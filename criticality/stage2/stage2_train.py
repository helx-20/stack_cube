"""
Stage2 trainer: load the stage1 SimpleClassifier checkpoint and continue
training it as a DQN q-network using the per-step replay buffer produced
by stage2_process.py.

State = obs (48), action = unit force (3); q_net input = state+action = 51.
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.criticality_model import SimpleClassifier
from criticality.utils.dqn import DQN, ReplayBuffer
from criticality.utils.data_utils import collect_npy_files, flatten_episodes, load_episodes
from criticality.stage1.stage1_train import precision_recall_curve

from sklearn.metrics import auc


def build_val_loader(pos_dir: str, neg_dir: str, batch_size: int, neg_subsample: float, seed: int):
    """Re-use stage1 episode files for held-out validation (classifier-style)."""
    rng = np.random.default_rng(seed)
    pos_files = collect_npy_files(pos_dir) if os.path.isdir(pos_dir) else []
    neg_files = collect_npy_files(neg_dir) if os.path.isdir(neg_dir) else []
    pos_eps = load_episodes(pos_files) if pos_files else []
    neg_eps = load_episodes(neg_files) if neg_files else []
    X_pos, y_pos = flatten_episodes(pos_eps, neg_subsample=1.0, rng=rng)
    X_neg, y_neg = flatten_episodes(neg_eps, neg_subsample=neg_subsample, rng=rng)
    if len(X_pos) == 0 and len(X_neg) == 0:
        return None
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def validate(model, loader, device, tag: str = "val"):
    model.eval()
    total = correct = tp = fp = fn = 0
    y_true, y_score = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            tp += int(((preds == 1) & (yb == 1)).sum().item())
            fp += int(((preds == 1) & (yb == 0)).sum().item())
            fn += int(((preds == 0) & (yb == 1)).sum().item())
            y_score.extend(probs.cpu().numpy().tolist())
            y_true.extend(yb.cpu().numpy().tolist())
    acc = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    print(f"[stage2][{tag}] acc={acc:.4f} p={precision:.4f} r={recall:.4f} auc={pr_auc:.4f}")
    return pr_auc


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Model: SimpleClassifier with input_dim = 48 + 3 = 51
    model = SimpleClassifier(input_dim=51, hidden=args.hidden, hidden_layer=args.hidden_layer)
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        print(f"[stage2] loading stage1 ckpt: {args.stage1_ckpt}")
        model.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu"))
    else:
        print(f"[stage2][WARN] no stage1 ckpt at {args.stage1_ckpt}; training from scratch")

    # 2. Replay buffer
    rb = ReplayBuffer(args.pos_path, args.neg_path, pos_ratio=args.pos_ratio)
    print(f"[stage2] replay buffer: pos={len(rb.pos_buf)}  neg={len(rb.neg_buf)}")

    # 3. DQN
    agent = DQN(model, learning_rate=args.lr, gamma=args.gamma,
                target_update=args.target_update, device=device)

    # 4. Validation loader (classifier-style PR-AUC on held-out episodes)
    val_loader = build_val_loader(args.val_pos_dir, args.val_neg_dir,
                                  batch_size=512, neg_subsample=args.val_neg_subsample,
                                  seed=args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, f"stage2_dqn_best_{args.worker_id}.pt")
    best_auc = -1.0

    if val_loader is not None:
        best_auc = validate(agent.q_net, val_loader, device, tag="init")

    # 5. Training loop
    for it in range(1, args.iters + 1):
        inputs, next_obs, rewards, dones = rb.sample(args.batch_size)
        loss = agent.update(inputs, next_obs, rewards, dones)

        if it % args.log_interval == 0:
            print(f"[stage2] iter {it}/{args.iters}  loss={loss:.6f}")

        if val_loader is not None and it % args.val_interval == 0:
            cur_auc = validate(agent.q_net, val_loader, device, tag=f"iter{it}")
            if cur_auc > best_auc:
                best_auc = cur_auc
                torch.save(agent.q_net.state_dict(), best_path)
                print(f"  -> saved new best ckpt (auc={best_auc:.4f}) to {best_path}")

    # final dump
    final_path = os.path.join(args.save_dir, f"stage2_dqn_final_{args.worker_id}.pt")
    torch.save(agent.q_net.state_dict(), final_path)
    print(f"[stage2] final ckpt saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=0)

    # TODO: 改成本机实际路径
    parser.add_argument("--stage1_ckpt", type=str,
                        default="/home/teamcommon/tyy/ManiSkill_stackcube/criticality/stage1/model_stackcube/stage1_criticality_best_1.pt")
    parser.add_argument("--save_dir", type=str,
                        default="/home/teamcommon/tyy/ManiSkill_stackcube/criticality/stage2/model_stackcube")
    parser.add_argument("--pos_path", type=str,
                        default="/mnt/mnt1/tyy/ManiSkill_stackcube/replay_buffer_pos.npy")
    parser.add_argument("--neg_path", type=str,
                        default="/mnt/mnt1/tyy/ManiSkill_stackcube/replay_buffer_neg.npy")

    # validation: reuse stage1's raw episodes
    parser.add_argument("--val_pos_dir", type=str,
                        default="/mnt/mnt1/tyy/ManiSkill_stackcube/positive")
    parser.add_argument("--val_neg_dir", type=str,
                        default="/mnt/mnt1/tyy/ManiSkill_stackcube/negative")
    parser.add_argument("--val_neg_subsample", type=float, default=0.05)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--target_update", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--hidden_layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("args:", args)
    main(args)
