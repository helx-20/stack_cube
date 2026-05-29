"""
Stage1 trainer for StackCube criticality.

Self-contained: loads raw NDE episode .npy files (produced by
stage1_collect.py), flattens them into per-step (obs+force, label) samples
where every step of a crash episode is positive (per user choice), and
trains a SimpleClassifier.

Expected on-disk layout (paths can be overridden by --pos_dir / --neg_dir):
    <pos_dir>/*.npy   -> crash (failure) episode files
    <neg_dir>/*.npy   -> success episode files
Each .npy is a 1-D array of episode dicts (see data_utils.py).
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Make `criticality.*` importable regardless of CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.criticality_model import SimpleClassifier
from criticality.utils.data_utils import collect_npy_files, flatten_episodes, load_episodes

from sklearn.metrics import auc


# ---------- metrics ----------

def precision_recall_curve(y_true, y_score, num_thresholds: int = 1000):
    """Compute precision/recall over a uniform threshold grid on [0,1]."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thr = np.linspace(1.0, 0.0, num_thresholds, endpoint=False)
    if y_true.size == 0 or y_score.size == 0:
        prec = np.concatenate(([1.0], np.zeros(num_thresholds)))
        rec = np.concatenate(([0.0], np.zeros(num_thresholds)))
        return prec, rec, thr

    positives = int(np.sum(y_true == 1))
    prec_list, rec_list = [], []
    for t in thr:
        preds = (y_score >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = positives - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(p)
        rec_list.append(r)
    prec = np.concatenate(([1.0], np.array(prec_list)))
    rec = np.concatenate(([0.0], np.array(rec_list)))
    return prec, rec, thr


# ---------- data loading ----------

def build_split(data_dir: str, rng: np.random.Generator | None = None, ratios=(0.8, 0.1, 0.1)):
    """Load all .npy episode files and flatten to (X, y)."""
    pos_files = collect_npy_files(os.path.join(data_dir, "raw", "positive"))
    neg_files = collect_npy_files(os.path.join(data_dir, "raw", "negative"))
    print(f"[stage1] pos files: {len(pos_files)} | neg files: {len(neg_files)}")

    pos_eps = load_episodes(pos_files) if pos_files else []
    neg_eps = load_episodes(neg_files) if neg_files else []
    print(f"[stage1] pos episodes: {len(pos_eps)} | neg episodes: {len(neg_eps)}")

    X_pos, y_pos = flatten_episodes(pos_eps)
    X_neg, y_neg = flatten_episodes(neg_eps)
    print(f"[stage1] pos steps: {len(y_pos)} | neg steps (after subsample): {len(y_neg)}")

    if len(X_pos) == 0 and len(X_neg) == 0:
        raise RuntimeError("No data found under the provided pos_dir / neg_dir")

    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    y_pos = np.array(y_pos)
    y_neg = np.array(y_neg)
    
    rng = rng or np.random.default_rng(0)
    n_pos = len(y_pos)
    n_neg = len(y_neg)
    idx_pos = np.arange(n_pos)
    idx_neg = np.arange(n_neg)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_train_pos = int(n_pos * ratios[0])
    n_val_pos = int(n_pos * ratios[1])
    n_train_neg = int(n_neg * ratios[0])
    n_val_neg = int(n_neg * ratios[1])
    tr_pos, va_pos, te_pos = idx_pos[:n_train_pos], idx_pos[n_train_pos:n_train_pos + n_val_pos], idx_pos[n_train_pos + n_val_pos:]
    tr_neg, va_neg, te_neg = idx_neg[:int(0.1*n_train_neg)], idx_neg[n_train_neg:n_train_neg + n_val_neg], idx_neg[n_train_neg + n_val_neg:]
    X_train = np.concatenate([X_pos[tr_pos], X_neg[tr_neg]], axis=0)
    y_train = np.concatenate([y_pos[tr_pos], y_neg[tr_neg]], axis=0)
    X_val = np.concatenate([X_pos[va_pos], X_neg[va_neg]], axis=0)
    y_val = np.concatenate([y_pos[va_pos], y_neg[va_neg]], axis=0)
    X_test = np.concatenate([X_pos[te_pos], X_neg[te_neg]], axis=0)
    y_test = np.concatenate([y_pos[te_pos], y_neg[te_neg]], axis=0)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------- evaluation ----------

def evaluate(model, loader, device):
    model.eval()
    total = correct = tp = fp = fn = 0
    y_true, y_score = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
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
    prec_arr, rec_arr, thr = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec_arr, prec_arr)
    return {
        "acc": acc, "precision": precision, "recall": recall, "auc": pr_auc,
        "prec_arr": prec_arr, "rec_arr": rec_arr, "thr": thr,
    }


def save_pr_curve(metrics: dict, path: str):
    if plt is None:
        return
    prec, rec, thr = metrics["prec_arr"], metrics["rec_arr"], metrics["thr"]
    if prec.size == 0:
        return
    fig = plt.figure()
    plt.step(rec, prec, where="post")
    point_num = 20
    idx = []
    for i in range(point_num):
        for tmp in range(len(thr)):
            if rec[tmp] >= max(rec) - (max(rec) - min(rec)) * i / point_num:
                idx.append(tmp)
                break
    points = [(rec[i], prec[i], thr[i]) for i in idx]
    for x, y, tval in points:
        plt.scatter([x], [y], color='red', s=24)
        plt.annotate(f'{tval:.2f}', xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=8, color='red')
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AUC={metrics['auc']:.4f})")
    plt.grid(True); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


# ---------- train / test ----------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    rng = np.random.default_rng(args.seed)

    if os.path.exists(os.path.join(args.data_dir, "train.pkl")):
        with open(os.path.join(args.data_dir, "train.pkl"), "rb") as f:
            train_data = pickle.load(f)
        X_tr = train_data['inputs']
        y_tr = train_data['labels']
        with open(os.path.join(args.data_dir, "val.pkl"), "rb") as f:
            val_data = pickle.load(f)
        X_va = val_data['inputs']
        y_va = val_data['labels']
        with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
            test_data = pickle.load(f)
        X_te = test_data['inputs']
        y_te = test_data['labels']
        print(f"[stage1] loaded data from {args.data_dir}")
    else:
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = build_split(args.data_dir, rng=rng)
        with open(os.path.join(args.data_dir, "train.pkl"), "wb") as f:
            pickle.dump({'inputs': X_tr, 'labels': y_tr}, f, protocol=4)
        with open(os.path.join(args.data_dir, "val.pkl"), "wb") as f:
            pickle.dump({'inputs': X_va, 'labels': y_va}, f, protocol=4)
        with open(os.path.join(args.data_dir, "test.pkl"), "wb") as f:
            pickle.dump({'inputs': X_te, 'labels': y_te}, f, protocol=4)
    print(f"[stage1] train={len(y_tr)} val={len(y_va)} test={len(y_te)} | input_dim={X_tr.shape[1]}")

    def make_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.from_numpy(Xa).float(), torch.from_numpy(ya).long())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(X_tr, y_tr, True)
    val_loader = make_loader(X_va, y_va, False)
    test_loader = make_loader(X_te, y_te, False)

    model = SimpleClassifier(input_dim=X_tr.shape[1], hidden=args.hidden, hidden_layer=args.hidden_layer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"stage1_criticality_best_{args.model_idx}.pt")

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
        train_acc = correct / max(total, 1)

        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_acc={train_acc:.4f} | "
              f"val_acc={val['acc']:.4f} p={val['precision']:.4f} r={val['recall']:.4f} auc={val['auc']:.4f}")

        if val["auc"] >= best_auc:
            best_auc = val["auc"]
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best ckpt (auc={best_auc:.4f}) to {save_path}")

    # final test on held-out
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    test = evaluate(model, test_loader, device)
    print(f"\n[stage1][TEST] acc={test['acc']:.4f} p={test['precision']:.4f} "
          f"r={test['recall']:.4f} auc={test['auc']:.4f}")
    save_pr_curve(test, os.path.join(args.save_dir, f"precision_recall_{args.model_idx}.png"))


def test_only(args):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    X_te = test_data['inputs']
    y_te = test_data['labels']
    print(f"[stage1] loaded data from {args.data_dir}")
    ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleClassifier(input_dim=X_te.shape[1], hidden=args.hidden, hidden_layer=args.hidden_layer).to(device)
    ckpt = os.path.join(args.save_dir, f"stage1_criticality_best_{args.model_idx}.pt")
    if not os.path.exists(ckpt):
        print(f"[stage1][TEST] no ckpt at {ckpt}, abort")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    m = evaluate(model, loader, device)
    print(f"[stage1][TEST] acc={m['acc']:.4f} p={m['precision']:.4f} r={m['recall']:.4f} auc={m['auc']:.4f}")
    save_pr_curve(m, os.path.join(args.save_dir, f"precision_recall_{args.model_idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/mnt1/linxuan/stack_cube_data/data/stage1", help="Folder with episode .npy files")
    parser.add_argument("--save_dir", type=str,
                        default="criticality/stage1/model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--hidden_layer", type=int, default=3)
    parser.add_argument("--model_idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="Only run evaluation on the best ckpt")
    args = parser.parse_args()

    print("args:", args)
    if args.test:
        test_only(args)
    else:
        train(args)
