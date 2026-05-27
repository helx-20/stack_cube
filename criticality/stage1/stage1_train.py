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

def build_split(pos_dir: str, neg_dir: str, neg_subsample: float, rng: np.random.Generator):
    """Load all .npy episode files and flatten to (X, y)."""
    pos_files = collect_npy_files(pos_dir) if os.path.isdir(pos_dir) else []
    neg_files = collect_npy_files(neg_dir) if os.path.isdir(neg_dir) else []
    print(f"[stage1] pos files: {len(pos_files)} | neg files: {len(neg_files)}")

    pos_eps = load_episodes(pos_files) if pos_files else []
    neg_eps = load_episodes(neg_files) if neg_files else []
    print(f"[stage1] pos episodes: {len(pos_eps)} | neg episodes: {len(neg_eps)}")

    X_pos, y_pos = flatten_episodes(pos_eps, neg_subsample=1.0, rng=rng)
    X_neg, y_neg = flatten_episodes(neg_eps, neg_subsample=neg_subsample, rng=rng)
    print(f"[stage1] pos steps: {len(y_pos)} | neg steps (after subsample): {len(y_neg)}")

    if len(X_pos) == 0 and len(X_neg) == 0:
        raise RuntimeError("No data found under the provided pos_dir / neg_dir")

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    return X, y


def train_val_test_split(X: np.ndarray, y: np.ndarray, ratios=(0.8, 0.1, 0.1),
                         rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(0)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


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

    X, y = build_split(args.pos_dir, args.neg_dir, args.neg_subsample, rng)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(X, y, rng=rng)
    print(f"[stage1] train={len(y_tr)} val={len(y_va)} test={len(y_te)} | input_dim={X.shape[1]}")

    def make_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.from_numpy(Xa).float(), torch.from_numpy(ya).long())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(X_tr, y_tr, True)
    val_loader = make_loader(X_va, y_va, False)
    test_loader = make_loader(X_te, y_te, False)

    model = SimpleClassifier(input_dim=X.shape[1], hidden=args.hidden, hidden_layer=args.hidden_layer).to(device)
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
    save_pr_curve(test, os.path.join(args.save_dir, "precision_recall.png"))


def test_only(args):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    rng = np.random.default_rng(args.seed)

    X, y = build_split(args.pos_dir, args.neg_dir, args.neg_subsample, rng)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleClassifier(input_dim=X.shape[1], hidden=args.hidden, hidden_layer=args.hidden_layer).to(device)
    ckpt = os.path.join(args.save_dir, f"stage1_criticality_best_{args.model_idx}.pt")
    if not os.path.exists(ckpt):
        print(f"[stage1][TEST] no ckpt at {ckpt}, abort")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    m = evaluate(model, loader, device)
    print(f"[stage1][TEST] acc={m['acc']:.4f} p={m['precision']:.4f} r={m['recall']:.4f} auc={m['auc']:.4f}")
    save_pr_curve(m, os.path.join(args.save_dir, "precision_recall.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_dir", type=str, default="/mnt/mnt1/linxuan/stack_cube_data/data/stage1/positive",
                        help="Folder with crash (failure) episode .npy files")
    parser.add_argument("--neg_dir", type=str, default="/mnt/mnt1/linxuan/stack_cube_data/data/stage1/negative",
                        help="Folder with success episode .npy files")
    parser.add_argument("--save_dir", type=str,
                        default="criticality/stage1/model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--hidden_layer", type=int, default=3)
    parser.add_argument("--model_idx", type=int, default=1)
    parser.add_argument("--neg_subsample", type=float, default=1,
                        help="Fraction of negative (success) steps to keep, to control class imbalance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="Only run evaluation on the best ckpt")
    args = parser.parse_args()

    print("args:", args)
    if args.test:
        test_only(args)
    else:
        train(args)
