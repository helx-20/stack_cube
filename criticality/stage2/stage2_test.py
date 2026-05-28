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


def test_only(args):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    with open(os.path.join(args.data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    X = test_data['inputs']
    y = test_data['labels']
    print(f"[stage1] loaded data from {args.data_dir}")
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleClassifier(input_dim=X.shape[1], hidden=args.hidden, hidden_layer=args.hidden_layer).to(device)
    ckpt = os.path.join(args.save_dir, f"stage2_dqn_iter{args.model_iter}.pt")
    if not os.path.exists(ckpt):
        print(f"[stage1][TEST] no ckpt at {ckpt}, abort")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    m = evaluate(model, loader, device)
    print(f"[stage1][TEST] acc={m['acc']:.4f} p={m['precision']:.4f} r={m['recall']:.4f} auc={m['auc']:.4f}")
    save_pr_curve(m, os.path.join(args.save_dir, f"precision_recall_{args.model_iter}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/mnt1/linxuan/stack_cube_data/data/stage1", help="Folder with episode .npy files")
    parser.add_argument("--save_dir", type=str,
                        default="criticality/stage2/model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--hidden_layer", type=int, default=3)
    parser.add_argument("--model_iter", type=int, default=4600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="Only run evaluation on the best ckpt")
    args = parser.parse_args()

    print("args:", args)
    test_only(args)
