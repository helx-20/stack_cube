"""
Convert stage2_collect.py output into per-step DQN replay-buffer files.

stage2_collect.py saves each batch as a list of episodes. Each episode is a
list of per-step dicts with at least:
    {
      "obs":     (48,)  state vector at step t
      "action":  ...    robot action (unused here)
      "weight":  float  IS weight
      "success": bool   per-step success flag
      "fx", "fy", "fz": float  unit force applied at step t
    }

We turn that into per-step transitions for DQN:
    {
      "input":      np.ndarray (48,)  obs at step t
      "action":     np.ndarray (3,)   (fx, fy, fz) at step t
      "next_input": np.ndarray (48,)  obs at step t+1
      "reward":     1.0 if crash episode else 0.0
      "done":       1.0 only on the final step of a crash episode, else 0.0
    }

Episode label rules (per user choice: whole crash episode is positive):
    - Crash episode (no success at any step)  -> all steps -> replay_buffer_pos
    - Success episode                         -> sub-sample -> replay_buffer_neg
"""

import argparse
import glob
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _obs_to_vec(o) -> np.ndarray:
    arr = np.asarray(o, dtype=np.float32).reshape(-1)
    return arr


def episode_is_crash(episode) -> bool:
    return not any(bool(s.get("success", False)) for s in episode)


def build_dataset(data_dir: str, out_dir: str | None = None,
                  neg_subsample: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    pos_files = sorted(glob.glob(os.path.join(data_dir, "positive", "*.npy")))
    neg_files = sorted(glob.glob(os.path.join(data_dir, "negative", "*.npy")))
    print(f"[stage2_process] found {len(pos_files)} pos files and {len(neg_files)} neg files under {data_dir}")

    pos, neg = [], []
    n_eps_pos = n_eps_neg = 0

    for fp in pos_files:
        batch = np.load(fp, allow_pickle=True)
        for ep in batch:
            L = len(ep['obs'])
            if L < 2:
                continue
            is_crash = True
            n_eps_pos += 1

            for t in range(L):
                cur_obs = _obs_to_vec(ep["obs"][t])
                nxt_obs = _obs_to_vec(ep["obs"][t + 1]) if t + 1 < L else cur_obs
                force = np.array(ep["force"][t], dtype=np.float32)
                done = 1.0 if (is_crash and t == L - 1) else 0.0
                reward = 1.0 if is_crash else 0.0
                rec = {
                    "input": cur_obs,
                    "action": force,
                    "next_input": nxt_obs,
                    "reward": reward,
                    "done": done,
                }
                pos.append(rec)
    
    for fp in neg_files:
        batch = np.load(fp, allow_pickle=True)
        for ep in batch:
            L = len(ep['obs'])
            if L < 2:
                continue
            is_crash = False
            n_eps_neg += 1

            for t in range(L):
                cur_obs = _obs_to_vec(ep["obs"][t])
                nxt_obs = _obs_to_vec(ep["obs"][t + 1]) if t + 1 < L else cur_obs
                force = np.array(ep["force"][t], dtype=np.float32)
                done = 0.0
                reward = 0.0
                rec = {
                    "input": cur_obs,
                    "action": force,
                    "next_input": nxt_obs,
                    "reward": reward,
                    "done": done,
                }
                if rng.random() < neg_subsample:
                    neg.append(rec)

    print("-" * 40)
    print(f"[stage2_process] crash ep={n_eps_pos}  success ep={n_eps_neg}")
    print(f"[stage2_process] pos transitions={len(pos)}  neg transitions={len(neg)} (subsampled)")

    pos_path = os.path.join(out_dir, "replay_buffer_pos.npy")
    neg_path = os.path.join(out_dir, "replay_buffer_neg.npy")
    np.save(pos_path, np.array(pos, dtype=object), allow_pickle=True)
    np.save(neg_path, np.array(neg, dtype=object), allow_pickle=True)
    print(f"[stage2_process] saved -> {pos_path}")
    print(f"[stage2_process] saved -> {neg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/mnt1/linxuan/stack_cube_data/data/stage1/raw")
    parser.add_argument("--out_dir", type=str, default="/mnt/mnt1/linxuan/stack_cube_data/data/stage2", help="Output folder for replay_buffer_*.npy (default: parent of data_dir)")
    parser.add_argument("--neg_subsample", type=float, default=0.1,
                        help="Keep this fraction of success-episode transitions in neg buffer")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_dataset(args.data_dir, args.out_dir, args.neg_subsample, args.seed)
