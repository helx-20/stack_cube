"""
StackCube per-step data utilities.

stage1_collect.py saves each rollout as a dict with per-step lists:
    {
      "obs":     [obs_0, obs_1, ...],          # each obs is shape (48,)
      "action":  [act_0, act_1, ...],          # robot action (unused here)
      "reward":  [r_0, r_1, ...],
      "force":   [f_0, f_1, ...],              # unit force (fx,fy,fz) in [-1,1]
      "success": 0 or 1,                       # episode-level label
    }

For SimpleClassifier we want per-step (input, label) where
    input = obs(48) + force(3) = 51
    label = 1 if the step belongs to a crash episode, else 0
(Per user choice: the whole crash episode is labeled positive.)
"""

import os
from typing import Iterable, List, Tuple

import numpy as np


def collect_npy_files(folder: str) -> List[str]:
    """Return sorted list of .npy files in `folder`."""
    return sorted(os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith(".npy"))


def episode_to_steps(episode: dict) -> Tuple[np.ndarray, int]:
    """Convert one episode dict into a (T, 51) feature array.

    Returns (features, episode_label). episode_label = 1 if crash (success==0).
    """
    obs = np.asarray(episode["obs"], dtype=np.float32)            # (T, 48)
    force = np.asarray(episode["force"], dtype=np.float32)        # (T, 3)
    if force.ndim == 1:
        force = force.reshape(-1, 3)
    T = min(obs.shape[0], force.shape[0])
    feats = np.concatenate([obs[:T], force[:T]], axis=1)          # (T, 51)
    ep_label = 0 if int(episode.get("success", 0)) == 1 else 1
    return feats, ep_label


def flatten_episodes(
    episodes: Iterable[dict],
    neg_subsample: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten a list of episode dicts into per-step samples.

    - Whole crash episode -> label 1
    - Whole success episode -> label 0 (subsampled by `neg_subsample` to control imbalance)

    Returns (X, y) with X shape (N, 51), y shape (N,).
    """
    if rng is None:
        rng = np.random.default_rng()

    X_chunks, y_chunks = [], []
    for ep in episodes:
        feats, ep_label = episode_to_steps(ep)
        if feats.shape[0] == 0:
            continue
        if ep_label == 0 and neg_subsample < 1.0:
            keep = rng.random(feats.shape[0]) < neg_subsample
            feats = feats[keep]
            if feats.shape[0] == 0:
                continue
        X_chunks.append(feats)
        y_chunks.append(np.full(feats.shape[0], ep_label, dtype=np.int64))

    if not X_chunks:
        return np.zeros((0, 51), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(X_chunks, axis=0), np.concatenate(y_chunks, axis=0)


def load_episodes(path_or_paths) -> List[dict]:
    """Load one or more .npy files of episode dicts and concatenate."""
    if isinstance(path_or_paths, str):
        path_or_paths = [path_or_paths]
    out: List[dict] = []
    for p in path_or_paths:
        data = np.load(p, allow_pickle=True)
        out.extend(list(data))
    return out
