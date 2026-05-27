"""
Per-step DQN for StackCube criticality stage2.

Convention:
  - state  = obs              shape (48,)
  - action = unit force (fx,fy,fz)  shape (3,)
  - q_net input = state + action    shape (51,)  -> SimpleClassifier
  - q_net output = logits (B, 2); index 1 acts as the criticality / Q signal

The 11^3 = 1331 (fx,fy,fz) grid is enumerated when computing target Q.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    """Disk-backed replay buffer split into positive (critical) and negative pools.

    Each transition is a dict with keys:
      input      : np.ndarray (state_dim,)      obs at step t
      action     : np.ndarray (action_dim,)     force at step t
      next_input : np.ndarray (state_dim,)      obs at step t+1
      reward     : float                        1.0 for critical, 0.0 otherwise
      done       : float                        1.0 at episode end, else 0.0
    """

    def __init__(self, pos_path: str, neg_path: str, pos_ratio: float = 0.5):
        self.pos_buf = list(np.load(pos_path, allow_pickle=True))
        self.neg_buf = list(np.load(neg_path, allow_pickle=True))
        self.pos_ratio = float(pos_ratio)

    def __len__(self):
        return len(self.pos_buf) + len(self.neg_buf)

    def _stack(self, samples):
        inputs = torch.tensor(
            np.stack([np.concatenate([np.asarray(t["input"]).ravel(),
                                       np.asarray(t["action"]).ravel()]) for t in samples]),
            dtype=torch.float,
        )
        next_inputs = torch.tensor(
            np.stack([np.asarray(t["next_input"]).ravel() for t in samples]),
            dtype=torch.float,
        )
        rewards = torch.tensor([float(t["reward"]) for t in samples], dtype=torch.float)
        dones = torch.tensor([float(t["done"]) for t in samples], dtype=torch.float)
        return inputs, next_inputs, rewards, dones

    def sample(self, batch_size: int):
        n_pos = max(1, int(batch_size * self.pos_ratio))
        n_neg = batch_size - n_pos
        pos = random.choices(self.pos_buf, k=n_pos) if len(self.pos_buf) < n_pos else random.sample(self.pos_buf, n_pos)
        neg = random.choices(self.neg_buf, k=n_neg) if len(self.neg_buf) < n_neg else random.sample(self.neg_buf, n_neg)
        p_in, p_next, p_r, p_d = self._stack(pos)
        n_in, n_next, n_r, n_d = self._stack(neg)
        return (
            torch.cat([p_in, n_in], dim=0),
            torch.cat([p_next, n_next], dim=0),
            torch.cat([p_r, n_r], dim=0),
            torch.cat([p_d, n_d], dim=0),
        )


def build_force_grid(step: float = 0.2) -> np.ndarray:
    """Build the 11^3 (fx,fy,fz) discrete force grid in [-1,1]."""
    vals = np.arange(-1.0, 1.0 + 1e-6, step, dtype=np.float32)
    fx, fy, fz = np.meshgrid(vals, vals, vals, indexing="ij")
    return np.stack([fx.reshape(-1), fy.reshape(-1), fz.reshape(-1)], axis=1).astype(np.float32)


class DQN:
    """Per-step DQN wrapper around a SimpleClassifier-style q_net.

    q_net(state + action) -> (B, 2) logits. We treat both output classes
    as Q values that go through the Bellman update jointly (matches the
    reference implementation that used the classifier head directly).
    """

    def __init__(self, q_net, learning_rate: float = 1e-3, gamma: float = 0.9,
                 target_update: int = 50, device: str = "cpu"):
        self.device = torch.device(device)
        self.q_net = q_net.to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_q_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.target_update = int(target_update)
        self.step_count = 0
        self.loss_fn = nn.MSELoss()

        # StackCube: 11^3 = 1331 discrete (fx, fy, fz) candidates.
        grid = build_force_grid(step=0.2)
        self.action_grid = torch.from_numpy(grid).to(self.device)  # (1331, 3)
        self.num_actions = self.action_grid.shape[0]

    def update(self, inputs: torch.Tensor, next_obs: torch.Tensor,
               rewards: torch.Tensor, dones: torch.Tensor) -> float:
        inputs = inputs.to(self.device)
        next_obs = next_obs.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        dones = dones.to(self.device).unsqueeze(1)

        self.q_net.train()
        q_vals = self.q_net(inputs)  # (B, 2)

        # Target Q: for each next_obs, enumerate all 1331 force actions and take max.
        with torch.no_grad():
            B = next_obs.shape[0]
            q_next_vals = torch.zeros_like(q_vals)
            for i in range(B):
                cur_obs = next_obs[i].unsqueeze(0).repeat(self.num_actions, 1)
                cur_input = torch.cat([cur_obs, self.action_grid], dim=1)  # (1331, 51)
                cur_q = self.target_q_net(cur_input)                        # (1331, 2)
                q_next_vals[i] = torch.max(cur_q, dim=0).values             # (2,)

        # Reward broadcasts to both logits; done masks bootstrap.
        q_targets = rewards + self.gamma * q_next_vals * (1.0 - dones)

        loss = self.loss_fn(q_vals, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.target_update > 0 and self.step_count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
