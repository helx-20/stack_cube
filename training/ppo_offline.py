#!/usr/bin/env python3
import os, sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.distributions.normal import Normal

# ==========================================
# 1. 网络定义 (保持不变)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_distribution(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

# ==========================================
# 2. 数据处理与加载 (已修改以支持多文件夹)
# ==========================================
def load_offline_dataset(data_dirs, gamma=0.95):
    """
    data_dirs: 可以是一个字符串路径，也可以是一个路径列表
    """
    all_obs, all_acts, all_returns, all_weights, all_log_probs = [], [], [], [], []
    
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    all_obs = np.empty((0, 48), dtype=np.float32)
    all_acts = np.empty((0, 8), dtype=np.float32)
    all_returns = np.empty((0,), dtype=np.float32)
    all_weights = np.empty((0,), dtype=np.float32)
    all_log_probs = np.empty((0,), dtype=np.float32)
    for data_dir in data_dirs:
        all_data_path = os.path.join(data_dir, 'all_data_unified_weight.npy')
        if os.path.exists(all_data_path):
            data = np.load(all_data_path, allow_pickle=True).item()
            obs = np.array(data['obs'], dtype=np.float32)
            acts = np.array(data['actions'], dtype=np.float32)
            returns = np.array(data['returns'], dtype=np.float32)
            weights = np.array(data['weights'], dtype=np.float32)
            log_prob = np.array(data['log_prob'], dtype=np.float32)

            all_obs = np.concatenate([all_obs, obs])
            all_acts = np.concatenate([all_acts, acts])
            all_returns = np.concatenate([all_returns, returns])
            all_weights = np.concatenate([all_weights, weights])
            all_log_probs = np.concatenate([all_log_probs, log_prob])
        else:
            tmp_obs = np.empty((0, 48), dtype=np.float32)
            tmp_acts = np.empty((0, 8), dtype=np.float32)
            tmp_returns = np.empty((0,), dtype=np.float32)
            tmp_weights = np.empty((0,), dtype=np.float32)
            tmp_log_probs = np.empty((0,), dtype=np.float32)
            for filename in os.listdir(data_dir):
                if filename.endswith('.npy') and not filename.startswith('all'):
                    path = os.path.join(data_dir, filename)
                    try:
                        data = np.load(path, allow_pickle=True)
                    except Exception as e:
                        continue
                    obs = np.array(data['obs'], dtype=np.float32)
                    acts = np.array(data['actions'], dtype=np.float32)
                    rews = np.array(data['rewards'], dtype=np.float32)
                    dones = np.array(data['dones'], dtype=bool)
                    weights = np.array(data['weights'], dtype=np.float32)
                    log_probs = np.array(data['log_probs'], dtype=np.float32).reshape(-1)

                    # 离线计算 Discounted Returns
                    returns = np.zeros_like(rews, dtype=np.float32)
                    G = 0.0
                    for i in reversed(range(len(rews))):
                        if dones[i]:
                            G = rews[i]
                        else:
                            G = rews[i] + gamma * G
                        returns[i] = G

                    tmp_obs = np.concatenate([tmp_obs, obs])
                    tmp_acts = np.concatenate([tmp_acts, acts])
                    tmp_returns = np.concatenate([tmp_returns, returns])
                    tmp_weights = np.concatenate([tmp_weights, weights])
                    tmp_log_probs = np.concatenate([tmp_log_probs, log_probs])
            
            np.save(all_data_path, {'obs': tmp_obs, 'actions': tmp_acts, 'returns': tmp_returns, 'weights': tmp_weights, 'log_prob': tmp_log_probs})
            all_obs = np.concatenate([all_obs, tmp_obs])
            all_acts = np.concatenate([all_acts, tmp_acts])
            all_returns = np.concatenate([all_returns, tmp_returns])
            all_weights = np.concatenate([all_weights, tmp_weights])
            all_log_probs = np.concatenate([all_log_probs, tmp_log_probs])

    obs_t = torch.tensor(all_obs, dtype=torch.float32)
    acts_t = torch.tensor(all_acts, dtype=torch.float32)
    returns_t = torch.tensor(all_returns, dtype=torch.float32)
    weights_t = torch.tensor(all_weights, dtype=torch.float32)
    weights_t = weights_t.clamp(max=1.0)
    weights_t = weights_t / (weights_t.mean() + 1e-8)
    log_probs_t = torch.tensor(all_log_probs, dtype=torch.float32)
    
    print(f"[*] 数据集加载完毕。总步数: {obs_t.shape[0]}")
    return obs_t, acts_t, returns_t, weights_t, log_probs_t

# ==========================================
# 3. 离线训练主循环 (保持不变)
# ==========================================
def train_offline(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 这里的 args.dataset 现在是一个列表
    obs_t, acts_t, returns_t, weights_t, log_probs_t = load_offline_dataset(args.dataset, gamma=args.gamma)
    
    obs_dim = obs_t.shape[1]
    act_dim = acts_t.shape[1]

    agent = Agent(obs_dim, act_dim).to(device)
    if args.initial_ckpt and os.path.exists(args.initial_ckpt):
        agent.load_state_dict(torch.load(args.initial_ckpt, map_location=device))
        print(f"[*] 成功加载预训练基线模型: {args.initial_ckpt}")

    if args.log_std is not None:
        with torch.no_grad():
            agent.actor_logstd.fill_(args.log_std)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=args.learning_rate, eps=1e-5)

    dataset = TensorDataset(obs_t, acts_t, returns_t, weights_t, log_probs_t)
    
    sample_weights = weights_t.clone()
    sample_weights = torch.clamp(sample_weights, min=1e-3, max=10.0) 
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    print("[*] 开始带有 Policy IS 的离线 PPO 训练...")
    
    agent.train()
    for epoch in range(1, args.epochs + 1):
        epoch_v_loss = 0.0
        epoch_p_loss = 0.0
        
        for b_obs, b_act, b_ret, b_weights, b_log_prob in dataloader:
            b_obs, b_act, b_ret, b_log_prob, b_weights = [x.to(device) for x in [b_obs, b_act, b_ret, b_log_prob, b_weights]]

            values = agent.get_value(b_obs).squeeze(-1)
            v_loss = F.mse_loss(values, b_ret)

            with torch.no_grad():
                adv = b_ret - values.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # awac_w = torch.exp(adv)
            # total_w = (adv * b_weights).clamp(max=args.combined_weight_max)

            dist = agent.get_action_distribution(b_obs)
            logp = dist.log_prob(b_act).sum(dim=1)
            
            log_ratio = logp - b_log_prob
            log_ratio = torch.clamp(log_ratio, min=-5.0, max=2.0) 
            ratio = torch.exp(log_ratio)
            clip_coef = 0.2
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
            ppo_loss = -(torch.min(surr1, surr2)).mean()
            
            # ppo_loss = -(ratio * adv * b_weights).mean()
            # ppo_loss = -(logp * total_w).sum() / (total_w.sum() + 1e-8)
            current_mean = agent.actor_mean(b_obs)
            anchor_loss = args.bc_coef * F.mse_loss(current_mean, b_act)

            p_loss = ppo_loss + anchor_loss
            if epoch <= args.warmup_epochs:
                loss = args.vf_coef * v_loss
            else:
                loss = p_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_v_loss += v_loss.item()
            epoch_p_loss += p_loss.item()
            
        avg_v_loss = epoch_v_loss / len(dataloader)
        avg_p_loss = epoch_p_loss / len(dataloader)
        current_std = torch.tensor(agent.actor_logstd).mean().item()
        
        print(f"Epoch: {epoch}/{args.epochs} | V Loss: {avg_v_loss:.4f} | P Loss: {avg_p_loss:.4f}| Mean Std: {current_std:.4f}")

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_path = os.path.join(args.out_dir, f"offline_model_ep{epoch}.pt")
            torch.save(agent.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs='+', default=["/mnt/mnt1/linxuan/stack_cube_data/data/training/round1"], 
                        help="一个或多个数据文件夹路径，用空格分隔")
    parser.add_argument("--initial_ckpt", type=str, default='examples/baselines/ppo/runs/StackCube-v1__ppo__1__1780033432/final_ckpt.pt')
    parser.add_argument("--out_dir", type=str, default="./training/models/round1")
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=1.0)
    parser.add_argument("--bc_coef", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--combined_weight_max", type=float, default=10.0)
    parser.add_argument("--save_freq", type=int, default=3)
    
    parser.add_argument("--log_std", default=None)

    args = parser.parse_args()
    print(args)
    train_offline(args)