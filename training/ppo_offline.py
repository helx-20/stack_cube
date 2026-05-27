#!/usr/bin/env python3
import os
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
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
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
def load_offline_dataset(data_dirs, gamma=0.8):
    """
    data_dirs: 可以是一个字符串路径，也可以是一个路径列表
    """
    all_obs, all_acts, all_returns, all_weights, all_log_probs = [], [], [], [], []
    
    # 如果传入的是单个字符串，转为列表统一处理
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    npy_files = []
    for d in data_dirs:
        files = glob.glob(os.path.join(d, "*.npy"))
        if not files:
            print(f"[!] 警告: 文件夹 {d} 中没有找到 .npy 文件。")
        npy_files.extend(files)

    if not npy_files:
        raise ValueError(f"在所有提供的路径 {data_dirs} 中均未找到 .npy 数据文件！")
        
    print(f"[*] 从 {len(data_dirs)} 个目录中共找到 {len(npy_files)} 个数据文件，正在处理...")

    for file in npy_files:
        try:
            data = np.load(file, allow_pickle=True).item()
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

            all_obs.append(obs)
            all_acts.append(acts)
            all_returns.append(returns)
            all_weights.append(weights)
            all_log_probs.append(log_probs)
        except Exception as e:
            print(f"[!] 读取文件 {file} 失败: {e}")

    obs_t = torch.tensor(np.concatenate(all_obs), dtype=torch.float32)
    acts_t = torch.tensor(np.concatenate(all_acts), dtype=torch.float32)
    returns_t = torch.tensor(np.concatenate(all_returns), dtype=torch.float32)
    weights_t = torch.tensor(np.concatenate(all_weights), dtype=torch.float32)
    log_probs_t = torch.tensor(np.concatenate(all_log_probs), dtype=torch.float32)
    
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

    with torch.no_grad():
        agent.actor_logstd.fill_(args.log_std)
        agent.actor_logstd.requires_grad = False

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

            # --- 2. 计算标准的优势值 (Advantage) ---
            with torch.no_grad():
                adv = b_ret - values.detach()
                # 🌟 关键修复：必须进行归一化，让优势值有正有负！
                # 绝对不能套用 torch.exp()，否则 IS 将无法正确惩罚坏动作。
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # --- 3. 🌟 核心：策略重要性采样 (Policy IS) ---
            dist = agent.get_action_distribution(b_obs)
            logp = dist.log_prob(b_act).sum(dim=1)
            
            # 🌟 关键补丁：在 exp 之前限制对数差的范围，防止数值爆炸！
            # 允许比率在 [e^-5, e^2] (即约 0.006 到 7.38) 之间波动，然后再交给 PPO Clip
            log_ratio = logp - b_log_prob
            log_ratio = torch.clamp(log_ratio, min=-5.0, max=2.0) 
            
            # 保留你的核心 IS 逻辑
            ratio = torch.exp(log_ratio)

            clip_coef = 0.2
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
            
            safe_env_w = torch.clamp(b_weights, min=0.5, max=5.0)
            ppo_loss = -(torch.min(surr1, surr2) * safe_env_w).mean()

            # --- 4. 弱化行为锚点 (Soft Anchor) ---
            # 因为数据来源有 -2.0 和 -1.5 两种 log_std，初始的 IS ratio 波动会很大
            # 我们需要把 anchor_loss 调低一点 (从 0.05 降到 0.01)，
            # 给 IS 留出足够的空间去调整 mean，从而让 ratio 尽快恢复到健康范围。
            current_mean = agent.actor_mean(b_obs)
            anchor_loss = 0.01 * F.mse_loss(current_mean, b_act)

            # --- 5. 组合 Loss ---
            p_loss = ppo_loss + anchor_loss
            loss = p_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_v_loss += v_loss.item()
            epoch_p_loss += p_loss.item()
            
        avg_v_loss = epoch_v_loss / len(dataloader)
        avg_p_loss = epoch_p_loss / len(dataloader)
        current_std = torch.exp(agent.actor_logstd).mean().item()
        
        print(f"Epoch: {epoch}/{args.epochs} | V Loss: {avg_v_loss:.4f} | P Loss: {avg_p_loss:.4f}| Mean Std: {current_std:.4f}")

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_path = os.path.join(args.out_dir, f"offline_model_ep{epoch}.pt")
            torch.save(agent.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 🌟 核心修改：nargs='+' 表示可以接收 1 个或多个参数
    parser.add_argument("--dataset", type=str, nargs='+', default=["./test_collect_data"], 
                        help="一个或多个数据文件夹路径，用空格分隔")
    parser.add_argument("--initial_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./runs/offline_training")
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--save_freq", type=int, default=10)
    
    parser.add_argument("--log_std", type=float, default=-2.0)

    args = parser.parse_args()
    train_offline(args)