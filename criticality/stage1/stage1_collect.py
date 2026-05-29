import os, sys
import pickle
import argparse
import numpy as np
import torch
import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mani_skill.envs  # 注册环境
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from examples.baselines.ppo.ppo import Agent

# StackCube state obs (panda_wristcam): qpos(9)+qvel(9)+tcp_pose(7)+cubeA_pose(7)
# +cubeB_pose(7)+tcp_to_cubeA_pos(3)+tcp_to_cubeB_pos(3)+cubeA_to_cubeB_pos(3) = 48
EXPECTED_OBS_DIM = 48


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.pos_dir, exist_ok=True)
    os.makedirs(args.neg_dir, exist_ok=True)

    # ===== 1) create env (like PPO evaluate-style, but no video) =====
    env_kwargs = dict(obs_mode=args.obs_mode, sim_backend=args.sim_backend)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    

    env = gym.make(
        args.env_id,
        num_envs=1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    # Use ManiSkillVectorEnv for consistent batched tensors (batch=1)
    env = ManiSkillVectorEnv(env, num_envs=1, ignore_terminations=not args.partial_reset, record_metrics=True)
    action_low = torch.tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    # ===== 2) load model =====
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    pos, neg = [], []
    pos_num = neg_num = 0
    total_pos = 0
    for ep in range(args.n):
        # reset
        obs, _ = env.reset(seed=args.seed + ep)

        # get cube actor AFTER reset (safe if reconfiguration happens)
        base_env = env._env.unwrapped
        cube_actor = base_env.cubeA  # StackCube: 红色被操纵方块

        # 一次性自检 obs 维度,防止后续 input_dim 不匹配
        if ep == 0:
            obs_dim = int(np.prod(to_np(obs).shape)) if obs.ndim == 1 else int(obs.shape[-1])
            print(f"[stage1_collect] detected state obs dim = {obs_dim} (expected {EXPECTED_OBS_DIM})")
            if obs_dim != EXPECTED_OBS_DIM:
                print(f"[stage1_collect][WARN] obs dim {obs_dim} != {EXPECTED_OBS_DIM};"
                      f" 请同步更新 Reward_Model.input_dim / stage2_process.step_dim / dqn.view 维度。")

        data_episode = {
            "obs": [],
            "action": [],
            "reward": [],
            "force": [], 
            "success": 0,  # episode-level label: success_once
        }

        success_once = False
        truncated_once = False
        done = False
        step = 0
        while not done:
            with torch.no_grad():
                action = agent.get_action(obs.to(device), deterministic=True)

            # ===== disturbance: 3D discrete grid sampling (-1..1 step 0.2 per dim) =====
            # StackCube: 从 fx, fy, fz 三个维度各自独立离散采样
            #   每个维度 ∈ {-1.0, -0.8, ..., 0.8, 1.0} (11 个值)
            #   总组合 = 11^3 = 1331。不做归一化，直接乘以 force_mag。
            if args.force_mag > 0 and np.random.rand() < args.force_prob:
                # 每个分量在 [-5, 5] 间随机选整数，再 * 0.2 得到 -1..1 步长 0.2
                f_unit = (np.random.randint(-5, 6, size=3).astype(np.float32)) * 0.2
                f_applied = f_unit * args.force_mag
                if args.xy_only:
                    f_applied[2] = 0.0
            else:
                f_unit = np.zeros(3, dtype=np.float32)
                f_applied = np.zeros(3, dtype=np.float32)

            # apply force: use tensor on GPU sim, numpy float32 on CPU sim
            if args.sim_backend == "physx_cuda":
                ft = torch.from_numpy(f_applied).to(device)
                cube_actor.apply_force(ft.view(1, 3))
            else:
                # PhysX CPU expects numpy float32 shape (3,)
                cube_actor.apply_force(f_applied.astype(np.float32))

            # record transition (store obs BEFORE step, like many collectors)
            data_episode["obs"].append(to_np(obs[0]))
            data_episode["action"].append(to_np(action[0]))
            # record UNIT force (-1..1, pre-force_mag scaling) — this is what the
            # criticality model receives as its 3-dim force feature.
            data_episode["force"].append(f_unit.copy())
            action = torch.clamp(action, action_low, action_high)
            # step
            next_obs, reward, terminated, truncated, info = env.step(action)
            data_episode["reward"].append(float(to_np(reward[0]).item()))

            # success_once (NO final_info)
            # StackCube should provide info["success"] each step
            if "_final_info" in info and bool(info["_final_info"][0].item()):
                ep_metrics = info["final_info"]["episode"]
                # ep_metrics["success_once"] 是 shape (1,) 的 tensor/bool
                success_once = bool(ep_metrics["success_once"][0].item())
            else:
                # fallback: treat terminated as success signal if success not provided
                success_once = success_once

            truncated_once = truncated_once or bool(truncated[0].item())

            obs = next_obs
            done = bool((terminated | truncated)[0].item())

        data_episode["success"] = 1 if success_once else 0
        step +=1
        # ===== pos/neg split (like lunarlander script style) =====
        if success_once:
            neg.append(data_episode)
        else:
            # keep the same idea: ignore truncated episodes for negative set
            pos.append(data_episode)

        if (ep + 1) % args.save_interval == 0:
            out_pos = os.path.join(args.pos_dir, f"pos_{args.worker_id}.npy")
            out_neg = os.path.join(args.neg_dir, f"neg_{args.worker_id}.npy")
            # Use pickle protocol 4 to allow serializing objects >4GiB
            with open(out_pos, "wb") as f:
                pickle.dump(np.array(pos, dtype=object), f, protocol=4)
            with open(out_neg, "wb") as f:
                pickle.dump(np.array(neg, dtype=object), f, protocol=4)

        if (ep + 1) % 10 == 0:
            print(f"[{ep+1}/{args.n}] pos_buf={pos_num} neg_buf={neg_num} total_pos={total_pos} last_ep_success={int(success_once)}")

    env.close()
    out_pos = os.path.join(args.pos_dir, f"pos_{args.worker_id}.npy")
    out_neg = os.path.join(args.neg_dir, f"neg_{args.worker_id}.npy")
    with open(out_pos, "wb") as f:
        pickle.dump(np.array(pos, dtype=object), f, protocol=4)
    with open(out_neg, "wb") as f:
        pickle.dump(np.array(neg, dtype=object), f, protocol=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # env / model
    p.add_argument("--env_id", type=str, default="StackCube-v1")
    p.add_argument("--checkpoint", type=str, default='examples/baselines/ppo/runs/StackCube-v1__ppo__1__1780033432/final_ckpt.pt')
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    # rollout amount / saving stylex``
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--pos_dir", type=str, default="data/stage1/positive")
    p.add_argument("--neg_dir", type=str, default="data/stage1/negative")

    # ManiSkill env knobs (keep simple)
    p.add_argument("--obs_mode", type=str, default="state")
    p.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    p.add_argument("--sim_backend", type=str, default="physx_cpu")
    p.add_argument("--reconfiguration_freq", type=int, default=None)
    p.add_argument("--partial_reset", action="store_true", default=True)

    # disturbance knobs (your "wind_power"-like params)
    p.add_argument("--force_mag", type=float, default=1.0)     # like wind_power
    p.add_argument("--force_prob", type=float, default=1.0)    # like how often wind applies
    p.add_argument("--xy_only", action="store_true", default=False)
    main(p.parse_args())