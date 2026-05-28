from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # register envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from examples.baselines.ppo.ppo import Agent
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_robot_qpos_qvel(agent):
    rob = getattr(agent, "robot", None)
    if rob is None:
        rob = agent
    qpos = getattr(rob, "qpos", None)
    qvel = getattr(rob, "qvel", None)
    return qpos, qvel


@dataclass
class NADEConfig:
    # StackCube: state obs is 48-dim
    base_feature_dim: int = 48
    env_param_dim: int = 3

    grid_size: int = 11
    xy_only: bool = False

    force_mag: float = 0.6
    force_prob: float = 1.0

    # recompute model-guided proposal center every k steps,
    # but sample a fresh continuous action every step.
    update_every: int = 1
    history_len: int = 1

    reward_threshold: float = 5.0
    kappa: float = 12.0

    # keep exact IS by default: no clipping / fallback in strict mode
    strict_is: bool = True

    # once cumulative importance weight becomes too small,
    # stop challenge actions and fall back to baseline sampling.
    weight_threshold: float = 1e-3



class ManiSkillOrdinaryNADE(gym.Wrapper):
    def __init__(self, env, cfg, args):
        # ManiSkillVectorEnv may not be a subclass of gym.Env, which makes
        # gym.Wrapper.__init__ assert. Try using the parent init when
        # appropriate, otherwise fall back to assigning self.env directly.
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            # fallback: don't call gym.Wrapper.__init__, just attach env
            self.env = env
        self.cfg = cfg
        self.device = torch.device(args.device)
        self.args = args

        # 1. 实例化模型架构 (SimpleClassifier per-step MLP)
        from criticality.utils.criticality_model import SimpleClassifier
        # StackCube: input_dim = 48 (obs) + 3 (force fx,fy,fz) = 51
        input_dim = 51
        self.criticality_model1 = SimpleClassifier(
            input_dim=input_dim,
            hidden=getattr(cfg, "hidden", 256),
            hidden_layer=getattr(cfg, "hidden_layer", 3),
        ).to(self.device)

        # StackCube: 11^3 = 1331 discrete (fx, fy, fz) candidates,
        # each component ∈ {-1.0, -0.8, ..., 0.8, 1.0}.
        vals = torch.arange(-1.0, 1.001, 0.2, device=self.device, dtype=torch.float32)
        fx_g, fy_g, fz_g = torch.meshgrid(vals, vals, vals, indexing="ij")
        self.force_grid = torch.stack([fx_g, fy_g, fz_g], dim=-1).reshape(-1, 3)  # (1331, 3)
        self.force_grid_np = self.force_grid.detach().cpu().numpy()
        self.total_actions = self.force_grid.shape[0]  # 1331

        # 2. 正确加载权重
        if args.criticality_ckpt is not None:
            print(f"正在从 {args.criticality_ckpt} 加载权重...")
            ckpt = torch.load(args.criticality_ckpt, map_location=self.device)
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt
            self.criticality_model1.load_state_dict(state_dict)

        self.criticality_model1.eval()
        self.current_state = None
        self.step_count = 0
        self.total_weight = 1.0
        self.env_action = np.zeros(3)

        # 初始化 criticality_info
        self.criticality_info = {
            "weight": 1.0,
            "p_list": np.ones(self.total_actions) / self.total_actions
        }

        # 指标记录
        self.record_metrics = True
        self.returns = []
        self.success_once = False
        self.fail_once = False

    # ---------- env plumbing ----------
    def _get_base_env(self):
        env = self.env
        if hasattr(env, "_env"):
            env = env._env
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        return env

    def get_wrapper_attr(self, name: str):
        """Delegate get_wrapper_attr to the underlying env if present."""
        if hasattr(self.env, "get_wrapper_attr"):
            return self.env.get_wrapper_attr(name)
        # fallback: try attribute access on the underlying env
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"Underlying env has no attribute '{name}'")

    @property
    def single_observation_space(self):
        if hasattr(self.env, "single_observation_space"):
            return getattr(self.env, "single_observation_space")
        if hasattr(self.env, "observation_space"):
            return getattr(self.env, "observation_space")
        # try wrapper accessor
        try:
            return self.get_wrapper_attr("single_observation_space")
        except Exception:
            raise AttributeError("Underlying env has no observation space attribute")

    @property
    def single_action_space(self):
        if hasattr(self.env, "single_action_space"):
            return getattr(self.env, "single_action_space")
        if hasattr(self.env, "action_space"):
            return getattr(self.env, "action_space")
        try:
            return self.get_wrapper_attr("single_action_space")
        except Exception:
            raise AttributeError("Underlying env has no action space attribute")

    def _format_model_output(self, outputs: Any) -> np.ndarray:
        # SimpleClassifier returns logits of shape (B, 2); convert to P(critical).
        if not torch.is_tensor(outputs):
            outputs = torch.as_tensor(outputs, device=self.device)
        probs = torch.softmax(outputs, dim=-1)[:, 1]
        return probs.detach().cpu().numpy()
    
    def _extract_state(self, obs):
        """辅助函数：将观测统一展平为 1D Numpy 数组"""
        if isinstance(obs, dict):
            state = common.to_numpy(common.flatten_state_dict(obs))
        else:
            state = common.to_numpy(obs)
        if len(state.shape) > 1: state = state[0]
        return state

    def calcu_q(self, obs=None):
        # Per-step SimpleClassifier: enumerate all 1331 (fx, fy, fz) candidates
        # over the same current observation -> batch shape (1331, 51).
        total_points = self.total_actions  # 1331

        cur_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)\
            .unsqueeze(0).repeat(total_points, 1)  # (1331, 48)
        cur_input = torch.cat([cur_obs, self.force_grid], dim=1)  # (1331, 51)

        with torch.no_grad():
            outputs = self.criticality_model1(cur_input)  # (1331, 2) logits
        return outputs
    
    def idx_to_action(self, action_idx):
        # StackCube: action_idx ∈ [0, 1331)，直接索引 11^3 离散 (fx, fy, fz) 网格
        return self.force_grid_np[int(action_idx)].astype(np.float32).copy()

    def get_env_action(self, obs: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        total_samples = self.total_actions  # 1331
        p_list = np.ones(total_samples) / total_samples

        if np.random.rand() > self.cfg.force_prob:
            return np.zeros(3), {"weight": 1.0, "p_list": p_list}

        if self.args.nade:
            outputs = self.calcu_q(obs)
            scores = self._format_model_output(outputs) 
            criticality = scores

            if np.max(criticality) > self.args.criticality_threshold:
                criticality_pdf = criticality / np.sum(criticality)
                epsilon = self.args.epsilon
                pdf_array = (1 - epsilon) * criticality_pdf + epsilon * p_list
                pdf_array /= np.sum(pdf_array) 
            else:
                pdf_array = p_list
        else:
            pdf_array = p_list

        pdf_array = pdf_array.astype(np.float64)
        action_idx = np.random.choice(total_samples, p=pdf_array)
        
        weight = p_list[action_idx] / pdf_array[action_idx]

        return self.idx_to_action(action_idx), {"weight": weight, "p_list": p_list}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            state = common.to_numpy(common.flatten_state_dict(obs))
        else:
            state = common.to_numpy(obs)

        if len(state.shape) > 1: state = state[0]
        self.current_state = self._extract_state(obs)
        self.step_count = 0
        self.total_weight = 1.0
        self.env_action = np.zeros(3)

        self.success_once = False

        return obs, info

    def step(self, action):
        if (self.step_count % self.cfg.update_every) == 0:
            new_env_action, criticality_info = self.get_env_action(self.current_state)
            
            threshold = self.cfg.weight_threshold
            if self.total_weight * criticality_info['weight'] < threshold:
                p_list = criticality_info['p_list']
                p_final = np.array(p_list, dtype='float64')
                p_sum = np.sum(p_final)
                p_final = np.ones_like(p_final)/len(p_final) if p_sum <= 0 else p_final/p_sum
                p_final /= np.sum(p_final)

                action_idx = np.random.choice(len(p_list), p=p_final)
                if np.random.rand() > self.cfg.force_prob:
                    self.env_action = np.zeros(3, dtype=np.float32)
                else:
                    # StackCube: 从 1331 离散 (fx, fy, fz) 网格中取出该方向
                    self.env_action = self.idx_to_action(action_idx)
                criticality_info['weight'] = 1.0
            else:
                self.env_action = new_env_action
                self.total_weight *= criticality_info['weight']

            self.criticality_info = criticality_info
            self.criticality_info['total_weight'] = self.total_weight
        else:
            self.criticality_info["weight"] = 1.0

        # 2. 取出本步施加的 3D 单位力 (fx, fy, fz) ∈ [-1, 1]
        force = self.env_action.detach().cpu().numpy().reshape(-1) if torch.is_tensor(self.env_action) else np.array(self.env_action).reshape(-1)

        # 3. 施加力到物理引擎
        ap_force = force * self.cfg.force_mag
        ap_force_tensor = torch.from_numpy(ap_force).to(self.device).float()
        # StackCube: apply force to cubeA (red cube, the one being stacked)
        if self.args.sim_backend == "physx_cuda":
            self.env.unwrapped.cubeA.apply_force(ap_force_tensor.reshape(1, 3))
        else:
            # PhysX CPU expects numpy float32 shape (3,)
            self.env.unwrapped.cubeA.apply_force(ap_force.astype(np.float32))
        
        # 4. 环境物理演进
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 5. 更新状态用于下一步
        self.current_state = self._extract_state(obs)
        
        # 6. 将当前步刚算出来的干净的权重写进 info，交给采集脚本
        info['criticality_info'] = self.criticality_info
        info['nade_env_action'] = self.env_action

        self.step_count += 1

        return common.unbatch(
            common.to_numpy(obs),
            common.to_numpy(reward),
            common.to_numpy(terminated),
            common.to_numpy(truncated),
            info,
        )


def make_env(args):
    import time
    import torch
    import gymnasium as gym
    import mani_skill.envs

    print("正在创建基础环境...")
    t0 = time.time()
    ignore_term = args.ignore_terminations if hasattr(args, "ignore_terminations") else False 
    # 1. 创建原生环境
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        render_mode=getattr(args, "render_mode", None),
        control_mode=getattr(args, "control_mode", "pd_joint_delta_pos"),
        sim_backend=args.sim_backend,
        reconfiguration_freq=None,
    )

    # 统一转换 ActionSpace
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)

    print(f"原生环境创建耗时: {time.time() - t0:.2f}s")

    # 2. 准备 NADE 配置
    print("正在加载 NADE 逻辑与模型...")
    t1 = time.time()
    nade_cfg = NADEConfig(
        grid_size=args.grid_size,
        force_mag=args.force_mag,
        force_prob=args.force_prob,
        update_every=args.update_every,
        xy_only=getattr(args, "xy_only", False),
        history_len=getattr(args, "history_len", 1)
    )
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    env = ManiSkillVectorEnv(
        env, 
        num_envs=1, 
        ignore_terminations=ignore_term, 
        record_metrics=True
    )
    # 3. 包装 NADE (严格匹配 __init__ 的位置参数)
    env = ManiSkillOrdinaryNADE(
        env,
        nade_cfg,
        args,
    )

    print(f"NADE 包装耗时: {time.time() - t1:.2f}s")

    return env