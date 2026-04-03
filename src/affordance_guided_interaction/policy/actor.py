"""完整 Actor 网络：多分支 encoder → RecurrentBackbone → ActionHead。

负责将 ``ActorObsBuilder`` 输出的嵌套字典观测拆解为分支张量，
分别编码后拼接、经循环主干网络产生隐状态特征，最终交由动作头
输出双臂 12 维关节力矩。

数据流概览::

    actor_obs
      ├── proprio   → MLP → f_proprio
      ├── gripper    → MLP → f_ee
      ├── context    → 直接拼接
      ├── stability  → MLP → f_stab
      └── z_aff      → MLP → f_vis
              ↓ concat
      RecurrentBackbone (GRU / LSTM)
              ↓
        ActionHead (Gaussian)
              ↓
        τ ∈ R^12
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import numpy as np

from .recurrent_backbone import RecurrentBackbone
from .action_head import ActionHead


# ======================================================================
# 数据规格常量（与 observations/ 保持一致）
# ======================================================================

NUM_JOINTS_PER_ARM = 6
TOTAL_ARM_JOINTS = NUM_JOINTS_PER_ARM * 2  # 12
Z_AFF_DIM = 768

# 每条臂的 gripper 状态维度: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13
_SINGLE_EE_DIM = 13
_DUAL_EE_DIM = _SINGLE_EE_DIM * 2  # 26

# context 维度: left_occ(1) + right_occ(1) = 2
_CONTEXT_DIM = 2

# 单臂稳定性 proxy 维度:
#   tilt(1) + lin_vel_norm(1) + lin_acc(3) + ang_vel_norm(1)
#   + ang_acc(3) + jerk(1) + recent_acc_history(k)
# 默认 k = 10 → 单臂 20，双臂 40
_DEFAULT_ACC_HISTORY_LEN = 10
_SINGLE_STAB_DIM = 1 + 1 + 3 + 1 + 3 + 1 + _DEFAULT_ACC_HISTORY_LEN  # 20
_DUAL_STAB_DIM = _SINGLE_STAB_DIM * 2  # 40


# ======================================================================
# 配置
# ======================================================================

@dataclass
class ActorConfig:
    """Actor 网络超参配置。"""

    # 分支 encoder 隐层维度
    proprio_hidden: int = 128
    proprio_out: int = 64
    ee_hidden: int = 64
    ee_out: int = 32
    stab_hidden: int = 64
    stab_out: int = 32
    vis_hidden: int = 256
    vis_out: int = 128

    # RecurrentBackbone
    rnn_hidden: int = 512
    rnn_layers: int = 1
    rnn_type: Literal["gru", "lstm"] = "gru"

    # ActionHead
    action_dim: int = TOTAL_ARM_JOINTS
    log_std_init: float = -0.5

    # 动作历史步数（与 ActorObsBuilder 保持一致）
    action_history_length: int = 3
    # 稳定性 proxy 加速度历史长度
    acc_history_length: int = _DEFAULT_ACC_HISTORY_LEN
    # 是否包含关节力矩输入
    include_torques: bool = True


# ======================================================================
# 分支 MLP 工厂
# ======================================================================

def _build_branch_encoder(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """构建一个 2 层 MLP 分支 encoder：Linear → LayerNorm → ReLU → Linear。"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


# ======================================================================
# 观测展平工具
# ======================================================================

def flatten_actor_obs(obs: dict, cfg: ActorConfig) -> dict[str, torch.Tensor]:
    """将 ``ActorObsBuilder.build()`` 输出的嵌套字典展平为分支张量。

    返回的字典包含以下 key，各值均为 ``(feature_dim,)`` 的 1-D 张量：

    - ``"proprio"`` — 关节位姿 + 速度 + [力矩] + 动作历史
    - ``"ee"``      — 左右臂 gripper 状态拼接
    - ``"context"`` — left_occ + right_occ
    - ``"stability"`` — 左右臂稳定性 proxy 拼接
    - ``"visual"``  — z_aff

    Parameters
    ----------
    obs : dict
        ``ActorObsBuilder.build()`` 返回的 actor_obs 字典。
    cfg : ActorConfig
        配置（决定是否包含 torque 等）。
    """
    proprio = obs["proprio"]

    # -- proprio --
    parts = [
        np.asarray(proprio["left_joint_positions"]).ravel(),
        np.asarray(proprio["left_joint_velocities"]).ravel(),
        np.asarray(proprio["right_joint_positions"]).ravel(),
        np.asarray(proprio["right_joint_velocities"]).ravel(),
    ]
    if cfg.include_torques:
        parts.append(
            np.asarray(proprio.get("left_joint_torques", np.zeros(NUM_JOINTS_PER_ARM))).ravel()
        )
        parts.append(
            np.asarray(proprio.get("right_joint_torques", np.zeros(NUM_JOINTS_PER_ARM))).ravel()
        )
    # 动作历史: (k, 12) → 展平为 (k*12,)
    prev_actions = np.asarray(proprio["previous_actions"])
    parts.append(prev_actions.ravel())

    proprio_vec = torch.from_numpy(np.concatenate(parts)).float()

    # -- ee --
    left_ee = obs["left_gripper_state"]
    right_ee = obs["right_gripper_state"]
    ee_vec = torch.from_numpy(np.concatenate([
        np.asarray(left_ee["position"]).ravel(),
        np.asarray(left_ee["orientation"]).ravel(),
        np.asarray(left_ee["linear_velocity"]).ravel(),
        np.asarray(left_ee["angular_velocity"]).ravel(),
        np.asarray(right_ee["position"]).ravel(),
        np.asarray(right_ee["orientation"]).ravel(),
        np.asarray(right_ee["linear_velocity"]).ravel(),
        np.asarray(right_ee["angular_velocity"]).ravel(),
    ])).float()

    # -- context --
    ctx = obs["context"]
    ctx_vec = torch.from_numpy(np.concatenate([
        np.asarray(ctx["left_occupied"]).ravel(),
        np.asarray(ctx["right_occupied"]).ravel(),
    ])).float()

    # -- stability --
    def _flatten_proxy(proxy: dict) -> np.ndarray:
        return np.concatenate([
            np.atleast_1d(proxy["tilt"]).ravel(),
            np.atleast_1d(proxy["linear_velocity_norm"]).ravel(),
            np.asarray(proxy["linear_acceleration"]).ravel(),
            np.atleast_1d(proxy["angular_velocity_norm"]).ravel(),
            np.asarray(proxy["angular_acceleration"]).ravel(),
            np.atleast_1d(proxy["jerk_proxy"]).ravel(),
            np.asarray(proxy["recent_acc_history"]).ravel(),
        ])

    stab_vec = torch.from_numpy(np.concatenate([
        _flatten_proxy(obs["left_stability_proxy"]),
        _flatten_proxy(obs["right_stability_proxy"]),
    ])).float()

    # -- visual --
    vis_vec = torch.from_numpy(
        np.asarray(obs["z_aff"]).ravel()
    ).float()

    return {
        "proprio": proprio_vec,
        "ee": ee_vec,
        "context": ctx_vec,
        "stability": stab_vec,
        "visual": vis_vec,
    }


def batch_flatten_actor_obs(
    obs_list: list[dict], cfg: ActorConfig
) -> dict[str, torch.Tensor]:
    """将多个 actor_obs 字典展平并 stack 为 batch 张量。

    Parameters
    ----------
    obs_list : list[dict]
        每个元素为 ``ActorObsBuilder.build()`` 返回的单步字典。
    cfg : ActorConfig

    Returns
    -------
    dict[str, torch.Tensor]
        每个值的形状为 ``(batch, feature_dim)``。
    """
    flat_list = [flatten_actor_obs(o, cfg) for o in obs_list]
    keys = flat_list[0].keys()
    return {k: torch.stack([f[k] for f in flat_list], dim=0) for k in keys}


# ======================================================================
# Actor 网络
# ======================================================================

class Actor(nn.Module):
    """完整 Actor：多分支 encoder → RecurrentBackbone → ActionHead。

    Parameters
    ----------
    cfg : ActorConfig
        超参配置，默认使用合理缺省值。
    """

    def __init__(self, cfg: ActorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ActorConfig()
        c = self.cfg

        # -- 计算各分支输入维度 --
        proprio_in = (
            NUM_JOINTS_PER_ARM * 4  # 左右 q + dq = 4 × 6 = 24
            + (NUM_JOINTS_PER_ARM * 2 if c.include_torques else 0)  # tau
            + c.action_history_length * TOTAL_ARM_JOINTS  # prev_actions
        )
        stab_in = (1 + 1 + 3 + 1 + 3 + 1 + c.acc_history_length) * 2  # 双臂

        # -- 分支 encoder --
        self.proprio_encoder = _build_branch_encoder(proprio_in, c.proprio_hidden, c.proprio_out)
        self.ee_encoder = _build_branch_encoder(_DUAL_EE_DIM, c.ee_hidden, c.ee_out)
        self.stab_encoder = _build_branch_encoder(stab_in, c.stab_hidden, c.stab_out)
        self.vis_encoder = _build_branch_encoder(Z_AFF_DIM, c.vis_hidden, c.vis_out)

        # -- 汇总维度 --
        concat_dim = c.proprio_out + c.ee_out + _CONTEXT_DIM + c.stab_out + c.vis_out

        # -- 循环主干 --
        self.backbone = RecurrentBackbone(
            input_dim=concat_dim,
            hidden_dim=c.rnn_hidden,
            num_layers=c.rnn_layers,
            rnn_type=c.rnn_type,
        )

        # -- 动作头 --
        self.action_head = ActionHead(
            input_dim=c.rnn_hidden,
            action_dim=c.action_dim,
            log_std_init=c.log_std_init,
        )

    # ------------------------------------------------------------------
    # 隐状态管理
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int = 1):
        """创建全零隐状态，在 episode 开始时调用。"""
        return self.backbone.init_hidden(batch_size)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(
        self,
        flat_obs: dict[str, torch.Tensor],
        hidden=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """Actor 前向：编码 → 循环 → 采样。

        Parameters
        ----------
        flat_obs : dict[str, Tensor]
            由 ``flatten_actor_obs()`` 或 ``batch_flatten_actor_obs()``
            生成的分支张量字典，每个值形状为 ``(batch, dim)``。
        hidden
            上一步的循环隐状态，None 表示 episode 开始。

        Returns
        -------
        action : ``(batch, 12)``
        log_prob : ``(batch,)``
        entropy : ``(batch,)``
        hidden_new
            更新后的循环隐状态。
        """
        # 各分支编码
        f_proprio = self.proprio_encoder(flat_obs["proprio"])  # (B, proprio_out)
        f_ee = self.ee_encoder(flat_obs["ee"])                 # (B, ee_out)
        f_stab = self.stab_encoder(flat_obs["stability"])      # (B, stab_out)
        f_vis = self.vis_encoder(flat_obs["visual"])            # (B, vis_out)
        ctx = flat_obs["context"]                              # (B, 3)

        # 拼接
        concat = torch.cat([f_proprio, f_ee, ctx, f_stab, f_vis], dim=-1)

        # 循环主干
        backbone_out, hidden_new = self.backbone(concat, hidden)

        # 采样动作
        action, log_prob = self.action_head.sample(backbone_out)
        _, entropy = self.action_head.evaluate(backbone_out, action)

        return action, log_prob, entropy, hidden_new

    def act(
        self,
        flat_obs: dict[str, torch.Tensor],
        hidden=None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, object]:
        """推理接口（部署/评估时使用）。

        Parameters
        ----------
        flat_obs : dict[str, Tensor]
            分支张量字典。
        hidden
            循环隐状态。
        deterministic : bool
            True 时返回均值动作，False 时采样。

        Returns
        -------
        action : ``(batch, 12)``
        hidden_new
        """
        # 各分支编码
        f_proprio = self.proprio_encoder(flat_obs["proprio"])
        f_ee = self.ee_encoder(flat_obs["ee"])
        f_stab = self.stab_encoder(flat_obs["stability"])
        f_vis = self.vis_encoder(flat_obs["visual"])
        ctx = flat_obs["context"]

        concat = torch.cat([f_proprio, f_ee, ctx, f_stab, f_vis], dim=-1)
        backbone_out, hidden_new = self.backbone(concat, hidden)

        if deterministic:
            action = self.action_head.deterministic(backbone_out)
        else:
            action, _ = self.action_head.sample(backbone_out)

        return action, hidden_new

    def evaluate_actions(
        self,
        flat_obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        hidden=None,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """PPO 更新时用：给定动作序列，计算 log_prob 和 entropy。

        Parameters
        ----------
        flat_obs : dict[str, Tensor]
        actions : ``(batch, 12)``
        hidden : 隐状态

        Returns
        -------
        log_prob : ``(batch,)``
        entropy : ``(batch,)``
        hidden_new
        """
        f_proprio = self.proprio_encoder(flat_obs["proprio"])
        f_ee = self.ee_encoder(flat_obs["ee"])
        f_stab = self.stab_encoder(flat_obs["stability"])
        f_vis = self.vis_encoder(flat_obs["visual"])
        ctx = flat_obs["context"]

        concat = torch.cat([f_proprio, f_ee, ctx, f_stab, f_vis], dim=-1)
        backbone_out, hidden_new = self.backbone(concat, hidden)

        log_prob, entropy = self.action_head.evaluate(backbone_out, actions)
        return log_prob, entropy, hidden_new
