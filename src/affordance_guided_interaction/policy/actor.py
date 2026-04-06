"""完整 Actor 网络：多分支 encoder → RecurrentBackbone → ActionHead。

负责将 ``DirectRLEnvAdapter`` 输出的嵌套字典观测拆解为分支张量，
分别编码后拼接、经循环主干网络产生隐状态特征，最终交由动作头
输出双臂 12 维关节力矩。

数据流概览::

    actor_obs
      ├── proprio   → MLP → f_proprio
      ├── ee         → MLP → f_ee
      ├── context    → 直接拼接
      ├── stability  → MLP → f_stab
      └── visual     → MLP → f_vis
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

from .recurrent_backbone import RecurrentBackbone
from .action_head import ActionHead


# ======================================================================
# 数据规格常量（与 observations/ 保持一致）
# ======================================================================

NUM_JOINTS_PER_ARM = 6
TOTAL_ARM_JOINTS = NUM_JOINTS_PER_ARM * 2  # 12
DOOR_EMBEDDING_DIM = 768
# visual 分支输入维度 = embedding(768) + visual_valid(1)
VISUAL_BRANCH_DIM = DOOR_EMBEDDING_DIM + 1

# 每条臂的末端状态维度:
# pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + lin_acc(3) + ang_acc(3) = 19
_SINGLE_EE_DIM = 19
_DUAL_EE_DIM = _SINGLE_EE_DIM * 2  # 38

# context 维度: left_occ(1) + right_occ(1) = 2
_CONTEXT_DIM = 2

# 单臂稳定性仅保留 tilt
_SINGLE_STAB_DIM = 1
_DUAL_STAB_DIM = _SINGLE_STAB_DIM * 2  # 2


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

def _t(v) -> torch.Tensor:
    """将任意值（CUDA tensor / CPU tensor / numpy array）转换为 1-D float32 CPU tensor。

    适配器将 obs 存储为 GPU tensor 切片；展平后由 rollout_collector 统一 .to(device)。
    """
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().float().reshape(-1)
    return torch.as_tensor(v, dtype=torch.float32).reshape(-1)


def flatten_actor_obs(obs: dict, cfg: ActorConfig) -> dict[str, torch.Tensor]:
    """将 ``DirectRLEnvAdapter`` 输出的嵌套字典展平为分支张量。

    返回的字典包含以下 key，各值均为 ``(feature_dim,)`` 的 1-D 张量：

    - ``"proprio"`` — 双臂关节位姿 + 速度 + [力矩] + 上一步动作
    - ``"ee"``      — 左右臂末端状态拼接
    - ``"context"`` — left_occ + right_occ
    - ``"stability"`` — 左右臂 tilt 拼接
    - ``"visual"``  — door_embedding

    Parameters
    ----------
    obs : dict
        ``DirectRLEnvAdapter`` 返回的 actor_obs 字典。
    cfg : ActorConfig
        配置（决定是否包含 torque 等）。
    """
    proprio = obs["proprio"]

    # -- proprio --
    parts = [
        _t(proprio["joint_positions"]),
        _t(proprio["joint_velocities"]),
    ]
    if cfg.include_torques:
        parts.append(_t(proprio.get("joint_torques", torch.zeros(TOTAL_ARM_JOINTS))))
    parts.append(_t(proprio["prev_action"]))
    proprio_vec = torch.cat(parts)

    # -- ee --
    left_ee = obs["ee"]["left"]
    right_ee = obs["ee"]["right"]
    ee_vec = torch.cat([
        _t(left_ee["position"]),
        _t(left_ee["orientation"]),
        _t(left_ee["linear_velocity"]),
        _t(left_ee["angular_velocity"]),
        _t(left_ee["linear_acceleration"]),
        _t(left_ee["angular_acceleration"]),
        _t(right_ee["position"]),
        _t(right_ee["orientation"]),
        _t(right_ee["linear_velocity"]),
        _t(right_ee["angular_velocity"]),
        _t(right_ee["linear_acceleration"]),
        _t(right_ee["angular_acceleration"]),
    ])

    # -- context --
    ctx = obs["context"]
    ctx_vec = torch.cat([_t(ctx["left_occupied"]), _t(ctx["right_occupied"])])

    # -- stability --
    stab = obs["stability"]
    stab_vec = torch.cat([_t(stab["left_tilt"]), _t(stab["right_tilt"])])

    # -- visual: embedding(768) + visual_valid(1) = 769 --
    vis_vec = torch.cat([
        _t(obs["visual"]["door_embedding"]),
        _t(obs["visual"].get("visual_valid", torch.zeros(1))),
    ])

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
        每个元素为 ``DirectRLEnvAdapter`` 返回的单步字典。
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
            TOTAL_ARM_JOINTS * 2  # q + dq
            + (TOTAL_ARM_JOINTS if c.include_torques else 0)  # tau
            + TOTAL_ARM_JOINTS  # prev_action
        )
        stab_in = _DUAL_STAB_DIM

        # -- 分支 encoder --
        self.proprio_encoder = _build_branch_encoder(proprio_in, c.proprio_hidden, c.proprio_out)
        self.ee_encoder = _build_branch_encoder(_DUAL_EE_DIM, c.ee_hidden, c.ee_out)
        self.stab_encoder = _build_branch_encoder(stab_in, c.stab_hidden, c.stab_out)
        self.vis_encoder = _build_branch_encoder(VISUAL_BRANCH_DIM, c.vis_hidden, c.vis_out)

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
        ctx = flat_obs["context"]                              # (B, 2)  # left_occ + right_occ

        # 拼接
        concat = torch.cat([f_proprio, f_ee, ctx, f_stab, f_vis], dim=-1).contiguous()

        # 循环主干
        backbone_out, hidden_new = self.backbone(concat, hidden)

        # 采样动作（单次 forward 同时得到 log_prob + entropy）
        action, log_prob, entropy = self.action_head.sample_with_entropy(backbone_out)

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

        concat = torch.cat([f_proprio, f_ee, ctx, f_stab, f_vis], dim=-1).contiguous()
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
