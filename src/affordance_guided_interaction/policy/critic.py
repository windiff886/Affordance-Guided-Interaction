"""非对称 Critic 网络：纯 MLP，接收 actor_obs + privileged 信息。

训练时使用 asymmetric actor-critic 架构——Critic 可额外访问
精确的物理状态和隐藏环境参数以辅助价值估计，但这些信息
对 Actor 不可见（Actor 只能通过循环隐状态和历史观测间接推断）。

部署时 Critic 不参与推理。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .actor import (
    ActorConfig,
    flatten_actor_obs,
    _build_branch_encoder,
    _t,
    TOTAL_ARM_JOINTS,
    VISUAL_BRANCH_DIM,
    _DUAL_EE_DIM,
    _CONTEXT_DIM,
    _DUAL_STAB_DIM,
)


# Privileged 信息维度:
#   door_pose(7) + door_joint_pos(1) + door_joint_vel(1)
#   + cup_mass(1) + door_mass(1) + door_damping(1) + base_pos(3)
#   + cup_dropped(1) = 16
_PRIVILEGED_DIM = 16


# ======================================================================
# 配置
# ======================================================================

@dataclass
class CriticConfig:
    """Critic 网络超参配置。"""

    # Critic MLP 各层维度
    hidden_dims: tuple[int, ...] = (512, 256, 128)


# ======================================================================
# 观测展平工具
# ======================================================================

def flatten_privileged(privileged: dict) -> torch.Tensor:
    """将 ``DirectRLEnvAdapter`` 输出的 privileged 字典展平为 1-D 张量。

    使用 ``_t()`` helper 处理任意输入类型（CUDA tensor / CPU tensor / numpy），
    与 ``flatten_actor_obs()`` 保持一致（P1 修复）。

    Parameters
    ----------
    privileged : dict
        ``critic_obs["privileged"]`` 字典。

    Returns
    -------
    torch.Tensor — ``(16,)``
    """
    parts = [
        _t(privileged["door_pose"]),        # 7
        _t(privileged["door_joint_pos"]),    # 1
        _t(privileged["door_joint_vel"]),    # 1
        _t(privileged["cup_mass"]),          # 1
        _t(privileged["door_mass"]),         # 1
        _t(privileged["door_damping"]),      # 1
        _t(privileged["base_pos"]),          # 3
        _t(privileged["cup_dropped"]),       # 1
    ]
    return torch.cat(parts)


def flatten_privileged_tensor(
    critic_obs: torch.Tensor,
    actor_obs_dim: int = 858,
) -> torch.Tensor:
    """从 flat critic obs tensor 直接切出 privileged 分支。"""
    return critic_obs[:, actor_obs_dim:]


def flatten_critic_obs(
    critic_obs: dict, actor_cfg: ActorConfig
) -> dict[str, torch.Tensor]:
    """将完整 critic_obs 展平为分支张量。

    Parameters
    ----------
    critic_obs : dict
        ``DirectRLEnvAdapter`` 返回的字典，包含
        ``"actor_obs"`` 和 ``"privileged"`` 两个 key。
    actor_cfg : ActorConfig

    Returns
    -------
    dict 包含:
        - ``"actor_branches"`` — actor 各分支的展平张量
        - ``"privileged"`` — privileged 展平张量
    """
    actor_flat = flatten_actor_obs(critic_obs["actor_obs"], actor_cfg)
    priv_flat = flatten_privileged(critic_obs["privileged"])
    return {
        "actor_branches": actor_flat,
        "privileged": priv_flat,
    }


# ======================================================================
# Critic 网络
# ======================================================================

class Critic(nn.Module):
    """非对称 Critic：纯 MLP，无循环结构。

    输入 = actor 各分支特征拼接 + privileged 特征 → MLP → 标量 V(s)。

    Parameters
    ----------
    actor_cfg : ActorConfig
        需要与 Actor 使用的配置一致，以便复用分支 encoder 或
        计算 actor 侧拼接维度。
    critic_cfg : CriticConfig | None
        Critic 独有的超参，默认使用合理缺省值。
    """

    def __init__(
        self,
        actor_cfg: ActorConfig | None = None,
        critic_cfg: CriticConfig | None = None,
    ) -> None:
        super().__init__()
        self.actor_cfg = actor_cfg or ActorConfig()
        self.critic_cfg = critic_cfg or CriticConfig()

        ac = self.actor_cfg

        # -- 独立的 actor 侧 encoder（与 Actor 权重不共享）--
        # L10: 与 Actor 保持相同公式，使用 TOTAL_ARM_JOINTS 避免隐式关系
        proprio_in = (
            TOTAL_ARM_JOINTS * 2                                     # q + dq
            + (TOTAL_ARM_JOINTS if ac.include_torques else 0)        # tau
            + TOTAL_ARM_JOINTS                                       # prev_action
        )
        stab_in = _DUAL_STAB_DIM  # L11: 使用常量而非硬编码 2

        self.proprio_encoder = _build_branch_encoder(proprio_in, ac.proprio_hidden, ac.proprio_out)
        self.ee_encoder = _build_branch_encoder(_DUAL_EE_DIM, ac.ee_hidden, ac.ee_out)
        self.stab_encoder = _build_branch_encoder(stab_in, ac.stab_hidden, ac.stab_out)
        self.vis_encoder = _build_branch_encoder(VISUAL_BRANCH_DIM, ac.vis_hidden, ac.vis_out)

        actor_concat_dim = ac.proprio_out + ac.ee_out + _CONTEXT_DIM + ac.stab_out + ac.vis_out

        # -- privileged encoder --
        self.priv_encoder = _build_branch_encoder(_PRIVILEGED_DIM, 64, 32)
        priv_out_dim = 32

        # -- 主 MLP --
        total_in = actor_concat_dim + priv_out_dim
        layers: list[nn.Module] = []
        prev_dim = total_in
        for h in self.critic_cfg.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(
        self,
        actor_flat: dict[str, torch.Tensor],
        privileged: torch.Tensor,
    ) -> torch.Tensor:
        """计算状态价值。

        Parameters
        ----------
        actor_flat : dict[str, Tensor]
            由 ``flatten_actor_obs()`` 产出的分支张量字典，
            每个值形状 ``(batch, dim)``。
        privileged : ``(batch, 16)``
            由 ``flatten_privileged()`` 产出的 privileged 张量。

        Returns
        -------
        value : ``(batch, 1)``
        """
        # actor 侧编码
        f_proprio = self.proprio_encoder(actor_flat["proprio"])
        f_ee = self.ee_encoder(actor_flat["ee"])
        f_stab = self.stab_encoder(actor_flat["stability"])
        f_vis = self.vis_encoder(actor_flat["visual"])
        ctx = actor_flat["context"]

        actor_features = torch.cat([f_proprio, f_ee, ctx, f_stab, f_vis], dim=-1)

        # privileged 编码
        priv_features = self.priv_encoder(privileged)

        # 拼接送入主 MLP
        combined = torch.cat([actor_features, priv_features], dim=-1)
        return self.value_head(combined)

    def evaluate(
        self,
        critic_flat: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """便捷接口：直接接收 ``flatten_critic_obs()`` 的输出。

        M5 修复：flatten_critic_obs() 产出 CPU tensor（_t() 搬到 CPU），
        此处自动迁移到模型所在 device，避免直接调用时 CPU/GPU 设备不匹配崩溃。

        Parameters
        ----------
        critic_flat : dict
            包含 ``"actor_branches"`` 和 ``"privileged"`` 两个 key。

        Returns
        -------
        value : ``(batch, 1)``
        """
        device = next(self.parameters()).device
        actor_branches = {k: v.to(device) for k, v in critic_flat["actor_branches"].items()}
        privileged = critic_flat["privileged"].to(device)
        return self.forward(actor_branches, privileged)
