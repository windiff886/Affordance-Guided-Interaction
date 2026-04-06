"""PhysX mass update helpers.

将 Isaac Sim/Isaac Lab 后端对不同 PhysX view 的张量形状要求显式编码：

- articulation mass 更新在部分 env reset 时，仍需写回完整质量张量
- rigid object mass 更新可以按 env 子集写回
"""

from __future__ import annotations

from torch import Tensor


def build_articulation_mass_update(
    *,
    masses: Tensor,
    env_ids: Tensor,
    all_env_ids: Tensor,
    body_idx: int,
    body_masses: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """构造 articulation 质量更新 payload。

    Parameters
    ----------
    masses:
        全量质量张量，形状通常为 ``(num_envs, num_bodies)``。
    env_ids:
        需要更新的环境索引。
    body_idx:
        要更新的 articulation body 索引。
    body_masses:
        与 ``env_ids`` 对齐的新质量值。
    """
    updated = masses.clone()
    updated[env_ids, body_idx] = body_masses
    return updated, all_env_ids


def build_rigid_body_mass_update(
    *,
    masses: Tensor,
    env_ids: Tensor,
    body_masses: Tensor,
) -> tuple[Tensor, Tensor]:
    """构造 rigid object 质量更新 payload。"""
    updated = masses.clone()
    updated[env_ids, 0] = body_masses
    return updated, env_ids
