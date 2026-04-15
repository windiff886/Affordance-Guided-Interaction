"""Gripper hold target helpers.

保持该模块不依赖 Isaac Lab，便于在普通 pytest 中直接验证角度逻辑。
"""

from __future__ import annotations

import torch


def build_gripper_hold_targets(
    left_occupied: torch.Tensor,
    right_occupied: torch.Tensor,
    occupied_deg: float,
    unoccupied_deg: float,
) -> torch.Tensor:
    """Build per-env left/right gripper hold targets in radians.

    Parameters
    ----------
    left_occupied, right_occupied:
        Shape ``(N,)`` 的布尔张量，表示左右臂是否处于持杯状态。
    occupied_deg:
        持杯侧夹爪保持角（单位：度）。
    unoccupied_deg:
        非持杯侧夹爪保持角（单位：度）。
    """
    if left_occupied.shape != right_occupied.shape:
        raise ValueError("left_occupied and right_occupied must have the same shape")

    occupied = torch.tensor(float(occupied_deg), device=left_occupied.device)
    unoccupied = torch.tensor(float(unoccupied_deg), device=left_occupied.device)

    left_target_deg = torch.where(left_occupied, occupied, unoccupied)
    right_target_deg = torch.where(right_occupied, occupied, unoccupied)
    targets_deg = torch.stack((left_target_deg, right_target_deg), dim=-1)
    return torch.deg2rad(targets_deg.to(dtype=torch.float32))
