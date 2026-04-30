"""Gripper hold target helpers.

保持该模块不依赖 Isaac Lab，便于在普通 pytest 中直接验证角度逻辑。
"""

from __future__ import annotations

import torch


def build_gripper_hold_targets(
    num_envs: int,
    neutral_deg: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build per-env left/right gripper hold targets in radians.

    Returns a ``(N, 2)`` tensor where both the left and right gripper target
    angles are set to *neutral_deg* (converted to radians).
    Both grippers always use the same neutral position.

    Parameters
    ----------
    num_envs:
        Number of parallel environments (``N``).
    neutral_deg:
        Neutral gripper hold angle in degrees.
    device:
        Torch device for the returned tensor.
    """
    target_rad = torch.tensor(float(neutral_deg), dtype=torch.float32, device=device)
    target_rad = torch.deg2rad(target_rad)
    # Shape (N, 2): [left_target, right_target]
    return target_rad.expand(num_envs, 2).clone()
