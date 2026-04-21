from __future__ import annotations

import torch
from torch import Tensor


def rescale_normalized_base_actions(
    actions: Tensor,
    *,
    max_lin_vel_x: float,
    max_lin_vel_y: float,
    max_ang_vel_z: float,
) -> Tensor:
    """Map normalized base actions in ``[-1, 1]`` to physical twist commands."""
    bounded_actions = torch.clamp(actions, -1.0, 1.0)
    scale = torch.tensor(
        [float(max_lin_vel_x), float(max_lin_vel_y), float(max_ang_vel_z)],
        dtype=bounded_actions.dtype,
        device=bounded_actions.device,
    )
    return bounded_actions * scale.unsqueeze(0)


def twist_to_wheel_angular_velocity_targets(
    twist_cmd: Tensor,
    *,
    wheel_radius: float,
    half_length: float,
    half_width: float,
) -> Tensor:
    """Map base-frame ``[vx, vy, wz]`` commands to four wheel angular velocities.

    Wheel order is ``[front_left, front_right, rear_left, rear_right]``.
    """
    vx = twist_cmd[:, 0]
    vy = twist_cmd[:, 1]
    wz = twist_cmd[:, 2]

    lever_arm = float(half_length) + float(half_width)
    inv_radius = 1.0 / float(wheel_radius)

    front_left = (vx - vy - lever_arm * wz) * inv_radius
    front_right = (vx + vy + lever_arm * wz) * inv_radius
    rear_left = (vx + vy - lever_arm * wz) * inv_radius
    rear_right = (vx - vy + lever_arm * wz) * inv_radius
    return torch.stack([front_left, front_right, rear_left, rear_right], dim=-1)


def clip_wheel_velocity_targets(
    wheel_targets: Tensor,
    *,
    velocity_limit: float,
) -> tuple[Tensor, Tensor]:
    """Clip wheel targets and report the per-env saturation ratio."""
    limit = float(velocity_limit)
    clipped = torch.clamp(wheel_targets, min=-limit, max=limit)
    saturated = torch.abs(wheel_targets) > limit
    saturation_ratio = saturated.to(dtype=wheel_targets.dtype).mean(dim=-1)
    return clipped, saturation_ratio
