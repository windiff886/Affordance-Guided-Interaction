"""Paper-aligned reward helpers for the handle-free push-door task.

Pure torch tensor math — importable in plain pytest without Isaac Lab runtime.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


def compute_opening_reward(theta: Tensor, theta_hat: float) -> Tensor:
    """Opening reward: r_od = 1 - |theta - theta_hat| / theta_hat."""
    return 1.0 - (theta - float(theta_hat)).abs() / float(theta_hat)


def compute_stage(theta: Tensor, theta_pass: float) -> Tensor:
    """Stage indicator: 1 if theta > theta_pass, else 0."""
    return (theta > float(theta_pass)).float()


def compute_passing_reward(
    progress_dir_base: Tensor,
    base_lin_vel_base: Tensor,
    max_speed: float = 0.5,
) -> Tensor:
    """Passing reward: min(1, dot(progress_dir, base_vel) / max_speed).

    Uses base velocity projection, not signed distance delta.
    """
    dot = (progress_dir_base * base_lin_vel_base).sum(dim=-1)
    return torch.clamp(dot / float(max_speed), min=0.0, max=1.0)


def compute_min_arm_motion_reward(arm_qd: Tensor, arm_qdd: Tensor) -> Tensor:
    """Minimize arm motion reward over 12 joints.

    r_ma = sum_i [exp(-0.01 * qd_i^2) + exp(-1e-6 * qdd_i^2)]
    """
    vel_term = torch.exp(-0.01 * arm_qd.square()).sum(dim=-1)
    acc_term = torch.exp(-1e-6 * arm_qdd.square()).sum(dim=-1)
    return vel_term + acc_term


def compute_stretched_arm_penalty(
    ee_pos_base: Tensor,
    shoulder_pos_base: Tensor,
    threshold: float = 0.5,
    scale: float = 0.1,
) -> Tensor:
    """Penalize stretched arms: -sum_j clip((||ee - shoulder|| - threshold) / scale, 0, 1).

    ee_pos_base: (N, 6) concatenated left+right EE positions in base frame.
    shoulder_pos_base: (N, 6) concatenated left+right shoulder positions in base frame.
    """
    left_ee = ee_pos_base[:, :3]
    right_ee = ee_pos_base[:, 3:]
    left_shoulder = shoulder_pos_base[:, :3]
    right_shoulder = shoulder_pos_base[:, 3:]

    left_dist = (left_ee - left_shoulder).norm(dim=-1)
    right_dist = (right_ee - right_shoulder).norm(dim=-1)

    left_penalty = torch.clamp((left_dist - float(threshold)) / float(scale), 0.0, 1.0)
    right_penalty = torch.clamp((right_dist - float(threshold)) / float(scale), 0.0, 1.0)
    return -(left_penalty + right_penalty)


def compute_end_effector_to_door_proximity_reward(
    left_ee_pos_base: Tensor,
    right_ee_pos_base: Tensor,
    door_pos_base: Tensor,
) -> Tensor:
    """Reward the closer end effector being near the door panel.

    r_eep = exp(-min(||ee_L - door||, ||ee_R - door||)).
    """
    left_dist = (left_ee_pos_base - door_pos_base).norm(dim=-1)
    right_dist = (right_ee_pos_base - door_pos_base).norm(dim=-1)
    return torch.exp(-torch.minimum(left_dist, right_dist))


def compute_command_limit_penalty(
    actions: Tensor,
    action_limits: float = 1.0,
    scale: float = 1.0,
) -> Tensor:
    """Penalize raw actions exceeding limits: -sum_i clip((|a_i| - limit) / scale, 0, 1)."""
    excess = torch.clamp((actions.abs() - float(action_limits)) / float(scale), 0.0, 1.0)
    return -excess.sum(dim=-1)


def compute_collision_penalty(hard_collision_mask: Tensor) -> Tensor:
    """Collision penalty: -1 for each hard collision event."""
    return -hard_collision_mask.float()
