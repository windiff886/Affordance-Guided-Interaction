from __future__ import annotations

import torch
from torch import Tensor


def build_grasp_init_joint_positions(
    *,
    joint_seed: Tensor,
    left_grasp_joint_ids: list[int],
    left_grasp_joint_targets: Tensor,
    right_grasp_joint_ids: list[int],
    right_grasp_joint_targets: Tensor,
    gripper_joint_ids: list[int],
    gripper_joint_targets: Tensor,
) -> Tensor:
    """Build grasp-init joint positions while preserving any prewritten base joints.

    The returned tensor starts from ``joint_seed`` so callers can pass the reset-time
    joint state that already contains the desired planar-base pose. Only the arm grasp
    joints and gripper joints are overwritten.
    """
    joint_pos = joint_seed.clone()
    joint_pos[:, left_grasp_joint_ids] = left_grasp_joint_targets
    joint_pos[:, right_grasp_joint_ids] = right_grasp_joint_targets
    joint_pos[:, gripper_joint_ids] = gripper_joint_targets
    return joint_pos


def rescale_normalized_joint_actions(actions: Tensor, q_min: Tensor, q_max: Tensor) -> Tensor:
    """Map normalized actions in ``[-1, 1]`` to joint-position targets."""
    bounded_actions = torch.clamp(actions, -1.0, 1.0)
    half_range = 0.5 * (q_max - q_min)
    center = 0.5 * (q_max + q_min)
    return center.unsqueeze(0) + bounded_actions * half_range.unsqueeze(0)


def compute_joint_limit_margin_penalty(
    q_target: Tensor,
    q_min: Tensor,
    q_max: Tensor,
    margin_ratio: float,
    beta: float,
) -> Tensor:
    """Penalize targets only inside a configurable edge band near the joint limits."""
    joint_range = torch.clamp(q_max - q_min, min=torch.finfo(q_target.dtype).eps)
    margin = torch.clamp(joint_range * margin_ratio, min=torch.finfo(q_target.dtype).eps)
    distance_to_nearest_limit = torch.minimum(q_target - q_min.unsqueeze(0), q_max.unsqueeze(0) - q_target)
    edge_intrusion = torch.clamp(margin.unsqueeze(0) - distance_to_nearest_limit, min=0.0)
    normalized_intrusion = edge_intrusion / margin.unsqueeze(0)
    return beta * torch.square(normalized_intrusion).sum(dim=-1)
