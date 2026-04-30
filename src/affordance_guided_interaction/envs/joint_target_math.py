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


def compute_torque_proxy_joint_targets(
    actions: Tensor,
    *,
    default_joint_pos: Tensor,
    current_joint_pos: Tensor,
    torque_limits: Tensor,
    stiffness: Tensor,
    action_scale: float,
    sigma: float,
) -> Tensor:
    """Compute joint-position targets using the torque-proxy clamping model.

    A raw target is first formed from the default pose plus a scaled action offset.
    The target is then clamped to a band around the *current* joint position whose
    half-width is proportional to ``torque_limits / stiffness`` (scaled by ``sigma``).
    This prevents the policy from requesting position targets that would require
    torques beyond the actuator limits.

    Parameters
    ----------
    actions:
        Normalized actions from the policy (shape ``[B, num_joints]``).
    default_joint_pos:
        Default (rest) joint positions (shape ``[num_joints]``).
    current_joint_pos:
        Current joint positions (shape ``[B, num_joints]``).
    torque_limits:
        Per-joint maximum torque (shape ``[num_joints]``).
    stiffness:
        Per-joint PD stiffness (shape ``[num_joints]``).
    action_scale:
        Multiplicative scale applied to actions before adding to defaults.
    sigma:
        Safety factor controlling how tightly the target is clamped around the
        current position (higher = looser clamp).
    """
    raw_target = default_joint_pos.unsqueeze(0) + actions * action_scale
    delta_limit = sigma * torque_limits / stiffness
    return torch.clamp(
        raw_target,
        current_joint_pos - delta_limit.unsqueeze(0),
        current_joint_pos + delta_limit.unsqueeze(0),
    )


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
