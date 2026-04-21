from __future__ import annotations

import torch
from torch import Tensor


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
