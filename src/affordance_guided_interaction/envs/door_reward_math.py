"""Door approach reward helpers that stay importable in plain pytest.

These utilities are pure torch tensor math and intentionally avoid Isaac Lab
runtime imports so the reward geometry can be regression-tested without Omni.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_point_to_panel_face_distance(
    *,
    points_base: Tensor,
    face_center_base: Tensor,
    face_rot_base: Tensor,
    half_extent_y: float,
    half_extent_z: float,
) -> Tensor:
    """Compute batched point-to-rectangle-face distances in base frame.

    The face is centered at ``face_center_base`` with local +X as the outward
    normal. The rectangle spans local ``[-half_extent_y, half_extent_y]`` on Y
    and ``[-half_extent_z, half_extent_z]`` on Z.
    """
    rel_base = points_base - face_center_base
    rel_local = torch.bmm(
        face_rot_base.transpose(-1, -2),
        rel_base.unsqueeze(-1),
    ).squeeze(-1)

    excess_y = torch.clamp(rel_local[:, 1].abs() - float(half_extent_y), min=0.0)
    excess_z = torch.clamp(rel_local[:, 2].abs() - float(half_extent_z), min=0.0)
    return torch.sqrt(rel_local[:, 0].square() + excess_y.square() + excess_z.square())



def compute_normalized_approach_score(
    *,
    current_dist: Tensor,
    initial_dist: Tensor,
    eps: float,
) -> Tensor:
    """Compute ``max(1 - a^2 / (b^2 + eps), 0)`` in batch form."""
    return torch.clamp(
        1.0 - current_dist.square() / (initial_dist.square() + float(eps)),
        min=0.0,
    )


def compute_point_to_segment_distance(
    *,
    points: Tensor,
    seg_start: Tensor,
    seg_end: Tensor,
) -> Tensor:
    """Compute batched point-to-line-segment distances in 3D."""
    seg_vec = seg_end - seg_start
    rel = points - seg_start
    seg_len_sq = seg_vec.square().sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
    t = torch.clamp((rel * seg_vec).sum(dim=-1, keepdim=True) / seg_len_sq, 0.0, 1.0)
    closest = seg_start + t * seg_vec
    return (points - closest).norm(dim=-1)


def compute_signed_distance_to_plane(
    *,
    points: Tensor,
    plane_points: Tensor,
    plane_normals: Tensor,
) -> Tensor:
    """Project point offsets onto the plane normal."""
    return ((points - plane_points) * plane_normals).sum(dim=-1)


def compute_base_approach_active_mask(
    *,
    base_crossed: Tensor,
    current_signed_distance: Tensor,
) -> Tensor:
    """Keep approach shaping active only while base_link is still outside the doorway plane."""
    return (~base_crossed) & (current_signed_distance > 0.0)


def compute_inside_progress_delta(
    *,
    previous_signed_distance: Tensor,
    current_signed_distance: Tensor,
    in_opening: Tensor,
) -> Tensor:
    """Reward only positive inside progress made while the base_link is in the doorway opening."""
    prev_inside_progress = torch.clamp(-previous_signed_distance, min=0.0)
    curr_inside_progress = torch.clamp(-current_signed_distance, min=0.0)
    return in_opening.float() * torch.clamp(curr_inside_progress - prev_inside_progress, min=0.0)


def update_crossing_latch(
    *,
    previous_crossed: Tensor,
    previous_signed_distance: Tensor,
    current_signed_distance: Tensor,
    in_opening: Tensor,
) -> Tensor:
    """Latch a valid outside-to-inside crossing event."""
    crossed_now = (
        (previous_signed_distance > 0.0)
        & (current_signed_distance <= 0.0)
        & in_opening
    )
    return previous_crossed | crossed_now


def compute_door_traverse_success(
    *,
    door_angle: Tensor,
    door_angle_target: float,
    cup_dropped: Tensor,
    base_crossed: Tensor,
) -> Tensor:
    """Combine the three task success conditions."""
    return (door_angle >= float(door_angle_target)) & (~cup_dropped) & base_crossed
