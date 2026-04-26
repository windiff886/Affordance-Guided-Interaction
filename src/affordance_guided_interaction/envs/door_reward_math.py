"""Door approach reward helpers that stay importable in plain pytest.

These utilities are pure torch tensor math and intentionally avoid Isaac Lab
runtime imports so the reward geometry can be regression-tested without Omni.
"""

from __future__ import annotations

import math

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


def _normalize_vectors(vectors: Tensor) -> Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1.0e-12)



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


def compute_base_alignment_gate(
    *,
    base_forward_world: Tensor,
    doorway_normal_world: Tensor,
    max_angle_deg: float,
) -> Tensor:
    """Gate base rewards by the angle to the push direction."""
    base_forward_unit = _normalize_vectors(base_forward_world)
    doorway_normal_unit = _normalize_vectors(doorway_normal_world)
    push_direction_unit = -doorway_normal_unit
    cos_threshold = math.cos(math.radians(float(max_angle_deg)))
    return (base_forward_unit * push_direction_unit).sum(dim=-1) >= cos_threshold


def compute_base_alignment_score(
    *,
    base_forward_world: Tensor,
    doorway_normal_world: Tensor,
    mid_angle_deg: float,
    temperature_deg: float,
) -> Tensor:
    """Smoothly score how well the base faces the push direction."""
    base_forward_unit = _normalize_vectors(base_forward_world)
    push_direction_unit = -_normalize_vectors(doorway_normal_world)
    cosine = torch.clamp((base_forward_unit * push_direction_unit).sum(dim=-1), -1.0, 1.0)
    angle = torch.acos(cosine)
    mid = math.radians(float(mid_angle_deg))
    temperature = max(math.radians(float(temperature_deg)), 1.0e-6)
    return torch.sigmoid((mid - angle) / temperature)


def compute_base_footprint_corners_door_frame(
    *,
    base_pos_door_xy: Tensor,
    base_yaw_door: Tensor,
    half_length: float,
    half_width: float,
) -> Tensor:
    """Compute the four base footprint corners in doorway-frame XY coordinates."""
    corners_local = torch.tensor(
        [
            [float(half_length), float(half_width)],
            [float(half_length), -float(half_width)],
            [-float(half_length), float(half_width)],
            [-float(half_length), -float(half_width)],
        ],
        device=base_pos_door_xy.device,
        dtype=base_pos_door_xy.dtype,
    ).unsqueeze(0)
    cos_yaw = torch.cos(base_yaw_door).unsqueeze(-1)
    sin_yaw = torch.sin(base_yaw_door).unsqueeze(-1)
    x_local = corners_local[..., 0]
    y_local = corners_local[..., 1]
    x_rot = x_local * cos_yaw - y_local * sin_yaw
    y_rot = x_local * sin_yaw + y_local * cos_yaw
    return torch.stack(
        (
            base_pos_door_xy[:, :1] + x_rot,
            base_pos_door_xy[:, 1:2] + y_rot,
        ),
        dim=-1,
    )


def compute_base_corridor_excess(
    *,
    corner_y: Tensor,
    corridor_half_width: float,
) -> Tensor:
    """Measure how far the base footprint extends outside the doorway corridor."""
    return torch.clamp(corner_y.abs().amax(dim=-1) - float(corridor_half_width), min=0.0)


def compute_base_range_score(
    *,
    corridor_excess: Tensor,
    tau: float,
) -> Tensor:
    """Keep full reward inside the doorway corridor and smoothly decay outside."""
    tau_value = max(float(tau), 1.0e-6)
    excess = torch.clamp(corridor_excess, min=0.0)
    return torch.exp(-0.5 * torch.square(excess / tau_value))


def compute_near_line_score(
    *,
    base_line_dist: Tensor,
    sigma: float,
) -> Tensor:
    """Score proximity to the doorway lower edge line segment."""
    sigma_value = max(float(sigma), 1.0e-6)
    return torch.exp(-0.5 * torch.square(base_line_dist / sigma_value))


def compute_forward_progress_delta(
    *,
    previous_signed_distance: Tensor,
    current_signed_distance: Tensor,
) -> Tensor:
    """Reward positive progress along the push direction in signed-distance space."""
    return torch.clamp(previous_signed_distance - current_signed_distance, min=0.0)


def compute_signed_progress_delta(
    *,
    previous_signed_distance: Tensor,
    current_signed_distance: Tensor,
) -> Tensor:
    """Measure signed progress along the push direction, preserving retreat."""
    return previous_signed_distance - current_signed_distance


def compute_target_rate_penalty(
    *,
    current_target: Tensor,
    previous_target: Tensor,
    beta: float,
    free_l2: float = 0.0,
) -> Tensor:
    """Penalize final arm-target jumps above an L2 free zone."""
    target_delta_l2 = (current_target - previous_target).norm(dim=-1)
    excess = torch.clamp(target_delta_l2 - float(free_l2), min=0.0)
    return float(beta) * torch.square(excess)


def compute_hand_near_gate(
    *,
    hand_dist: Tensor,
    near_dist: float,
    tau: float,
) -> Tensor:
    """Gate coordination rewards by absolute hand distance to the door face."""
    width = max(float(tau), 1.0e-6)
    return torch.sigmoid((float(near_dist) - hand_dist) / width)


def compute_base_net_progress_reward(
    *,
    align_score: Tensor,
    range_score: Tensor,
    signed_progress: Tensor,
    weight: float,
) -> Tensor:
    """Reward forward base motion and penalize retreat under alignment/range gates."""
    return float(weight) * align_score * range_score * signed_progress


def compute_soft_capped_velocity_penalty(
    *,
    velocity: Tensor,
    occupied: Tensor,
    free_speed: float,
    weight: float,
) -> Tensor:
    """Penalize only velocity norm above a free-speed cap for occupied cup sides."""
    excess = torch.clamp(velocity.norm(dim=-1) - float(free_speed), min=0.0)
    return -occupied.float() * float(weight) * torch.square(excess)


def compute_door_angle_push_gate(
    *,
    door_angle: Tensor,
    start_angle: float,
    end_angle: float,
    start_tau: float,
    end_tau: float,
) -> Tensor:
    """Gate base-assist reward to the mid opening range where push coordination matters."""
    start_width = max(float(start_tau), 1.0e-6)
    end_width = max(float(end_tau), 1.0e-6)
    start_gate = torch.sigmoid((door_angle - float(start_angle)) / start_width)
    end_gate = torch.sigmoid((float(end_angle) - door_angle) / end_width)
    return start_gate * end_gate


def compute_door_angle_cross_gate(
    *,
    door_angle: Tensor,
    cross_angle: float,
    tau: float,
) -> Tensor:
    """Smoothly activate base-cross progress before the hard task success angle."""
    width = max(float(tau), 1.0e-6)
    return torch.sigmoid((door_angle - float(cross_angle)) / width)


def compute_base_assist_reward(
    *,
    align_score: Tensor,
    range_score: Tensor,
    hand_score: Tensor,
    push_gate: Tensor,
    base_push_progress: Tensor,
    weight: float,
) -> Tensor:
    """Reward base progress only when hand contact, alignment and doorway range agree."""
    return (
        float(weight)
        * align_score
        * range_score
        * hand_score
        * push_gate
        * base_push_progress
    )


def compute_base_door_sync_reward(
    *,
    align_score: Tensor,
    range_score: Tensor,
    door_delta: Tensor,
    base_push_progress: Tensor,
    weight: float,
    eps: float = 0.0,
) -> Tensor:
    """Reward simultaneous positive door opening and base push progress."""
    coupled_progress = torch.clamp(door_delta, min=0.0) * torch.clamp(
        base_push_progress,
        min=0.0,
    )
    reward = float(weight) * align_score * range_score * torch.sqrt(coupled_progress + float(eps))
    return torch.where(coupled_progress > 0.0, reward, torch.zeros_like(reward))


def compute_base_align_reward(
    *,
    align_score: Tensor,
    range_score: Tensor,
    near_score: Tensor,
    weight: float,
) -> Tensor:
    """Combine smooth factors into the base alignment shaping reward."""
    return float(weight) * align_score * range_score * near_score


def compute_base_heading_penalty(
    *,
    base_forward_world: Tensor,
    doorway_tangent_world: Tensor,
) -> Tensor:
    """Penalize heading projected onto the doorway tangent direction."""
    base_forward_unit = _normalize_vectors(base_forward_world)
    doorway_tangent_unit = _normalize_vectors(doorway_tangent_world)
    return torch.square((base_forward_unit * doorway_tangent_unit).sum(dim=-1))


def compute_base_speed_squared(
    *,
    base_lin_vel_base: Tensor,
    base_ang_vel_base: Tensor,
) -> Tensor:
    """Compute the raw mobile-base speed energy."""
    return (
        torch.square(base_lin_vel_base[:, 0])
        + torch.square(base_lin_vel_base[:, 1])
        + torch.square(base_ang_vel_base[:, 2])
    )


def compute_base_zero_speed_reward(
    *,
    speed_sq: Tensor,
    weight: float,
    decay: float,
) -> Tensor:
    """Compute the documented near-zero-speed reward."""
    return float(weight) * torch.exp(-float(decay) * speed_sq)


def compute_base_speed_penalty(
    *,
    speed_sq: Tensor,
    weight: float,
) -> Tensor:
    """Compute the documented raw speed penalty."""
    return float(weight) * speed_sq


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
