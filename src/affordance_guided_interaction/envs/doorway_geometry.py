"""Pure doorway geometry helpers shared by observations, rewards, and tests."""

from __future__ import annotations

import torch
from torch import Tensor

from .batch_math import batch_quat_to_rotation_matrix

DOORWAY_INNER_CORNERS_LOCAL = torch.tensor(
    [
        [0.0, -0.51, 0.0],
        [0.0, 0.51, 0.0],
        [0.0, -0.51, 2.05],
        [0.0, 0.51, 2.05],
    ],
    dtype=torch.float32,
)
DOORWAY_LOWER_EDGE_START_LOCAL = DOORWAY_INNER_CORNERS_LOCAL[0].clone()
DOORWAY_LOWER_EDGE_END_LOCAL = DOORWAY_INNER_CORNERS_LOCAL[1].clone()


def _expand_points(points: Tensor, batch_size: int) -> Tensor:
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.shape[0] == 1 and batch_size != 1:
        points = points.expand(batch_size, -1, -1)
    return points


def transform_doorway_points_to_world(
    points_local: Tensor,
    door_pos_w: Tensor,
    door_quat_w: Tensor,
) -> Tensor:
    """Transform batched doorway points from door-local frame into world frame."""
    batch_size = door_pos_w.shape[0]
    points_local = _expand_points(points_local, batch_size).to(
        device=door_pos_w.device,
        dtype=door_pos_w.dtype,
    )
    rot = batch_quat_to_rotation_matrix(door_quat_w)
    rotated = torch.bmm(rot, points_local.transpose(1, 2)).transpose(1, 2)
    return rotated + door_pos_w.unsqueeze(1)


def transform_doorway_points_to_base(
    *,
    points_world: Tensor,
    base_pos_w: Tensor,
    base_quat_w: Tensor,
) -> Tensor:
    """Transform batched doorway world points into the current base frame."""
    batch_size = base_pos_w.shape[0]
    points_world = _expand_points(points_world, batch_size).to(
        device=base_pos_w.device,
        dtype=base_pos_w.dtype,
    )
    rel_world = points_world - base_pos_w.unsqueeze(1)
    rot = batch_quat_to_rotation_matrix(base_quat_w)
    rel_base = torch.bmm(rot.transpose(-1, -2), rel_world.transpose(1, 2)).transpose(1, 2)
    return rel_base
