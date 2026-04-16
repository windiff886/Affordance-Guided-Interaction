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
