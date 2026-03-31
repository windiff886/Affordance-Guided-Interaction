"""Compute the low-dimensional geometric affordance vector z_aff.

z_aff layout (default ~25-D):
    c_door          (3)   door panel centroid
    n_door          (3)   door plane unit normal
    b_door          (3)   door bounding-box extents (dx, dy, dz)
    c_handle        (3)   handle centroid
    c_button        (3)   button centroid
    d_g_door        (1)   gripper-to-door-plane signed distance
    d_g_handle      (1)   gripper-to-handle distance
    d_g_button      (1)   gripper-to-button distance
    affordance_type (4)   one-hot [push, press, handle, sequential]
    confidence      (3)   per-part segmentation confidence
    ---
    total = 25
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from affordance_guided_interaction.door_perception.point_cloud_processing import (
    PlaneFitResult,
    fit_plane_ransac,
)

# Affordance type indices
AFFORDANCE_PUSH = 0
AFFORDANCE_PRESS = 1
AFFORDANCE_HANDLE = 2
AFFORDANCE_SEQUENTIAL = 3
_NUM_AFFORDANCE_TYPES = 4

Z_AFF_DIM = 25


@dataclass(slots=True)
class PartGeometry:
    """Geometric summary for a single detected part."""

    centroid: np.ndarray   # (3,)
    bbox_extents: np.ndarray  # (3,) axis-aligned extents
    normal: np.ndarray | None = None  # (3,) only for planar parts
    confidence: float = 0.0
    present: bool = False


def _compute_part_geometry(
    points: np.ndarray,
    confidence: float,
    fit_plane: bool = False,
) -> PartGeometry:
    """Summarise a single part point cloud."""
    if len(points) == 0:
        return PartGeometry(
            centroid=np.zeros(3),
            bbox_extents=np.zeros(3),
            normal=None,
            confidence=0.0,
            present=False,
        )

    centroid = points.mean(axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extents = bbox_max - bbox_min

    normal = None
    if fit_plane:
        plane = fit_plane_ransac(points)
        if plane is not None:
            normal = plane.normal

    return PartGeometry(
        centroid=centroid,
        bbox_extents=extents,
        normal=normal,
        confidence=confidence,
        present=True,
    )


def _point_to_plane_distance(
    point: np.ndarray,
    plane_center: np.ndarray,
    plane_normal: np.ndarray,
) -> float:
    """Signed distance from *point* to an infinite plane."""
    return float(np.dot(point - plane_center, plane_normal))


def compute_z_aff(
    door_points: np.ndarray,
    handle_points: np.ndarray,
    button_points: np.ndarray,
    gripper_pos: np.ndarray,
    affordance_type: int = AFFORDANCE_PUSH,
    door_confidence: float = 0.0,
    handle_confidence: float = 0.0,
    button_confidence: float = 0.0,
) -> np.ndarray:
    """Build the z_aff vector from per-part point clouds.

    Parameters
    ----------
    door_points, handle_points, button_points : np.ndarray (N, 3)
        Cleaned point clouds for each part (may be empty).
    gripper_pos : np.ndarray (3,)
        Current gripper position in the same coordinate frame.
    affordance_type : int
        Index into [push, press, handle, sequential].
    door_confidence, handle_confidence, button_confidence : float
        Segmentation confidence for each part.

    Returns
    -------
    np.ndarray (Z_AFF_DIM,)
    """
    door_geom = _compute_part_geometry(door_points, door_confidence, fit_plane=True)
    handle_geom = _compute_part_geometry(handle_points, handle_confidence)
    button_geom = _compute_part_geometry(button_points, button_confidence)

    # --- per-part features ---
    c_door = door_geom.centroid
    n_door = door_geom.normal if door_geom.normal is not None else np.zeros(3)
    b_door = door_geom.bbox_extents

    c_handle = handle_geom.centroid
    c_button = button_geom.centroid

    # --- gripper distances ---
    if door_geom.present and door_geom.normal is not None:
        d_g_door = _point_to_plane_distance(gripper_pos, c_door, n_door)
    else:
        d_g_door = 0.0

    d_g_handle = (
        float(np.linalg.norm(gripper_pos - c_handle)) if handle_geom.present else 0.0
    )
    d_g_button = (
        float(np.linalg.norm(gripper_pos - c_button)) if button_geom.present else 0.0
    )

    # --- affordance type one-hot ---
    aff_onehot = np.zeros(_NUM_AFFORDANCE_TYPES, dtype=np.float64)
    if 0 <= affordance_type < _NUM_AFFORDANCE_TYPES:
        aff_onehot[affordance_type] = 1.0

    # --- confidence ---
    conf = np.array(
        [door_geom.confidence, handle_geom.confidence, button_geom.confidence],
        dtype=np.float64,
    )

    z_aff = np.concatenate(
        [
            c_door,          # 3
            n_door,          # 3
            b_door,          # 3
            c_handle,        # 3
            c_button,        # 3
            [d_g_door],      # 1
            [d_g_handle],    # 1
            [d_g_button],    # 1
            aff_onehot,      # 4
            conf,            # 3
        ]
    )
    assert z_aff.shape == (Z_AFF_DIM,), f"Expected {Z_AFF_DIM}, got {z_aff.shape}"
    return z_aff
