"""Depth back-projection: mask + depth map -> 3-D point cloud."""

from __future__ import annotations

import numpy as np

from affordance_guided_interaction.door_perception.config import CameraIntrinsics


def backproject_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a depth image (or a masked subset) to a 3-D point cloud.

    Parameters
    ----------
    depth : np.ndarray
        (H, W) depth map in metres.
    intrinsics : CameraIntrinsics
        Pin-hole camera intrinsic parameters.
    mask : np.ndarray | None
        (H, W) boolean mask.  Only pixels where ``mask == True`` are
        projected.  If *None* the entire image is used.
    extrinsic : np.ndarray | None
        (4, 4) camera-to-world transform.  When given the returned points
        are expressed in the world frame; otherwise they stay in the camera
        frame.

    Returns
    -------
    np.ndarray
        (N, 3) point cloud.
    """
    h, w = depth.shape[:2]

    # Build pixel coordinate grids
    us = np.arange(w, dtype=np.float64)
    vs = np.arange(h, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(us, vs)  # both (H, W)

    if mask is not None:
        valid = mask & (depth > 0)
    else:
        valid = depth > 0

    u_vals = u_grid[valid]
    v_vals = v_grid[valid]
    d_vals = depth[valid].astype(np.float64)

    # Pin-hole back-projection
    x = (u_vals - intrinsics.cx) * d_vals / intrinsics.fx
    y = (v_vals - intrinsics.cy) * d_vals / intrinsics.fy
    z = d_vals

    points = np.stack([x, y, z], axis=-1)  # (N, 3)

    if extrinsic is not None:
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        points = (R @ points.T).T + t

    return points


def backproject_masks(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    masks: dict[str, np.ndarray],
    extrinsic: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: back-project multiple named masks at once.

    Parameters
    ----------
    depth : np.ndarray
        (H, W) depth map in metres.
    intrinsics : CameraIntrinsics
        Pin-hole camera intrinsic parameters.
    masks : dict[str, np.ndarray]
        Mapping from label (e.g. ``"door"``) to (H, W) boolean mask.
    extrinsic : np.ndarray | None
        Optional camera-to-world transform.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from label to (N_i, 3) point cloud.
    """
    return {
        label: backproject_depth(depth, intrinsics, mask=m, extrinsic=extrinsic)
        for label, m in masks.items()
    }
