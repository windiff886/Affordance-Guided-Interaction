"""Point cloud post-processing: downsampling, outlier removal, plane fitting.

Uses Open3D when available; falls back to pure-numpy implementations so that
the module stays importable and testable without Open3D installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from affordance_guided_interaction.door_perception.config import PointCloudConfig

logger = logging.getLogger(__name__)

try:
    import open3d as o3d  # type: ignore[import-untyped]

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


@dataclass(slots=True)
class PlaneFitResult:
    """Result of RANSAC plane fitting."""

    normal: np.ndarray   # (3,) unit normal
    center: np.ndarray   # (3,) centroid of inliers
    d: float             # plane equation: n·x + d = 0
    inlier_indices: np.ndarray  # integer indices into the input cloud


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid downsampling.

    Uses Open3D when available, otherwise a numpy approximation.
    """
    if _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points)

    # Numpy fallback: hash points into voxel bins, keep one per bin
    indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(indices, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def statistical_outlier_removal(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers."""
    if _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(pcd.points)

    # Numpy fallback: remove points whose mean kNN distance > mean + std_ratio * std
    from scipy.spatial import KDTree  # type: ignore[import-untyped]

    if len(points) <= nb_neighbors:
        return points
    tree = KDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return points[mean_dists < threshold]


def radius_outlier_removal(
    points: np.ndarray,
    radius: float = 0.02,
    min_neighbors: int = 5,
) -> np.ndarray:
    """Remove radius outliers."""
    if _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        return np.asarray(pcd.points)

    from scipy.spatial import KDTree  # type: ignore[import-untyped]

    if len(points) == 0:
        return points
    tree = KDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True)
    return points[np.asarray(counts) >= min_neighbors]


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> PlaneFitResult | None:
    """Fit a plane to *points* using RANSAC (Open3D).

    Returns *None* when the cloud has fewer than *ransac_n* points.
    """
    if len(points) < ransac_n:
        return None

    if _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane_model, inlier_idx = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return None
        normal /= norm
        inlier_idx = np.asarray(inlier_idx)
        center = points[inlier_idx].mean(axis=0)
        return PlaneFitResult(
            normal=normal, center=center, d=float(d), inlier_indices=inlier_idx
        )

    # Numpy RANSAC fallback
    rng = np.random.default_rng()
    best_inliers: np.ndarray | None = None
    best_normal = np.zeros(3)
    best_d = 0.0
    for _ in range(num_iterations):
        idx = rng.choice(len(points), size=ransac_n, replace=False)
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n /= n_norm
        d_val = -np.dot(n, p0)
        dists = np.abs(points @ n + d_val)
        inliers = np.where(dists < distance_threshold)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = n
            best_d = d_val

    if best_inliers is None or len(best_inliers) == 0:
        return None
    center = points[best_inliers].mean(axis=0)
    return PlaneFitResult(
        normal=best_normal, center=center, d=float(best_d), inlier_indices=best_inliers
    )


def clean_point_cloud(points: np.ndarray, config: PointCloudConfig) -> np.ndarray:
    """Run the full cleaning pipeline: voxel downsample -> statistical -> radius filter.

    Returns an empty (0, 3) array when the input is empty.
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts = voxel_downsample(points, config.voxel_size)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts = statistical_outlier_removal(
        pts,
        nb_neighbors=config.statistical_nb_neighbors,
        std_ratio=config.statistical_std_ratio,
    )
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts = radius_outlier_removal(
        pts,
        radius=config.radius_filter_radius,
        min_neighbors=config.radius_filter_min_neighbors,
    )
    return pts if len(pts) > 0 else np.zeros((0, 3), dtype=np.float64)


def random_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Randomly sample (or pad with zeros) to exactly *n* points."""
    if len(points) == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if len(points) >= n:
        idx = np.random.choice(len(points), size=n, replace=False)
    else:
        idx = np.random.choice(len(points), size=n, replace=True)
    return points[idx]
