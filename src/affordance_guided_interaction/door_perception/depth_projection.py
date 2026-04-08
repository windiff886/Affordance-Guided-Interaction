"""深度反投影：mask + depth map -> 3D 点云，以及基础的点云降采样工具。"""

from __future__ import annotations

import numpy as np
import torch

from affordance_guided_interaction.door_perception.config import CameraIntrinsics


def backproject_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
) -> np.ndarray:
    """将深度图（或其 mask 子集）反投影为 3D 点云。

    Parameters
    ----------
    depth : np.ndarray
        (H, W) 深度图，单位：米。
    intrinsics : CameraIntrinsics
        针孔相机内参。
    mask : np.ndarray | None
        (H, W) 布尔掩模。仅 ``mask == True`` 的像素会被投影。
        为 None 时使用全图。
    extrinsic : np.ndarray | None
        (4, 4) 相机到世界坐标系的变换矩阵。提供时返回世界坐标系点云。

    Returns
    -------
    np.ndarray
        (N, 3) 点云。
    """
    h, w = depth.shape[:2]

    us = np.arange(w, dtype=np.float64)
    vs = np.arange(h, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(us, vs)

    if mask is not None:
        valid = mask & (depth > 0)
    else:
        valid = depth > 0

    u_vals = u_grid[valid]
    v_vals = v_grid[valid]
    d_vals = depth[valid].astype(np.float64)

    x = (u_vals - intrinsics.cx) * d_vals / intrinsics.fx
    y = (v_vals - intrinsics.cy) * d_vals / intrinsics.fy
    z = d_vals

    points = np.stack([x, y, z], axis=-1)

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
    """便捷包装：对多个命名 mask 批量反投影。

    Returns
    -------
    dict[str, np.ndarray]
        标签 → (N_i, 3) 点云的映射。
    """
    return {
        label: backproject_depth(depth, intrinsics, mask=m, extrinsic=extrinsic)
        for label, m in masks.items()
    }


def backproject_depth_batch(
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
    mask: torch.Tensor | None = None,
    extrinsic: torch.Tensor | None = None,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将一批深度图反投影为定长 batched 点云。"""
    batch, height, width = depth.shape
    device = depth.device

    u = torch.arange(width, device=device, dtype=torch.float32)
    v = torch.arange(height, device=device, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    u_grid = u_grid.unsqueeze(0).expand(batch, -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(batch, -1, -1)

    valid = depth > 0
    if mask is not None:
        valid = valid & mask.to(device=device, dtype=torch.bool)

    x = (u_grid - float(intrinsics.cx)) * depth / float(intrinsics.fx)
    y = (v_grid - float(intrinsics.cy)) * depth / float(intrinsics.fy)
    points = torch.stack([x, y, depth], dim=-1).reshape(batch, -1, 3)
    valid_flat = valid.reshape(batch, -1)

    if extrinsic is not None:
        rot = extrinsic[:, :3, :3].to(device=device, dtype=points.dtype)
        trans = extrinsic[:, :3, 3].to(device=device, dtype=points.dtype)
        points = torch.bmm(points, rot.transpose(1, 2)) + trans.unsqueeze(1)

    return sample_or_pad_batch(points, valid_flat, max_points)


# ---------------------------------------------------------------------------
# 点云降采样工具（管线使用，替代旧 point_cloud_processing.py 中的复杂滤波）
# ---------------------------------------------------------------------------


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素网格降采样（纯 numpy 实现）。

    将点云按 voxel_size 划分体素格，每个非空体素内只保留一个点。
    主要目的是控制送入 Point-MAE 的点数上限。

    Parameters
    ----------
    points : np.ndarray
        (N, 3) 点云。
    voxel_size : float
        体素边长（米）。

    Returns
    -------
    np.ndarray
        (M, 3) 降采样后的点云，M <= N。
    """
    if len(points) == 0 or voxel_size <= 0:
        return points

    indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(indices, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def sample_or_pad(points: np.ndarray, n: int) -> np.ndarray:
    """将点云对齐到恰好 n 个点。

    当点数 >= n 时随机采样（不重复），不足时重复采样填充。
    空点云返回全零数组。

    Parameters
    ----------
    points : np.ndarray
        (M, 3) 输入点云。
    n : int
        目标点数。

    Returns
    -------
    np.ndarray
        (n, 3) 对齐后的点云。
    """
    if len(points) == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if len(points) >= n:
        idx = np.random.choice(len(points), size=n, replace=False)
    else:
        idx = np.random.choice(len(points), size=n, replace=True)
    return points[idx]


def sample_or_pad_batch(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 batched 点云对齐到固定点数，并返回有效点标记。"""
    batch = points.shape[0]
    aligned = torch.zeros(batch, n, 3, dtype=points.dtype, device=points.device)
    valid_out = torch.zeros(batch, n, dtype=torch.bool, device=points.device)

    for idx in range(batch):
        current = points[idx][valid_mask[idx]]
        count = int(current.shape[0])
        if count == 0:
            continue
        take = min(count, n)
        aligned[idx, :take] = current[:take]
        valid_out[idx, :take] = True
        if count < n:
            repeat_count = n - count
            repeat_idx = torch.arange(repeat_count, device=points.device) % count
            aligned[idx, count:n] = current[repeat_idx]

    return aligned, valid_out
