"""GPU 批量四元数与坐标变换工具 — 纯 torch tensor 操作。

替代 ``door_env.py`` 中逐环境调用的 numpy 版本，支持 ``(N, ...)``
批量输入，在 GPU 上并行计算所有环境的坐标变换。

四元数约定：``(w, x, y, z)``，与 Isaac Lab 一致。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════
# 四元数基础运算
# ═══════════════════════════════════════════════════════════════════════


def batch_quat_conjugate(q: Tensor) -> Tensor:
    """四元数共轭。

    Parameters
    ----------
    q : (N, 4)  wxyz 格式

    Returns
    -------
    (N, 4) wxyz 格式
    """
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def batch_quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton 四元数乘法  a * b。

    Parameters
    ----------
    a, b : (..., 4)  wxyz 格式

    Returns
    -------
    (..., 4) wxyz 格式
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def batch_quat_normalize(q: Tensor) -> Tensor:
    """单位化四元数。

    Parameters
    ----------
    q : (..., 4)

    Returns
    -------
    (..., 4) 范数为 1 的四元数
    """
    norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return q / norm


def batch_quat_to_rotation_matrix(q: Tensor) -> Tensor:
    """四元数 → 3x3 旋转矩阵。

    Parameters
    ----------
    q : (..., 4)  wxyz 格式

    Returns
    -------
    (..., 3, 3) 旋转矩阵
    """
    q = batch_quat_normalize(q)
    w, x, y, z = q.unbind(-1)

    # 预计算
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


# ═══════════════════════════════════════════════════════════════════════
# yaw ↔ 四元数
# ═══════════════════════════════════════════════════════════════════════


def batch_quat_from_yaw(yaw: Tensor) -> Tensor:
    """绕 Z 轴的 yaw 角 → 四元数 (wxyz)。

    Parameters
    ----------
    yaw : (N,) 弧度

    Returns
    -------
    (N, 4) wxyz 格式
    """
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack([half.cos(), zeros, zeros, half.sin()], dim=-1)


def batch_yaw_from_quat(q: Tensor) -> Tensor:
    """从四元数 (wxyz) 提取 yaw 角。

    Parameters
    ----------
    q : (N, 4)  wxyz 格式

    Returns
    -------
    (N,)  yaw 弧度
    """
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


# ═══════════════════════════════════════════════════════════════════════
# 坐标系变换（世界系 → base_link 系）
# ═══════════════════════════════════════════════════════════════════════


def batch_vector_world_to_base(
    vector_world: Tensor,
    base_quat_world: Tensor,
) -> Tensor:
    """将世界系 3D 向量旋转到 base_link 坐标系。

    Parameters
    ----------
    vector_world : (N, 3)
    base_quat_world : (N, 4)  wxyz 格式

    Returns
    -------
    (N, 3) base_link 坐标系下的向量
    """
    R_world_from_base = batch_quat_to_rotation_matrix(base_quat_world)  # (N, 3, 3)
    # R_base_from_world = R_world_from_base.transpose(-1, -2)
    # result = R_base_from_world @ vector_world.unsqueeze(-1)
    return torch.bmm(
        R_world_from_base.transpose(-1, -2),
        vector_world.unsqueeze(-1),
    ).squeeze(-1)


def batch_orientation_world_to_base(
    quat_world: Tensor,
    base_quat_world: Tensor,
) -> Tensor:
    """将世界系四元数转为 base_link 相对四元数。

    Parameters
    ----------
    quat_world : (N, 4)  wxyz
    base_quat_world : (N, 4)  wxyz

    Returns
    -------
    (N, 4) base_link 系下的相对四元数  wxyz
    """
    return batch_quat_normalize(
        batch_quat_multiply(
            batch_quat_conjugate(base_quat_world),
            quat_world,
        )
    )


def batch_pose_world_to_base(
    pos_world: Tensor,
    quat_world: Tensor,
    base_pos_world: Tensor,
    base_quat_world: Tensor,
) -> Tensor:
    """将世界系 pose 转为 base_link 系，返回 pos(3)+quat(4)=7 的拼接。

    Parameters
    ----------
    pos_world : (N, 3)
    quat_world : (N, 4) wxyz
    base_pos_world : (N, 3)
    base_quat_world : (N, 4) wxyz

    Returns
    -------
    (N, 7)  [position_base(3), orientation_base(4)]
    """
    pos_base = batch_vector_world_to_base(
        pos_world - base_pos_world,
        base_quat_world,
    )
    quat_base = batch_orientation_world_to_base(quat_world, base_quat_world)
    return torch.cat([pos_base, quat_base], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# 基座位置采样（门外侧扇形环）
# ═══════════════════════════════════════════════════════════════════════


def sample_base_poses(
    n: int,
    *,
    door_center_xy: tuple[float, float] = (2.95, 0.00),
    base_reference_xy: tuple[float, float] = (3.72, 0.27),
    base_height: float = 0.12,
    radius_range: tuple[float, float] = (0.45, 0.60),
    sector_half_angle_deg: float = 20.0,
    yaw_delta_deg: float = 10.0,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """在门外侧扇形环中批量采样 base pose。

    Parameters
    ----------
    n : int
        采样数量。

    Returns
    -------
    positions : (n, 3)  世界坐标 [x, y, z]
    yaws : (n,)  弧度
    """
    dev = torch.device(device)
    center = torch.tensor(door_center_xy, dtype=torch.float32, device=dev)
    reference = torch.tensor(base_reference_xy, dtype=torch.float32, device=dev)

    # 标称角度（从门中心指向参考点）
    diff = reference - center
    nominal_angle = torch.atan2(diff[1], diff[0])
    sector_half = math.radians(sector_half_angle_deg)

    # 采样半径与角度
    radii = torch.empty(n, device=dev).uniform_(radius_range[0], radius_range[1])
    angles = torch.empty(n, device=dev).uniform_(
        float(nominal_angle) - sector_half,
        float(nominal_angle) + sector_half,
    )

    # 计算 xy 坐标
    base_x = center[0] + radii * angles.cos()
    base_y = center[1] + radii * angles.sin()
    base_z = torch.full((n,), base_height, device=dev)
    positions = torch.stack([base_x, base_y, base_z], dim=-1)

    # 采样 yaw：朝向门中心 ± delta
    yaw_delta = math.radians(yaw_delta_deg)
    nominal_yaws = torch.atan2(
        center[1] - base_y,
        center[0] - base_x,
    )
    yaws = nominal_yaws + torch.empty(n, device=dev).uniform_(-yaw_delta, yaw_delta)

    return positions, yaws


# ═══════════════════════════════════════════════════════════════════════
# 相对位置旋转（持杯偏移随 base yaw 旋转）
# ═══════════════════════════════════════════════════════════════════════


def batch_rotate_relative_by_yaw(
    relative_xyz: Tensor,
    yaw: Tensor,
) -> Tensor:
    """将 base_link 局部偏移按 yaw 角旋转到世界系。

    Parameters
    ----------
    relative_xyz : (N, 3) 或 (3,)  base_link 局部坐标
    yaw : (N,) 弧度

    Returns
    -------
    (N, 3) 旋转后的世界系偏移
    """
    cos_y = yaw.cos()
    sin_y = yaw.sin()

    if relative_xyz.dim() == 1:
        relative_xyz = relative_xyz.unsqueeze(0).expand(yaw.shape[0], -1)

    x_local = relative_xyz[..., 0]
    y_local = relative_xyz[..., 1]
    z_local = relative_xyz[..., 2]

    x_world = x_local * cos_y - y_local * sin_y
    y_world = x_local * sin_y + y_local * cos_y

    return torch.stack([x_world, y_world, z_local], dim=-1)


def compute_relative_point_velocity_world(
    *,
    point_pos_w: Tensor,
    point_lin_vel_w: Tensor,
    base_pos_w: Tensor,
    base_lin_vel_w: Tensor,
    base_ang_vel_w: Tensor,
) -> Tensor:
    """Compute the point linear velocity relative to a moving base in world frame.

    The base contribution includes translational motion and the rigid-body
    rotational component ``omega_base x (p_point - p_base)``.
    """
    relative_pos_w = point_pos_w - base_pos_w
    rigid_body_velocity_w = base_lin_vel_w + torch.linalg.cross(base_ang_vel_w, relative_pos_w, dim=-1)
    return point_lin_vel_w - rigid_body_velocity_w


def compute_relative_angular_velocity_world(
    *,
    point_ang_vel_w: Tensor,
    base_ang_vel_w: Tensor,
) -> Tensor:
    """Compute the angular velocity of a point relative to the base in world frame."""
    return point_ang_vel_w - base_ang_vel_w
