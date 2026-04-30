from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def default_dingo_mecanum_angles_deg(wheel_joint_names: Sequence[str]) -> tuple[float, ...]:
    """Return the standard X-drive mecanum roller angles for Dingo wheel order."""
    angle_by_prefix = {
        "front_left": 45.0,
        "front_right": -45.0,
        "rear_left": -45.0,
        "rear_right": 45.0,
    }
    angles: list[float] = []
    for joint_name in wheel_joint_names:
        for prefix, angle in angle_by_prefix.items():
            if prefix in joint_name:
                angles.append(angle)
                break
        else:
            raise ValueError(f"Unsupported wheel joint name for Dingo mecanum layout: {joint_name}")
    return tuple(angles)


def resolve_holonomic_wheel_axis(
    *,
    wheel_axis: Sequence[float],
    wheel_joint_names: Sequence[str],
) -> tuple[float, float, float]:
    """Normalize imported wheel-axis conventions to Isaac holonomic X-drive semantics."""
    axis = tuple(float(value) for value in wheel_axis)
    if list(wheel_joint_names) == [
        "front_left_wheel",
        "front_right_wheel",
        "rear_left_wheel",
        "rear_right_wheel",
    ] and axis == (0.0, 1.0, 0.0):
        return (1.0, 0.0, 0.0)
    return axis


def build_holonomic_wheel_target_matrix(
    *,
    wheel_radius: Sequence[float],
    wheel_positions: Sequence[Sequence[float]],
    wheel_orientations: Sequence[Sequence[float]],
    mecanum_angles_deg: Sequence[float],
    wheel_axis: Sequence[float],
    up_axis: Sequence[float],
    controller_cls=None,
) -> Tensor:
    """Probe Isaac Sim's holonomic controller with basis commands and build a wheel map."""
    if controller_cls is None:
        from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController

        controller_cls = HolonomicController

    controller = controller_cls(
        name="door_push_mobile_base",
        wheel_radius=np.asarray(wheel_radius, dtype=float),
        wheel_positions=np.asarray(wheel_positions, dtype=float),
        wheel_orientations=np.asarray(wheel_orientations, dtype=float),
        mecanum_angles=np.asarray(mecanum_angles_deg, dtype=float),
        wheel_axis=np.asarray(wheel_axis, dtype=float),
        up_axis=np.asarray(up_axis, dtype=float),
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        max_wheel_speed=1.0e20,
    )
    basis_commands = (
        np.asarray([1.0, 0.0, 0.0], dtype=float),
        np.asarray([0.0, 1.0, 0.0], dtype=float),
        np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    columns = [
        np.asarray(controller.forward(command).joint_velocities, dtype=float) for command in basis_commands
    ]
    return torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)


def project_base_twist_to_wheel_targets(
    base_cmd: Tensor,
    *,
    wheel_target_matrix: Tensor,
) -> Tensor:
    """Project batched ``[vx, vy, wz]`` base commands through a precomputed wheel map."""
    matrix = wheel_target_matrix.to(device=base_cmd.device, dtype=base_cmd.dtype)
    return base_cmd @ matrix.transpose(0, 1)


def compute_root_force_torque_targets(
    base_cmd: Tensor,
    *,
    base_lin_vel_base: Tensor,
    base_ang_vel_base: Tensor,
    base_mass: Tensor | float,
    base_inertia_zz: Tensor | float,
    lin_accel_gain_xy: Sequence[float] | float,
    ang_accel_gain_z: float,
    force_limit_xy: Sequence[float] | float,
    torque_limit_z: float,
) -> tuple[Tensor, Tensor]:
    """Compute local-frame force/torque targets that track ``[vx, vy, wz]`` commands.

    The controller is a velocity-error proportional law expressed as desired acceleration:

    ``force_xy = mass * gain_xy * (cmd_xy - vel_xy)``
    ``torque_z = inertia_zz * gain_z * (cmd_wz - ang_vel_z)``

    The output wrench is clipped axis-wise and returned in the base/body local frame.
    """

    def _resolve_pair(value: Sequence[float] | float, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        if isinstance(value, Sequence):
            pair = torch.tensor(list(value), dtype=dtype, device=device)
        else:
            pair = torch.full((2,), float(value), dtype=dtype, device=device)
        if pair.shape != (2,):
            raise ValueError(f"Expected a scalar or length-2 sequence, got shape {tuple(pair.shape)}")
        return pair

    batch_size = base_cmd.shape[0]
    dtype = base_cmd.dtype
    device = base_cmd.device

    lin_gain = _resolve_pair(lin_accel_gain_xy, dtype=dtype, device=device)
    force_limit = _resolve_pair(force_limit_xy, dtype=dtype, device=device)
    mass = torch.as_tensor(base_mass, dtype=dtype, device=device).reshape(-1)
    inertia_zz = torch.as_tensor(base_inertia_zz, dtype=dtype, device=device).reshape(-1)

    if mass.numel() == 1:
        mass = mass.repeat(batch_size)
    if inertia_zz.numel() == 1:
        inertia_zz = inertia_zz.repeat(batch_size)
    if mass.shape != (batch_size,):
        raise ValueError(f"Expected base_mass to broadcast to shape ({batch_size},), got {tuple(mass.shape)}")
    if inertia_zz.shape != (batch_size,):
        raise ValueError(
            f"Expected base_inertia_zz to broadcast to shape ({batch_size},), got {tuple(inertia_zz.shape)}"
        )

    lin_error_xy = base_cmd[:, :2] - base_lin_vel_base[:, :2]
    ang_error_z = base_cmd[:, 2] - base_ang_vel_base[:, 2]

    force_xy = mass.unsqueeze(-1) * lin_gain.unsqueeze(0) * lin_error_xy
    force_xy = torch.clamp(force_xy, min=-force_limit.unsqueeze(0), max=force_limit.unsqueeze(0))

    torque_z = inertia_zz * float(ang_accel_gain_z) * ang_error_z
    torque_z = torch.clamp(torque_z, min=-float(torque_limit_z), max=float(torque_limit_z))

    forces = torch.zeros((batch_size, 3), dtype=dtype, device=device)
    torques = torch.zeros((batch_size, 3), dtype=dtype, device=device)
    forces[:, :2] = force_xy
    torques[:, 2] = torque_z
    return forces, torques


def project_body_twist_to_planar_joint_targets(
    base_cmd: Tensor,
    *,
    base_yaw: Tensor,
) -> Tensor:
    """Map body-frame ``[vx, vy, wz]`` commands to planar-joint velocity targets.

    The planar base is modeled as world-frame ``x/y`` prismatic joints plus a world-frame
    ``yaw`` revolute joint. Therefore the translational command must be rotated from the
    robot body frame into the world frame before it can be written as joint velocities.
    """
    cos_yaw = torch.cos(base_yaw)
    sin_yaw = torch.sin(base_yaw)

    vx_body = base_cmd[:, 0]
    vy_body = base_cmd[:, 1]
    wz = base_cmd[:, 2]

    x_dot_world = cos_yaw * vx_body - sin_yaw * vy_body
    y_dot_world = sin_yaw * vx_body + cos_yaw * vy_body
    return torch.stack([x_dot_world, y_dot_world, wz], dim=-1)


def rescale_normalized_base_actions(
    actions: Tensor,
    *,
    max_lin_vel_x: float,
    max_lin_vel_y: float,
    max_ang_vel_z: float,
) -> Tensor:
    """Map normalized base actions in ``[-1, 1]`` to physical twist commands."""
    bounded_actions = torch.clamp(actions, -1.0, 1.0)
    scale = torch.tensor(
        [float(max_lin_vel_x), float(max_lin_vel_y), float(max_ang_vel_z)],
        dtype=bounded_actions.dtype,
        device=bounded_actions.device,
    )
    return bounded_actions * scale.unsqueeze(0)


def twist_to_wheel_angular_velocity_targets(
    twist_cmd: Tensor,
    *,
    wheel_radius: float,
    half_length: float,
    half_width: float,
) -> Tensor:
    """Map base-frame ``[vx, vy, wz]`` commands to four wheel angular velocities.

    Wheel order is ``[front_left, front_right, rear_left, rear_right]``.
    """
    vx = twist_cmd[:, 0]
    vy = twist_cmd[:, 1]
    wz = twist_cmd[:, 2]

    lever_arm = float(half_length) + float(half_width)
    inv_radius = 1.0 / float(wheel_radius)

    front_left = (vx - vy - lever_arm * wz) * inv_radius
    front_right = (vx + vy + lever_arm * wz) * inv_radius
    rear_left = (vx + vy - lever_arm * wz) * inv_radius
    rear_right = (vx - vy + lever_arm * wz) * inv_radius
    return torch.stack([front_left, front_right, rear_left, rear_right], dim=-1)


def clip_wheel_velocity_targets(
    wheel_targets: Tensor,
    *,
    velocity_limit: float,
) -> tuple[Tensor, Tensor]:
    """Clip wheel targets and report the per-env saturation ratio."""
    limit = float(velocity_limit)
    clipped = torch.clamp(wheel_targets, min=-limit, max=limit)
    saturated = torch.abs(wheel_targets) > limit
    saturation_ratio = saturated.to(dtype=wheel_targets.dtype).mean(dim=-1)
    return clipped, saturation_ratio


def map_raw_base_actions_to_command(
    actions: Tensor,
    *,
    max_lin_vel_x: float,
    max_lin_vel_y: float,
    max_ang_vel_z: float,
    deadband: float,
) -> Tensor:
    """Map raw base actions to command-layer velocity with saturation and deadband.

    Does NOT clamp actions; clamps only the scaled command.
    """
    scale = torch.tensor(
        [float(max_lin_vel_x), float(max_lin_vel_y), float(max_ang_vel_z)],
        device=actions.device,
        dtype=actions.dtype,
    )
    raw_cmd = actions * scale.unsqueeze(0)
    cmd = torch.clamp(raw_cmd, -scale.unsqueeze(0), scale.unsqueeze(0))
    return torch.where(cmd.abs() < float(deadband), torch.zeros_like(cmd), cmd)
