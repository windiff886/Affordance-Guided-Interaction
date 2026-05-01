"""GPU-batch handle-free push-door environment — Isaac Lab DirectRLEnv.

One teacher PPO policy controls dual arms + Clearpath Dingo base to push open
a handle-free door and traverse the doorway.

Key design:
    - All per-env state uses (num_envs, ...) shaped torch tensors
    - Observation/reward/termination are pure tensor ops, no Python loops
    - 79D symmetric actor-critic observation
    - 15D raw Gaussian action (NOT clipped to [-1,1])
    - Paper-aligned opening/passing/shaping reward
    - Door-frame reset distribution
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .door_push_env_cfg import (
    DoorPushEnvCfg,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    PLANAR_BASE_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    BASE_LINK_NAME,
    LEFT_EE_LINK_NAME,
    RIGHT_EE_LINK_NAME,
    LEFT_ARM_JOINT1_ANCHOR_LINK_NAME,
    RIGHT_ARM_JOINT1_ANCHOR_LINK_NAME,
    DOOR_LEAF_BODY_NAME,
    GRIPPER_CLOSED_DEG,
)
from .batch_math import (
    batch_quat_to_rotation_matrix,
    batch_yaw_from_quat,
    batch_pose_world_to_base,
    batch_vector_world_to_base,
    sample_base_poses_in_door_frame,
)
from .base_control_math import (
    map_raw_base_actions_to_command,
    project_body_twist_to_planar_joint_targets,
    compute_root_force_torque_targets,
    clip_wheel_velocity_targets,
    build_holonomic_wheel_target_matrix,
    resolve_holonomic_wheel_axis,
    project_base_twist_to_wheel_targets,
    twist_to_wheel_angular_velocity_targets,
)
from .door_reward_math import (
    compute_opening_reward,
    compute_stage,
    compute_passing_reward,
    compute_min_arm_motion_reward,
    compute_stretched_arm_penalty,
    compute_end_effector_to_door_proximity_reward,
    compute_command_limit_penalty,
    compute_collision_penalty,
)
from .joint_target_math import compute_torque_proxy_joint_targets
from .physx_mass_ops import (
    build_articulation_mass_update,
)

logger = logging.getLogger(__name__)

# Door geometry offsets in DoorLeaf local frame
_DOOR_CENTER_OFFSET_LOCAL = (0.02, 0.45, 1.0)
_DOOR_NORMAL_LOCAL = (1.0, 0.0, 0.0)
# Doorway center offset from door root (hinge) in world frame.
# Hinge is at one edge of the doorway; center is half door width away laterally.
_DOORWAY_CENTER_OFFSET_FROM_ROOT = (0.02, 0.45, 1.0)

# Reward/extras keys
_EPISODE_REWARD_KEYS = (
    "opening",
    "opening/open_door_target",
    "passing",
    "shaping",
    "shaping/min_arm_motion",
    "shaping/stretched_arm",
    "shaping/end_effector_to_panel",
    "shaping/command_limit",
    "shaping/collision",
    "total",
)

_ACTOR_OBS_DIM = 79
_CRITIC_OBS_DIM = 79


class DoorPushEnv(DirectRLEnv):
    """GPU-batch handle-free push-door environment."""

    cfg: DoorPushEnvCfg

    def __init__(
        self,
        cfg: DoorPushEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]

        self._arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
        self._gripper_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
        self._planar_joint_ids, _ = robot.find_joints(PLANAR_BASE_JOINT_NAMES)
        self._wheel_joint_ids, _ = robot.find_joints(WHEEL_JOINT_NAMES)
        self._base_body_idx = robot.find_bodies([BASE_LINK_NAME])[0][0]
        self._left_ee_body_idx = robot.find_bodies([LEFT_EE_LINK_NAME])[0][0]
        self._right_ee_body_idx = robot.find_bodies([RIGHT_EE_LINK_NAME])[0][0]
        self._left_joint1_anchor_body_idx = robot.find_bodies([LEFT_ARM_JOINT1_ANCHOR_LINK_NAME])[0][0]
        self._right_joint1_anchor_body_idx = robot.find_bodies([RIGHT_ARM_JOINT1_ANCHOR_LINK_NAME])[0][0]

        self._door_hinge_ids, _ = door.find_joints(".*")
        self._door_panel_body_idx = door.find_bodies([DOOR_LEAF_BODY_NAME])[0][0]

        # Per-env persistent state tensors
        N = self.num_envs
        dev = self.device

        self._prev_action = torch.zeros(N, 15, device=dev)
        self._prev_arm_qd = torch.zeros(N, 12, device=dev)
        self._step_count = torch.zeros(N, dtype=torch.long, device=dev)
        self._prev_door_angle = torch.zeros(N, device=dev)

        # Domain randomization tensors
        self._door_mass = torch.zeros(N, device=dev)
        self._door_hinge_resistance = torch.zeros(N, device=dev)
        self._door_hinge_air_damping = torch.zeros(N, device=dev)
        self._door_closer_damping = torch.zeros(N, device=dev)
        self._door_hinge_dyn_torque = torch.zeros(N, device=dev)
        self._arm_kp = torch.full((N, 12), float(cfg.arm_pd_stiffness), device=dev)
        self._arm_kd = torch.full((N, 12), float(cfg.arm_pd_damping), device=dev)

        self._base_pos = torch.zeros(N, 3, device=dev)
        self._base_yaw = torch.zeros(N, device=dev)

        # Control dt and backend
        self._control_dt = self.physics_dt * self.cfg.decimation
        self._holonomic_wheel_target_matrix: Tensor | None = None
        self._base_controller_backend = str(self.cfg.base_control_backend)
        self._training_planar_base_only = bool(getattr(self.cfg, "training_planar_base_only", False))
        self._emit_wheel_debug_state = bool(getattr(self.cfg, "emit_wheel_debug_state", True))
        self._base_force_body_idx = self._base_body_idx
        self._base_force_body_name = BASE_LINK_NAME
        self._base_force_body_mass = torch.ones(N, device=dev)
        self._base_force_body_inertia_zz = torch.ones(N, device=dev)
        self._initialize_base_controller_backend()

        # Precomputed constants
        self._door_center_offset_local = torch.tensor(
            _DOOR_CENTER_OFFSET_LOCAL, device=dev, dtype=torch.float32
        )
        self._door_normal_local = torch.tensor(
            _DOOR_NORMAL_LOCAL, device=dev, dtype=torch.float32
        )
        self._doorway_center_offset_from_root = _DOORWAY_CENTER_OFFSET_FROM_ROOT
        self._arm_default_pos = torch.tensor(
            cfg.arm_default_joint_pos, device=dev, dtype=torch.float32
        )
        self._torque_limits = torch.tensor(
            cfg.effort_limits, device=dev, dtype=torch.float32
        )

        # Episode reward tracking
        self._episode_reward_sums = {
            key: torch.zeros(N, device=dev) for key in _EPISODE_REWARD_KEYS
        }

        # Gripper closed targets
        self._gripper_closed_targets = torch.full(
            (len(self._gripper_joint_ids),),
            math.radians(GRIPPER_CLOSED_DEG),
            device=dev,
            dtype=torch.float32,
        )

        # Hard collision body mask: chassis_link + arm links (left_link*, right_link*).
        # Build it in ContactSensor body order, which may differ from articulation body order.
        if "hard_contact" in self.scene.keys():
            hard_contact_body_names = self.scene["hard_contact"].body_names
        else:
            hard_contact_body_names = []
        self._hard_collision_body_mask = torch.zeros(
            len(hard_contact_body_names), dtype=torch.bool, device=dev,
        )
        for i, name in enumerate(hard_contact_body_names):
            if name == "chassis_link" or name.startswith("left_link") or name.startswith("right_link"):
                self._hard_collision_body_mask[i] = True

    def _initialize_base_controller_backend(self) -> None:
        backend = str(self.cfg.base_control_backend)
        if backend == "planar_joint_velocity":
            self._configure_planar_joint_velocity_backend()
            self._base_controller_backend = backend
            return
        if backend == "root_force_torque":
            self._configure_root_force_torque_backend()
            self._base_controller_backend = backend
            return
        if backend == "analytic_mecanum_fallback":
            self._holonomic_wheel_target_matrix = None
            self._base_controller_backend = backend
            return
        if backend != "isaac_holonomic_controller":
            raise ValueError(f"Unsupported base_control_backend: {backend}")
        try:
            self._holonomic_wheel_target_matrix = self._build_holonomic_wheel_target_matrix_from_usd()
            self._base_controller_backend = "isaac_holonomic_controller"
        except Exception as exc:
            self._holonomic_wheel_target_matrix = None
            self._base_controller_backend = "analytic_mecanum_fallback"
            logger.warning("Falling back to analytic mecanum wheel mapping: %s", exc)

    def _configure_planar_joint_velocity_backend(self) -> None:
        robot: Articulation = self.scene["robot"]
        if len(self._planar_joint_ids) != len(PLANAR_BASE_JOINT_NAMES):
            raise RuntimeError(
                f"Planar base backend requires {PLANAR_BASE_JOINT_NAMES}, "
                f"found {len(self._planar_joint_ids)}."
            )
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_effort_limit_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device),
            joint_ids=self._wheel_joint_ids,
        )

    def _configure_root_force_torque_backend(self) -> None:
        robot: Articulation = self.scene["robot"]
        force_body_name = self.cfg.base_force_body_name
        if force_body_name not in robot.body_names:
            force_body_name = BASE_LINK_NAME
        self._base_force_body_name = force_body_name
        self._base_force_body_idx = robot.find_bodies([force_body_name])[0][0]
        self._base_force_body_mass = robot.data.default_mass[:, self._base_force_body_idx].clone()
        self._base_force_body_inertia_zz = robot.data.default_inertia[:, self._base_force_body_idx, 8].clone()
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_effort_limit_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device),
            joint_ids=self._wheel_joint_ids,
        )

    def _build_holonomic_wheel_target_matrix_from_usd(self) -> Tensor:
        from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup

        robot_prim_path = self.cfg.scene.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        com_prim_path = f"{robot_prim_path}/{BASE_LINK_NAME}"
        setup = HolonomicRobotUsdSetup(robot_prim_path=robot_prim_path, com_prim_path=com_prim_path)
        if len(setup.wheel_dof_names) == 0:
            raise RuntimeError(f"No holonomic wheel metadata found under {robot_prim_path}.")

        matrix = build_holonomic_wheel_target_matrix(
            wheel_radius=setup.wheel_radius,
            wheel_positions=setup.wheel_positions,
            wheel_orientations=setup.wheel_orientations,
            mecanum_angles_deg=setup.mecanum_angles,
            wheel_axis=resolve_holonomic_wheel_axis(
                wheel_axis=setup.wheel_axis,
                wheel_joint_names=WHEEL_JOINT_NAMES,
            ),
            up_axis=setup.up_axis,
        )
        name_to_row = {str(name): index for index, name in enumerate(setup.wheel_dof_names)}
        ordered_rows = [name_to_row[name] for name in WHEEL_JOINT_NAMES]
        return matrix[ordered_rows].to(device=self.device)

    def _compute_wheel_targets(self, base_cmd: Tensor) -> Tensor:
        if self._holonomic_wheel_target_matrix is not None:
            return project_base_twist_to_wheel_targets(
                base_cmd, wheel_target_matrix=self._holonomic_wheel_target_matrix,
            )
        return twist_to_wheel_angular_velocity_targets(
            base_cmd,
            wheel_radius=self.cfg.wheel_radius,
            half_length=self.cfg.wheel_base_half_length,
            half_width=self.cfg.wheel_base_half_width,
        )

    def _compute_base_force_torque_targets(self, base_cmd: Tensor) -> tuple[Tensor, Tensor]:
        robot: Articulation = self.scene["robot"]
        body_quat = robot.data.body_quat_w[:, self._base_force_body_idx]
        base_lin_vel_base = batch_vector_world_to_base(
            robot.data.body_lin_vel_w[:, self._base_force_body_idx], body_quat,
        )
        base_ang_vel_base = batch_vector_world_to_base(
            robot.data.body_ang_vel_w[:, self._base_force_body_idx], body_quat,
        )
        return compute_root_force_torque_targets(
            base_cmd,
            base_lin_vel_base=base_lin_vel_base,
            base_ang_vel_base=base_ang_vel_base,
            base_mass=self._base_force_body_mass,
            base_inertia_zz=self._base_force_body_inertia_zz,
            lin_accel_gain_xy=self.cfg.base_lin_accel_gain_xy,
            ang_accel_gain_z=self.cfg.base_ang_accel_gain_z,
            force_limit_xy=self.cfg.base_force_limit_xy,
            torque_limit_z=self.cfg.base_torque_limit_z,
        )

    def _compute_planar_joint_targets(self, base_cmd: Tensor) -> Tensor:
        robot: Articulation = self.scene["robot"]
        base_yaw = batch_yaw_from_quat(robot.data.body_quat_w[:, self._base_body_idx])
        return project_body_twist_to_planar_joint_targets(base_cmd, base_yaw=base_yaw)

    # Scene setup
    def _setup_scene(self) -> None:
        return

    # Action execution
    def _pre_physics_step(self, actions: Tensor) -> None:
        robot: Articulation = self.scene["robot"]

        # Cache raw actions before any mapping
        self._prev_action = actions.clone()

        arm_actions = actions[:, :12]
        base_actions = actions[:, 12:]

        # Arm: torque-proxy target mapping
        current_q = robot.data.joint_pos[:, self._arm_joint_ids]
        q_target_cmd = compute_torque_proxy_joint_targets(
            arm_actions,
            default_joint_pos=self._arm_default_pos,
            current_joint_pos=current_q,
            torque_limits=self._torque_limits,
            stiffness=self._arm_kp,
            action_scale=self.cfg.arm_action_scale_rad,
            sigma=self.cfg.torque_proxy_sigma,
        )

        # Base: raw action to command with saturation and deadband
        base_cmd = map_raw_base_actions_to_command(
            base_actions,
            max_lin_vel_x=self.cfg.base_max_lin_vel_x,
            max_lin_vel_y=self.cfg.base_max_lin_vel_y,
            max_ang_vel_z=self.cfg.base_max_ang_vel_z,
            deadband=self.cfg.base_command_deadband,
        )

        # Write arm targets
        robot.set_joint_position_target(q_target_cmd, joint_ids=self._arm_joint_ids)

        # Write base commands
        if self._base_controller_backend == "planar_joint_velocity":
            planar_targets = self._compute_planar_joint_targets(base_cmd)
            robot.set_joint_velocity_target(planar_targets, joint_ids=self._planar_joint_ids)
        elif self._base_controller_backend == "root_force_torque":
            force_cmd, torque_cmd = self._compute_base_force_torque_targets(base_cmd)
            robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=force_cmd.unsqueeze(1),
                torques=torque_cmd.unsqueeze(1),
                body_ids=[self._base_force_body_idx],
                is_global=False,
            )
        else:
            wheel_targets = self._compute_wheel_targets(base_cmd)
            wheel_targets, _ = clip_wheel_velocity_targets(
                wheel_targets, velocity_limit=self.cfg.wheel_velocity_limit,
            )
            robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_ids)

        # Gripper hold closed
        robot.set_joint_position_target(
            self._gripper_closed_targets.unsqueeze(0).expand(self.num_envs, -1),
            joint_ids=self._gripper_joint_ids,
        )

    def _apply_action(self) -> None:
        if self._base_controller_backend == "root_force_torque":
            pass  # wrench already written in _pre_physics_step
        return

    # Observation: 79D symmetric
    def _get_observations(self) -> dict:
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]

        # Arm joint state
        arm_q = robot.data.joint_pos[:, self._arm_joint_ids]    # (N, 12)
        arm_qd = robot.data.joint_vel[:, self._arm_joint_ids]   # (N, 12)

        # Base state
        body_pos_w = robot.data.body_pos_w
        body_quat_w = robot.data.body_quat_w
        base_pos = body_pos_w[:, self._base_body_idx]       # (N, 3)
        base_quat = body_quat_w[:, self._base_body_idx]     # (N, 4)
        base_lv = body_pos_w[:, self._base_body_idx]        # will be overwritten
        base_lv = robot.data.body_lin_vel_w[:, self._base_body_idx]
        base_av = robot.data.body_ang_vel_w[:, self._base_body_idx]

        # Base twist in base frame
        base_lin_vel_base = batch_vector_world_to_base(base_lv, base_quat)  # (N, 3)
        base_ang_vel_base = batch_vector_world_to_base(base_av, base_quat)  # (N, 3)

        base_twist = torch.cat([
            base_lin_vel_base[:, :2],    # v_x, v_y
            base_ang_vel_base[:, 2:3],   # omega_z
        ], dim=-1)  # (N, 3)

        left_ee_pose_base = batch_pose_world_to_base(
            robot.data.body_pos_w[:, self._left_ee_body_idx],
            robot.data.body_quat_w[:, self._left_ee_body_idx],
            base_pos,
            base_quat,
        )
        right_ee_pose_base = batch_pose_world_to_base(
            robot.data.body_pos_w[:, self._right_ee_body_idx],
            robot.data.body_quat_w[:, self._right_ee_body_idx],
            base_pos,
            base_quat,
        )

        # Door state
        door_root_pos_w = door.data.root_pos_w       # (N, 3)
        door_root_quat_w = door.data.root_quat_w     # (N, 4)
        door_leaf_pos_w = door.data.body_pos_w[:, self._door_panel_body_idx]
        door_leaf_quat_w = door.data.body_quat_w[:, self._door_panel_body_idx]
        theta = door.data.joint_pos[:, 0]             # (N,)
        theta_dot = door.data.joint_vel[:, 0]          # (N,)

        # Per-step hinge dynamics torque: reflects actual modeled dynamics
        # tau_dyn = resistance + K_ar * |theta_dot|^2 + K_dc * |theta_dot|
        abs_theta_dot = theta_dot.abs()
        self._door_hinge_dyn_torque = (
            self._door_hinge_resistance
            + self._door_hinge_air_damping * abs_theta_dot.square()
            + self._door_closer_damping * abs_theta_dot
        )

        # Apply hinge resistance + quadratic damping as joint effort
        # Linear damping is handled by the joint damping parameter.
        hinge_effort = torch.zeros(self.num_envs, 1, device=self.device)
        motion_sign = torch.sign(theta_dot)
        hinge_effort[:, 0] = -(
            self._door_hinge_resistance * motion_sign
            + self._door_hinge_air_damping * theta_dot.square() * motion_sign
        )
        door.set_joint_effort_target(hinge_effort)

        # Door geometry in base frame
        R_world_from_leaf = batch_quat_to_rotation_matrix(door_leaf_quat_w)

        # Door center in world, then to base frame
        door_center_w = door_leaf_pos_w + torch.bmm(
            R_world_from_leaf,
            self._door_center_offset_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)
        door_center_base = batch_vector_world_to_base(door_center_w - base_pos, base_quat)

        # Door normal in base frame
        door_normal_w = torch.bmm(
            R_world_from_leaf,
            self._door_normal_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)
        door_normal_base = batch_vector_world_to_base(door_normal_w, base_quat)

        # Doorway frame geometry in base frame
        # Doorway center = door root + fixed offset (center of the opening)
        doorway_center_offset = torch.tensor(
            self._doorway_center_offset_from_root, device=self.device, dtype=torch.float32
        )
        doorway_center_w = door_root_pos_w + doorway_center_offset.unsqueeze(0)
        doorway_center_base = batch_vector_world_to_base(doorway_center_w - base_pos, base_quat)

        # Cross direction: outside -> inside, expressed in base frame
        cross_dir_world = torch.tensor(
            [float(self.cfg.door_cross_dir_xy[0]),
             float(self.cfg.door_cross_dir_xy[1]),
             0.0],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)
        cross_dir_base = batch_vector_world_to_base(cross_dir_world, base_quat)

        # Lateral direction
        lat_dir_world = torch.tensor(
            [float(self.cfg.door_lateral_dir_xy[0]),
             float(self.cfg.door_lateral_dir_xy[1]),
             0.0],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)
        lat_dir_base = batch_vector_world_to_base(lat_dir_world, base_quat)

        # Hinge reference point in base frame (door root position)
        hinge_point_base = batch_vector_world_to_base(door_root_pos_w - base_pos, base_quat)

        # Stage
        stage = compute_stage(theta, self.cfg.theta_pass)  # (N,)

        # Build 79D observation in exact order
        obs = torch.cat([
            base_twist,              # 3
            arm_q,                   # 12
            arm_qd,                  # 12
            left_ee_pose_base,       # 7
            right_ee_pose_base,      # 7
            self._prev_action,       # 15
            doorway_center_base,     # 3
            cross_dir_base,          # 3
            lat_dir_base,            # 3
            hinge_point_base,        # 3
            door_normal_base,        # 3
            door_center_base,        # 3
            theta.unsqueeze(-1),     # 1
            theta_dot.unsqueeze(-1), # 1
            self._door_mass.unsqueeze(-1),         # 1
            self._door_hinge_dyn_torque.unsqueeze(-1),  # 1
            stage.unsqueeze(-1),     # 1
        ], dim=-1)  # (N, 79)

        return {"policy": obs, "critic": obs}

    # Reward computation
    def _get_rewards(self) -> Tensor:
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]

        theta = door.data.joint_pos[:, 0]
        theta_dot = door.data.joint_vel[:, 0]

        reward_info: dict[str, Tensor] = {}

        # Opening reward
        r_od = compute_opening_reward(theta, self.cfg.theta_hat)
        r_o = self.cfg.rew_opening_scale * r_od

        # Stage
        stage = compute_stage(theta, self.cfg.theta_pass)

        # Passing reward — piecewise progress vector
        body_pos_w = robot.data.body_pos_w[:, self._base_body_idx]
        body_quat_w = robot.data.body_quat_w[:, self._base_body_idx]
        base_lin_vel_base = batch_vector_world_to_base(
            robot.data.body_lin_vel_w[:, self._base_body_idx], body_quat_w
        )
        door_root_pos_w = door.data.root_pos_w
        door_leaf_pos_w = door.data.body_pos_w[:, self._door_panel_body_idx]
        door_leaf_quat_w = door.data.body_quat_w[:, self._door_panel_body_idx]
        R_world_from_leaf = batch_quat_to_rotation_matrix(door_leaf_quat_w)
        door_center_w = door_leaf_pos_w + torch.bmm(
            R_world_from_leaf,
            self._door_center_offset_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)
        door_center_base = batch_vector_world_to_base(door_center_w - body_pos_w, body_quat_w)

        # Doorway center in world frame (fixed offset from door root)
        doorway_center_offset = torch.tensor(
            self._doorway_center_offset_from_root, device=self.device, dtype=torch.float32
        )
        doorway_center_w = door_root_pos_w + doorway_center_offset.unsqueeze(0)

        # Cross progress to decide piecewise branch
        cross_progress = self._compute_base_cross_progress(body_pos_w, door_root_pos_w)
        not_crossed = cross_progress < 0  # still on outside

        # Before crossing: progress dir toward doorway center
        rel_to_doorway = doorway_center_w - body_pos_w
        dir_to_doorway = rel_to_doorway / rel_to_doorway.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dir_to_doorway_base = batch_vector_world_to_base(dir_to_doorway, body_quat_w)

        # After crossing: use e_cross in base frame
        cross_dir_world = torch.tensor(
            [float(self.cfg.door_cross_dir_xy[0]),
             float(self.cfg.door_cross_dir_xy[1]),
             0.0],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)
        cross_dir_base = batch_vector_world_to_base(cross_dir_world, body_quat_w)

        progress_dir_base = torch.where(
            not_crossed.unsqueeze(-1).expand_as(dir_to_doorway_base),
            dir_to_doorway_base,
            cross_dir_base,
        )

        r_p = compute_passing_reward(
            progress_dir_base, base_lin_vel_base,
            max_speed=self.cfg.rew_passing_max_speed,
        )

        # Shaping rewards
        arm_qd = robot.data.joint_vel[:, self._arm_joint_ids]
        arm_qdd = (arm_qd - self._prev_arm_qd) / self._control_dt
        r_ma = compute_min_arm_motion_reward(arm_qd, arm_qdd)

        # EE positions for stretched arm penalty
        left_ee_pos_w = robot.data.body_pos_w[:, self._left_ee_body_idx]
        right_ee_pos_w = robot.data.body_pos_w[:, self._right_ee_body_idx]
        left_ee_base = batch_vector_world_to_base(left_ee_pos_w - body_pos_w, body_quat_w)
        right_ee_base = batch_vector_world_to_base(right_ee_pos_w - body_pos_w, body_quat_w)
        ee_pos_base = torch.cat([left_ee_base, right_ee_base], dim=-1)

        # Shoulder anchors: left/right arm joint1 anchor links in base frame.
        left_shoulder_pos_w = robot.data.body_pos_w[:, self._left_joint1_anchor_body_idx]
        right_shoulder_pos_w = robot.data.body_pos_w[:, self._right_joint1_anchor_body_idx]
        left_shoulder_base = batch_vector_world_to_base(left_shoulder_pos_w - body_pos_w, body_quat_w)
        right_shoulder_base = batch_vector_world_to_base(right_shoulder_pos_w - body_pos_w, body_quat_w)
        shoulder_pos_base = torch.cat([left_shoulder_base, right_shoulder_base], dim=-1)

        r_psa = compute_stretched_arm_penalty(
            ee_pos_base, shoulder_pos_base,
            threshold=self.cfg.rew_stretched_arm_threshold,
            scale=self.cfg.rew_stretched_arm_scale,
        )
        r_eep = compute_end_effector_to_door_proximity_reward(
            left_ee_base, right_ee_base, door_center_base
        )

        r_pcl = compute_command_limit_penalty(
            self._prev_action,
            action_limits=self.cfg.rew_cmd_limit_threshold,
        )

        # Collision: hard collision detection via PhysX contact forces
        hard_collision = self._detect_hard_collision()
        r_pc = compute_collision_penalty(hard_collision)

        r_s = (self.cfg.rew_ma_weight * r_ma
               + self.cfg.rew_psa_weight * r_psa
               + self.cfg.rew_eep_weight * r_eep
               + self.cfg.rew_pcl_weight * r_pcl
               + self.cfg.rew_pc_weight * r_pc)

        # Total reward
        r_o_max = self.cfg.rew_opening_scale  # max opening reward
        r_total = torch.where(
            stage < 0.5,
            r_o + r_s,
            r_o_max + r_p + r_s,
        )

        reward_info["opening"] = r_o
        reward_info["opening/open_door_target"] = r_od
        reward_info["passing"] = r_p
        reward_info["shaping"] = r_s
        reward_info["shaping/min_arm_motion"] = r_ma
        reward_info["shaping/stretched_arm"] = r_psa
        reward_info["shaping/end_effector_to_panel"] = r_eep
        reward_info["shaping/command_limit"] = r_pcl
        reward_info["shaping/collision"] = r_pc
        reward_info["total"] = r_total

        for key, value in reward_info.items():
            self._episode_reward_sums[key] += value

        # Episode extras for done envs
        success = self._compute_success()
        opened_enough = theta >= self.cfg.theta_open
        truncated = (self._step_count + 1) >= self.max_episode_length
        hard_collision_event = hard_collision
        reverse_open_event = theta < self.cfg.reverse_angle_limit
        fail_timeout = truncated & ~success

        done_mask = success | truncated
        if done_mask.any():
            done_env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self.extras["episode_reward_info"] = {
                key: value[done_env_ids].clone() for key, value in self._episode_reward_sums.items()
            }
            self.extras["episode_reward_info"]["_step_count"] = (self._step_count[done_env_ids] + 1).float().clone()
            self.extras["success"] = success[done_env_ids].clone()
            self.extras["opened_enough"] = opened_enough[done_env_ids].clone()
            self.extras["passed_through"] = self._compute_passed_through()[done_env_ids].clone()
            self.extras["door_angle"] = theta[done_env_ids].clone()
            self.extras["base_cross_progress"] = self._compute_base_cross_progress(body_pos_w, door_root_pos_w)[done_env_ids].clone()
            self.extras["hard_collision"] = hard_collision_event[done_env_ids].clone()
            self.extras["reverse_open"] = reverse_open_event[done_env_ids].clone()
            self.extras["fail_timeout"] = fail_timeout[done_env_ids].clone()
            self.extras["no_collision"] = ~hard_collision_event[done_env_ids].clone()

            # Task diagnostics
            self.extras["door_angular_velocity"] = theta_dot[done_env_ids].clone()
            self.extras["stage"] = stage[done_env_ids].clone()

            lat_dir = torch.tensor(
                [float(self.cfg.door_lateral_dir_xy[0]),
                 float(self.cfg.door_lateral_dir_xy[1]),
                 0.0],
                device=self.device, dtype=torch.float32,
            )
            lateral_error = ((body_pos_w[done_env_ids] - door_root_pos_w[done_env_ids]) * lat_dir).sum(dim=-1)
            self.extras["lateral_error"] = lateral_error.clone()

            base_yaw_done = batch_yaw_from_quat(body_quat_w[done_env_ids])
            desired_yaw = math.atan2(float(self.cfg.door_cross_dir_xy[1]), float(self.cfg.door_cross_dir_xy[0]))
            heading_error = (base_yaw_done - desired_yaw).abs()
            heading_error = torch.remainder(heading_error + math.pi, 2 * math.pi) - math.pi
            self.extras["heading_error"] = heading_error.abs().clone()

            # Randomization diagnostics (scalar means across done envs)
            self.extras["random/door_mass"] = self._door_mass[done_env_ids].clone()
            self.extras["random/hinge_resistance"] = self._door_hinge_resistance[done_env_ids].clone()
            self.extras["random/reset_x"] = self._base_pos[done_env_ids, 0].clone()
            self.extras["random/reset_y"] = self._base_pos[done_env_ids, 1].clone()
            self.extras["random/reset_yaw"] = self._base_yaw[done_env_ids].clone()

        # Update caches
        self._prev_door_angle = theta.clone()
        self._prev_arm_qd = arm_qd.clone()

        return r_total

    # Termination
    def _get_dones(self) -> tuple[Tensor, Tensor]:
        self._step_count += 1

        success = self._compute_success()
        terminated = success
        truncated = self._step_count >= self.max_episode_length

        return terminated, truncated

    # Reset
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = self.scene["robot"]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # 1. Sample domain randomization
        self._sample_domain_randomization(env_ids)

        # 2. Sample base pose in door frame
        base_pos, base_yaw = sample_base_poses_in_door_frame(
            n,
            door_origin_xy=self.cfg.door_center_xy,
            door_cross_dir_xy=self.cfg.door_cross_dir_xy,
            door_lateral_dir_xy=self.cfg.door_lateral_dir_xy,
            base_height=self.cfg.base_height,
            distance_to_wall_range=self.cfg.reset_distance_to_wall_range,
            lateral_offset_range=self.cfg.reset_lateral_offset_range,
            yaw_range=self.cfg.reset_yaw_range,
            device=self.device,
        )
        self._base_pos[env_ids] = base_pos
        self._base_yaw[env_ids] = base_yaw

        # 3. Write robot root state
        robot: Articulation = self.scene["robot"]
        default_root = robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        default_root[:, 7:] = 0.0
        robot.write_root_state_to_sim(default_root, env_ids)

        # Set arm joints to the configured default pose.
        default_jpos = robot.data.default_joint_pos[env_ids].clone()
        default_jvel = robot.data.default_joint_vel[env_ids].clone()

        # Arm defaults
        arm_default = torch.tensor(self.cfg.arm_default_joint_pos, device=self.device, dtype=torch.float32)
        default_jpos[:, self._arm_joint_ids] = arm_default.unsqueeze(0).expand(n, -1)

        # Gripper closed (position, not just target)
        default_jpos[:, self._gripper_joint_ids] = self._gripper_closed_targets.unsqueeze(0).expand(n, -1)

        # Planar base pose
        default_jpos[:, self._planar_joint_ids[0]] = self._base_pos[env_ids, 0]
        default_jpos[:, self._planar_joint_ids[1]] = self._base_pos[env_ids, 1]
        default_jpos[:, self._planar_joint_ids[2]] = self._base_yaw[env_ids]

        # Initial base planar velocity
        lin_vel_range = self.cfg.reset_base_lin_vel_xy_range
        default_jvel[:, self._planar_joint_ids[0]] = torch.empty(n, device=self.device).uniform_(*lin_vel_range)
        default_jvel[:, self._planar_joint_ids[1]] = torch.empty(n, device=self.device).uniform_(*lin_vel_range)
        default_jvel[:, self._planar_joint_ids[2]] = self.cfg.reset_base_ang_vel_z
        default_jvel[:, self._arm_joint_ids] = 0.0
        default_jvel[:, self._wheel_joint_ids] = 0.0

        robot.write_joint_state_to_sim(default_jpos, default_jvel, None, env_ids)

        # Gripper closed
        gripper_pos = self._gripper_closed_targets.unsqueeze(0).expand(n, -1)
        robot.set_joint_position_target(gripper_pos, joint_ids=self._gripper_joint_ids, env_ids=env_ids)

        # 4. Reset door
        door: Articulation = self.scene["door"]
        zero_door_pos = torch.zeros(n, door.num_joints, device=self.device)
        zero_door_vel = torch.zeros(n, door.num_joints, device=self.device)
        door.write_joint_state_to_sim(zero_door_pos, zero_door_vel, None, env_ids)

        # 5. Control targets
        arm_target = default_jpos[:, self._arm_joint_ids]
        robot.set_joint_position_target(arm_target, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        zero_planar_vel = torch.zeros((n, len(self._planar_joint_ids)), device=self.device)
        robot.set_joint_velocity_target(zero_planar_vel, joint_ids=self._planar_joint_ids, env_ids=env_ids)
        if not self._training_planar_base_only:
            zero_wheel_vel = torch.zeros((n, len(self._wheel_joint_ids)), device=self.device)
            robot.set_joint_velocity_target(zero_wheel_vel, joint_ids=self._wheel_joint_ids, env_ids=env_ids)
        self.scene.write_data_to_sim()

        # 6. Apply domain randomization
        self._apply_domain_params(env_ids)

        # 7. Reset per-env state
        self._step_count[env_ids] = 0
        self._prev_door_angle[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._prev_arm_qd[env_ids] = 0.0
        for value in self._episode_reward_sums.values():
            value[env_ids] = 0.0

        robot.instantaneous_wrench_composer.reset(env_ids)

    def _sample_domain_randomization(self, env_ids: Tensor) -> None:
        n = len(env_ids)
        cfg = self.cfg

        # Door mass [15, 75]
        self._door_mass[env_ids] = torch.empty(n, device=self.device).uniform_(*cfg.door_mass_range)

        # Hinge resistance [0, 30], zero with prob 0.2
        resistance = torch.empty(n, device=self.device).uniform_(*cfg.door_hinge_resistance_range)
        zero_mask = torch.empty(n, device=self.device).uniform_(0, 1) < cfg.door_hinge_resistance_zero_prob
        resistance[zero_mask] = 0.0
        self._door_hinge_resistance[env_ids] = resistance

        # Hinge air damping [0, 4]
        self._door_hinge_air_damping[env_ids] = torch.empty(n, device=self.device).uniform_(*cfg.door_hinge_air_damping_range)

        # Closer damping = alpha * resistance, zero with prob 0.4
        alpha = torch.empty(n, device=self.device).uniform_(*cfg.door_closer_damping_alpha_range)
        closer_damping = alpha * resistance
        damping_zero_mask = torch.empty(n, device=self.device).uniform_(0, 1) < cfg.door_hinge_damping_zero_prob
        closer_damping[damping_zero_mask] = 0.0
        self._door_closer_damping[env_ids] = closer_damping

        self._arm_kp[env_ids] = float(cfg.arm_pd_stiffness)
        self._arm_kd[env_ids] = float(cfg.arm_pd_damping)

    def _apply_domain_params(self, env_ids: Tensor) -> None:
        door: Articulation = self.scene["door"]
        n = len(env_ids)
        dev = self.device

        # Door mass
        door_masses = door.root_physx_view.get_masses()
        if door_masses is not None:
            payload, target_env_ids = build_articulation_mass_update(
                masses=door_masses.to(dev),
                env_ids=env_ids,
                all_env_ids=door._ALL_INDICES.to(dev),
                body_idx=self._door_panel_body_idx,
                body_masses=self._door_mass[env_ids],
            )
            door.root_physx_view.set_masses(payload.cpu(), target_env_ids.cpu())

        # Door hinge damping
        if hasattr(door, "write_joint_damping_to_sim"):
            total_damping = (self._door_hinge_air_damping[env_ids]
                             + self._door_closer_damping[env_ids])
            damping = total_damping.unsqueeze(-1).expand(n, door.num_joints)
            door.write_joint_damping_to_sim(damping, env_ids=env_ids)

        # Arm PD gains (if supported per-env)
        robot: Articulation = self.scene["robot"]
        if hasattr(robot, "write_joint_stiffness_to_sim"):
            robot.write_joint_stiffness_to_sim(
                self._arm_kp[env_ids], joint_ids=self._arm_joint_ids, env_ids=env_ids
            )
            robot.write_joint_damping_to_sim(
                self._arm_kd[env_ids], joint_ids=self._arm_joint_ids, env_ids=env_ids
            )

    # Success/failure helpers
    def _compute_success(self) -> Tensor:
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        body_pos_w = robot.data.body_pos_w[:, self._base_body_idx]
        door_root_pos_w = door.data.root_pos_w
        cross_progress = self._compute_base_cross_progress(body_pos_w, door_root_pos_w)
        door_angle = door.data.joint_pos[:, 0]
        opened_enough = door_angle >= self.cfg.theta_open
        return opened_enough & (cross_progress >= 0.5)

    def _compute_passed_through(self) -> Tensor:
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        body_pos_w = robot.data.body_pos_w[:, self._base_body_idx]
        door_root_pos_w = door.data.root_pos_w
        return self._compute_base_cross_progress(body_pos_w, door_root_pos_w) >= 0.5

    def _compute_base_cross_progress(self, base_pos_w: Tensor, door_root_pos_w: Tensor) -> Tensor:
        cross_dir = torch.tensor(
            [float(self.cfg.door_cross_dir_xy[0]),
             float(self.cfg.door_cross_dir_xy[1]),
             0.0],
            device=self.device, dtype=torch.float32
        )
        rel = base_pos_w - door_root_pos_w
        # x^D = dot(base - door_root, e_cross)
        # cross_dir points outside->inside; outside gives negative progress.
        return (rel * cross_dir).sum(dim=-1)

    def _detect_hard_collision(self) -> Tensor:
        """Detect hard collisions on chassis and arm links via ContactSensor.

        Reads net contact forces from the ``hard_contact`` scene sensor and
        checks bodies matching ``chassis_link``, ``left_link*``, ``right_link*``.
        Returns all-false if the sensor is missing or not yet initialized.
        """
        if "hard_contact" not in self.scene.keys():
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        sensor = self.scene["hard_contact"]
        net_forces_w = sensor.data.net_forces_w
        if net_forces_w is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        force_mag = net_forces_w.norm(dim=-1)  # (num_envs, num_bodies)

        # Fail closed if sensor body count doesn't match the cached sensor-body mask.
        if force_mag.shape[1] != self._hard_collision_body_mask.shape[0]:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        monitored = force_mag[:, self._hard_collision_body_mask]
        threshold = self.cfg.hard_collision_force_threshold
        return (monitored > threshold).any(dim=-1)

    # Debug and external interfaces
    def get_debug_state(self) -> dict[str, Tensor]:
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        base_pos_w = robot.data.body_pos_w[:, self._base_body_idx].clone()
        base_quat_w = robot.data.body_quat_w[:, self._base_body_idx].clone()
        base_lin_vel_base = batch_vector_world_to_base(
            robot.data.body_lin_vel_w[:, self._base_body_idx],
            robot.data.body_quat_w[:, self._base_body_idx],
        )
        base_ang_vel_base = batch_vector_world_to_base(
            robot.data.body_ang_vel_w[:, self._base_body_idx],
            robot.data.body_quat_w[:, self._base_body_idx],
        )
        door_angle = door.data.joint_pos[:, 0].clone()

        # Arm joint targets (from previous action mapping)
        arm_target = self._prev_action[:, :12] * self.cfg.arm_action_scale_rad + self._arm_default_pos.unsqueeze(0)

        # Base command from previous action
        base_cmd = map_raw_base_actions_to_command(
            self._prev_action[:, 12:],
            max_lin_vel_x=self.cfg.base_max_lin_vel_x,
            max_lin_vel_y=self.cfg.base_max_lin_vel_y,
            max_ang_vel_z=self.cfg.base_max_ang_vel_z,
            deadband=self.cfg.base_command_deadband,
        )

        # Planar joint state
        planar_pos = robot.data.joint_pos[:, self._planar_joint_ids].clone()
        planar_vel = robot.data.joint_vel[:, self._planar_joint_ids].clone()

        # Wheel state
        wheel_vel = robot.data.joint_vel[:, self._wheel_joint_ids].clone()

        return {
            "door_angle": door_angle,
            "opened_enough": door_angle >= self.cfg.theta_open,
            "success": self._compute_success(),
            "passed_through": self._compute_passed_through(),
            "arm_joint_positions": robot.data.joint_pos[:, self._arm_joint_ids].clone(),
            "arm_joint_targets": arm_target.clone(),
            "base_pos_w": base_pos_w,
            "base_quat_w": base_quat_w,
            "base_lin_vel_base": base_lin_vel_base.clone(),
            "base_ang_vel_base": base_ang_vel_base.clone(),
            "base_cmd": base_cmd.clone(),
            "base_controller_backend": self._base_controller_backend,
            "base_force_body_name": self._base_force_body_name,
            "base_force_cmd": torch.zeros(self.num_envs, 3, device=self.device),
            "base_torque_cmd": torch.zeros(self.num_envs, 3, device=self.device),
            "planar_joint_positions": planar_pos,
            "planar_joint_velocities": planar_vel,
            "planar_joint_targets": torch.zeros_like(planar_vel),
            "prev_action": self._prev_action.clone(),
            "wheel_joint_velocities": wheel_vel,
            "wheel_saturation_ratio": torch.zeros(self.num_envs, device=self.device),
            "door_mass": self._door_mass.clone(),
            "hinge_resistance": self._door_hinge_resistance.clone(),
            "hinge_dyn_torque": self._door_hinge_dyn_torque.clone(),
            "arm_kp": self._arm_kp.clone(),
            "arm_kd": self._arm_kd.clone(),
        }

    def get_visual_observation(self) -> dict | None:
        return None
