"""DirectRLEnv 门推交互任务 — GPU 批量并行实现。

替代旧的 ``VecDoorEnv`` (Python for-loop) + ``DoorInteractionEnv`` (单环境)
架构，使用 Isaac Lab 的 Cloner + ArticulationView 实现真正的 GPU 批量仿真。

关键设计：
    - 所有 per-env 状态使用 ``(num_envs, ...)`` 形状的 torch tensor
    - 观测/奖励/终止判定均为纯 tensor 操作，无 Python 循环
    - 杯体预生成在场景中，reset 时 teleport 到夹爪位置或远处
    - Actor/Critic 不对称观测：Actor 含噪声 + 视觉 embedding，
      Critic 无噪声 + privileged 物理信息
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .door_push_env_cfg import (
    DoorPushEnvCfg,
    # 名称常量
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    PLANAR_BASE_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    BASE_LINK_NAME,
    LEFT_EE_LINK_NAME,
    RIGHT_EE_LINK_NAME,
    DOOR_LEAF_BODY_NAME,
    # 持杯初始化常量
    LEFT_CUP_RELATIVE_XYZ,
    RIGHT_CUP_RELATIVE_XYZ,
    LEFT_ARM_GRASP_INIT_DEG,
    RIGHT_ARM_GRASP_INIT_DEG,
    GRIPPER_OPEN_DEG,
    GRIPPER_CLOSE_DEG,
    GRIPPER_FULLY_CLOSED_DEG,
    POSE_SETTLE_STEPS,
    POST_SPAWN_SETTLE_STEPS,
    GRIPPER_CLOSE_STEPS,
    POST_CLOSE_SETTLE_STEPS,
    POST_REMOVE_SETTLE_STEPS,
    TRAY_SIZE_XYZ,
)
from .batch_math import (
    batch_quat_from_yaw,
    batch_quat_to_rotation_matrix,
    batch_yaw_from_quat,
    batch_orientation_world_to_base,
    batch_pose_world_to_base,
    batch_rotate_relative_by_yaw,
    batch_vector_world_to_base,
    sample_base_poses,
)
from .base_control_math import (
    build_holonomic_wheel_target_matrix,
    clip_wheel_velocity_targets,
    compute_root_force_torque_targets,
    project_body_twist_to_planar_joint_targets,
    resolve_holonomic_wheel_axis,
    project_base_twist_to_wheel_targets,
    rescale_normalized_base_actions,
    twist_to_wheel_angular_velocity_targets,
)
from .door_reward_math import (
    compute_base_align_reward,
    compute_base_alignment_score,
    compute_base_assist_reward,
    compute_base_corridor_excess,
    compute_base_door_sync_reward,
    compute_base_net_progress_reward,
    compute_base_range_score,
    compute_base_footprint_corners_door_frame,
    compute_base_heading_penalty,
    compute_door_angle_cross_gate,
    compute_door_angle_push_gate,
    compute_forward_progress_delta,
    compute_hand_near_gate,
    compute_base_speed_penalty,
    compute_base_speed_squared,
    compute_base_zero_speed_reward,
    compute_door_traverse_success,
    compute_inside_progress_delta,
    compute_near_line_score,
    compute_normalized_approach_score,
    compute_point_to_panel_face_distance,
    compute_point_to_segment_distance,
    compute_soft_capped_velocity_penalty,
    compute_signed_progress_delta,
    compute_signed_distance_to_plane,
    compute_target_rate_penalty,
    update_crossing_latch,
)
from .doorway_geometry import (
    DOORWAY_INNER_CORNERS_LOCAL,
    DOORWAY_LOWER_EDGE_END_LOCAL,
    DOORWAY_LOWER_EDGE_START_LOCAL,
    transform_doorway_points_to_base,
    transform_doorway_points_to_world,
)
from .gripper_hold import build_gripper_hold_targets
from .joint_target_math import (
    build_grasp_init_joint_positions,
    compute_joint_limit_margin_penalty,
    rescale_normalized_joint_actions,
)
from .physx_mass_ops import (
    build_articulation_mass_update,
    build_rigid_body_mass_update,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 门几何观测常量
# ═══════════════════════════════════════════════════════════════════════

# DoorLeaf 局部坐标系下的门板中心偏移
_DOOR_CENTER_OFFSET_LOCAL = (0.02, 0.45, 1.0)
# DoorLeaf 局部坐标系下的推门侧大表面中心偏移
_DOOR_PANEL_FACE_CENTER_OFFSET_LOCAL = (0.04, 0.45, 1.0)
# DoorLeaf 局部坐标系下的推门侧法向量 (+X)
_DOOR_NORMAL_LOCAL = (1.0, 0.0, 0.0)
_DOOR_PANEL_HALF_EXTENT_Y = 0.45
_DOOR_PANEL_HALF_EXTENT_Z = 1.0
_EPISODE_REWARD_KEYS = (
    "task",
    "task/delta",
    "task/open_bonus",
    "task/approach",
    "task/approach_raw",
    "task/base_align",
    "task/base_forward",
    "task/base_centerline",
    "task/base_net_progress",
    "task/base_assist",
    "task/base_door_sync",
    "task/base_cross",
    "stab_left",
    "stab_left/zero_acc",
    "stab_left/zero_ang",
    "stab_left/acc",
    "stab_left/ang",
    "stab_left/tilt",
    "stab_left/ee_lin_vel",
    "stab_left/ee_ang_vel",
    "stab_right",
    "stab_right/zero_acc",
    "stab_right/zero_ang",
    "stab_right/acc",
    "stab_right/ang",
    "stab_right/tilt",
    "stab_right/ee_lin_vel",
    "stab_right/ee_ang_vel",
    "safe",
    "safe/joint_vel",
    "safe/target_limit",
    "safe/target_rate",
    "safe/cup_door_prox",
    "safe/cup_drop",
    "safe/base_zero_speed",
    "safe/base_speed",
    "safe/base_cmd_delta",
    "safe/base_heading",
    "safe/base_corridor",
    "total",
)
_EPISODE_STATE_SUM_KEYS = (
    "base_push_progress_sum",
    "base_net_progress_sum",
    "base_inside_progress_sum",
    "base_corridor_excess_sum",
    "base_align_score_sum",
    "base_range_score_sum",
    "ee_left_speed_world_sum",
    "ee_right_speed_world_sum",
    "ee_left_ang_speed_world_sum",
    "ee_right_ang_speed_world_sum",
    "target_rate_l2_sum",
    "target_rate_sq_sum",
)
_EPISODE_STATE_MAX_KEYS = (
    "base_corridor_excess_max",
    "ee_left_speed_world_max",
    "ee_right_speed_world_max",
    "ee_left_ang_speed_world_max",
    "ee_right_ang_speed_world_max",
    "target_rate_l2_max",
)

# 观测维度
# actor: proprio(36) + ee(38) + context(2) + stability(2) + door_geometry(6)
#        + doorway_corners(12) + base_twist/cmd(6) = 102
# critic: actor_obs(102) + privileged(13) = 115
_ACTOR_OBS_DIM = 102
_CRITIC_OBS_DIM = 115
_DOOR_GEOMETRY_DIM = 6
_DOOR_FRAME_CORNERS_DIM = 12
_PRIVILEGED_DIM = 13


class DoorPushEnv(DirectRLEnv):
    """GPU 批量门推交互环境 — Isaac Lab DirectRLEnv 子类。

    Parameters
    ----------
    cfg : DoorPushEnvCfg
        环境与场景配置。
    render_mode : str | None
        渲染模式。
    """

    cfg: DoorPushEnvCfg

    def __init__(
        self,
        cfg: DoorPushEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # ── 解析关节 / body 索引（一次性，对所有 env 共享）──────
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        self._camera = self.scene.sensors.get("tiled_camera")

        self._arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
        self._gripper_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
        self._planar_joint_ids, _ = robot.find_joints(PLANAR_BASE_JOINT_NAMES)
        self._wheel_joint_ids, _ = robot.find_joints(WHEEL_JOINT_NAMES)
        self._base_body_idx = robot.find_bodies([BASE_LINK_NAME])[0][0]
        self._left_ee_body_idx = robot.find_bodies([LEFT_EE_LINK_NAME])[0][0]
        self._right_ee_body_idx = robot.find_bodies([RIGHT_EE_LINK_NAME])[0][0]
        self._left_grasp_joint_ids = [robot.find_joints([jname])[0][0] for jname in LEFT_ARM_GRASP_INIT_DEG]
        self._right_grasp_joint_ids = [robot.find_joints([jname])[0][0] for jname in RIGHT_ARM_GRASP_INIT_DEG]
        self._left_grasp_joint_targets = torch.tensor(
            [math.radians(deg) for deg in LEFT_ARM_GRASP_INIT_DEG.values()],
            device=self.device,
            dtype=torch.float32,
        )
        self._right_grasp_joint_targets = torch.tensor(
            [math.radians(deg) for deg in RIGHT_ARM_GRASP_INIT_DEG.values()],
            device=self.device,
            dtype=torch.float32,
        )
        self._gripper_close_targets = torch.full(
            (len(self._gripper_joint_ids),),
            math.radians(GRIPPER_CLOSE_DEG),
            device=self.device,
            dtype=torch.float32,
        )

        # NOTE: 相机传感器已从默认场景配置移除。环境不再持有相机句柄。

        # 门铰链（通常只有 1 个关节）
        self._door_hinge_ids, _ = door.find_joints(".*")
        self._door_panel_body_idx = door.find_bodies([DOOR_LEAF_BODY_NAME])[0][0]

        # ── Per-env 持久状态 tensor ──────────────────────────────
        N = self.num_envs
        dev = self.device

        self._prev_joint_target = torch.zeros(N, 12, device=dev)
        self._prev_arm_joint_pos = torch.zeros(N, 12, device=dev)
        self._cached_joint_target_delta = torch.zeros(N, 12, device=dev)
        self._prev_base_cmd = torch.zeros(N, 3, device=dev)
        self._left_occupied = torch.zeros(N, dtype=torch.bool, device=dev)
        self._right_occupied = torch.zeros(N, dtype=torch.bool, device=dev)
        self._step_count = torch.zeros(N, dtype=torch.long, device=dev)
        self._prev_door_angle = torch.zeros(N, device=dev)
        self._already_succeeded = torch.zeros(N, dtype=torch.bool, device=dev)

        # 域随机化参数 tensors
        self._cup_mass = torch.zeros(N, device=dev)
        self._door_mass = torch.zeros(N, device=dev)
        self._door_damping = torch.zeros(N, device=dev)
        self._base_pos = torch.zeros(N, 3, device=dev)
        self._base_yaw = torch.zeros(N, device=dev)

        # 上一帧的 EE 世界系线速度/角速度，用于数值差分计算加速度
        self._prev_left_ee_lin_vel_w = torch.zeros(N, 3, device=dev)
        self._prev_left_ee_ang_vel_w = torch.zeros(N, 3, device=dev)
        self._prev_right_ee_lin_vel_w = torch.zeros(N, 3, device=dev)
        self._prev_right_ee_ang_vel_w = torch.zeros(N, 3, device=dev)

        self._episode_reset_fn = None

        nan = float("nan")
        self._pending_cup_mass = torch.full((N,), nan, device=dev)
        self._pending_door_mass = torch.full((N,), nan, device=dev)
        self._pending_door_damping = torch.full((N,), nan, device=dev)
        self._pending_base_pos = torch.full((N, 3), nan, device=dev)
        self._pending_base_yaw = torch.full((N,), nan, device=dev)

        # 预计算控制 dt
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

        # ── 奖励计算用缓存（由 _get_observations 填充）───────────
        self._cached_left_ee_la = torch.zeros(N, 3, device=dev)
        self._cached_left_ee_aa = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_la = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_aa = torch.zeros(N, 3, device=dev)
        self._cached_left_ee_lin_vel_w = torch.zeros(N, 3, device=dev)
        self._cached_left_ee_ang_vel_w = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_lin_vel_w = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_ang_vel_w = torch.zeros(N, 3, device=dev)
        self._cached_left_tilt_perp = torch.zeros(N, 2, device=dev)
        self._cached_right_tilt_perp = torch.zeros(N, 2, device=dev)
        # L12: 每步在 _get_observations() 中缓存，供 _get_rewards/_get_dones 共用
        self._cached_cup_dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._cached_left_ee_pos_base = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_pos_base = torch.zeros(N, 3, device=dev)
        self._cached_door_panel_face_center_base = torch.zeros(N, 3, device=dev)
        self._cached_door_panel_face_rot_base = torch.eye(3, device=dev).unsqueeze(0).repeat(N, 1, 1)
        self._cached_doorway_corners_world = torch.zeros(N, 4, 3, device=dev)
        self._cached_doorway_corners_base = torch.zeros(N, 4, 3, device=dev)
        self._cached_base_signed_doorway_distance = torch.zeros(N, device=dev)
        self._prev_base_signed_doorway_distance = torch.full((N,), float("nan"), device=dev)
        self._initial_base_signed_doorway_distance = torch.full((N,), float("nan"), device=dev)
        self._cached_base_in_doorway_opening = torch.zeros(N, dtype=torch.bool, device=dev)
        self._base_link_crossed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._cached_episode_success = torch.zeros(N, dtype=torch.bool, device=dev)
        self._cached_approach_dist = torch.zeros(N, device=dev)
        self._cached_left_approach_dist = torch.zeros(N, device=dev)
        self._cached_right_approach_dist = torch.zeros(N, device=dev)
        self._cached_base_cmd_delta = torch.zeros(N, 3, device=dev)
        self._cached_base_force_cmd = torch.zeros(N, 3, device=dev)
        self._cached_base_torque_cmd = torch.zeros(N, 3, device=dev)
        self._cached_planar_joint_targets = torch.zeros(N, len(self._planar_joint_ids), device=dev)
        self._cached_wheel_saturation_ratio = torch.zeros(N, device=dev)
        self._initial_approach_dist = torch.full((N,), float("nan"), device=dev)
        self._initial_left_approach_dist = torch.full((N,), float("nan"), device=dev)
        self._initial_right_approach_dist = torch.full((N,), float("nan"), device=dev)
        self._episode_reward_sums = {
            key: torch.zeros(N, device=dev) for key in _EPISODE_REWARD_KEYS
        }
        self._episode_state_sums = {
            key: torch.zeros(N, device=dev) for key in _EPISODE_STATE_SUM_KEYS
        }
        self._episode_state_max = {
            key: torch.zeros(N, device=dev) for key in _EPISODE_STATE_MAX_KEYS
        }
        self._door_center_offset_local = torch.tensor(_DOOR_CENTER_OFFSET_LOCAL, device=dev, dtype=torch.float32)
        self._door_panel_face_center_offset_local = torch.tensor(
            _DOOR_PANEL_FACE_CENTER_OFFSET_LOCAL,
            device=dev,
            dtype=torch.float32,
        )
        self._door_normal_local = torch.tensor(_DOOR_NORMAL_LOCAL, device=dev, dtype=torch.float32)
        self._doorway_inner_corners_local = DOORWAY_INNER_CORNERS_LOCAL.to(device=dev)
        self._doorway_lower_edge_start_local = DOORWAY_LOWER_EDGE_START_LOCAL.to(device=dev)
        self._doorway_lower_edge_end_local = DOORWAY_LOWER_EDGE_END_LOCAL.to(device=dev)
        self._doorway_plane_normal_local = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=torch.float32)
        self._left_cup_relative_xyz = torch.tensor(LEFT_CUP_RELATIVE_XYZ, device=dev, dtype=torch.float32)
        self._right_cup_relative_xyz = torch.tensor(RIGHT_CUP_RELATIVE_XYZ, device=dev, dtype=torch.float32)
        self._initialize_reward_stage_schedule()

    def _initialize_reward_stage_schedule(self) -> None:
        schedule = getattr(self.cfg, "rew_stage_schedule", None)
        stages = schedule.get("stages", []) if isinstance(schedule, dict) else []
        self._reward_stage_schedule = schedule if isinstance(schedule, dict) else {"enabled": False, "stages": []}
        self._reward_stage_active = bool(self._reward_stage_schedule.get("enabled", False) and stages)
        self._reward_stage_index = -1
        self._reward_stage_name = "default"
        staged_attrs = {
            attr
            for stage in stages
            if isinstance(stage, dict)
            for attr in (stage.get("overrides", {}) or {})
            if hasattr(self.cfg, attr)
        }
        self._reward_stage_base_values = {attr: getattr(self.cfg, attr) for attr in staged_attrs}
        if self._reward_stage_active:
            self._apply_reward_stage_schedule(force=True)

    def _select_reward_stage_index(self) -> int:
        stages = self._reward_stage_schedule.get("stages", [])
        if not self._reward_stage_active or not stages:
            return -1

        step_counter = getattr(self, "common_step_counter", 0)
        if isinstance(step_counter, torch.Tensor):
            step_counter = int(step_counter.item())
        global_frames = int(step_counter) * int(self.num_envs)
        for index, stage in enumerate(stages):
            until_frames = stage.get("until_frames")
            if until_frames is None or global_frames < int(until_frames):
                return index
        return len(stages) - 1

    def _apply_reward_stage_schedule(self, *, force: bool = False) -> None:
        if not self._reward_stage_active:
            return

        stage_index = self._select_reward_stage_index()
        if stage_index == self._reward_stage_index and not force:
            return

        stages = self._reward_stage_schedule.get("stages", [])
        stage = stages[stage_index]
        for attr, value in self._reward_stage_base_values.items():
            setattr(self.cfg, attr, value)
        for attr, value in (stage.get("overrides", {}) or {}).items():
            if hasattr(self.cfg, attr):
                setattr(self.cfg, attr, float(value))

        self._reward_stage_index = int(stage_index)
        self._reward_stage_name = str(stage.get("name", f"stage_{stage_index}"))

    def _initialize_base_controller_backend(self) -> None:
        """Configure the selected mobile-base backend."""
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
        except Exception as exc:  # pragma: no cover - exercised only inside live Isaac Sim sessions.
            self._holonomic_wheel_target_matrix = None
            self._base_controller_backend = "analytic_mecanum_fallback"
            logger.warning("Falling back to analytic mecanum wheel mapping: %s", exc)

    def _configure_planar_joint_velocity_backend(self) -> None:
        """Disable wheel actuation so the mobile base is driven only by planar joints."""
        robot: Articulation = self.scene["robot"]
        if len(self._planar_joint_ids) != len(PLANAR_BASE_JOINT_NAMES):
            raise RuntimeError(
                "Planar base backend requires planar joints "
                f"{PLANAR_BASE_JOINT_NAMES}, but only found {len(self._planar_joint_ids)}."
            )
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_effort_limit_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device),
            joint_ids=self._wheel_joint_ids,
        )

    def _configure_root_force_torque_backend(self) -> None:
        """Cache the driven body properties and disable wheel actuation feedback."""
        robot: Articulation = self.scene["robot"]
        force_body_name = self.cfg.base_force_body_name
        if force_body_name not in robot.body_names:
            logger.warning(
                "Requested base_force_body_name '%s' was not found; falling back to '%s'.",
                force_body_name,
                BASE_LINK_NAME,
            )
            force_body_name = BASE_LINK_NAME
        self._base_force_body_name = force_body_name
        self._base_force_body_idx = robot.find_bodies([force_body_name])[0][0]
        self._base_force_body_mass = robot.data.default_mass[:, self._base_force_body_idx].clone()
        self._base_force_body_inertia_zz = robot.data.default_inertia[:, self._base_force_body_idx, 8].clone()

        # Root-wrench control should not compete with implicit wheel velocity actuators.
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.write_joint_effort_limit_to_sim(0.0, joint_ids=self._wheel_joint_ids)
        robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device),
            joint_ids=self._wheel_joint_ids,
        )

    def _build_holonomic_wheel_target_matrix_from_usd(self) -> Tensor:
        """Read wheel metadata from the cloned robot USD and derive a batched wheel-target matrix."""
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
        try:
            ordered_rows = [name_to_row[name] for name in WHEEL_JOINT_NAMES]
        except KeyError as exc:
            raise RuntimeError(
                f"Holonomic metadata is missing wheel joint '{exc.args[0]}' in {robot_prim_path}."
            ) from exc
        return matrix[ordered_rows].to(device=self.device)

    def _compute_wheel_targets(self, base_cmd: Tensor) -> Tensor:
        """Project base twists to wheel velocity targets using the selected backend."""
        if self._holonomic_wheel_target_matrix is not None:
            return project_base_twist_to_wheel_targets(
                base_cmd,
                wheel_target_matrix=self._holonomic_wheel_target_matrix,
            )
        return twist_to_wheel_angular_velocity_targets(
            base_cmd,
            wheel_radius=self.cfg.wheel_radius,
            half_length=self.cfg.wheel_base_half_length,
            half_width=self.cfg.wheel_base_half_width,
        )

    def _compute_base_force_torque_targets(self, base_cmd: Tensor) -> tuple[Tensor, Tensor]:
        """Track ``[vx, vy, wz]`` using local-frame wrench commands on the chosen base body."""
        robot: Articulation = self.scene["robot"]
        body_quat = robot.data.body_quat_w[:, self._base_force_body_idx]
        base_lin_vel_base = batch_vector_world_to_base(
            robot.data.body_lin_vel_w[:, self._base_force_body_idx],
            body_quat,
        )
        base_ang_vel_base = batch_vector_world_to_base(
            robot.data.body_ang_vel_w[:, self._base_force_body_idx],
            body_quat,
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
        """Track body-frame ``[vx, vy, wz]`` with world-frame planar joint velocities."""
        robot: Articulation = self.scene["robot"]
        base_yaw = batch_yaw_from_quat(robot.data.body_quat_w[:, self._base_body_idx])
        return project_body_twist_to_planar_joint_targets(base_cmd, base_yaw=base_yaw)


    # ═══════════════════════════════════════════════════════════════════
    # 场景装配（由 Cloner 调用）
    # ═══════════════════════════════════════════════════════════════════

    def _setup_scene(self) -> None:
        """场景实体已由 ``DoorPushSceneCfg`` 自动注册，此处无需额外装配。"""
        return

    @staticmethod
    def _compute_point_to_panel_face_distance(
        *,
        points_base: Tensor,
        face_center_base: Tensor,
        face_rot_base: Tensor,
        half_extent_y: float,
        half_extent_z: float,
    ) -> Tensor:
        """Thin wrapper around the pure reward-geometry helper."""
        return compute_point_to_panel_face_distance(
            points_base=points_base,
            face_center_base=face_center_base,
            face_rot_base=face_rot_base,
            half_extent_y=half_extent_y,
            half_extent_z=half_extent_z,
        )

    @staticmethod
    def _compute_normalized_approach_score(
        *,
        current_dist: Tensor,
        initial_dist: Tensor,
        eps: float,
    ) -> Tensor:
        """Thin wrapper around the normalized approach helper."""
        return compute_normalized_approach_score(
            current_dist=current_dist,
            initial_dist=initial_dist,
            eps=eps,
        )

    def _compute_doorway_state(
        self,
        *,
        base_pos_w: Tensor,
        base_quat_w: Tensor,
        door_root_pos_w: Tensor,
        door_root_quat_w: Tensor,
    ) -> dict[str, Tensor]:
        """Compute doorway geometry in world, door-root, and base frames."""
        doorway_corners_world = transform_doorway_points_to_world(
            self._doorway_inner_corners_local.unsqueeze(0).expand(base_pos_w.shape[0], -1, -1),
            door_root_pos_w,
            door_root_quat_w,
        )
        doorway_corners_base = transform_doorway_points_to_base(
            points_world=doorway_corners_world,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
        )
        doorway_lower_start_w = doorway_corners_world[:, 0]
        doorway_lower_end_w = doorway_corners_world[:, 1]

        base_pos_ground_w = base_pos_w.clone()
        base_pos_ground_w[:, 2] = 0.0
        base_line_dist = compute_point_to_segment_distance(
            points=base_pos_ground_w,
            seg_start=doorway_lower_start_w,
            seg_end=doorway_lower_end_w,
        )

        doorway_plane_normal_w = torch.bmm(
            batch_quat_to_rotation_matrix(door_root_quat_w),
            self._doorway_plane_normal_local.view(1, 3, 1).expand(base_pos_w.shape[0], -1, -1),
        ).squeeze(-1)
        doorway_tangent_w = doorway_lower_end_w - doorway_lower_start_w
        base_rot_w = batch_quat_to_rotation_matrix(base_quat_w)
        base_forward_world = base_rot_w[:, :, 0]
        base_pos_door = batch_vector_world_to_base(base_pos_w - door_root_pos_w, door_root_quat_w)
        base_forward_door = batch_vector_world_to_base(base_forward_world, door_root_quat_w)
        base_yaw_door = torch.atan2(base_forward_door[:, 1], base_forward_door[:, 0])
        base_corners_door = compute_base_footprint_corners_door_frame(
            base_pos_door_xy=base_pos_door[:, :2],
            base_yaw_door=base_yaw_door,
            half_length=self.cfg.wheel_base_half_length,
            half_width=self.cfg.wheel_base_half_width,
        )
        corridor_half_width = max(
            abs(float(self._doorway_lower_edge_start_local[1])),
            abs(float(self._doorway_lower_edge_end_local[1])),
        )
        base_corridor_excess = compute_base_corridor_excess(
            corner_y=base_corners_door[:, :, 1],
            corridor_half_width=corridor_half_width,
        )
        base_corridor_gate = base_corridor_excess <= 1.0e-6
        base_signed_distance = compute_signed_distance_to_plane(
            points=base_pos_w,
            plane_points=door_root_pos_w,
            plane_normals=doorway_plane_normal_w,
        )
        base_in_opening = (
            (base_pos_door[:, 1] >= float(self._doorway_lower_edge_start_local[1]))
            & (base_pos_door[:, 1] <= float(self._doorway_lower_edge_end_local[1]))
        )

        return {
            "doorway_corners_world": doorway_corners_world,
            "doorway_corners_base": doorway_corners_base,
            "doorway_lower_start_w": doorway_lower_start_w,
            "doorway_lower_end_w": doorway_lower_end_w,
            "doorway_plane_normal_w": doorway_plane_normal_w,
            "doorway_tangent_w": doorway_tangent_w,
            "base_pos_door": base_pos_door,
            "base_forward_world": base_forward_world,
            "base_corners_door": base_corners_door,
            "base_corridor_excess": base_corridor_excess,
            "base_corridor_gate": base_corridor_gate,
            "base_line_dist": base_line_dist,
            "base_signed_distance": base_signed_distance,
            "base_in_opening": base_in_opening,
        }

    def _set_gripper_hold_targets(self, env_ids: Tensor | None = None) -> None:
        """刷新夹爪保持目标。

        持杯侧保持在预设抓持角，非持杯侧保持完全闭合，避免 episode 开始后
        gripper 因无持续控制而被接触力顶开。
        """
        if env_ids is not None and env_ids.numel() == 0:
            return

        robot: Articulation = self.scene["robot"]
        left_occupied = self._left_occupied if env_ids is None else self._left_occupied[env_ids]
        right_occupied = self._right_occupied if env_ids is None else self._right_occupied[env_ids]
        hold_targets = build_gripper_hold_targets(
            left_occupied=left_occupied,
            right_occupied=right_occupied,
            occupied_deg=GRIPPER_CLOSE_DEG,
            unoccupied_deg=GRIPPER_FULLY_CLOSED_DEG,
        )
        robot.set_joint_position_target(
            hold_targets,
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )
        robot.set_joint_velocity_target(
            torch.zeros_like(hold_targets),
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 动作执行
    # ═══════════════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: Tensor) -> None:
        """将策略输出的 arm/base 命令写入仿真 — 批量操作所有 env。

        控制链路：
            arm: a_norm[-1,1] → joint limits → (optional) noise → position target
            base:
                - ``planar_joint_velocity``: a_norm[-1,1] → [vx, vy, wz] → world-frame planar joint velocities
                - ``root_force_torque``: a_norm[-1,1] → [vx, vy, wz] → local wrench
                - wheel backends: a_norm[-1,1] → [vx, vy, wz] → wheel velocity targets
        """
        robot: Articulation = self.scene["robot"]
        arm_actions = actions[:, : len(self._arm_joint_ids)]
        base_actions = actions[:, len(self._arm_joint_ids) :]

        # 1. 将 arm 归一化动作映射到 joint limits
        joint_limits = robot.data.soft_joint_pos_limits[0, self._arm_joint_ids]  # (12, 2)
        q_min = joint_limits[:, 0]  # (12,)
        q_max = joint_limits[:, 1]  # (12,)
        q_target_cmd = rescale_normalized_joint_actions(arm_actions, q_min, q_max)

        # 2. 可选：注入位置目标噪声
        noise_std = self.cfg.position_target_noise_std
        if noise_std > 0:
            noise = torch.randn_like(q_target_cmd) * noise_std
            q_target_cmd = torch.clamp(
                q_target_cmd + noise,
                q_min.unsqueeze(0),
                q_max.unsqueeze(0),
            )

        # 3. 将底盘动作映射为驱动命令，并记录命令变化
        base_cmd = rescale_normalized_base_actions(
            base_actions,
            max_lin_vel_x=self.cfg.base_max_lin_vel_x,
            max_lin_vel_y=self.cfg.base_max_lin_vel_y,
            max_ang_vel_z=self.cfg.base_max_ang_vel_z,
        )
        self._cached_base_cmd_delta = base_cmd - self._prev_base_cmd
        if self._base_controller_backend == "planar_joint_velocity":
            planar_joint_targets = self._compute_planar_joint_targets(base_cmd)
            robot.set_joint_velocity_target(
                planar_joint_targets,
                joint_ids=self._planar_joint_ids,
            )
            if not self._training_planar_base_only:
                robot.set_joint_velocity_target(
                    torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device),
                    joint_ids=self._wheel_joint_ids,
                )
            self._cached_planar_joint_targets = planar_joint_targets
            self._cached_base_force_cmd.zero_()
            self._cached_base_torque_cmd.zero_()
            self._cached_wheel_saturation_ratio.zero_()
        elif self._base_controller_backend == "root_force_torque":
            force_cmd, torque_cmd = self._compute_base_force_torque_targets(base_cmd)
            wheel_targets = torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device)
            self._cached_base_force_cmd = force_cmd
            self._cached_base_torque_cmd = torque_cmd
            self._cached_planar_joint_targets.zero_()
            self._cached_wheel_saturation_ratio.zero_()
        else:
            wheel_targets = self._compute_wheel_targets(base_cmd)
            wheel_targets, saturation_ratio = clip_wheel_velocity_targets(
                wheel_targets,
                velocity_limit=self.cfg.wheel_velocity_limit,
            )
            self._cached_base_force_cmd.zero_()
            self._cached_base_torque_cmd.zero_()
            self._cached_planar_joint_targets.zero_()
            self._cached_wheel_saturation_ratio = saturation_ratio

        robot.set_joint_position_target(
            q_target_cmd,
            joint_ids=self._arm_joint_ids,
        )
        if self._base_controller_backend not in ("root_force_torque", "planar_joint_velocity"):
            robot.set_joint_velocity_target(
                wheel_targets,
                joint_ids=self._wheel_joint_ids,
            )

        # 4. 保持 gripper hold targets
        self._set_gripper_hold_targets()

        # 5. 缓存最终目标，供下一步 obs 和边界惩罚使用
        self._cached_joint_target_delta = q_target_cmd - self._prev_joint_target
        self._prev_joint_target = q_target_cmd.clone()
        self._prev_base_cmd = base_cmd.clone()

    def _apply_action(self) -> None:
        """Write per-substep actions into articulation buffers before each simulator step."""
        if self._base_controller_backend == "root_force_torque":
            robot: Articulation = self.scene["robot"]
            robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=self._cached_base_force_cmd.unsqueeze(1),
                torques=self._cached_base_torque_cmd.unsqueeze(1),
                body_ids=[self._base_force_body_idx],
                is_global=False,
            )
        return

    # ═══════════════════════════════════════════════════════════════════
    # 观测构建
    # ═══════════════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        """批量构建 Actor + Critic 观测 — 纯 tensor 操作。

        Returns
        -------
        dict
            ``{"policy": actor_obs (N, num_observations),
               "critic": critic_obs (N, num_states)}``
        """
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]

        # ── 读取关节状态 ──────────────────────────────────────────
        all_q = robot.data.joint_pos[:, self._arm_joint_ids]    # (N, 12)
        all_dq = robot.data.joint_vel[:, self._arm_joint_ids]   # (N, 12)

        # ── 读取 body 状态（世界系）─────────────────────────────
        body_pos_w = robot.data.body_pos_w     # (N, B, 3)
        body_quat_w = robot.data.body_quat_w   # (N, B, 4)
        body_lin_vel_w = robot.data.body_lin_vel_w
        body_ang_vel_w = robot.data.body_ang_vel_w

        base_pos = body_pos_w[:, self._base_body_idx]     # (N, 3)
        base_quat = body_quat_w[:, self._base_body_idx]   # (N, 4)
        base_lv = body_lin_vel_w[:, self._base_body_idx]   # (N, 3)
        base_av = body_ang_vel_w[:, self._base_body_idx]   # (N, 3)

        left_ee_pos_w = body_pos_w[:, self._left_ee_body_idx]
        left_ee_quat_w = body_quat_w[:, self._left_ee_body_idx]
        left_ee_lv_w = body_lin_vel_w[:, self._left_ee_body_idx]
        left_ee_av_w = body_ang_vel_w[:, self._left_ee_body_idx]
        right_ee_pos_w = body_pos_w[:, self._right_ee_body_idx]
        right_ee_quat_w = body_quat_w[:, self._right_ee_body_idx]
        right_ee_lv_w = body_lin_vel_w[:, self._right_ee_body_idx]
        right_ee_av_w = body_ang_vel_w[:, self._right_ee_body_idx]

        # ── 左/右臂 EE（base_link 相对系）──────────────────────
        left_ee_pos_base, left_ee_quat_base, left_ee_lv_base, left_ee_av_base = self._ee_world_to_base(
            left_ee_pos_w,
            left_ee_quat_w,
            left_ee_lv_w,
            left_ee_av_w,
            base_pos,
            base_quat,
            base_lv,
            base_av,
        )
        right_ee_pos_base, right_ee_quat_base, right_ee_lv_base, right_ee_av_base = self._ee_world_to_base(
            right_ee_pos_w,
            right_ee_quat_w,
            right_ee_lv_w,
            right_ee_av_w,
            base_pos,
            base_quat,
            base_lv,
            base_av,
        )

        # ── 数值微分计算加速度（对世界系速度差分，再转到当前 base 系）──
        inv_dt = 1.0 / max(self._control_dt, 1e-6)
        left_ee_la = batch_vector_world_to_base(
            (left_ee_lv_w - self._prev_left_ee_lin_vel_w) * inv_dt,
            base_quat,
        )
        left_ee_aa = batch_vector_world_to_base(
            (left_ee_av_w - self._prev_left_ee_ang_vel_w) * inv_dt,
            base_quat,
        )
        right_ee_la = batch_vector_world_to_base(
            (right_ee_lv_w - self._prev_right_ee_lin_vel_w) * inv_dt,
            base_quat,
        )
        right_ee_aa = batch_vector_world_to_base(
            (right_ee_av_w - self._prev_right_ee_ang_vel_w) * inv_dt,
            base_quat,
        )

        self._cached_left_ee_lin_vel_w = left_ee_lv_w.clone()
        self._cached_left_ee_ang_vel_w = left_ee_av_w.clone()
        self._cached_right_ee_lin_vel_w = right_ee_lv_w.clone()
        self._cached_right_ee_ang_vel_w = right_ee_av_w.clone()

        # 更新速度缓存
        self._prev_left_ee_lin_vel_w = left_ee_lv_w.clone()
        self._prev_left_ee_ang_vel_w = left_ee_av_w.clone()
        self._prev_right_ee_lin_vel_w = right_ee_lv_w.clone()
        self._prev_right_ee_ang_vel_w = right_ee_av_w.clone()

        # ── 稳定性 proxy: tilt ──────────────────────────────────
        left_tilt, left_tilt_perp = self._compute_tilt(left_ee_quat_w)   # (N, 1), (N, 2)
        right_tilt, right_tilt_perp = self._compute_tilt(right_ee_quat_w)  # (N, 1), (N, 2)

        # ── 缓存 EE 位姿/加速度和 tilt_perp，供 _get_rewards 使用 ──
        self._cached_left_ee_pos_base = left_ee_pos_base.clone()
        self._cached_right_ee_pos_base = right_ee_pos_base.clone()
        self._cached_left_ee_la = left_ee_la.clone()
        self._cached_left_ee_aa = left_ee_aa.clone()
        self._cached_right_ee_la = right_ee_la.clone()
        self._cached_right_ee_aa = right_ee_aa.clone()
        self._cached_left_tilt_perp = left_tilt_perp.clone()
        self._cached_right_tilt_perp = right_tilt_perp.clone()

        # ── 上下文 ──────────────────────────────────────────────
        left_occ = self._left_occupied.float().unsqueeze(-1)   # (N, 1)
        right_occ = self._right_occupied.float().unsqueeze(-1)  # (N, 1)
        base_lin_vel_base = batch_vector_world_to_base(base_lv, base_quat)
        base_ang_vel_base = batch_vector_world_to_base(base_av, base_quat)
        base_twist_cmd_obs = torch.cat(
            [
                base_lin_vel_base[:, :2],
                base_ang_vel_base[:, 2:3],
                self._prev_base_cmd,
            ],
            dim=-1,
        )  # (N, 6)

        # ── 门几何观测（base_link 系）─────────────────────────────
        door_root_pos_w = door.data.root_pos_w    # (N, 3)
        door_root_quat_w = door.data.root_quat_w  # (N, 4)
        door_leaf_pos_w = door.data.body_pos_w[:, self._door_panel_body_idx]   # (N, 3)
        door_leaf_quat_w = door.data.body_quat_w[:, self._door_panel_body_idx]  # (N, 4)

        R_world_from_leaf = batch_quat_to_rotation_matrix(door_leaf_quat_w)  # (N, 3, 3)
        door_center_w = door_leaf_pos_w + torch.bmm(
            R_world_from_leaf,
            self._door_center_offset_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)  # (N, 3)
        door_panel_face_center_w = door_leaf_pos_w + torch.bmm(
            R_world_from_leaf,
            self._door_panel_face_center_offset_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)  # (N, 3)
        door_normal_w = torch.bmm(
            R_world_from_leaf,
            self._door_normal_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)  # (N, 3)

        door_center_base = batch_vector_world_to_base(
            door_center_w - base_pos, base_quat
        )  # (N, 3)
        door_panel_face_center_base = batch_vector_world_to_base(
            door_panel_face_center_w - base_pos, base_quat
        )  # (N, 3)
        door_normal_base = batch_vector_world_to_base(
            door_normal_w, base_quat
        )  # (N, 3)
        door_panel_face_quat_base = batch_orientation_world_to_base(
            door_leaf_quat_w, base_quat
        )
        door_panel_face_rot_base = batch_quat_to_rotation_matrix(
            door_panel_face_quat_base
        )
        self._cached_door_panel_face_center_base = door_panel_face_center_base.clone()
        self._cached_door_panel_face_rot_base = door_panel_face_rot_base.clone()

        doorway_state = self._compute_doorway_state(
            base_pos_w=base_pos,
            base_quat_w=base_quat,
            door_root_pos_w=door_root_pos_w,
            door_root_quat_w=door_root_quat_w,
        )
        doorway_corners_base = doorway_state["doorway_corners_base"]
        self._cached_doorway_corners_world = doorway_state["doorway_corners_world"].clone()
        self._cached_doorway_corners_base = doorway_corners_base.clone()
        self._cached_base_signed_doorway_distance = doorway_state["base_signed_distance"].clone()
        self._cached_base_in_doorway_opening = doorway_state["base_in_opening"].clone()
        base_dist_uninitialized = torch.isnan(self._initial_base_signed_doorway_distance)
        self._initial_base_signed_doorway_distance = torch.where(
            base_dist_uninitialized,
            doorway_state["base_signed_distance"],
            self._initial_base_signed_doorway_distance,
        )
        self._prev_base_signed_doorway_distance = torch.where(
            torch.isnan(self._prev_base_signed_doorway_distance),
            doorway_state["base_signed_distance"],
            self._prev_base_signed_doorway_distance,
        )
        left_approach_dist = self._compute_point_to_panel_face_distance(
            points_base=left_ee_pos_base,
            face_center_base=door_panel_face_center_base,
            face_rot_base=door_panel_face_rot_base,
            half_extent_y=_DOOR_PANEL_HALF_EXTENT_Y,
            half_extent_z=_DOOR_PANEL_HALF_EXTENT_Z,
        )
        right_approach_dist = self._compute_point_to_panel_face_distance(
            points_base=right_ee_pos_base,
            face_center_base=door_panel_face_center_base,
            face_rot_base=door_panel_face_rot_base,
            half_extent_y=_DOOR_PANEL_HALF_EXTENT_Y,
            half_extent_z=_DOOR_PANEL_HALF_EXTENT_Z,
        )
        self._cached_left_approach_dist = left_approach_dist.clone()
        self._cached_right_approach_dist = right_approach_dist.clone()
        current_approach_dist = torch.minimum(left_approach_dist, right_approach_dist)
        self._cached_approach_dist = current_approach_dist.clone()
        uninitialized = torch.isnan(self._initial_approach_dist)
        self._initial_approach_dist = torch.where(
            uninitialized,
            current_approach_dist,
            self._initial_approach_dist,
        )
        left_uninitialized = torch.isnan(self._initial_left_approach_dist)
        self._initial_left_approach_dist = torch.where(
            left_uninitialized,
            left_approach_dist,
            self._initial_left_approach_dist,
        )
        right_uninitialized = torch.isnan(self._initial_right_approach_dist)
        self._initial_right_approach_dist = torch.where(
            right_uninitialized,
            right_approach_dist,
            self._initial_right_approach_dist,
        )

        door_geometry = torch.cat([door_center_base, door_normal_base], dim=-1)  # (N, 6)
        doorway_frame_obs = doorway_corners_base.reshape(self.num_envs, -1)  # (N, 12)

        # ── Actor obs (含噪声) ──────────────────────────────────
        noisy_q = all_q
        noisy_dq = all_dq
        if self.cfg.obs_noise_std > 0:
            noisy_q = all_q + torch.randn_like(all_q) * self.cfg.obs_noise_std
            noisy_dq = all_dq + torch.randn_like(all_dq) * self.cfg.obs_noise_std

        # actor: proprio(36) + ee(38) + context(2) + stability(2) + door_geometry(6)
        #        + doorway_frame(12) + base_twist/cmd(6) = 102
        actor_obs = torch.cat([
            # proprio: q(12) + dq(12) + prev_joint_target(12) = 36
            noisy_q, noisy_dq, self._prev_joint_target,
            # left_ee: pos(3) + quat(4) + lv(3) + av(3) + la(3) + aa(3) = 19
            left_ee_pos_base, left_ee_quat_base,
            left_ee_lv_base, left_ee_av_base,
            left_ee_la, left_ee_aa,
            # right_ee: 同上 = 19
            right_ee_pos_base, right_ee_quat_base,
            right_ee_lv_base, right_ee_av_base,
            right_ee_la, right_ee_aa,
            # context: 2
            left_occ, right_occ,
            # stability: 2
            left_tilt, right_tilt,
            # door_geometry: center(3) + normal(3) = 6
            door_geometry,
            # doorway_frame: 4 inner-frame corners x xyz = 12
            doorway_frame_obs,
            # base_twist/cmd: v_xy(2) + wz(1) + prev_base_cmd(3) = 6
            base_twist_cmd_obs,
        ], dim=-1)  # (N, 102)

        # ── Critic obs (无噪声 + privileged) ───────────────────
        # 门状态 (base_link 系)
        door_pose_base = batch_pose_world_to_base(
            door_root_pos_w, door_root_quat_w, base_pos, base_quat,
        )  # (N, 7)
        door_joint_pos = door.data.joint_pos[:, 0:1]  # (N, 1)
        door_joint_vel = door.data.joint_vel[:, 0:1]  # (N, 1)

        # 杯体掉落检测 (用于 critic privileged info)；同时缓存供本步后续方法复用（L12）
        cup_dropped = self._check_cup_dropped()  # (N,)
        self._cached_cup_dropped = cup_dropped

        critic_obs = torch.cat([
            # 无噪声 proprio
            all_q, all_dq, self._prev_joint_target,
            # EE（与 actor 相同，但无噪声 q/dq）
            left_ee_pos_base, left_ee_quat_base,
            left_ee_lv_base, left_ee_av_base,
            left_ee_la, left_ee_aa,
            right_ee_pos_base, right_ee_quat_base,
            right_ee_lv_base, right_ee_av_base,
            right_ee_la, right_ee_aa,
            # context + stability
            left_occ, right_occ, left_tilt, right_tilt,
            # door_geometry（与 actor 相同）
            door_geometry,
            doorway_frame_obs,
            # base_twist/cmd（与 actor 相同）
            base_twist_cmd_obs,
            # privileged: door_pose(7) + door_joint(2) + domain_params(3) + cup_dropped(1) = 13
            door_pose_base,
            door_joint_pos, door_joint_vel,
            self._cup_mass.unsqueeze(-1),
            self._door_mass.unsqueeze(-1),
            self._door_damping.unsqueeze(-1),
            cup_dropped.float().unsqueeze(-1),
        ], dim=-1)  # (N, 115)

        return {"policy": actor_obs, "critic": critic_obs}

    # ═══════════════════════════════════════════════════════════════════
    # 奖励计算
    # ═══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> Tensor:
        """批量计算完整奖励，并在回合完成时输出聚合后的 TensorBoard 分项信息。"""
        self._apply_reward_stage_schedule()

        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        theta = door.data.joint_pos[:, 0]   # (N,)
        theta_prev = self._prev_door_angle

        reward_info: dict[str, Tensor] = {}

        # ══════════════════════════════════════════════════════════════
        # §4 任务奖励：角度增量 + 一次性成功 bonus
        # ══════════════════════════════════════════════════════════════
        delta = theta - theta_prev

        target = self.cfg.door_angle_target
        w_below = torch.full_like(theta, self.cfg.rew_w_delta)
        w_above = self.cfg.rew_w_delta * torch.clamp(
            1.0 - self.cfg.rew_k_decay * (theta - target),
            min=self.cfg.rew_alpha,
        )
        weight = torch.where(theta <= target, w_below, w_above)
        r_task_delta = weight * delta

        newly_succeeded = (theta >= target) & ~self._already_succeeded
        r_task_open_bonus = newly_succeeded.float() * self.cfg.rew_w_open
        r_task_approach_raw = self._compute_normalized_approach_score(
            current_dist=self._cached_approach_dist,
            initial_dist=self._initial_approach_dist,
            eps=self.cfg.rew_approach_eps,
        )
        approach_active = (theta < self.cfg.rew_approach_stop_angle).float()
        r_task_approach = (
            approach_active
            * self.cfg.rew_w_approach
            * r_task_approach_raw
        )
        left_approach_score = self._compute_normalized_approach_score(
            current_dist=self._cached_left_approach_dist,
            initial_dist=self._initial_left_approach_dist,
            eps=self.cfg.rew_approach_eps,
        )
        right_approach_score = self._compute_normalized_approach_score(
            current_dist=self._cached_right_approach_dist,
            initial_dist=self._initial_right_approach_dist,
            eps=self.cfg.rew_approach_eps,
        )
        left_hand_near_score = compute_hand_near_gate(
            hand_dist=self._cached_left_approach_dist,
            near_dist=self.cfg.rew_hand_near_dist,
            tau=self.cfg.rew_hand_near_tau,
        )
        right_hand_near_score = compute_hand_near_gate(
            hand_dist=self._cached_right_approach_dist,
            near_dist=self.cfg.rew_hand_near_dist,
            tau=self.cfg.rew_hand_near_tau,
        )
        hand_near_score = torch.maximum(left_hand_near_score, right_hand_near_score)
        self._already_succeeded = self._already_succeeded | newly_succeeded

        door_root_pos_w = door.data.root_pos_w
        door_root_quat_w = door.data.root_quat_w
        base_pos_w = robot.data.body_pos_w[:, self._base_body_idx]
        base_quat_w = robot.data.body_quat_w[:, self._base_body_idx]
        doorway_state = self._compute_doorway_state(
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            door_root_pos_w=door_root_pos_w,
            door_root_quat_w=door_root_quat_w,
        )
        current_base_line_dist = doorway_state["base_line_dist"]
        current_base_signed_distance = doorway_state["base_signed_distance"]
        prev_base_signed_distance = torch.where(
            torch.isnan(self._prev_base_signed_doorway_distance),
            current_base_signed_distance,
            self._prev_base_signed_doorway_distance,
        )

        base_align_score = compute_base_alignment_score(
            base_forward_world=doorway_state["base_forward_world"],
            doorway_normal_world=doorway_state["doorway_plane_normal_w"],
            mid_angle_deg=self.cfg.rew_base_align_mid_angle_deg,
            temperature_deg=self.cfg.rew_base_align_temperature_deg,
        )
        base_range_score = compute_base_range_score(
            corridor_excess=doorway_state["base_corridor_excess"],
            tau=self.cfg.rew_base_range_tau,
        )
        base_near_score = compute_near_line_score(
            base_line_dist=current_base_line_dist,
            sigma=self.cfg.rew_base_near_sigma,
        )
        r_task_base_align = compute_base_align_reward(
            align_score=base_align_score,
            range_score=base_range_score,
            near_score=base_near_score,
            weight=self.cfg.rew_w_base_align,
        )
        base_forward_delta = compute_forward_progress_delta(
            previous_signed_distance=prev_base_signed_distance,
            current_signed_distance=current_base_signed_distance,
        )
        base_signed_progress_delta = compute_signed_progress_delta(
            previous_signed_distance=prev_base_signed_distance,
            current_signed_distance=current_base_signed_distance,
        )
        r_task_base_forward = (
            self.cfg.rew_w_base_forward
            * base_align_score
            * base_range_score
            * base_forward_delta
        )
        r_task_base_net_progress = compute_base_net_progress_reward(
            align_score=base_align_score,
            range_score=base_range_score,
            signed_progress=base_signed_progress_delta,
            weight=self.cfg.rew_w_base_net_progress,
        )
        base_centerline_score = torch.exp(
            -0.5
            * torch.square(
                doorway_state["base_pos_door"][:, 1]
                / max(float(self.cfg.rew_base_centerline_sigma), 1.0e-6)
            )
        )
        r_task_base_centerline = (
            self.cfg.rew_w_base_centerline
            * base_align_score
            * base_range_score
            * base_centerline_score
        )
        base_push_gate = compute_door_angle_push_gate(
            door_angle=theta,
            start_angle=self.cfg.rew_base_push_start_angle,
            end_angle=self.cfg.rew_base_push_end_angle,
            start_tau=self.cfg.rew_base_push_start_tau,
            end_tau=self.cfg.rew_base_push_end_tau,
        )
        r_task_base_assist = compute_base_assist_reward(
            align_score=base_align_score,
            range_score=base_range_score,
            hand_score=hand_near_score,
            push_gate=base_push_gate,
            base_push_progress=base_forward_delta,
            weight=self.cfg.rew_w_base_assist,
        )
        r_task_base_door_sync = compute_base_door_sync_reward(
            align_score=base_align_score,
            range_score=base_range_score,
            door_delta=delta,
            base_push_progress=base_forward_delta,
            weight=self.cfg.rew_w_base_door_sync,
        )
        base_inside_progress = compute_inside_progress_delta(
            previous_signed_distance=prev_base_signed_distance,
            current_signed_distance=current_base_signed_distance,
            in_opening=torch.ones_like(doorway_state["base_in_opening"], dtype=torch.bool),
        )
        base_cross_gate = compute_door_angle_cross_gate(
            door_angle=theta,
            cross_angle=self.cfg.rew_base_cross_open_gate,
            tau=self.cfg.rew_base_cross_tau,
        )
        r_task_base_cross = (
            base_align_score
            * base_range_score
            * base_cross_gate
            * self.cfg.rew_w_base_cross
            * base_inside_progress
        )
        self._base_link_crossed = update_crossing_latch(
            previous_crossed=self._base_link_crossed,
            previous_signed_distance=prev_base_signed_distance,
            current_signed_distance=current_base_signed_distance,
            in_opening=doorway_state["base_corridor_gate"],
        )
        self._cached_base_signed_doorway_distance = current_base_signed_distance.clone()
        self._cached_base_in_doorway_opening = doorway_state["base_in_opening"].clone()
        self._prev_base_signed_doorway_distance = current_base_signed_distance.clone()

        cup_dropped = self._cached_cup_dropped
        success = compute_door_traverse_success(
            door_angle=theta,
            door_angle_target=self.cfg.door_angle_target,
            cup_dropped=cup_dropped,
            base_crossed=self._base_link_crossed,
        )
        self._cached_episode_success = success

        r_task = (
            r_task_delta
            + r_task_open_bonus
            + r_task_approach
            + r_task_base_align
            + r_task_base_forward
            + r_task_base_centerline
            + r_task_base_net_progress
            + r_task_base_assist
            + r_task_base_door_sync
            + r_task_base_cross
        )

        reward_info["task"] = r_task
        reward_info["task/delta"] = r_task_delta
        reward_info["task/open_bonus"] = r_task_open_bonus
        reward_info["task/approach"] = r_task_approach
        reward_info["task/approach_raw"] = r_task_approach_raw
        reward_info["task/base_align"] = r_task_base_align
        reward_info["task/base_forward"] = r_task_base_forward
        reward_info["task/base_centerline"] = r_task_base_centerline
        reward_info["task/base_net_progress"] = r_task_base_net_progress
        reward_info["task/base_assist"] = r_task_base_assist
        reward_info["task/base_door_sync"] = r_task_base_door_sync
        reward_info["task/base_cross"] = r_task_base_cross

        # ══════════════════════════════════════════════════════════════
        # §5 稳定性奖励：5 子项 × 双臂（使用缓存的加速度 + tilt_perp）
        # ══════════════════════════════════════════════════════════════
        m_l = self._left_occupied.float()  # (N,)
        m_r = self._right_occupied.float()

        r_stab_left = torch.zeros(self.num_envs, device=self.device)
        r_stab_right = torch.zeros(self.num_envs, device=self.device)

        for side_name, m, la, aa, tilt_perp, ee_lv_w, ee_av_w in [
            ("left", m_l, self._cached_left_ee_la, self._cached_left_ee_aa,
             self._cached_left_tilt_perp, self._cached_left_ee_lin_vel_w,
             self._cached_left_ee_ang_vel_w),
            ("right", m_r, self._cached_right_ee_la, self._cached_right_ee_aa,
             self._cached_right_tilt_perp, self._cached_right_ee_lin_vel_w,
             self._cached_right_ee_ang_vel_w),
        ]:
            la_sq = (la * la).sum(-1)
            aa_sq = (aa * aa).sum(-1)
            tilt_sq = (tilt_perp * tilt_perp).sum(-1)

            side_terms = {
                "zero_acc": m * (
                    self.cfg.rew_w_zero_acc * torch.exp(-self.cfg.rew_lambda_acc * la_sq)
                ),
                "zero_ang": m * (
                    self.cfg.rew_w_zero_ang * torch.exp(-self.cfg.rew_lambda_ang * aa_sq)
                ),
                "acc": m * (-self.cfg.rew_w_acc * la_sq),
                "ang": m * (-self.cfg.rew_w_ang * aa_sq),
                "tilt": m * (-self.cfg.rew_w_tilt * tilt_sq),
                "ee_lin_vel": compute_soft_capped_velocity_penalty(
                    velocity=ee_lv_w,
                    occupied=m.bool(),
                    free_speed=self.cfg.rew_ee_lin_vel_free,
                    weight=self.cfg.rew_w_ee_lin_vel,
                ),
                "ee_ang_vel": compute_soft_capped_velocity_penalty(
                    velocity=ee_av_w,
                    occupied=m.bool(),
                    free_speed=self.cfg.rew_ee_ang_vel_free,
                    weight=self.cfg.rew_w_ee_ang_vel,
                ),
            }
            r_stab_side = sum(side_terms.values())
            reward_info[f"stab_{side_name}"] = r_stab_side
            for term_name, term_value in side_terms.items():
                reward_info[f"stab_{side_name}/{term_name}"] = term_value

            if side_name == "left":
                r_stab_left = r_stab_side
            else:
                r_stab_right = r_stab_side

        # ══════════════════════════════════════════════════════════════
        # §6 安全惩罚：arm + mobile-base 子项（正惩罚量）
        # ══════════════════════════════════════════════════════════════
        # 关节速度超限惩罚
        joint_vel = robot.data.joint_vel[:, self._arm_joint_ids]  # (N, 12)
        if hasattr(robot.data, "soft_joint_vel_limits"):
            vel_limit = robot.data.soft_joint_vel_limits[0, self._arm_joint_ids]  # (12,)
        else:
            raise RuntimeError(
                "robot.data.soft_joint_vel_limits 不可用，"
                "无法获取关节最大转速。请检查 Isaac Lab 版本或 Articulation 配置。"
            )
        vel_threshold = self.cfg.rew_mu * vel_limit
        vel_excess = torch.clamp(torch.abs(joint_vel) - vel_threshold, min=0)
        r_safe_joint_vel = self.cfg.rew_beta_vel * (vel_excess ** 2).sum(-1)

        # 目标角边界带惩罚（基于最终执行目标，只在接近 limits 时激活）
        joint_limits = robot.data.soft_joint_pos_limits[0, self._arm_joint_ids]  # (12, 2)
        q_min = joint_limits[:, 0]  # (12,)
        q_max = joint_limits[:, 1]  # (12,)
        r_safe_target_limit = compute_joint_limit_margin_penalty(
            q_target=self._prev_joint_target,
            q_min=q_min,
            q_max=q_max,
            margin_ratio=self.cfg.rew_target_margin_ratio,
            beta=self.cfg.rew_beta_target,
        )

        r_safe_cup_drop = cup_dropped.float() * self.cfg.rew_w_drop

        # 关节目标变化率惩罚：约束策略写入 PD 的最终目标变化
        r_safe_target_rate = compute_target_rate_penalty(
            current_target=self._prev_joint_target,
            previous_target=self._prev_joint_target - self._cached_joint_target_delta,
            beta=self.cfg.rew_beta_target_rate,
            free_l2=self.cfg.rew_target_rate_free_l2,
        )

        # 杯体-门板接近惩罚：只对持杯侧生效
        cup_left: RigidObject = self.scene["cup_left"]
        cup_right: RigidObject = self.scene["cup_right"]
        left_cup_base = batch_vector_world_to_base(
            cup_left.data.root_pos_w - base_pos_w, base_quat_w
        )
        right_cup_base = batch_vector_world_to_base(
            cup_right.data.root_pos_w - base_pos_w, base_quat_w
        )
        left_cup_door_dist = self._compute_point_to_panel_face_distance(
            points_base=left_cup_base,
            face_center_base=self._cached_door_panel_face_center_base,
            face_rot_base=self._cached_door_panel_face_rot_base,
            half_extent_y=_DOOR_PANEL_HALF_EXTENT_Y,
            half_extent_z=_DOOR_PANEL_HALF_EXTENT_Z,
        )
        right_cup_door_dist = self._compute_point_to_panel_face_distance(
            points_base=right_cup_base,
            face_center_base=self._cached_door_panel_face_center_base,
            face_rot_base=self._cached_door_panel_face_rot_base,
            half_extent_y=_DOOR_PANEL_HALF_EXTENT_Y,
            half_extent_z=_DOOR_PANEL_HALF_EXTENT_Z,
        )
        prox_thresh = self.cfg.rew_cup_door_prox_threshold
        left_prox_intrusion = torch.clamp(prox_thresh - left_cup_door_dist, min=0)
        right_prox_intrusion = torch.clamp(prox_thresh - right_cup_door_dist, min=0)
        r_safe_cup_door_prox = self.cfg.rew_beta_cup_door_prox * (
            m_l * left_prox_intrusion ** 2 + m_r * right_prox_intrusion ** 2
        )
        base_lv = batch_vector_world_to_base(
            robot.data.body_lin_vel_w[:, self._base_body_idx],
            robot.data.body_quat_w[:, self._base_body_idx],
        )
        base_av = batch_vector_world_to_base(
            robot.data.body_ang_vel_w[:, self._base_body_idx],
            robot.data.body_quat_w[:, self._base_body_idx],
        )
        base_speed_sq = compute_base_speed_squared(
            base_lin_vel_base=base_lv,
            base_ang_vel_base=base_av,
        )
        r_safe_base_zero_speed = compute_base_zero_speed_reward(
            speed_sq=base_speed_sq,
            weight=self.cfg.rew_w_base_zero_speed,
            decay=self.cfg.rew_lambda_base_speed,
        )
        r_safe_base_speed = compute_base_speed_penalty(
            speed_sq=base_speed_sq,
            weight=self.cfg.rew_w_base_speed,
        )
        base_cmd_delta_norm = self._cached_base_cmd_delta.norm(dim=-1)
        r_safe_base_cmd_delta = self.cfg.rew_beta_base_cmd * torch.square(base_cmd_delta_norm)
        r_safe_base_heading = self.cfg.rew_beta_base_heading * compute_base_heading_penalty(
            base_forward_world=doorway_state["base_forward_world"],
            doorway_tangent_world=doorway_state["doorway_tangent_w"],
        )
        r_safe_base_corridor = self.cfg.rew_beta_base_corridor * torch.square(
            doorway_state["base_corridor_excess"]
        )

        r_safe = (
            r_safe_joint_vel
            + r_safe_target_limit
            + r_safe_cup_drop
            + r_safe_target_rate
            + r_safe_cup_door_prox
            - r_safe_base_zero_speed
            + r_safe_base_speed
            + r_safe_base_cmd_delta
            + r_safe_base_heading
            + r_safe_base_corridor
        )

        reward_info["safe"] = r_safe
        reward_info["safe/joint_vel"] = r_safe_joint_vel
        reward_info["safe/target_limit"] = r_safe_target_limit
        reward_info["safe/cup_drop"] = r_safe_cup_drop
        reward_info["safe/target_rate"] = r_safe_target_rate
        reward_info["safe/cup_door_prox"] = r_safe_cup_door_prox
        reward_info["safe/base_zero_speed"] = r_safe_base_zero_speed
        reward_info["safe/base_speed"] = r_safe_base_speed
        reward_info["safe/base_cmd_delta"] = r_safe_base_cmd_delta
        reward_info["safe/base_heading"] = r_safe_base_heading
        reward_info["safe/base_corridor"] = r_safe_base_corridor

        r_total = r_task + r_stab_left + r_stab_right - r_safe
        reward_info["total"] = r_total
        for key, value in reward_info.items():
            self._episode_reward_sums[key] += value

        target_rate_l2 = self._cached_joint_target_delta.norm(dim=-1)
        target_rate_sq = torch.square(self._cached_joint_target_delta).sum(dim=-1)
        left_ee_speed_world = self._cached_left_ee_lin_vel_w.norm(dim=-1)
        right_ee_speed_world = self._cached_right_ee_lin_vel_w.norm(dim=-1)
        left_ee_ang_speed_world = self._cached_left_ee_ang_vel_w.norm(dim=-1)
        right_ee_ang_speed_world = self._cached_right_ee_ang_vel_w.norm(dim=-1)

        self._episode_state_sums["base_push_progress_sum"] += base_forward_delta
        self._episode_state_sums["base_net_progress_sum"] += base_signed_progress_delta
        self._episode_state_sums["base_inside_progress_sum"] += base_inside_progress
        self._episode_state_sums["base_corridor_excess_sum"] += doorway_state["base_corridor_excess"]
        self._episode_state_sums["base_align_score_sum"] += base_align_score
        self._episode_state_sums["base_range_score_sum"] += base_range_score
        self._episode_state_sums["ee_left_speed_world_sum"] += m_l * left_ee_speed_world
        self._episode_state_sums["ee_right_speed_world_sum"] += m_r * right_ee_speed_world
        self._episode_state_sums["ee_left_ang_speed_world_sum"] += m_l * left_ee_ang_speed_world
        self._episode_state_sums["ee_right_ang_speed_world_sum"] += m_r * right_ee_ang_speed_world
        self._episode_state_sums["target_rate_l2_sum"] += target_rate_l2
        self._episode_state_sums["target_rate_sq_sum"] += target_rate_sq

        self._episode_state_max["base_corridor_excess_max"] = torch.maximum(
            self._episode_state_max["base_corridor_excess_max"],
            doorway_state["base_corridor_excess"],
        )
        self._episode_state_max["ee_left_speed_world_max"] = torch.maximum(
            self._episode_state_max["ee_left_speed_world_max"],
            m_l * left_ee_speed_world,
        )
        self._episode_state_max["ee_right_speed_world_max"] = torch.maximum(
            self._episode_state_max["ee_right_speed_world_max"],
            m_r * right_ee_speed_world,
        )
        self._episode_state_max["ee_left_ang_speed_world_max"] = torch.maximum(
            self._episode_state_max["ee_left_ang_speed_world_max"],
            m_l * left_ee_ang_speed_world,
        )
        self._episode_state_max["ee_right_ang_speed_world_max"] = torch.maximum(
            self._episode_state_max["ee_right_ang_speed_world_max"],
            m_r * right_ee_ang_speed_world,
        )
        self._episode_state_max["target_rate_l2_max"] = torch.maximum(
            self._episode_state_max["target_rate_l2_max"],
            target_rate_l2,
        )

        door_open_met = theta >= self.cfg.door_angle_target
        truncated = (self._step_count + 1) >= self.max_episode_length
        done_mask = cup_dropped | success | truncated
        if done_mask.any():
            done_env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            episode_lengths = (self._step_count[done_env_ids] + 1).to(dtype=torch.float32).clamp(min=1.0)
            nan_state = torch.full_like(episode_lengths, float("nan"))
            left_done = self._left_occupied[done_env_ids]
            right_done = self._right_occupied[done_env_ids]
            episode_state_info = {
                "base_signed_distance_final": current_base_signed_distance[done_env_ids].clone(),
                "base_signed_distance_delta": (
                    self._initial_base_signed_doorway_distance[done_env_ids]
                    - current_base_signed_distance[done_env_ids]
                ).clone(),
                "base_push_progress_sum": self._episode_state_sums["base_push_progress_sum"][done_env_ids].clone(),
                "base_net_progress_sum": self._episode_state_sums["base_net_progress_sum"][done_env_ids].clone(),
                "base_inside_progress_sum": self._episode_state_sums["base_inside_progress_sum"][done_env_ids].clone(),
                "base_corridor_excess_mean": (
                    self._episode_state_sums["base_corridor_excess_sum"][done_env_ids] / episode_lengths
                ).clone(),
                "base_corridor_excess_max": self._episode_state_max["base_corridor_excess_max"][done_env_ids].clone(),
                "base_align_score_mean": (
                    self._episode_state_sums["base_align_score_sum"][done_env_ids] / episode_lengths
                ).clone(),
                "base_range_score_mean": (
                    self._episode_state_sums["base_range_score_sum"][done_env_ids] / episode_lengths
                ).clone(),
                "ee_left_speed_world_mean": torch.where(
                    left_done,
                    self._episode_state_sums["ee_left_speed_world_sum"][done_env_ids] / episode_lengths,
                    nan_state,
                ).clone(),
                "ee_right_speed_world_mean": torch.where(
                    right_done,
                    self._episode_state_sums["ee_right_speed_world_sum"][done_env_ids] / episode_lengths,
                    nan_state,
                ).clone(),
                "ee_left_speed_world_max": torch.where(
                    left_done,
                    self._episode_state_max["ee_left_speed_world_max"][done_env_ids],
                    nan_state,
                ).clone(),
                "ee_right_speed_world_max": torch.where(
                    right_done,
                    self._episode_state_max["ee_right_speed_world_max"][done_env_ids],
                    nan_state,
                ).clone(),
                "ee_left_ang_speed_world_mean": torch.where(
                    left_done,
                    self._episode_state_sums["ee_left_ang_speed_world_sum"][done_env_ids] / episode_lengths,
                    nan_state,
                ).clone(),
                "ee_right_ang_speed_world_mean": torch.where(
                    right_done,
                    self._episode_state_sums["ee_right_ang_speed_world_sum"][done_env_ids] / episode_lengths,
                    nan_state,
                ).clone(),
                "ee_left_ang_speed_world_max": torch.where(
                    left_done,
                    self._episode_state_max["ee_left_ang_speed_world_max"][done_env_ids],
                    nan_state,
                ).clone(),
                "ee_right_ang_speed_world_max": torch.where(
                    right_done,
                    self._episode_state_max["ee_right_ang_speed_world_max"][done_env_ids],
                    nan_state,
                ).clone(),
                "target_rate_l2_mean": (
                    self._episode_state_sums["target_rate_l2_sum"][done_env_ids] / episode_lengths
                ).clone(),
                "target_rate_l2_max": self._episode_state_max["target_rate_l2_max"][done_env_ids].clone(),
                "target_rate_sq_sum": self._episode_state_sums["target_rate_sq_sum"][done_env_ids].clone(),
            }
            self.extras["episode_reward_info"] = {
                key: value[done_env_ids].clone() for key, value in self._episode_reward_sums.items()
            }
            self.extras["episode_state_info"] = episode_state_info
            self.extras["reward_stage_index"] = torch.full(
                (done_env_ids.numel(),),
                float(self._reward_stage_index),
                device=self.device,
            )
            self.extras["success"] = success[done_env_ids].clone()
            self.extras["episode_left_occupied"] = self._left_occupied[done_env_ids].clone()
            self.extras["episode_right_occupied"] = self._right_occupied[done_env_ids].clone()
            self.extras["door_angle"] = theta[done_env_ids].clone()
            self.extras["base_crossed"] = self._base_link_crossed[done_env_ids].clone()
            self.extras["door_open_met"] = door_open_met[done_env_ids].clone()
            self.extras["door_open_but_not_crossed"] = (
                door_open_met & ~self._base_link_crossed
            )[done_env_ids].clone()
            self.extras["fail_cup_drop"] = cup_dropped[done_env_ids].clone()
            self.extras["fail_timeout"] = (truncated & ~cup_dropped & ~success)[done_env_ids].clone()
            self.extras["fail_not_crossed"] = (
                truncated & ~cup_dropped & ~self._base_link_crossed
            )[done_env_ids].clone()

        # 更新门角度缓存
        self._prev_door_angle = theta.clone()
        # 更新臂关节位置缓存
        self._prev_arm_joint_pos = robot.data.joint_pos[:, self._arm_joint_ids].clone()

        return r_total

    # ═══════════════════════════════════════════════════════════════════
    # 终止判定
    # ═══════════════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """批量判定终止。

        Returns
        -------
        terminated : (N,) bool — 杯掉落 or 门角度达标
        truncated : (N,) bool — 达到最大步数
        """
        self._step_count += 1

        door: Articulation = self.scene["door"]
        theta = door.data.joint_pos[:, 0]

        cup_dropped = self._cached_cup_dropped  # L12: 复用 _get_observations 的缓存
        success = self._cached_episode_success

        terminated = cup_dropped | success
        truncated = self._step_count >= self.max_episode_length

        return terminated, truncated

    # ═══════════════════════════════════════════════════════════════════
    # 选择性重置
    # ═══════════════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """选择性重置指定 env — 包含域随机化和持杯初始化。"""
        if env_ids is None:
            env_ids = self.scene["robot"]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # ── 1. 采样域随机化参数 ──────────────────────────────────
        self._cup_mass[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.cup_mass_range
        )
        self._door_mass[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.door_mass_range
        )
        self._door_damping[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.door_damping_range
        )

        base_pos, base_yaw = sample_base_poses(
            n,
            door_center_xy=self.cfg.door_center_xy,
            base_reference_xy=self.cfg.base_reference_xy,
            base_height=self.cfg.base_height,
            radius_range=self.cfg.base_radius_range,
            sector_half_angle_deg=self.cfg.base_sector_half_angle_deg,
            yaw_delta_deg=self.cfg.base_yaw_delta_deg,
            device=self.device,
        )
        self._base_pos[env_ids] = base_pos
        self._base_yaw[env_ids] = base_yaw
        explicit_override_mask = self._has_pending_domain_overrides(env_ids)
        self._apply_pending_domain_overrides(env_ids)
        self._apply_episode_reset_callback_overrides(
            env_ids, skip_mask=explicit_override_mask
        )
        self._base_pos[env_ids, 2] = self.cfg.base_height

        # ── 2. 写入机器人 root state 与 planar joint pose ──────
        robot: Articulation = self.scene["robot"]
        default_root = robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        default_root[:, 7:] = 0.0  # 零速度
        robot.write_root_state_to_sim(default_root, env_ids)

        # 重置关节到默认位置，并显式写入 planar base 关节
        default_jpos = robot.data.default_joint_pos[env_ids].clone()
        default_jvel = robot.data.default_joint_vel[env_ids].clone()
        default_jpos[:, self._planar_joint_ids[0]] = self._base_pos[env_ids, 0]
        default_jpos[:, self._planar_joint_ids[1]] = self._base_pos[env_ids, 1]
        default_jpos[:, self._planar_joint_ids[2]] = self._base_yaw[env_ids]
        default_jvel[:, self._planar_joint_ids] = 0.0
        default_jvel[:, self._wheel_joint_ids] = 0.0
        robot.write_joint_state_to_sim(default_jpos, default_jvel, None, env_ids)

        # ── 3. 重置门关节角度 ────────────────────────────────────
        door: Articulation = self.scene["door"]
        zero_door_pos = torch.zeros(n, door.num_joints, device=self.device)
        zero_door_vel = torch.zeros(n, door.num_joints, device=self.device)
        door.write_joint_state_to_sim(zero_door_pos, zero_door_vel, None, env_ids)

        # ── 3.5 初始化控制目标（不再额外推进物理步）─────────────────
        arm_target = default_jpos[:, self._arm_joint_ids]
        zero_planar_vel = torch.zeros((n, len(self._planar_joint_ids)), device=self.device)
        zero_wheel_vel = torch.zeros((n, len(self._wheel_joint_ids)), device=self.device)
        robot.set_joint_position_target(
            arm_target,
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        robot.set_joint_velocity_target(
            zero_planar_vel,
            joint_ids=self._planar_joint_ids,
            env_ids=env_ids,
        )
        if not self._training_planar_base_only:
            robot.set_joint_velocity_target(
                zero_wheel_vel,
                joint_ids=self._wheel_joint_ids,
                env_ids=env_ids,
            )
        self.scene.write_data_to_sim()

        # ── 4. occupancy 已由 reset 事件或外部接口设置 ──────────
        # 不在此处重置，保留 Isaac Lab reset event 或手动脚本注入的 occupancy。
        # 如果没有任何外部写入，occupancy 保持初始化时的 False。

        # ── 5. 杯体处理 ─────────────────────────────────────────
        # 将不需要的杯体 teleport 到远处
        self._teleport_cups_to_park(env_ids)

        # 对需要持杯的 env 执行批量杯抓取初始化
        need_cup = self._left_occupied[env_ids] | self._right_occupied[env_ids]
        if need_cup.any():
            cup_env_ids = env_ids[need_cup]
            self._batch_cup_grasp_init(
                cup_env_ids,
                joint_pos_seed=default_jpos[need_cup],
            )

        # 重置完成后立即刷新 gripper hold target，避免沿用上一回合目标。
        self._set_gripper_hold_targets(env_ids)

        # ── 6. 应用域随机化物理参数 ──────────────────────────────
        self._apply_domain_params(env_ids)

        # ── 7. 重置 per-env 状态 ────────────────────────────────
        self._step_count[env_ids] = 0
        self._prev_door_angle[env_ids] = 0.0
        current_arm_pos = robot.data.joint_pos[env_ids][:, self._arm_joint_ids].clone()
        self._prev_joint_target[env_ids] = current_arm_pos
        self._prev_arm_joint_pos[env_ids] = current_arm_pos
        self._cached_joint_target_delta[env_ids] = 0.0
        self._prev_base_cmd[env_ids] = 0.0
        self._already_succeeded[env_ids] = False
        self._prev_left_ee_lin_vel_w[env_ids] = 0.0
        self._prev_left_ee_ang_vel_w[env_ids] = 0.0
        self._prev_right_ee_lin_vel_w[env_ids] = 0.0
        self._prev_right_ee_ang_vel_w[env_ids] = 0.0
        self._cached_left_ee_lin_vel_w[env_ids] = 0.0
        self._cached_left_ee_ang_vel_w[env_ids] = 0.0
        self._cached_right_ee_lin_vel_w[env_ids] = 0.0
        self._cached_right_ee_ang_vel_w[env_ids] = 0.0
        self._cached_cup_dropped[env_ids] = False
        self._cached_doorway_corners_world[env_ids] = 0.0
        self._cached_doorway_corners_base[env_ids] = 0.0
        self._cached_base_signed_doorway_distance[env_ids] = 0.0
        self._prev_base_signed_doorway_distance[env_ids] = float("nan")
        self._initial_base_signed_doorway_distance[env_ids] = float("nan")
        self._cached_base_in_doorway_opening[env_ids] = False
        self._base_link_crossed[env_ids] = False
        self._cached_episode_success[env_ids] = False
        self._cached_approach_dist[env_ids] = 0.0
        self._cached_left_approach_dist[env_ids] = 0.0
        self._cached_right_approach_dist[env_ids] = 0.0
        self._cached_base_cmd_delta[env_ids] = 0.0
        self._cached_base_force_cmd[env_ids] = 0.0
        self._cached_base_torque_cmd[env_ids] = 0.0
        self._cached_planar_joint_targets[env_ids] = 0.0
        self._cached_wheel_saturation_ratio[env_ids] = 0.0
        self._initial_approach_dist[env_ids] = float("nan")
        self._initial_left_approach_dist[env_ids] = float("nan")
        self._initial_right_approach_dist[env_ids] = float("nan")
        for value in self._episode_reward_sums.values():
            value[env_ids] = 0.0
        for value in self._episode_state_sums.values():
            value[env_ids] = 0.0
        for value in self._episode_state_max.values():
            value[env_ids] = 0.0

        robot.instantaneous_wrench_composer.reset(env_ids)

    # ═══════════════════════════════════════════════════════════════════
    # 视口帧捕获
    # ═══════════════════════════════════════════════════════════════════

    def get_visual_observation(self) -> dict | None:
        """捕获当前视口渲染帧，返回 ``{"rgb": ndarray(H, W, 3)}``。

        需要在 Isaac Sim 运行时且渲染已启用时才能工作。
        不可用时返回 ``None``。
        """
        import numpy as np

        # ── 初始化 RGB annotator（仅首次调用）──────────────────────
        if not hasattr(self, "_rgb_annotator"):
            self._rgb_annotator = None
            try:
                import omni.replicator.core as rep
                from omni.kit.viewport.utility import get_active_viewport

                vp = get_active_viewport()
                if vp is None:
                    print("[DIAG] get_active_viewport() 返回 None")
                else:
                    # ── 一次性诊断：列出 viewport 可用方法 ──
                    vp_methods = [m for m in dir(vp) if not m.startswith("_")]
                    print(f"[DIAG] ViewportAPI type: {type(vp)}")
                    print(f"[DIAG] ViewportAPI methods: {vp_methods}")

                    rp_path = getattr(vp, "render_product_path", None)
                    print(f"[DIAG] render_product_path: {rp_path}")

                    # 尝试直接 get_texture
                    get_tex = getattr(vp, "get_texture", None)
                    print(f"[DIAG] get_texture method: {get_tex}")

                    if get_tex is not None:
                        self.sim.render()
                        tex = get_tex()
                        print(f"[DIAG] texture object: {tex}, type={type(tex)}")
                        if tex is not None:
                            tex_methods = [m for m in dir(tex) if not m.startswith("_")]
                            print(f"[DIAG] texture methods: {tex_methods}")
                            data = getattr(tex, "get_byte_array", lambda: None)()
                            print(f"[DIAG] byte_array: size={len(data) if data is not None else None}")
                            res = getattr(vp, "get_texture_resolution", lambda: (0,0))()
                            print(f"[DIAG] texture resolution: {res}")

                    # 也尝试 annotator 方式作为备选
                    if rp_path:
                        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        self._rgb_annotator.attach([rp_path])
                        print(f"[DIAG] RGB annotator 已绑定到 {rp_path}")
            except Exception as e:
                print(f"[DIAG] 初始化失败: {e}")
                import traceback; traceback.print_exc()

        # ── 获取帧数据 ─────────────────────────────────────────────
        # 模式：先 render()，再从 annotator 读上一帧的结果。
        # 这有一帧延迟，但对可视化 rollout 无影响。
        # 不要用 rep.orchestrator.step() — 它会阻塞/死锁。
        if self._rgb_annotator is not None:
            try:
                self.sim.render()
                data = self._rgb_annotator.get_data()
                if data is not None and data.size > 0:
                    rgb = data[:, :, :3] if data.shape[-1] >= 3 else data
                    return {"rgb": np.asarray(rgb, dtype=np.uint8).copy()}
                # 首帧可能为空（渲染尚未完成），不打印重复警告
                return None
            except Exception as e:
                logger.warning("get_visual_observation: 获取帧失败: %s", e)

        return None

    # ═══════════════════════════════════════════════════════════════════
    # 课程注入接口
    # ═══════════════════════════════════════════════════════════════════

    def set_occupancy(
        self,
        left_occupied: Tensor,
        right_occupied: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """设置持杯 occupancy，支持整批或局部 env 更新。

        Parameters
        ----------
        left_occupied : (N,) bool
        right_occupied : (N,) bool
        env_ids : (K,) long, optional
            为 None 时覆盖所有 env；否则只覆盖给定 env_ids 对应的子集。
        """
        if env_ids is None:
            self._left_occupied[:] = left_occupied
            self._right_occupied[:] = right_occupied
            return

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._left_occupied[env_ids] = left_occupied
        self._right_occupied[env_ids] = right_occupied

    def get_debug_state(self) -> dict[str, Tensor]:
        """返回当前环境调试状态，供手动验证脚本读取。"""
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        cup_dropped = self._check_cup_dropped()
        door_angle = door.data.joint_pos[:, 0].clone()
        door_open_met = door_angle >= self.cfg.door_angle_target
        episode_success = door_open_met & ~cup_dropped & self._base_link_crossed
        base_pos = robot.data.body_pos_w[:, self._base_body_idx].clone()
        base_quat = robot.data.body_quat_w[:, self._base_body_idx].clone()
        base_lin_vel_base = batch_vector_world_to_base(
            robot.data.body_lin_vel_w[:, self._base_body_idx],
            robot.data.body_quat_w[:, self._base_body_idx],
        )
        base_ang_vel_base = batch_vector_world_to_base(
            robot.data.body_ang_vel_w[:, self._base_body_idx],
            robot.data.body_quat_w[:, self._base_body_idx],
        )

        state = {
            "left_occupied": self._left_occupied.clone(),
            "right_occupied": self._right_occupied.clone(),
            "door_angle": door_angle,
            "door_open_met": door_open_met,
            "cup_dropped": cup_dropped,
            "base_crossed": self._base_link_crossed.clone(),
            "episode_success": episode_success,
            "arm_joint_positions": robot.data.joint_pos[:, self._arm_joint_ids].clone(),
            "arm_joint_targets": self._prev_joint_target.clone(),
            "base_pos_w": base_pos,
            "base_quat_w": base_quat,
            "base_lin_vel_base": base_lin_vel_base.clone(),
            "base_ang_vel_base": base_ang_vel_base.clone(),
            "base_cmd": self._prev_base_cmd.clone(),
            "base_controller_backend": self._base_controller_backend,
            "base_force_body_name": self._base_force_body_name,
            "base_force_cmd": self._cached_base_force_cmd.clone(),
            "base_torque_cmd": self._cached_base_torque_cmd.clone(),
            "planar_joint_positions": robot.data.joint_pos[:, self._planar_joint_ids].clone(),
            "planar_joint_velocities": robot.data.joint_vel[:, self._planar_joint_ids].clone(),
            "planar_joint_targets": self._cached_planar_joint_targets.clone(),
        }
        if self._emit_wheel_debug_state:
            state["wheel_joint_velocities"] = robot.data.joint_vel[:, self._wheel_joint_ids].clone()
            state["wheel_saturation_ratio"] = self._cached_wheel_saturation_ratio.clone()
        return state

    def set_episode_reset_fn(self, fn) -> None:
        """注册可选的 reset 覆写回调，保留给外部脚本或实验接口使用。"""
        self._episode_reset_fn = fn

    def set_domain_params_batch(
        self, params_list: list[dict[str, float]]
    ) -> None:
        """从外部注入域随机化参数，供下次 _reset_idx 使用。

        Parameters
        ----------
        params_list : list[dict]
            长度 = num_envs，每个 dict 含 cup_mass/door_mass/door_damping 等。
        """
        for i, p in enumerate(params_list):
            if p is None:
                continue
            if "cup_mass" in p:
                self._pending_cup_mass[i] = float(p["cup_mass"])
            if "door_mass" in p:
                self._pending_door_mass[i] = float(p["door_mass"])
            if "door_damping" in p:
                self._pending_door_damping[i] = float(p["door_damping"])
            if "base_pos" in p:
                self._pending_base_pos[i] = torch.as_tensor(
                    p["base_pos"], device=self.device, dtype=torch.float32
                )
            if "base_yaw" in p:
                self._pending_base_yaw[i] = float(p["base_yaw"])

    # ═══════════════════════════════════════════════════════════════════
    # 内部工具方法
    # ═══════════════════════════════════════════════════════════════════

    def _ee_world_to_base(
        self,
        ee_pos_w: Tensor,
        ee_quat_w: Tensor,
        ee_lv_w: Tensor,
        ee_av_w: Tensor,
        base_pos_w: Tensor,
        base_quat_w: Tensor,
        base_lv_w: Tensor,
        base_av_w: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """将世界系 EE 状态变换到 base_link 相对系。

        位置和姿态变换到 base_link 相对系；速度直接旋转到 base_link 系表达（世界系速度）。
        """
        pos_base = batch_vector_world_to_base(ee_pos_w - base_pos_w, base_quat_w)
        quat_base = batch_orientation_world_to_base(ee_quat_w, base_quat_w)
        lv_base = batch_vector_world_to_base(ee_lv_w, base_quat_w)
        av_base = batch_vector_world_to_base(ee_av_w, base_quat_w)
        return pos_base, quat_base, lv_base, av_base

    def _has_pending_domain_overrides(self, env_ids: Tensor) -> Tensor:
        """返回每个 env 是否存在显式注入的域随机化覆写。"""
        return (
            ~torch.isnan(self._pending_cup_mass[env_ids])
            | ~torch.isnan(self._pending_door_mass[env_ids])
            | ~torch.isnan(self._pending_door_damping[env_ids])
            | ~torch.isnan(self._pending_base_yaw[env_ids])
            | ~torch.isnan(self._pending_base_pos[env_ids, 0])
        )

    def _apply_pending_domain_overrides(self, env_ids: Tensor) -> None:
        """应用由 `set_domain_params_batch()` 预写入的覆写值。"""
        cup_mass_mask = ~torch.isnan(self._pending_cup_mass[env_ids])
        if cup_mass_mask.any():
            target_ids = env_ids[cup_mass_mask]
            self._cup_mass[target_ids] = self._pending_cup_mass[target_ids]
            self._pending_cup_mass[target_ids] = float("nan")

        door_mass_mask = ~torch.isnan(self._pending_door_mass[env_ids])
        if door_mass_mask.any():
            target_ids = env_ids[door_mass_mask]
            self._door_mass[target_ids] = self._pending_door_mass[target_ids]
            self._pending_door_mass[target_ids] = float("nan")

        door_damping_mask = ~torch.isnan(self._pending_door_damping[env_ids])
        if door_damping_mask.any():
            target_ids = env_ids[door_damping_mask]
            self._door_damping[target_ids] = self._pending_door_damping[target_ids]
            self._pending_door_damping[target_ids] = float("nan")

        base_pos_mask = ~torch.isnan(self._pending_base_pos[env_ids, 0])
        if base_pos_mask.any():
            target_ids = env_ids[base_pos_mask]
            self._base_pos[target_ids] = self._pending_base_pos[target_ids]
            self._pending_base_pos[target_ids] = float("nan")

        base_yaw_mask = ~torch.isnan(self._pending_base_yaw[env_ids])
        if base_yaw_mask.any():
            target_ids = env_ids[base_yaw_mask]
            self._base_yaw[target_ids] = self._pending_base_yaw[target_ids]
            self._pending_base_yaw[target_ids] = float("nan")

    def _apply_episode_reset_callback_overrides(
        self,
        env_ids: Tensor,
        *,
        skip_mask: Tensor | None = None,
    ) -> None:
        """对未显式覆写的 env 调用外部 auto-reset 回调。"""
        if self._episode_reset_fn is None:
            return

        for local_idx, env_id in enumerate(env_ids.tolist()):
            if skip_mask is not None and bool(skip_mask[local_idx]):
                continue

            override = self._normalize_episode_reset_override(
                self._episode_reset_fn(int(env_id))
            )
            if override is None:
                continue

            left_occupied = override.get("left_occupied")
            right_occupied = override.get("right_occupied")
            if left_occupied is not None:
                self._left_occupied[env_id] = bool(left_occupied)
            if right_occupied is not None:
                self._right_occupied[env_id] = bool(right_occupied)

            domain_params = override.get("domain_params")
            if not isinstance(domain_params, dict):
                continue

            if "cup_mass" in domain_params:
                self._cup_mass[env_id] = float(domain_params["cup_mass"])
            if "door_mass" in domain_params:
                self._door_mass[env_id] = float(domain_params["door_mass"])
            if "door_damping" in domain_params:
                self._door_damping[env_id] = float(domain_params["door_damping"])
            if "base_pos" in domain_params:
                self._base_pos[env_id] = torch.as_tensor(
                    domain_params["base_pos"],
                    device=self.device,
                    dtype=torch.float32,
                )
            if "base_yaw" in domain_params:
                self._base_yaw[env_id] = float(domain_params["base_yaw"])

    @staticmethod
    def _normalize_episode_reset_override(override):
        """兼容 tuple/dict 两种 reset 回调返回格式。"""
        if override is None:
            return None
        if isinstance(override, dict):
            return {
                "domain_params": override.get("domain_params", override),
                "left_occupied": override.get("left_occupied"),
                "right_occupied": override.get("right_occupied"),
                "door_type": override.get("door_type"),
            }
        if isinstance(override, (tuple, list)) and len(override) == 4:
            domain_params, door_type, left_occupied, right_occupied = override
            return {
                "domain_params": domain_params,
                "door_type": door_type,
                "left_occupied": left_occupied,
                "right_occupied": right_occupied,
            }
        return None

    @staticmethod
    def _compute_tilt(quat_world: Tensor) -> tuple[Tensor, Tensor]:
        """计算 cup tilt-to-gravity proxy — 重力在 EE 局部系中偏离 Y 轴的投影。

        抓取姿态下 joint6 = ±90° (绕 X 轴) 使 gripperMover Y 轴对齐世界竖直方向，
        因此取 g_local 的 xz 分量（偏离 Y 轴的分量）作为倾斜度量。
        当杯子竖直时 g_local 沿 Y 轴，tilt = 0；倾斜时 xz 分量增大，tilt > 0。

        Parameters
        ----------
        quat_world : (N, 4) wxyz

        Returns
        -------
        tilt_norm : (N, 1) tilt 标量范数
        tilt_perp : (N, 2) tilt 的 xz 分量（重力偏离 EE Y 轴的分量，用于奖励计算）
        """
        R = batch_quat_to_rotation_matrix(quat_world)  # (N, 3, 3)
        g_world = torch.tensor(
            [0.0, 0.0, -9.81], device=quat_world.device
        ).expand(quat_world.shape[0], 3)
        # 直接使用 EE 在世界系下的姿态投影世界重力，不依赖 base 是否水平。
        g_local = torch.bmm(
            R.transpose(-1, -2),
            g_world.unsqueeze(-1),
        ).squeeze(-1)  # (N, 3)
        # 抓取姿态下 EE Y 轴 ≈ 世界 Z 轴（竖直），取 xz 分量度量偏离竖直的程度
        tilt_perp = g_local[:, [0, 2]]  # (N, 2)
        return tilt_perp.norm(dim=-1, keepdim=True), tilt_perp  # (N, 1), (N, 2)

    def _check_cup_dropped(self) -> Tensor:
        """检测杯体是否脱落 — 通过 cup-EE 距离阈值判定。

        Returns
        -------
        (N,) bool
        """
        robot: Articulation = self.scene["robot"]
        cup_left: RigidObject = self.scene["cup_left"]
        cup_right: RigidObject = self.scene["cup_right"]

        left_ee_pos = robot.data.body_pos_w[:, self._left_ee_body_idx]  # (N, 3)
        right_ee_pos = robot.data.body_pos_w[:, self._right_ee_body_idx]
        left_cup_pos = cup_left.data.root_pos_w  # (N, 3)
        right_cup_pos = cup_right.data.root_pos_w

        threshold = self.cfg.cup_drop_threshold

        left_dist = (left_cup_pos - left_ee_pos).norm(dim=-1)
        right_dist = (right_cup_pos - right_ee_pos).norm(dim=-1)

        left_dropped = self._left_occupied & (left_dist > threshold)
        right_dropped = self._right_occupied & (right_dist > threshold)

        return left_dropped | right_dropped

    def _teleport_cups_to_park(self, env_ids: Tensor) -> None:
        """将不需要的杯体 teleport 到远处。"""
        cup_left: RigidObject = self.scene["cup_left"]
        cup_right: RigidObject = self.scene["cup_right"]

        # 将不持杯的 env 杯体放到远处
        left_park = env_ids[~self._left_occupied[env_ids]]
        if len(left_park) > 0:
            park_l = torch.zeros(len(left_park), 13, device=self.device)
            park_l[:, 0] = 100.0
            park_l[:, 3] = 1.0
            park_l[:, :3] += self.scene.env_origins[left_park]
            cup_left.write_root_state_to_sim(park_l, left_park)

        right_park = env_ids[~self._right_occupied[env_ids]]
        if len(right_park) > 0:
            park_r = torch.zeros(len(right_park), 13, device=self.device)
            park_r[:, 0] = 100.0
            park_r[:, 1] = 1.0
            park_r[:, 3] = 1.0
            park_r[:, :3] += self.scene.env_origins[right_park]
            cup_right.write_root_state_to_sim(park_r, right_park)

    def _apply_domain_params(self, env_ids: Tensor) -> None:
        """将域随机化物理参数写入仿真引擎。

        PhysX tensors API（本版本）要求 ``set_masses`` 的数据和索引都在 CPU。
        计算在 GPU 完成后，传参时统一 ``.cpu()``。
        """
        door: Articulation = self.scene["door"]
        cup_left: RigidObject = self.scene["cup_left"]
        cup_right: RigidObject = self.scene["cup_right"]
        dev = self.device

        n = len(env_ids)

        # 门板质量
        door_masses = door.root_physx_view.get_masses()
        if door_masses is not None:
            payload, target_env_ids = build_articulation_mass_update(
                masses=door_masses.to(dev),
                env_ids=env_ids,
                all_env_ids=door._ALL_INDICES.to(dev),
                body_idx=self._door_panel_body_idx,
                body_masses=self._door_mass[env_ids],
            )
            door.root_physx_view.set_masses(
                payload.cpu(),
                target_env_ids.cpu(),
            )

        # 门铰链阻尼
        if hasattr(door, "write_joint_damping_to_sim"):
            damping = self._door_damping[env_ids].unsqueeze(-1).expand(n, door.num_joints)
            door.write_joint_damping_to_sim(damping, env_ids=env_ids)

        # 杯体质量
        if hasattr(cup_left, "root_physx_view"):
            left_cup_mass = cup_left.root_physx_view.get_masses()
            if left_cup_mass is not None:
                payload_l, target_env_ids_l = build_rigid_body_mass_update(
                    masses=left_cup_mass.to(dev),
                    env_ids=env_ids,
                    body_masses=self._cup_mass[env_ids],
                )
                cup_left.root_physx_view.set_masses(
                    payload_l.cpu(),
                    target_env_ids_l.cpu(),
                )

        if hasattr(cup_right, "root_physx_view"):
            right_cup_mass = cup_right.root_physx_view.get_masses()
            if right_cup_mass is not None:
                payload_r, target_env_ids_r = build_rigid_body_mass_update(
                    masses=right_cup_mass.to(dev),
                    env_ids=env_ids,
                    body_masses=self._cup_mass[env_ids],
                )
                cup_right.root_physx_view.set_masses(
                    payload_r.cpu(),
                    target_env_ids_r.cpu(),
                )

    def _batch_cup_grasp_init(self, env_ids: Tensor, *, joint_pos_seed: Tensor | None = None) -> None:
        """批量持杯初始化 — 纯 teleport 方式，不调用 sim.step()。

        直接将臂关节设到关闭抓取姿态，并将杯体 teleport 到夹爪位置。
        避免了 sim.step() 推进所有环境导致非目标 env 状态被破坏的问题。

        流程：
            1. 设置臂关节到预设抓取姿态（最终状态）
            2. 直接关闭 gripper
            3. 计算杯体相对基座的世界坐标并 teleport
        """
        robot: Articulation = self.scene["robot"]
        cup_left: RigidObject = self.scene["cup_left"]
        cup_right: RigidObject = self.scene["cup_right"]

        n = len(env_ids)

        # ── 1. 设置臂关节到预设抓取姿态（直接到最终状态）──────────
        # 从 reset 已写好的 joint seed 开始，保留 planar base pose，不要回退到底层默认关节状态。
        if joint_pos_seed is None:
            joint_pos_seed = robot.data.default_joint_pos[env_ids]
        joint_pos = build_grasp_init_joint_positions(
            joint_seed=joint_pos_seed,
            left_grasp_joint_ids=self._left_grasp_joint_ids,
            left_grasp_joint_targets=self._left_grasp_joint_targets,
            right_grasp_joint_ids=self._right_grasp_joint_ids,
            right_grasp_joint_targets=self._right_grasp_joint_targets,
            gripper_joint_ids=self._gripper_joint_ids,
            gripper_joint_targets=self._gripper_close_targets,
        )

        joint_vel = torch.zeros_like(joint_pos)  # (n, num_joints)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── 2. 计算杯体世界坐标并 teleport（不需要物理步进）─────
        # 使用 base_pos 和 base_yaw 计算杯体的世界坐标
        left_cup_ids = env_ids[self._left_occupied[env_ids]]
        right_cup_ids = env_ids[self._right_occupied[env_ids]]

        if len(left_cup_ids) > 0:
            local_yaw = self._base_yaw[left_cup_ids]
            base_p = self._base_pos[left_cup_ids] + self.scene.env_origins[left_cup_ids]
            cup_world = base_p + batch_rotate_relative_by_yaw(self._left_cup_relative_xyz, local_yaw)
            cup_state = torch.zeros(len(left_cup_ids), 13, device=self.device)
            cup_state[:, :3] = cup_world
            cup_state[:, 3] = 1.0  # quat w
            cup_left.write_root_state_to_sim(cup_state, left_cup_ids)

        if len(right_cup_ids) > 0:
            local_yaw = self._base_yaw[right_cup_ids]
            base_p = self._base_pos[right_cup_ids] + self.scene.env_origins[right_cup_ids]
            cup_world = base_p + batch_rotate_relative_by_yaw(self._right_cup_relative_xyz, local_yaw)
            cup_state = torch.zeros(len(right_cup_ids), 13, device=self.device)
            cup_state[:, :3] = cup_world
            cup_state[:, 3] = 1.0
            cup_right.write_root_state_to_sim(cup_state, right_cup_ids)
