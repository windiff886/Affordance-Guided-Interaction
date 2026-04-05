"""门交互任务环境 — 单环境主体实现。

数据流总览（单步 step）：

    action (12,)
        │
        ├── 1. clip(action, -effort_limit, effort_limit)
        ├── 2. 注入噪声 ε_a（如提供）
        ├── 3. sim_step × decimation
        ├── 4. 读取物理状态
        │      ├── 关节: q, dq, tau
        │      ├── 末端: pos, quat, v, ω
        │      ├── 门: angle, pose
        │      └── 杯: pose, vel
        ├── 5. ContactMonitor.update() → ContactSummary
        ├── 6. TaskManager.update() → TaskStatus
        ├── 7. ActorObsBuilder.build() → actor_obs
        ├── 8. CriticObsBuilder.build() → critic_obs
        └── 9. RewardManager.step() → (reward, terminate, info)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_env import BaseEnv, EnvConfig
from .scene_factory import SceneFactory, SceneHandles, _HAS_ISAAC_LAB
from .contact_monitor import ContactMonitor, ContactSummary
from .task_manager import TaskManager, TaskStatus, TerminationReason

from affordance_guided_interaction.observations.actor_obs_builder import (
    ActorObsBuilder,
)
from affordance_guided_interaction.observations.stability_proxy import (
    StabilityProxyState,
    build_stability_proxy,
)
from affordance_guided_interaction.observations.critic_obs_builder import (
    CriticObsBuilder,
)
from affordance_guided_interaction.rewards.reward_manager import RewardManager

# Isaac Lab 条件导入
if _HAS_ISAAC_LAB:
    import torch


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(lhs, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(rhs, dtype=np.float64)
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dtype=np.float64)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _context_name(left_occupied: bool, right_occupied: bool) -> str:
    if left_occupied and right_occupied:
        return "both"
    if left_occupied:
        return "left_only"
    if right_occupied:
        return "right_only"
    return "none"


class DoorInteractionEnv(BaseEnv):
    """门交互任务单环境实现。

    继承 BaseEnv，组装 SceneFactory / ContactMonitor / TaskManager /
    ActorObsBuilder / CriticObsBuilder / RewardManager 实现完整的
    reset() / step() 循环。

    Parameters
    ----------
    cfg : EnvConfig | None
        环境配置。
    sim_context : Any | None
        Isaac Lab SimulationContext 引用（无头训练时从外部传入）。
    """

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        sim_context: Any = None,
    ) -> None:
        super().__init__(cfg)

        # ── 内部组件 ──────────────────────────────────────────
        self._scene = SceneFactory(
            physics_dt=self.cfg.physics_dt,
            sim_context=sim_context,
        )
        self._sim_context = sim_context
        self._contact_monitor = ContactMonitor(
            force_threshold=self.cfg.contact_force_threshold,
            cup_drop_threshold=self.cfg.cup_drop_threshold,
        )
        self._task_manager = TaskManager(
            success_angle_threshold=float(self.cfg.reward_cfg["task"]["theta_target"]),
            episode_end_angle_threshold=self.cfg.door_angle_target,
            max_episode_steps=self.cfg.max_episode_steps,
        )
        self._actor_obs_builder = ActorObsBuilder(
            action_history_length=self.cfg.action_history_length,
        )
        self._reward_manager = RewardManager(self.cfg.reward_cfg)

        # ── 回合级状态 ────────────────────────────────────────
        self._handles: SceneHandles | None = None
        self._domain_params: dict[str, Any] = {}
        self._left_occupied: bool = False
        self._right_occupied: bool = False
        self._episode_context: str = "none"
        self._prev_action: np.ndarray = np.zeros(
            self.cfg.total_joints, dtype=np.float64
        )
        self._effort_limits: np.ndarray = self.cfg.get_effort_limits()
        self._left_stability_state = StabilityProxyState()
        self._right_stability_state = StabilityProxyState()

    # ═══════════════════════════════════════════════════════════════════
    # reset
    # ═══════════════════════════════════════════════════════════════════

    def reset(
        self,
        *,
        domain_params: dict[str, Any] | None = None,
        door_type: str = "push",
        left_occupied: bool = False,
        right_occupied: bool = False,
    ) -> tuple[dict, dict]:
        """重置环境，装配场景，返回初始观测。"""
        self._domain_params = domain_params or {}
        self._left_occupied = left_occupied
        self._right_occupied = right_occupied
        self._episode_context = _context_name(left_occupied, right_occupied)
        self._prev_action = np.zeros(self.cfg.total_joints, dtype=np.float64)
        self._left_stability_state = StabilityProxyState()
        self._right_stability_state = StabilityProxyState()

        # 1. 装配场景
        self._handles = self._scene.build(
            door_type=door_type,
            left_occupied=left_occupied,
            right_occupied=right_occupied,
            domain_params=self._domain_params,
        )

        # 2. 重置内部组件
        self._task_manager.reset()
        self._actor_obs_builder.reset()
        self._reward_manager.reset_episode()

        # 3. 读取初始物理状态
        state = self._read_physics_state()

        # 4. 构建初始观测
        actor_obs = self._build_actor_obs(state, action_taken=None)
        critic_obs = self._build_critic_obs(actor_obs, state, cup_dropped=0.0)

        return actor_obs, critic_obs

    # ═══════════════════════════════════════════════════════════════════
    # step
    # ═══════════════════════════════════════════════════════════════════

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, dict, float, bool, dict[str, Any]]:
        """执行一步仿真。

        Parameters
        ----------
        action : (12,) ndarray
            绝对物理力矩（N·m）。
        """
        raw_action = np.asarray(action, dtype=np.float64)

        # ── 1. 力矩安全截断 ────────────────────────────────────
        clipped_action = np.clip(raw_action, -self._effort_limits, self._effort_limits)

        # ── 2. 物理仿真步进（decimation 次）──────────────────
        self._sim_step(clipped_action)

        # ── 3. 读取新物理状态 ───────────────────────────────────
        state = self._read_physics_state()

        # ── 4. 接触事件汇总 ────────────────────────────────────
        contact_summary = self._contact_monitor.update(
            scene_handles=self._handles,
            left_ee_pos=state["left_ee_position"],
            right_ee_pos=state["right_ee_position"],
            cup_pos=state.get("cup_position"),
            left_occupied=self._left_occupied,
            right_occupied=self._right_occupied,
        )

        # ── 5. 任务状态判定 ────────────────────────────────────
        task_status = self._task_manager.update(
            door_angle=state["door_joint_pos"],
            cup_dropped=contact_summary.cup_dropped,
        )

        # ── 6. 构建观测 ───────────────────────────────────────
        actor_obs = self._build_actor_obs(state, action_taken=clipped_action)
        critic_obs = self._build_critic_obs(
            actor_obs,
            state,
            cup_dropped=float(contact_summary.cup_dropped),
        )

        # ── 7. 计算奖励 ───────────────────────────────────────
        reward, should_terminate, reward_info = self._compute_reward(
            state=state,
            task_status=task_status,
            contact_summary=contact_summary,
            raw_action=raw_action,
            applied_action=clipped_action,
        )

        # ── 8. 最终 done 判定 ─────────────────────────────────
        # TaskManager 和 RewardManager 都可触发终止
        done = task_status.done or should_terminate

        # ── 9. 组装 info ──────────────────────────────────────
        info: dict[str, Any] = {
            "success": task_status.success,
            "success_reached": task_status.success_reached,
            "success_time_step": task_status.success_time_step,
            "termination_reason": task_status.reason.name,
            "step_count": task_status.step_count,
            "door_angle": task_status.door_angle,
            "episode_context": self._episode_context,
            "cup_dropped": contact_summary.cup_dropped,
            **reward_info,
        }

        # 记录本步动作，供下一步 observations 使用
        self._prev_action = clipped_action.copy()

        return actor_obs, critic_obs, reward, done, info

    # ═══════════════════════════════════════════════════════════════════
    # close
    # ═══════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """释放仿真资源。"""
        self._handles = None

    # ═══════════════════════════════════════════════════════════════════
    # 内部方法 — 物理交互
    # ═══════════════════════════════════════════════════════════════════

    def _sim_step(self, action: np.ndarray) -> None:
        """推进 decimation 次物理仿真步。

        通过 Isaac Lab 的 Articulation.write_joint_effort_to_sim()
        设置关节力矩，然后调用 SimulationContext.step() 推进物理引擎。
        """
        if not _HAS_ISAAC_LAB or self._handles is None:
            return

        h = self._handles
        robot = h.robot_view

        # 检查是否为占位模式
        if isinstance(robot, type(None)) or not hasattr(robot, 'write_joint_effort_to_sim'):
            return

        # 将 numpy action 转为 torch tensor
        # action 维度: (12,) → 需要构建完整关节力矩向量
        effort_tensor = torch.zeros(
            (1, robot.num_joints), dtype=torch.float32, device=robot.device
        )

        # 只对双臂关节施加力矩（使用预解析的索引）
        if h.arm_joint_indices is not None:
            action_t = torch.tensor(
                action, dtype=torch.float32, device=robot.device
            ).unsqueeze(0)
            effort_tensor[:, h.arm_joint_indices] = action_t

        # 写入力矩并步进
        robot.write_joint_effort_to_sim(effort_tensor)

        # 多次物理步进（decimation）
        sim = self._sim_context
        if sim is not None:
            for _ in range(self.cfg.decimation):
                sim.step(render=False)

            # 更新 Isaac Lab 内部缓存
            robot.update(self.cfg.physics_dt * self.cfg.decimation)
            if h.door_view is not None and hasattr(h.door_view, 'update'):
                h.door_view.update(self.cfg.physics_dt * self.cfg.decimation)
            if h.cup_view is not None and hasattr(h.cup_view, 'update'):
                h.cup_view.update(self.cfg.physics_dt * self.cfg.decimation)

    def _read_physics_state(self) -> dict[str, Any]:
        """从 Isaac Lab 读取所有需要的物理量。

        返回一个扁平字典，包含机器人本体、末端、门、杯体的完整状态。
        """
        n = self.cfg.joints_per_arm  # 6

        # 占位模式：返回全零状态
        if (not _HAS_ISAAC_LAB
                or self._handles is None
                or not hasattr(self._handles.robot_view, 'data')):
            return self._get_zero_state(n)

        h = self._handles
        robot = h.robot_view

        # ── 读取机器人关节状态 ────────────────────────────────
        # 获取双臂 12 关节的 q / dq / tau
        all_q = robot.data.joint_pos[0]      # (num_joints,) GPU tensor
        all_dq = robot.data.joint_vel[0]     # (num_joints,)
        all_tau = robot.data.applied_torque[0] if hasattr(robot.data, 'applied_torque') \
            else torch.zeros_like(all_q)

        if h.arm_joint_indices is not None:
            arm_q = all_q[h.arm_joint_indices].cpu().numpy()
            arm_dq = all_dq[h.arm_joint_indices].cpu().numpy()
            arm_tau = all_tau[h.arm_joint_indices].cpu().numpy()
        else:
            arm_q = np.zeros(12, dtype=np.float64)
            arm_dq = np.zeros(12, dtype=np.float64)
            arm_tau = np.zeros(12, dtype=np.float64)

        # 分为左右臂各 6 维
        left_q, right_q = arm_q[:n], arm_q[n:]
        left_dq, right_dq = arm_dq[:n], arm_dq[n:]
        left_tau, right_tau = arm_tau[:n], arm_tau[n:]

        # ── 读取末端执行器位姿/速度 ──────────────────────────
        # 通过 body 索引从 robot.data 获取
        body_pos = robot.data.body_pos_w[0]     # (num_bodies, 3)
        body_quat = robot.data.body_quat_w[0]   # (num_bodies, 4)
        body_lin_vel_w = robot.data.body_lin_vel_w
        body_ang_vel_w = robot.data.body_ang_vel_w
        body_lin_vel = body_lin_vel_w[0]   # (num_bodies, 3)
        body_ang_vel = body_ang_vel_w[0]   # (num_bodies, 3)
        body_lin_acc = self._extract_robot_body_tensor(
            robot.data,
            preferred=("body_lin_acc_w", "body_com_lin_acc_w"),
        )[0]
        body_ang_acc = self._extract_robot_body_tensor(
            robot.data,
            preferred=("body_ang_acc_w", "body_com_ang_acc_w"),
        )[0]
        if h.base_body_idx < 0:
            raise RuntimeError("未解析到 base_link body 索引，无法构造 base_link 相对坐标。")

        base_pos_w = body_pos[h.base_body_idx].cpu().numpy()
        base_quat_w = body_quat[h.base_body_idx].cpu().numpy()
        base_lin_vel_w = body_lin_vel[h.base_body_idx].cpu().numpy()
        base_ang_vel_w = body_ang_vel[h.base_body_idx].cpu().numpy()
        base_lin_acc_w = body_lin_acc[h.base_body_idx].cpu().numpy()
        base_ang_acc_w = body_ang_acc[h.base_body_idx].cpu().numpy()

        if h.left_ee_body_idx >= 0:
            left_ee_pos = self._vector_world_to_base(
                body_pos[h.left_ee_body_idx].cpu().numpy() - base_pos_w,
                base_quat_w,
            )
            left_ee_quat = self._orientation_world_to_base(
                body_quat[h.left_ee_body_idx].cpu().numpy(),
                base_quat_w,
            )
            left_ee_lin_vel = self._vector_world_to_base(
                body_lin_vel[h.left_ee_body_idx].cpu().numpy() - base_lin_vel_w,
                base_quat_w,
            )
            left_ee_ang_vel = self._vector_world_to_base(
                body_ang_vel[h.left_ee_body_idx].cpu().numpy() - base_ang_vel_w,
                base_quat_w,
            )
            left_ee_lin_acc = self._vector_world_to_base(
                body_lin_acc[h.left_ee_body_idx].cpu().numpy() - base_lin_acc_w,
                base_quat_w,
            )
            left_ee_ang_acc = self._vector_world_to_base(
                body_ang_acc[h.left_ee_body_idx].cpu().numpy() - base_ang_acc_w,
                base_quat_w,
            )
        else:
            left_ee_pos = np.zeros(3, dtype=np.float64)
            left_ee_quat = np.array([1, 0, 0, 0], dtype=np.float64)
            left_ee_lin_vel = np.zeros(3, dtype=np.float64)
            left_ee_ang_vel = np.zeros(3, dtype=np.float64)
            left_ee_lin_acc = np.zeros(3, dtype=np.float64)
            left_ee_ang_acc = np.zeros(3, dtype=np.float64)

        if h.right_ee_body_idx >= 0:
            right_ee_pos = self._vector_world_to_base(
                body_pos[h.right_ee_body_idx].cpu().numpy() - base_pos_w,
                base_quat_w,
            )
            right_ee_quat = self._orientation_world_to_base(
                body_quat[h.right_ee_body_idx].cpu().numpy(),
                base_quat_w,
            )
            right_ee_lin_vel = self._vector_world_to_base(
                body_lin_vel[h.right_ee_body_idx].cpu().numpy() - base_lin_vel_w,
                base_quat_w,
            )
            right_ee_ang_vel = self._vector_world_to_base(
                body_ang_vel[h.right_ee_body_idx].cpu().numpy() - base_ang_vel_w,
                base_quat_w,
            )
            right_ee_lin_acc = self._vector_world_to_base(
                body_lin_acc[h.right_ee_body_idx].cpu().numpy() - base_lin_acc_w,
                base_quat_w,
            )
            right_ee_ang_acc = self._vector_world_to_base(
                body_ang_acc[h.right_ee_body_idx].cpu().numpy() - base_ang_acc_w,
                base_quat_w,
            )
        else:
            right_ee_pos = np.zeros(3, dtype=np.float64)
            right_ee_quat = np.array([1, 0, 0, 0], dtype=np.float64)
            right_ee_lin_vel = np.zeros(3, dtype=np.float64)
            right_ee_ang_vel = np.zeros(3, dtype=np.float64)
            right_ee_lin_acc = np.zeros(3, dtype=np.float64)
            right_ee_ang_acc = np.zeros(3, dtype=np.float64)

        # ── 读取门关节状态 ────────────────────────────────────
        door_joint_pos = 0.0
        door_joint_vel = 0.0
        door_pose = np.zeros(7, dtype=np.float64)
        door_pose[3] = 1.0  # pos(3) + quat(4)

        if h.door_view is not None and hasattr(h.door_view, 'data'):
            door_data = h.door_view.data
            if door_data.joint_pos is not None and door_data.joint_pos.numel() > 0:
                door_joint_pos = float(door_data.joint_pos[0, 0].cpu())
                door_joint_vel = float(door_data.joint_vel[0, 0].cpu())
            # 门体根位姿
            if door_data.root_pos_w is not None:
                door_root_pos = door_data.root_pos_w[0].cpu().numpy()
                door_root_quat = door_data.root_quat_w[0].cpu().numpy()
                door_pose = self._pose_world_to_base(
                    position_world=door_root_pos,
                    orientation_world=door_root_quat,
                    base_position_world=base_pos_w,
                    base_orientation_world=base_quat_w,
                )

        # ── 读取杯体状态 ──────────────────────────────────────
        cup_position = None
        cup_pose = np.zeros(7, dtype=np.float64)
        cup_pose[3] = 1.0
        cup_lin_vel = np.zeros(3, dtype=np.float64)
        cup_ang_vel = np.zeros(3, dtype=np.float64)

        if (self._left_occupied or self._right_occupied) \
                and h.cup_view is not None and hasattr(h.cup_view, 'data'):
            cup_data = h.cup_view.data
            if cup_data.root_pos_w is not None:
                cup_pos_t = cup_data.root_pos_w[0].cpu().numpy()
                cup_quat_t = cup_data.root_quat_w[0].cpu().numpy()
                cup_position = self._vector_world_to_base(
                    cup_pos_t - base_pos_w,
                    base_quat_w,
                )
                cup_pose = self._pose_world_to_base(
                    position_world=cup_pos_t,
                    orientation_world=cup_quat_t,
                    base_position_world=base_pos_w,
                    base_orientation_world=base_quat_w,
                )
            if cup_data.root_lin_vel_w is not None:
                cup_lin_vel = self._vector_world_to_base(
                    cup_data.root_lin_vel_w[0].cpu().numpy() - base_lin_vel_w,
                    base_quat_w,
                )
            if cup_data.root_ang_vel_w is not None:
                cup_ang_vel = self._vector_world_to_base(
                    cup_data.root_ang_vel_w[0].cpu().numpy() - base_ang_vel_w,
                    base_quat_w,
                )

        # ── 组装状态字典 ──────────────────────────────────────
        state: dict[str, Any] = {
            # 关节状态（左右臂各 6 维）
            "left_joint_positions": left_q.astype(np.float64),
            "left_joint_velocities": left_dq.astype(np.float64),
            "left_joint_torques": left_tau.astype(np.float64),
            "right_joint_positions": right_q.astype(np.float64),
            "right_joint_velocities": right_dq.astype(np.float64),
            "right_joint_torques": right_tau.astype(np.float64),
            # 左臂末端
            "left_ee_position": left_ee_pos.astype(np.float64),
            "left_ee_orientation": left_ee_quat.astype(np.float64),
            "left_ee_linear_velocity": left_ee_lin_vel.astype(np.float64),
            "left_ee_angular_velocity": left_ee_ang_vel.astype(np.float64),
            "left_ee_linear_acceleration": left_ee_lin_acc.astype(np.float64),
            "left_ee_angular_acceleration": left_ee_ang_acc.astype(np.float64),
            # 右臂末端
            "right_ee_position": right_ee_pos.astype(np.float64),
            "right_ee_orientation": right_ee_quat.astype(np.float64),
            "right_ee_linear_velocity": right_ee_lin_vel.astype(np.float64),
            "right_ee_angular_velocity": right_ee_ang_vel.astype(np.float64),
            "right_ee_linear_acceleration": right_ee_lin_acc.astype(np.float64),
            "right_ee_angular_acceleration": right_ee_ang_acc.astype(np.float64),
            # 门状态
            "door_joint_pos": door_joint_pos,
            "door_joint_vel": door_joint_vel,
            "door_pose": door_pose,
            # 杯体状态
            "cup_position": cup_position,
            "cup_pose": cup_pose,
            "cup_linear_vel": cup_lin_vel,
            "cup_angular_vel": cup_ang_vel,
            # 感知视觉 latent（来自 door_perception，步进时由外部注入或延迟加载）
            "door_embedding": None,
        }

        self._attach_stability_proxies(state)
        return state

    def _get_zero_state(self, n: int) -> dict[str, Any]:
        """返回全零占位状态（无 Isaac Lab 时）。"""
        state: dict[str, Any] = {
            "left_joint_positions": np.zeros(n, dtype=np.float64),
            "left_joint_velocities": np.zeros(n, dtype=np.float64),
            "left_joint_torques": np.zeros(n, dtype=np.float64),
            "right_joint_positions": np.zeros(n, dtype=np.float64),
            "right_joint_velocities": np.zeros(n, dtype=np.float64),
            "right_joint_torques": np.zeros(n, dtype=np.float64),
            "left_ee_position": np.zeros(3, dtype=np.float64),
            "left_ee_orientation": np.array([1, 0, 0, 0], dtype=np.float64),
            "left_ee_linear_velocity": np.zeros(3, dtype=np.float64),
            "left_ee_angular_velocity": np.zeros(3, dtype=np.float64),
            "left_ee_linear_acceleration": np.zeros(3, dtype=np.float64),
            "left_ee_angular_acceleration": np.zeros(3, dtype=np.float64),
            "right_ee_position": np.zeros(3, dtype=np.float64),
            "right_ee_orientation": np.array([1, 0, 0, 0], dtype=np.float64),
            "right_ee_linear_velocity": np.zeros(3, dtype=np.float64),
            "right_ee_angular_velocity": np.zeros(3, dtype=np.float64),
            "right_ee_linear_acceleration": np.zeros(3, dtype=np.float64),
            "right_ee_angular_acceleration": np.zeros(3, dtype=np.float64),
            "door_joint_pos": 0.0,
            "door_joint_vel": 0.0,
            "door_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "cup_position": None,
            "cup_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "cup_linear_vel": np.zeros(3, dtype=np.float64),
            "cup_angular_vel": np.zeros(3, dtype=np.float64),
            "door_embedding": None,
        }
        if self._left_occupied or self._right_occupied:
            state["cup_position"] = np.zeros(3, dtype=np.float64)
        self._attach_stability_proxies(state)
        return state

    # ═══════════════════════════════════════════════════════════════════
    # 内部方法 — 观测构建
    # ═══════════════════════════════════════════════════════════════════

    def _build_actor_obs(
        self,
        state: dict[str, Any],
        action_taken: np.ndarray | None,
    ) -> dict:
        """调用 ActorObsBuilder 构建 actor 观测。"""
        return self._actor_obs_builder.build(
            left_joint_positions=state["left_joint_positions"],
            left_joint_velocities=state["left_joint_velocities"],
            right_joint_positions=state["right_joint_positions"],
            right_joint_velocities=state["right_joint_velocities"],
            left_joint_torques=state["left_joint_torques"],
            right_joint_torques=state["right_joint_torques"],
            left_ee_position=state["left_ee_position"],
            left_ee_orientation=state["left_ee_orientation"],
            left_ee_linear_velocity=state["left_ee_linear_velocity"],
            left_ee_angular_velocity=state["left_ee_angular_velocity"],
            left_ee_linear_acceleration=state["left_ee_linear_acceleration"],
            left_ee_angular_acceleration=state["left_ee_angular_acceleration"],
            left_stability_proxy=state["left_stability_proxy"],
            right_ee_position=state["right_ee_position"],
            right_ee_orientation=state["right_ee_orientation"],
            right_ee_linear_velocity=state["right_ee_linear_velocity"],
            right_ee_angular_velocity=state["right_ee_angular_velocity"],
            right_ee_linear_acceleration=state["right_ee_linear_acceleration"],
            right_ee_angular_acceleration=state["right_ee_angular_acceleration"],
            right_stability_proxy=state["right_stability_proxy"],
            left_occupied=float(self._left_occupied),
            right_occupied=float(self._right_occupied),
            door_embedding=state.get("door_embedding"),
            action_taken=action_taken,
        )

    def _build_critic_obs(
        self,
        actor_obs: dict,
        state: dict[str, Any],
        *,
        cup_dropped: float = 0.0,
    ) -> dict:
        """调用 CriticObsBuilder 追加 privileged 信息。"""
        return CriticObsBuilder.build(
            actor_obs=actor_obs,
            door_pose=state["door_pose"],
            door_joint_pos=state["door_joint_pos"],
            door_joint_vel=state["door_joint_vel"],
            cup_mass=self._domain_params.get("cup_mass", 0.0),
            door_mass=self._domain_params.get("door_mass", 0.0),
            door_damping=self._domain_params.get("door_damping", 0.0),
            base_pos=self._domain_params.get("base_pos"),
            cup_dropped=cup_dropped,
        )

    def get_visual_observation(self) -> dict[str, Any] | None:
        """返回当前环境可用于视觉编码的原始观测。

        当 Isaac 相机不可用或当前帧读取失败时返回 ``None``，
        训练侧 ``PerceptionRuntime`` 会退化为缓存零向量。
        """
        handles = self._handles
        if handles is None:
            return None

        camera = getattr(handles, "camera_view", None)
        if camera is None:
            return None

        try:
            rgba = camera.get_rgba()
            frame = camera.get_current_frame()
        except Exception:
            return None

        if rgba is None or np.size(rgba) == 0:
            return None
        if not frame or "distance_to_image_plane" not in frame:
            return None

        depth = frame["distance_to_image_plane"]
        if depth is None or np.size(depth) == 0:
            return None

        extrinsic = self._camera_to_base_extrinsic()
        if extrinsic is None:
            extrinsic = np.eye(4, dtype=np.float64)

        return {
            "rgb": np.asarray(rgba, dtype=np.uint8)[..., :3].copy(),
            "depth": np.asarray(depth, dtype=np.float32).copy(),
            "extrinsic": extrinsic,
        }

    def _camera_to_base_extrinsic(self) -> np.ndarray | None:
        """计算相机坐标系到 ``base_link`` 的变换矩阵。"""
        if not _HAS_ISAAC_LAB or self._handles is None:
            return None

        camera = getattr(self._handles, "camera_view", None)
        if camera is None:
            return None

        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None

            cam_prim_path = getattr(camera, "prim_path", None)
            if not cam_prim_path:
                return None

            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            if not cam_prim.IsValid():
                return None

            cam_to_world = np.array(
                UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0.0),
                dtype=np.float64,
            )

            h = self._handles
            robot = h.robot_view
            if robot is None or h.base_body_idx < 0:
                return None

            base_pos_w = robot.data.body_pos_w[0][h.base_body_idx].cpu().numpy()
            base_quat_w = robot.data.body_quat_w[0][h.base_body_idx].cpu().numpy()

            base_to_world = np.eye(4, dtype=np.float64)
            base_to_world[:3, :3] = _quat_to_rotation_matrix(base_quat_w)
            base_to_world[:3, 3] = base_pos_w

            world_to_base = np.linalg.inv(base_to_world)
            return world_to_base @ cam_to_world
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    # 内部方法 — 奖励计算
    # ═══════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        *,
        state: dict[str, Any],
        task_status: TaskStatus,
        contact_summary: ContactSummary,
        raw_action: np.ndarray,
        applied_action: np.ndarray,
    ) -> tuple[float, bool, dict[str, float]]:
        """调用 RewardManager 计算完整奖励。

        将环境层采集的物理真值映射为 RewardManager.step() 所需的参数。
        """
        # 拼接双臂完整力矩向量
        full_torques = applied_action
        prev_torques = self._prev_action

        return self._reward_manager.step(
            theta_t=task_status.door_angle,
            theta_prev=task_status.door_angle_prev,
            left_stability_proxy=state["left_stability_proxy"],
            right_stability_proxy=state["right_stability_proxy"],
            left_occupied=self._left_occupied,
            right_occupied=self._right_occupied,
            torques=full_torques,
            prev_torques=prev_torques,
            self_collision=contact_summary.self_collision,
            policy_torques=raw_action,
            torque_limits=self._effort_limits,
            cup_dropped=contact_summary.cup_dropped,
        )

    def _attach_stability_proxies(self, state: dict[str, Any]) -> None:
        """基于环境侧原生加速度为当前状态补齐共享稳定性 proxy。"""
        state["left_stability_proxy"] = build_stability_proxy(
            quat_ee=state["left_ee_orientation"],
            linear_velocity=state["left_ee_linear_velocity"],
            angular_velocity=state["left_ee_angular_velocity"],
            linear_acceleration=state["left_ee_linear_acceleration"],
            angular_acceleration=state["left_ee_angular_acceleration"],
            dt=self.cfg.control_dt,
            state=self._left_stability_state,
            acc_history_length=self.cfg.acc_history_length,
        ).to_dict()
        state["right_stability_proxy"] = build_stability_proxy(
            quat_ee=state["right_ee_orientation"],
            linear_velocity=state["right_ee_linear_velocity"],
            angular_velocity=state["right_ee_angular_velocity"],
            linear_acceleration=state["right_ee_linear_acceleration"],
            angular_acceleration=state["right_ee_angular_acceleration"],
            dt=self.cfg.control_dt,
            state=self._right_stability_state,
            acc_history_length=self.cfg.acc_history_length,
        ).to_dict()

    @staticmethod
    def _extract_robot_body_tensor(
        robot_data: Any,
        *,
        preferred: tuple[str, ...],
    ):
        """优先读取 Isaac Lab 原生 body 张量，缺失时显式报错。"""
        for attr_name in preferred:
            tensor = getattr(robot_data, attr_name, None)
            if tensor is not None:
                return tensor
        available = ", ".join(sorted(name for name in dir(robot_data) if "acc" in name))
        raise RuntimeError(
            "当前 Isaac Lab 数据对象未暴露所需的原生加速度张量。"
            f" 期望字段之一: {preferred}; 可见加速度相关字段: {available or '无'}"
        )

    @staticmethod
    def _vector_world_to_base(
        vector_world: np.ndarray,
        base_orientation_world: np.ndarray,
    ) -> np.ndarray:
        """将世界系向量旋转到 base_link 坐标系。"""
        rotation_world_from_base = _quat_to_rotation_matrix(base_orientation_world)
        return rotation_world_from_base.T @ np.asarray(vector_world, dtype=np.float64)

    @staticmethod
    def _orientation_world_to_base(
        orientation_world: np.ndarray,
        base_orientation_world: np.ndarray,
    ) -> np.ndarray:
        """将世界系四元数转换为 base_link 相对四元数。"""
        relative = _quat_multiply(
            _quat_conjugate(base_orientation_world),
            np.asarray(orientation_world, dtype=np.float64),
        )
        return _quat_normalize(relative)

    @classmethod
    def _pose_world_to_base(
        cls,
        *,
        position_world: np.ndarray,
        orientation_world: np.ndarray,
        base_position_world: np.ndarray,
        base_orientation_world: np.ndarray,
    ) -> np.ndarray:
        """将世界系 pose 转为 base_link 系，并按 pos(3)+quat(4) 拼接。"""
        position_base = cls._vector_world_to_base(
            np.asarray(position_world, dtype=np.float64) - np.asarray(base_position_world, dtype=np.float64),
            base_orientation_world,
        )
        orientation_base = cls._orientation_world_to_base(
            orientation_world,
            base_orientation_world,
        )
        return np.concatenate([position_base, orientation_base])
