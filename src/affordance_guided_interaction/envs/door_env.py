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
from affordance_guided_interaction.observations.critic_obs_builder import (
    CriticObsBuilder,
)
from affordance_guided_interaction.rewards.reward_manager import RewardManager

# Isaac Lab 条件导入
if _HAS_ISAAC_LAB:
    import torch


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
            door_angle_target=self.cfg.door_angle_target,
            max_episode_steps=self.cfg.max_episode_steps,
        )
        self._actor_obs_builder = ActorObsBuilder(
            action_history_length=self.cfg.action_history_length,
            acc_history_length=self.cfg.acc_history_length,
            dt=self.cfg.control_dt,
        )
        self._reward_manager = RewardManager(self.cfg.reward_cfg)

        # ── 回合级状态 ────────────────────────────────────────
        self._handles: SceneHandles | None = None
        self._domain_params: dict[str, Any] = {}
        self._left_occupied: bool = False
        self._right_occupied: bool = False
        self._prev_action: np.ndarray = np.zeros(
            self.cfg.total_joints, dtype=np.float64
        )
        self._effort_limits: np.ndarray = self.cfg.get_effort_limits()

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
        self._prev_action = np.zeros(self.cfg.total_joints, dtype=np.float64)

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
        critic_obs = self._build_critic_obs(actor_obs, state)

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
        action = np.asarray(action, dtype=np.float64)

        # ── 1. 力矩安全截断 ────────────────────────────────────
        clipped_action = np.clip(action, -self._effort_limits, self._effort_limits)

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
        critic_obs = self._build_critic_obs(actor_obs, state)

        # ── 7. 计算奖励 ───────────────────────────────────────
        reward, should_terminate, reward_info = self._compute_reward(
            state=state,
            task_status=task_status,
            contact_summary=contact_summary,
            action=clipped_action,
        )

        # ── 8. 最终 done 判定 ─────────────────────────────────
        # TaskManager 和 RewardManager 都可触发终止
        done = task_status.done or should_terminate

        # ── 9. 组装 info ──────────────────────────────────────
        info: dict[str, Any] = {
            "success": task_status.success,
            "termination_reason": task_status.reason.name,
            "step_count": task_status.step_count,
            "door_angle": task_status.door_angle,
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
        body_lin_vel = robot.data.body_lin_vel_w[0]   # (num_bodies, 3)
        body_ang_vel = robot.data.body_ang_vel_w[0]   # (num_bodies, 3)

        if h.left_ee_body_idx >= 0:
            left_ee_pos = body_pos[h.left_ee_body_idx].cpu().numpy()
            left_ee_quat = body_quat[h.left_ee_body_idx].cpu().numpy()
            left_ee_lin_vel = body_lin_vel[h.left_ee_body_idx].cpu().numpy()
            left_ee_ang_vel = body_ang_vel[h.left_ee_body_idx].cpu().numpy()
        else:
            left_ee_pos = np.zeros(3, dtype=np.float64)
            left_ee_quat = np.array([1, 0, 0, 0], dtype=np.float64)
            left_ee_lin_vel = np.zeros(3, dtype=np.float64)
            left_ee_ang_vel = np.zeros(3, dtype=np.float64)

        if h.right_ee_body_idx >= 0:
            right_ee_pos = body_pos[h.right_ee_body_idx].cpu().numpy()
            right_ee_quat = body_quat[h.right_ee_body_idx].cpu().numpy()
            right_ee_lin_vel = body_lin_vel[h.right_ee_body_idx].cpu().numpy()
            right_ee_ang_vel = body_ang_vel[h.right_ee_body_idx].cpu().numpy()
        else:
            right_ee_pos = np.zeros(3, dtype=np.float64)
            right_ee_quat = np.array([1, 0, 0, 0], dtype=np.float64)
            right_ee_lin_vel = np.zeros(3, dtype=np.float64)
            right_ee_ang_vel = np.zeros(3, dtype=np.float64)

        # ── 读取门关节状态 ────────────────────────────────────
        door_joint_pos = 0.0
        door_joint_vel = 0.0
        door_pose = np.zeros(7, dtype=np.float64)
        door_pose[0] = 1.0  # 单位四元数

        if h.door_view is not None and hasattr(h.door_view, 'data'):
            door_data = h.door_view.data
            if door_data.joint_pos is not None and door_data.joint_pos.numel() > 0:
                door_joint_pos = float(door_data.joint_pos[0, 0].cpu())
                door_joint_vel = float(door_data.joint_vel[0, 0].cpu())
            # 门体根位姿
            if door_data.root_pos_w is not None:
                door_root_pos = door_data.root_pos_w[0].cpu().numpy()
                door_root_quat = door_data.root_quat_w[0].cpu().numpy()
                door_pose = np.concatenate([door_root_quat, door_root_pos])

        # ── 读取杯体状态 ──────────────────────────────────────
        cup_position = None
        cup_pose = np.zeros(7, dtype=np.float64)
        cup_pose[0] = 1.0
        cup_lin_vel = np.zeros(3, dtype=np.float64)
        cup_ang_vel = np.zeros(3, dtype=np.float64)

        if (self._left_occupied or self._right_occupied) \
                and h.cup_view is not None and hasattr(h.cup_view, 'data'):
            cup_data = h.cup_view.data
            if cup_data.root_pos_w is not None:
                cup_pos_t = cup_data.root_pos_w[0].cpu().numpy()
                cup_quat_t = cup_data.root_quat_w[0].cpu().numpy()
                cup_position = cup_pos_t
                cup_pose = np.concatenate([cup_quat_t, cup_pos_t])
            if cup_data.root_lin_vel_w is not None:
                cup_lin_vel = cup_data.root_lin_vel_w[0].cpu().numpy()
            if cup_data.root_ang_vel_w is not None:
                cup_ang_vel = cup_data.root_ang_vel_w[0].cpu().numpy()

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
            # 右臂末端
            "right_ee_position": right_ee_pos.astype(np.float64),
            "right_ee_orientation": right_ee_quat.astype(np.float64),
            "right_ee_linear_velocity": right_ee_lin_vel.astype(np.float64),
            "right_ee_angular_velocity": right_ee_ang_vel.astype(np.float64),
            # 门状态
            "door_joint_pos": door_joint_pos,
            "door_joint_vel": door_joint_vel,
            "door_pose": door_pose,
            # 杯体状态
            "cup_position": cup_position,
            "cup_pose": cup_pose,
            "cup_linear_vel": cup_lin_vel,
            "cup_angular_vel": cup_ang_vel,
            # 门点云 embedding（来自 door_perception，步进时由外部注入或延迟加载）
            "door_embedding": None,
        }

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
            "right_ee_position": np.zeros(3, dtype=np.float64),
            "right_ee_orientation": np.array([1, 0, 0, 0], dtype=np.float64),
            "right_ee_linear_velocity": np.zeros(3, dtype=np.float64),
            "right_ee_angular_velocity": np.zeros(3, dtype=np.float64),
            "door_joint_pos": 0.0,
            "door_joint_vel": 0.0,
            "door_pose": np.zeros(7, dtype=np.float64),
            "cup_position": None,
            "cup_pose": np.zeros(7, dtype=np.float64),
            "cup_linear_vel": np.zeros(3, dtype=np.float64),
            "cup_angular_vel": np.zeros(3, dtype=np.float64),
            "door_embedding": None,
        }
        if self._left_occupied or self._right_occupied:
            state["cup_position"] = np.zeros(3, dtype=np.float64)
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
            right_ee_position=state["right_ee_position"],
            right_ee_orientation=state["right_ee_orientation"],
            right_ee_linear_velocity=state["right_ee_linear_velocity"],
            right_ee_angular_velocity=state["right_ee_angular_velocity"],
            left_occupied=float(self._left_occupied),
            right_occupied=float(self._right_occupied),
            door_embedding=state.get("door_embedding"),
            action_taken=action_taken,
        )

    def _build_critic_obs(
        self,
        actor_obs: dict,
        state: dict[str, Any],
    ) -> dict:
        """调用 CriticObsBuilder 追加 privileged 信息。"""
        return CriticObsBuilder.build(
            actor_obs=actor_obs,
            door_pose=state["door_pose"],
            door_joint_pos=state["door_joint_pos"],
            door_joint_vel=state["door_joint_vel"],
            cup_pose=state["cup_pose"],
            cup_linear_vel=state["cup_linear_vel"],
            cup_angular_vel=state["cup_angular_vel"],
            cup_mass=self._domain_params.get("cup_mass", 0.0),
            door_mass=self._domain_params.get("door_mass", 0.0),
            door_damping=self._domain_params.get("door_damping", 0.0),
            base_pos=self._domain_params.get("base_pos"),
        )

    # ═══════════════════════════════════════════════════════════════════
    # 内部方法 — 奖励计算
    # ═══════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        *,
        state: dict[str, Any],
        task_status: TaskStatus,
        contact_summary: ContactSummary,
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, float]]:
        """调用 RewardManager 计算完整奖励。

        将环境层采集的物理真值映射为 RewardManager.step() 所需的参数。
        """
        # 为稳定性 proxy 准备简化输入
        left_stab_proxy = None
        right_stab_proxy = None

        if self._left_occupied:
            left_stab_proxy = {
                "lin_acc": state["left_ee_linear_velocity"],
                "ang_acc": state["left_ee_angular_velocity"],
                "tilt_xy": np.zeros(2, dtype=np.float64),
            }
        if self._right_occupied:
            right_stab_proxy = {
                "lin_acc": state["right_ee_linear_velocity"],
                "ang_acc": state["right_ee_angular_velocity"],
                "tilt_xy": np.zeros(2, dtype=np.float64),
            }

        # 拼接双臂完整力矩向量
        full_torques = action
        prev_torques = self._prev_action

        return self._reward_manager.step(
            theta_t=task_status.door_angle,
            theta_prev=task_status.door_angle_prev,
            left_stability_proxy=left_stab_proxy,
            right_stability_proxy=right_stab_proxy,
            left_occupied=self._left_occupied,
            right_occupied=self._right_occupied,
            torques=full_torques,
            prev_torques=prev_torques,
            contact_forces=contact_summary.link_forces,
            self_collision=contact_summary.self_collision,
            cup_dropped=contact_summary.cup_dropped,
        )
