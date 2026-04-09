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
from isaaclab.sensors import ContactSensor

from .door_push_env_cfg import (
    DoorPushEnvCfg,
    # 名称常量
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
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
    POSE_SETTLE_STEPS,
    POST_SPAWN_SETTLE_STEPS,
    GRIPPER_CLOSE_STEPS,
    POST_CLOSE_SETTLE_STEPS,
    POST_REMOVE_SETTLE_STEPS,
    TRAY_SIZE_XYZ,
)
from .batch_math import (
    batch_quat_conjugate,
    batch_quat_from_yaw,
    batch_quat_multiply,
    batch_quat_normalize,
    batch_quat_to_rotation_matrix,
    batch_orientation_world_to_base,
    batch_pose_world_to_base,
    batch_rotate_relative_by_yaw,
    batch_vector_world_to_base,
    batch_yaw_from_quat,
    sample_base_poses,
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
# DoorLeaf 局部坐标系下的推门侧法向量 (+X)
_DOOR_NORMAL_LOCAL = (1.0, 0.0, 0.0)

# 观测维度
# actor: proprio(48) + ee(38) + context(2) + stability(2) + door_geometry(6) = 96
# critic: actor_obs(96) + privileged(13) = 109
_ACTOR_OBS_DIM = 96
_CRITIC_OBS_DIM = 109
_DOOR_GEOMETRY_DIM = 6
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
        self._base_body_idx = robot.find_bodies([BASE_LINK_NAME])[0][0]
        self._left_ee_body_idx = robot.find_bodies([LEFT_EE_LINK_NAME])[0][0]
        self._right_ee_body_idx = robot.find_bodies([RIGHT_EE_LINK_NAME])[0][0]

        # 门铰链（通常只有 1 个关节）
        self._door_hinge_ids, _ = door.find_joints(".*")
        self._door_panel_body_idx = door.find_bodies([DOOR_LEAF_BODY_NAME])[0][0]

        # ── Per-env 持久状态 tensor ──────────────────────────────
        N = self.num_envs
        dev = self.device

        self._prev_action = torch.zeros(N, 12, device=dev)
        self._prev_prev_action = torch.zeros(N, 12, device=dev)
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

        # 上一帧的 EE 线速度/角速度，用于计算加速度
        self._prev_left_ee_lin_vel = torch.zeros(N, 3, device=dev)
        self._prev_left_ee_ang_vel = torch.zeros(N, 3, device=dev)
        self._prev_right_ee_lin_vel = torch.zeros(N, 3, device=dev)
        self._prev_right_ee_ang_vel = torch.zeros(N, 3, device=dev)

        self._episode_reset_fn = None

        nan = float("nan")
        self._pending_cup_mass = torch.full((N,), nan, device=dev)
        self._pending_door_mass = torch.full((N,), nan, device=dev)
        self._pending_door_damping = torch.full((N,), nan, device=dev)
        self._pending_base_pos = torch.full((N, 3), nan, device=dev)
        self._pending_base_yaw = torch.full((N,), nan, device=dev)

        # 预计算控制 dt
        self._control_dt = self.physics_dt * self.cfg.decimation

        # ── 奖励计算用缓存（由 _get_observations 填充）───────────
        self._cached_left_ee_la = torch.zeros(N, 3, device=dev)
        self._cached_left_ee_aa = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_la = torch.zeros(N, 3, device=dev)
        self._cached_right_ee_aa = torch.zeros(N, 3, device=dev)
        self._cached_left_tilt_xy = torch.zeros(N, 2, device=dev)
        self._cached_right_tilt_xy = torch.zeros(N, 2, device=dev)
        self._cached_raw_action = torch.zeros(N, 12, device=dev)
        # L12: 每步在 _get_observations() 中缓存，供 _get_rewards/_get_dones 共用
        self._cached_cup_dropped = torch.zeros(N, dtype=torch.bool, device=dev)

        # 自碰撞检测分组索引（一次性解析）
        self._self_collision_groups = self._resolve_collision_groups(robot)

    # ═══════════════════════════════════════════════════════════════════
    # 场景装配（由 Cloner 调用）
    # ═══════════════════════════════════════════════════════════════════

    def _setup_scene(self) -> None:
        """场景实体已由 ``DoorPushSceneCfg`` 自动注册，此处无需额外装配。"""
        return

    # ═══════════════════════════════════════════════════════════════════
    # 动作执行
    # ═══════════════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: Tensor) -> None:
        """将策略输出的力矩写入关节 — 批量操作所有 env。"""
        # 缓存原始动作（clip 前），用于力矩超限惩罚 §6.4
        self._cached_raw_action = actions.clone()

        # 力矩裁剪
        clipped = torch.clamp(actions, -self.cfg.effort_limit, self.cfg.effort_limit)

        # 注入步级动作噪声 (training only)
        if self.cfg.action_noise_std > 0:
            noise = torch.randn_like(clipped) * self.cfg.action_noise_std
            clipped = torch.clamp(
                clipped + noise,
                -self.cfg.effort_limit,
                self.cfg.effort_limit,
            )

        # 构建全关节力矩向量（策略只控制 12 个臂关节）
        robot: Articulation = self.scene["robot"]
        efforts = torch.zeros(
            self.num_envs, robot.num_joints, device=self.device
        )
        efforts[:, self._arm_joint_ids] = clipped
        robot.set_joint_effort_target(efforts)

        # 保存动作供下一步 obs 使用
        self._prev_prev_action = self._prev_action.clone()
        self._prev_action = clipped.clone()

    def _apply_action(self) -> None:
        """动作已在 `_pre_physics_step()` 中写入 articulation 缓冲，这里无需重复处理。"""
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
        all_tau = robot.data.applied_torque[:, self._arm_joint_ids] \
            if hasattr(robot.data, "applied_torque") \
            else torch.zeros_like(all_q)

        # ── 读取 body 状态（世界系）─────────────────────────────
        body_pos_w = robot.data.body_pos_w     # (N, B, 3)
        body_quat_w = robot.data.body_quat_w   # (N, B, 4)
        body_lin_vel_w = robot.data.body_lin_vel_w
        body_ang_vel_w = robot.data.body_ang_vel_w

        base_pos = body_pos_w[:, self._base_body_idx]     # (N, 3)
        base_quat = body_quat_w[:, self._base_body_idx]   # (N, 4)
        base_lv = body_lin_vel_w[:, self._base_body_idx]   # (N, 3)
        base_av = body_ang_vel_w[:, self._base_body_idx]   # (N, 3)

        # ── 左臂 EE（base_link 相对系）─────────────────────────
        left_ee_pos_base, left_ee_quat_base, left_ee_lv_base, left_ee_av_base = \
            self._ee_world_to_base(
                body_pos_w[:, self._left_ee_body_idx],
                body_quat_w[:, self._left_ee_body_idx],
                body_lin_vel_w[:, self._left_ee_body_idx],
                body_ang_vel_w[:, self._left_ee_body_idx],
                base_pos, base_quat, base_lv, base_av,
            )

        # ── 右臂 EE（base_link 相对系）─────────────────────────
        right_ee_pos_base, right_ee_quat_base, right_ee_lv_base, right_ee_av_base = \
            self._ee_world_to_base(
                body_pos_w[:, self._right_ee_body_idx],
                body_quat_w[:, self._right_ee_body_idx],
                body_lin_vel_w[:, self._right_ee_body_idx],
                body_ang_vel_w[:, self._right_ee_body_idx],
                base_pos, base_quat, base_lv, base_av,
            )

        # ── 数值微分计算加速度 ──────────────────────────────────
        # 注：base_pos/yaw 仅在 episode reset 时更新，episode 内保持不变，
        # 因此前后两帧速度均在同一 base frame 中，差分结果正确。
        inv_dt = 1.0 / max(self._control_dt, 1e-6)
        left_ee_la = (left_ee_lv_base - self._prev_left_ee_lin_vel) * inv_dt
        left_ee_aa = (left_ee_av_base - self._prev_left_ee_ang_vel) * inv_dt
        right_ee_la = (right_ee_lv_base - self._prev_right_ee_lin_vel) * inv_dt
        right_ee_aa = (right_ee_av_base - self._prev_right_ee_ang_vel) * inv_dt

        # 更新速度缓存
        self._prev_left_ee_lin_vel = left_ee_lv_base.clone()
        self._prev_left_ee_ang_vel = left_ee_av_base.clone()
        self._prev_right_ee_lin_vel = right_ee_lv_base.clone()
        self._prev_right_ee_ang_vel = right_ee_av_base.clone()

        # ── 稳定性 proxy: tilt ──────────────────────────────────
        left_tilt, left_tilt_xy = self._compute_tilt(left_ee_quat_base)   # (N, 1), (N, 2)
        right_tilt, right_tilt_xy = self._compute_tilt(right_ee_quat_base)  # (N, 1), (N, 2)

        # ── 缓存加速度和 tilt_xy，供 _get_rewards 使用 ──────────
        self._cached_left_ee_la = left_ee_la.clone()
        self._cached_left_ee_aa = left_ee_aa.clone()
        self._cached_right_ee_la = right_ee_la.clone()
        self._cached_right_ee_aa = right_ee_aa.clone()
        self._cached_left_tilt_xy = left_tilt_xy.clone()
        self._cached_right_tilt_xy = right_tilt_xy.clone()

        # ── 上下文 ──────────────────────────────────────────────
        left_occ = self._left_occupied.float().unsqueeze(-1)   # (N, 1)
        right_occ = self._right_occupied.float().unsqueeze(-1)  # (N, 1)

        # ── 门几何观测（base_link 系）─────────────────────────────
        door_leaf_pos_w = door.data.body_pos_w[:, self._door_panel_body_idx]   # (N, 3)
        door_leaf_quat_w = door.data.body_quat_w[:, self._door_panel_body_idx]  # (N, 4)

        center_offset_local = torch.tensor(
            _DOOR_CENTER_OFFSET_LOCAL, device=self.device, dtype=torch.float32
        )
        normal_local = torch.tensor(
            _DOOR_NORMAL_LOCAL, device=self.device, dtype=torch.float32
        )

        R_world_from_leaf = batch_quat_to_rotation_matrix(door_leaf_quat_w)  # (N, 3, 3)
        door_center_w = door_leaf_pos_w + torch.bmm(
            R_world_from_leaf,
            center_offset_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)  # (N, 3)
        door_normal_w = torch.bmm(
            R_world_from_leaf,
            normal_local.view(1, 3, 1).expand(self.num_envs, -1, -1),
        ).squeeze(-1)  # (N, 3)

        door_center_base = batch_vector_world_to_base(
            door_center_w - base_pos, base_quat
        )  # (N, 3)
        door_normal_base = batch_vector_world_to_base(
            door_normal_w, base_quat
        )  # (N, 3)
        door_geometry = torch.cat([door_center_base, door_normal_base], dim=-1)  # (N, 6)

        # ── Actor obs (含噪声) ──────────────────────────────────
        noisy_q = all_q
        noisy_dq = all_dq
        if self.cfg.obs_noise_std > 0:
            noisy_q = all_q + torch.randn_like(all_q) * self.cfg.obs_noise_std
            noisy_dq = all_dq + torch.randn_like(all_dq) * self.cfg.obs_noise_std

        # actor: proprio(48) + ee(38) + context(2) + stability(2) + door_geometry(6) = 96
        actor_obs = torch.cat([
            # proprio: q(12) + dq(12) + tau(12) + prev_action(12) = 48
            noisy_q, noisy_dq, all_tau, self._prev_action,
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
        ], dim=-1)  # (N, 96)

        # ── Critic obs (无噪声 + privileged) ───────────────────
        # 门状态 (base_link 系)
        door_root_pos_w = door.data.root_pos_w    # (N, 3)
        door_root_quat_w = door.data.root_quat_w  # (N, 4)
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
            all_q, all_dq, all_tau, self._prev_action,
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
            # privileged: door_pose(7) + door_joint(2) + domain_params(3) + cup_dropped(1) = 13
            door_pose_base,
            door_joint_pos, door_joint_vel,
            self._cup_mass.unsqueeze(-1),
            self._door_mass.unsqueeze(-1),
            self._door_damping.unsqueeze(-1),
            cup_dropped.float().unsqueeze(-1),
        ], dim=-1)  # (N, 109)

        return {"policy": actor_obs, "critic": critic_obs}

    # ═══════════════════════════════════════════════════════════════════
    # 奖励计算
    # ═══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> Tensor:
        """批量计算完整奖励，并缓存 TensorBoard 用分项信息。"""
        robot: Articulation = self.scene["robot"]
        door: Articulation = self.scene["door"]
        theta = door.data.joint_pos[:, 0]   # (N,)
        theta_prev = self._prev_door_angle

        reward_info: dict[str, Tensor] = {}

        # ══════════════════════════════════════════════════════════════
        # §4 任务奖励：角度增量 + 一次性成功 bonus
        # ══════════════════════════════════════════════════════════════
        delta = theta - theta_prev

        target = self.cfg.success_angle_threshold
        w_below = torch.full_like(theta, self.cfg.rew_w_delta)
        w_above = self.cfg.rew_w_delta * torch.clamp(
            1.0 - self.cfg.rew_k_decay * (theta - target),
            min=self.cfg.rew_alpha,
        )
        weight = torch.where(theta <= target, w_below, w_above)
        r_task_delta = weight * delta

        newly_succeeded = (theta >= target) & ~self._already_succeeded
        r_task_open_bonus = newly_succeeded.float() * self.cfg.rew_w_open
        r_task = r_task_delta + r_task_open_bonus
        self._already_succeeded = self._already_succeeded | newly_succeeded

        reward_info["task"] = r_task
        reward_info["task/delta"] = r_task_delta
        reward_info["task/open_bonus"] = r_task_open_bonus

        # ══════════════════════════════════════════════════════════════
        # §5 稳定性奖励：7 子项 × 双臂（使用缓存的加速度 + tilt_xy）
        # ══════════════════════════════════════════════════════════════
        m_l = self._left_occupied.float()  # (N,)
        m_r = self._right_occupied.float()

        all_tau = self._prev_action  # (N, 12) — 已 clip 后的力矩
        r_stab_left = torch.zeros(self.num_envs, device=self.device)
        r_stab_right = torch.zeros(self.num_envs, device=self.device)

        for side_idx, (side_name, m, la, aa, tilt_xy, tau_slice) in enumerate([
            ("left", m_l, self._cached_left_ee_la, self._cached_left_ee_aa,
             self._cached_left_tilt_xy, all_tau[:, :6]),
            ("right", m_r, self._cached_right_ee_la, self._cached_right_ee_aa,
             self._cached_right_tilt_xy, all_tau[:, 6:]),
        ]):
            prev_tau_slice = (
                self._prev_prev_action[:, :6] if side_idx == 0
                else self._prev_prev_action[:, 6:]
            )

            la_sq = (la * la).sum(-1)
            aa_sq = (aa * aa).sum(-1)
            tilt_sq = (tilt_xy * tilt_xy).sum(-1)
            tau_diff_sq = ((tau_slice - prev_tau_slice) ** 2).sum(-1)
            tau_sq = (tau_slice * tau_slice).sum(-1)

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
                "smooth": m * (-self.cfg.rew_w_smooth * tau_diff_sq),
                "reg": m * (-self.cfg.rew_w_reg * tau_sq),
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
        # §6 安全惩罚：5 子项（正惩罚量）
        # ══════════════════════════════════════════════════════════════
        self_collision = self._compute_batch_self_collision()  # (N,) bool
        r_safe_self_collision = self_collision.float() * self.cfg.rew_beta_self

        joint_pos = robot.data.joint_pos[:, self._arm_joint_ids]  # (N, 12)
        if hasattr(robot.data, "soft_joint_pos_limits"):
            joint_limits = robot.data.soft_joint_pos_limits[0, self._arm_joint_ids]  # (12, 2)
        else:
            raise RuntimeError(
                "robot.data.soft_joint_pos_limits 不可用，"
                "无法获取关节物理限位。请检查 Isaac Lab 版本或 Articulation 配置。"
            )
        center = (joint_limits[:, 0] + joint_limits[:, 1]) / 2.0
        half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2.0
        threshold = self.cfg.rew_mu * half_range
        excess = torch.clamp(torch.abs(joint_pos - center) - threshold, min=0)
        r_safe_joint_limit = self.cfg.rew_beta_limit * (excess ** 2).sum(-1)

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

        torque_excess = torch.clamp(
            torch.abs(self._cached_raw_action) - self.cfg.effort_limit, min=0
        )
        r_safe_torque_limit = self.cfg.rew_beta_torque * (torque_excess ** 2).sum(-1)

        cup_dropped = self._cached_cup_dropped
        r_safe_cup_drop = cup_dropped.float() * self.cfg.rew_w_drop

        r_safe = (
            r_safe_self_collision
            + r_safe_joint_limit
            + r_safe_joint_vel
            + r_safe_torque_limit
            + r_safe_cup_drop
        )

        reward_info["safe"] = r_safe
        reward_info["safe/self_collision"] = r_safe_self_collision
        reward_info["safe/joint_limit"] = r_safe_joint_limit
        reward_info["safe/joint_vel"] = r_safe_joint_vel
        reward_info["safe/torque_limit"] = r_safe_torque_limit
        reward_info["safe/cup_drop"] = r_safe_cup_drop

        r_total = r_task + r_stab_left + r_stab_right - r_safe
        reward_info["total"] = r_total

        self.extras["reward_info"] = {
            key: value.clone() for key, value in reward_info.items()
        }

        # 更新门角度缓存
        self._prev_door_angle = theta.clone()

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
        angle_reached = theta >= self.cfg.door_angle_target

        terminated = cup_dropped | angle_reached
        truncated = self._step_count >= self.max_episode_length

        # C3: 在 auto-reset 之前将 per-env success 写入 extras，
        # 供 DirectRLEnvAdapter 通过 info_dict 消费（reset 后状态已归零无法重读）
        self.extras["success"] = angle_reached & ~cup_dropped

        # D7: 缓存 pre-reset occupancy，auto-reset 中 _episode_reset_fn 会覆写
        # occupancy，adapter 需要读到完成 episode 的 occupancy 而非新 episode 的
        self.extras["episode_left_occupied"] = self._left_occupied.clone()
        self.extras["episode_right_occupied"] = self._right_occupied.clone()

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
            push_plate_center_xy=self.cfg.push_plate_center_xy,
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

        # ── 2. 写入机器人 base pose ─────────────────────────────
        robot: Articulation = self.scene["robot"]
        default_root = robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] = self._base_pos[env_ids] + self.scene.env_origins[env_ids]
        base_quat = batch_quat_from_yaw(self._base_yaw[env_ids])
        default_root[:, 3:7] = base_quat
        default_root[:, 7:] = 0.0  # 零速度
        robot.write_root_state_to_sim(default_root, env_ids)

        # 重置关节到默认位置
        default_jpos = robot.data.default_joint_pos[env_ids]
        default_jvel = robot.data.default_joint_vel[env_ids]
        robot.write_joint_state_to_sim(default_jpos, default_jvel, None, env_ids)

        # ── 3. 重置门关节角度 ────────────────────────────────────
        door: Articulation = self.scene["door"]
        zero_door_pos = torch.zeros(n, door.num_joints, device=self.device)
        zero_door_vel = torch.zeros(n, door.num_joints, device=self.device)
        door.write_joint_state_to_sim(zero_door_pos, zero_door_vel, None, env_ids)

        # ── 4. occupancy 已由外部 set_occupancy() 设置 ──────────
        # 不在此处重置 — 保留外部课程管理器注入的 occupancy。
        # 注意：如果没有外部注入，occupancy 保持初始化时的 False。

        # ── 5. 杯体处理 ─────────────────────────────────────────
        # 将不需要的杯体 teleport 到远处
        self._teleport_cups_to_park(env_ids)

        # 对需要持杯的 env 执行批量杯抓取初始化
        need_cup = self._left_occupied[env_ids] | self._right_occupied[env_ids]
        if need_cup.any():
            cup_env_ids = env_ids[need_cup]
            self._batch_cup_grasp_init(cup_env_ids)

        # ── 6. 应用域随机化物理参数 ──────────────────────────────
        self._apply_domain_params(env_ids)

        # ── 7. 重置 per-env 状态 ────────────────────────────────
        self._step_count[env_ids] = 0
        self._prev_door_angle[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._prev_prev_action[env_ids] = 0.0
        self._already_succeeded[env_ids] = False
        self._prev_left_ee_lin_vel[env_ids] = 0.0
        self._prev_left_ee_ang_vel[env_ids] = 0.0
        self._prev_right_ee_lin_vel[env_ids] = 0.0
        self._prev_right_ee_ang_vel[env_ids] = 0.0
        self._cached_cup_dropped[env_ids] = False

    # ═══════════════════════════════════════════════════════════════════
    # 课程注入接口
    # ═══════════════════════════════════════════════════════════════════

    def set_occupancy(
        self,
        left_occupied: Tensor,
        right_occupied: Tensor,
    ) -> None:
        """外部课程管理器调用，设置所有 env 的持杯 occupancy。

        Parameters
        ----------
        left_occupied : (N,) bool
        right_occupied : (N,) bool
        """
        self._left_occupied[:] = left_occupied
        self._right_occupied[:] = right_occupied

    def set_episode_reset_fn(self, fn) -> None:
        """注册 auto-reset 回调，在 `_reset_idx()` 中逐 env 调用。"""
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
        """将世界系 EE 状态变换到 base_link 相对系。"""
        pos_base = batch_vector_world_to_base(ee_pos_w - base_pos_w, base_quat_w)
        quat_base = batch_orientation_world_to_base(ee_quat_w, base_quat_w)
        lv_base = batch_vector_world_to_base(ee_lv_w - base_lv_w, base_quat_w)
        av_base = batch_vector_world_to_base(ee_av_w - base_av_w, base_quat_w)
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
    def _compute_tilt(quat_base: Tensor) -> tuple[Tensor, Tensor]:
        """计算 cup tilt-to-gravity proxy — 重力在 EE 局部系中的 xy 投影。

        Parameters
        ----------
        quat_base : (N, 4) wxyz

        Returns
        -------
        tilt_norm : (N, 1) tilt 标量范数
        tilt_xy : (N, 2) tilt 的 xy 分量（用于奖励计算）
        """
        R = batch_quat_to_rotation_matrix(quat_base)  # (N, 3, 3)
        # 注：g_world 直接使用世界坐标系重力方向，隐含 base Z 轴与世界 Z 轴对齐。
        # 对于平坦地面上的移动机器人（base 水平放置），此假设成立。
        g_world = torch.tensor(
            [0.0, 0.0, -9.81], device=quat_base.device
        ).expand(quat_base.shape[0], 3)
        # g_local = R^T @ g_world_in_base ≈ R^T @ g_world（base 水平时等价）
        g_local = torch.bmm(
            R.transpose(-1, -2),
            g_world.unsqueeze(-1),
        ).squeeze(-1)  # (N, 3)
        tilt_xy = g_local[:, :2]  # (N, 2)
        return tilt_xy.norm(dim=-1, keepdim=True), tilt_xy  # (N, 1), (N, 2)

    def _resolve_collision_groups(
        self, robot: Articulation
    ) -> list[tuple[list[int], list[int]]]:
        """一次性解析自碰撞检测分组的 body 索引。

        Returns
        -------
        list of (group_a_indices, group_b_indices)
        """
        body_names = robot.body_names

        left_arm_links = {
            "left_link00", "left_link01", "left_link02",
            "left_link03", "left_link04", "left_link05", "left_link06",
            "left_gripperStator", "left_gripperMover",
        }
        right_arm_links = {
            "right_link00", "right_link01", "right_link02",
            "right_link03", "right_link04", "right_link05", "right_link06",
            "right_gripperStator", "right_gripperMover",
        }
        base_links = {"base_link"}

        def _name_to_indices(names: set[str]) -> list[int]:
            return [i for i, n in enumerate(body_names) if n in names]

        left_ids = _name_to_indices(left_arm_links)
        right_ids = _name_to_indices(right_arm_links)
        base_ids = _name_to_indices(base_links)

        groups = []
        if left_ids and right_ids:
            groups.append((left_ids, right_ids))
        if left_ids and base_ids:
            groups.append((left_ids, base_ids))
        if right_ids and base_ids:
            groups.append((right_ids, base_ids))
        return groups

    def _compute_batch_self_collision(self) -> Tensor:
        """检测自碰撞 — 使用 contact sensor 的交叉分组方法。

        NOTE: 这是一个近似检测。检查两个 body 分组是否*同时*有接触力，
        作为自碰撞的代理指标。较高的力阈值（1.0N）减少环境接触的误报。

        Returns
        -------
        (N,) bool — 每个环境是否发生自碰撞
        """
        contact_sensor: ContactSensor = self.scene["contact_sensor"]
        # net_forces_w: (N, num_bodies, 3)
        net_forces = contact_sensor.data.net_forces_w

        # 使用较高的力阈值减少误报（地面接触、杯体接触等）
        force_mag = net_forces.norm(dim=-1)  # (N, num_bodies)
        active = force_mag > 1.0  # (N, num_bodies) bool

        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for group_a_ids, group_b_ids in self._self_collision_groups:
            a_active = active[:, group_a_ids].any(dim=-1)  # (N,)
            b_active = active[:, group_b_ids].any(dim=-1)  # (N,)
            result = result | (a_active & b_active)

        return result

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

    def _batch_cup_grasp_init(self, env_ids: Tensor) -> None:
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
        # 使用 default_joint_pos 作为基准（而非 stale 的 data.joint_pos）
        joint_pos = robot.data.default_joint_pos[env_ids].clone()  # (n, num_joints)
        for jname, deg in LEFT_ARM_GRASP_INIT_DEG.items():
            jids, _ = robot.find_joints([jname])
            if jids is not None and len(jids) > 0:
                joint_pos[:, jids[0]] = math.radians(deg)
        for jname, deg in RIGHT_ARM_GRASP_INIT_DEG.items():
            jids, _ = robot.find_joints([jname])
            if jids is not None and len(jids) > 0:
                joint_pos[:, jids[0]] = math.radians(deg)

        # Gripper 直接设到关闭角度（跳过渐进关闭）
        for gname in GRIPPER_JOINT_NAMES:
            gids, _ = robot.find_joints([gname])
            if gids is not None and len(gids) > 0:
                joint_pos[:, gids[0]] = math.radians(GRIPPER_CLOSE_DEG)

        joint_vel = torch.zeros_like(joint_pos)  # (n, num_joints)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── 2. 计算杯体世界坐标并 teleport（不需要物理步进）─────
        # 使用 base_pos 和 base_yaw 计算杯体的世界坐标
        left_cup_ids = env_ids[self._left_occupied[env_ids]]
        right_cup_ids = env_ids[self._right_occupied[env_ids]]

        if len(left_cup_ids) > 0:
            rel = torch.tensor(LEFT_CUP_RELATIVE_XYZ, device=self.device).float()
            local_yaw = self._base_yaw[left_cup_ids]
            base_p = self._base_pos[left_cup_ids] + self.scene.env_origins[left_cup_ids]
            cup_world = base_p + batch_rotate_relative_by_yaw(rel, local_yaw)
            cup_state = torch.zeros(len(left_cup_ids), 13, device=self.device)
            cup_state[:, :3] = cup_world
            cup_state[:, 3] = 1.0  # quat w
            cup_left.write_root_state_to_sim(cup_state, left_cup_ids)

        if len(right_cup_ids) > 0:
            rel = torch.tensor(RIGHT_CUP_RELATIVE_XYZ, device=self.device).float()
            local_yaw = self._base_yaw[right_cup_ids]
            base_p = self._base_pos[right_cup_ids] + self.scene.env_origins[right_cup_ids]
            cup_world = base_p + batch_rotate_relative_by_yaw(rel, local_yaw)
            cup_state = torch.zeros(len(right_cup_ids), 13, device=self.device)
            cup_state[:, :3] = cup_world
            cup_state[:, 3] = 1.0
            cup_right.write_root_state_to_sim(cup_state, right_cup_ids)
