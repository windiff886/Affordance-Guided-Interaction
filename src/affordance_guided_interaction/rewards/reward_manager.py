"""奖励管理器 — 聚合所有分项奖励并执行动态缩放。

实现 README 中的以下核心逻辑：
- §2  总体奖励公式：r_task + mask·(bonus + s_t·penalty) - r_safe
- §7  动态缩放因子 s_t 的全局步数线性退火
- §10 分项监控日志输出
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .task_reward import compute_task_reward
from .stability_reward import compute_stability_reward
from .safety_penalty import compute_safety_penalty


class RewardManager:
    """统一聚合奖励分项、执行动态缩放、输出监控日志。"""

    def __init__(self, cfg: dict) -> None:
        """初始化奖励管理器。

        参数:
            cfg: 完整奖励配置字典，包含 task / stability / safety / scaling 四组。
        """
        self._cfg_task: dict = cfg["task"]
        self._cfg_stab: dict = cfg["stability"]
        self._cfg_safe: dict = cfg["safety"]
        self._cfg_scale: dict = cfg["scaling"]

        # §7.2 全局步数计数器
        self._global_step: int = 0

        # 任务成功状态跟踪（一次性 bonus）
        self._already_succeeded: bool = False

    # ═══════════════════════════════════════════════════════════════════
    # §7.2 动态缩放因子
    # ═══════════════════════════════════════════════════════════════════

    def compute_scaling_factor(self) -> float:
        """§7.2: 基于全局训练步数的线性退火。

        s_t = s_min + (1.0 - s_min) · min(1.0, N_step / N_anneal)
        """
        s_min: float = self._cfg_scale["s_min"]
        n_anneal: int = self._cfg_scale["n_anneal"]

        progress = min(1.0, self._global_step / max(n_anneal, 1))
        return s_min + (1.0 - s_min) * progress

    # ═══════════════════════════════════════════════════════════════════
    # §2 总公式聚合
    # ═══════════════════════════════════════════════════════════════════

    def step(
        self,
        *,
        # 任务进展
        theta_t: float,
        theta_prev: float,
        # 稳定性 proxy（左右臂，None 表示该侧无数据）
        left_stability_proxy: dict[str, Any] | None = None,
        right_stability_proxy: dict[str, Any] | None = None,
        # 持杯 mask
        left_occupied: bool = False,
        right_occupied: bool = False,
        # 力矩（双臂 12 维）
        torques: np.ndarray | None = None,
        prev_torques: np.ndarray | None = None,
        # 安全事件
        self_collision: bool = False,
        joint_pos: np.ndarray | None = None,
        joint_vel: np.ndarray | None = None,
        joint_limits: np.ndarray | None = None,
        joint_vel_limits: np.ndarray | None = None,
        policy_torques: np.ndarray | None = None,
        torque_limits: np.ndarray | None = None,
        cup_dropped: bool = False,
    ) -> tuple[float, bool, dict[str, float]]:
        """执行一步完整的奖励计算。

        返回:
            (total_reward, should_terminate, info_dict)
        """
        info: dict[str, float] = {}

        # ── 1. 动态缩放因子 §7 ────────────────────────────────────────
        s_t = self.compute_scaling_factor()
        info["scaling/s_t"] = s_t

        # ── 2. 主任务奖励 §4 ──────────────────────────────────────────
        r_task, newly_succeeded, task_info = compute_task_reward(
            theta_t=theta_t,
            theta_prev=theta_prev,
            already_succeeded=self._already_succeeded,
            cfg=self._cfg_task,
        )
        if newly_succeeded:
            self._already_succeeded = True
        info.update(task_info)

        # ── 3. 稳定性奖励 §5 ──────────────────────────────────────────
        # 为双臂力矩准备默认值
        if torques is None:
            torques = np.zeros(12)
        if prev_torques is None:
            prev_torques = np.zeros(12)

        # 左臂
        stab_bonus_l, stab_penalty_l = 0.0, 0.0
        if left_occupied and left_stability_proxy is not None:
            bonus_l, penalty_l, stab_info_l = compute_stability_reward(
                lin_acc=left_stability_proxy["linear_acceleration"],
                ang_acc=left_stability_proxy["angular_acceleration"],
                tilt_xy=left_stability_proxy["tilt_xy"],
                torques=torques[:6],
                prev_torques=prev_torques[:6],
                cfg=self._cfg_stab,
            )
            stab_bonus_l = bonus_l
            stab_penalty_l = penalty_l
            # 写入左臂日志
            for key, val in stab_info_l.items():
                info[f"stability/left_{key}"] = val

        # 右臂
        stab_bonus_r, stab_penalty_r = 0.0, 0.0
        if right_occupied and right_stability_proxy is not None:
            bonus_r, penalty_r, stab_info_r = compute_stability_reward(
                lin_acc=right_stability_proxy["linear_acceleration"],
                ang_acc=right_stability_proxy["angular_acceleration"],
                tilt_xy=right_stability_proxy["tilt_xy"],
                torques=torques[6:],
                prev_torques=prev_torques[6:],
                cfg=self._cfg_stab,
            )
            stab_bonus_r = bonus_r
            stab_penalty_r = penalty_r
            # 写入右臂日志
            for key, val in stab_info_r.items():
                info[f"stability/right_{key}"] = val

        # 持杯 mask 处理（§5.6 + §2 公式）
        m_l = 1.0 if left_occupied else 0.0
        m_r = 1.0 if right_occupied else 0.0

        r_stab = (
            m_l * (stab_bonus_l + s_t * stab_penalty_l)
            + m_r * (stab_bonus_r + s_t * stab_penalty_r)
        )
        info["stability_total"] = r_stab

        # ── 4. 安全惩罚 §6 ────────────────────────────────────────────
        r_safe, should_terminate, safe_info = compute_safety_penalty(
            self_collision=self_collision,
            joint_pos=joint_pos if joint_pos is not None else np.zeros(12),
            joint_vel=joint_vel if joint_vel is not None else np.zeros(12),
            joint_limits=joint_limits if joint_limits is not None else np.zeros((12, 2)),
            joint_vel_limits=joint_vel_limits if joint_vel_limits is not None else np.ones(12),
            policy_torques=policy_torques,
            torque_limits=torque_limits,
            cup_dropped=cup_dropped,
            cfg=self._cfg_safe,
        )
        info.update(safe_info)
        info["safety_total"] = r_safe

        # ── 5. §2 总公式 ──────────────────────────────────────────────
        total_reward = r_task + r_stab - r_safe
        info["total_reward"] = total_reward

        # ── 6. 推进全局步数 ───────────────────────────────────────────
        self._global_step += 1

        return total_reward, should_terminate, info

    # ═══════════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════════

    def reset_episode(self) -> None:
        """重置单回合状态（成功标记等），全局步数不重置。"""
        self._already_succeeded = False

    @property
    def global_step(self) -> int:
        """返回当前全局环境步数。"""
        return self._global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        """允许外部同步全局步数（如恢复训练时）。"""
        self._global_step = value
