"""任务进度状态机 — 判定 episode 的成功/失败/超时。

职责：
    每步接收当前物理状态，判断是否触发 episode 终止。
    终止条件三选一（互斥）：
        1. 成功：门铰链角度达到目标 θ_d ≥ θ_target
        2. 失败：杯体脱落（由 ContactMonitor 检测到）
        3. 超时：步数达到上限 T_max
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TerminationReason(Enum):
    """终止原因枚举。"""
    NONE = auto()          # 未终止
    SUCCESS = auto()       # 任务成功完成
    CUP_DROPPED = auto()   # 杯体脱落
    TIMEOUT = auto()       # 超时


@dataclass(slots=True)
class TaskStatus:
    """单步任务状态判定结果。

    Attributes
    ----------
    done : bool
        episode 是否终止。
    success : bool
        是否因任务完成而终止。
    reason : TerminationReason
        终止原因。
    door_angle : float
        当前门铰链角度（rad）。
    door_angle_prev : float
        上一步门铰链角度（rad）。
    step_count : int
        当前 episode 已执行的步数。
    """

    done: bool = False
    success: bool = False
    reason: TerminationReason = TerminationReason.NONE
    door_angle: float = 0.0
    door_angle_prev: float = 0.0
    step_count: int = 0


class TaskManager:
    """任务进度状态机。

    Parameters
    ----------
    door_angle_target : float
        推门成功角度阈值（rad），默认 1.2 ≈ 69°。
    max_episode_steps : int
        单 episode 最大步数，默认 500。
    """

    def __init__(
        self,
        door_angle_target: float = 1.2,
        max_episode_steps: int = 500,
    ) -> None:
        self._door_angle_target = door_angle_target
        self._max_episode_steps = max_episode_steps

        # 内部状态
        self._step_count: int = 0
        self._prev_door_angle: float = 0.0

    def reset(self) -> None:
        """每个 episode 开始时调用，重置内部计数器。"""
        self._step_count = 0
        self._prev_door_angle = 0.0

    def update(
        self,
        *,
        door_angle: float,
        cup_dropped: bool = False,
    ) -> TaskStatus:
        """判定当前步的任务状态。

        Parameters
        ----------
        door_angle : float
            当前门铰链角度（rad）。
        cup_dropped : bool
            杯体是否脱落（由 ContactMonitor 提供）。

        Returns
        -------
        TaskStatus
        """
        prev_angle = self._prev_door_angle
        self._step_count += 1
        self._prev_door_angle = door_angle

        # 检查优先级：失败 > 成功 > 超时

        # ── 杯体脱落 → 立即失败终止 ────────────────────────────
        if cup_dropped:
            return TaskStatus(
                done=True,
                success=False,
                reason=TerminationReason.CUP_DROPPED,
                door_angle=door_angle,
                door_angle_prev=prev_angle,
                step_count=self._step_count,
            )

        # ── 门角度达标 → 任务成功 ─────────────────────────────
        if door_angle >= self._door_angle_target:
            return TaskStatus(
                done=True,
                success=True,
                reason=TerminationReason.SUCCESS,
                door_angle=door_angle,
                door_angle_prev=prev_angle,
                step_count=self._step_count,
            )

        # ── 超时 ──────────────────────────────────────────────
        if self._step_count >= self._max_episode_steps:
            return TaskStatus(
                done=True,
                success=False,
                reason=TerminationReason.TIMEOUT,
                door_angle=door_angle,
                door_angle_prev=prev_angle,
                step_count=self._step_count,
            )

        # ── 继续 ──────────────────────────────────────────────
        return TaskStatus(
            done=False,
            success=False,
            reason=TerminationReason.NONE,
            door_angle=door_angle,
            door_angle_prev=prev_angle,
            step_count=self._step_count,
        )

    @property
    def step_count(self) -> int:
        """当前 episode 已执行步数。"""
        return self._step_count
