"""任务进度状态机 — 维护成功标签与 episode 终止语义。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TerminationReason(Enum):
    """终止原因枚举。"""
    NONE = auto()          # 未终止
    ANGLE_LIMIT_REACHED = auto()  # 达到 episode 角度终止阈值
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
        本回合是否已经达到任务成功阈值。
    success_reached : bool
        与 ``success`` 同义，显式保留以匹配训练目标文档。
    reason : TerminationReason
        终止原因。
    door_angle : float
        当前门铰链角度（rad）。
    door_angle_prev : float
        上一步门铰链角度（rad）。
    step_count : int
        当前 episode 已执行的步数。
    success_time_step : int | None
        首次达到任务成功阈值的时间步。
    """

    done: bool = False
    success: bool = False
    success_reached: bool = False
    reason: TerminationReason = TerminationReason.NONE
    door_angle: float = 0.0
    door_angle_prev: float = 0.0
    step_count: int = 0
    success_time_step: int | None = None


class TaskManager:
    """任务进度状态机。

    Parameters
    ----------
    success_angle_threshold : float
        任务成功阈值（rad），默认 1.2。
    episode_end_angle_threshold : float
        episode 结束阈值（rad），默认 1.57。
    max_episode_steps : int
        单 episode 最大步数，默认 500。
    """

    def __init__(
        self,
        success_angle_threshold: float = 1.2,
        episode_end_angle_threshold: float = 1.57,
        max_episode_steps: int = 500,
    ) -> None:
        self._success_angle_threshold = success_angle_threshold
        self._episode_end_angle_threshold = episode_end_angle_threshold
        self._max_episode_steps = max_episode_steps

        # 内部状态
        self._step_count: int = 0
        self._prev_door_angle: float = 0.0
        self._success_reached: bool = False
        self._success_time_step: int | None = None

    def reset(self) -> None:
        """每个 episode 开始时调用，重置内部计数器。"""
        self._step_count = 0
        self._prev_door_angle = 0.0
        self._success_reached = False
        self._success_time_step = None

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

        if (not self._success_reached) and door_angle >= self._success_angle_threshold:
            self._success_reached = True
            self._success_time_step = self._step_count

        # 检查优先级：失败 > 角度终止 > 超时
        if cup_dropped:
            return TaskStatus(
                done=True,
                success=self._success_reached,
                success_reached=self._success_reached,
                reason=TerminationReason.CUP_DROPPED,
                door_angle=door_angle,
                door_angle_prev=prev_angle,
                step_count=self._step_count,
                success_time_step=self._success_time_step,
            )

        if door_angle >= self._episode_end_angle_threshold:
            return TaskStatus(
                done=True,
                success=self._success_reached,
                success_reached=self._success_reached,
                reason=TerminationReason.ANGLE_LIMIT_REACHED,
                door_angle=door_angle,
                door_angle_prev=prev_angle,
                step_count=self._step_count,
                success_time_step=self._success_time_step,
            )

        if self._step_count >= self._max_episode_steps:
            return TaskStatus(
                done=True,
                success=self._success_reached,
                success_reached=self._success_reached,
                reason=TerminationReason.TIMEOUT,
                door_angle=door_angle,
                door_angle_prev=prev_angle,
                step_count=self._step_count,
                success_time_step=self._success_time_step,
            )

        return TaskStatus(
            done=False,
            success=self._success_reached,
            success_reached=self._success_reached,
            reason=TerminationReason.NONE,
            door_angle=door_angle,
            door_angle_prev=prev_angle,
            step_count=self._step_count,
            success_time_step=self._success_time_step,
        )

    @property
    def step_count(self) -> int:
        """当前 episode 已执行步数。"""
        return self._step_count
