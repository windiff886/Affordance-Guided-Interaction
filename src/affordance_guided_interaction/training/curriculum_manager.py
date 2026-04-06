"""课程管理器 — push-only 三阶段自动跃迁与配置分发。

实现 training/README.md §4 中定义的课程学习机制：

- §4.2 三阶段自动跃迁：从无杯 push 到单臂持杯，再到最终混合分布
- §4.3 跃迁条件数学判据：滑动窗口平均成功率 ≥ η_thresh
- §4.4 课程与持杯稳定性约束的协同

本模块只负责逻辑判定与配置输出，不直接操作仿真环境或奖励管理器。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


_OCCUPANCY_CONTEXTS = ("none", "left_only", "right_only", "both")


@dataclass
class StageConfig:
    """单个课程阶段的配置。"""

    name: str
    stage_id: int
    context_probabilities: dict[str, float]
    door_types: list[str]           # 当前阶段允许的门类型
    description: str                # 核心学习目标

    def __post_init__(self) -> None:
        invalid = set(self.context_probabilities) - set(_OCCUPANCY_CONTEXTS)
        if invalid:
            raise ValueError(f"未知 occupancy context: {sorted(invalid)}")

        total = float(sum(self.context_probabilities.values()))
        if total <= 0.0:
            raise ValueError("context_probabilities 必须包含正概率")
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"context_probabilities 概率和必须为 1.0，当前为 {total}"
            )

        for context, prob in self.context_probabilities.items():
            if prob < 0.0:
                raise ValueError(
                    f"context_probabilities[{context!r}] 不能为负数，当前为 {prob}"
                )

    def sample_occupancy_batch(
        self,
        n_envs: int,
        rng: object | None = None,
    ) -> tuple[list[bool], list[bool]]:
        """按当前阶段的上下文分布采样一批左右持杯标记。"""
        chooser = np.random if rng is None else rng
        contexts = list(self.context_probabilities.keys())
        probs = [self.context_probabilities[c] for c in contexts]
        sampled = chooser.choice(contexts, size=n_envs, p=probs)

        left_occupied: list[bool] = []
        right_occupied: list[bool] = []
        for context in sampled.tolist():
            if context == "none":
                left_occupied.append(False)
                right_occupied.append(False)
            elif context == "left_only":
                left_occupied.append(True)
                right_occupied.append(False)
            elif context == "right_only":
                left_occupied.append(False)
                right_occupied.append(True)
            elif context == "both":
                left_occupied.append(True)
                right_occupied.append(True)
            else:
                raise ValueError(f"未知采样结果: {context}")

        return left_occupied, right_occupied


# ── 三阶段定义（对齐 README §4.2）──────────────────────────────────

STAGE_CONFIGS: list[StageConfig] = [
    StageConfig(
        name="stage_1",
        stage_id=1,
        context_probabilities={"none": 1.0},
        door_types=["push"],
        description="基础视觉引导接触，跑通网络闭环",
    ),
    StageConfig(
        name="stage_2",
        stage_id=2,
        context_probabilities={
            "left_only": 0.5,
            "right_only": 0.5,
        },
        door_types=["push"],
        description="在单臂持杯约束下学会稳定推门",
    ),
    StageConfig(
        name="stage_3",
        stage_id=3,
        context_probabilities={
            "none": 0.25,
            "left_only": 0.25,
            "right_only": 0.25,
            "both": 0.25,
        },
        door_types=["push"],
        description="在最终混合分布下统一覆盖无杯、单臂持杯和双臂持杯",
    ),
]


class CurriculumManager:
    """push-only 三阶段课程管理器。

    通过滑动窗口平均成功率判断阶段跃迁：

    .. math::

        \\frac{1}{M} \\sum_{e=E-M+1}^{E} \\eta_e \\geq \\eta_{\\text{thresh}}

    Parameters
    ----------
    window_size : int
        滑动窗口长度 M（epoch 数），默认 50。
    threshold : float
        成功率跃迁阈值 η_thresh，默认 0.8。
    stages : list[StageConfig] | None
        自定义阶段配置列表。为 None 时使用默认三阶段。
    initial_stage : str | int | None
        初始阶段名或阶段编号。为 None 时从第一个阶段开始。
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.8,
        stages: list[StageConfig] | None = None,
        initial_stage: str | int | None = None,
    ) -> None:
        self._stages = stages or STAGE_CONFIGS
        self._window_size = window_size
        self._threshold = threshold

        self._current_idx = self._resolve_initial_index(initial_stage)
        self._success_window: deque[float] = deque(maxlen=window_size)
        self._total_epochs: int = 0

    def _resolve_initial_index(self, initial_stage: str | int | None) -> int:
        """解析初始阶段配置。"""
        if initial_stage is None:
            return 0

        if isinstance(initial_stage, int):
            for idx, stage in enumerate(self._stages):
                if stage.stage_id == initial_stage:
                    return idx
            raise ValueError(f"未知初始阶段编号: {initial_stage}")

        for idx, stage in enumerate(self._stages):
            if stage.name == initial_stage:
                return idx

        raise ValueError(f"未知初始阶段名称: {initial_stage}")

    # ═══════════════════════════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════════════════════════

    @property
    def current_stage(self) -> int:
        """当前阶段编号（1-indexed）。"""
        return self._stages[self._current_idx].stage_id

    @property
    def current_stage_name(self) -> str:
        """当前阶段名称。"""
        return self._stages[self._current_idx].name

    @property
    def is_final_stage(self) -> bool:
        """是否已到达最终阶段。"""
        return self._current_idx >= len(self._stages) - 1

    @property
    def total_stages(self) -> int:
        """总阶段数。"""
        return len(self._stages)

    def get_stage_config(self) -> StageConfig:
        """获取当前阶段的完整配置。"""
        return self._stages[self._current_idx]

    @property
    def window_mean(self) -> float:
        """当前滑动窗口的平均成功率。"""
        if not self._success_window:
            return 0.0
        return sum(self._success_window) / len(self._success_window)

    # ═══════════════════════════════════════════════════════════════════
    # 跃迁判定（§4.3）
    # ═══════════════════════════════════════════════════════════════════

    def report_epoch(self, success_rate: float) -> bool:
        """报告一个 epoch 的成功率，判断是否触发阶段跃迁。

        Parameters
        ----------
        success_rate : float
            本轮 epoch 的任务成功率 η_e ∈ [0, 1]。

        Returns
        -------
        bool
            是否在本次调用中发生了阶段跃迁。
        """
        self._total_epochs += 1
        self._success_window.append(success_rate)

        # 已经在最终阶段，无需跃迁
        if self.is_final_stage:
            return False

        # 窗口未填满时不触发跃迁
        if len(self._success_window) < self._window_size:
            return False

        # §4.3 判据
        if self.window_mean >= self._threshold:
            self._advance()
            return True

        return False

    def _advance(self) -> None:
        """执行阶段跃迁，清空滑动窗口。"""
        self._current_idx = min(self._current_idx + 1, len(self._stages) - 1)
        self._success_window.clear()

    # ═══════════════════════════════════════════════════════════════════
    # 序列化 / 恢复
    # ═══════════════════════════════════════════════════════════════════

    def state_dict(self) -> dict:
        """导出状态（用于 checkpoint 保存）。"""
        return {
            "current_idx": self._current_idx,
            "success_window": list(self._success_window),
            "total_epochs": self._total_epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复状态（用于 checkpoint 加载）。"""
        self._current_idx = max(0, min(state["current_idx"], len(self._stages) - 1))
        self._success_window = deque(
            state["success_window"], maxlen=self._window_size
        )
        self._total_epochs = state["total_epochs"]
