"""课程管理器 — 五阶段自动跃迁与配置分发。

实现 training/README.md §4 中定义的课程学习机制：

- §4.2 五阶段自动跃迁：从基础视觉引导接触到全域泛化
- §4.3 跃迁条件数学判据：滑动窗口平均成功率 ≥ η_thresh
- §4.4 课程与奖励缩放 s_t 的协同

本模块只负责逻辑判定与配置输出，不直接操作仿真环境或奖励管理器。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class StageConfig:
    """单个课程阶段的配置。"""

    name: str
    stage_id: int
    cup_probability: float          # P(occupied)
    door_types: list[str]           # 允许的门类型
    description: str                # 核心学习目标


# ── 五阶段定义（对齐 README §4.2）──────────────────────────────────

STAGE_CONFIGS: list[StageConfig] = [
    StageConfig(
        name="stage_1",
        stage_id=1,
        cup_probability=0.0,
        door_types=["push"],
        description="基础视觉引导接触，跑通网络闭环",
    ),
    StageConfig(
        name="stage_2",
        stage_id=2,
        cup_probability=1.0,
        door_types=["push"],
        description="在稳定性约束 r_stab 与 s_t 下学会力控",
    ),
    StageConfig(
        name="stage_3",
        stage_id=3,
        cup_probability=0.5,
        door_types=["push", "pull"],
        description="视觉区分 affordance 类型，调整接触策略",
    ),
    StageConfig(
        name="stage_4",
        stage_id=4,
        cup_probability=0.5,
        door_types=["handle_push", "handle_pull"],
        description="学习时序子任务组合，依靠 RNN 跨越 reward delay",
    ),
    StageConfig(
        name="stage_5",
        stage_id=5,
        cup_probability=0.5,
        door_types=["push", "pull", "handle_push", "handle_pull"],
        description="高强度域随机化下的全域泛化",
    ),
]


class CurriculumManager:
    """五阶段课程管理器。

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
        自定义阶段配置列表。为 None 时使用默认五阶段。
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.8,
        stages: list[StageConfig] | None = None,
    ) -> None:
        self._stages = stages or STAGE_CONFIGS
        self._window_size = window_size
        self._threshold = threshold

        self._current_idx: int = 0
        self._success_window: deque[float] = deque(maxlen=window_size)
        self._total_epochs: int = 0

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
        self._current_idx = state["current_idx"]
        self._success_window = deque(
            state["success_window"], maxlen=self._window_size
        )
        self._total_epochs = state["total_epochs"]
