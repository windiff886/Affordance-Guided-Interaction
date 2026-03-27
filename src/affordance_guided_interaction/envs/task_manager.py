from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TaskManager:
    """管理任务类型、阶段和目标的占位对象。"""

    task_type: str = "push"
    stage: str = "init"

