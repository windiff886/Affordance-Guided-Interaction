from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CurriculumManager:
    """课程调度占位类。"""

    stage: str = "stage_1"

