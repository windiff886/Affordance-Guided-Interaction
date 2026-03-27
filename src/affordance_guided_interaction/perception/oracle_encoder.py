from __future__ import annotations

from typing import Any


class OracleAffordanceEncoder:
    """从仿真真值直接构造表示的占位版本。"""

    def encode(
        self,
        *,
        observation: dict[str, Any],
        task_goal: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"source": "oracle", "task_goal": task_goal},
            {"source": "oracle", "progress": observation.get("progress", "unknown")},
        )

