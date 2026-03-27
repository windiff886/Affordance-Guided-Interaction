from __future__ import annotations

from typing import Any, Protocol


class AffordanceEncoder(Protocol):
    """定义 affordance/progress 编码器接口。"""

    def encode(
        self,
        *,
        observation: dict[str, Any],
        task_goal: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

