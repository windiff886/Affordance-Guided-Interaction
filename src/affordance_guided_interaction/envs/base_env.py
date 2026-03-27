from __future__ import annotations

from typing import Any


class BaseEnv:
    """仿真环境占位基类。"""

    def reset(self) -> dict[str, Any]:
        raise NotImplementedError("待接入 Isaac Sim 环境重置逻辑")

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raise NotImplementedError("待接入 Isaac Sim 环境步进逻辑")

