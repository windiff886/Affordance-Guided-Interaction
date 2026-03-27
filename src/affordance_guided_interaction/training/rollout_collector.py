from __future__ import annotations


class RolloutCollector:
    """并行采样器占位类。"""

    def collect(self) -> None:
        raise NotImplementedError("待实现多环境 rollout 采样")

