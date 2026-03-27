from __future__ import annotations


class Critic:
    """价值网络占位类。"""

    def evaluate(self, _critic_obs: dict[str, object]) -> float:
        raise NotImplementedError("待实现 critic 前向逻辑")

