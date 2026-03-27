from __future__ import annotations


class RolloutBuffer:
    """轨迹缓存占位类。"""

    def add(self, _transition: dict[str, object]) -> None:
        raise NotImplementedError("待实现 PPO 所需轨迹存储")

