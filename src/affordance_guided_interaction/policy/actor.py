from __future__ import annotations


class Actor:
    """低层执行策略占位类。"""

    def act(self, _actor_obs: dict[str, object]) -> object:
        raise NotImplementedError("待实现 actor 前向与采样逻辑")

