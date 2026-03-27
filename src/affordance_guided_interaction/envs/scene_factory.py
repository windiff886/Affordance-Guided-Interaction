from __future__ import annotations


class SceneFactory:
    """场景装配占位工厂。"""

    def build(self) -> None:
        raise NotImplementedError("待实现门、按钮、把手与杯体资产装配")

