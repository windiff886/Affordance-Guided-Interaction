from __future__ import annotations


class AffordanceHead:
    """对象 affordance 预测头占位类。"""

    def forward(self, _features: object) -> dict[str, object]:
        raise NotImplementedError("待定义 affordance 表示形式")

