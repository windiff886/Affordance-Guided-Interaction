from __future__ import annotations


class ActionHead:
    """动作输出头占位类。"""

    def forward(self, _features: object) -> object:
        raise NotImplementedError("待定义 torque action 参数化方式")

