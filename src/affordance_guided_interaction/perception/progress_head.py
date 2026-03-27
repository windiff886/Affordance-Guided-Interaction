from __future__ import annotations


class ProgressHead:
    """任务进展表示头占位类。"""

    def forward(self, _features: object) -> dict[str, object]:
        raise NotImplementedError("待定义 progress 表示形式")

