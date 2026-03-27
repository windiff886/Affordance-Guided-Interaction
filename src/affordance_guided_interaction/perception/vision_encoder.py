from __future__ import annotations


class VisionEncoder:
    """视觉骨干网络占位类。"""

    def forward(self, _inputs: object) -> object:
        raise NotImplementedError("待选定视觉输入形式和网络结构")

