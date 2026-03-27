from __future__ import annotations


class RecurrentBackbone:
    """循环骨干网络占位类。"""

    def forward(self, _inputs: object, _hidden_state: object | None = None) -> object:
        raise NotImplementedError("待接入 GRU 或 LSTM 实现")

