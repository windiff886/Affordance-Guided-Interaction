"""保存最近若干步历史数据的轻量缓存。

用于维护动作历史、速度/加速度历史等，为稳定性 proxy 和 actor
观测中的时间序列特征提供支撑。
"""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class HistoryBuffer(Generic[T]):
    """固定长度的 FIFO 缓存，保存最近 *max_length* 步数据。

    Parameters
    ----------
    max_length : int
        缓存容量（必须 > 0）。
    fill_value : T | None
        如果提供，则在初始化时用该值填满缓存（减少冷启动时的
        ``len < max_length`` 判断）。
    """

    def __init__(self, max_length: int, *, fill_value: T | None = None) -> None:
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self._max_length = max_length
        self._buffer: deque[T] = deque(maxlen=max_length)
        if fill_value is not None:
            for _ in range(max_length):
                self._buffer.append(fill_value)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def append(self, item: T) -> None:
        """追加一条数据，超出容量时自动丢弃最旧的。"""
        self._buffer.append(item)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def items(self) -> list[T]:
        """返回缓存内容的列表拷贝（从旧到新）。"""
        return list(self._buffer)

    def latest(self, n: int = 1) -> list[T]:
        """返回最近 *n* 条数据（从旧到新）。

        如果缓存不足 *n* 条，返回已有的全部。
        """
        buf_list = list(self._buffer)
        return buf_list[-n:]

    @property
    def last(self) -> T | None:
        """返回最新的一条数据，若缓存为空则返回 ``None``。"""
        if len(self._buffer) == 0:
            return None
        return self._buffer[-1]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self._max_length

    def reset(self, fill_value: T | None = None) -> None:
        """清空缓存。如果给了 *fill_value* 则重新填满。"""
        self._buffer.clear()
        if fill_value is not None:
            for _ in range(self._max_length):
                self._buffer.append(fill_value)

    def to_numpy(self) -> np.ndarray:
        """将缓存内容堆叠为 numpy 数组。

        要求缓存中的每一项都可以被 ``np.asarray`` 转换。
        """
        if len(self._buffer) == 0:
            return np.array([])
        return np.stack([np.asarray(x) for x in self._buffer], axis=0)
