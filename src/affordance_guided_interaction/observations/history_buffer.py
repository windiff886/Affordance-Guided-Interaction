from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar


T = TypeVar("T")


class HistoryBuffer(Generic[T]):
    """保存最近若干步历史数据的轻量缓存。"""

    def __init__(self, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._buffer: deque[T] = deque(maxlen=max_length)

    def append(self, item: T) -> None:
        self._buffer.append(item)

    def items(self) -> list[T]:
        return list(self._buffer)

