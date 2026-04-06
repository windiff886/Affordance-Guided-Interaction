"""观测相关工具模块。

提供通用的历史数据缓存和稳定性计算工具。

注意：观测构建现在直接在 DoorPushEnv._get_observations() 中完成，
不再通过独立的 ActorObsBuilder/CriticObsBuilder 类。

保留的工具类
-----------
- :class:`HistoryBuffer`    — 通用历史数据缓存
- :class:`StabilityProxy`   — 稳定性 proxy（含 tilt 计算参考实现）
- :func:`build_stability_proxy` — 单步稳定性 proxy 构建
- :func:`compute_tilt`      — 计算 body tilt 角度
- :func:`compute_tilt_xy`   — 计算 body 前后/左右 tilt 分量
"""

from .history_buffer import HistoryBuffer
from .stability_proxy import (
    StabilityProxy,
    StabilityProxyState,
    build_stability_proxy,
    compute_tilt,
    compute_tilt_xy,
)

__all__ = [
    "HistoryBuffer",
    "StabilityProxy",
    "StabilityProxyState",
    "build_stability_proxy",
    "compute_tilt",
    "compute_tilt_xy",
]
