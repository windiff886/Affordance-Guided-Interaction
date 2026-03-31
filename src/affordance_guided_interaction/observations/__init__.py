"""观测构建相关模块。

提供从仿真原始状态到 actor / critic 结构化观测的完整构建管线。

主要对外接口
-----------
- :class:`ActorObsBuilder`  — 有状态的 actor 观测构建器
- :class:`CriticObsBuilder` — 无状态的 critic 观测构建器
- :class:`HistoryBuffer`    — 通用历史数据缓存
- :func:`estimate_stability_proxy` — 单步稳定性 proxy 估计
"""

from .actor_obs_builder import ActorObsBuilder, NUM_ARM_JOINTS
from .critic_obs_builder import CriticObsBuilder
from .history_buffer import HistoryBuffer
from .stability_proxy import (
    StabilityProxy,
    StabilityProxyState,
    estimate_stability_proxy,
    compute_tilt,
)

__all__ = [
    "ActorObsBuilder",
    "CriticObsBuilder",
    "HistoryBuffer",
    "StabilityProxy",
    "StabilityProxyState",
    "estimate_stability_proxy",
    "compute_tilt",
    "NUM_ARM_JOINTS",
]
