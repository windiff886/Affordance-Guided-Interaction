"""观测构建相关模块（双臂平台）。

提供从仿真原始状态到 actor / critic 结构化观测的完整构建管线。
当前系统为双臂 Z1 + RealSense 深度相机平台，无移动底座与云台。

主要对外接口
-----------
- :class:`ActorObsBuilder`  — 有状态的双臂 actor 观测构建器
- :class:`CriticObsBuilder` — 无状态的 critic 观测构建器
- :class:`HistoryBuffer`    — 通用历史数据缓存
- :func:`build_stability_proxy` — 单步稳定性 proxy 构建
- ``NUM_JOINTS_PER_ARM``    — 每条 Z1 臂的关节数 (6)
- ``TOTAL_ARM_JOINTS``      — 双臂总关节数 (12)
- ``DOOR_EMBEDDING_DIM``   — Point-MAE 门点云 embedding 维度 (768)
"""

from .actor_obs_builder import (
    ActorObsBuilder,
    NUM_JOINTS_PER_ARM,
    TOTAL_ARM_JOINTS,
    DOOR_EMBEDDING_DIM,
)
from .critic_obs_builder import CriticObsBuilder
from .history_buffer import HistoryBuffer
from .stability_proxy import (
    StabilityProxy,
    StabilityProxyState,
    build_stability_proxy,
    compute_tilt,
    compute_tilt_xy,
)

__all__ = [
    "ActorObsBuilder",
    "CriticObsBuilder",
    "HistoryBuffer",
    "StabilityProxy",
    "StabilityProxyState",
    "build_stability_proxy",
    "compute_tilt",
    "compute_tilt_xy",
    "NUM_JOINTS_PER_ARM",
    "TOTAL_ARM_JOINTS",
    "DOOR_EMBEDDING_DIM",
]
