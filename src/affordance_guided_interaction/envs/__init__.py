"""环境封装模块。

保持包级导入轻量，避免在仅使用 `batch_math` 等纯 Python 子模块时
立即触发 Isaac Lab / omni 运行时依赖。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["DoorPushEnv", "DoorPushEnvCfg", "DoorPushLiteEnvCfg", "DoorPushSceneCfg", "DoorPushLiteSceneCfg", "DirectRLEnvAdapter"]


if TYPE_CHECKING:
    from .direct_rl_env_adapter import DirectRLEnvAdapter
    from .door_push_env import DoorPushEnv
    from .door_push_env_cfg import DoorPushEnvCfg, DoorPushLiteEnvCfg, DoorPushSceneCfg, DoorPushLiteSceneCfg


def __getattr__(name: str) -> Any:
    if name == "DoorPushEnv":
        from .door_push_env import DoorPushEnv

        return DoorPushEnv
    if name == "DoorPushEnvCfg":
        from .door_push_env_cfg import DoorPushEnvCfg

        return DoorPushEnvCfg
    if name == "DoorPushLiteEnvCfg":
        from .door_push_env_cfg import DoorPushLiteEnvCfg

        return DoorPushLiteEnvCfg
    if name == "DoorPushSceneCfg":
        from .door_push_env_cfg import DoorPushSceneCfg

        return DoorPushSceneCfg
    if name == "DoorPushLiteSceneCfg":
        from .door_push_env_cfg import DoorPushLiteSceneCfg

        return DoorPushLiteSceneCfg
    if name == "DirectRLEnvAdapter":
        from .direct_rl_env_adapter import DirectRLEnvAdapter

        return DirectRLEnvAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
