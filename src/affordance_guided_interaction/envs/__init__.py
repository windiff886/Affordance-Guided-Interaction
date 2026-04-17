"""Environment package exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["DoorPushEnv", "DoorPushEnvCfg", "DoorPushSceneCfg"]

if TYPE_CHECKING:
    from .door_push_env import DoorPushEnv
    from .door_push_env_cfg import DoorPushEnvCfg, DoorPushSceneCfg


def __getattr__(name: str) -> Any:
    if name == "DoorPushEnv":
        from .door_push_env import DoorPushEnv

        return DoorPushEnv
    if name == "DoorPushEnvCfg":
        from .door_push_env_cfg import DoorPushEnvCfg

        return DoorPushEnvCfg
    if name == "DoorPushSceneCfg":
        from .door_push_env_cfg import DoorPushSceneCfg

        return DoorPushSceneCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
