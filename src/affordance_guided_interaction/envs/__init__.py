"""环境封装模块。

对外暴露：
    - BaseEnv, EnvConfig     基类与配置
    - DoorInteractionEnv     单环境实现
    - VecDoorEnv             向量化并行封装
    - SceneFactory           场景装配工厂
    - ContactMonitor, ContactSummary   接触事件
    - TaskManager, TaskStatus, TerminationReason  任务状态机
"""

from .base_env import BaseEnv, EnvConfig
from .door_env import DoorInteractionEnv
from .vec_env import VecDoorEnv
from .scene_factory import SceneFactory, SceneHandles
from .contact_monitor import ContactMonitor, ContactSummary
from .task_manager import TaskManager, TaskStatus, TerminationReason

__all__ = [
    "BaseEnv",
    "EnvConfig",
    "DoorInteractionEnv",
    "VecDoorEnv",
    "SceneFactory",
    "SceneHandles",
    "ContactMonitor",
    "ContactSummary",
    "TaskManager",
    "TaskStatus",
    "TerminationReason",
]
