"""训练模块 — 策略优化、轨迹收集、课程编排与域随机化。

提供 Affordance-guided interaction 框架中训练层的全部组件：

- :class:`PPOTrainer`          — PPO 策略梯度更新器
- :class:`PPOConfig`           — PPO 超参配置
- :class:`RolloutBuffer`       — GAE + TBPTT 轨迹缓存
- :class:`RolloutCollector`    — 并行轨迹采集器
- :class:`CurriculumManager`   — 五阶段课程管理器
- :class:`StageConfig`         — 课程阶段配置
- :class:`DomainRandomizer`    — 域随机化采样器
- :class:`RandomizationConfig` — 域随机化超参
- :class:`TrainingMetrics`     — 训练指标聚合器
"""

from .ppo_trainer import PPOTrainer, PPOConfig
from .rollout_buffer import RolloutBuffer
from .rollout_collector import RolloutCollector
from .curriculum_manager import CurriculumManager, StageConfig, STAGE_CONFIGS
from .domain_randomizer import DomainRandomizer, RandomizationConfig
from .metrics import TrainingMetrics

__all__ = [
    "PPOTrainer",
    "PPOConfig",
    "RolloutBuffer",
    "RolloutCollector",
    "CurriculumManager",
    "StageConfig",
    "STAGE_CONFIGS",
    "DomainRandomizer",
    "RandomizationConfig",
    "TrainingMetrics",
]
