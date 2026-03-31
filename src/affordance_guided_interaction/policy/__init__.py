"""策略网络模块。

提供 Affordance-guided interaction 框架中约束感知执行层的全部组件：

- :class:`RecurrentBackbone` — GRU / LSTM 循环主干网络
- :class:`ActionHead`        — 高斯动作参数化头（12 维力矩）
- :class:`Actor`             — 多分支 encoder + backbone + head 完整 actor
- :class:`ActorConfig`       — Actor 超参配置
- :class:`Critic`            — 非对称 MLP critic（接收 privileged info）
- :class:`CriticConfig`      — Critic 超参配置
- :func:`flatten_actor_obs`  — actor_obs 字典 → 分支张量
- :func:`flatten_privileged` — privileged 字典 → 1-D 张量
- :func:`flatten_critic_obs` — critic_obs 字典 → 分支 + privileged 张量
"""

from .recurrent_backbone import RecurrentBackbone
from .action_head import ActionHead
from .actor import Actor, ActorConfig, flatten_actor_obs, batch_flatten_actor_obs
from .critic import Critic, CriticConfig, flatten_privileged, flatten_critic_obs

__all__ = [
    "RecurrentBackbone",
    "ActionHead",
    "Actor",
    "ActorConfig",
    "Critic",
    "CriticConfig",
    "flatten_actor_obs",
    "batch_flatten_actor_obs",
    "flatten_privileged",
    "flatten_critic_obs",
]
