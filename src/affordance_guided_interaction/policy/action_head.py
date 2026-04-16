"""高斯动作头：将 backbone 隐状态解码为连续动作分布。

输出双臂 12 维关节位置目标 (rad) 的对角高斯分布参数 (μ, σ)，
提供 sample / evaluate / deterministic 三种调用模式供 PPO 使用。
不进行关节限位 clip——位置限位由仿真环境层负责。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActionHead(nn.Module):
    """高斯动作参数化头。

    Parameters
    ----------
    input_dim : int
        来自 RecurrentBackbone 的隐状态维度。
    action_dim : int
        输出动作维度，默认 12（双臂各 6 关节位置目标，单位 rad）。
    log_std_init : float
        可学习 log_std 的初始值，默认 -0.5。
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 12,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        # 均值映射层
        self.mu_layer = nn.Linear(input_dim, action_dim)

        # 可学习的 log_std（state-independent，与输入无关）
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )

    # ------------------------------------------------------------------
    # 前向与采样
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算高斯分布参数。

        Parameters
        ----------
        features : ``(batch, input_dim)``

        Returns
        -------
        mu : ``(batch, action_dim)``
        std : ``(batch, action_dim)``
        """
        mu = self.mu_layer(features)  # (B, action_dim)
        std = torch.clamp(self.log_std, -20.0, 2.0).exp().expand_as(mu)  # (B, action_dim)
        return mu, std

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """采样动作并返回对应的 log probability。

        Parameters
        ----------
        features : ``(batch, input_dim)``

        Returns
        -------
        action : ``(batch, action_dim)``
        log_prob : ``(batch,)``
            各维度 log_prob 之和（对角高斯）。
        """
        mu, std = self.forward(features)
        dist = Normal(mu, std)
        action = dist.rsample()  # 可微采样
        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)
        return action, log_prob

    def sample_with_entropy(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作并同时返回 log_prob 和 entropy（单次 forward，避免 L2 重复计算）。

        Returns
        -------
        action : ``(batch, action_dim)``
        log_prob : ``(batch,)``
        entropy : ``(batch,)``
        """
        mu, std = self.forward(features)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate(
        self, features: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """给定已执行的动作，计算 log_prob 和 entropy（PPO 训练用）。

        Parameters
        ----------
        features : ``(batch, input_dim)``
        action : ``(batch, action_dim)``

        Returns
        -------
        log_prob : ``(batch,)``
        entropy : ``(batch,)``
        """
        mu, std = self.forward(features)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic(self, features: torch.Tensor) -> torch.Tensor:
        """返回确定性动作（均值），用于部署推理。

        Parameters
        ----------
        features : ``(batch, input_dim)``

        Returns
        -------
        action : ``(batch, action_dim)``
        """
        mu, _ = self.forward(features)
        return mu
