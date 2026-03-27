from __future__ import annotations


class PPOTrainer:
    """PPO 训练器占位类。"""

    def train_step(self) -> None:
        raise NotImplementedError("待接入 rollout buffer 与 PPO loss")

