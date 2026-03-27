from __future__ import annotations


class RewardManager:
    """统一聚合奖励分项。"""

    def combine(
        self,
        *,
        task_progress: float,
        carry_stability: float,
        effective_contact: float,
        invalid_collision: float,
        safety_penalty: float,
    ) -> dict[str, float]:
        total_reward = (
            task_progress
            + carry_stability
            + effective_contact
            - invalid_collision
            - safety_penalty
        )
        return {
            "task_progress": task_progress,
            "carry_stability": carry_stability,
            "effective_contact": effective_contact,
            "invalid_collision": invalid_collision,
            "safety_penalty": safety_penalty,
            "total_reward": total_reward,
        }

