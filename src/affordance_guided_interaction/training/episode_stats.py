"""训练阶段的 episode 统计辅助函数。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def compute_episode_outcome_stats(
    *,
    infos: Sequence[Mapping[str, Any]],
    dones: np.ndarray,
) -> dict[str, float]:
    """从 env.step 的 `infos` 与 `dones` 计算真实 episode 统计量。"""
    completed_episodes = 0
    successful_episodes = 0

    for done, info in zip(dones, infos, strict=False):
        if not bool(done):
            continue
        completed_episodes += 1
        if bool(info.get("success", False)):
            successful_episodes += 1

    episode_success_rate = (
        successful_episodes / completed_episodes
        if completed_episodes > 0
        else 0.0
    )

    return {
        "collect/completed_episodes": float(completed_episodes),
        "collect/successful_episodes": float(successful_episodes),
        "collect/episode_success_rate": float(episode_success_rate),
    }


def extract_curriculum_success_rate(collect_stats: Mapping[str, float]) -> float:
    """提取课程跃迁应使用的 epoch 成功率。"""
    return float(collect_stats.get("collect/episode_success_rate", 0.0))
