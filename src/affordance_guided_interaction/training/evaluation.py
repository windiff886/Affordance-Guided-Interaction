"""评估结果汇总工具。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def summarize_evaluation_outcomes(
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """将逐回合评估结果汇总为 mixed / per-context 成功率。"""
    totals = {
        "none": 0,
        "left_only": 0,
        "right_only": 0,
        "both": 0,
    }
    successes = {
        "none": 0,
        "left_only": 0,
        "right_only": 0,
        "both": 0,
    }

    total_episodes = 0
    total_successes = 0

    for outcome in outcomes:
        total_episodes += 1
        success = bool(outcome.get("success", False))
        if success:
            total_successes += 1

        context_name = str(outcome.get("episode_context", ""))
        if context_name in totals:
            totals[context_name] += 1
            if success:
                successes[context_name] += 1

    return {
        "evaluation/episode_count": float(total_episodes),
        "evaluation/success_mixed": _safe_rate(total_successes, total_episodes),
        "evaluation/success_none": _safe_rate(successes["none"], totals["none"]),
        "evaluation/success_left_only": _safe_rate(
            successes["left_only"], totals["left_only"]
        ),
        "evaluation/success_right_only": _safe_rate(
            successes["right_only"], totals["right_only"]
        ),
        "evaluation/success_both": _safe_rate(successes["both"], totals["both"]),
    }


def _safe_rate(successes: int, totals: int) -> float:
    if totals <= 0:
        return 0.0
    return float(successes / totals)
