from __future__ import annotations

from typing import Any


def compute_carry_stability_reward(
    *,
    occupied: int,
    stability_proxy: dict[str, Any],
    stability_level: str,
) -> float:
    _ = stability_proxy
    if not occupied:
        return 0.0
    if stability_level == "high":
        return 0.0
    return 0.0

