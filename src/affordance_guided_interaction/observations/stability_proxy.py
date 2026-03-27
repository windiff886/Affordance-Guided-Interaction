from __future__ import annotations

from typing import Any


def estimate_stability_proxy(
    *,
    gripper_state: dict[str, Any],
    previous_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    linear_velocity = gripper_state.get("linear_velocity", [0.0, 0.0, 0.0])
    angular_velocity = gripper_state.get("angular_velocity", [0.0, 0.0, 0.0])
    _ = previous_state

    return {
        "tilt": 0.0,
        "linear_velocity_norm": _norm(linear_velocity),
        "angular_velocity_norm": _norm(angular_velocity),
        "linear_acceleration_norm": 0.0,
        "angular_acceleration_norm": 0.0,
        "jerk_proxy": 0.0,
    }


def _norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5

