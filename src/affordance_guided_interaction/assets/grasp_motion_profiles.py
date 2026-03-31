from __future__ import annotations

from math import radians


def build_gripper_closure_targets(
    open_deg: float,
    close_deg: float,
    steps: int,
) -> list[float]:
    if steps <= 1:
        return [radians(close_deg)]

    delta = (close_deg - open_deg) / (steps - 1)
    return [radians(open_deg + delta * index) for index in range(steps)]


def build_joint_space_targets(
    start_deg: dict[str, float],
    end_deg: dict[str, float],
    steps: int,
) -> list[dict[str, float]]:
    if steps < 2:
        raise ValueError("steps")
    if set(start_deg) != set(end_deg):
        raise ValueError("joint sets")

    joint_names = sorted(start_deg)
    targets: list[dict[str, float]] = []
    for index in range(steps):
        alpha = index / (steps - 1)
        targets.append(
            {
                name: start_deg[name] + alpha * (end_deg[name] - start_deg[name])
                for name in joint_names
            }
        )
    return targets
