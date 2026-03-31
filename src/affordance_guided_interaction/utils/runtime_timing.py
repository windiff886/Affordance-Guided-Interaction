from __future__ import annotations


def run_observation_pause(
    update_callback,
    sleep_callback,
    duration_seconds: float,
    hz: int = 60,
) -> int:
    if duration_seconds <= 0.0:
        return 0

    steps = max(1, int(round(duration_seconds * hz)))
    step_duration = duration_seconds / steps

    for _ in range(steps):
        update_callback()
        sleep_callback(step_duration)

    return steps
