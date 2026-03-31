from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_run_observation_pause_updates_and_sleeps_for_expected_duration() -> None:
    from affordance_guided_interaction.utils.runtime_timing import run_observation_pause

    updates = []
    sleeps = []

    def _update() -> None:
        updates.append(True)

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    steps = run_observation_pause(
        update_callback=_update,
        sleep_callback=_sleep,
        duration_seconds=10.0,
        hz=20,
    )

    assert steps == 200
    assert len(updates) == 200
    assert len(sleeps) == 200
    assert sum(sleeps) == pytest.approx(10.0)


def test_run_observation_pause_skips_non_positive_duration() -> None:
    from affordance_guided_interaction.utils.runtime_timing import run_observation_pause

    updates = []
    sleeps = []

    steps = run_observation_pause(
        update_callback=lambda: updates.append(True),
        sleep_callback=lambda seconds: sleeps.append(seconds),
        duration_seconds=0.0,
    )

    assert steps == 0
    assert updates == []
    assert sleeps == []
