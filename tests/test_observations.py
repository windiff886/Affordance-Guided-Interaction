from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_actor_observation_builder_merges_core_fields() -> None:
    from affordance_guided_interaction.observations.actor_obs_builder import (
        build_actor_observation,
    )

    actor_obs = build_actor_observation(
        proprio={"q": [0.0], "dq": [0.0]},
        gripper_state={"pose": [0.0] * 7, "velocity": [0.0] * 6},
        context={"occupied": 1, "stability_level": "high"},
        stability_proxy={"tilt": 0.1},
        z_aff={"type": "push"},
        z_prog={"stage": "approach"},
    )

    assert actor_obs["context"]["occupied"] == 1
    assert actor_obs["z_aff"]["type"] == "push"
    assert actor_obs["z_prog"]["stage"] == "approach"


def test_critic_observation_builder_attaches_privileged_data() -> None:
    from affordance_guided_interaction.observations.critic_obs_builder import (
        build_critic_observation,
    )

    critic_obs = build_critic_observation(
        actor_obs={"context": {"occupied": 0}},
        privileged={"door_mass": 1.2},
    )

    assert critic_obs["actor_obs"]["context"]["occupied"] == 0
    assert critic_obs["privileged"]["door_mass"] == 1.2


def test_history_buffer_keeps_recent_items() -> None:
    from affordance_guided_interaction.observations.history_buffer import HistoryBuffer

    buffer = HistoryBuffer(max_length=2)
    buffer.append("first")
    buffer.append("second")
    buffer.append("third")

    assert buffer.items() == ["second", "third"]


def test_stability_proxy_returns_default_safe_values() -> None:
    from affordance_guided_interaction.observations.stability_proxy import (
        estimate_stability_proxy,
    )

    proxy = estimate_stability_proxy(
        gripper_state={
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }
    )

    assert proxy["tilt"] == 0.0
    assert proxy["jerk_proxy"] == 0.0
