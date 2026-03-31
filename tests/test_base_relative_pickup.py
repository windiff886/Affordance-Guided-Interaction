from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_compute_support_center_from_pickup_point() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        compute_support_center_xyz,
    )

    assert compute_support_center_xyz((0.4, 0.25, 0.85), (0.12, 0.12, 0.02)) == (
        0.4,
        0.25,
        0.84,
    )


def test_compute_cup_root_uses_pickup_point_directly() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        compute_cup_root_xyz,
    )

    assert compute_cup_root_xyz((0.4, 0.25, 0.85)) == (0.4, 0.25, 0.85)


def test_pickup_success_requires_minimum_lift_height() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        pickup_lift_succeeds,
    )

    assert pickup_lift_succeeds(
        current_height=0.90,
        support_top_z=0.85,
        minimum_lift=0.03,
    )
    assert not pickup_lift_succeeds(
        current_height=0.87,
        support_top_z=0.85,
        minimum_lift=0.03,
    )


def test_pickup_success_requires_gripper_proximity() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        pickup_proximity_succeeds,
    )

    assert pickup_proximity_succeeds(distance_to_grasp_frame=0.05, threshold=0.08)
    assert not pickup_proximity_succeeds(distance_to_grasp_frame=0.10, threshold=0.08)


def test_pickup_success_requires_upright_tilt() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        pickup_tilt_succeeds,
    )

    assert pickup_tilt_succeeds(tilt_deg=10.0, max_tilt_deg=20.0)
    assert not pickup_tilt_succeeds(tilt_deg=25.0, max_tilt_deg=20.0)
    assert pickup_tilt_succeeds(tilt_deg=-10.0, max_tilt_deg=20.0)
    assert not pickup_tilt_succeeds(tilt_deg=-25.0, max_tilt_deg=20.0)


def test_pickup_success_requires_settle_speed_below_threshold() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        pickup_settle_speed_succeeds,
    )

    assert pickup_settle_speed_succeeds(linear_speed_mps=0.30, max_speed_mps=0.50)
    assert not pickup_settle_speed_succeeds(linear_speed_mps=0.70, max_speed_mps=0.50)


def test_build_pickup_failure_payload_includes_stage_and_scalars() -> None:
    from affordance_guided_interaction.assets.base_relative_pickup import (
        build_pickup_failure_payload,
    )

    payload = build_pickup_failure_payload(
        stage="lift",
        support_top_z=0.85,
        cup_height=0.87,
        cup_tilt_deg=12.0,
        settle_speed_mps=0.20,
        distance_to_grasp_frame=0.04,
        gripper_joint_name="left_jointGripper",
    )

    assert payload["stage"] == "lift"
    assert payload["cup_height"] == 0.87
    assert payload["distance_to_grasp_frame"] == 0.04
    assert payload["gripper_joint_name"] == "left_jointGripper"
