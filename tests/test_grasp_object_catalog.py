from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def make_valid_cup_catalog(pickup_overrides: dict | None = None) -> dict:
    pickup = {
        "schema_version": 1,
        "arm": "left",
        "pickup_point_xyz_m": [0.40, 0.25, 0.85],
        "support_size_xyz_m": [0.12, 0.12, 0.02],
        "grasp_frame_parent_link": "left_gripperStator",
        "grasp_frame_translation_xyz_m": [0.105, 0.0, -0.015],
        "grasp_frame_rotate_xyz_deg": [0.0, 0.0, 0.0],
        "pregrasp_joint_deg": {
            "left_joint1": 0.0,
            "left_joint2": 0.0,
            "left_joint3": 0.0,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_joint6": 90.0,
        },
        "grasp_joint_deg": {
            "left_joint1": 0.0,
            "left_joint2": 20.0,
            "left_joint3": 0.0,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_joint6": 90.0,
        },
        "capture_joint_deg": {
            "left_joint1": 0.0,
            "left_joint2": 15.0,
            "left_joint3": 0.0,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_joint6": 90.0,
        },
        "carry_standby_joint_deg": {
            "left_joint1": 0.0,
            "left_joint2": 15.0,
            "left_joint3": 0.0,
            "left_joint4": -15.0,
            "left_joint5": 0.0,
            "left_joint6": 90.0,
        },
        "gripper_open_deg": -55.0,
        "gripper_close_deg": 0.0,
        "pregrasp_steps": 90,
        "approach_steps": 45,
        "capture_steps": 30,
        "closure_steps": 60,
        "settle_steps": 90,
        "lift_steps": 90,
        "retreat_steps": 60,
        "lift_delta_z_m": 0.06,
        "max_tilt_deg": 20.0,
        "min_lift_height_m": 0.03,
        "max_settle_linear_speed_mps": 0.50,
        "gripper_proximity_threshold_m": 0.08,
    }
    if pickup_overrides:
        pickup.update(pickup_overrides)
    return {
        "cup": {
            "default_variant": "carry_cup",
            "grasp_mode": "base_relative_pickup",
            "pickup": pickup,
            "variants": {
                "carry_cup": {
                    "usd_path": "cup/carry_cup.usda",
                    "root_prim": "/CarryCup",
                    "grasp_frame_path": "/CarryCup/GraspFrames/Pinch",
                    "randomization": {},
                }
            },
        }
    }


def test_load_grasp_object_catalog_exposes_supported_families() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        load_grasp_object_catalog,
    )

    catalog = load_grasp_object_catalog()

    assert set(catalog) == {"cup", "tray"}
    assert catalog["cup"]["default_variant"] == "carry_cup"
    assert catalog["tray"]["default_variant"] == "carry_tray"


def test_build_grasp_object_config_resolves_default_variant_path() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    config = build_grasp_object_config("cup", arm="left")

    assert config["family"] == "cup"
    assert config["variant"] == "carry_cup"
    assert config["usd_path"].is_file()
    assert config["gripper_joint"] == "left_jointGripper"
    assert config["grasp_mode"] == "base_relative_pickup"
    assert config["pickup"]["arm"] == "left"
    assert len(config["pickup"]["pickup_point_xyz_m"]) == 3


def test_build_grasp_object_config_honors_requested_variant_and_arm() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    config = build_grasp_object_config("tray", variant="carry_tray", arm="right")

    assert config["family"] == "tray"
    assert config["variant"] == "carry_tray"
    assert config["gripper_joint"] == "right_jointGripper"
    assert config["grasp_mode"] == "scene_spawn"
    assert config["spawn_position_xyz"] == (4.0, 0.65, 0.95)


def test_build_grasp_object_config_rejects_unknown_family() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    with pytest.raises(ValueError, match="Unknown grasp object family"):
        build_grasp_object_config("bottle")


def test_build_grasp_object_config_exposes_left_cup_arm_preset() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    config = build_grasp_object_config("cup", arm="left")

    assert config["pickup"]["pregrasp_joint_deg"]["left_joint6"] == 90.0
    assert config["pickup"]["carry_standby_joint_deg"]["left_joint6"] == 90.0
    assert config["pickup"]["settle_steps"] == 90


def test_build_grasp_object_config_uses_distinct_lift_pose_for_cup_pickup() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    config = build_grasp_object_config("cup", arm="left")

    assert config["pickup"]["carry_standby_joint_deg"] != config["pickup"]["grasp_joint_deg"]


def test_build_grasp_object_config_exposes_cup_pickup_parent_link() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    config = build_grasp_object_config("cup", arm="left")

    assert config["pickup"]["grasp_frame_parent_link"] == "left_gripperStator"


def test_build_grasp_object_config_exposes_capture_stage_for_cup_pickup() -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    config = build_grasp_object_config("cup", arm="left")

    assert set(config["pickup"]["capture_joint_deg"]) == {
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
    }
    assert isinstance(config["pickup"]["gripper_open_deg"], float)
    assert config["pickup"]["approach_steps"] > 0
    assert config["pickup"]["approach_settle_steps"] >= 0
    assert config["pickup"]["capture_steps"] > 0


def test_build_grasp_object_config_rejects_malformed_pickup_schema(monkeypatch) -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    bad_catalog = make_valid_cup_catalog({"pickup_point_xyz_m": None})

    monkeypatch.setattr(
        "affordance_guided_interaction.assets.grasp_object_catalog.load_grasp_object_catalog",
        lambda: bad_catalog,
    )

    with pytest.raises(ValueError, match="pickup"):
        build_grasp_object_config("cup", arm="left")


def test_build_grasp_object_config_rejects_non_positive_support_dimension(
    monkeypatch,
) -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    bad_catalog = make_valid_cup_catalog({"support_size_xyz_m": [0.12, 0.12, 0.0]})

    monkeypatch.setattr(
        "affordance_guided_interaction.assets.grasp_object_catalog.load_grasp_object_catalog",
        lambda: bad_catalog,
    )

    with pytest.raises(ValueError, match="support_size_xyz_m"):
        build_grasp_object_config("cup", arm="left")


def test_build_grasp_object_config_rejects_negative_threshold(monkeypatch) -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    bad_catalog = make_valid_cup_catalog({"gripper_proximity_threshold_m": -0.01})

    monkeypatch.setattr(
        "affordance_guided_interaction.assets.grasp_object_catalog.load_grasp_object_catalog",
        lambda: bad_catalog,
    )

    with pytest.raises(ValueError, match="gripper_proximity_threshold_m"):
        build_grasp_object_config("cup", arm="left")


def test_build_grasp_object_config_rejects_bad_vector_length(monkeypatch) -> None:
    from affordance_guided_interaction.assets.grasp_object_catalog import (
        build_grasp_object_config,
    )

    bad_catalog = make_valid_cup_catalog({"pickup_point_xyz_m": [0.40, 0.25]})

    monkeypatch.setattr(
        "affordance_guided_interaction.assets.grasp_object_catalog.load_grasp_object_catalog",
        lambda: bad_catalog,
    )

    with pytest.raises(ValueError, match="pickup_point_xyz_m"):
        build_grasp_object_config("cup", arm="left")
