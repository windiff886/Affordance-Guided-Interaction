from __future__ import annotations

from typing import Any


EXPECTED_LEFT_ARM_JOINTS = tuple(f"left_joint{index}" for index in range(1, 7))


def compute_support_center_xyz(
    pickup_point_xyz: tuple[float, float, float],
    support_size_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        pickup_point_xyz[0],
        pickup_point_xyz[1],
        pickup_point_xyz[2] - 0.5 * support_size_xyz[2],
    )


def compute_cup_root_xyz(
    pickup_point_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(pickup_point_xyz)


def pickup_lift_succeeds(
    current_height: float,
    support_top_z: float,
    minimum_lift: float,
) -> bool:
    return current_height - support_top_z >= minimum_lift


def pickup_proximity_succeeds(
    distance_to_grasp_frame: float,
    threshold: float,
) -> bool:
    return distance_to_grasp_frame <= threshold


def pickup_tilt_succeeds(
    tilt_deg: float,
    max_tilt_deg: float,
) -> bool:
    return abs(tilt_deg) <= max_tilt_deg


def pickup_settle_speed_succeeds(
    linear_speed_mps: float,
    max_speed_mps: float,
) -> bool:
    return linear_speed_mps <= max_speed_mps


def build_pickup_failure_payload(
    *,
    stage: str,
    support_top_z: float,
    cup_height: float,
    cup_tilt_deg: float,
    settle_speed_mps: float,
    distance_to_grasp_frame: float,
    gripper_joint_name: str,
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "support_top_z": support_top_z,
        "cup_height": cup_height,
        "cup_tilt_deg": cup_tilt_deg,
        "settle_speed_mps": settle_speed_mps,
        "distance_to_grasp_frame": distance_to_grasp_frame,
        "gripper_joint_name": gripper_joint_name,
    }


def _require_vector3(
    name: str,
    value: Any,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(name)
    return (float(value[0]), float(value[1]), float(value[2]))


def _require_positive_vector3(
    name: str,
    value: Any,
) -> tuple[float, float, float]:
    vector = _require_vector3(name, value)
    if any(component <= 0.0 for component in vector):
        raise ValueError(name)
    return vector


def _require_non_negative_float(
    name: str,
    value: Any,
) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError(name)
    return float(value)


def _require_positive_int(
    name: str,
    value: Any,
) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(name)
    return value


def _require_non_negative_int(
    name: str,
    value: Any,
) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(name)
    return value


def _require_joint_targets(
    name: str,
    value: Any,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(name)
    if tuple(sorted(value)) != EXPECTED_LEFT_ARM_JOINTS:
        raise ValueError(name)
    return {joint_name: float(value[joint_name]) for joint_name in EXPECTED_LEFT_ARM_JOINTS}


def build_base_relative_pickup_config(
    raw_pickup: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    if not isinstance(raw_pickup, dict):
        raise ValueError("pickup")
    if arm != "left":
        raise ValueError("pickup arm")
    if raw_pickup.get("arm") != arm:
        raise ValueError("pickup arm")

    return {
        "schema_version": int(raw_pickup["schema_version"]),
        "arm": raw_pickup["arm"],
        "pickup_point_xyz_m": _require_vector3(
            "pickup_point_xyz_m", raw_pickup.get("pickup_point_xyz_m")
        ),
        "support_size_xyz_m": _require_positive_vector3(
            "support_size_xyz_m", raw_pickup.get("support_size_xyz_m")
        ),
        "grasp_frame_parent_link": str(raw_pickup["grasp_frame_parent_link"]),
        "grasp_frame_translation_xyz_m": _require_vector3(
            "grasp_frame_translation_xyz_m",
            raw_pickup.get("grasp_frame_translation_xyz_m"),
        ),
        "grasp_frame_rotate_xyz_deg": _require_vector3(
            "grasp_frame_rotate_xyz_deg",
            raw_pickup.get("grasp_frame_rotate_xyz_deg"),
        ),
        "pregrasp_joint_deg": _require_joint_targets(
            "pregrasp_joint_deg", raw_pickup.get("pregrasp_joint_deg")
        ),
        "grasp_joint_deg": _require_joint_targets(
            "grasp_joint_deg", raw_pickup.get("grasp_joint_deg")
        ),
        "capture_joint_deg": _require_joint_targets(
            "capture_joint_deg",
            raw_pickup.get("capture_joint_deg", raw_pickup.get("grasp_joint_deg")),
        ),
        "carry_standby_joint_deg": _require_joint_targets(
            "carry_standby_joint_deg",
            raw_pickup.get("carry_standby_joint_deg"),
        ),
        "gripper_open_deg": float(raw_pickup["gripper_open_deg"]),
        "gripper_close_deg": float(raw_pickup["gripper_close_deg"]),
        "pregrasp_steps": _require_positive_int(
            "pregrasp_steps", raw_pickup.get("pregrasp_steps")
        ),
        "approach_steps": _require_positive_int(
            "approach_steps", raw_pickup.get("approach_steps")
        ),
        "approach_settle_steps": _require_non_negative_int(
            "approach_settle_steps",
            raw_pickup.get("approach_settle_steps", 0),
        ),
        "capture_steps": _require_positive_int(
            "capture_steps", raw_pickup.get("capture_steps", raw_pickup.get("closure_steps"))
        ),
        "closure_steps": _require_positive_int(
            "closure_steps", raw_pickup.get("closure_steps")
        ),
        "settle_steps": _require_positive_int(
            "settle_steps", raw_pickup.get("settle_steps")
        ),
        "lift_steps": _require_positive_int("lift_steps", raw_pickup.get("lift_steps")),
        "retreat_steps": _require_positive_int(
            "retreat_steps", raw_pickup.get("retreat_steps")
        ),
        "lift_delta_z_m": _require_non_negative_float(
            "lift_delta_z_m", raw_pickup.get("lift_delta_z_m")
        ),
        "max_tilt_deg": _require_non_negative_float(
            "max_tilt_deg", raw_pickup.get("max_tilt_deg")
        ),
        "min_lift_height_m": _require_non_negative_float(
            "min_lift_height_m", raw_pickup.get("min_lift_height_m")
        ),
        "max_settle_linear_speed_mps": _require_non_negative_float(
            "max_settle_linear_speed_mps",
            raw_pickup.get("max_settle_linear_speed_mps"),
        ),
        "gripper_proximity_threshold_m": _require_non_negative_float(
            "gripper_proximity_threshold_m",
            raw_pickup.get("gripper_proximity_threshold_m"),
        ),
    }
