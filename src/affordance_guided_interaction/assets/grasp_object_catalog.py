from __future__ import annotations

import json
from typing import Any

from affordance_guided_interaction.assets.base_relative_pickup import (
    build_base_relative_pickup_config,
)
from affordance_guided_interaction.assets.grasp_motion_profiles import (
    build_gripper_closure_targets,
)
from affordance_guided_interaction.utils.paths import GRASP_OBJECTS_ROOT


CATALOG_PATH = GRASP_OBJECTS_ROOT / "catalog.json"


def load_grasp_object_catalog() -> dict[str, dict[str, Any]]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_grasp_object_config(
    family: str,
    variant: str | None = None,
    arm: str = "left",
) -> dict[str, Any]:
    catalog = load_grasp_object_catalog()
    if family not in catalog:
        raise ValueError(f"Unknown grasp object family: {family}")
    if arm not in {"left", "right"}:
        raise ValueError(f"Unsupported arm side: {arm}")

    family_entry = catalog[family]
    chosen_variant = variant or family_entry["default_variant"]
    variants = family_entry["variants"]
    if chosen_variant not in variants:
        raise ValueError(
            f"Unknown variant '{chosen_variant}' for grasp object family '{family}'"
        )

    variant_entry = variants[chosen_variant]
    config = {
        "family": family,
        "variant": chosen_variant,
        "usd_path": GRASP_OBJECTS_ROOT / variant_entry["usd_path"],
        "root_prim": variant_entry["root_prim"],
        "grasp_frame_path": variant_entry["grasp_frame_path"],
        "gripper_joint": f"{arm}_jointGripper",
        "randomization": variant_entry["randomization"],
    }

    grasp_mode = family_entry["grasp_mode"]
    config["grasp_mode"] = grasp_mode

    if grasp_mode == "base_relative_pickup":
        config["pickup"] = build_base_relative_pickup_config(
            family_entry["pickup"],
            arm=arm,
        )
    elif grasp_mode == "contact_grasp":
        arm_preset = {
            f"{arm}_{joint_name}": joint_value
            for joint_name, joint_value in family_entry["arm_preset_deg"].items()
        }
        config.update(
            {
                "grasp_frame_name": f"{arm}_{family_entry['grasp_frame_name']}",
                "grasp_frame_parent_name": f"{arm}_{family_entry['grasp_frame_parent_suffix']}",
                "grasp_frame_translation_xyz": tuple(
                    family_entry["grasp_frame_translation_xyz"]
                ),
                "grasp_frame_rotate_xyz_deg": tuple(
                    family_entry["grasp_frame_rotate_xyz_deg"]
                ),
                "arm_preset_deg": arm_preset,
                "gripper_targets_rad": {
                    "open": build_gripper_closure_targets(
                        family_entry["gripper_open_deg"],
                        family_entry["gripper_open_deg"],
                        1,
                    )[0],
                    "close": build_gripper_closure_targets(
                        family_entry["gripper_close_deg"],
                        family_entry["gripper_close_deg"],
                        1,
                    )[0],
                },
                "spawn_pre_align_updates": family_entry["spawn_pre_align_updates"],
                "pre_grasp_steps": family_entry["pre_grasp_steps"],
                "closure_steps": family_entry["closure_steps"],
                "grasp_settle_steps": family_entry["grasp_settle_steps"],
            }
        )
    elif grasp_mode == "scene_spawn":
        config["spawn_position_xyz"] = tuple(family_entry["spawn_position_xyz"])
    else:
        raise ValueError(f"Unsupported grasp mode: {grasp_mode}")

    return config
