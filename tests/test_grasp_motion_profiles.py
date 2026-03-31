from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_build_gripper_closure_targets_returns_monotonic_radians() -> None:
    from affordance_guided_interaction.assets.grasp_motion_profiles import (
        build_gripper_closure_targets,
    )

    targets = build_gripper_closure_targets(open_deg=-55.0, close_deg=-8.0, steps=5)

    assert len(targets) == 5
    assert targets[0] < targets[-1] < 0.0
    assert sorted(targets) == targets


def test_build_joint_space_targets_interpolates_all_named_joints() -> None:
    from affordance_guided_interaction.assets.grasp_motion_profiles import (
        build_joint_space_targets,
    )

    targets = build_joint_space_targets(
        {"left_joint1": 0.0, "left_joint6": 0.0},
        {"left_joint1": 10.0, "left_joint6": 90.0},
        steps=3,
    )

    assert targets == [
        {"left_joint1": 0.0, "left_joint6": 0.0},
        {"left_joint1": 5.0, "left_joint6": 45.0},
        {"left_joint1": 10.0, "left_joint6": 90.0},
    ]


def test_build_joint_space_targets_rejects_mismatched_joint_sets() -> None:
    from affordance_guided_interaction.assets.grasp_motion_profiles import (
        build_joint_space_targets,
    )

    with pytest.raises(ValueError, match="joint sets"):
        build_joint_space_targets(
            {"left_joint1": 0.0},
            {"left_joint1": 10.0, "left_joint6": 90.0},
            steps=3,
        )


def test_build_joint_space_targets_rejects_non_positive_steps() -> None:
    from affordance_guided_interaction.assets.grasp_motion_profiles import (
        build_joint_space_targets,
    )

    with pytest.raises(ValueError, match="steps"):
        build_joint_space_targets(
            {"left_joint1": 0.0},
            {"left_joint1": 10.0},
            steps=0,
        )
