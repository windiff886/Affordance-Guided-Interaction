from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleop_cup_grasp.grasp_demo_io import (
    GraspDemo,
    due_events,
    load_grasp_demo_npz,
    sample_joint_targets,
    save_grasp_demo_npz,
)


class GraspDemoIOTests(unittest.TestCase):
    def test_save_and_load_round_trip_preserves_arrays_and_metadata(self) -> None:
        demo = GraspDemo(
            t=np.array([0.0, 0.1, 0.2], dtype=np.float64),
            q_arm=np.array(
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                ],
                dtype=np.float64,
            ),
            q_gripper=np.array([0.0, -0.2, -0.4], dtype=np.float64),
            joint_names=np.array(
                [
                    "left_joint1",
                    "left_joint2",
                    "left_joint3",
                    "left_joint4",
                    "left_joint5",
                    "left_joint6",
                    "left_jointGripper",
                ]
            ),
            cup_world_pos=np.array([0.45, 0.20, 0.55], dtype=np.float64),
            cup_world_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            robot_initial_q=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0], dtype=np.float64),
            support_center_world=np.array([0.45, 0.20, 0.54], dtype=np.float64),
            support_scale=np.array([0.12, 0.12, 0.02], dtype=np.float64),
            remove_support_time=0.2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "grasp_demo.npz"
            save_grasp_demo_npz(path, demo)
            loaded = load_grasp_demo_npz(path)

        np.testing.assert_allclose(loaded.t, demo.t)
        np.testing.assert_allclose(loaded.q_arm, demo.q_arm)
        np.testing.assert_allclose(loaded.q_gripper, demo.q_gripper)
        np.testing.assert_array_equal(loaded.joint_names, demo.joint_names)
        np.testing.assert_allclose(loaded.cup_world_pos, demo.cup_world_pos)
        np.testing.assert_allclose(loaded.cup_world_quat_wxyz, demo.cup_world_quat_wxyz)
        np.testing.assert_allclose(loaded.robot_initial_q, demo.robot_initial_q)
        np.testing.assert_allclose(loaded.support_center_world, demo.support_center_world)
        np.testing.assert_allclose(loaded.support_scale, demo.support_scale)
        self.assertAlmostEqual(loaded.remove_support_time, demo.remove_support_time)

    def test_sample_joint_targets_linearly_interpolates_between_samples(self) -> None:
        demo = GraspDemo(
            t=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            q_arm=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ],
                dtype=np.float64,
            ),
            q_gripper=np.array([0.0, -0.5, -1.0], dtype=np.float64),
            joint_names=np.array(
                [
                    "left_joint1",
                    "left_joint2",
                    "left_joint3",
                    "left_joint4",
                    "left_joint5",
                    "left_joint6",
                    "left_jointGripper",
                ]
            ),
            cup_world_pos=np.zeros(3, dtype=np.float64),
            cup_world_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            robot_initial_q=np.zeros(7, dtype=np.float64),
            support_center_world=np.zeros(3, dtype=np.float64),
            support_scale=np.ones(3, dtype=np.float64),
            remove_support_time=0.75,
        )

        q_arm, q_gripper = sample_joint_targets(demo, 0.25)

        np.testing.assert_allclose(q_arm, np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]))
        self.assertAlmostEqual(q_gripper, -0.25)

    def test_due_events_triggers_when_remove_support_time_is_crossed(self) -> None:
        events = due_events(
            prev_time=0.70,
            cur_time=0.80,
            event_times={"remove_support": 0.75},
        )

        self.assertEqual(events, ["remove_support"])


if __name__ == "__main__":
    unittest.main()
