from __future__ import annotations

import unittest

from teleop_cup_grasp.moveit_bridge_python import (
    build_bridge_config,
    joint_trajectory_point_to_targets,
    make_pose_goal,
)


class MoveItBridgePythonTests(unittest.TestCase):
    def test_bridge_topics_default_to_servo_node_pose_and_trajectory_topics(self) -> None:
        config = build_bridge_config()

        self.assertEqual(config.pose_topic, "/servo_node/target_pose")
        self.assertEqual(config.servo_twist_topic, "/servo_node/delta_twist_cmds")
        self.assertEqual(config.joint_command_topic, "/servo_node/joint_trajectory")
        self.assertEqual(config.status_topic, "/servo_node/status")
        self.assertEqual(config.command_type_service, "/servo_node/switch_command_type")
        self.assertEqual(config.start_servo_service, "/servo_node/start_servo")
        self.assertEqual(config.move_group_name, "left_arm")
        self.assertEqual(config.planning_frame, "base_link")
        self.assertEqual(config.ee_frame, "left_gripperStator")

    def test_make_pose_goal_uses_wxyz_quaternion_and_frame_id(self) -> None:
        pose_goal = make_pose_goal(
            position_xyz=(0.29, 0.16, 0.64),
            quat_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
            frame_id="base_link",
        )

        self.assertEqual(pose_goal["header"]["frame_id"], "base_link")
        self.assertAlmostEqual(pose_goal["pose"]["position"]["x"], 0.29)
        self.assertAlmostEqual(pose_goal["pose"]["orientation"]["w"], 0.70710678)
        self.assertAlmostEqual(pose_goal["pose"]["orientation"]["y"], 0.70710678)

    def test_joint_trajectory_point_to_targets_maps_joint_names_to_positions(self) -> None:
        targets = joint_trajectory_point_to_targets(
            joint_names=["left_joint1", "left_joint2", "left_joint3"],
            positions=[0.1, 0.2, 0.3],
        )

        self.assertEqual(
            targets,
            {"left_joint1": 0.1, "left_joint2": 0.2, "left_joint3": 0.3},
        )


if __name__ == "__main__":
    unittest.main()
