from __future__ import annotations

import unittest

import numpy as np

from affordance_guided_interaction.utils.pose_alignment import (
    calibrate_pose_alignment,
    model_pose_to_sim_frame,
    relative_pose_in_parent_frame,
    rotation_distance_deg,
    row_major_rotation_to_column_major,
    sim_pose_to_model_frame,
)


def rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PoseAlignmentTests(unittest.TestCase):
    def test_row_major_rotation_to_column_major_transposes_rotation_block(self) -> None:
        expected_rot = rot_z(np.deg2rad(20.0)) @ rot_y(np.deg2rad(-15.0))
        usd_row_major_rot = expected_rot.T
        np.testing.assert_allclose(
            row_major_rotation_to_column_major(usd_row_major_rot),
            expected_rot,
            atol=1e-9,
        )

    def test_relative_pose_in_parent_frame_matches_manual_transform(self) -> None:
        parent_rot = rot_z(np.deg2rad(35.0))
        child_rot = parent_rot @ rot_x(np.deg2rad(70.0))
        parent_pos = np.array([0.3, -0.2, 0.1])
        local_pos_expected = np.array([0.4, 0.05, -0.02])
        child_pos = parent_pos + parent_rot @ local_pos_expected

        local_pos, local_rot = relative_pose_in_parent_frame(
            parent_pos, parent_rot, child_pos, child_rot
        )

        np.testing.assert_allclose(local_pos, local_pos_expected, atol=1e-9)
        np.testing.assert_allclose(
            local_rot, rot_x(np.deg2rad(70.0)), atol=1e-9
        )

    def test_calibrate_pose_alignment_recovers_constant_offset(self) -> None:
        model_pos = np.array([0.18, 0.11, 0.74])
        model_rot = rot_z(np.deg2rad(15.0)) @ rot_y(np.deg2rad(-10.0))
        align_pos = np.array([0.01, -0.02, 0.03])
        align_rot = rot_x(np.deg2rad(90.0)) @ rot_z(np.deg2rad(5.0))

        sim_pos, sim_rot = model_pose_to_sim_frame(
            model_pos, model_rot, align_pos, align_rot
        )
        got_pos, got_rot = calibrate_pose_alignment(
            model_pos, model_rot, sim_pos, sim_rot
        )

        np.testing.assert_allclose(got_pos, align_pos, atol=1e-9)
        np.testing.assert_allclose(got_rot, align_rot, atol=1e-9)

    def test_sim_pose_to_model_frame_round_trip(self) -> None:
        model_pos = np.array([0.2, -0.1, 0.6])
        model_rot = rot_z(np.deg2rad(-20.0)) @ rot_x(np.deg2rad(35.0))
        align_pos = np.array([-0.04, 0.03, 0.01])
        align_rot = rot_y(np.deg2rad(25.0))

        sim_pos, sim_rot = model_pose_to_sim_frame(
            model_pos, model_rot, align_pos, align_rot
        )
        round_trip_pos, round_trip_rot = sim_pose_to_model_frame(
            sim_pos, sim_rot, align_pos, align_rot
        )

        np.testing.assert_allclose(round_trip_pos, model_pos, atol=1e-9)
        np.testing.assert_allclose(round_trip_rot, model_rot, atol=1e-9)

    def test_rotation_distance_deg_matches_expected_angle(self) -> None:
        rot_a = np.eye(3)
        rot_b = rot_z(np.deg2rad(70.0))
        self.assertAlmostEqual(rotation_distance_deg(rot_a, rot_b), 70.0, places=6)


if __name__ == "__main__":
    unittest.main()
