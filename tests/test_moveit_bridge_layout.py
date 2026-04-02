from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = PROJECT_ROOT / "src/teleop_cup_grasp/moveit_bridge"


class MoveItBridgeLayoutTests(unittest.TestCase):
    def test_moveit_bridge_package_contains_required_files(self) -> None:
        required_paths = [
            BRIDGE_ROOT / "package.xml",
            BRIDGE_ROOT / "CMakeLists.txt",
            BRIDGE_ROOT / "launch/servo_teleop.launch.py",
            BRIDGE_ROOT / "config/robot.srdf",
            BRIDGE_ROOT / "config/kinematics.yaml",
            BRIDGE_ROOT / "config/joint_limits.yaml",
            BRIDGE_ROOT / "config/moveit_controllers.yaml",
            BRIDGE_ROOT / "config/servo_parameters.yaml",
            BRIDGE_ROOT / "rviz/servo.rviz",
        ]

        missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_paths if not path.is_file()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
