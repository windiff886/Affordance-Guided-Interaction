from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = PROJECT_ROOT / "src/teleop_cup_grasp/start_moveit_capture.sh"


class CaptureLauncherLayoutTests(unittest.TestCase):
    def test_capture_launcher_contains_moveit_launch_and_isaac_bridge_commands(self) -> None:
        content = LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertIn("ros2 launch cup_grasp_moveit_bridge servo_teleop.launch.py", content)
        self.assertIn("python", content)
        self.assertIn("isaac_servo_bridge.py", content)
        self.assertIn("trap cleanup EXIT", content)
        self.assertIn("/opt/ros/humble/setup.bash", content)


if __name__ == "__main__":
    unittest.main()
