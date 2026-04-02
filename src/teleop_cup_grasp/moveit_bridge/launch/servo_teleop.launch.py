from __future__ import annotations

from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("cup_grasp_moveit_bridge"))
    urdf_path = package_share / "urdf/uni_dingo_dual_arm_absolute.urdf"
    srdf_path = package_share / "config/robot.srdf"
    kinematics_path = package_share / "config/kinematics.yaml"
    joint_limits_path = package_share / "config/joint_limits.yaml"
    controllers_path = package_share / "config/moveit_controllers.yaml"
    servo_params_path = package_share / "config/servo_parameters.yaml"
    rviz_config_path = package_share / "rviz/servo.rviz"

    robot_description = {"robot_description": _load_text(urdf_path)}
    robot_description_semantic = {"robot_description_semantic": _load_text(srdf_path)}
    robot_description_kinematics = {"robot_description_kinematics": _load_yaml(kinematics_path)}
    robot_description_planning = {"robot_description_planning": _load_yaml(joint_limits_path)}
    moveit_controller_manager = _load_yaml(controllers_path)
    servo_parameters = {"moveit_servo": _load_yaml(servo_params_path)}

    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[robot_description],
            ),
            Node(
                package="moveit_servo",
                executable="servo_node_main",
                name="servo_node",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    robot_description_kinematics,
                    robot_description_planning,
                    moveit_controller_manager,
                    servo_parameters,
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", str(rviz_config_path)],
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    robot_description_kinematics,
                    robot_description_planning,
                ],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
