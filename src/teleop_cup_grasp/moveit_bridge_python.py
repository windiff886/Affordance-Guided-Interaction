from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


ARM_JOINT_NAMES = (
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
)
GRIPPER_JOINT_NAME = "left_jointGripper"


@dataclass(frozen=True)
class MoveItBridgeConfig:
    pose_topic: str
    servo_twist_topic: str
    joint_command_topic: str
    joint_state_topic: str
    status_topic: str
    command_type_service: str
    start_servo_service: str
    move_group_name: str
    planning_frame: str
    ee_frame: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_name: str


def build_bridge_config(
    servo_namespace: str = "/servo_node",
    move_group_name: str = "left_arm",
    planning_frame: str = "base_link",
    ee_frame: str = "left_gripperStator",
) -> MoveItBridgeConfig:
    servo_namespace = servo_namespace.rstrip("/")
    if not servo_namespace:
        servo_namespace = "/servo_node"
    return MoveItBridgeConfig(
        pose_topic=f"{servo_namespace}/target_pose",
        servo_twist_topic=f"{servo_namespace}/delta_twist_cmds",
        joint_command_topic=f"{servo_namespace}/joint_trajectory",
        joint_state_topic="/joint_states",
        status_topic=f"{servo_namespace}/status",
        command_type_service=f"{servo_namespace}/switch_command_type",
        start_servo_service=f"{servo_namespace}/start_servo",
        move_group_name=move_group_name,
        planning_frame=planning_frame,
        ee_frame=ee_frame,
        arm_joint_names=ARM_JOINT_NAMES,
        gripper_joint_name=GRIPPER_JOINT_NAME,
    )


def make_pose_goal(
    position_xyz: Sequence[float],
    quat_wxyz: Sequence[float],
    frame_id: str,
) -> dict[str, dict[str, dict[str, float] | str]]:
    px, py, pz = [float(v) for v in position_xyz]
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]
    return {
        "header": {"frame_id": frame_id},
        "pose": {
            "position": {"x": px, "y": py, "z": pz},
            "orientation": {"w": qw, "x": qx, "y": qy, "z": qz},
        },
    }


def joint_trajectory_point_to_targets(
    joint_names: Iterable[str],
    positions: Sequence[float],
) -> dict[str, float]:
    joint_names = list(joint_names)
    if len(joint_names) != len(positions):
        raise ValueError("joint_names and positions must have the same length")
    return {joint_name: float(position) for joint_name, position in zip(joint_names, positions)}
