from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.utils.pose_alignment import (
    relative_pose_in_parent_frame,
    row_major_rotation_to_column_major,
)
from affordance_guided_interaction.utils.runtime_env import configure_omniverse_client_environment
from affordance_guided_interaction.utils.usd_assets import to_usd_asset_path
from teleop_cup_grasp.grasp_demo_io import GraspDemo, save_grasp_demo_npz
from teleop_cup_grasp.moveit_bridge_python import (
    build_bridge_config,
    joint_trajectory_point_to_targets,
)


try:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectory
except ImportError as exc:  # pragma: no cover - depends on local ROS install
    raise SystemExit(
        "ROS 2 Humble and MoveIt 2 must be sourced before running "
        "src/teleop_cup_grasp/isaac_servo_bridge.py"
    ) from exc


configure_omniverse_client_environment(os.environ)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import carb
import omni
import omni.ui as ui
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics


DOOR_SCENE = PROJECT_ROOT / "assets/minimal_push_door/minimal_push_door.usda"
ROBOT_USD = PROJECT_ROOT / "assets/robot/uni_dingo_dual_arm.usd"
CUP_USD = PROJECT_ROOT / "assets/grasp_objects/cup/carry_cup.usda"
DEMO_OUTPUT_PATH = PROJECT_ROOT / "src/teleop_cup_grasp/grasp_demo.npz"
PICKUP_POINT_BASE = (0.45, 0.20, 0.55)
RECORD_DT = 1.0 / 60.0
ARM_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
]
GRIPPER_JOINT = "left_jointGripper"
ALL_RECORD_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
STEP_CM = 0.01
STEP_DEG = 2.0
LINEAR_GAIN = 4.0
ANGULAR_GAIN = 3.0
MAX_LINEAR_SPEED = 0.20
MAX_ANGULAR_SPEED = 1.20


def rot_from_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def ypr_from_rot(rot: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    return np.degrees([yaw, pitch, roll])


def rotation_vector_from_matrix(rot: np.ndarray) -> np.ndarray:
    cos_angle = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle


def clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec * (max_norm / norm)


def find_prim(stage: Usd.Stage, name: str):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def get_world_transform(stage: Usd.Stage, prim) -> tuple[np.ndarray, np.ndarray]:
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    row_major_rot = np.array([[xf[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
    rot = row_major_rotation_to_column_major(row_major_rot)
    pos = np.array(xf.ExtractTranslation(), dtype=np.float64)
    return rot, pos


def get_ee_in_baselink(stage: Usd.Stage, base_prim, ee_prim) -> tuple[np.ndarray, np.ndarray]:
    base_rot, base_pos = get_world_transform(stage, base_prim)
    ee_rot, ee_pos = get_world_transform(stage, ee_prim)
    return relative_pose_in_parent_frame(base_pos, base_rot, ee_pos, ee_rot)


class MoveItServoBridgeNode(Node):
    def __init__(self) -> None:
        self.config = build_bridge_config()
        super().__init__("isaac_moveit_servo_bridge")
        self.joint_state_pub = self.create_publisher(JointState, self.config.joint_state_topic, 10)
        self.twist_pub = self.create_publisher(TwistStamped, self.config.servo_twist_topic, 10)
        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            self.config.joint_command_topic,
            self._on_joint_trajectory,
            10,
        )
        self.start_servo_client = self.create_client(Trigger, self.config.start_servo_service)
        self.latest_joint_targets: dict[str, float] = {}

    def _on_joint_trajectory(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        self.latest_joint_targets = joint_trajectory_point_to_targets(
            msg.joint_names,
            msg.points[0].positions,
        )

    def publish_joint_state(self, joint_names: list[str], positions: list[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(value) for value in positions]
        self.joint_state_pub.publish(msg)

    def publish_twist_from_pose_error(
        self,
        current_pos: np.ndarray,
        current_rot: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
    ) -> None:
        pos_err = target_pos - current_pos
        ori_err = rotation_vector_from_matrix(target_rot @ current_rot.T)
        linear = clamp_norm(LINEAR_GAIN * pos_err, MAX_LINEAR_SPEED)
        angular = clamp_norm(ANGULAR_GAIN * ori_err, MAX_ANGULAR_SPEED)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.config.planning_frame
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self.twist_pub.publish(msg)

    def try_start_servo(self, retries: int = 20) -> bool:
        for _ in range(retries):
            if self.start_servo_client.wait_for_service(timeout_sec=0.2):
                future = self.start_servo_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                result = future.result()
                return bool(result and result.success)
        return False


def main() -> int:
    rclpy.init()
    bridge_node = MoveItServoBridgeNode()

    omni.usd.get_context().new_stage()
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()

    if DOOR_SCENE.exists():
        stage.GetRootLayer().subLayerPaths.append(str(DOOR_SCENE))
    else:
        PhysicsSchemaTools.addPhysicsScene(stage, "/World/PhysicsScene")
        light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/SphereLight"))
        light.GetRadiusAttr().Set(2.0)
        light.GetIntensityAttr().Set(30000.0)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 5.0))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for _ in range(20):
        simulation_app.update()

    robot_prim = stage.DefinePrim("/World/Robot", "Xform")
    robot_prim.GetReferences().AddReference(to_usd_asset_path(ROBOT_USD))
    for _ in range(10):
        simulation_app.update()
    robot_xf = UsdGeom.Xformable(robot_prim)
    robot_xf.ClearXformOpOrder()
    robot_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.12))

    ground_path = "/World/TeleopGround"
    if not stage.GetPrimAtPath(ground_path).IsValid():
        ground = UsdGeom.Cube.Define(stage, Sdf.Path(ground_path))
        ground.GetSizeAttr().Set(1.0)
        ground.GetDisplayColorAttr().Set([(0.52, 0.54, 0.56)])
        ground_xf = UsdGeom.Xformable(ground.GetPrim())
        ground_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, 0.0, -0.05))
        ground_xf.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.1))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    base_link_prim = stage.GetPrimAtPath("/World/Robot/base_link")
    if base_link_prim.IsValid():
        base_world = UsdGeom.Xformable(base_link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        cup_world = base_world.Transform(Gf.Vec3d(*PICKUP_POINT_BASE))
    else:
        cup_world = Gf.Vec3d(*PICKUP_POINT_BASE)

    support_scale = (0.12, 0.12, 0.02)
    support_center = Gf.Vec3f(float(cup_world[0]), float(cup_world[1]), float(cup_world[2]) - 0.5 * support_scale[2])
    support_prim = UsdGeom.Cube.Define(stage, Sdf.Path("/World/PickupSupport")).GetPrim()
    support_geom = UsdGeom.Cube(support_prim)
    support_geom.GetSizeAttr().Set(1.0)
    support_geom.GetDisplayColorAttr().Set([(0.45, 0.37, 0.25)])
    support_xf = UsdGeom.Xformable(support_prim)
    support_xf.ClearXformOpOrder()
    support_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(support_center)
    support_xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*support_scale))
    UsdPhysics.CollisionAPI.Apply(support_prim)

    cup_prim = stage.DefinePrim("/World/CarryObject", "Xform")
    cup_prim.GetReferences().AddReference(to_usd_asset_path(CUP_USD))
    cup_xf = UsdGeom.Xformable(cup_prim)
    cup_xf.ClearXformOpOrder()
    cup_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(float(cup_world[0]), float(cup_world[1]), float(cup_world[2]))
    )
    for _ in range(20):
        simulation_app.update()

    omni.timeline.get_timeline_interface().play()
    for _ in range(120):
        simulation_app.update()

    dc = _dynamic_control.acquire_dynamic_control_interface()
    art = dc.get_articulation("/World/Robot/base_link")
    if art == _dynamic_control.INVALID_HANDLE:
        raise SystemExit("Cannot get articulation from /World/Robot/base_link")

    def set_pos(joint_name: str, rad: float) -> None:
        dof = dc.find_articulation_dof(art, joint_name)
        if dof != _dynamic_control.INVALID_HANDLE:
            dc.set_dof_position_target(dof, float(rad))

    def get_pos(joint_name: str) -> float:
        dof = dc.find_articulation_dof(art, joint_name)
        if dof == _dynamic_control.INVALID_HANDLE:
            return 0.0
        return float(dc.get_dof_position(dof))

    base_prim = stage.GetPrimAtPath("/World/Robot/base_link")
    ee_prim = stage.GetPrimAtPath("/World/Robot/left_gripperStator")
    if not base_prim.IsValid():
        base_prim = find_prim(stage, "base_link")
    if not ee_prim.IsValid():
        ee_prim = find_prim(stage, "left_gripperStator")

    cur_ee_pos, cur_ee_rot = get_ee_in_baselink(stage, base_prim, ee_prim)
    ik_target_pos = cur_ee_pos.copy()
    ik_target_ypr = list(ypr_from_rot(cur_ee_rot))

    pos_labels: dict[int, ui.Label] = {}
    ypr_labels: dict[int, ui.Label] = {}
    cur_pos_labels: dict[int, ui.Label] = {}
    cur_ypr_labels: dict[int, ui.Label] = {}

    window = ui.Window("Cup Grasp Teleop (MoveIt Servo)", width=560, height=480)
    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("EE Control (base_link frame)", style={"font_size": 14})
            ui.Spacer(height=4)

            with ui.HStack(height=20, spacing=4):
                ui.Label("", width=80)
                ui.Label("", width=50)
                ui.Label("Target", width=65, style={"font_size": 11, "color": 0xFF88FF88})
                ui.Label("", width=50)
                ui.Label("Current", width=65, style={"font_size": 11, "color": 0xFFFFCC66})

            for idx, label in enumerate(["X (fwd)", "Y (left)", "Z (up)"]):
                with ui.HStack(height=28, spacing=4):
                    ui.Label(label, width=80)

                    def make_minus(axis: int = idx):
                        def _click():
                            ik_target_pos[axis] -= STEP_CM

                        return _click

                    def make_plus(axis: int = idx):
                        def _click():
                            ik_target_pos[axis] += STEP_CM

                        return _click

                    ui.Button("  -  ", width=50, clicked_fn=make_minus())
                    pos_labels[idx] = ui.Label(f"{ik_target_pos[idx]:.3f}", width=65, style={"color": 0xFF88FF88})
                    ui.Button("  +  ", width=50, clicked_fn=make_plus())
                    cur_pos_labels[idx] = ui.Label("---", width=65, style={"color": 0xFFFFCC66})

            ui.Spacer(height=6)
            ui.Separator(height=2)
            ui.Spacer(height=4)

            with ui.HStack(height=20, spacing=4):
                ui.Label("", width=80)
                ui.Label("", width=50)
                ui.Label("Target", width=55, style={"font_size": 11, "color": 0xFF88FF88})
                ui.Label("", width=50)
                ui.Label("Current", width=55, style={"font_size": 11, "color": 0xFFFFCC66})

            for idx, label in enumerate(["Yaw", "Pitch", "Roll"]):
                with ui.HStack(height=28, spacing=4):
                    ui.Label(label, width=80)

                    def make_minus(axis: int = idx):
                        def _click():
                            ik_target_ypr[axis] -= STEP_DEG

                        return _click

                    def make_plus(axis: int = idx):
                        def _click():
                            ik_target_ypr[axis] += STEP_DEG

                        return _click

                    ui.Button("  -  ", width=50, clicked_fn=make_minus())
                    ypr_labels[idx] = ui.Label(f"{ik_target_ypr[idx]:.1f}°", width=55, style={"color": 0xFF88FF88})
                    ui.Button("  +  ", width=50, clicked_fn=make_plus())
                    cur_ypr_labels[idx] = ui.Label("---", width=55, style={"color": 0xFFFFCC66})

            ui.Spacer(height=6)
            ui.Separator(height=2)
            ui.Spacer(height=4)

            with ui.HStack(height=26, spacing=4):
                ui.Label("Gripper", width=80)
                slider = ui.FloatSlider(min=-90, max=0)
                slider.model.set_value(np.degrees(get_pos(GRIPPER_JOINT)))
                slider_value = ui.Label(f"{np.degrees(get_pos(GRIPPER_JOINT)):.1f}°", width=55)

                def _on_gripper(model, label_widget=slider_value):
                    degrees = model.as_float
                    label_widget.text = f"{degrees:.1f}°"
                    set_pos(GRIPPER_JOINT, np.radians(degrees))

                slider.model.add_value_changed_fn(_on_gripper)

            ui.Spacer(height=6)
            ui.Label("[SPACE] remove support & save", style={"font_size": 12, "color": 0xFFAAAAFF})

    record_times: list[float] = []
    record_q_arm: list[np.ndarray] = []
    record_q_gripper: list[float] = []
    recording = True
    remove_support_time = None

    def read_cup_pose() -> tuple[np.ndarray, np.ndarray]:
        rigid_body = dc.get_rigid_body("/World/CarryObject")
        if rigid_body != _dynamic_control.INVALID_HANDLE:
            pose = dc.get_rigid_body_pose(rigid_body)
            return (
                np.array([pose.p.x, pose.p.y, pose.p.z], dtype=np.float64),
                np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z], dtype=np.float64),
            )
        return (
            np.array([float(cup_world[0]), float(cup_world[1]), float(cup_world[2])], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    cup_initial_pos, cup_initial_quat = read_cup_pose()
    robot_initial_q = np.array([get_pos(joint) for joint in ALL_RECORD_JOINTS], dtype=np.float64)
    support_center_world = np.array([float(support_center[0]), float(support_center[1]), float(support_center[2])], dtype=np.float64)
    support_scale_arr = np.array(support_scale, dtype=np.float64)

    def on_kb(ev, *args, **kwargs):
        nonlocal recording, remove_support_time
        if ev.type == carb.input.KeyboardEventType.KEY_PRESS and ev.input == carb.input.KeyboardInput.SPACE:
            print("\n[Space] Removing support...")
            support_prim.GetAttribute("visibility").Set("invisible")
            collision_api = UsdPhysics.CollisionAPI(support_prim)
            if collision_api:
                collision_api.GetCollisionEnabledAttr().Set(False)
            if record_times and remove_support_time is None:
                remove_support_time = float(record_times[-1])
            recording = False

    input_iface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    subscription = input_iface.subscribe_to_keyboard_events(keyboard, on_kb)

    servo_started = bridge_node.try_start_servo()
    print(f"MoveIt Servo start: {'ok' if servo_started else 'missing service / failed'}")

    try:
        while simulation_app.is_running():
            simulation_app.update()
            rclpy.spin_once(bridge_node, timeout_sec=0.0)

            q_arm = np.array([get_pos(joint) for joint in ARM_JOINTS], dtype=np.float64)
            q_gripper = float(get_pos(GRIPPER_JOINT))
            bridge_node.publish_joint_state(ALL_RECORD_JOINTS, [*q_arm.tolist(), q_gripper])

            cur_ee_pos, cur_ee_rot = get_ee_in_baselink(stage, base_prim, ee_prim)
            cur_ypr = ypr_from_rot(cur_ee_rot)
            target_rot = rot_from_ypr(
                np.radians(ik_target_ypr[0]),
                np.radians(ik_target_ypr[1]),
                np.radians(ik_target_ypr[2]),
            )

            bridge_node.publish_twist_from_pose_error(cur_ee_pos, cur_ee_rot, ik_target_pos, target_rot)

            if bridge_node.latest_joint_targets:
                for joint_name in ARM_JOINTS:
                    if joint_name in bridge_node.latest_joint_targets:
                        set_pos(joint_name, bridge_node.latest_joint_targets[joint_name])

            for axis in range(3):
                pos_labels[axis].text = f"{ik_target_pos[axis]:.3f}"
                ypr_labels[axis].text = f"{ik_target_ypr[axis]:.1f}°"
                cur_pos_labels[axis].text = f"{cur_ee_pos[axis]:.3f}"
                cur_ypr_labels[axis].text = f"{cur_ypr[axis]:.1f}°"

            if recording:
                record_times.append(len(record_times) * RECORD_DT)
                record_q_arm.append(q_arm.copy())
                record_q_gripper.append(q_gripper)
            elif record_times:
                demo = GraspDemo(
                    t=np.asarray(record_times, dtype=np.float64),
                    q_arm=np.asarray(record_q_arm, dtype=np.float64),
                    q_gripper=np.asarray(record_q_gripper, dtype=np.float64),
                    joint_names=np.array(ALL_RECORD_JOINTS),
                    cup_world_pos=cup_initial_pos,
                    cup_world_quat_wxyz=cup_initial_quat,
                    robot_initial_q=robot_initial_q,
                    support_center_world=support_center_world,
                    support_scale=support_scale_arr,
                    remove_support_time=(
                        float(remove_support_time) if remove_support_time is not None else float(record_times[-1])
                    ),
                )
                save_grasp_demo_npz(DEMO_OUTPUT_PATH, demo)
                print(f"Saved {len(record_times)} samples -> {DEMO_OUTPUT_PATH}")
                record_times = []
                record_q_arm = []
                record_q_gripper = []
    except KeyboardInterrupt:
        print("\nUser interrupted")
    finally:
        input_iface.unsubscribe_to_keyboard_events(keyboard, subscription)
        try:
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        bridge_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
