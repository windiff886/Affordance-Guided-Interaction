"""
在 Isaac Sim 中查看和控制双臂机器人 USD。

功能:
    1. 加载 assets/robot/usd/uni_dingo_dual_arm.usd
    2. GUI 切换 visual / collision / both 显示模式
    3. GUI 滑条控制底盘线速度/角速度
    4. GUI 滑条控制双臂、夹爪、云台关节

用法:
    conda activate isaaclab
    python assets/robot/scripts/visualize_in_isaacsim.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD_PATH = PROJECT_ROOT / "assets/robot/usd/uni_dingo_dual_arm.usd"
DEFAULT_URDF_PATH = PROJECT_ROOT / "assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf"

WHEEL_RADIUS_METERS = 0.049
TRACK_WIDTH_METERS = 0.4523232
WHEEL_JOINTS = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
DISPLAY_MODES = ("visual", "collision", "both")


@dataclass(frozen=True)
class JointControlSpec:
    name: str
    lower_rad: float
    upper_rad: float


def ground_visual_spec() -> dict[str, object]:
    thickness = 0.05
    return {
        "size_xy": 30.0,
        "thickness": thickness,
        "translate": (0.0, 0.0, -thickness / 2.0),
        "color": (0.58, 0.6, 0.62),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim 机器人 USD 可视化与滑条控制")
    parser.add_argument(
        "--usd",
        type=Path,
        default=DEFAULT_USD_PATH,
        help="机器人 USD 路径",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="用于读取关节范围的 URDF 路径",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无窗口模式运行（仅调试，不推荐可视化时使用）",
    )
    return parser


def display_mode_flags(mode: str) -> tuple[bool, bool]:
    normalized = mode.strip().lower()
    if normalized == "visual":
        return True, False
    if normalized == "collision":
        return False, True
    if normalized == "both":
        return True, True
    raise ValueError(f"Unsupported display mode: {mode}")


def compute_wheel_targets(
    *,
    linear_velocity: float,
    angular_velocity: float,
    wheel_radius: float = WHEEL_RADIUS_METERS,
    track_width: float = TRACK_WIDTH_METERS,
) -> dict[str, float]:
    half_track = track_width / 2.0
    left_linear = linear_velocity - angular_velocity * half_track
    right_linear = linear_velocity + angular_velocity * half_track
    left_target = left_linear / wheel_radius
    right_target = right_linear / wheel_radius
    return {
        "front_left_wheel": left_target,
        "rear_left_wheel": left_target,
        "front_right_wheel": right_target,
        "rear_right_wheel": right_target,
    }


def collect_joint_specs(urdf_path: Path) -> list[JointControlSpec]:
    root = ET.parse(str(urdf_path)).getroot()
    specs: list[JointControlSpec] = []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        if name in WHEEL_JOINTS:
            continue
        joint_type = joint.attrib.get("type", "")
        if joint_type in {"fixed", "floating", "planar"}:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.attrib.get("lower", -math.pi))
        upper = float(limit.attrib.get("upper", math.pi))
        specs.append(JointControlSpec(name=name, lower_rad=lower, upper_rad=upper))
    return specs


def collect_link_names(urdf_path: Path) -> list[str]:
    root = ET.parse(str(urdf_path)).getroot()
    return [link.attrib["name"] for link in root.findall("link")]


def slider_label(name: str) -> str:
    return name.replace("_", " ")


def shared_geometry_scope_name(kind: str) -> str:
    if kind == "visuals":
        return "visuals"
    if kind == "collisions":
        return "colliders"
    raise ValueError(f"Unsupported geometry kind: {kind}")


def geometry_root_paths(
    robot_root_path: str,
    robot_container_path: str,
    link_names: list[str],
    kind: str,
) -> list[str]:
    paths: list[str] = []
    shared_scope = shared_geometry_scope_name(kind)
    for link_name in link_names:
        paths.append(f"{robot_root_path}/{link_name}/{kind}")
        paths.append(f"{robot_container_path}/{shared_scope}/{link_name}")
    return paths


def run_viewer(args: argparse.Namespace) -> int:
    usd_path = args.usd.resolve()
    urdf_path = args.urdf.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD 不存在: {usd_path}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

    joint_specs = collect_joint_specs(urdf_path)
    link_names = collect_link_names(urdf_path)

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": args.headless, "width": 1440, "height": 900})
    try:
        import omni
        import omni.ui as ui
        from omni.isaac.dynamic_control import _dynamic_control
        from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

        def log(message: str) -> None:
            print(message, flush=True)

        omni.usd.get_context().new_stage()
        sim_app.update()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        PhysicsSchemaTools.addGroundPlane(
            stage,
            "/World/GroundPlane",
            "Z",
            1500.0,
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(0.5),
        )
        ground_spec = ground_visual_spec()
        ground_visual = UsdGeom.Cube.Define(stage, Sdf.Path("/World/GroundVisual"))
        ground_visual.CreateSizeAttr(1.0)
        ground_visual.CreateDisplayColorAttr(
            [Gf.Vec3f(*ground_spec["color"])]
        )
        ground_xform = UsdGeom.Xformable(ground_visual.GetPrim())
        ground_xform.AddTranslateOp().Set(Gf.Vec3d(*ground_spec["translate"]))
        ground_xform.AddScaleOp().Set(
            Gf.Vec3d(
                ground_spec["size_xy"],
                ground_spec["size_xy"],
                ground_spec["thickness"],
            )
        )
        UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(500)
        UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(250)

        physics_scene = stage.GetPrimAtPath("/World/PhysicsScene")
        if not physics_scene.IsValid():
            UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
            physics_scene = stage.GetPrimAtPath("/World/PhysicsScene")
        physics_scene.GetAttribute("physics:gravityDirection").Set(Gf.Vec3f(0, 0, -1))
        physics_scene.GetAttribute("physics:gravityMagnitude").Set(9.81)

        robot_container = stage.DefinePrim("/World/Robot")
        robot_container.GetReferences().AddReference(str(usd_path))
        sim_app.update()
        sim_app.update()

        robot_root_path = None
        for child in robot_container.GetChildren():
            if child and child.IsValid():
                robot_root_path = str(child.GetPath())
                break
        if robot_root_path is None:
            raise RuntimeError("未找到机器人根 prim")

        def set_visibility_for_paths(paths: list[str], visible: bool) -> None:
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                imageable = UsdGeom.Imageable(prim)
                if not imageable:
                    continue
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

        visual_paths = geometry_root_paths(
            robot_root_path,
            str(robot_container.GetPath()),
            link_names,
            "visuals",
        )
        collision_paths = geometry_root_paths(
            robot_root_path,
            str(robot_container.GetPath()),
            link_names,
            "collisions",
        )

        current_mode = {"value": "visual"}

        def apply_display_mode(mode: str) -> None:
            show_visual, show_collision = display_mode_flags(mode)
            set_visibility_for_paths(visual_paths, show_visual)
            set_visibility_for_paths(collision_paths, show_collision)
            current_mode["value"] = mode
            if status_label is not None:
                status_label.text = f"Display: {mode}"

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(60):
            sim_app.update()

        dc = _dynamic_control.acquire_dynamic_control_interface()
        articulation = None
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root_path)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation = dc.get_articulation(str(prim.GetPath()))
                if articulation != 0:
                    break
        if articulation is None or articulation == 0:
            raise RuntimeError("未找到机器人 articulation")

        dof_map: dict[str, int] = {}
        for index in range(dc.get_articulation_dof_count(articulation)):
            dof = dc.get_articulation_dof(articulation, index)
            dof_map[dc.get_dof_name(dof)] = dof

        class JointSlider:
            def __init__(self, spec: JointControlSpec):
                self.spec = spec
                self.lower_deg = math.degrees(spec.lower_rad)
                self.upper_deg = math.degrees(spec.upper_rad)
                self.default_deg = min(max(0.0, self.lower_deg), self.upper_deg)
                self.model = ui.SimpleFloatModel(self.default_deg)
                self.value_label = None

        joint_sliders = [JointSlider(spec) for spec in joint_specs if spec.name in dof_map]
        base_linear_model = ui.SimpleFloatModel(0.0)
        base_angular_model = ui.SimpleFloatModel(0.0)

        status_label = None
        base_linear_value = None
        base_angular_value = None

        def reset_targets() -> None:
            base_linear_model.set_value(0.0)
            base_angular_model.set_value(0.0)
            for joint_slider in joint_sliders:
                joint_slider.model.set_value(joint_slider.default_deg)
            apply_display_mode("visual")

        def make_slider_row(
            label_text: str,
            model,
            range_min: float,
            range_max: float,
            step: float,
            value_format: str,
        ):
            value_label = None
            with ui.HStack(height=28, spacing=6):
                ui.Label(label_text, width=145)

                def on_minus():
                    model.set_value(max(range_min, model.get_value_as_float() - step))

                def on_plus():
                    model.set_value(min(range_max, model.get_value_as_float() + step))

                ui.Button("-", width=24, clicked_fn=on_minus)
                ui.FloatSlider(model=model, min=range_min, max=range_max)
                ui.Button("+", width=24, clicked_fn=on_plus)
                value_label = ui.Label(value_format.format(model.get_value_as_float()), width=70)
            return value_label

        window = ui.Window("Robot USD Viewer", width=520, height=860)
        with window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8):
                    ui.Label("Display Mode", height=22)
                    with ui.HStack(height=32, spacing=6):
                        ui.Button("Visual", clicked_fn=lambda: apply_display_mode("visual"))
                        ui.Button("Collision", clicked_fn=lambda: apply_display_mode("collision"))
                        ui.Button("Both", clicked_fn=lambda: apply_display_mode("both"))

                    ui.Spacer(height=4)
                    ui.Label("Base Control", height=22)
                    base_linear_value = make_slider_row(
                        "Linear Velocity (m/s)",
                        base_linear_model,
                        -1.5,
                        1.5,
                        0.05,
                        "{:.2f}",
                    )
                    base_angular_value = make_slider_row(
                        "Angular Velocity (rad/s)",
                        base_angular_model,
                        -2.5,
                        2.5,
                        0.05,
                        "{:.2f}",
                    )

                    ui.Spacer(height=4)
                    ui.Label("Joint Control (deg)", height=22)
                    for joint_slider in joint_sliders:
                        joint_slider.value_label = make_slider_row(
                            slider_label(joint_slider.spec.name),
                            joint_slider.model,
                            joint_slider.lower_deg,
                            joint_slider.upper_deg,
                            1.0,
                            "{:.1f}",
                        )

                    ui.Spacer(height=6)
                    with ui.HStack(height=32, spacing=8):
                        ui.Button("Reset", clicked_fn=reset_targets)
                        ui.Button("Stop Base", clicked_fn=lambda: (base_linear_model.set_value(0.0), base_angular_model.set_value(0.0)))

                    status_label = ui.Label("Display: visual", height=20)

        apply_display_mode("visual")

        log(f"Loaded USD: {usd_path}")
        log(f"Articulation DOFs: {len(dof_map)}")

        while sim_app.is_running():
            sim_app.update()
            dc.wake_up_articulation(articulation)

            wheel_targets = compute_wheel_targets(
                linear_velocity=base_linear_model.get_value_as_float(),
                angular_velocity=base_angular_model.get_value_as_float(),
            )
            for joint_name, target in wheel_targets.items():
                if joint_name in dof_map:
                    dc.set_dof_velocity_target(dof_map[joint_name], target)

            for joint_slider in joint_sliders:
                target_deg = joint_slider.model.get_value_as_float()
                dc.set_dof_position_target(
                    dof_map[joint_slider.spec.name],
                    math.radians(target_deg),
                )
                if joint_slider.value_label is not None:
                    joint_slider.value_label.text = f"{target_deg:.1f}"

            if base_linear_value is not None:
                base_linear_value.text = f"{base_linear_model.get_value_as_float():.2f}"
            if base_angular_value is not None:
                base_angular_value.text = f"{base_angular_model.get_value_as_float():.2f}"

        return 0
    finally:
        sim_app.close()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
