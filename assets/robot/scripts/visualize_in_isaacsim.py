"""
Isaac Sim Lite Robot Viewer

Visualize uni_dingo_lite.usd with display mode switching and joint control.

Features:
    1. Load uni_dingo_lite.usd
    2. GUI toggle: visual / collision / both
    3. GUI sliders for arm joint control

Usage:
    conda activate isaaclab
    python assets/robot/scripts/visualize_in_isaacsim.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD_PATH = PROJECT_ROOT / "assets/robot/usd/uni_dingo_lite.usd"
DEFAULT_URDF_PATH = PROJECT_ROOT / "assets/robot/urdf/uni_dingo_lite.urdf"

ARM_JOINTS = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6", "left_jointGripper",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6", "right_jointGripper",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite Robot USD Viewer")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD_PATH)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--headless", action="store_true")
    return parser


def collect_link_names(urdf_path: Path) -> list[str]:
    root = ET.parse(str(urdf_path)).getroot()
    return [link.attrib["name"] for link in root.findall("link")]


def collect_joint_specs(urdf_path: Path) -> list[tuple[str, float, float]]:
    root = ET.parse(str(urdf_path)).getroot()
    specs = []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        if name not in ARM_JOINTS:
            continue
        jtype = joint.attrib.get("type", "")
        if jtype in {"fixed", "floating", "planar"}:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.attrib.get("lower", -math.pi))
        upper = float(limit.attrib.get("upper", math.pi))
        specs.append((name, lower, upper))
    return specs


def run_viewer(args: argparse.Namespace) -> int:
    usd_path = args.usd.resolve()
    urdf_path = args.urdf.resolve()
    for p in (usd_path, urdf_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    link_names = collect_link_names(urdf_path)
    joint_specs = collect_joint_specs(urdf_path)

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": args.headless, "width": 1440, "height": 900})

    import omni
    import omni.ui as ui
    from omni.isaac.dynamic_control import _dynamic_control
    from pxr import (
        Gf,
        PhysicsSchemaTools,
        Sdf,
        Usd,
        UsdGeom,
        UsdLux,
        UsdPhysics,
    )

    def log(msg: str) -> None:
        print(msg, flush=True)

    # ── Stage setup ────────────────────────────────────────────
    omni.usd.get_context().new_stage()
    sim_app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 1500.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
    )
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(500)
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(250)

    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ps = stage.GetPrimAtPath("/World/PhysicsScene")
    ps.GetAttribute("physics:gravityDirection").Set(Gf.Vec3f(0, 0, -1))
    ps.GetAttribute("physics:gravityMagnitude").Set(9.81)
    sim_app.update()

    # ── Load robot ─────────────────────────────────────────────
    robot_prim = stage.DefinePrim("/World/Robot")
    robot_prim.GetReferences().AddReference(str(usd_path))
    sim_app.update()

    # ── Find robot root (filter out non-robot scopes) ──────────
    robot_root_path = None
    for child in robot_prim.GetChildren():
        if not (child and child.IsValid() and child.GetTypeName() == "Xform"):
            continue
        child_name = child.GetName()
        if child_name not in ("visuals", "colliders", "meshes", "Looks"):
            robot_root_path = str(child.GetPath())
            break
    if robot_root_path is None:
        raise RuntimeError("Robot root prim not found")
    log(f"Robot root: {robot_root_path}")

    # ── Collect geometry paths ─────────────────────────────────
    def collect_geometry_paths(kind: str) -> list[str]:
        scope_name = "visuals" if kind == "visuals" else "colliders"
        paths = []
        for link_name in link_names:
            paths.append(f"{robot_root_path}/{link_name}/{kind}")
            paths.append(f"/World/Robot/{scope_name}/{link_name}")
        return paths

    visual_paths = collect_geometry_paths("visuals")
    collision_paths = collect_geometry_paths("collisions")

    # ── Visibility helpers ─────────────────────────────────────
    def set_visibility(paths: list[str], visible: bool) -> None:
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

    def fix_guide_purpose(paths: list[str]) -> None:
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            for p in Usd.PrimRange(prim):
                img = UsdGeom.Imageable(p)
                if img:
                    attr = img.GetPurposeAttr()
                    if attr and attr.Get() == "guide":
                        attr.Set("default")

    display_mode_label = None

    def apply_display_mode(mode: str) -> None:
        if mode == "visual":
            set_visibility(visual_paths, True)
            set_visibility(collision_paths, False)
        elif mode == "collision":
            fix_guide_purpose(collision_paths)
            set_visibility(visual_paths, False)
            set_visibility(collision_paths, True)
        elif mode == "both":
            fix_guide_purpose(collision_paths)
            set_visibility(visual_paths, True)
            set_visibility(collision_paths, True)
        if display_mode_label is not None:
            display_mode_label.text = f"Display: {mode}"

    apply_display_mode("visual")

    # ── Start simulation ───────────────────────────────────────
    omni.timeline.get_timeline_interface().play()
    for _ in range(60):
        sim_app.update()

    # ── Find articulation ──────────────────────────────────────
    dc = _dynamic_control.acquire_dynamic_control_interface()
    art = None
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = dc.get_articulation(str(prim.GetPath()))
            if art != 0:
                log(f"Articulation found: {prim.GetPath()}")
                break

    if art is None or art == 0:
        raise RuntimeError("Articulation not found")

    dof_map: dict[str, int] = {}
    for i in range(dc.get_articulation_dof_count(art)):
        dof = dc.get_articulation_dof(art, i)
        dof_map[dc.get_dof_name(dof)] = dof
    log(f"DOFs: {len(dof_map)}")

    # ── Build GUI ──────────────────────────────────────────────
    joint_data: dict[str, dict] = {}
    for name, lower, upper in joint_specs:
        if name in dof_map:
            joint_data[name] = {
                "model": ui.SimpleFloatModel(0.0),
                "lower_deg": math.degrees(lower),
                "upper_deg": math.degrees(upper),
            }

    def reset() -> None:
        for info in joint_data.values():
            info["model"].set_value(0.0)
        apply_display_mode("visual")

    def make_step_cb(model, delta, lo, hi):
        def cb():
            v = model.get_value_as_float() + delta
            model.set_value(max(lo, min(hi, v)))
        return cb

    window = ui.Window("Lite Robot Viewer", width=520, height=860)
    with window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8):
                ui.Label("Display Mode", height=22)
                with ui.HStack(height=32, spacing=6):
                    ui.Button("Visual", clicked_fn=lambda: apply_display_mode("visual"))
                    ui.Button("Collision", clicked_fn=lambda: apply_display_mode("collision"))
                    ui.Button("Both", clicked_fn=lambda: apply_display_mode("both"))
                display_mode_label = ui.Label("Display: visual", height=20)

                ui.Spacer(height=4)
                ui.Label("Left Arm (deg)", height=22)
                left_labels: dict[str, object] = {}
                for name, info in joint_data.items():
                    if not name.startswith("left_"):
                        continue
                    m = info["model"]
                    lo = info["lower_deg"]
                    hi = info["upper_deg"]
                    with ui.HStack(height=28, spacing=6):
                        ui.Label(name.replace("left_", "").replace("_", " "), width=120)
                        ui.Button("-", width=24, clicked_fn=make_step_cb(m, -1.0, lo, hi))
                        ui.FloatSlider(model=m, min=lo, max=hi)
                        ui.Button("+", width=24, clicked_fn=make_step_cb(m, 1.0, lo, hi))
                        vl = ui.Label("0.0", width=70)
                        left_labels[name] = vl

                ui.Spacer(height=4)
                ui.Label("Right Arm (deg)", height=22)
                right_labels: dict[str, object] = {}
                for name, info in joint_data.items():
                    if not name.startswith("right_"):
                        continue
                    m = info["model"]
                    lo = info["lower_deg"]
                    hi = info["upper_deg"]
                    with ui.HStack(height=28, spacing=6):
                        ui.Label(name.replace("right_", "").replace("_", " "), width=120)
                        ui.Button("-", width=24, clicked_fn=make_step_cb(m, -1.0, lo, hi))
                        ui.FloatSlider(model=m, min=lo, max=hi)
                        ui.Button("+", width=24, clicked_fn=make_step_cb(m, 1.0, lo, hi))
                        vl = ui.Label("0.0", width=70)
                        right_labels[name] = vl

                ui.Spacer(height=6)
                with ui.HStack(height=32, spacing=8):
                    ui.Button("Reset", clicked_fn=reset)

    log("UI ready.")

    # ── Main loop ──────────────────────────────────────────────
    try:
        while sim_app.is_running():
            sim_app.update()
            dc.wake_up_articulation(art)

            for name, info in joint_data.items():
                deg = info["model"].get_value_as_float()
                dc.set_dof_position_target(dof_map[name], math.radians(deg))
                label = left_labels.get(name) or right_labels.get(name)
                if label is not None:
                    label.text = f"{deg:.1f}"
    except KeyboardInterrupt:
        log("Interrupted")

    omni.timeline.get_timeline_interface().stop()
    sim_app.close()
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
