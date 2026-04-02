from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.utils.runtime_env import configure_omniverse_client_environment
from affordance_guided_interaction.utils.usd_assets import to_usd_asset_path
from teleop_cup_grasp.grasp_demo_io import due_events, load_grasp_demo_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a recorded cup grasp demo in Isaac Sim.")
    parser.add_argument(
        "--demo",
        type=Path,
        default=PROJECT_ROOT / "src/teleop_cup_grasp/grasp_demo.npz",
        help="Recorded grasp demo .npz file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without a viewport.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=120,
        help="Extra simulation steps to hold the final joint target after replay.",
    )
    return parser.parse_args()


ARGS = parse_args()
configure_omniverse_client_environment(os.environ)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": ARGS.headless, "width": 1280, "height": 720}
)

import omni
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from omni.isaac.dynamic_control import _dynamic_control


DOOR_SCENE = PROJECT_ROOT / "assets/minimal_push_door/minimal_push_door.usda"
ROBOT_USD = PROJECT_ROOT / "assets/robot/uni_dingo_dual_arm.usd"
ARM_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
]
GRIPPER_JOINT = "left_jointGripper"


def add_ground_if_needed(stage: Usd.Stage) -> None:
    if stage.GetPrimAtPath("/World/TeleopGround").IsValid():
        return
    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/TeleopGround"))
    ground.GetSizeAttr().Set(1.0)
    ground.GetDisplayColorAttr().Set([(0.52, 0.54, 0.56)])
    xf = UsdGeom.Xformable(ground.GetPrim())
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0, 0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(6, 6, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())


def spawn_support(
    stage: Usd.Stage,
    center_world: np.ndarray,
    scale_xyz: np.ndarray,
) -> Usd.Prim:
    support_prim = UsdGeom.Cube.Define(stage, Sdf.Path("/World/PickupSupport")).GetPrim()
    geom = UsdGeom.Cube(support_prim)
    geom.GetSizeAttr().Set(1.0)
    geom.GetDisplayColorAttr().Set([(0.45, 0.37, 0.25)])
    xf = UsdGeom.Xformable(support_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*center_world.astype(np.float32)))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*scale_xyz.astype(np.float32)))
    UsdPhysics.CollisionAPI.Apply(support_prim)
    return support_prim


def spawn_cup(stage: Usd.Stage, cup_world_pos: np.ndarray, cup_world_quat_wxyz: np.ndarray) -> None:
    cup_prim = stage.DefinePrim("/World/CarryObject", "Xform")
    cup_prim.GetReferences().AddReference(
        to_usd_asset_path(PROJECT_ROOT / "assets/grasp_objects/cup/carry_cup.usda")
    )
    xf = UsdGeom.Xformable(cup_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*cup_world_pos.astype(np.float32))
    )
    w, x, y, z = cup_world_quat_wxyz.astype(np.float32)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def remove_support(support_prim: Usd.Prim) -> None:
    support_prim.GetAttribute("visibility").Set("invisible")
    collision_api = UsdPhysics.CollisionAPI(support_prim)
    if collision_api:
        collision_api.GetCollisionEnabledAttr().Set(False)


def main() -> int:
    demo = load_grasp_demo_npz(ARGS.demo)

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
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3f(0, 0, 5))

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
    robot_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0.12))

    add_ground_if_needed(stage)
    support_prim = spawn_support(stage, demo.support_center_world, demo.support_scale)
    spawn_cup(stage, demo.cup_world_pos, demo.cup_world_quat_wxyz)
    for _ in range(20):
        simulation_app.update()

    omni.timeline.get_timeline_interface().play()
    for _ in range(120):
        simulation_app.update()

    dc = _dynamic_control.acquire_dynamic_control_interface()
    art = dc.get_articulation("/World/Robot/base_link")
    if art == _dynamic_control.INVALID_HANDLE:
        print("Cannot get articulation")
        return 1

    def set_pos(joint_name: str, rad: float) -> None:
        dof = dc.find_articulation_dof(art, joint_name)
        if dof != _dynamic_control.INVALID_HANDLE:
            dc.set_dof_position_target(dof, float(rad))

    def get_pos(joint_name: str) -> float:
        dof = dc.find_articulation_dof(art, joint_name)
        if dof == _dynamic_control.INVALID_HANDLE:
            return 0.0
        return float(dc.get_dof_position(dof))

    for _ in range(120):
        for joint_name, joint_value in zip(ARM_JOINTS, demo.robot_initial_q[:6]):
            set_pos(joint_name, joint_value)
        set_pos(GRIPPER_JOINT, float(demo.robot_initial_q[6]))
        simulation_app.update()

    support_removed = False
    prev_t = float(demo.t[0] - (demo.t[1] - demo.t[0] if len(demo.t) > 1 else 1.0 / 60.0))
    for step_idx, replay_t in enumerate(demo.t):
        target_arm = demo.q_arm[step_idx]
        target_gripper = float(demo.q_gripper[step_idx])

        for joint_name, joint_value in zip(ARM_JOINTS, target_arm):
            set_pos(joint_name, joint_value)
        set_pos(GRIPPER_JOINT, target_gripper)

        for event_name in due_events(
            prev_t=prev_t,
            cur_time=float(replay_t),
            event_times={"remove_support": demo.remove_support_time},
        ):
            if event_name == "remove_support" and not support_removed:
                print(f"[Replay] remove_support at t={replay_t:.3f}s")
                remove_support(support_prim)
                support_removed = True

        simulation_app.update()

        if step_idx % 120 == 0:
            actual_q = np.array([get_pos(j) for j in ARM_JOINTS], dtype=np.float64)
            tracking_error = float(np.linalg.norm(actual_q - target_arm))
            print(
                f"[Replay] step={step_idx:04d} "
                f"t={replay_t:.3f}s tracking_error={tracking_error:.4f} rad"
            )

        prev_t = float(replay_t)

    if not support_removed and demo.remove_support_time <= float(demo.t[-1]):
        remove_support(support_prim)

    final_arm = demo.q_arm[-1]
    final_gripper = float(demo.q_gripper[-1])
    for _ in range(max(0, ARGS.hold_steps)):
        for joint_name, joint_value in zip(ARM_JOINTS, final_arm):
            set_pos(joint_name, joint_value)
        set_pos(GRIPPER_JOINT, final_gripper)
        simulation_app.update()

    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass
    simulation_app.close()
    print(f"Replay complete from {ARGS.demo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
