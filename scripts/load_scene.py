"""
推门场景 + Uni-Dingo 机器人  —  Isaac Sim Standalone 脚本

功能:
    1. 创建新 stage，以 sublayer 方式加载 minimal_push_door.usda
    2. 以 reference 方式加载预转换的机器人 USD (含自碰撞 + 关节驱动)
    3. 提供交互式 UI 控制面板（滑块控制关节角度、底盘速度）

前置条件:
    先运行一次 convert_urdf_to_usd.py 生成机器人 USD 文件:
        python assets/robot/scripts/convert_urdf_to_usd.py

用法:
    conda activate isaaclab
    python scripts/load_scene.py
    python scripts/load_scene.py --show-collisions   # 碰撞体可视化
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Iterable
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.assets import (
    build_joint_space_targets,
    build_grasp_object_config,
    build_gripper_closure_targets,
    build_pickup_failure_payload,
    compute_cup_root_xyz,
    compute_support_center_xyz,
    pickup_lift_succeeds,
    pickup_proximity_succeeds,
    pickup_settle_speed_succeeds,
    pickup_tilt_succeeds,
)
from affordance_guided_interaction.utils.runtime_env import (
    configure_omniverse_client_environment,
)
from affordance_guided_interaction.utils.runtime_timing import (
    run_observation_pause,
)
from affordance_guided_interaction.utils.usd_assets import to_usd_asset_path
from affordance_guided_interaction.utils.usd_math import (
    quat_to_float_components,
    vec3_to_float_components,
)

# =========================================================
#  Step 0: 解析参数 & 启动 SimulationApp
# =========================================================

parser = argparse.ArgumentParser(description="推门场景 + Uni-Dingo 机器人")
parser.add_argument("--headless", action="store_true", help="无窗口模式")
parser.add_argument("--show-collisions", action="store_true", help="显示碰撞体调试可视化")
parser.add_argument("--show-camera", action="store_true", help="启用机器人第一人称视角可视化")
parser.add_argument(
    "--grasp-object",
    choices=["none", "cup", "tray"],
    default="none",
    help="启动时加载的持物资产",
)
parser.add_argument("--grasp-variant", default=None, help="指定持物资产变体名称")
parser.add_argument(
    "--grasp-arm",
    choices=["left", "right"],
    default="left",
    help="将持物资产挂到哪一侧夹爪",
)
parser.add_argument(
    "--attach-grasp-object",
    action="store_true",
    help="已废弃；当前版本会执行物理抓取初始化，不再创建 fixed joint",
)
parser.add_argument(
    "--gripper-closed-position",
    type=float,
    default=None,
    help="启动时夹爪闭合关节目标值；默认读取资产目录配置",
)
args, _ = parser.parse_known_args()

configure_omniverse_client_environment(os.environ)

from isaacsim import SimulationApp

CONFIG = {
    "headless": args.headless,
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(launch_config=CONFIG)
print("✅ Isaac Sim 已启动")

# =========================================================
#  Step 1: 创建 stage 并加载推门场景
# =========================================================

import omni
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools

DOOR_SCENE = PROJECT_ROOT / "assets/minimal_push_door/minimal_push_door.usda"
ROBOT_USD = PROJECT_ROOT / "assets/robot/uni_dingo_dual_arm.usd"


def find_first_named_prim(stage: Usd.Stage, prim_names: list[str]) -> Usd.Prim:
    for prim_name in prim_names:
        for prim in stage.Traverse():
            if prim.GetName() == prim_name:
                return prim
    return Usd.Prim()


def align_xform_prim_to_world_pose(prim: Usd.Prim, world_xform: Gf.Matrix4d) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)
    orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(
        Gf.Vec3f(*vec3_to_float_components(world_xform.ExtractTranslation()))
    )
    real, ix, iy, iz = quat_to_float_components(world_xform.ExtractRotation().GetQuat())
    orient_op.Set(Gf.Quatf(real, ix, iy, iz))


def set_local_xform(
    prim: Usd.Prim,
    translation: Gf.Vec3d,
    rotation_deg: Gf.Vec3d | None = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*vec3_to_float_components(translation))
    )
    if rotation_deg is not None:
        xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(*vec3_to_float_components(rotation_deg))
        )


def spawn_carry_object(
    stage: Usd.Stage,
    object_config: dict,
) -> Usd.Prim:
    carry_prim_path = "/World/CarryObject"
    carry_prim = stage.DefinePrim(carry_prim_path, "Xform")
    carry_prim.GetReferences().AddReference(
        to_usd_asset_path(object_config["usd_path"])
    )

    for _ in range(object_config.get("spawn_pre_align_updates", 0)):
        simulation_app.update()

    return carry_prim


def ensure_grasp_frame(
    stage: Usd.Stage,
    parent_prim: Usd.Prim,
    frame_name: str,
    translation_xyz: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
) -> Usd.Prim:
    frame_path = parent_prim.GetPath().AppendChild(frame_name)
    frame_prim = stage.DefinePrim(frame_path, "Xform")
    set_local_xform(
        frame_prim,
        Gf.Vec3d(*translation_xyz),
        Gf.Vec3d(*rotation_xyz_deg),
    )
    simulation_app.update()
    return frame_prim


def resolve_referenced_child_prim(
    stage: Usd.Stage,
    referenced_prim: Usd.Prim,
    asset_root_prim_path: str,
    asset_target_prim_path: str,
) -> Usd.Prim:
    if asset_target_prim_path == asset_root_prim_path:
        return referenced_prim
    if not asset_target_prim_path.startswith(f"{asset_root_prim_path}/"):
        raise RuntimeError(
            f"asset target path is not under root prim: {asset_target_prim_path}"
        )

    suffix = asset_target_prim_path[len(asset_root_prim_path) :]
    resolved_path = Sdf.Path(f"{referenced_prim.GetPath()}{suffix}")
    resolved_prim = stage.GetPrimAtPath(resolved_path)
    if not resolved_prim.IsValid():
        raise RuntimeError(f"missing referenced child prim: {resolved_path}")
    return resolved_prim

# 验证文件存在
selected_grasp_object = None
if args.grasp_object != "none":
    selected_grasp_object = build_grasp_object_config(
        args.grasp_object,
        variant=args.grasp_variant,
        arm=args.grasp_arm,
    )

required_files = [(DOOR_SCENE, "门场景"), (ROBOT_USD, "机器人 USD")]
if selected_grasp_object is not None:
    required_files.append((selected_grasp_object["usd_path"], "持物资产"))

for f, desc in required_files:
    if not f.exists():
        hint = ""
        if f == ROBOT_USD:
            hint = "\n   💡 请先运行: python assets/robot/scripts/convert_urdf_to_usd.py"
        print(f"❌ {desc} 不存在: {f}{hint}")
        simulation_app.close()
        sys.exit(1)

# 创建新 stage 并以 sublayer 方式加载门场景
# 使用 sublayer（而非 reference）确保关节的绝对路径引用被正确解析
print("📂 创建 stage 并加载推门场景...")
omni.usd.get_context().new_stage()
simulation_app.update()

stage = omni.usd.get_context().get_stage()
if not stage:
    print("❌ 无法创建 stage")
    simulation_app.close()
    sys.exit(1)

# 以 sublayer 方式引入门场景
stage.GetRootLayer().subLayerPaths.append(str(DOOR_SCENE))

# 设置场景元数据
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

for _ in range(20):
    simulation_app.update()

print("✅ 推门场景已加载 (sublayer 方式)")

# =========================================================
#  Step 2: 读取 SuggestedRobotBase 位置
# =========================================================

robot_base_pos = Gf.Vec3d(4.6, 0, 0)
robot_base_rot = Gf.Vec3d(0, 0, 180)

suggested_prim = stage.GetPrimAtPath("/World/SuggestedRobotBase")
if suggested_prim.IsValid():
    xformable = UsdGeom.Xformable(suggested_prim)
    local_xform = xformable.GetLocalTransformation()
    robot_base_pos = local_xform.ExtractTranslation()
    print(f"📍 SuggestedRobotBase 位置: {robot_base_pos}")
else:
    print(f"⚠️  未找到 SuggestedRobotBase, 使用默认位置: {robot_base_pos}")

# =========================================================
#  Step 3: 加载预转换的机器人 USD (reference 方式)
# =========================================================
#  机器人 USD 由 convert_urdf_to_usd.py 预生成，已包含:
#    - Articulation 自碰撞检测
#    - 关节驱动配置 (手臂位置控制 / 轮子速度控制)

print(f"\n📂 加载机器人 USD: {ROBOT_USD}")

robot_prim_path = "/World/Robot"
robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
robot_prim.GetReferences().AddReference(to_usd_asset_path(ROBOT_USD))

for _ in range(10):
    simulation_app.update()

print(f"✅ 机器人已加载: {robot_prim_path} (reference 方式, 含自碰撞 + 关节驱动)")

# =========================================================
#  Step 4: 设置机器人位置
# =========================================================

xformable = UsdGeom.Xformable(robot_prim)
xformable.ClearXformOpOrder()
translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
translate_op.Set(Gf.Vec3d(float(robot_base_pos[0]), float(robot_base_pos[1]), 0.12))
rotate_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
rotate_op.Set(Gf.Vec3d(0.0, 0.0, 180.0))
simulation_app.update()
print(f"📍 机器人已放置到 ({robot_base_pos[0]:.1f}, {robot_base_pos[1]:.1f}, 0.12)")

# =========================================================
#  Step 4b: 添加室外地面
# =========================================================

outdoor_floor_path = "/World/Room/OutdoorFloor"
if not stage.GetPrimAtPath(outdoor_floor_path).IsValid():
    outdoor_floor = UsdGeom.Cube.Define(stage, Sdf.Path(outdoor_floor_path))
    outdoor_floor.GetSizeAttr().Set(1.0)
    outdoor_floor.GetDisplayColorAttr().Set([(0.52, 0.54, 0.56)])
    xf = UsdGeom.Xformable(outdoor_floor.GetPrim())
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(5.0, 0.0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(6.0, 8.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(outdoor_floor.GetPrim())
    simulation_app.update()
    print("✅ 已添加室外地面")

# =========================================================
#  Step 5: 碰撞体可视化 (可选)
# =========================================================

if args.show_collisions:
    import carb
    settings = carb.settings.get_settings()
    settings.set_bool("/persistent/physics/visualizationDisplayColliders", True)
    settings.set_bool("/physics/debugDraw", True)
    settings.set_int("/persistent/physics/visualizationDisplayCollidersMeshMode", 2)
    print("🔍 碰撞体调试可视化已开启")

# =========================================================
#  Step 6: 交互式控制 UI
# =========================================================

from omni.isaac.dynamic_control import _dynamic_control
import omni.ui as ui

dc = _dynamic_control.acquire_dynamic_control_interface()
GRASP_OBSERVATION_SECONDS = 10.0
active_grasp_observation_seconds = 0.0 if args.headless else GRASP_OBSERVATION_SECONDS

# 启动仿真
omni.timeline.get_timeline_interface().play()
print("\n🎮 仿真已启动, 等待初始化...")

for _ in range(120):
    simulation_app.update()

# 获取 Articulation
art = dc.get_articulation("/World/Robot/base_link")
if art == 0:
    print("❌ 无法获取 Articulation!")
    simulation_app.close()
    sys.exit(1)

print("✅ Articulation 已就绪")

num_dofs = dc.get_articulation_dof_count(art)
print(f"\n📋 机器人关节数量: {num_dofs}")
for i in range(num_dofs):
    name = dc.get_dof_name(dc.get_articulation_dof(art, i))
    print(f"   [{i:>2}] {name}")

# 辅助函数
def set_pos(joint_name, rad):
    dof = dc.find_articulation_dof(art, joint_name)
    if dof != 0:
        dc.set_dof_position_target(dof, rad)

def set_vel(joint_name, vel):
    dof = dc.find_articulation_dof(art, joint_name)
    if dof != 0:
        dc.set_dof_velocity_target(dof, vel)


def set_joint_pose_deg(joint_targets_deg: dict[str, float]) -> None:
    for joint_name, degrees in joint_targets_deg.items():
        set_pos(joint_name, np.radians(degrees))


def transform_point(
    world_xform: Gf.Matrix4d,
    local_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    point = world_xform.Transform(Gf.Vec3d(*local_xyz))
    return (float(point[0]), float(point[1]), float(point[2]))


def transform_point_inverse(
    world_xform: Gf.Matrix4d,
    world_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    point = world_xform.GetInverse().Transform(Gf.Vec3d(*world_xyz))
    return (float(point[0]), float(point[1]), float(point[2]))


def transform_direction_to_base(
    base_world_xform: Gf.Matrix4d,
    prim_world_xform: Gf.Matrix4d,
    local_direction_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    origin_world = prim_world_xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    direction_world = prim_world_xform.Transform(Gf.Vec3d(*local_direction_xyz))
    origin_base = base_world_xform.GetInverse().Transform(origin_world)
    direction_base = base_world_xform.GetInverse().Transform(direction_world)
    return (
        float(direction_base[0] - origin_base[0]),
        float(direction_base[1] - origin_base[1]),
        float(direction_base[2] - origin_base[2]),
    )


def spawn_pickup_support(
    stage: Usd.Stage,
    world_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
) -> Usd.Prim:
    support_prim = UsdGeom.Cube.Define(stage, Sdf.Path("/World/PickupSupport")).GetPrim()
    support_geom = UsdGeom.Cube(support_prim)
    support_geom.GetSizeAttr().Set(1.0)
    support_geom.GetDisplayColorAttr().Set([(0.45, 0.37, 0.25)])
    xformable = UsdGeom.Xformable(support_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*world_xyz)
    )
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*size_xyz))
    UsdPhysics.CollisionAPI.Apply(support_prim)
    return support_prim


def read_current_joint_targets_deg(joint_names: Iterable[str]) -> dict[str, float]:
    current_targets_deg: dict[str, float] = {}
    for joint_name in joint_names:
        dof = dc.find_articulation_dof(art, joint_name)
        if dof == 0:
            raise RuntimeError(f"missing dof: {joint_name}")
        current_targets_deg[joint_name] = float(
            np.degrees(dc.get_dof_position_target(dof))
        )
    return current_targets_deg


def run_joint_stage(
    stage_name: str,
    joint_targets_deg: dict[str, float],
    steps: int,
) -> None:
    print(f"➡️ {stage_name}")
    current_targets_deg = read_current_joint_targets_deg(joint_targets_deg.keys())
    if all(
        np.isclose(current_targets_deg[joint_name], target_deg, atol=1e-4)
        for joint_name, target_deg in joint_targets_deg.items()
    ):
        return
    if steps <= 1:
        set_joint_pose_deg(joint_targets_deg)
        simulation_app.update()
        return
    for target in build_joint_space_targets(
        current_targets_deg,
        joint_targets_deg,
        steps,
    ):
        set_joint_pose_deg(target)
        simulation_app.update()


def run_gripper_stage(
    stage_name: str,
    joint_name: str,
    open_deg: float,
    close_deg: float,
    steps: int,
) -> None:
    print(f"➡️ {stage_name}")
    for target_rad in build_gripper_closure_targets(open_deg, close_deg, steps):
        set_pos(joint_name, target_rad)
        simulation_app.update()


def run_settle_stage(stage_name: str, steps: int) -> None:
    print(f"➡️ {stage_name}")
    for _ in range(steps):
        simulation_app.update()


def read_world_translation(prim: Usd.Prim) -> Gf.Vec3d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()


def find_first_rigid_body_prim(root_prim: Usd.Prim) -> Usd.Prim:
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return Usd.Prim()


def read_support_top_z(
    support_prim: Usd.Prim,
    support_size_xyz: tuple[float, float, float],
) -> float:
    center = read_world_translation(support_prim)
    return float(center[2] + 0.5 * support_size_xyz[2])


def read_cup_root_height(carry_prim: Usd.Prim) -> float:
    measurement_prim = find_first_rigid_body_prim(carry_prim)
    if not measurement_prim or not measurement_prim.IsValid():
        measurement_prim = carry_prim
    return float(read_world_translation(measurement_prim)[2])


def read_cup_root_tilt_deg(carry_prim: Usd.Prim) -> float:
    measurement_prim = find_first_rigid_body_prim(carry_prim)
    if not measurement_prim or not measurement_prim.IsValid():
        measurement_prim = carry_prim

    world_xform = UsdGeom.Xformable(measurement_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    origin = world_xform.ExtractTranslation()
    up_point = world_xform.Transform(Gf.Vec3d(0.0, 0.0, 1.0))
    local_up = np.array(
        [up_point[0] - origin[0], up_point[1] - origin[1], up_point[2] - origin[2]],
        dtype=np.float64,
    )
    local_up /= np.linalg.norm(local_up)
    dot = np.clip(np.dot(local_up, np.array([0.0, 0.0, 1.0], dtype=np.float64)), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def read_cup_linear_speed(carry_prim: Usd.Prim) -> float:
    measurement_prim = find_first_rigid_body_prim(carry_prim)
    if not measurement_prim or not measurement_prim.IsValid():
        raise RuntimeError("missing carry rigid body prim")
    rigid_body = dc.get_rigid_body(str(measurement_prim.GetPath()))
    if rigid_body == _dynamic_control.INVALID_HANDLE:
        raise RuntimeError(f"missing rigid body handle: {measurement_prim.GetPath()}")
    velocity = dc.get_rigid_body_linear_velocity(rigid_body)
    return float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))


def read_distance_between_prims(
    measurement_prim: Usd.Prim,
    reference_prim: Usd.Prim,
) -> float:
    carry_translation = read_world_translation(measurement_prim)
    frame_translation = read_world_translation(reference_prim)
    return float(
        np.linalg.norm(
            np.array(
                [
                    carry_translation[0] - frame_translation[0],
                    carry_translation[1] - frame_translation[1],
                    carry_translation[2] - frame_translation[2],
                ],
                dtype=np.float64,
            )
        )
    )


def log_prim_pose_in_base(
    label: str,
    prim: Usd.Prim,
    base_world: Gf.Matrix4d,
) -> None:
    world_xyz = tuple(float(component) for component in read_world_translation(prim))
    print(f"   {label}(world): {world_xyz}")
    print(f"   {label}(base): {transform_point_inverse(base_world, world_xyz)}")


def log_prim_pose_against_current_base(
    label: str,
    prim: Usd.Prim,
    base_link_prim: Usd.Prim,
) -> None:
    current_base_world = UsdGeom.Xformable(base_link_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_xyz = tuple(float(component) for component in read_world_translation(prim))
    print(
        f"   {label}(current_base): "
        f"{transform_point_inverse(current_base_world, world_xyz)}"
    )


def fail_pickup(
    stage_name: str,
    object_config: dict,
    runtime_context: dict[str, Usd.Prim],
) -> None:
    pickup = object_config["pickup"]
    payload = build_pickup_failure_payload(
        stage=stage_name,
        support_top_z=read_support_top_z(
            runtime_context["support_prim"],
            pickup["support_size_xyz_m"],
        ),
        cup_height=read_cup_root_height(runtime_context["carry_prim"]),
        cup_tilt_deg=read_cup_root_tilt_deg(runtime_context["carry_prim"]),
        settle_speed_mps=read_cup_linear_speed(runtime_context["carry_prim"]),
        distance_to_grasp_frame=read_distance_between_prims(
            runtime_context["carry_grasp_frame_prim"],
            runtime_context["grasp_frame_prim"],
        ),
        gripper_joint_name=object_config["gripper_joint"],
    )
    raise RuntimeError(f"抓杯失败 [{stage_name}]: {payload}")


def initialize_base_relative_pickup(
    stage: Usd.Stage,
    object_config: dict,
) -> dict[str, Usd.Prim]:
    print(
        "🤲 执行基座相对抓杯初始化: "
        f"{object_config['family']} / {object_config['variant']}"
    )
    pickup = object_config["pickup"]
    gripper_joint = object_config["gripper_joint"]

    base_link_prim = stage.GetPrimAtPath("/World/Robot/base_link")
    if not base_link_prim.IsValid():
        raise RuntimeError("missing /World/Robot/base_link")

    base_world = UsdGeom.Xformable(base_link_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    grasp_parent = find_first_named_prim(stage, [pickup["grasp_frame_parent_link"]])
    if not grasp_parent.IsValid():
        raise RuntimeError(
            f"missing left_cup_grasp_frame parent: {pickup['grasp_frame_parent_link']}"
        )

    grasp_frame_prim = ensure_grasp_frame(
        stage,
        grasp_parent,
        f"{pickup['arm']}_cup_grasp_frame",
        pickup["grasp_frame_translation_xyz_m"],
        pickup["grasp_frame_rotate_xyz_deg"],
    )
    if not grasp_frame_prim.IsValid():
        raise RuntimeError("failed to create left_cup_grasp_frame")
    print(f"📐 已创建抓取参考系: {grasp_frame_prim.GetPath()}")
    grasp_frame_world_xform = UsdGeom.Xformable(
        grasp_frame_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    log_prim_pose_in_base("grasp_frame", grasp_frame_prim, base_world)
    log_prim_pose_against_current_base("grasp_frame", grasp_frame_prim, base_link_prim)
    print(
        "   grasp_axes(base): "
        f"x={transform_direction_to_base(base_world, grasp_frame_world_xform, (1.0, 0.0, 0.0))}, "
        f"y={transform_direction_to_base(base_world, grasp_frame_world_xform, (0.0, 1.0, 0.0))}, "
        f"z={transform_direction_to_base(base_world, grasp_frame_world_xform, (0.0, 0.0, 1.0))}"
    )

    set_pos(gripper_joint, np.radians(pickup["gripper_open_deg"]))
    simulation_app.update()

    run_joint_stage("pregrasp", pickup["pregrasp_joint_deg"], pickup["pregrasp_steps"])
    log_prim_pose_in_base("grasp_frame_after_pregrasp", grasp_frame_prim, base_world)
    log_prim_pose_against_current_base(
        "grasp_frame_after_pregrasp",
        grasp_frame_prim,
        base_link_prim,
    )

    pickup_base_world = UsdGeom.Xformable(base_link_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    support_center_xyz = compute_support_center_xyz(
        pickup["pickup_point_xyz_m"],
        pickup["support_size_xyz_m"],
    )
    support_world_xyz = transform_point(pickup_base_world, support_center_xyz)
    cup_world_xyz = transform_point(
        pickup_base_world,
        compute_cup_root_xyz(pickup["pickup_point_xyz_m"]),
    )

    support_prim = spawn_pickup_support(
        stage,
        support_world_xyz,
        pickup["support_size_xyz_m"],
    )
    if not support_prim.IsValid():
        raise RuntimeError("failed to spawn pickup support")
    print(f"🪵 已创建取杯支撑台: {support_prim.GetPath()}")

    carry_prim = spawn_carry_object(stage, object_config)
    if not carry_prim.IsValid():
        raise RuntimeError("failed to spawn carry cup")
    set_local_xform(carry_prim, Gf.Vec3d(*cup_world_xyz))
    carry_grasp_frame_prim = resolve_referenced_child_prim(
        stage,
        carry_prim,
        object_config["root_prim"],
        object_config["grasp_frame_path"],
    )
    print(f"🥤 已创建水杯: {carry_prim.GetPath()}")
    print(f"   cup_spawn(world): {cup_world_xyz}")
    print(f"   cup_spawn(base): {pickup['pickup_point_xyz_m']}")
    log_prim_pose_in_base("cup_grasp_frame", carry_grasp_frame_prim, pickup_base_world)
    log_prim_pose_against_current_base(
        "cup_grasp_frame",
        carry_grasp_frame_prim,
        base_link_prim,
    )
    print("🚀 pickup sequence started")

    runtime_context = {
        "support_prim": support_prim,
        "carry_prim": carry_prim,
        "carry_grasp_frame_prim": carry_grasp_frame_prim,
        "grasp_frame_prim": grasp_frame_prim,
    }

    run_joint_stage("approach", pickup["grasp_joint_deg"], pickup["approach_steps"])
    if pickup["approach_settle_steps"] > 0:
        run_settle_stage("approach_settle", pickup["approach_settle_steps"])
    log_prim_pose_in_base(
        "grasp_frame_after_approach",
        grasp_frame_prim,
        pickup_base_world,
    )
    log_prim_pose_against_current_base(
        "grasp_frame_after_approach",
        grasp_frame_prim,
        base_link_prim,
    )
    log_prim_pose_in_base(
        "cup_grasp_frame_after_approach",
        carry_grasp_frame_prim,
        pickup_base_world,
    )
    log_prim_pose_against_current_base(
        "cup_grasp_frame_after_approach",
        carry_grasp_frame_prim,
        base_link_prim,
    )
    print(
        "   distance_before_closure: "
        f"{read_distance_between_prims(carry_grasp_frame_prim, grasp_frame_prim):.4f} m"
    )

    close_deg = (
        np.degrees(args.gripper_closed_position)
        if args.gripper_closed_position is not None
        else pickup["gripper_close_deg"]
    )
    run_gripper_stage(
        "closure",
        gripper_joint,
        pickup["gripper_open_deg"],
        close_deg,
        pickup["closure_steps"],
    )

    run_joint_stage("capture", pickup["capture_joint_deg"], pickup["capture_steps"])
    log_prim_pose_in_base(
        "grasp_frame_after_capture",
        grasp_frame_prim,
        pickup_base_world,
    )
    log_prim_pose_against_current_base(
        "grasp_frame_after_capture",
        grasp_frame_prim,
        base_link_prim,
    )
    log_prim_pose_in_base(
        "cup_grasp_frame_after_capture",
        carry_grasp_frame_prim,
        pickup_base_world,
    )
    log_prim_pose_against_current_base(
        "cup_grasp_frame_after_capture",
        carry_grasp_frame_prim,
        base_link_prim,
    )

    run_settle_stage("settle", pickup["settle_steps"])
    if not pickup_settle_speed_succeeds(
        read_cup_linear_speed(carry_prim),
        pickup["max_settle_linear_speed_mps"],
    ) or not pickup_tilt_succeeds(
        read_cup_root_tilt_deg(carry_prim),
        pickup["max_tilt_deg"],
    ):
        fail_pickup("settle", object_config, runtime_context)

    run_joint_stage("lift", pickup["carry_standby_joint_deg"], pickup["lift_steps"])
    log_prim_pose_in_base(
        "grasp_frame_after_lift",
        grasp_frame_prim,
        pickup_base_world,
    )
    log_prim_pose_against_current_base(
        "grasp_frame_after_lift",
        grasp_frame_prim,
        base_link_prim,
    )
    log_prim_pose_in_base(
        "cup_grasp_frame_after_lift",
        carry_grasp_frame_prim,
        pickup_base_world,
    )
    log_prim_pose_against_current_base(
        "cup_grasp_frame_after_lift",
        carry_grasp_frame_prim,
        base_link_prim,
    )
    if not pickup_lift_succeeds(
        read_cup_root_height(carry_prim),
        read_support_top_z(support_prim, pickup["support_size_xyz_m"]),
        pickup["min_lift_height_m"],
    ) or not pickup_proximity_succeeds(
        read_distance_between_prims(carry_grasp_frame_prim, grasp_frame_prim),
        pickup["gripper_proximity_threshold_m"],
    ):
        fail_pickup("lift", object_config, runtime_context)

    run_joint_stage(
        "retreat",
        pickup["carry_standby_joint_deg"],
        pickup["retreat_steps"],
    )
    if not pickup_proximity_succeeds(
        read_distance_between_prims(carry_grasp_frame_prim, grasp_frame_prim),
        pickup["gripper_proximity_threshold_m"],
    ) or not pickup_tilt_succeeds(
        read_cup_root_tilt_deg(carry_prim),
        pickup["max_tilt_deg"],
    ):
        fail_pickup("retreat", object_config, runtime_context)

    return runtime_context


def initialize_scene_object(
    stage: Usd.Stage,
    object_config: dict,
) -> dict[str, Usd.Prim]:
    if object_config["grasp_mode"] == "base_relative_pickup":
        return initialize_base_relative_pickup(stage, object_config)

    carry_prim = spawn_carry_object(stage, object_config)
    set_local_xform(carry_prim, Gf.Vec3d(*object_config["spawn_position_xyz"]))
    simulation_app.update()
    return {
        "carry_prim": carry_prim,
    }


runtime_context = None
if selected_grasp_object is not None:
    if args.attach_grasp_object:
        print("ℹ️  已忽略 --attach-grasp-object；当前版本使用物理抓取初始化，不再创建 fixed joint")
    print(f"⏸  观察窗口: {active_grasp_observation_seconds:.0f}s 后开始执行抓取初始化...")
    run_observation_pause(
        update_callback=simulation_app.update,
        sleep_callback=time.sleep,
        duration_seconds=active_grasp_observation_seconds,
    )
    try:
        runtime_context = initialize_scene_object(stage, selected_grasp_object)
    except Exception as exc:
        print(f"❌ 持物资产初始化失败: {exc}")
        simulation_app.close()
        sys.exit(1)
    print(f"✅ 持物资产已初始化: {runtime_context['carry_prim'].GetPath()}")

# ---- 构建 UI ----
print("\n🖥️  创建控制面板...")

slider_state = {}

LEFT_ARM_UI = [
    ("left_joint1",  -150, 150,  0, "Shoulder Z"),
    ("left_joint2",     0, 170,  0, "Shoulder Y"),
    ("left_joint3",  -165,   0,  0, "Elbow Y"),
    ("left_joint4",  -165, 165,  0, "Forearm Y"),
    ("left_joint5",  -165, 165,  0, "Wrist Z"),
    ("left_joint6",   -90,  90,  0, "Wrist X"),
    ("left_jointGripper", -90, 0, 0, "Gripper"),
]
RIGHT_ARM_UI = [
    ("right_joint1",  -150, 150,  0, "Shoulder Z"),
    ("right_joint2",     0, 170,  0, "Shoulder Y"),
    ("right_joint3",  -165,   0,  0, "Elbow Y"),
    ("right_joint4",  -165, 165,  0, "Forearm Y"),
    ("right_joint5",  -165, 165,  0, "Wrist Z"),
    ("right_joint6",   -90,  90,  0, "Wrist X"),
    ("right_jointGripper", -90, 0, 0, "Gripper"),
]
HEAD_UI = [
    ("pan_tilt_yaw_joint",   -90, 90, 0, "Pan Yaw"),
    ("pan_tilt_pitch_joint", -90, 90, 0, "Tilt Pitch"),
]


def make_joint_slider(joint_name, min_deg, max_deg, default_deg, label):
    with ui.HStack(height=28, spacing=4):
        ui.Label(label, width=100)
        slider = ui.FloatSlider(min=min_deg, max=max_deg)
        slider.model.set_value(default_deg)
        value_label = ui.Label(f"{default_deg:.0f}°", width=50)

        def on_changed(model, jn=joint_name, vl=value_label):
            deg = model.as_float
            vl.text = f"{deg:.0f}°"
            set_pos(jn, np.radians(deg))

        slider.model.add_value_changed_fn(on_changed)
        slider_state[joint_name] = slider


chassis_speed = {"forward": 0.0, "turn": 0.0}


def update_wheels():
    fwd = chassis_speed["forward"]
    trn = chassis_speed["turn"]
    set_vel("front_left_wheel",  fwd + trn)
    set_vel("rear_left_wheel",   fwd + trn)
    set_vel("front_right_wheel", fwd - trn)
    set_vel("rear_right_wheel",  fwd - trn)


# 创建窗口
window = ui.Window("Uni-Dingo Control Panel", width=420, height=780)

with window.frame:
    with ui.ScrollingFrame():
        with ui.VStack(spacing=6):
            ui.Spacer(height=4)

            # ---- 底盘 ----
            with ui.CollapsableFrame("Chassis", height=0):
                with ui.VStack(spacing=4):
                    ui.Spacer(height=2)
                    with ui.HStack(height=28, spacing=4):
                        ui.Label("Fwd / Rev", width=100)
                        fwd_slider = ui.FloatSlider(min=-10, max=10)
                        fwd_slider.model.set_value(0.0)
                        fwd_label = ui.Label("0.0", width=50)

                        def on_fwd(model, fl=fwd_label):
                            chassis_speed["forward"] = model.as_float
                            fl.text = f"{model.as_float:.1f}"
                            update_wheels()
                        fwd_slider.model.add_value_changed_fn(on_fwd)

                    with ui.HStack(height=28, spacing=4):
                        ui.Label("Turn L/R", width=100)
                        turn_slider = ui.FloatSlider(min=-5, max=5)
                        turn_slider.model.set_value(0.0)
                        turn_label = ui.Label("0.0", width=50)

                        def on_turn(model, tl=turn_label):
                            chassis_speed["turn"] = model.as_float
                            tl.text = f"{model.as_float:.1f}"
                            update_wheels()
                        turn_slider.model.add_value_changed_fn(on_turn)

                    def on_stop():
                        fwd_slider.model.set_value(0.0)
                        turn_slider.model.set_value(0.0)
                        chassis_speed["forward"] = 0.0
                        chassis_speed["turn"] = 0.0
                        update_wheels()
                    ui.Button("STOP", height=32, clicked_fn=on_stop)

            # ---- 左臂 ----
            with ui.CollapsableFrame("Left Arm", height=0):
                with ui.VStack(spacing=2):
                    ui.Spacer(height=2)
                    for jn, mn, mx, dv, lb in LEFT_ARM_UI:
                        make_joint_slider(jn, mn, mx, dv, lb)

            # ---- 右臂 ----
            with ui.CollapsableFrame("Right Arm", height=0):
                with ui.VStack(spacing=2):
                    ui.Spacer(height=2)
                    for jn, mn, mx, dv, lb in RIGHT_ARM_UI:
                        make_joint_slider(jn, mn, mx, dv, lb)

            # ---- 云台 ----
            with ui.CollapsableFrame("Head Pan/Tilt", height=0):
                with ui.VStack(spacing=2):
                    ui.Spacer(height=2)
                    for jn, mn, mx, dv, lb in HEAD_UI:
                        make_joint_slider(jn, mn, mx, dv, lb)

            # ---- 快捷姿态 ----
            with ui.CollapsableFrame("Presets", height=0):
                with ui.VStack(spacing=4):
                    ui.Spacer(height=2)

                    def preset_zero():
                        for jn, sl in slider_state.items():
                            sl.model.set_value(0.0)
                            set_pos(jn, 0.0)

                    def preset_arms_forward():
                        poses = {
                            "left_joint1": 0, "left_joint2": 90,
                            "left_joint3": -10, "left_joint4": 0,
                            "left_joint5": 0, "left_joint6": 0,
                            "right_joint1": 0, "right_joint2": 90,
                            "right_joint3": -10, "right_joint4": 0,
                            "right_joint5": 0, "right_joint6": 0,
                        }
                        for jn, deg in poses.items():
                            if jn in slider_state:
                                slider_state[jn].model.set_value(deg)
                            set_pos(jn, np.radians(deg))

                    def preset_arms_up():
                        poses = {
                            "left_joint1": 0, "left_joint2": 170,
                            "left_joint3": 0, "left_joint4": 0,
                            "right_joint1": 0, "right_joint2": 170,
                            "right_joint3": 0, "right_joint4": 0,
                        }
                        for jn, deg in poses.items():
                            if jn in slider_state:
                                slider_state[jn].model.set_value(deg)
                            set_pos(jn, np.radians(deg))

                    ui.Button("Reset All", height=30, clicked_fn=preset_zero)
                    ui.Button("Arms Forward", height=30, clicked_fn=preset_arms_forward)
                    ui.Button("Arms Up", height=30, clicked_fn=preset_arms_up)

            ui.Spacer(height=8)

print("✅ 控制面板已创建!")
print("🎮 在 Isaac Sim 窗口中拖动滑块控制机器人")
print("   按 Ctrl+C 退出\n")

import cv2
from omni.isaac.sensor import Camera

d455_camera = None
if not args.show_camera or args.headless:
    print("ℹ️ 未启用相机视图或处于 headless 模式，跳过深度相机可视化")
else:
    # 查找 head_d455_link 并创建 Camera
    head_prim = find_first_named_prim(stage, ["head_d455_link"])
    cam_prim_path = (
        str(head_prim.GetPath()) + "/Camera"
        if (head_prim and head_prim.IsValid())
        else "/World/Robot/base_link/head_d455_link/Camera"
    )

    print(f"📷 初始化深度相机: {cam_prim_path}")
    try:
        d455_camera = Camera(
            prim_path=cam_prim_path,
            frequency=30,
            resolution=(640, 480),
        )
        d455_camera.initialize()
        d455_camera.add_distance_to_image_plane_to_frame()

        # 强制设置 D455 的广角 FOV (约 86 度水平视野)
        cam_geom = UsdGeom.Camera(stage.GetPrimAtPath(cam_prim_path))
        if cam_geom:
            cam_geom.GetFocalLengthAttr().Set(11.2)
            cam_geom.GetHorizontalApertureAttr().Set(20.955)
            # 640x480 比例下 vertical aperture 为 20.955 * (480/640)
            cam_geom.GetVerticalApertureAttr().Set(15.716)
            cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    except Exception as e:
        print(f"⚠️ 深度相机初始化失败: {e}")
        d455_camera = None

# 主循环
try:
    while simulation_app.is_running():
        simulation_app.update()

        if d455_camera:
            try:
                rgba = d455_camera.get_rgba()
                if rgba is not None and rgba.size > 0:
                    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    cv2.imshow("D455 RGB", bgr)

                frame = d455_camera.get_current_frame()
                if frame and "distance_to_image_plane" in frame:
                    depth = frame["distance_to_image_plane"]
                    if depth is not None and depth.size > 0:
                        depth_norm = np.clip(depth / 5.0, 0, 1.0)
                        depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                        cv2.imshow("D455 Depth", depth_color)

                cv2.waitKey(1)
            except Exception as e:
                print(f"⚠️ 深度相机读取失败，已禁用相机可视化: {e}")
                d455_camera = None
                cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\n⏹ 用户中断")

# 清理
try:
    omni.timeline.get_timeline_interface().stop()
except Exception:
    pass
if d455_camera:
    cv2.destroyAllWindows()
simulation_app.close()
print("👋 已退出")
