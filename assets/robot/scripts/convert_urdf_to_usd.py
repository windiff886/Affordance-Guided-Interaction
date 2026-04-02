"""
URDF → USD 一次性转换脚本

将 Uni-Dingo 机器人 URDF 预转换为 USD 文件，并在其中：
    1. 启用 Articulation 自碰撞检测
    2. 配置关节驱动 (手臂位置控制 / 轮子速度控制)

转换后的 USD 可被 load_scene.py 直接以 reference 方式加载，
无需每次启动都重新解析 URDF，启动更快、配置不丢失。

用法:
    conda activate isaaclab
    python scripts/convert_urdf_to_usd.py
    python scripts/convert_urdf_to_usd.py --output assets/robot/my_robot.usd
"""

import os
import sys
import argparse

# =========================================================
#  解析参数 & 启动 SimulationApp
# =========================================================

parser = argparse.ArgumentParser(description="URDF → USD 转换")
parser.add_argument("--headless", action="store_true", default=True,
                    help="无窗口模式 (默认开启)")
parser.add_argument("--output", type=str, default=None,
                    help="输出 USD 文件路径")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

CONFIG = {"headless": args.headless, "width": 1280, "height": 720}
simulation_app = SimulationApp(launch_config=CONFIG)
print("✅ Isaac Sim 已启动")

# =========================================================
#  导入依赖
# =========================================================

import omni
import omni.kit.commands
from pxr import Usd, UsdGeom, Sdf, UsdPhysics, PhysxSchema

PROJECT_ROOT = os.path.expanduser("~/Code/Affordance-Guided-Interaction")
URDF_PATH = os.path.join(PROJECT_ROOT,
                         "assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf")
OUTPUT_PATH = args.output or os.path.join(PROJECT_ROOT,
                                          "assets/robot/uni_dingo_dual_arm.usd")

if not os.path.exists(URDF_PATH):
    print(f"❌ URDF 不存在: {URDF_PATH}")
    simulation_app.close()
    sys.exit(1)

# =========================================================
#  Step 1: 创建新 stage 并导入 URDF
# =========================================================

omni.usd.get_context().new_stage()
simulation_app.update()

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

print(f"📂 导入 URDF: {URDF_PATH}")

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.collision_from_visuals = False  # 修改为 False: 使用 URDF 原生的 collision 描述
import_config.self_collision = True
import_config.distance_scale = 1.0
import_config.create_physics_scene = False

status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
)
simulation_app.update()

if not robot_prim_path:
    print("❌ URDF 导入失败")
    simulation_app.close()
    sys.exit(1)

print(f"✅ URDF 已导入: {robot_prim_path}")

# =========================================================
#  Step 2: 启用 Articulation 自碰撞 (双重保险)
# =========================================================

robot_prim = stage.GetPrimAtPath(robot_prim_path)
if robot_prim.IsValid():
    # 确保 PhysxArticulationAPI 已 Apply
    if not robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    # 使用底层 USD 属性设置自碰撞 (兼容不同 Isaac Sim 版本)
    attr = robot_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
    if not attr.IsValid():
        attr = robot_prim.CreateAttribute("physxArticulation:enabledSelfCollisions",
                                          Sdf.ValueTypeNames.Bool, True)
    attr.Set(True)
    print("✅ 已启用 Articulation 自碰撞检测")
else:
    print("⚠️  未找到 robot prim, 无法设置自碰撞")

# =========================================================
#  Step 3: 配置关节驱动
# =========================================================

ARM_JOINTS = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6",
    "left_jointGripper",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6",
    "right_jointGripper",
    "pan_tilt_yaw_joint", "pan_tilt_pitch_joint",
]
WHEEL_JOINTS = [
    "front_left_wheel", "front_right_wheel",
    "rear_left_wheel", "rear_right_wheel",
]

drive_count = 0
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
        continue
    name = prim.GetName()
    prim_path = str(prim.GetPath())
    if robot_prim_path not in prim_path:
        continue

    is_wheel = any(w in name for w in WHEEL_JOINTS)
    is_arm = any(j in name for j in ARM_JOINTS)

    if is_wheel:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("velocity")
        drive.CreateDampingAttr(1500.0)
        drive.CreateStiffnessAttr(0.0)
        drive_count += 1
    elif is_arm:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(1000.0)
        drive.CreateDampingAttr(100.0)
        drive_count += 1

print(f"✅ 已配置 {drive_count} 个关节驱动")

# =========================================================
#  Step 4: 设置 defaultPrim 并保存
# =========================================================

stage.SetDefaultPrim(robot_prim)
simulation_app.update()

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

stage.GetRootLayer().Export(OUTPUT_PATH)
print(f"\n✅ USD 已保存: {OUTPUT_PATH}")
print(f"   文件大小: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

# =========================================================
#  清理
# =========================================================

simulation_app.close()
print("👋 转换完成!")
print(f"\n💡 提示: load_scene.py 将自动使用此 USD 文件加载机器人。")
print(f"   如果修改了 URDF, 请重新运行此脚本更新 USD。")
