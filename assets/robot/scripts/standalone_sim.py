"""
Uni-Dingo 双臂机器人 Isaac Sim Standalone 脚本
无需 GUI 操作，命令行直接运行

用法:
    conda activate isaaclab
    python assets/robot/scripts/standalone_sim.py

功能:
    1. 导入 URDF → USD
    2. 配置关节驱动
    3. 运行仿真，控制机器人运动
    4. 保存场景为 USD 文件
"""

import os
import sys
import numpy as np

# =========================================================
#  Step 0: 启动 SimulationApp (必须在 import omni 之前)
# =========================================================

from isaacsim import SimulationApp

# headless=False 会弹出渲染窗口; headless=True 则完全无窗口
CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(launch_config=CONFIG)
print("✅ Isaac Sim 已启动")

# =========================================================
#  Step 1: 导入 URDF
# =========================================================

import omni
import omni.kit.commands
from pxr import Usd, UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools

# 路径配置
PROJECT_ROOT = os.path.expanduser("~/Code/Affordance-Guided-Interaction")
URDF_PATH = os.path.join(PROJECT_ROOT, "assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf")
USD_OUTPUT = os.path.join(PROJECT_ROOT, "assets/robot/usd/uni_dingo_dual_arm.usd")

# 确保 URDF 存在
if not os.path.exists(URDF_PATH):
    print(f"❌ URDF 不存在: {URDF_PATH}")
    simulation_app.close()
    sys.exit(1)

# 新建 Stage
omni.usd.get_context().new_stage()
simulation_app.update()

print(f"📂 导入 URDF: {URDF_PATH}")

# 配置导入参数
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.collision_from_visuals = False
import_config.distance_scale = 1.0
import_config.create_physics_scene = True

# 执行导入
status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
)
simulation_app.update()

if robot_prim_path:
    print(f"✅ 机器人已导入: {robot_prim_path}")
else:
    print("❌ URDF 导入失败")
    simulation_app.close()
    sys.exit(1)

# =========================================================
#  Step 2: 配置场景 (地面 + 灯光)
# =========================================================

stage = omni.usd.get_context().get_stage()

# 添加地面
PhysicsSchemaTools.addGroundPlane(
    stage, "/World/groundPlane", "Z", 1500,
    Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
)

# 添加灯光
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distantLight.CreateIntensityAttr(500)

simulation_app.update()
print("✅ 场景配置完成 (地面 + 灯光)")

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

    is_wheel = any(w in name for w in WHEEL_JOINTS)
    is_arm = any(j in name for j in ARM_JOINTS)

    if is_wheel:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("velocity")
        drive.CreateDampingAttr(500.0)
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
#  Step 4: 保存 USD
# =========================================================

os.makedirs(os.path.dirname(USD_OUTPUT), exist_ok=True)
omni.usd.get_context().save_as_stage(USD_OUTPUT)
print(f"💾 场景已保存: {USD_OUTPUT}")

# =========================================================
#  Step 5: 运行仿真 + 控制机器人
# =========================================================

from omni.isaac.dynamic_control import _dynamic_control

print("\n🎮 开始仿真...")
print("   机器人将执行预设动作序列")
print("   按 Ctrl+C 退出\n")

# --- 重新获取 stage (save_as_stage 后原引用失效) ---
stage = omni.usd.get_context().get_stage()

# --- 遍历 Stage 找到 Articulation Root 的真实路径 ---
art_root_path = None
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_root_path = str(prim.GetPath())
        print(f"🔍 找到 Articulation Root: {art_root_path}")
        break

if art_root_path is None:
    print("⚠️  Stage 中未找到 ArticulationRootAPI, 使用 URDF 导入返回路径")
    art_root_path = robot_prim_path

# 启动物理仿真
omni.timeline.get_timeline_interface().play()

# 等足够帧数让物理引擎完全初始化 (100帧 ≈ 1.7秒)
for _ in range(100):
    simulation_app.update()

# 获取 Dynamic Control 接口
dc = _dynamic_control.acquire_dynamic_control_interface()
art = None

# 尝试候选路径 (用 try/except 防止 null prim 崩溃)
candidate_paths = [
    art_root_path,
    robot_prim_path,
    "/World" + robot_prim_path,
    "/World/uni_dingo_dual_arm",
    "/uni_dingo_dual_arm",
]
# 去重并保持顺序
seen = set()
unique_paths = []
for p in candidate_paths:
    if p and p not in seen:
        seen.add(p)
        unique_paths.append(p)

for path in unique_paths:
    try:
        handle = dc.get_articulation(path)
        if handle != 0:
            art = handle
            print(f"✅ Articulation 句柄获取成功: {path}")
            break
        else:
            print(f"   ❌ {path} → handle=0")
    except RuntimeError:
        print(f"   ❌ {path} → prim 不存在")

if art is None:
    print("\n❌ 无法获取 articulation, 跳过控制演示")
    print("   可能原因:")
    print("   1. 仿真未处于 PLAY 状态")
    print("   2. 机器人缺少 ArticulationRootAPI")
    print("   3. prim 路径不正确")
    print("\n   Stage 中所有 prim (前 30 个):")
    for i, prim in enumerate(Usd.PrimRange(stage.GetPseudoRoot())):
        if i >= 30:
            print("   ... (更多省略)")
            break
        print(f"   {prim.GetPath()}")
else:
    # 列出关节
    num_dofs = dc.get_articulation_dof_count(art)
    print(f"\n📋 关节数量: {num_dofs}")
    for i in range(num_dofs):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        print(f"   [{i:>2}] {name}")

    def set_pos(joint_name, rad):
        dof = dc.find_articulation_dof(art, joint_name)
        if dof != 0:
            dc.set_dof_position_target(dof, rad)

    def set_vel(joint_name, vel):
        dof = dc.find_articulation_dof(art, joint_name)
        if dof != 0:
            dc.set_dof_velocity_target(dof, vel)

    # === 动作序列 ===
    actions = [
        ("🏠 零位", 2.0, lambda: [
            set_pos(j, 0.0) for j in ARM_JOINTS
        ]),
        ("🙌 双臂抬起", 3.0, lambda: [
            set_pos("left_joint2", np.radians(45)),
            set_pos("left_joint3", np.radians(-90)),
            set_pos("left_joint5", np.radians(45)),
            set_pos("right_joint2", np.radians(45)),
            set_pos("right_joint3", np.radians(-90)),
            set_pos("right_joint5", np.radians(45)),
        ]),
        ("👀 云台环顾", 2.0, lambda: [
            set_pos("pan_tilt_yaw_joint", np.radians(30)),
            set_pos("pan_tilt_pitch_joint", np.radians(-15)),
        ]),
        ("🚗 底盘前进", 3.0, lambda: [
            set_vel("front_left_wheel", 5.0),
            set_vel("front_right_wheel", 5.0),
            set_vel("rear_left_wheel", 5.0),
            set_vel("rear_right_wheel", 5.0),
        ]),
        ("🛑 停止", 2.0, lambda: [
            set_vel("front_left_wheel", 0.0),
            set_vel("front_right_wheel", 0.0),
            set_vel("rear_left_wheel", 0.0),
            set_vel("rear_right_wheel", 0.0),
        ]),
        ("🏠 回零位", 2.0, lambda: [
            set_pos(j, 0.0) for j in ARM_JOINTS
        ] + [
            set_pos("pan_tilt_yaw_joint", 0.0),
            set_pos("pan_tilt_pitch_joint", 0.0),
        ]),
    ]

    try:
        for action_name, duration_sec, action_fn in actions:
            print(f"\n▶ {action_name}")
            action_fn()

            # 仿真运行 duration_sec 秒 (60fps)
            for _ in range(int(duration_sec * 60)):
                simulation_app.update()

        print("\n✅ 动作序列完成!")

        # 保持仿真运行，等待用户关闭
        print("按 Ctrl+C 退出...")
        while simulation_app.is_running():
            simulation_app.update()

    except KeyboardInterrupt:
        print("\n⏹ 用户中断")

# 清理
try:
    omni.timeline.get_timeline_interface().stop()
except Exception:
    pass
simulation_app.close()
print("👋 已退出")
