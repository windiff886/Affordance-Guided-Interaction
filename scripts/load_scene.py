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
import argparse
import numpy as np

# =========================================================
#  Step 0: 解析参数 & 启动 SimulationApp
# =========================================================

parser = argparse.ArgumentParser(description="推门场景 + Uni-Dingo 机器人")
parser.add_argument("--headless", action="store_true", help="无窗口模式")
parser.add_argument("--show-collisions", action="store_true", help="显示碰撞体调试可视化")
args, _ = parser.parse_known_args()

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

PROJECT_ROOT = os.path.expanduser("~/Code/Affordance-Guided-Interaction")
DOOR_SCENE = os.path.join(PROJECT_ROOT, "assets/minimal_push_door/minimal_push_door.usda")
ROBOT_USD = os.path.join(PROJECT_ROOT, "assets/robot/uni_dingo_dual_arm.usd")

# 验证文件存在
for f, desc in [(DOOR_SCENE, "门场景"), (ROBOT_USD, "机器人 USD")]:
    if not os.path.exists(f):
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
stage.GetRootLayer().subLayerPaths.append(DOOR_SCENE)

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
robot_prim.GetReferences().AddReference(ROBOT_USD)

for _ in range(10):
    simulation_app.update()

print(f"✅ 机器人已加载: {robot_prim_path} (reference 方式, 含自碰撞 + 关节驱动)")

# =========================================================
#  Step 4: 设置机器人位置
# =========================================================

xformable = UsdGeom.Xformable(robot_prim)
xformable.ClearXformOpOrder()
translate_op = xformable.AddTranslateOp()
translate_op.Set(Gf.Vec3d(robot_base_pos[0], robot_base_pos[1], 0.12))
rotate_op = xformable.AddRotateXYZOp()
rotate_op.Set(Gf.Vec3d(0, 0, 180))
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
    xf.AddTranslateOp().Set(Gf.Vec3d(5.0, 0, -0.05))
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
    ("left_jointGripper", -1, 1, 0, "Gripper"),
]
RIGHT_ARM_UI = [
    ("right_joint1",  -150, 150,  0, "Shoulder Z"),
    ("right_joint2",     0, 170,  0, "Shoulder Y"),
    ("right_joint3",  -165,   0,  0, "Elbow Y"),
    ("right_joint4",  -165, 165,  0, "Forearm Y"),
    ("right_joint5",  -165, 165,  0, "Wrist Z"),
    ("right_joint6",   -90,  90,  0, "Wrist X"),
    ("right_jointGripper", -1, 1, 0, "Gripper"),
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

# 主循环
try:
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
