"""
抓取回放 — 左/右臂各自抓取水杯, 可独立启用/禁用。

流程 (对每个启用的臂):
    1. 机器人所有关节归零
    2. 臂 joint6 预设角度, gripper 张开 (-90°)
    3. 杯子按 base_link + 相对偏移放置
    4. 夹爪从 -90° 平滑闭合到 -34°
    5. 仿真持续运行, 可通过滑块操作所有关节

左右臂对称关系 (URDF 中左右臂关于 Y=0 平面镜像):
    - 杯子位置: Y 取反
    - joint1 (yaw): 角度取反
    - joint6 (wrist roll): 角度取反
    - joint2~5, gripper: 角度不变

用法:
    conda activate isaaclab
    python src/teleop_cup_grasp/grasp_replay.py
"""

import os
import sys
import math
import xml.etree.ElementTree as ET
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

# ===================================================================
# 左/右臂抓取开关 — 设为 True 则该臂执行抓取
# ===================================================================

ENABLE_LEFT_GRASP = True
ENABLE_RIGHT_GRASP = True

# ===================================================================
# 左臂基准数据 (来自 grasp_demo.py 按 q 记录)
# ===================================================================

# 杯子相对 base_link 的坐标
LEFT_CUP_RELATIVE = [0.2900, 0.1111, 0.6814]

# 左臂预抓取姿态 (度)
LEFT_ARM_INIT_DEG = {
    "left_joint1": 0.0,
    "left_joint2": 0.0,
    "left_joint3": 0.0,
    "left_joint4": 0.0,
    "left_joint5": 0.0,
    "left_joint6": 90.0,
    "left_jointGripper": -90.0,
}

# ===================================================================
# 右臂数据 — 由左臂镜像得出 (Y 取反, J1/J6 取反)
# ===================================================================

RIGHT_CUP_RELATIVE = [LEFT_CUP_RELATIVE[0],
                      -LEFT_CUP_RELATIVE[1],
                      LEFT_CUP_RELATIVE[2]]

# J1, J6 取反; J2~J5, Gripper 不变
RIGHT_ARM_INIT_DEG = {
    "right_joint1": -LEFT_ARM_INIT_DEG["left_joint1"],
    "right_joint2":  LEFT_ARM_INIT_DEG["left_joint2"],
    "right_joint3":  LEFT_ARM_INIT_DEG["left_joint3"],
    "right_joint4": -LEFT_ARM_INIT_DEG["left_joint4"],
    "right_joint5":  LEFT_ARM_INIT_DEG["left_joint5"],
    "right_joint6": -LEFT_ARM_INIT_DEG["left_joint6"],
    "right_jointGripper": LEFT_ARM_INIT_DEG["left_jointGripper"],
}

# 夹爪闭合参数
GRIPPER_CLOSE_DEG = -34.0
GRIPPER_CLOSE_FRAMES = 120

# ===================================================================
# 可调参数
# ===================================================================

TRAY_SIZE = [0.12, 0.12, 0.008]

# ===================================================================
# 路径
# ===================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_USD = os.path.join(PROJECT_ROOT, "assets", "robot", "usd", "uni_dingo_lite.usd")
ROBOT_URDF = os.path.join(PROJECT_ROOT, "assets", "robot", "urdf", "uni_dingo_lite.urdf")
CUP_USDA = os.path.join(PROJECT_ROOT, "assets", "grasp_objects", "cup", "carry_cup.usda")

# ===================================================================
# 从 URDF 解析关节信息
# ===================================================================

WHEEL_JOINTS = set()  # lite 版本无轮子


def collect_link_names(urdf_path):
    root = ET.parse(urdf_path).getroot()
    return [link.attrib["name"] for link in root.findall("link")]


def collect_joint_specs(urdf_path):
    root = ET.parse(urdf_path).getroot()
    specs = []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        if name in WHEEL_JOINTS:
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


LINK_NAMES = collect_link_names(ROBOT_URDF)
JOINT_SPECS = collect_joint_specs(ROBOT_URDF)

# ===================================================================
# 构建启用列表
# ===================================================================

# 每个启用的臂: (side, cup_relative, arm_init_deg, cup_prim_name, tray_prim_name)
grasp_tasks = []
if ENABLE_LEFT_GRASP:
    grasp_tasks.append(("left", LEFT_CUP_RELATIVE, LEFT_ARM_INIT_DEG,
                        "/World/CupLeft", "/World/TrayLeft"))
if ENABLE_RIGHT_GRASP:
    grasp_tasks.append(("right", RIGHT_CUP_RELATIVE, RIGHT_ARM_INIT_DEG,
                        "/World/CupRight", "/World/TrayRight"))

enabled_str = []
if ENABLE_LEFT_GRASP:
    enabled_str.append("LEFT")
if ENABLE_RIGHT_GRASP:
    enabled_str.append("RIGHT")
print(f"Grasp tasks: {' + '.join(enabled_str) if enabled_str else 'NONE'}")

# ===================================================================
# 启动 Isaac Sim
# ===================================================================

print("Starting Isaac Sim...")
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
print("Isaac Sim started OK")

import omni
import omni.ui as ui
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf, PhysicsSchemaTools, UsdShade, PhysxSchema

# ===================================================================
# 搭建场景
# ===================================================================

omni.usd.get_context().new_stage()
sim_app.update()
stage = omni.usd.get_context().get_stage()

PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 1500,
                                  Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5))
UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(500)

physics_path = "/World/PhysicsScene"
if not stage.GetPrimAtPath(physics_path).IsValid():
    UsdPhysics.Scene.Define(stage, physics_path)
pp = stage.GetPrimAtPath(physics_path)
pp.GetAttribute("physics:gravityDirection").Set(Gf.Vec3f(0, 0, -1))
pp.GetAttribute("physics:gravityMagnitude").Set(9.81)
sim_app.update()

# ── 加载机器人 ──
robot_prim = stage.DefinePrim("/World/Robot")
robot_prim.GetReferences().AddReference(ROBOT_USD)
sim_app.update()

# ── 收紧碰撞体的 contact offset ──
def set_tight_contact(prim):
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
    prim.GetAttribute("physxCollision:contactOffset").Set(0.001)
    prim.GetAttribute("physxCollision:restOffset").Set(0.0001)

tight_count = 0
for p in Usd.PrimRange(robot_prim):
    if p.HasAPI(UsdPhysics.CollisionAPI):
        set_tight_contact(p)
        tight_count += 1
# lite USD 可能未对 collision mesh 施加 CollisionAPI，主动修复
repaired = 0
for p in Usd.PrimRange(robot_prim):
    if p.GetName() == "collisions" or str(p.GetPath()).endswith("/collisions"):
        for child in p.GetChildren():
            if child.IsValid() and child.GetTypeName() in ("Mesh", "Cube", "Cylinder", "Sphere", "Cone", "Capsule"):
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(child)
                set_tight_contact(child)
                repaired += 1
tight_count += repaired
print(f"Set tight contactOffset on {tight_count} robot collision prims (repaired {repaired})")
sim_app.update()

# ── 找到机器人根路径 ──
robot_root_path = None
for child in robot_prim.GetChildren():
    if child and child.IsValid() and child.GetTypeName() == "Xform":
        child_name = child.GetName()
        if child_name not in ("visuals", "colliders", "meshes"):
            robot_root_path = str(child.GetPath())
            break
if robot_root_path is None:
    print("ERROR: Robot root prim not found!")
    sim_app.close()
    sys.exit(1)
print(f"Robot root: {robot_root_path}")

# ===================================================================
# 显示模式切换
# ===================================================================

def _collect_geometry_paths(kind):
    scope_name = "visuals" if kind == "visuals" else "colliders"
    paths = []
    for link_name in LINK_NAMES:
        paths.append(f"{robot_root_path}/{link_name}/{kind}")
        paths.append(f"/World/Robot/{scope_name}/{link_name}")
    return paths

visual_paths = _collect_geometry_paths("visuals")
collision_paths = _collect_geometry_paths("collisions")
display_mode_label = None


def set_visibility(paths, visible):
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


def apply_display_mode(mode):
    if mode == "visual":
        set_visibility(visual_paths, True)
        set_visibility(collision_paths, False)
    elif mode == "collision":
        set_visibility(visual_paths, False)
        set_visibility(collision_paths, True)
    elif mode == "both":
        set_visibility(visual_paths, True)
        set_visibility(collision_paths, True)
    if display_mode_label is not None:
        display_mode_label.text = f"Display: {mode}"


apply_display_mode("visual")

# ===================================================================
# 启动仿真 + 获取 dynamic_control
# ===================================================================

omni.timeline.get_timeline_interface().play()
for _ in range(60):
    sim_app.update()

from omni.isaac.dynamic_control import _dynamic_control
dc = _dynamic_control.acquire_dynamic_control_interface()

art = None
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art = dc.get_articulation(str(prim.GetPath()))
        if art != 0:
            print(f"Articulation found: {prim.GetPath()}")
            break

if art is None or art == 0:
    print("ERROR: No articulation found!")
    sim_app.close()
    sys.exit(1)

num_dofs = dc.get_articulation_dof_count(art)
dof_map = {}
for i in range(num_dofs):
    dof = dc.get_articulation_dof(art, i)
    dof_map[dc.get_dof_name(dof)] = dof
print(f"DOFs: {num_dofs}")


def set_joint(name, deg):
    if name in dof_map:
        dc.set_dof_position_target(dof_map[name], math.radians(deg))


def get_body_pos(path):
    b = dc.get_rigid_body(path)
    if b != 0:
        p = dc.get_rigid_body_pose(b)
        if p:
            return np.array([p.p.x, p.p.y, p.p.z])
    return np.zeros(3)


def get_base_link_pos():
    # lite USD 的根 prim 就是 base_link，无需再拼子路径
    if robot_root_path.endswith("/base_link"):
        return get_body_pos(robot_root_path)
    return get_body_pos(f"{robot_root_path}/base_link")

# ===================================================================
# 辅助: 创建板子 + 杯子
# ===================================================================

def create_tray(prim_path, position):
    """创建一个 kinematic 浮空板, 返回 translate op。"""
    tray_prim = stage.DefinePrim(prim_path, "Cube")
    tray_xf = UsdGeom.Xformable(tray_prim)
    tray_xf.ClearXformOpOrder()
    translate_op = tray_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*position))
    tray_xf.AddScaleOp().Set(Gf.Vec3f(TRAY_SIZE[0]/2, TRAY_SIZE[1]/2, TRAY_SIZE[2]/2))
    UsdPhysics.RigidBodyAPI.Apply(tray_prim)
    tray_prim.GetAttribute("physics:kinematicEnabled").Set(True)
    UsdPhysics.CollisionAPI.Apply(tray_prim)
    tray_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
        [Gf.Vec3f(0.7, 0.7, 0.75)])

    mat_path = prim_path + "/PhysMat"
    mat_prim = stage.DefinePrim(mat_path)
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(2.0)
    mat_prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(1.5)
    mat_prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(tray_prim).Bind(
        UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")

    return translate_op


def create_cup(prim_path, position):
    """创建杯子, 收紧 contactOffset, 返回 translate op。"""
    cup_prim = stage.DefinePrim(prim_path)
    cup_prim.GetReferences().AddReference(CUP_USDA)

    n = 0
    for p in Usd.PrimRange(cup_prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            set_tight_contact(p)
            n += 1
    print(f"  Set tight contactOffset on {n} collision prims for {prim_path}")

    cup_xf = UsdGeom.Xformable(cup_prim)
    cup_xf.ClearXformOpOrder()
    translate_op = cup_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*position))
    return translate_op

# ===================================================================
# 阶段 1: 所有关节归零
# ===================================================================

print("Initializing all joints to zero...")
for s in ["left", "right"]:
    for i in range(1, 7):
        set_joint(f"{s}_joint{i}", 0.0)
    set_joint(f"{s}_jointGripper", 0.0)
# lite 版本无 pan_tilt 关节，跳过

for _ in range(240):
    sim_app.update()
print("  Robot at zero position")

# ===================================================================
# 阶段 2: 设置各臂预抓取姿态
# ===================================================================

# 合并所有启用臂的初始姿态
all_arm_init = {}
for side, cup_rel, arm_init, _, _ in grasp_tasks:
    all_arm_init.update(arm_init)

print("Setting arm replay poses...")
for jname, deg in all_arm_init.items():
    set_joint(jname, deg)
    print(f"  {jname} -> {deg:.1f} deg")

for _ in range(240):
    sim_app.update()
print("  Arms at replay pose")

# ===================================================================
# 阶段 3: 放杯子 — 每个臂各自一个杯子
# ===================================================================

base_pos = get_base_link_pos()
print(f"base_link at: [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]")

omni.timeline.get_timeline_interface().pause()
sim_app.update()

# 记录每个任务的运行时数据
task_data = []  # [(side, tray_translate_op, cup_translate_op, tray_pos_list)]

for side, cup_rel, arm_init, cup_path, tray_path in grasp_tasks:
    cup_world = base_pos + np.array(cup_rel)
    tray_world = cup_world.copy()
    tray_world[2] -= TRAY_SIZE[2] / 2

    print(f"  [{side}] cup target: [{cup_world[0]:.4f}, {cup_world[1]:.4f}, {cup_world[2]:.4f}]")

    tray_op = create_tray(tray_path, tray_world.tolist())
    sim_app.update()
    cup_op = create_cup(cup_path, cup_world.tolist())
    sim_app.update()

    task_data.append((side, tray_op, cup_op, tray_world.tolist()))

sim_app.update()
omni.timeline.get_timeline_interface().play()

for _ in range(120):
    sim_app.update()
print("  Cups placed")

# ===================================================================
# 阶段 4: 所有启用臂的夹爪同时闭合
# ===================================================================

GRIP_OPEN = -90.0
GRIP_CLOSE = GRIPPER_CLOSE_DEG

print(f"Closing grippers: {GRIP_OPEN:.1f} -> {GRIP_CLOSE:.1f} over {GRIPPER_CLOSE_FRAMES} frames...")

for frame in range(GRIPPER_CLOSE_FRAMES):
    t = (frame + 1) / GRIPPER_CLOSE_FRAMES
    t_smooth = t * t * (3.0 - 2.0 * t)
    grip_deg = GRIP_OPEN + t_smooth * (GRIP_CLOSE - GRIP_OPEN)
    for side, _, _, _ in task_data:
        set_joint(f"{side}_jointGripper", grip_deg)
    dc.wake_up_articulation(art)
    sim_app.update()

for _ in range(60):
    dc.wake_up_articulation(art)
    sim_app.update()

print(f"  Grippers closed at {GRIP_CLOSE:.1f} deg")

# ── 删除浮空板 ──
for side, cup_rel, arm_init, cup_path, tray_path in grasp_tasks:
    stage.RemovePrim(tray_path)
    print(f"  Removed tray: {tray_path}")
sim_app.update()

# ===================================================================
# UI 窗口
# ===================================================================

POS_STEP = 0.005
JOINT_STEP = 2.0

window = ui.Window("Grasp Replay Control", width=440, height=850)

SLIDER_RANGE_X = (-0.2, 0.8)
SLIDER_RANGE_Y = (-0.5, 0.5)
SLIDER_RANGE_Z = (0.2, 1.0)

# 关节 models — 启用的臂用闭合后角度, 其余为 0
joint_models = {}
joint_labels = {}
for jname, lower_rad, upper_rad in JOINT_SPECS:
    # 先查是否在启用臂的初始角度中
    init_deg = all_arm_init.get(jname, 0.0)
    # 如果是启用臂的夹爪, 用闭合后的角度
    for side, _, _, _ in task_data:
        if jname == f"{side}_jointGripper":
            init_deg = GRIP_CLOSE
    joint_models[jname] = ui.SimpleFloatModel(init_deg)

cup_pos_label = None


def make_slider_row(label_text, model, range_min, range_max, step, fmt=".3f"):
    vl = None
    with ui.HStack(height=28, spacing=4):
        ui.Label(label_text, width=130)

        def on_minus(m=model, s=step, lo=range_min):
            m.set_value(max(lo, m.get_value_as_float() - s))

        def on_plus(m=model, s=step, hi=range_max):
            m.set_value(min(hi, m.get_value_as_float() + s))

        ui.Button("-", width=24, clicked_fn=on_minus)
        ui.FloatSlider(model=model, min=range_min, max=range_max)
        ui.Button("+", width=24, clicked_fn=on_plus)
        vl = ui.Label(f"{model.get_value_as_float():{fmt}}", width=60)
    return vl


def _joint_slider_label(name):
    if "pan_tilt" in name:
        return name.replace("pan_tilt_", "PT ").replace("_joint", "")
    n = name
    n = n.replace("left_", "L ").replace("right_", "R ")
    n = n.replace("joint", "J").replace("Gripper", "Grip")
    return n


with window.frame:
    with ui.ScrollingFrame():
        with ui.VStack(spacing=4):
            # 显示模式
            ui.Label("Display Mode", height=20)
            with ui.HStack(height=28, spacing=6):
                ui.Button("Visual", clicked_fn=lambda: apply_display_mode("visual"))
                ui.Button("Collision", clicked_fn=lambda: apply_display_mode("collision"))
                ui.Button("Both", clicked_fn=lambda: apply_display_mode("both"))
            display_mode_label = ui.Label("Display: visual", height=18)

            ui.Spacer(height=4)

            # Left Arm
            ui.Label("Left Arm", alignment=ui.Alignment.CENTER, height=20)
            for jname, lower_rad, upper_rad in JOINT_SPECS:
                if not jname.startswith("left_"):
                    continue
                lo_deg = math.degrees(lower_rad)
                hi_deg = math.degrees(upper_rad)
                joint_labels[jname] = make_slider_row(
                    _joint_slider_label(jname), joint_models[jname],
                    lo_deg, hi_deg, JOINT_STEP, fmt=".1f")

            ui.Spacer(height=4)

            # Right Arm
            ui.Label("Right Arm", alignment=ui.Alignment.CENTER, height=20)
            for jname, lower_rad, upper_rad in JOINT_SPECS:
                if not jname.startswith("right_"):
                    continue
                lo_deg = math.degrees(lower_rad)
                hi_deg = math.degrees(upper_rad)
                joint_labels[jname] = make_slider_row(
                    _joint_slider_label(jname), joint_models[jname],
                    lo_deg, hi_deg, JOINT_STEP, fmt=".1f")

            ui.Spacer(height=4)

            # Pan-Tilt
            ui.Label("Pan-Tilt", alignment=ui.Alignment.CENTER, height=20)
            for jname, lower_rad, upper_rad in JOINT_SPECS:
                if "pan_tilt" not in jname:
                    continue
                lo_deg = math.degrees(lower_rad)
                hi_deg = math.degrees(upper_rad)
                joint_labels[jname] = make_slider_row(
                    _joint_slider_label(jname), joint_models[jname],
                    lo_deg, hi_deg, JOINT_STEP, fmt=".1f")

            ui.Spacer(height=6)
            cup_pos_label = ui.Label("", height=20)

            tasks_str = ", ".join(s.upper() for s, _, _, _ in task_data)
            ui.Label(f"Active: {tasks_str}. Grippers closed. Free control.", height=20)

print("UI ready. You can now operate all joints freely.")

# ===================================================================
# 主循环
# ===================================================================

try:
    while sim_app.is_running():
        sim_app.update()

        # 所有关节跟随滑块
        dc.wake_up_articulation(art)
        for jname, _, _ in JOINT_SPECS:
            if jname not in dof_map or jname not in joint_models:
                continue
            deg = joint_models[jname].get_value_as_float()
            dc.set_dof_position_target(dof_map[jname], math.radians(deg))
            if jname in joint_labels and joint_labels[jname] is not None:
                joint_labels[jname].text = f"{deg:.1f}"

        # 杯子位置显示
        if cup_pos_label:
            parts = []
            for side, _, _, _ in task_data:
                cup_path = f"/World/Cup{side.capitalize()}"
                for try_path in (cup_path, cup_path + "/CarryCup"):
                    pos = get_body_pos(try_path)
                    if np.any(pos != 0):
                        parts.append(f"{side[0].upper()}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                        break
            cup_pos_label.text = "  ".join(parts) if parts else ""

except KeyboardInterrupt:
    print("\nUser interrupted (Ctrl+C)")

omni.timeline.get_timeline_interface().stop()
sim_app.close()
print("Done")
