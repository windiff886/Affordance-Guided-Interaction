"""
杯子抓取演示 — 全关节滑块控制 + 终端按 q 记录杯子相对底盘坐标。

场景:
    - 机器人所有关节初始化为零位
    - 一个浮空小板子上放着水杯
    - 通过 UI 滑块实时调整板子+杯子的 XYZ 位置
    - 通过 UI 滑块控制全部 16 个关节 (双臂各 6+gripper, 云台 yaw/pitch)
    - 可切换 Visual / Collision / Both 显示模式
    - 终端输入 q 记录当前杯子相对 base_link 的坐标 (不退出程序)

用法:
    conda activate isaaclab
    python src/teleop_cup_grasp/grasp_demo.py
"""

import os
import sys
import math
import select
import xml.etree.ElementTree as ET
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

# ===================================================================
# 可调参数
# ===================================================================

# 杯子/板子初始位置 (米)
INITIAL_POS = [0.30, 0.15, 0.50]

# 滑块范围 (米)
SLIDER_RANGE_X = (-0.2, 0.8)
SLIDER_RANGE_Y = (-0.5, 0.5)
SLIDER_RANGE_Z = (0.2, 1.0)

# 浮空板尺寸 (米)
TRAY_SIZE = [0.12, 0.12, 0.008]

SIDE = "left"

# ===================================================================
# 路径
# ===================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_USD = os.path.join(PROJECT_ROOT, "assets", "robot", "usd", "uni_dingo_dual_arm.usd")
ROBOT_URDF = os.path.join(PROJECT_ROOT, "assets", "robot", "urdf", "uni_dingo_dual_arm_absolute.urdf")
CUP_USDA = os.path.join(PROJECT_ROOT, "assets", "grasp_objects", "cup", "carry_cup.usda")

# ===================================================================
# 从 URDF 解析关节信息
# ===================================================================

WHEEL_JOINTS = {"front_left_wheel", "front_right_wheel",
                "rear_left_wheel", "rear_right_wheel"}


def collect_link_names(urdf_path):
    root = ET.parse(urdf_path).getroot()
    return [link.attrib["name"] for link in root.findall("link")]


def collect_joint_specs(urdf_path):
    """返回 [(name, lower_rad, upper_rad), ...] 排除 wheel 和 fixed。"""
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
# 非阻塞终端输入
# ===================================================================

def setup_nonblocking_stdin():
    """设置终端为非阻塞模式 (仅 Linux/macOS)。"""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old_settings


def restore_stdin(old_settings):
    """恢复终端设置。"""
    import termios
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def poll_stdin():
    """非阻塞检查 stdin 是否有输入, 返回字符或 None。"""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None

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

# 地面 + 灯光
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 1500,
                                  Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5))
UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(500)

# 物理场景
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

# ── 收紧碰撞体的 contact offset (默认 2cm 太大, 改为 1mm) ──
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
print(f"Set tight contactOffset on {tight_count} robot collision prims")
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
# 显示模式切换: Visual / Collision / Both
# ===================================================================

def _collect_geometry_paths(kind):
    """收集所有 link 下的 visuals 或 collisions 路径。"""
    scope_name = "visuals" if kind == "visuals" else "colliders"
    paths = []
    for link_name in LINK_NAMES:
        paths.append(f"{robot_root_path}/{link_name}/{kind}")
        paths.append(f"/World/Robot/{scope_name}/{link_name}")
    return paths

visual_paths = _collect_geometry_paths("visuals")
collision_paths = _collect_geometry_paths("collisions")
current_display_mode = "visual"
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
    global current_display_mode
    current_display_mode = mode
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


# 初始设为 visual 模式 (collision 隐藏)
apply_display_mode("visual")

# ── 创建浮空板 (Kinematic 刚体) ──
tray_prim = stage.DefinePrim("/World/Tray", "Cube")
tray_xf = UsdGeom.Xformable(tray_prim)
tray_xf.ClearXformOpOrder()
tray_translate_op = tray_xf.AddTranslateOp()
tray_translate_op.Set(Gf.Vec3d(*INITIAL_POS))
tray_xf.AddScaleOp().Set(Gf.Vec3f(TRAY_SIZE[0]/2, TRAY_SIZE[1]/2, TRAY_SIZE[2]/2))
UsdPhysics.RigidBodyAPI.Apply(tray_prim)
tray_prim.GetAttribute("physics:kinematicEnabled").Set(True)
UsdPhysics.CollisionAPI.Apply(tray_prim)
tray_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
    [Gf.Vec3f(0.7, 0.7, 0.75)])

# 板子高摩擦材质
tray_mat = stage.DefinePrim("/World/Tray/PhysMat")
UsdPhysics.MaterialAPI.Apply(tray_mat)
tray_mat.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(2.0)
tray_mat.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(1.5)
tray_mat.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(0.0)
UsdShade.MaterialBindingAPI.Apply(tray_prim).Bind(
    UsdShade.Material(tray_mat), UsdShade.Tokens.weakerThanDescendants, "physics")
sim_app.update()

print("Scene built (robot + tray, cup will be added after stabilization)")

# ===================================================================
# 启动仿真
# ===================================================================

omni.timeline.get_timeline_interface().play()

# 等物理初始化
for _ in range(60):
    sim_app.update()

# 获取 dynamic_control
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


def get_cup_pos():
    b = dc.get_rigid_body("/World/Cup")
    if b == 0:
        b = dc.get_rigid_body("/World/Cup/CarryCup")
    if b != 0:
        p = dc.get_rigid_body_pose(b)
        if p:
            return np.array([p.p.x, p.p.y, p.p.z])
    return np.zeros(3)


def get_base_link_pos():
    """获取 base_link 刚体的世界坐标。"""
    base_path = f"{robot_root_path}/base_link"
    b = dc.get_rigid_body(base_path)
    if b != 0:
        p = dc.get_rigid_body_pose(b)
        if p:
            return np.array([p.p.x, p.p.y, p.p.z])
    return np.zeros(3)


def record_relative_pose():
    """记录杯子相对于 base_link 的坐标并打印。"""
    cup = get_cup_pos()
    base = get_base_link_pos()
    rel = cup - base
    print("\n" + "=" * 60)
    print("  [RECORD] Cup relative to base_link")
    print(f"    base_link world: [{base[0]:.4f}, {base[1]:.4f}, {base[2]:.4f}]")
    print(f"    cup world:       [{cup[0]:.4f}, {cup[1]:.4f}, {cup[2]:.4f}]")
    print(f"    relative (cup-base): [{rel[0]:.4f}, {rel[1]:.4f}, {rel[2]:.4f}]")
    print("=" * 60 + "\n")


# ── 所有关节归零 ──
print("Initializing all joints to zero...")
for s in ["left", "right"]:
    for i in range(1, 7):
        set_joint(f"{s}_joint{i}", 0.0)
    set_joint(f"{s}_jointGripper", 0.0)
set_joint("pan_tilt_yaw_joint", 0.0)
set_joint("pan_tilt_pitch_joint", 0.0)

# 等机器人稳定在零位
for _ in range(240):
    sim_app.update()
print("  Robot at zero position")

# ── 放杯子到板子上 ──
print("Placing cup on tray...")

omni.timeline.get_timeline_interface().pause()
sim_app.update()

cup_prim = stage.DefinePrim("/World/Cup")
cup_prim.GetReferences().AddReference(CUP_USDA)

# 收紧杯子碰撞体 contact offset
cup_tight = 0
for p in Usd.PrimRange(cup_prim):
    if p.HasAPI(UsdPhysics.CollisionAPI):
        set_tight_contact(p)
        cup_tight += 1
print(f"  Set tight contactOffset on {cup_tight} cup collision prims")
cup_xf = UsdGeom.Xformable(cup_prim)
cup_xf.ClearXformOpOrder()
cup_translate_op = cup_xf.AddTranslateOp()
cup_on_tray_z = INITIAL_POS[2] + TRAY_SIZE[2] / 2
cup_translate_op.Set(Gf.Vec3d(INITIAL_POS[0], INITIAL_POS[1], cup_on_tray_z))

sim_app.update()
sim_app.update()

omni.timeline.get_timeline_interface().play()

# 等杯子稳定
for _ in range(120):
    sim_app.update()

cup_pos = get_cup_pos()
print(f"  Cup placed at: [{cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}]")

# ===================================================================
# UI 窗口
# ===================================================================

POS_STEP = 0.005    # 位置步进 5mm
JOINT_STEP = 2.0    # 角度步进 2 deg

# 当前板子位置 (跟踪变化)
tray_pos = list(INITIAL_POS)

window = ui.Window("Cup Grasp Control", width=440, height=850)

# ── 杯子位置 models ──
model_x = ui.SimpleFloatModel(INITIAL_POS[0])
model_y = ui.SimpleFloatModel(INITIAL_POS[1])
model_z = ui.SimpleFloatModel(INITIAL_POS[2])
val_x = val_y = val_z = None

# ── 关节 models: 按 JOINT_SPECS 顺序创建 ──
joint_models = {}   # name -> ui.SimpleFloatModel (度)
joint_labels = {}   # name -> ui.Label (显示值)
for jname, lower_rad, upper_rad in JOINT_SPECS:
    joint_models[jname] = ui.SimpleFloatModel(0.0)

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
    """关节名 → 短标签, 如 left_joint1 → L J1。"""
    n = name
    n = n.replace("left_", "L ").replace("right_", "R ")
    n = n.replace("joint", "J").replace("Gripper", "Grip")
    n = n.replace("pan_tilt_yaw_joint", "PT Yaw")
    n = n.replace("pan_tilt_pitch_joint", "PT Pitch")
    # 如果上面的替换没有命中 pan_tilt, 原样返回
    if "pan_tilt" in name:
        n = name.replace("pan_tilt_", "PT ").replace("_joint", "")
    return n


with window.frame:
    with ui.ScrollingFrame():
        with ui.VStack(spacing=4):
            # ── 显示模式切换 ──
            ui.Label("Display Mode", height=20)
            with ui.HStack(height=28, spacing=6):
                ui.Button("Visual", clicked_fn=lambda: apply_display_mode("visual"))
                ui.Button("Collision", clicked_fn=lambda: apply_display_mode("collision"))
                ui.Button("Both", clicked_fn=lambda: apply_display_mode("both"))
            display_mode_label = ui.Label("Display: visual", height=18)

            ui.Spacer(height=4)

            # ── 杯子/板子位置 ──
            ui.Label("Cup / Tray Position", alignment=ui.Alignment.CENTER, height=20)
            val_x = make_slider_row("X (m)", model_x, *SLIDER_RANGE_X, POS_STEP)
            val_y = make_slider_row("Y (m)", model_y, *SLIDER_RANGE_Y, POS_STEP)
            val_z = make_slider_row("Z (m)", model_z, *SLIDER_RANGE_Z, POS_STEP)

            ui.Spacer(height=4)

            # ── Left Arm ──
            ui.Label("Left Arm", alignment=ui.Alignment.CENTER, height=20)
            for jname, lower_rad, upper_rad in JOINT_SPECS:
                if not jname.startswith("left_"):
                    continue
                lo_deg = math.degrees(lower_rad)
                hi_deg = math.degrees(upper_rad)
                lbl = _joint_slider_label(jname)
                joint_labels[jname] = make_slider_row(
                    lbl, joint_models[jname], lo_deg, hi_deg, JOINT_STEP, fmt=".1f")

            ui.Spacer(height=4)

            # ── Right Arm ──
            ui.Label("Right Arm", alignment=ui.Alignment.CENTER, height=20)
            for jname, lower_rad, upper_rad in JOINT_SPECS:
                if not jname.startswith("right_"):
                    continue
                lo_deg = math.degrees(lower_rad)
                hi_deg = math.degrees(upper_rad)
                lbl = _joint_slider_label(jname)
                joint_labels[jname] = make_slider_row(
                    lbl, joint_models[jname], lo_deg, hi_deg, JOINT_STEP, fmt=".1f")

            ui.Spacer(height=4)

            # ── Pan-Tilt ──
            ui.Label("Pan-Tilt", alignment=ui.Alignment.CENTER, height=20)
            for jname, lower_rad, upper_rad in JOINT_SPECS:
                if "pan_tilt" not in jname:
                    continue
                lo_deg = math.degrees(lower_rad)
                hi_deg = math.degrees(upper_rad)
                lbl = _joint_slider_label(jname)
                joint_labels[jname] = make_slider_row(
                    lbl, joint_models[jname], lo_deg, hi_deg, JOINT_STEP, fmt=".1f")

            ui.Spacer(height=6)
            cup_pos_label = ui.Label("Cup: ---", height=20)

print("UI ready.")
print(">>> Press 'q' in terminal to record cup-vs-base_link relative pose <<<")

# ===================================================================
# 主循环
# ===================================================================

old_term = setup_nonblocking_stdin()

try:
    while sim_app.is_running():
        sim_app.update()

        # ── 非阻塞检查终端输入 ──
        ch = poll_stdin()
        if ch == "q":
            record_relative_pose()

        # ── 杯子/板子位置跟随滑块 ──
        x = model_x.get_value_as_float()
        y = model_y.get_value_as_float()
        z = model_z.get_value_as_float()

        if (abs(x - tray_pos[0]) > 1e-5 or
            abs(y - tray_pos[1]) > 1e-5 or
            abs(z - tray_pos[2]) > 1e-5):
            tray_pos[:] = [x, y, z]
            tray_translate_op.Set(Gf.Vec3d(x, y, z))
            cup_on_tray_z = z + TRAY_SIZE[2] / 2
            cup_translate_op.Set(Gf.Vec3d(x, y, cup_on_tray_z))

        if val_x: val_x.text = f"{x:.3f}"
        if val_y: val_y.text = f"{y:.3f}"
        if val_z: val_z.text = f"{z:.3f}"

        # ── 所有关节跟随滑块 ──
        dc.wake_up_articulation(art)
        for jname, _, _ in JOINT_SPECS:
            if jname not in dof_map or jname not in joint_models:
                continue
            deg = joint_models[jname].get_value_as_float()
            dc.set_dof_position_target(dof_map[jname], math.radians(deg))
            if jname in joint_labels and joint_labels[jname] is not None:
                joint_labels[jname].text = f"{deg:.1f}"

        # ── 更新杯子位置显示 ──
        cup_pos = get_cup_pos()
        if cup_pos_label:
            cup_pos_label.text = f"Cup: [{cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}]"

except KeyboardInterrupt:
    print("\nUser interrupted (Ctrl+C)")

finally:
    restore_stdin(old_term)

omni.timeline.get_timeline_interface().stop()
sim_app.close()
print("Done")
