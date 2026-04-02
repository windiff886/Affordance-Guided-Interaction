"""
Teleop Cup Grasping - XYZ end-effector control with geometric Jacobian IK.

Controls:
    XYZ sliders  -> move left arm end-effector via IK
    Yaw/Pitch/Roll -> end-effector orientation target
    Gripper      -> left_jointGripper
    [Space]      -> remove support platform & save trajectory
"""

import os, sys
from pathlib import Path
import numpy as np

# =========================================================
#  Environment
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.utils.runtime_env import configure_omniverse_client_environment
from affordance_guided_interaction.utils.pose_alignment import (
    calibrate_pose_alignment,
    model_pose_to_sim_frame,
    relative_pose_in_parent_frame,
    rotation_distance_deg,
    row_major_rotation_to_column_major,
    sim_pose_to_model_frame,
)
from affordance_guided_interaction.utils.usd_assets import to_usd_asset_path
from teleop_cup_grasp.grasp_demo_io import GraspDemo, save_grasp_demo_npz
configure_omniverse_client_environment(os.environ)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
print("SimulationApp ready")

import omni, carb
from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics, Sdf, PhysicsSchemaTools
from omni.isaac.dynamic_control import _dynamic_control
import omni.ui as ui

# =========================================================
#  Asset paths
# =========================================================
DOOR_SCENE = PROJECT_ROOT / "assets/minimal_push_door/minimal_push_door.usda"
ROBOT_USD  = PROJECT_ROOT / "assets/robot/uni_dingo_dual_arm.usd"
CUP_USD    = PROJECT_ROOT / "assets/grasp_objects/cup/carry_cup.usda"
DEMO_OUTPUT_PATH = PROJECT_ROOT / "src/teleop_cup_grasp/grasp_demo.npz"
RECORD_DT = 1.0 / 60.0

for f in [ROBOT_USD, CUP_USD]:
    if not f.exists():
        print(f"MISSING: {f}"); simulation_app.close(); sys.exit(1)

# =========================================================
#  Step 1 - Stage + scene
# =========================================================
omni.usd.get_context().new_stage(); simulation_app.update()
stage = omni.usd.get_context().get_stage()

if DOOR_SCENE.exists():
    stage.GetRootLayer().subLayerPaths.append(str(DOOR_SCENE))
else:
    PhysicsSchemaTools.addPhysicsScene(stage, "/World/PhysicsScene")
    lt = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/SphereLight"))
    lt.GetRadiusAttr().Set(2.0); lt.GetIntensityAttr().Set(30000.0)
    UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3f(0,0,5))
    fl = UsdGeom.Cube.Define(stage, Sdf.Path("/World/GroundPlane"))
    fl.GetSizeAttr().Set(1.0); fl.GetDisplayColorAttr().Set([(0.52,0.54,0.56)])
    xf=UsdGeom.Xformable(fl.GetPrim())
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0,0,-0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(10,10,0.1))
    UsdPhysics.CollisionAPI.Apply(fl.GetPrim())

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
for _ in range(20): simulation_app.update()

# =========================================================
#  Step 2 - Robot
# =========================================================
robot_prim = stage.DefinePrim("/World/Robot", "Xform")
robot_prim.GetReferences().AddReference(to_usd_asset_path(ROBOT_USD))
for _ in range(10): simulation_app.update()

xf = UsdGeom.Xformable(robot_prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0,0,0.12))
simulation_app.update()

# =========================================================
#  Step 3 - Ground
# =========================================================
gp = "/World/TeleopGround"
if not stage.GetPrimAtPath(gp).IsValid():
    g = UsdGeom.Cube.Define(stage, Sdf.Path(gp))
    g.GetSizeAttr().Set(1.0); g.GetDisplayColorAttr().Set([(0.52,0.54,0.56)])
    xf=UsdGeom.Xformable(g.GetPrim())
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0,0,-0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(6,6,0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim()); simulation_app.update()

# =========================================================
#  Step 4 - Cup + support
# =========================================================
PICKUP_POINT_BASE = (0.45, 0.20, 0.55)
bl = stage.GetPrimAtPath("/World/Robot/base_link")
if bl.IsValid():
    bw = UsdGeom.Xformable(bl).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cwp = bw.Transform(Gf.Vec3d(*PICKUP_POINT_BASE))
else:
    cwp = Gf.Vec3d(*PICKUP_POINT_BASE)

SS = (0.12, 0.12, 0.02)
sc = Gf.Vec3f(float(cwp[0]), float(cwp[1]), float(cwp[2]) - 0.5*SS[2])
support_prim = UsdGeom.Cube.Define(stage, Sdf.Path("/World/PickupSupport")).GetPrim()
UsdGeom.Cube(support_prim).GetSizeAttr().Set(1.0)
UsdGeom.Cube(support_prim).GetDisplayColorAttr().Set([(0.45,0.37,0.25)])
xs = UsdGeom.Xformable(support_prim); xs.ClearXformOpOrder()
xs.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(sc)
xs.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*SS))
UsdPhysics.CollisionAPI.Apply(support_prim)

carry_prim = stage.DefinePrim("/World/CarryObject", "Xform")
carry_prim.GetReferences().AddReference(to_usd_asset_path(CUP_USD))
xc = UsdGeom.Xformable(carry_prim); xc.ClearXformOpOrder()
xc.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(
    Gf.Vec3f(float(cwp[0]), float(cwp[1]), float(cwp[2])))
simulation_app.update()
for _ in range(20): simulation_app.update()

# =========================================================
#  Step 5 - Start sim + articulation
# =========================================================
omni.timeline.get_timeline_interface().play()
for _ in range(120): simulation_app.update()

dc = _dynamic_control.acquire_dynamic_control_interface()
art = dc.get_articulation("/World/Robot/base_link")
if art == _dynamic_control.INVALID_HANDLE:
    print("Cannot get articulation"); simulation_app.close(); sys.exit(1)

num_dofs = dc.get_articulation_dof_count(art)
for i in range(num_dofs):
    print(f"  [{i:>2}] {dc.get_dof_name(dc.get_articulation_dof(art, i))}")

def set_pos(jn, rad):
    d = dc.find_articulation_dof(art, jn)
    if d != _dynamic_control.INVALID_HANDLE: dc.set_dof_position_target(d, rad)

def get_pos(jn):
    d = dc.find_articulation_dof(art, jn)
    return float(dc.get_dof_position(d)) if d != _dynamic_control.INVALID_HANDLE else 0.0

# =========================================================
#  Step 6 - Pinocchio IK setup + Isaac Sim EE readout
# =========================================================
import pinocchio as pin

URDF_PATH = str(PROJECT_ROOT / "assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf")
IK_JOINTS = ["left_joint1","left_joint2","left_joint3","left_joint4",
             "left_joint5","left_joint6"]
ALL_RECORD_JOINTS = IK_JOINTS + ["left_jointGripper"]

# Load full model
full_model = pin.buildModelFromUrdf(URDF_PATH)

# Build reduced model: fix all joints except left arm 1-6
left_arm_jids = [full_model.getJointId(n) for n in IK_JOINTS]
fix_jids = [i for i in range(1, full_model.njoints) if i not in left_arm_jids]
q_ref = pin.neutral(full_model)
pin_model = pin.buildReducedModel(full_model, fix_jids, q_ref)
pin_data  = pin_model.createData()

# EE frame id in reduced model
EE_FRAME_ID = pin_model.getFrameId("left_gripperStator")
print(f"Pinocchio reduced model: nq={pin_model.nq}, nv={pin_model.nv}")
print(f"EE frame: {pin_model.frames[EE_FRAME_ID].name} (id={EE_FRAME_ID})")

def pin_jacobian(q_vec):
    """Compute 6xN frame Jacobian in LOCAL_WORLD_ALIGNED frame.
    In the reduced model, 'world' = base_link frame."""
    q = np.array(q_vec, dtype=np.float64)
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin.computeFrameJacobian(
        pin_model, pin_data, q, EE_FRAME_ID,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

def pin_fk(q_vec):
    """Compute EE pose in base_link frame using Pinocchio FK.
    Returns (position_3, rotation_3x3)."""
    q = np.array(q_vec, dtype=np.float64)
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf = pin_data.oMf[EE_FRAME_ID]
    return oMf.translation.copy(), oMf.rotation.copy()

# --- Isaac Sim prim helpers (ground truth) ---
def find_prim(name):
    for p in stage.Traverse():
        if p.GetName() == name: return p
    return None

BASE_LINK_PRIM = stage.GetPrimAtPath("/World/Robot/base_link")
EE_PRIM = stage.GetPrimAtPath("/World/Robot/left_gripperStator")
if not BASE_LINK_PRIM.IsValid():
    BASE_LINK_PRIM = find_prim("base_link")
if not EE_PRIM.IsValid():
    EE_PRIM = find_prim("left_gripperStator")
print(f"Isaac Sim prims: base_link={BASE_LINK_PRIM.GetPath() if BASE_LINK_PRIM else None}, "
      f"EE={EE_PRIM.GetPath() if EE_PRIM else None}")

def get_world_transform(prim):
    """Get 4x4 world transform from a USD prim."""
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    # Gf matrices use row-vector convention, so transpose the rotation block
    # before using it with NumPy column-vector math.
    row_major_rot = np.array([[xf[r][c] for c in range(3)] for r in range(3)],
                             dtype=np.float64)
    R = row_major_rotation_to_column_major(row_major_rot)
    t = np.array(xf.ExtractTranslation(), dtype=np.float64)
    return R, t

def get_ee_in_baselink():
    """Get EE pose relative to base_link from Isaac Sim (ground truth).
    Returns (position_3, rotation_3x3) in base_link frame."""
    R_base, t_base = get_world_transform(BASE_LINK_PRIM)
    R_ee, t_ee     = get_world_transform(EE_PRIM)
    return relative_pose_in_parent_frame(t_base, R_base, t_ee, R_ee)

# --- Euler angle utilities ---
def rot_from_ypr(yaw, pitch, roll):
    """Build rotation matrix from Yaw(Z)-Pitch(Y)-Roll(X), angles in rad."""
    return pin.rpy.rpyToMatrix(roll, pitch, yaw)

def ypr_from_rot(R):
    """Extract Yaw-Pitch-Roll from rotation matrix, returns degrees."""
    rpy = pin.rpy.matrixToRpy(R)  # returns [roll, pitch, yaw]
    return np.degrees(rpy[2]), np.degrees(rpy[1]), np.degrees(rpy[0])

for _ in range(30): simulation_app.update()

# Initial EE pose from Pinocchio FK (consistent with IK Jacobian)
q_init = np.array([get_pos(jn) for jn in IK_JOINTS], dtype=np.float64)
init_model_pos, init_model_rot = pin_fk(q_init)
init_sim_pos, init_sim_rot = get_ee_in_baselink()
ee_align_pos, ee_align_rot = calibrate_pose_alignment(
    init_model_pos, init_model_rot, init_sim_pos, init_sim_rot)
ik_target_pos = init_sim_pos.copy()
init_ypr = ypr_from_rot(init_sim_rot)
ik_target_ypr = list(init_ypr)  # [yaw_deg, pitch_deg, roll_deg]
print(f"[USD EE] init pos: {ik_target_pos}")
print(f"[USD EE] init YPR: yaw={init_ypr[0]:.1f}° "
      f"pitch={init_ypr[1]:.1f}° roll={init_ypr[2]:.1f}°")
print(f"[EE model->USD] translation offset: {ee_align_pos.round(6)}")
print(f"[EE model->USD] rotation offset: {rotation_distance_deg(ee_align_rot, np.eye(3)):.2f} deg")

# =========================================================
#  Step 7 - UI: XYZ + YPR + Gripper sliders
# =========================================================
STEP_CM  = 0.01  # 每次平移步进 1 cm
STEP_DEG = 2.0   # 每次旋转步进 2 度
pos_labels = {}       # target position labels
ypr_labels = {}       # target YPR labels
cur_pos_labels = {}   # current position labels
cur_ypr_labels = {}   # current YPR labels

window = ui.Window("Cup Grasp Teleop", width=560, height=480)
with window.frame:
    with ui.VStack(spacing=4):
        ui.Label("EE Control (base_link frame)", style={"font_size": 14})
        ui.Spacer(height=4)

        # --- Header row ---
        with ui.HStack(height=20, spacing=4):
            ui.Label("", width=80)
            ui.Label("", width=50)
            ui.Label("Target", width=65, style={"font_size": 11, "color": 0xFF88FF88})
            ui.Label("", width=50)
            ui.Label("Current", width=65, style={"font_size": 11, "color": 0xFFFFCC66})

        # --- Position: XYZ +/- buttons ---
        for idx, lb in enumerate(["X (fwd)", "Y (left)", "Z (up)"]):
            with ui.HStack(height=28, spacing=4):
                ui.Label(lb, width=80)
                def make_minus(a=idx):
                    def _click():
                        ik_target_pos[a] -= STEP_CM
                    return _click
                def make_plus(a=idx):
                    def _click():
                        ik_target_pos[a] += STEP_CM
                    return _click
                ui.Button("  -  ", width=50, clicked_fn=make_minus())
                vt = ui.Label(f"{ik_target_pos[idx]:.3f}", width=65,
                              style={"color": 0xFF88FF88})
                ui.Button("  +  ", width=50, clicked_fn=make_plus())
                vc = ui.Label("---", width=65,
                              style={"color": 0xFFFFCC66})
                pos_labels[idx] = vt
                cur_pos_labels[idx] = vc

        ui.Spacer(height=6)
        ui.Separator(height=2)
        ui.Spacer(height=4)

        # --- Orientation: YPR +/- 增量按钮 ---
        with ui.HStack(height=20, spacing=4):
            ui.Label("", width=80)
            ui.Label("", width=50)
            ui.Label("Target", width=55, style={"font_size": 11, "color": 0xFF88FF88})
            ui.Label("", width=50)
            ui.Label("Current", width=55, style={"font_size": 11, "color": 0xFFFFCC66})
        for idx, lb in enumerate(["Yaw", "Pitch", "Roll"]):
            with ui.HStack(height=28, spacing=4):
                ui.Label(lb, width=80)
                def make_ypr_minus(a=idx):
                    def _click():
                        ik_target_ypr[a] -= STEP_DEG
                    return _click
                def make_ypr_plus(a=idx):
                    def _click():
                        ik_target_ypr[a] += STEP_DEG
                    return _click
                ui.Button("  -  ", width=50, clicked_fn=make_ypr_minus())
                vl = ui.Label(f"{ik_target_ypr[idx]:.1f}°", width=55,
                              style={"color": 0xFF88FF88})
                ui.Button("  +  ", width=50, clicked_fn=make_ypr_plus())
                vcl = ui.Label("---", width=55,
                               style={"color": 0xFFFFCC66})
                ypr_labels[idx] = vl
                cur_ypr_labels[idx] = vcl

        ui.Spacer(height=6)
        ui.Separator(height=2)
        ui.Spacer(height=4)

        # --- Gripper slider ---
        with ui.HStack(height=26, spacing=4):
            ui.Label("Gripper", width=80)
            sg = ui.FloatSlider(min=-90, max=0)
            sg.model.set_value(np.degrees(get_pos("left_jointGripper")))
            vg = ui.Label(f"{np.degrees(get_pos('left_jointGripper')):.1f}°", width=55)
            def _on_gripper(m, v=vg):
                d = m.as_float; v.text = f"{d:.1f}°"
                set_pos("left_jointGripper", np.radians(d))
            sg.model.add_value_changed_fn(_on_gripper)

        ui.Spacer(height=6)
        ui.Label("[SPACE] remove support & save",
                 style={"font_size": 12, "color": 0xFFAAAAFF})

print("Panel ready. base_link frame control, [Space] to verify.\n")

# =========================================================
#  Step 8 - Recording + keyboard
# =========================================================
record_times = []
record_q_arm = []
record_q_gripper = []
recording = True
remove_support_time = None


def read_cup_pose() -> tuple[np.ndarray, np.ndarray]:
    rb = dc.get_rigid_body("/World/CarryObject")
    if rb != _dynamic_control.INVALID_HANDLE:
        pose = dc.get_rigid_body_pose(rb)
        return (
            np.array([pose.p.x, pose.p.y, pose.p.z], dtype=np.float64),
            np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z], dtype=np.float64),
        )
    return (
        np.array([float(cwp[0]), float(cwp[1]), float(cwp[2])], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


cup_initial_pos, cup_initial_quat = read_cup_pose()
robot_initial_q = np.array([get_pos(j) for j in ALL_RECORD_JOINTS], dtype=np.float64)
support_center_world = np.array([float(sc[0]), float(sc[1]), float(sc[2])], dtype=np.float64)
support_scale = np.array(SS, dtype=np.float64)

def on_kb(ev, *a, **k):
    global recording, remove_support_time
    if ev.type==carb.input.KeyboardEventType.KEY_PRESS and \
       ev.input==carb.input.KeyboardInput.SPACE:
        print("\n[Space] Removing support...")
        support_prim.GetAttribute("visibility").Set("invisible")
        ca = UsdPhysics.CollisionAPI(support_prim)
        if ca: ca.GetCollisionEnabledAttr().Set(False)
        if record_times and remove_support_time is None:
            remove_support_time = float(record_times[-1])
        recording = False

inp = carb.input.acquire_input_interface()
aw = omni.appwindow.get_default_app_window(); kb = aw.get_keyboard()
ks = inp.subscribe_to_keyboard_events(kb, on_kb)

# =========================================================
#  Main loop - Pinocchio Jacobian IK (base_link frame)
# =========================================================
GAIN = 1.0
DAMP = 1e-4
MAX_DQ = 0.1    # max joint step per iteration (rad)
IK_ITERS = 10   # IK iterations per frame
frame_count = 0
try:
    while simulation_app.is_running():
        simulation_app.update()

        # Read current joint angles from simulation
        q_cur = np.array([get_pos(jn) for jn in IK_JOINTS], dtype=np.float64)

        # Compute current EE pose from both kinematic model and USD ground truth
        cur_model_pos, cur_model_rot = pin_fk(q_cur)
        cur_ee_pos, cur_ee_rot = get_ee_in_baselink()
        cur_model_in_sim_pos, cur_model_in_sim_rot = model_pose_to_sim_frame(
            cur_model_pos, cur_model_rot, ee_align_pos, ee_align_rot)
        cur_ypr_vals = ypr_from_rot(cur_ee_rot)

        # Update target labels
        for ax in range(3):
            if ax in pos_labels:
                pos_labels[ax].text = f"{ik_target_pos[ax]:.3f}"
            if ax in ypr_labels:
                ypr_labels[ax].text = f"{ik_target_ypr[ax]:.1f}°"

        # Update current pose labels
        for ax in range(3):
            if ax in cur_pos_labels:
                cur_pos_labels[ax].text = f"{cur_ee_pos[ax]:.3f}"
            if ax in cur_ypr_labels:
                cur_ypr_labels[ax].text = f"{cur_ypr_vals[ax]:.1f}°"

        # Build desired rotation from target YPR (in base_link frame)
        R_des = rot_from_ypr(
            np.radians(ik_target_ypr[0]),
            np.radians(ik_target_ypr[1]),
            np.radians(ik_target_ypr[2]))
        target_model_pos, target_model_rot = sim_pose_to_model_frame(
            ik_target_pos, R_des, ee_align_pos, ee_align_rot)

        # --- Multi-iteration IK step ---
        q_ik = q_cur.copy()
        for _it in range(IK_ITERS):
            # FK at current IK iterate
            fk_pos, fk_rot = pin_fk(q_ik)

            # 6D error
            pos_err = target_model_pos - fk_pos
            R_err = target_model_rot @ fk_rot.T
            ori_err = pin.log3(R_err)
            err6 = np.concatenate([pos_err, ori_err])

            if np.linalg.norm(err6) < 1e-4:
                break

            # Jacobian at current iterate
            J = pin_jacobian(q_ik)

            # Damped least-squares
            dq = GAIN * J.T @ np.linalg.solve(
                J @ J.T + DAMP * np.eye(6), err6)

            # Clamp joint step
            dq_norm = np.linalg.norm(dq)
            if dq_norm > MAX_DQ:
                dq *= MAX_DQ / dq_norm

            q_ik = q_ik + dq

        # Apply final joint targets
        for i, jn in enumerate(IK_JOINTS):
            set_pos(jn, q_ik[i])

        if frame_count % 120 == 0:
            model_vs_sim_pos = float(np.linalg.norm(cur_model_in_sim_pos - cur_ee_pos))
            model_vs_sim_rot = rotation_distance_deg(cur_model_in_sim_rot, cur_ee_rot)
            target_vs_sim_pos = float(np.linalg.norm(ik_target_pos - cur_ee_pos))
            target_vs_sim_rot = rotation_distance_deg(R_des, cur_ee_rot)
            print(
                "[EE diag] "
                f"model_vs_sim_pos={model_vs_sim_pos:.4f} m "
                f"model_vs_sim_rot={model_vs_sim_rot:.2f} deg "
                f"target_vs_sim_pos={target_vs_sim_pos:.4f} m "
                f"target_vs_sim_rot={target_vs_sim_rot:.2f} deg"
            )
        frame_count += 1

        # Recording
        if recording:
            record_times.append(len(record_times) * RECORD_DT)
            record_q_arm.append(q_cur.copy())
            record_q_gripper.append(get_pos("left_jointGripper"))
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
                support_scale=support_scale,
                remove_support_time=(
                    float(remove_support_time)
                    if remove_support_time is not None
                    else float(record_times[-1])
                ),
            )
            save_grasp_demo_npz(DEMO_OUTPUT_PATH, demo)
            print(f"Saved {len(record_times)} samples -> {DEMO_OUTPUT_PATH}")
            record_times = []
            record_q_arm = []
            record_q_gripper = []
except KeyboardInterrupt:
    print("\nUser interrupted")

inp.unsubscribe_to_keyboard_events(kb, ks)
try: omni.timeline.get_timeline_interface().stop()
except: pass
simulation_app.close()
print("Teleop exit")
