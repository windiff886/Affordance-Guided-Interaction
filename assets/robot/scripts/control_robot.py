"""
Uni-Dingo 双臂机器人 Isaac Sim 控制脚本

在 Isaac Sim 的 Script Editor (Window > Script Editor) 中运行本脚本。

前提条件:
  1. 已导入 URDF 并在 Stage 中看到机器人
  2. 已按下 PLAY 按钮（仿真必须处于运行状态）

功能:
  - 列出所有关节及其当前状态
  - 通过位置/速度命令控制各关节
  - 提供预设动作：抬臂、挥手、底盘前进等
"""

import numpy as np
import asyncio
from pxr import UsdPhysics, Sdf, Gf

# =========================================================
#  配置 - 根据你的 Stage 修改
# =========================================================

# URDF 导入后机器人的 prim 路径
# 默认 URDF 导入器会在 /World/ 下以 robot name 创建 prim
# 如果你不确定路径, 运行下方的 find_robot() 函数
ROBOT_PRIM_PATH = "/uni_dingo_dual_arm"

# =========================================================
#  关节名称映射
# =========================================================

# 左臂关节 (6DOF + 夹爪)
LEFT_ARM_JOINTS = [
    "left_joint1",   # 肩部旋转
    "left_joint2",   # 肩部俯仰
    "left_joint3",   # 肘部
    "left_joint4",   # 腕部旋转
    "left_joint5",   # 腕部俯仰
    "left_joint6",   # 腕部滚转
]
LEFT_GRIPPER = "left_jointGripper"

# 右臂关节 (6DOF + 夹爪)
RIGHT_ARM_JOINTS = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
]
RIGHT_GRIPPER = "right_jointGripper"

# 底盘轮子 (4个全向轮)
WHEEL_JOINTS = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

# 云台关节
PAN_TILT_JOINTS = [
    "pan_tilt_yaw_joint",    # 偏航
    "pan_tilt_pitch_joint",  # 俯仰
]


# =========================================================
#  工具函数
# =========================================================

def find_robot():
    """搜索 Stage 中所有 Articulation Root, 帮助你找到机器人的 prim 路径"""
    stage = omni.usd.get_context().get_stage()
    print("=" * 60)
    print("  搜索 Stage 中的 Articulation Roots")
    print("=" * 60)
    count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            count += 1
            print(f"  📦 {prim.GetPath()}")
    if count == 0:
        print("  ❌ 未找到 Articulation Root!")
        print("  请确认:")
        print("    1. 已通过 File > Import 导入了 URDF")
        print("    2. 导入时没有勾选 'Merge Fixed Joints'")
    print(f"\n总计: {count} 个 Articulation Root")
    print("=" * 60)


def get_dc():
    """获取 Dynamic Control 接口"""
    from omni.isaac.dynamic_control import _dynamic_control
    return _dynamic_control.acquire_dynamic_control_interface()


def get_articulation():
    """获取机器人 Articulation 句柄"""
    dc = get_dc()
    art = dc.get_articulation(ROBOT_PRIM_PATH)
    if art == 0:
        # 尝试常见的路径变体
        for path in [
            ROBOT_PRIM_PATH,
            "/World" + ROBOT_PRIM_PATH,
            "/World/uni_dingo_dual_arm",
            "/uni_dingo_dual_arm",
        ]:
            art = dc.get_articulation(path)
            if art != 0:
                print(f"✅ 机器人路径: {path}")
                break
    if art == 0:
        print("❌ 无法找到机器人 Articulation!")
        print("   请运行 find_robot() 获取正确路径")
        print("   然后修改 ROBOT_PRIM_PATH 变量")
        return None, None
    return dc, art


def list_joints():
    """列出所有关节及其当前状态"""
    dc, art = get_articulation()
    if art is None:
        return

    num_dofs = dc.get_articulation_dof_count(art)
    print("=" * 70)
    print(f"  机器人关节列表 (共 {num_dofs} 个自由度)")
    print("=" * 70)
    print(f"{'序号':>4} {'关节名称':<30} {'类型':<12} {'位置 (rad)':>12} {'速度':>10}")
    print("-" * 70)

    for i in range(num_dofs):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        dof_type = dc.get_dof_type(dof)
        pos = dc.get_dof_position(dof)
        vel = dc.get_dof_velocity(dof)

        type_str = {0: "rotation", 1: "translation"}.get(dof_type, "unknown")
        print(f"{i:>4} {name:<30} {type_str:<12} {pos:>12.4f} {vel:>10.4f}")

    print("=" * 70)


def set_joint_position(joint_name, target_rad):
    """设置单个关节的目标位置 (弧度)"""
    dc, art = get_articulation()
    if art is None:
        return
    dof = dc.find_articulation_dof(art, joint_name)
    if dof == 0:
        print(f"❌ 关节 '{joint_name}' 未找到")
        return
    dc.set_dof_position_target(dof, target_rad)
    print(f"➡️  {joint_name} → {target_rad:.3f} rad ({np.degrees(target_rad):.1f}°)")


def set_joint_velocity(joint_name, target_vel):
    """设置单个关节的目标速度 (rad/s)"""
    dc, art = get_articulation()
    if art is None:
        return
    dof = dc.find_articulation_dof(art, joint_name)
    if dof == 0:
        print(f"❌ 关节 '{joint_name}' 未找到")
        return
    dc.set_dof_velocity_target(dof, target_vel)
    print(f"➡️  {joint_name} 速度 → {target_vel:.3f} rad/s")


def set_arm_positions(side, positions_deg):
    """
    设置一只臂的6个关节位置 (角度制)

    Args:
        side: "left" 或 "right"
        positions_deg: 6个关节角度的列表 (单位: 度)
    """
    joints = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
    assert len(positions_deg) == 6, "需要提供6个关节角度"
    for j, deg in zip(joints, positions_deg):
        set_joint_position(j, np.radians(deg))


def set_gripper(side, open_ratio=1.0):
    """
    控制夹爪

    Args:
        side: "left" 或 "right"
        open_ratio: 0.0 (关闭) 到 1.0 (完全打开)
    """
    gripper = LEFT_GRIPPER if side == "left" else RIGHT_GRIPPER
    # Z1 夹爪范围大约 0 ~ 0.04 rad
    target = open_ratio * 0.04
    set_joint_position(gripper, target)


def set_pan_tilt(yaw_deg=0, pitch_deg=0):
    """
    设置云台角度

    Args:
        yaw_deg: 偏航角度 (度), ±60
        pitch_deg: 俯仰角度 (度), ±60
    """
    set_joint_position("pan_tilt_yaw_joint", np.radians(yaw_deg))
    set_joint_position("pan_tilt_pitch_joint", np.radians(pitch_deg))


def drive_base(linear_vel=0.0, angular_vel=0.0):
    """
    控制底盘运动 (简化双轮差速模型近似)

    Args:
        linear_vel: 前进速度 (rad/s on wheels)
        angular_vel: 旋转速度 (rad/s, 正=逆时针)
    """
    # 对于全向底盘, 简化为差速驱动
    left_vel = linear_vel - angular_vel
    right_vel = linear_vel + angular_vel

    set_joint_velocity("front_left_wheel", left_vel)
    set_joint_velocity("rear_left_wheel", left_vel)
    set_joint_velocity("front_right_wheel", right_vel)
    set_joint_velocity("rear_right_wheel", right_vel)


# =========================================================
#  预设动作
# =========================================================

def home_position():
    """所有关节回到零位"""
    print("\n🏠 回到零位")
    set_arm_positions("left",  [0, 0, 0, 0, 0, 0])
    set_arm_positions("right", [0, 0, 0, 0, 0, 0])
    set_gripper("left", 0.5)
    set_gripper("right", 0.5)
    set_pan_tilt(0, 0)
    drive_base(0, 0)


def raise_arms():
    """双臂抬起"""
    print("\n🙌 双臂抬起")
    set_arm_positions("left",  [0, 45, -90, 0, 45, 0])
    set_arm_positions("right", [0, 45, -90, 0, 45, 0])


def ready_pose():
    """就绪姿态 (双臂前伸)"""
    print("\n🤖 就绪姿态")
    set_arm_positions("left",  [0, 30, -60, 0, 30, 0])
    set_arm_positions("right", [0, 30, -60, 0, 30, 0])
    set_gripper("left", 1.0)
    set_gripper("right", 1.0)


def look_around():
    """云台环顾"""
    print("\n👀 云台看左")
    set_pan_tilt(yaw_deg=30, pitch_deg=-15)


def forward():
    """底盘前进"""
    print("\n🚗 前进")
    drive_base(linear_vel=5.0, angular_vel=0.0)


def stop():
    """底盘停止"""
    print("\n🛑 停止")
    drive_base(0, 0)


def turn_left():
    """底盘左转"""
    print("\n↰ 左转")
    drive_base(linear_vel=0.0, angular_vel=2.0)


# =========================================================
#  设置关节驱动属性 (首次导入后需要运行一次)
# =========================================================

def setup_joint_drives(stiffness=1000.0, damping=100.0):
    """
    为所有 revolute 关节设置 Angular Drive 属性
    URDF 导入后关节默认可能没有驱动, 需要手动添加

    Args:
        stiffness: 位置控制刚度 (越大越硬)
        damping: 阻尼 (越大越稳)
    """
    stage = omni.usd.get_context().get_stage()
    print("=" * 60)
    print("  设置关节驱动属性")
    print("=" * 60)

    all_joints = LEFT_ARM_JOINTS + [LEFT_GRIPPER] + \
                 RIGHT_ARM_JOINTS + [RIGHT_GRIPPER] + \
                 PAN_TILT_JOINTS

    count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint):
            prim_name = prim.GetName()
            # 检查是否是我们的关节
            is_target = any(j in prim_name for j in all_joints + WHEEL_JOINTS)

            if is_target:
                # 添加 DriveAPI
                if "wheel" in prim_name.lower():
                    # 轮子用速度控制
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.CreateTypeAttr("velocity")
                    drive.CreateDampingAttr(500.0)
                    drive.CreateStiffnessAttr(0.0)
                    print(f"  🔧 {prim_name}: velocity drive (damping={500.0})")
                else:
                    # 臂/云台用位置控制
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.CreateTypeAttr("force")
                    drive.CreateStiffnessAttr(stiffness)
                    drive.CreateDampingAttr(damping)
                    print(f"  🔧 {prim_name}: position drive (K={stiffness}, D={damping})")
                count += 1

    print(f"\n✅ 已配置 {count} 个关节驱动")
    if count == 0:
        print("⚠️  未找到匹配的关节! 请检查 Stage 中的 prim 路径")
        print("   提示: 可以在 Stage 树中展开机器人查看关节名称")
    print("=" * 60)


# =========================================================
#  主程序 - 在 Script Editor 中运行
# =========================================================

print("""
╔══════════════════════════════════════════════════════╗
║      Uni-Dingo 双臂机器人控制器 (Isaac Sim)         ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  首次使用:                                           ║
║    1. find_robot()       查找机器人路径              ║
║    2. setup_joint_drives() 配置关节驱动              ║
║    3. 按 PLAY 按钮                                   ║
║    4. list_joints()      查看所有关节状态            ║
║                                                      ║
║  控制命令:                                           ║
║    home_position()    所有关节回零位                  ║
║    raise_arms()       双臂抬起                       ║
║    ready_pose()       就绪姿态                       ║
║    look_around()      云台环顾                       ║
║    forward()          底盘前进                       ║
║    stop()             底盘停止                       ║
║    turn_left()        底盘左转                       ║
║                                                      ║
║  精确控制:                                           ║
║    set_arm_positions("left", [j1,j2,j3,j4,j5,j6])  ║
║    set_gripper("right", 1.0)  # 0=关, 1=开          ║
║    set_pan_tilt(yaw_deg=30, pitch_deg=-15)           ║
║    set_joint_position("left_joint1", 0.5)  # rad     ║
║    drive_base(linear_vel=5.0, angular_vel=0.0)       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

# 自动搜索机器人
find_robot()
