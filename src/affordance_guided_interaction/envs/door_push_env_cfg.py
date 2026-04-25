"""DirectRLEnv 场景与环境配置 — 门推交互任务。

使用 Isaac Lab 的 ``@configclass`` + ``InteractiveSceneCfg`` 声明式定义场景，
由 Cloner 自动为每个并行环境复制完整场景子树，实现 GPU 批量并行仿真。

场景包含：
    - 双臂移动底盘机器人 (UniDingo Lite Z1)
    - 推门 (minimal_push_door)
    - 左/右杯体（预生成，按 reset-time occupancy 启停）
    - 地面平面 + 照明

基座位姿在每次 episode reset 时通过扇形环采样随机化。
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg  # optional, kept for future visual experiments
from isaaclab.utils import configclass

from .direct_rl_env_window import DirectRLEnvWindow

# ═══════════════════════════════════════════════════════════════════════
# 资产路径解析
# ═══════════════════════════════════════════════════════════════════════

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]  # src/affordance_.../envs → 项目根

_ROBOT_USD = str(_PROJECT_ROOT / "assets/robot/usd/uni_dingo_dual_arm.usd")
_LITE_ROBOT_USD = str(_PROJECT_ROOT / "assets/robot/usd/uni_dingo_lite.usd")
_DOOR_USD = str(_PROJECT_ROOT / "assets/minimal_push_door/solid_push_door.usda")
_ROOM_USD = str(_PROJECT_ROOT / "assets/minimal_push_door/room_shell.usda")
_CUP_USD = str(_PROJECT_ROOT / "assets/grasp_objects/cup/carry_cup.usda")

_CAMERA_WIDTH = 640
_CAMERA_HEIGHT = 480
_CAMERA_FOCAL_LENGTH = 11.2
_CAMERA_HORIZONTAL_APERTURE = 20.955
_CAMERA_VERTICAL_APERTURE = 15.71625

# ═══════════════════════════════════════════════════════════════════════
# 机器人关节与 body 名称常量
# ═══════════════════════════════════════════════════════════════════════

# 双臂各 6 关节（不含 gripper，纯策略控制的关节）
ARM_JOINT_NAMES: list[str] = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6",
]

# gripper 关节（初始化时使用，非策略控制）
GRIPPER_JOINT_NAMES: list[str] = [
    "left_jointGripper",
    "right_jointGripper",
]

# 底盘轮子关节（由 3 维底盘速度命令间接驱动）
WHEEL_JOINT_NAMES: list[str] = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

# 平面底盘关节（世界系 x/y 平移 + z 轴偏航）
PLANAR_BASE_JOINT_NAMES: list[str] = [
    "base_x_joint",
    "base_y_joint",
    "base_yaw_joint",
]

# 云台关节（相机姿态；不进入策略动作空间）
PAN_TILT_JOINT_NAMES: list[str] = [
    "pan_tilt_yaw_joint",
    "pan_tilt_pitch_joint",
]

# 末端执行器 / 基座 body 名称
BASE_LINK_NAME: str = "base_link"
LEFT_EE_LINK_NAME: str = "left_gripperMover"
RIGHT_EE_LINK_NAME: str = "right_gripperMover"

# 平面底盘抽象下，planar root 固定在地面 z=0，机器人本体通过 yaw 关节的
# 固定竖直偏移放到正确视觉高度。该高度满足 wheel joint center z (0.035145 m)
# 减去轮半径 (0.050 m) 后，轮底刚好接触地面：
#     base_link_height = wheel_radius - wheel_axis_z = 0.014855 m
BASE_LINK_SPAWN_HEIGHT: float = 0.014855

DOOR_CENTER_XY: tuple[float, float] = (2.95, 0.00)
BASE_REFERENCE_XY: tuple[float, float] = (3.72, 0.00)
BASE_REFERENCE_YAW: float = math.atan2(
    DOOR_CENTER_XY[1] - BASE_REFERENCE_XY[1],
    DOOR_CENTER_XY[0] - BASE_REFERENCE_XY[0],
)

# 门板 body 名称
DOOR_LEAF_BODY_NAME: str = "DoorLeaf"

# ═══════════════════════════════════════════════════════════════════════
# 持杯初始化预设常量（来自 cup_grasp_initializer.py CupGraspPreset）
# ═══════════════════════════════════════════════════════════════════════

# 杯体相对 base_link 的局部偏移 (x, y, z)
LEFT_CUP_RELATIVE_XYZ: tuple[float, float, float] = (0.29, 0.1111, 0.6814)
RIGHT_CUP_RELATIVE_XYZ: tuple[float, float, float] = (0.29, -0.12306, 0.6814)

# 抓取初始化臂关节角度（度）
LEFT_ARM_GRASP_INIT_DEG: dict[str, float] = {
    "left_joint1": 0.0,
    "left_joint2": 0.0,
    "left_joint3": 0.0,
    "left_joint4": 0.0,
    "left_joint5": 0.0,
    "left_joint6": 90.0,
}
RIGHT_ARM_GRASP_INIT_DEG: dict[str, float] = {
    "right_joint1": 0.0,
    "right_joint2": 0.0,
    "right_joint3": 0.0,
    "right_joint4": 0.0,
    "right_joint5": 0.0,
    "right_joint6": 90.0,
}

# Gripper 角度
GRIPPER_OPEN_DEG: float = -90.0
GRIPPER_CLOSE_DEG: float = -32.0
GRIPPER_FULLY_CLOSED_DEG: float = 0.0

# 持杯初始化物理步进数
POSE_SETTLE_STEPS: int = 240
POST_SPAWN_SETTLE_STEPS: int = 120
GRIPPER_CLOSE_STEPS: int = 120
POST_CLOSE_SETTLE_STEPS: int = 60
POST_REMOVE_SETTLE_STEPS: int = 30

# 托盘尺寸 (用于临时支撑杯体)
TRAY_SIZE_XYZ: tuple[float, float, float] = (0.12, 0.12, 0.008)


# ═══════════════════════════════════════════════════════════════════════
# Reset 随机化事件
# ═══════════════════════════════════════════════════════════════════════


def randomize_reset_occupancy(env, env_ids) -> None:
    """在 reset 时为指定 env 均匀采样 empty/left/right/both occupancy。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    sampled_modes = torch.randint(0, 4, (len(env_ids),), device=env.device)
    left_occupied = (sampled_modes == 1) | (sampled_modes == 3)
    right_occupied = (sampled_modes == 2) | (sampled_modes == 3)
    env.set_occupancy(left_occupied, right_occupied, env_ids=env_ids)


@configclass
class EventCfg:
    """Configuration for reset-time randomization."""

    randomize_occupancy = EventTerm(
        func=randomize_reset_occupancy,
        mode="reset",
    )


# ═══════════════════════════════════════════════════════════════════════
# 场景配置
# ═══════════════════════════════════════════════════════════════════════


@configclass
class DoorPushSceneCfg(InteractiveSceneCfg):
    """门推交互任务的场景配置（轻量化单配置）。

    Cloner 将 ``{ENV_REGEX_NS}`` 替换为 ``/World/envs/env_\\d+``，
    自动为每个并行环境复制整棵子树。

    特征：
        - 保留 ``room`` 字段但默认禁用（``room=None``）
        - 使用精简版机器人 USD
        - 保留双臂、gripper 与 4 个底盘轮子 actuator
        - 机器人自碰撞保持启用
        - 基座位姿在每次 episode reset 时随机化（扇形环采样）
    """

    # ── 房间（保留配置字段，但默认不实例化）────────────────────────────
    room: AssetBaseCfg | None = None

    # ── 地面 + 照明 ──────────────────────────────────────────────────
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.05),
        ),
    )
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # ── 双臂机器人（轻量化移动底盘）──────────────────────────────
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_LITE_ROBOT_USD if Path(_LITE_ROBOT_USD).exists() else _ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 平面底盘根节点固定在每个 env 地面原点，机器人本体高度由资产里的
            # base_yaw_joint 固定竖直偏移提供。
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz identity；朝向由 base_yaw_joint 单独控制
            joint_pos={
                "base_x_joint": BASE_REFERENCE_XY[0],
                "base_y_joint": BASE_REFERENCE_XY[1],
                "base_yaw_joint": BASE_REFERENCE_YAW,
                "left_joint.*": 0.0,
                "right_joint.*": 0.0,
            },
        ),
        actuators={
            # 肩关节 (joint2) 力矩上限 60 N·m，其余臂关节 30 N·m（来自 Z1 URDF）
            # PD 位置控制：stiffness/damping 由 env cfg 的 arm_pd_stiffness/arm_pd_damping 注入
            "shoulder_joints": ImplicitActuatorCfg(
                joint_names_expr=["left_joint2", "right_joint2"],
                effort_limit=60.0,
                velocity_limit=2.175,
                stiffness=1000.0,
                damping=100.0,
            ),
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_joint1", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6",
                    "right_joint1", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6",
                ],
                effort_limit=30.0,
                velocity_limit=2.175,
                stiffness=1000.0,
                damping=100.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_jointGripper", "right_jointGripper"],
                effort_limit=30.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=40.0,
            ),
            "planar_base_joints": ImplicitActuatorCfg(
                joint_names_expr=PLANAR_BASE_JOINT_NAMES,
                effort_limit=1.0e6,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=1.0e5,
            ),
        },
    )

    # ── 推门 ────────────────────────────────────────────────────────
    door: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DOOR_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.93, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=0.0,  # 门铰链无主动力矩
                velocity_limit=100.0,
                stiffness=0.0,
                damping=2.0,  # 标称阻尼，域随机化时覆盖
            ),
        },
    )

    # ── 左杯体（预生成，默认远处；reset 时按 occupancy teleport）──
    cup_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CupLeft",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CUP_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 默认放在远处（不影响仿真），reset 时 teleport
            pos=(100.0, 0.0, 0.0),
        ),
    )

    # ── 右杯体（同上）──────────────────────────────────────────────
    cup_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CupRight",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CUP_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(100.0, 1.0, 0.0),
        ),
    )


    # NOTE: tiled_camera 已从默认场景移除。当前训练使用 door_geometry(6D) 作为
    # 唯一门相关输入，不需要相机传感器。若未来实验需恢复，可在子配置中重新声明
    # TiledCameraCfg 并在 launch_simulation_app(enable_cameras=True) 中启用。


# ═══════════════════════════════════════════════════════════════════════
# 环境超参配置
# ═══════════════════════════════════════════════════════════════════════


@configclass
class DoorPushEnvCfg(DirectRLEnvCfg):
    """门推交互任务的 DirectRLEnv 配置。

    包含场景配置引用和所有任务级超参数。

    使用轻量化场景（移动底盘、`room=None`、自碰撞启用），
    基座位姿在每次 episode reset 时通过扇形环采样随机化。
    """

    # ── 场景 ────────────────────────────────────────────────────────
    ui_window_class_type = DirectRLEnvWindow

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        create_stage_in_memory=True,
        physx=sim_utils.PhysxCfg(
            # 6144 env + dual-arm self-collision + door/cup contacts can exceed
            # Isaac Lab's default rigid patch buffer during GPU narrow phase.
            gpu_max_rigid_patch_count=2**19,
        ),
    )

    scene: DoorPushSceneCfg = DoorPushSceneCfg(
        num_envs=64,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    events: EventCfg = EventCfg()

    # ── 仿真步进 ────────────────────────────────────────────────────
    decimation: int = 2           # 策略频率 = physics_dt / decimation = 60 Hz
    episode_length_s: float = 15.0  # 900 steps × (1/60 s)
    num_rerenders_on_reset: int = 3

    # ── 动作空间：双臂 12 维 + 底盘 3 维速度命令 = 15 ─────────────
    num_actions: int = 15
    action_space: gym.spaces.Box = gym.spaces.Box(low=-1.0, high=1.0, shape=(15,))

    # ── 观测空间 ────────────────────────────────────────────────────
    # Actor obs: proprio(36) + ee(38) + context(2) + stability(2) +
    #            door_geometry(6) + door_frame_corners(12) + base_twist/cmd(6) = 102
    num_observations: int = 102
    observation_space: int = 102

    # Critic obs (privileged): actor_obs(102) + door_pose(7) +
    #     door_joint_pos(1) + door_joint_vel(1) + cup_mass(1) +
    #     door_mass(1) + door_damping(1) + cup_dropped(1) = 115
    num_states: int = 115
    state_space: int = 115

    # ── 任务判定阈值 ────────────────────────────────────────────────
    # 门角度达到 1.2rad 后才允许 episode success（仍需杯体未掉落且 base_link 过门）
    door_angle_target: float = 1.2
    # 杯体脱落距离检测阈值 (m)
    cup_drop_threshold: float = 0.15

    # ── 力矩限幅（来自 Z1 URDF，per-joint）───────────────────────
    # 顺序与 ARM_JOINT_NAMES 一致：left_joint1..6, right_joint1..6
    # joint2（肩关节）为 60 N·m，其余为 30 N·m
    # 注：位置控制模式下，effort_limits 限制 PD 控制器输出的力矩上限
    effort_limits: tuple[float, ...] = (
        30.0, 60.0, 30.0, 30.0, 30.0, 30.0,   # left arm
        30.0, 60.0, 30.0, 30.0, 30.0, 30.0,   # right arm
    )

    # ── 控制器参数 ────────────────────────────────────────────────
    control_action_type: str = "joint_position"
    arm_pd_stiffness: float = 1000.0
    arm_pd_damping: float = 100.0
    position_target_noise_std: float = 0.01
    base_control_backend: str = "planar_joint_velocity"
    training_planar_base_only: bool = False
    emit_wheel_debug_state: bool = True
    base_force_body_name: str = "chassis_link"
    base_max_lin_vel_x: float = 0.6
    base_max_lin_vel_y: float = 0.6
    base_max_ang_vel_z: float = 1.2
    base_lin_accel_gain_xy: tuple[float, float] = (20.0, 20.0)
    base_ang_accel_gain_z: float = 20.0
    base_force_limit_xy: tuple[float, float] = (600.0, 600.0)
    base_torque_limit_z: float = 200.0
    wheel_radius: float = 0.05
    wheel_base_half_length: float = 0.285
    wheel_base_half_width: float = 0.2104
    wheel_velocity_limit: float = 40.0

    # ── 域随机化范围（回合级静态参数）──────────────────────────────
    cup_mass_range: tuple[float, float] = (0.1, 0.8)
    door_mass_range: tuple[float, float] = (5.0, 20.0)
    door_damping_range: tuple[float, float] = (0.5, 5.0)

    # 基座采样几何（门外侧扇形环）
    # Z1 臂有效水平可达 ≈ 0.95m（肩偏移 0.15m + 臂链 0.80m），
    # 半径需留 ≥ 0.10m 安全余量。
    door_center_xy: tuple[float, float] = DOOR_CENTER_XY
    base_reference_xy: tuple[float, float] = BASE_REFERENCE_XY
    base_height: float = BASE_LINK_SPAWN_HEIGHT
    base_radius_range: tuple[float, float] = (0.45, 0.60)
    base_sector_half_angle_deg: float = 20.0
    base_yaw_delta_deg: float = 10.0

    # ── 步级噪声标准差 ──────────────────────────────────────────────
    action_noise_std: float = 0.02   # σ_a
    obs_noise_std: float = 0.01      # σ_o

    # ── 稳定性 proxy ────────────────────────────────────────────────
    acc_history_length: int = 10

    # ── 奖励超参 ────────────────────────────────────────────────────
    # 任务奖励 (§4)
    rew_w_delta: float = 10.0
    rew_alpha: float = 0.3
    rew_k_decay: float = 0.5
    rew_w_open: float = 15.0
    rew_w_approach: float = 0.5
    rew_approach_eps: float = 1.0e-6
    rew_approach_stop_angle: float = 0.70
    rew_w_base_align: float = 0.005
    rew_w_base_forward: float = 25.0
    rew_w_base_centerline: float = 0.03
    rew_base_align_mid_angle_deg: float = 35.0
    rew_base_align_temperature_deg: float = 5.0
    rew_base_near_sigma: float = 0.8
    rew_base_range_tau: float = 0.05
    rew_base_centerline_sigma: float = 0.25
    rew_w_base_cross: float = 50.0
    rew_base_cross_open_gate: float = 1.2

    # 稳定性奖励 (§5)
    rew_w_zero_acc: float = 0.045
    rew_lambda_acc: float = 3.0
    rew_w_zero_ang: float = 0.055
    rew_lambda_ang: float = 1.2
    rew_w_acc: float = 0.015
    rew_w_ang: float = 0.0005
    rew_w_tilt: float = 3.0

    # 安全惩罚 (§6)
    rew_mu: float = 0.9
    rew_beta_vel: float = 0.1
    rew_beta_target: float = 0.1
    rew_target_margin_ratio: float = 0.05
    rew_beta_joint_move: float = 7.0
    rew_beta_cup_door_prox: float = 10000.0
    rew_cup_door_prox_threshold: float = 0.20
    rew_w_drop: float = 25.0
    rew_w_base_zero_speed: float = 0.02
    rew_lambda_base_speed: float = 10.0
    rew_w_base_speed: float = 1.0
    rew_beta_base_cmd: float = 2.0
    rew_beta_base_heading: float = 2.0
    rew_beta_base_corridor: float = 1500.0

    # ── 门几何观测 ────────────────────────────────────────────────────
    door_geometry_dim: int = 6  # center(3) + normal(3)
    door_frame_corner_dim: int = 12  # 4 inner-frame corners in base_link frame
    visual_refresh_interval: int = 4  # deprecated, kept for config compat
