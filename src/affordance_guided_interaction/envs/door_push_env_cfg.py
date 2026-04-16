"""DirectRLEnv 场景与环境配置 — 门推交互任务。

使用 Isaac Lab 的 ``@configclass`` + ``InteractiveSceneCfg`` 声明式定义场景，
由 Cloner 自动为每个并行环境复制完整场景子树，实现 GPU 批量并行仿真。

场景包含：
    - 双臂固定底座机器人 (UniDingo Lite Z1)
    - 推门 (minimal_push_door)
    - 左/右杯体（预生成，按课程 occupancy 启停）
    - 地面平面 + 照明

基座位姿在每次 episode reset 时通过扇形环采样随机化。
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
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

# 云台关节（相机姿态；不进入策略动作空间）
PAN_TILT_JOINT_NAMES: list[str] = [
    "pan_tilt_yaw_joint",
    "pan_tilt_pitch_joint",
]

# 末端执行器 / 基座 body 名称
BASE_LINK_NAME: str = "base_link"
LEFT_EE_LINK_NAME: str = "left_gripperMover"
RIGHT_EE_LINK_NAME: str = "right_gripperMover"

# 门板 body 名称
DOOR_LEAF_BODY_NAME: str = "DoorLeaf"

# ═══════════════════════════════════════════════════════════════════════
# 持杯初始化预设常量（来自 cup_grasp_initializer.py CupGraspPreset）
# ═══════════════════════════════════════════════════════════════════════

# 杯体相对 base_link 的局部偏移 (x, y, z)
LEFT_CUP_RELATIVE_XYZ: tuple[float, float, float] = (0.29, 0.1111, 0.6814)
RIGHT_CUP_RELATIVE_XYZ: tuple[float, float, float] = (0.29, -0.1111, 0.6814)

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
    "right_joint6": -90.0,
}

# Gripper 角度
GRIPPER_OPEN_DEG: float = -90.0
GRIPPER_CLOSE_DEG: float = -34.0
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
# 场景配置
# ═══════════════════════════════════════════════════════════════════════


@configclass
class DoorPushSceneCfg(InteractiveSceneCfg):
    """门推交互任务的场景配置（轻量化单配置）。

    Cloner 将 ``{ENV_REGEX_NS}`` 替换为 ``/World/envs/env_\\d+``，
    自动为每个并行环境复制整棵子树。

    特征：
        - 保留 ``room`` 字段但默认禁用（``room=None``）
        - 使用精简版机器人 USD（无轮子/云台/支架），底座固定
        - 仅保留双臂 + gripper actuator
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

    # ── 双臂机器人（轻量化，固定底座）──────────────────────────────
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
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 标称初始位置：推板正前方，在采样范围中心
            pos=(3.72, 0.27, 0.12),
            rot=(0.0, 0.0, 0.0, 1.0),  # wxyz 180° yaw — 匹配 scene_factory
            joint_pos={
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

    使用轻量化场景（固定底座、无轮子/云台、`room=None`、自碰撞启用），
    基座位姿在每次 episode reset 时通过扇形环采样随机化。
    """

    # ── 场景 ────────────────────────────────────────────────────────
    ui_window_class_type = DirectRLEnvWindow

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        create_stage_in_memory=True,
    )

    scene: DoorPushSceneCfg = DoorPushSceneCfg(
        num_envs=64,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # ── 仿真步进 ────────────────────────────────────────────────────
    decimation: int = 2           # 策略频率 = physics_dt / decimation = 60 Hz
    episode_length_s: float = 15.0  # 900 steps × (1/60 s)
    num_rerenders_on_reset: int = 3

    # ── 动作空间：双臂 6+6 = 12 关节位置目标 (rad) ──────────────
    num_actions: int = 12
    action_space: int = 12

    # ── 观测空间 ────────────────────────────────────────────────────
    # Actor obs: proprio(36) + ee(38) + context(2) + stability(2) +
    #            door_geometry(6) = 84
    num_observations: int = 84
    observation_space: int = 84

    # Critic obs (privileged): actor_obs(84) + door_pose(7) +
    #     door_joint_pos(1) + door_joint_vel(1) + cup_mass(1) +
    #     door_mass(1) + door_damping(1) + cup_dropped(1) = 97
    num_states: int = 97
    state_space: int = 97

    # ── 任务判定阈值 ────────────────────────────────────────────────
    # 门角度到达 target → terminated(success)
    door_angle_target: float = 1.57
    # 一次性成功 bonus 的角度阈值 (< door_angle_target)
    success_angle_threshold: float = 1.2
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
    position_target_noise_std: float = 0.0

    # ── 域随机化范围（回合级静态参数）──────────────────────────────
    cup_mass_range: tuple[float, float] = (0.1, 0.8)
    door_mass_range: tuple[float, float] = (5.0, 20.0)
    door_damping_range: tuple[float, float] = (0.5, 5.0)

    # 基座采样几何（门外侧扇形环）
    # Z1 臂有效水平可达 ≈ 0.95m（肩偏移 0.15m + 臂链 0.80m），
    # 半径需留 ≥ 0.10m 安全余量。
    push_plate_center_xy: tuple[float, float] = (2.98, 0.27)
    base_reference_xy: tuple[float, float] = (3.72, 0.27)
    base_height: float = 0.12
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
    rew_w_open: float = 50.0
    rew_w_approach: float = 200.0
    rew_approach_eps: float = 1.0e-6
    rew_approach_stop_angle: float = 0.10

    # 稳定性奖励 (§5)
    rew_w_zero_acc: float = 1.0
    rew_lambda_acc: float = 2.0
    rew_w_zero_ang: float = 0.5
    rew_lambda_ang: float = 1.0
    rew_w_acc: float = 0.5
    rew_w_ang: float = 0.3
    rew_w_tilt: float = 0.3

    # 安全惩罚 (§6)
    rew_mu: float = 0.9
    rew_beta_vel: float = 0.5
    rew_beta_target: float = 1.0
    rew_w_drop: float = 100.0

    # ── 门几何观测 ────────────────────────────────────────────────────
    door_geometry_dim: int = 6  # center(3) + normal(3)
    visual_refresh_interval: int = 4  # deprecated, kept for config compat
