"""DirectRLEnv 场景与环境配置 — 门推交互任务。

使用 Isaac Lab 的 ``@configclass`` + ``InteractiveSceneCfg`` 声明式定义场景，
由 Cloner 自动为每个并行环境复制完整场景子树，实现 GPU 批量并行仿真。

场景包含：
    - 双臂移动机器人 (UniDingo Dual-Arm Z1)
    - 推门 (minimal_push_door)
    - 左/右杯体（预生成，按课程 occupancy 启停）
    - 地面平面 + 照明
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
    """门推交互任务的完整场景配置。

    Cloner 将 ``{ENV_REGEX_NS}`` 替换为 ``/World/envs/env_\\d+``，
    自动为每个并行环境复制整棵子树。
    """

    # ── 房间（墙壁 + 地板）────────────────────────────────────────────
    room: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ROOM_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

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

    # ── 双臂移动机器人 ──────────────────────────────────────────────
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
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
                "pan_tilt_.*": 0.0,
                ".*wheel": 0.0,
            },
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["left_joint.*", "right_joint.*"],
                effort_limit=33.5,
                velocity_limit=2.175,
                stiffness=0.0,
                damping=0.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel"],
                effort_limit=20.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "pan_tilt": ImplicitActuatorCfg(
                joint_names_expr=PAN_TILT_JOINT_NAMES,
                effort_limit=5.0,
                velocity_limit=2.0,
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
    """

    # ── 场景 ────────────────────────────────────────────────────────
    ui_window_class_type = DirectRLEnvWindow

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        create_stage_in_memory=True,
    )

    scene: DoorPushSceneCfg = DoorPushSceneCfg(
        num_envs=64,
        env_spacing=10.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # ── 仿真步进 ────────────────────────────────────────────────────
    decimation: int = 2           # 策略频率 = physics_dt / decimation = 60 Hz
    episode_length_s: float = 90.0  # 5400 steps × (1/60 s)
    num_rerenders_on_reset: int = 3

    # ── 动作空间：双臂 6+6 = 12 关节力矩 ──────────────────────────
    num_actions: int = 12
    action_space: int = 12

    # ── 观测空间 ────────────────────────────────────────────────────
    # Actor obs: proprio(48) + ee(38) + context(2) + stability(2) +
    #            door_geometry(6) = 96
    num_observations: int = 96
    observation_space: int = 96

    # Critic obs (privileged): actor_obs(96) + door_pose(7) +
    #     door_joint_pos(1) + door_joint_vel(1) + cup_mass(1) +
    #     door_mass(1) + door_damping(1) + cup_dropped(1) = 109
    num_states: int = 109
    state_space: int = 109

    # ── 任务判定阈值 ────────────────────────────────────────────────
    # 门角度到达 target → terminated(success)
    door_angle_target: float = 1.57
    # 一次性成功 bonus 的角度阈值 (< door_angle_target)
    success_angle_threshold: float = 1.2
    # 杯体脱落距离检测阈值 (m)
    cup_drop_threshold: float = 0.15

    # ── 力矩限幅 ────────────────────────────────────────────────────
    effort_limit: float = 33.5

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

    # 稳定性奖励 (§5)
    rew_w_zero_acc: float = 1.0
    rew_lambda_acc: float = 2.0
    rew_w_zero_ang: float = 0.5
    rew_lambda_ang: float = 1.0
    rew_w_acc: float = 0.5
    rew_w_ang: float = 0.3
    rew_w_tilt: float = 0.3
    rew_w_smooth: float = 0.1
    rew_w_reg: float = 0.01

    # 安全惩罚 (§6)
    rew_beta_limit: float = 1.0
    rew_mu: float = 0.9
    rew_beta_vel: float = 0.5
    rew_beta_torque: float = 0.01
    rew_w_drop: float = 100.0

    # ── 门几何观测 ────────────────────────────────────────────────────
    door_geometry_dim: int = 6  # center(3) + normal(3)
    visual_refresh_interval: int = 4  # deprecated, kept for config compat


@configclass
class DoorPushLiteSceneCfg(DoorPushSceneCfg):
    """门-杯-双臂训练轻量版场景。

    Phase 2 轻量化：
    - 使用精简版机器人 USD（uni_dingo_lite.usd，无轮子/云台/相机/支架）
    - 删除 room
    - 固定机器人 root
    - 仅保留双臂 actuator
    - 维持现有 DoorPushEnv 逻辑兼容

    注意：uni_dingo_lite.usd 需要先通过
    ``python assets/robot/scripts/convert_lite_urdf_to_usd.py``
    从 uni_dingo_lite.urdf 转换生成。
    如果 lite USD 尚未生成，会回退到完整机器人 USD。
    """

    def __post_init__(self):
        super().__post_init__()
        self.room = None
        # 切换到轻量版机器人 USD（如文件存在）
        if Path(_LITE_ROBOT_USD).exists():
            self.robot.spawn.usd_path = _LITE_ROBOT_USD
        self.robot.spawn.articulation_props.fix_root_link = True
        self.robot.actuators.pop("wheels", None)
        self.robot.actuators.pop("pan_tilt", None)
        self.robot.init_state.joint_pos.pop(".*wheel", None)
        self.robot.init_state.joint_pos.pop("pan_tilt_.*", None)


@configclass
class DoorPushLiteEnvCfg(DoorPushEnvCfg):
    """固定底座双臂持杯推门训练配置。"""

    scene: DoorPushLiteSceneCfg = DoorPushLiteSceneCfg(
        num_envs=64,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    def __post_init__(self):
        super().__post_init__()
        # 固定在当前标称门前位姿，避免把底盘采样当作训练内容。
        self.base_radius_range = (0.74, 0.74)
        self.base_sector_half_angle_deg = 0.0
        self.base_yaw_delta_deg = 0.0
