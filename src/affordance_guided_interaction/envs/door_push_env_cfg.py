"""DirectRLEnv scene and environment config for the handle-free push-door task.

One teacher PPO policy controls dual arms plus Clearpath Dingo base to push
open a handle-free door and traverse the doorway.

Architecture: one Isaac Lab DirectRLEnv, one 79D teacher observation, one 15D
raw Gaussian action, one scalar reward, and one symmetric actor-critic training
configuration. Raw policy actions are NOT clipped to [-1, 1]; only mapped
control commands are safety-limited.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .direct_rl_env_window import DirectRLEnvWindow

# ═══════════════════════════════════════════════════════════════════════
# Asset paths
# ═══════════════════════════════════════════════════════════════════════

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]

_ROBOT_USD = str(_PROJECT_ROOT / "assets/robot/usd/uni_dingo_dual_arm.usd")
_LITE_ROBOT_USD = str(_PROJECT_ROOT / "assets/robot/usd/uni_dingo_lite.usd")
_DOOR_USD = str(_PROJECT_ROOT / "assets/minimal_push_door/solid_push_door.usda")
_DOOR_SIDE_WALLS_USD = str(_PROJECT_ROOT / "assets/minimal_push_door/door_side_walls.usda")

# ═══════════════════════════════════════════════════════════════════════
# Robot joint and body name constants
# ═══════════════════════════════════════════════════════════════════════

ARM_JOINT_NAMES: list[str] = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6",
]

GRIPPER_JOINT_NAMES: list[str] = [
    "left_jointGripper",
    "right_jointGripper",
]

WHEEL_JOINT_NAMES: list[str] = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

PLANAR_BASE_JOINT_NAMES: list[str] = [
    "base_x_joint",
    "base_y_joint",
    "base_yaw_joint",
]

BASE_LINK_NAME: str = "base_link"
LEFT_EE_LINK_NAME: str = "left_gripperMover"
RIGHT_EE_LINK_NAME: str = "right_gripperMover"
LEFT_ARM_JOINT1_ANCHOR_LINK_NAME: str = "left_link00"
RIGHT_ARM_JOINT1_ANCHOR_LINK_NAME: str = "right_link00"

BASE_LINK_SPAWN_HEIGHT: float = 0.014855

DOOR_CENTER_XY: tuple[float, float] = (2.95, 0.00)
DOOR_CROSS_DIR_XY: tuple[float, float] = (-1.0, 0.0)
DOOR_LATERAL_DIR_XY: tuple[float, float] = (0.0, 1.0)

DOOR_LEAF_BODY_NAME: str = "DoorLeaf"

# Gripper constants (non-policy-controlled)
GRIPPER_CLOSED_DEG: float = -32.0


# ═══════════════════════════════════════════════════════════════════════
# Scene configuration
# ═══════════════════════════════════════════════════════════════════════


@configclass
class DoorPushSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the handle-free push-door task."""

    room: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DoorSideWalls",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DOOR_SIDE_WALLS_USD,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

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

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_LITE_ROBOT_USD if Path(_LITE_ROBOT_USD).exists() else _ROBOT_USD,
            activate_contact_sensors=True,
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
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "base_x_joint": DOOR_CENTER_XY[0] - 1.5,
                "base_y_joint": DOOR_CENTER_XY[1],
                "base_yaw_joint": 0.0,
                "left_joint.*": 0.0,
                "right_joint.*": 0.0,
            },
        ),
        actuators={
            "shoulder_joints": ImplicitActuatorCfg(
                joint_names_expr=["left_joint2", "right_joint2"],
                effort_limit=60.0,
                velocity_limit=2.175,
                stiffness=50.0,
                damping=4.5,
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
                stiffness=50.0,
                damping=4.5,
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
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=2.0,
            ),
        },
    )

    hard_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(chassis_link|left_link.*|right_link.*)",
    )


# ═══════════════════════════════════════════════════════════════════════
# Environment hyperparameter config
# ═══════════════════════════════════════════════════════════════════════


@configclass
class DoorPushEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv config for the handle-free push-door task.

    One teacher policy, 79D symmetric actor-critic observation, 15D raw
    Gaussian action. No cup, no occupancy, no student, no handle.
    """

    ui_window_class_type = DirectRLEnvWindow

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        create_stage_in_memory=True,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=2**19,
        ),
    )

    scene: DoorPushSceneCfg = DoorPushSceneCfg(
        num_envs=64,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Simulation stepping
    decimation: int = 2
    episode_length_s: float = 10.0  # 600 steps at 60 Hz
    num_rerenders_on_reset: int = 3

    # Action space: 12 arm raw actions + 3 base raw actions = 15
    num_actions: int = 15
    action_space: gym.spaces.Box = gym.spaces.Box(
        low=-1.0e6, high=1.0e6, shape=(15,)
    )

    # Observation: 79D symmetric actor-critic
    num_observations: int = 79
    observation_space: int = 79
    num_states: int = 79
    state_space: int = 79

    # Task thresholds
    door_angle_target: float = math.pi / 6.0  # opened_enough at 30 deg
    theta_open: float = math.pi / 6.0
    theta_pass: float = 70.0 * math.pi / 180.0
    theta_hat: float = 75.0 * math.pi / 180.0
    reverse_angle_limit: float = -0.05

    # Torque limits (from Z1 URDF)
    effort_limits: tuple[float, ...] = (
        30.0, 60.0, 30.0, 30.0, 30.0, 30.0,
        30.0, 60.0, 30.0, 30.0, 30.0, 30.0,
    )

    # Control parameters
    control_action_type: str = "joint_position"
    arm_action_scale_rad: float = 0.25
    torque_proxy_sigma: float = 0.7
    arm_default_joint_pos: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0,
        0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0,
    )
    arm_pd_stiffness: float = 50.0
    arm_pd_damping: float = 4.5
    base_control_backend: str = "planar_joint_velocity"
    training_planar_base_only: bool = False
    emit_wheel_debug_state: bool = True
    base_force_body_name: str = "chassis_link"
    base_max_lin_vel_x: float = 0.5
    base_max_lin_vel_y: float = 0.5
    base_max_ang_vel_z: float = 1.0
    base_command_deadband: float = 0.1
    base_lin_accel_gain_xy: tuple[float, float] = (20.0, 20.0)
    base_ang_accel_gain_z: float = 20.0
    base_force_limit_xy: tuple[float, float] = (600.0, 600.0)
    base_torque_limit_z: float = 200.0
    wheel_radius: float = 0.05
    wheel_base_half_length: float = 0.285
    wheel_base_half_width: float = 0.2104
    wheel_velocity_limit: float = 40.0

    # Reset distribution (paper-aligned)
    door_center_xy: tuple[float, float] = DOOR_CENTER_XY
    door_cross_dir_xy: tuple[float, float] = DOOR_CROSS_DIR_XY
    door_lateral_dir_xy: tuple[float, float] = DOOR_LATERAL_DIR_XY
    base_height: float = BASE_LINK_SPAWN_HEIGHT
    reset_distance_to_wall_range: tuple[float, float] = (1.0, 2.0)
    reset_lateral_offset_range: tuple[float, float] = (-2.0, 2.0)
    reset_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    reset_base_lin_vel_xy_range: tuple[float, float] = (-0.5, 0.5)
    reset_base_ang_vel_z: float = 0.0

    # Door randomization
    door_mass_range: tuple[float, float] = (15.0, 75.0)
    door_hinge_resistance_range: tuple[float, float] = (0.0, 30.0)
    door_hinge_resistance_zero_prob: float = 0.2
    door_hinge_air_damping_range: tuple[float, float] = (0.0, 4.0)
    door_closer_damping_alpha_range: tuple[float, float] = (1.5, 3.0)
    door_hinge_damping_zero_prob: float = 0.4

    # Reward parameters
    rew_opening_scale: float = 3.0
    rew_passing_max_speed: float = 0.5
    rew_ma_weight: float = 0.3
    rew_psa_weight: float = 1.0
    rew_eep_weight: float = 1.0
    rew_pcl_weight: float = 0.1
    rew_pc_weight: float = 2.0
    rew_stretched_arm_threshold: float = 0.5
    rew_stretched_arm_scale: float = 0.1
    rew_cmd_limit_threshold: float = 1.0
    hard_collision_force_threshold: float = 1.0
