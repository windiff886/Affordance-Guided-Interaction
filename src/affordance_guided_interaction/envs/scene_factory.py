"""场景装配工厂 — 封装所有 Isaac Lab 资产加载与物理属性修改。

本模块是 envs 层中**唯一直接调用 Isaac Lab API** 的位置。
在无 Isaac Lab 环境下以 fallback 模式运行（返回占位句柄）。

职责：
    1. 加载机器人模型（双臂 Z1 + Dingo 底座）
    2. 根据课程阶段决定门类型（当前仅支持 push）
    3. 根据 occupied 标记决定是否生成杯体
    4. 应用域随机化参数（质量、阻尼、基座位置）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Isaac Lab 条件导入
# ═══════════════════════════════════════════════════════════════════════
# 当 Isaac Lab 不可用时（如纯 CPU 开发/测试），使用占位模式

_HAS_ISAAC_LAB = False

try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from omni.isaac.lab.sim import SimulationContext
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    _HAS_ISAAC_LAB = True
except ImportError:
    try:
        # Isaac Lab 2024+ 使用 isaaclab 命名空间
        import isaaclab.sim as sim_utils
        from isaaclab.assets import (
            Articulation,
            ArticulationCfg,
            RigidObject,
            RigidObjectCfg,
        )
        from isaaclab.sim import SimulationContext
        from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
        _HAS_ISAAC_LAB = True
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# 项目根目录解析 + 资产路径
# ═══════════════════════════════════════════════════════════════════════

# 从当前文件定位项目根目录
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]  # src/affordance_.../envs → 项目根

# 门资产 USD 路径映射（当前仅支持 push 门）
_DOOR_ASSET_MAP: dict[str, str] = {
    "push": str(_PROJECT_ROOT / "assets/minimal_push_door/minimal_push_door.usda"),
}

# 机器人 USD 路径
_ROBOT_ASSET_PATH = str(_PROJECT_ROOT / "assets/robot/uni_dingo_dual_arm.usd")

# 杯体 USD 路径（来自 grasp_objects 目录）
_CUP_ASSET_PATH = str(_PROJECT_ROOT / "assets/grasp_objects")


class _PlaceholderView:
    """Isaac Lab 视图的占位类型，用于无 Isaac Lab 环境下的开发。"""
    pass


@dataclass
class SceneHandles:
    """场景中各物体的仿真句柄集合。

    将所有 Isaac Lab 视图引用集中管理，避免在环境代码中散落。
    """

    # 机器人 Articulation 句柄
    robot_view: Any = None
    # 左臂 gripper link 名称（用于 body_physx_view 索引）
    left_ee_view: Any = None
    # 右臂 gripper link 名称
    right_ee_view: Any = None
    # 门的 Articulation 句柄（含铰链关节）
    door_view: Any = None
    # 门板 RigidObject 句柄（用于读取位姿）
    door_panel_view: Any = None
    # 杯体 RigidObject 句柄（None 表示本局无杯体）
    cup_view: Any = None

    # 当前场景元信息
    door_type: str = "push"
    left_occupied: bool = False
    right_occupied: bool = False

    # 机器人关节名称映射（在初始化后填充）
    arm_joint_names: list[str] = field(default_factory=list)
    arm_joint_indices: Any = None  # Tensor 或 None

    # 末端执行器 body 索引
    left_ee_body_idx: int = -1
    right_ee_body_idx: int = -1

    # 门铰链关节索引
    door_hinge_joint_idx: Any = None

    # Isaac Lab SimulationContext 引用
    sim_context: Any = None


class SceneFactory:
    """Isaac Lab 场景装配工厂。

    每次 episode reset 时调用 ``build()`` 重建场景。
    在 Isaac Lab 可达时使用真实物理引擎；否则返回占位句柄。

    Parameters
    ----------
    physics_dt : float
        物理仿真步长。
    sim_context : Any | None
        Isaac Lab SimulationContext 引用。如果为 None 则在 build() 时尝试获取。
    """

    def __init__(
        self,
        physics_dt: float = 1.0 / 120.0,
        sim_context: Any = None,
    ) -> None:
        self._physics_dt = physics_dt
        self._sim_context = sim_context
        self._handles: SceneHandles | None = None
        self._scene_built: bool = False

        # 末端执行器 link 名称（与 URDF 中定义一致）
        self._left_ee_link = "left_gripper_link"
        self._right_ee_link = "right_gripper_link"

        # 机械臂关节名称（双臂各 6 个关节）
        self._arm_joint_names = [
            "left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6",
            "right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6",
        ]

    @property
    def handles(self) -> SceneHandles:
        """当前场景句柄。必须在 build() 之后访问。"""
        assert self._handles is not None, "必须先调用 build() 装配场景"
        return self._handles

    # ═══════════════════════════════════════════════════════════════════
    # 场景装配
    # ═══════════════════════════════════════════════════════════════════

    def build(
        self,
        *,
        door_type: str = "push",
        left_occupied: bool = False,
        right_occupied: bool = False,
        domain_params: dict[str, Any] | None = None,
    ) -> SceneHandles:
        """装配完整场景并返回句柄集。

        Parameters
        ----------
        door_type : str
            门类型，当前仅支持 "push"。
        left_occupied : bool
            左臂是否生成杯体。
        right_occupied : bool
            右臂是否生成杯体。
        domain_params : dict | None
            域随机化参数，由 DomainRandomizer.sample_episode_params() 产出。
        """
        params = domain_params or {}

        # 1. 清理旧场景（仅首次构建后需要清理）
        if self._scene_built:
            self._clear_scene()

        # 2. 加载机器人
        robot_view = self._spawn_robot()

        # 3. 加载门
        door_view, door_panel_view = self._spawn_door(door_type)

        # 4. 读取末端执行器视图
        left_ee_view = self._get_ee_view("left")
        right_ee_view = self._get_ee_view("right")

        # 5. 按需加载杯体
        cup_view = None
        if left_occupied or right_occupied:
            cup_view = self._spawn_cup(left_occupied, right_occupied)

        # 6. 组装句柄
        self._handles = SceneHandles(
            robot_view=robot_view,
            left_ee_view=left_ee_view,
            right_ee_view=right_ee_view,
            door_view=door_view,
            door_panel_view=door_panel_view,
            cup_view=cup_view,
            door_type=door_type,
            left_occupied=left_occupied,
            right_occupied=right_occupied,
            arm_joint_names=self._arm_joint_names,
            sim_context=self._sim_context,
        )

        # 7. 在 Isaac Lab 模式下解析关节/body 索引
        if _HAS_ISAAC_LAB and not isinstance(robot_view, _PlaceholderView):
            self._resolve_indices(self._handles)

        # 8. 应用域随机化参数
        if params:
            self.apply_domain_params(params)

        self._scene_built = True
        return self._handles

    # ═══════════════════════════════════════════════════════════════════
    # 索引解析（Isaac Lab 专用）
    # ═══════════════════════════════════════════════════════════════════

    def _resolve_indices(self, handles: SceneHandles) -> None:
        """从 Articulation 视图中解析关节和 body 索引。"""
        robot = handles.robot_view

        # 解析双臂关节索引
        joint_indices, _ = robot.find_joints(self._arm_joint_names)
        handles.arm_joint_indices = joint_indices

        # 解析末端执行器 body 索引
        left_ee_indices, _ = robot.find_bodies([self._left_ee_link])
        right_ee_indices, _ = robot.find_bodies([self._right_ee_link])

        if left_ee_indices is not None and len(left_ee_indices) > 0:
            handles.left_ee_body_idx = int(left_ee_indices[0])
        if right_ee_indices is not None and len(right_ee_indices) > 0:
            handles.right_ee_body_idx = int(right_ee_indices[0])

        # 解析门铰链关节索引
        if handles.door_view is not None and not isinstance(handles.door_view, _PlaceholderView):
            door = handles.door_view
            # 查找所有门关节（通常只有一个铰链）
            door_joint_indices, _ = door.find_joints(".*")
            handles.door_hinge_joint_idx = door_joint_indices

    # ═══════════════════════════════════════════════════════════════════
    # 域随机化参数落地
    # ═══════════════════════════════════════════════════════════════════

    def apply_domain_params(self, params: dict[str, Any]) -> None:
        """将域随机化参数写入物理引擎。

        Parameters
        ----------
        params : dict
            包含 cup_mass, door_mass, door_damping, base_pos。
        """
        h = self.handles

        if "cup_mass" in params and h.cup_view is not None:
            self._set_rigid_body_mass(h.cup_view, params["cup_mass"])

        if "door_mass" in params and h.door_panel_view is not None:
            self._set_rigid_body_mass(h.door_panel_view, params["door_mass"])

        if "door_damping" in params and h.door_view is not None:
            self._set_joint_damping(h.door_view, params["door_damping"])

        if "base_pos" in params and h.robot_view is not None:
            self._teleport_base(h.robot_view, np.asarray(params["base_pos"]))

    # ═══════════════════════════════════════════════════════════════════
    # Isaac Lab API 封装（所有实际仿真调用集中于此）
    # ═══════════════════════════════════════════════════════════════════

    def _clear_scene(self) -> None:
        """清理当前场景中的所有动态 prim。"""
        if not _HAS_ISAAC_LAB:
            return

        from pxr import Usd
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        # 删除可复用的动态实体
        for prim_path in ["/World/Cup"]:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.RemovePrim(prim_path)

    def _spawn_robot(self) -> Any:
        """加载双臂机器人模型并返回 Articulation 句柄。"""
        if not _HAS_ISAAC_LAB:
            return _PlaceholderView()

        # 使用 Isaac Lab 的 ArticulationCfg 来描述机器人
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_ROBOT_ASSET_PATH,
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
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(4.6, 0.0, 0.12),
                rot=(0.0, 0.0, 1.0, 0.0),  # 180° yaw（四元数 wxyz）
                joint_pos={
                    "left_joint.*": 0.0,
                    "right_joint.*": 0.0,
                    ".*wheel": 0.0,
                },
            ),
            actuators={
                "arms": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=["left_joint.*", "right_joint.*"],
                    effort_limit=33.5,
                    velocity_limit=2.175,
                    stiffness=0.0,    # 力矩控制模式
                    damping=0.0,      # 力矩控制模式
                ),
                "wheels": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=[".*wheel"],
                    effort_limit=20.0,
                    velocity_limit=10.0,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )

        robot = Articulation(robot_cfg)
        return robot

    def _spawn_door(self, door_type: str) -> tuple[Any, Any]:
        """加载指定类型的门并返回 (Articulation, RigidObject|None)。"""
        if not _HAS_ISAAC_LAB:
            return _PlaceholderView(), _PlaceholderView()

        asset_path = _DOOR_ASSET_MAP.get(door_type, _DOOR_ASSET_MAP["push"])

        # 门作为带铰链的 Articulation
        door_cfg = ArticulationCfg(
            prim_path="/World/Door",
            spawn=sim_utils.UsdFileCfg(
                usd_path=asset_path,
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
                joint_pos={".*": 0.0},
            ),
            actuators={
                "hinge": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit=0.0,     # 门自由转动
                    velocity_limit=100.0,
                    stiffness=0.0,
                    damping=2.0,          # 默认阻尼，可被域随机化覆盖
                ),
            },
        )

        door = Articulation(door_cfg)
        # 门板的 RigidObject 在 stage 加载后通过 prim 路径获取
        # 暂时用 door 本身作为门板引用
        door_panel = None

        return door, door_panel

    def _get_ee_view(self, side: str) -> Any:
        """获取指定侧的末端执行器引用。

        在 Isaac Lab 中，末端 body 的位姿通过
        robot.data.body_pos_w / body_quat_w 按索引获取，
        因此此处仅返回标识字符串，实际索引在 _resolve_indices 中解析。
        """
        if not _HAS_ISAAC_LAB:
            return _PlaceholderView()

        link_name = self._left_ee_link if side == "left" else self._right_ee_link
        return link_name

    def _spawn_cup(
        self, left_occupied: bool, right_occupied: bool
    ) -> Any:
        """在持杯臂的末端生成杯体 RigidObject。"""
        if not _HAS_ISAAC_LAB:
            return _PlaceholderView()

        # 杯体作为独立 RigidObject
        # 杯体的初始位姿由预录抓取轨迹决定，此处给一个默认位置
        # 实际运行时由 teleop_cup_grasp 回放覆盖
        cup_cfg = RigidObjectCfg(
            prim_path="/World/Cup",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_CUP_ASSET_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(4.0, 0.0, 1.0),  # 默认位置，待杯子抓取轨迹覆盖
            ),
        )

        cup = RigidObject(cup_cfg)
        return cup

    def _set_rigid_body_mass(self, view: Any, mass: float) -> None:
        """修改刚体的质量属性。"""
        if not _HAS_ISAAC_LAB or isinstance(view, _PlaceholderView):
            return

        if isinstance(view, RigidObject):
            # Isaac Lab RigidObject 可通过 root_physx_view 设置质量
            import torch
            masses = torch.tensor([[mass]], dtype=torch.float32, device=view.device)
            view.root_physx_view.set_masses(masses)
        elif isinstance(view, Articulation):
            # 对 Articulation 的各 body 设置质量
            import torch
            current_masses = view.root_physx_view.get_masses()
            current_masses[:] = mass
            view.root_physx_view.set_masses(current_masses)

    def _set_joint_damping(self, view: Any, damping: float) -> None:
        """修改关节阻尼系数。"""
        if not _HAS_ISAAC_LAB or isinstance(view, _PlaceholderView):
            return

        if isinstance(view, Articulation):
            import torch
            num_joints = view.num_joints
            dampings = torch.full(
                (view.num_instances, num_joints),
                damping,
                dtype=torch.float32,
                device=view.device,
            )
            view.write_joint_damping_to_sim(dampings)

    def _teleport_base(self, view: Any, pos: np.ndarray) -> None:
        """将机器人基座传送至指定世界坐标位置。"""
        if not _HAS_ISAAC_LAB or isinstance(view, _PlaceholderView):
            return

        if isinstance(view, Articulation):
            import torch
            # 保持当前朝向，只修改位置
            pos_t = torch.tensor(
                pos.reshape(1, 3), dtype=torch.float32, device=view.device
            )
            view.write_root_pose_to_sim(
                root_pose=torch.cat([
                    pos_t,
                    view.data.root_quat_w,
                ], dim=-1)
            )
