"""场景装配工厂 — 封装所有 Isaac Sim 资产加载与物理属性修改。

本模块是 envs 层中**唯一直接调用 Isaac Sim API** 的位置。
所有仿真 API 调用集中在标注 ``# [ISAAC_API]`` 的行，
便于在真实 Isaac Sim 设备上一次性对接。

职责：
    1. 加载机器人模型（双臂 Z1 + Dingo 底座）
    2. 根据课程阶段决定门类型（push / pull / handle_push / handle_pull）
    3. 根据 occupied 标记决定是否生成杯体
    4. 应用域随机化参数（质量、阻尼、基座位置）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Isaac Sim 视图句柄的类型别名
# ═══════════════════════════════════════════════════════════════════════
# 实际对接时替换为:
#   from omni.isaac.core.articulations import ArticulationView
#   from omni.isaac.core.prims import RigidPrimView


class _PlaceholderView:
    """Isaac Sim 视图的占位类型，用于无 Isaac Sim 环境下的开发。

    在真实设备上，应替换为 ArticulationView / RigidPrimView。
    """
    pass


# 门资产 USD 路径映射（示例路径，实际部署时替换）
_DOOR_ASSET_MAP: dict[str, str] = {
    "push": "assets/doors/push_door.usd",
    "pull": "assets/doors/pull_door.usd",
    "handle_push": "assets/doors/handle_push_door.usd",
    "handle_pull": "assets/doors/handle_pull_door.usd",
}

# 机器人 USD 路径
_ROBOT_ASSET_PATH = "assets/robots/dual_z1_dingo.usd"

# 杯体 USD 路径
_CUP_ASSET_PATH = "assets/objects/cup.usd"


@dataclass
class SceneHandles:
    """场景中各物体的仿真句柄集合。

    将所有 Isaac Sim 视图引用集中管理，避免在环境代码中散落。
    """

    # 机器人 ArticulationView
    robot_view: Any = None
    # 左臂 gripper RigidPrimView
    left_ee_view: Any = None
    # 右臂 gripper RigidPrimView
    right_ee_view: Any = None
    # 门的 ArticulationView（含铰链关节）
    door_view: Any = None
    # 门板 RigidPrimView（用于读取位姿）
    door_panel_view: Any = None
    # 杯体 RigidPrimView（None 表示本局无杯体）
    cup_view: Any = None

    # 当前场景元信息
    door_type: str = "push"
    left_occupied: bool = False
    right_occupied: bool = False


class SceneFactory:
    """Isaac Sim 场景装配工厂。

    每次 episode reset 时调用 ``build()`` 重建场景。

    Parameters
    ----------
    physics_dt : float
        物理仿真步长。
    """

    def __init__(self, physics_dt: float = 1.0 / 120.0) -> None:
        self._physics_dt = physics_dt
        self._handles: SceneHandles | None = None

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
            门类型，取值 "push" / "pull" / "handle_push" / "handle_pull"。
        left_occupied : bool
            左臂是否生成杯体。
        right_occupied : bool
            右臂是否生成杯体。
        domain_params : dict | None
            域随机化参数，由 DomainRandomizer.sample_episode_params() 产出。
        """
        params = domain_params or {}

        # 1. 清理旧场景
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
        )

        # 7. 应用域随机化参数
        if params:
            self.apply_domain_params(params)

        return self._handles

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
    # Isaac Sim API 封装（所有实际仿真调用集中于此）
    # ═══════════════════════════════════════════════════════════════════

    def _clear_scene(self) -> None:
        """清理当前场景中的所有 prim。"""
        # [ISAAC_API] 实际实现：
        #   from omni.isaac.core.utils.prims import delete_prim
        #   delete_prim("/World/Robot")
        #   delete_prim("/World/Door")
        #   delete_prim("/World/Cup")
        pass

    def _spawn_robot(self) -> Any:
        """加载双臂机器人模型并返回 ArticulationView。"""
        # [ISAAC_API] 实际实现：
        #   from omni.isaac.core.utils.stage import add_reference_to_stage
        #   add_reference_to_stage(_ROBOT_ASSET_PATH, "/World/Robot")
        #   robot = ArticulationView(prim_paths_expr="/World/Robot",
        #                            name="robot_view")
        #   world.scene.add(robot)
        #   return robot
        return _PlaceholderView()

    def _spawn_door(self, door_type: str) -> tuple[Any, Any]:
        """加载指定类型的门并返回 (ArticulationView, RigidPrimView)。"""
        asset_path = _DOOR_ASSET_MAP.get(door_type, _DOOR_ASSET_MAP["push"])
        # [ISAAC_API] 实际实现：
        #   add_reference_to_stage(asset_path, "/World/Door")
        #   door_art = ArticulationView(prim_paths_expr="/World/Door",
        #                               name="door_view")
        #   door_panel = RigidPrimView(prim_paths_expr="/World/Door/panel",
        #                              name="door_panel_view")
        #   world.scene.add(door_art)
        #   world.scene.add(door_panel)
        #   return door_art, door_panel
        return _PlaceholderView(), _PlaceholderView()

    def _get_ee_view(self, side: str) -> Any:
        """获取指定侧的末端执行器 RigidPrimView。"""
        # [ISAAC_API] 实际实现：
        #   prim_path = f"/World/Robot/{side}_gripper_link"
        #   ee = RigidPrimView(prim_paths_expr=prim_path,
        #                      name=f"{side}_ee_view")
        #   world.scene.add(ee)
        #   return ee
        return _PlaceholderView()

    def _spawn_cup(
        self, left_occupied: bool, right_occupied: bool
    ) -> Any:
        """在持杯臂的末端生成杯体。"""
        # [ISAAC_API] 实际实现：
        #   add_reference_to_stage(_CUP_ASSET_PATH, "/World/Cup")
        #   cup = RigidPrimView(prim_paths_expr="/World/Cup",
        #                       name="cup_view")
        #   # 将杯体位姿对齐到持杯臂的末端
        #   world.scene.add(cup)
        #   return cup
        return _PlaceholderView()

    def _set_rigid_body_mass(self, view: Any, mass: float) -> None:
        """修改刚体的质量属性。"""
        # [ISAAC_API] 实际实现：
        #   from omni.isaac.core.utils.prims import set_prim_property
        #   set_prim_property(view.prim_paths[0],
        #                     "physics:mass", mass)
        pass

    def _set_joint_damping(self, view: Any, damping: float) -> None:
        """修改关节阻尼系数。"""
        # [ISAAC_API] 实际实现：
        #   view.set_joint_dampings(
        #       np.array([[damping]]))
        pass

    def _teleport_base(self, view: Any, pos: np.ndarray) -> None:
        """将机器人基座传送至指定世界坐标位置。"""
        # [ISAAC_API] 实际实现：
        #   from omni.isaac.core.utils.transformations import ...
        #   view.set_world_poses(positions=pos.reshape(1, 3))
        pass
