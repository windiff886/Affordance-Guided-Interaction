"""接触事件监控器 — 从 Isaac Lab ContactSensor 提取接触信息。

本模块封装 ContactSensor.data 的读取逻辑，提供以下两层抽象：

    1. 原始接触数据 → 按 link 分组的接触力向量
    2. 语义事件：自碰撞标志、外部碰撞力汇总、杯体脱落检测

数据流：
    ContactSensor.data.net_forces_w → _read_raw_contacts()
    → 力阈值过滤 → 分桶（自碰撞 / 外部碰撞）→ ContactSummary
    → 杯体脱落检测（距离法）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ContactSummary:
    """单步接触事件汇总。

    Attributes
    ----------
    self_collision : bool
        是否发生自碰撞。
    cup_dropped : bool
        持杯臂的杯体是否脱落。
    link_forces : dict[str, np.ndarray]
        每个接触 link 的合力向量 (3,)，key = link prim name。
    total_external_force : float
        所有非自碰撞接触的合力大小（N）。
    max_contact_force : float
        单次接触的最大力大小（N）。
    """

    self_collision: bool = False
    cup_dropped: bool = False
    link_forces: dict[str, np.ndarray] = field(default_factory=dict)
    total_external_force: float = 0.0
    max_contact_force: float = 0.0


class ContactMonitor:
    """接触事件监控器。

    Parameters
    ----------
    force_threshold : float
        力阈值（N），低于此值的微碰撞被忽略。
    cup_drop_threshold : float
        杯体脱落判定距离阈值（m）。
        当杯体中心与持杯末端之间的距离超过此阈值时，判定为脱落。
    """

    # 自碰撞源 link 集合（机器人自身部件之间的碰撞）
    _SELF_COLLISION_PAIRS: set[str] = {
        "left_gripper_link",
        "right_gripper_link",
        "base_link",
        "left_link1", "left_link2", "left_link3",
        "left_link4", "left_link5", "left_link6",
        "right_link1", "right_link2", "right_link3",
        "right_link4", "right_link5", "right_link6",
    }

    def __init__(
        self,
        force_threshold: float = 0.1,
        cup_drop_threshold: float = 0.15,
    ) -> None:
        self._force_threshold = force_threshold
        self._cup_drop_threshold = cup_drop_threshold

    def update(
        self,
        scene_handles: Any,
        left_ee_pos: np.ndarray,
        right_ee_pos: np.ndarray,
        cup_pos: np.ndarray | None = None,
        left_occupied: bool = False,
        right_occupied: bool = False,
    ) -> ContactSummary:
        """更新接触状态。

        Parameters
        ----------
        scene_handles : SceneHandles
            场景句柄集合。
        left_ee_pos : (3,) ndarray
            左臂末端世界坐标。
        right_ee_pos : (3,) ndarray
            右臂末端世界坐标。
        cup_pos : (3,) ndarray | None
            杯体世界坐标。如果无杯体则为 None。
        left_occupied : bool
            左臂是否持杯。
        right_occupied : bool
            右臂是否持杯。
        """
        # 读取原始接触数据
        raw_forces = self._read_raw_contacts(scene_handles)

        # 初始化汇总
        summary = ContactSummary()

        # ── 力阈值过滤 + 分类 ─────────────────────────────────
        self_collision_detected = False
        total_external = 0.0
        max_force = 0.0

        for link_name, force_vec in raw_forces.items():
            force_mag = float(np.linalg.norm(force_vec))

            # 低于阈值的噪抖忽略
            if force_mag < self._force_threshold:
                continue

            max_force = max(max_force, force_mag)
            summary.link_forces[link_name] = force_vec

            # 判断是否为自碰撞
            if link_name in self._SELF_COLLISION_PAIRS:
                self_collision_detected = True
            else:
                total_external += force_mag

        summary.self_collision = self_collision_detected
        summary.total_external_force = total_external
        summary.max_contact_force = max_force

        # ── 杯体脱落检测（距离法）──────────────────────────────
        if cup_pos is not None:
            if left_occupied:
                dist = float(np.linalg.norm(cup_pos - left_ee_pos))
                if dist > self._cup_drop_threshold:
                    summary.cup_dropped = True
            if right_occupied and not summary.cup_dropped:
                dist = float(np.linalg.norm(cup_pos - right_ee_pos))
                if dist > self._cup_drop_threshold:
                    summary.cup_dropped = True

        return summary

    def _read_raw_contacts(
        self, scene_handles: Any
    ) -> dict[str, np.ndarray]:
        """从 Isaac Lab ContactSensor 读取每个 link 的净接触力。

        Returns
        -------
        dict[str, np.ndarray]
            {link_name: force_vector_world (3,)}
        """
        result: dict[str, np.ndarray] = {}

        # 检查 Isaac Lab 可用性
        if scene_handles is None:
            return result

        robot_view = scene_handles.robot_view

        # 占位模式：无接触数据
        if robot_view is None or not hasattr(robot_view, 'data'):
            return result

        # Isaac Lab ContactSensor 读取路径：
        # robot.data.body_contact_net_forces_w → (num_instances, num_bodies, 3)
        # 该属性在 activate_contact_sensors=True 时可用
        try:
            net_forces = robot_view.data.body_contact_net_forces_w
            if net_forces is None:
                return result

            # net_forces shape: (1, num_bodies, 3) → 取第一个实例
            forces = net_forces[0]  # (num_bodies, 3)

            # 遍历所有 body，构建 {link_name: force_vec}
            body_names = robot_view.body_names
            for i, name in enumerate(body_names):
                force_vec = forces[i].cpu().numpy().astype(np.float64)
                # 只记录有显著力的 body
                if np.linalg.norm(force_vec) > 0:
                    result[name] = force_vec

        except (AttributeError, RuntimeError):
            # Isaac Lab 版本差异或未启用 contact sensor 时 graceful fallback
            pass

        return result
