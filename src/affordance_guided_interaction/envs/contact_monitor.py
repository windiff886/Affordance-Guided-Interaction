"""接触事件汇总器 — 从 Isaac Sim Contact Sensor 提取并过滤接触信息。

职责：
    1. 每步从物理引擎拉取原始接触数据
    2. 按力阈值过滤微碰撞噪声
    3. 按 link 归类接触力大小
    4. 检测自碰撞（同一 articulation 内的 link pair）
    5. 检测杯体脱落（杯体与末端距离超阈值）

输出 ContactSummary 供 rewards/safety_penalty.py 消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ContactSummary:
    """单步接触事件的结构化摘要。

    Attributes
    ----------
    link_forces : dict[str, float]
        各 link 名称到接触法向力合力大小（N）的映射。
        仅包含超过力阈值的有效接触。
    self_collision : bool
        机器人是否发生自碰撞（任意两个自身 link 接触）。
    cup_dropped : bool
        杯体是否脱落（杯体与持握末端的距离超过阈值）。
    max_contact_force : float
        本步所有有效接触中力最大的那个值（N）。
    """

    link_forces: dict[str, float] = field(default_factory=dict)
    self_collision: bool = False
    cup_dropped: bool = False
    max_contact_force: float = 0.0


# 机器人自身 link 名称列表（用于自碰撞检测）
# 实际部署时根据 URDF 补全
_ROBOT_LINK_NAMES: set[str] = {
    "left_link0", "left_link1", "left_link2",
    "left_link3", "left_link4", "left_link5",
    "left_gripper_link",
    "right_link0", "right_link1", "right_link2",
    "right_link3", "right_link4", "right_link5",
    "right_gripper_link",
    "base_link", "torso_link",
}

# 不计入碰撞惩罚的 link 对（相邻关节的接触是正常物理）
_ADJACENT_PAIRS: set[frozenset[str]] = {
    frozenset({"left_link0", "left_link1"}),
    frozenset({"left_link1", "left_link2"}),
    frozenset({"left_link2", "left_link3"}),
    frozenset({"left_link3", "left_link4"}),
    frozenset({"left_link4", "left_link5"}),
    frozenset({"left_link5", "left_gripper_link"}),
    frozenset({"right_link0", "right_link1"}),
    frozenset({"right_link1", "right_link2"}),
    frozenset({"right_link2", "right_link3"}),
    frozenset({"right_link3", "right_link4"}),
    frozenset({"right_link4", "right_link5"}),
    frozenset({"right_link5", "right_gripper_link"}),
}


class ContactMonitor:
    """接触事件汇总器。

    Parameters
    ----------
    force_threshold : float
        接触力过滤阈值（N），低于此值的微碰撞被忽略。
    cup_drop_threshold : float
        杯体脱落检测距离（m）。
    """

    def __init__(
        self,
        force_threshold: float = 0.1,
        cup_drop_threshold: float = 0.15,
    ) -> None:
        self._force_threshold = force_threshold
        self._cup_drop_threshold = cup_drop_threshold

    def update(
        self,
        *,
        scene_handles: Any,
        left_ee_pos: np.ndarray | None = None,
        right_ee_pos: np.ndarray | None = None,
        cup_pos: np.ndarray | None = None,
        left_occupied: bool = False,
        right_occupied: bool = False,
    ) -> ContactSummary:
        """从物理引擎拉取接触数据并生成摘要。

        Parameters
        ----------
        scene_handles
            SceneHandles 实例，用于访问 Isaac Sim 视图。
        left_ee_pos : (3,) | None
            左臂末端世界坐标位置。
        right_ee_pos : (3,) | None
            右臂末端世界坐标位置。
        cup_pos : (3,) | None
            杯体世界坐标位置（无杯体时为 None）。
        left_occupied : bool
            左臂是否持杯。
        right_occupied : bool
            右臂是否持杯。

        Returns
        -------
        ContactSummary
        """
        # ── 1. 从物理引擎读取原始接触 ────────────────────────────
        raw_contacts = self._read_raw_contacts(scene_handles)

        # ── 2. 阈值过滤 + 按 link 归类 ─────────────────────────
        link_forces: dict[str, float] = {}
        max_force = 0.0

        for contact in raw_contacts:
            body_a: str = contact["body_a"]
            body_b: str = contact["body_b"]
            force_mag: float = contact["force_magnitude"]

            if force_mag < self._force_threshold:
                continue

            # 记录各 link 的接触力
            if body_a in _ROBOT_LINK_NAMES:
                link_forces[body_a] = link_forces.get(body_a, 0.0) + force_mag
            if body_b in _ROBOT_LINK_NAMES:
                link_forces[body_b] = link_forces.get(body_b, 0.0) + force_mag

            max_force = max(max_force, force_mag)

        # ── 3. 自碰撞检测 ───────────────────────────────────────
        self_collision = self._check_self_collision(raw_contacts)

        # ── 4. 杯体脱落检测 ─────────────────────────────────────
        cup_dropped = self._check_cup_dropped(
            cup_pos=cup_pos,
            left_ee_pos=left_ee_pos,
            right_ee_pos=right_ee_pos,
            left_occupied=left_occupied,
            right_occupied=right_occupied,
        )

        return ContactSummary(
            link_forces=link_forces,
            self_collision=self_collision,
            cup_dropped=cup_dropped,
            max_contact_force=max_force,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════════

    def _read_raw_contacts(self, scene_handles: Any) -> list[dict]:
        """从 Isaac Sim 读取原始接触对列表。

        Returns
        -------
        list[dict]
            每个元素形如：
            {"body_a": str, "body_b": str, "force_magnitude": float}
        """
        # [ISAAC_API] 实际实现：
        #   from omni.isaac.core.utils.physics import get_contact_data
        #   contacts = get_contact_data(...)
        #   return [{"body_a": c.body0, "body_b": c.body1,
        #            "force_magnitude": np.linalg.norm(c.impulse / dt)}
        #           for c in contacts]
        return []

    def _check_self_collision(self, raw_contacts: list[dict]) -> bool:
        """检查是否存在非相邻 link 之间的自碰撞。"""
        for contact in raw_contacts:
            body_a = contact["body_a"]
            body_b = contact["body_b"]
            force_mag = contact["force_magnitude"]

            if force_mag < self._force_threshold:
                continue

            # 两个碰撞体都是机器人自身的 link
            if body_a in _ROBOT_LINK_NAMES and body_b in _ROBOT_LINK_NAMES:
                pair = frozenset({body_a, body_b})
                # 排除相邻关节的正常接触
                if pair not in _ADJACENT_PAIRS:
                    return True
        return False

    def _check_cup_dropped(
        self,
        *,
        cup_pos: np.ndarray | None,
        left_ee_pos: np.ndarray | None,
        right_ee_pos: np.ndarray | None,
        left_occupied: bool,
        right_occupied: bool,
    ) -> bool:
        """检查杯体是否脱落（与持握末端距离超限）。"""
        if cup_pos is None:
            return False

        # 检查左臂持杯情况
        if left_occupied and left_ee_pos is not None:
            dist = float(np.linalg.norm(cup_pos - left_ee_pos))
            if dist > self._cup_drop_threshold:
                return True

        # 检查右臂持杯情况
        if right_occupied and right_ee_pos is not None:
            dist = float(np.linalg.norm(cup_pos - right_ee_pos))
            if dist > self._cup_drop_threshold:
                return True

        return False
