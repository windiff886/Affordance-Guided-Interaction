"""在 actor observation 基础上追加目标规格 privileged information。"""

from __future__ import annotations

import numpy as np


class CriticObsBuilder:
    """无状态的 critic observation 构建器。

    只需在每步将 actor_obs 与 privileged info 合并即可。
    """

    @staticmethod
    def build(
        *,
        actor_obs: dict,
        door_pose: np.ndarray | None = None,
        door_joint_pos: float = 0.0,
        door_joint_vel: float = 0.0,
        cup_mass: float = 0.0,
        door_mass: float = 0.0,
        door_damping: float = 0.0,
        base_pos: np.ndarray | None = None,
        cup_dropped: float = 0.0,
    ) -> dict:
        """构建单步 critic observation。

        Parameters
        ----------
        actor_obs : dict
            完整的 actor observation（由 ``ActorObsBuilder.build()`` 返回）。
        door_pose : (7,) | None
            门板相对 `base_link` 的 pose：pos(3) + quat(4)。
        door_joint_pos : float
            门铰链角度 (rad)。
        door_joint_vel : float
            门铰链角速度 (rad/s)。
        cup_mass : float
            杯体质量 (kg)，通过随机化模拟不同装载量。
        door_mass : float
            门板质量 (kg)。
        door_damping : float
            门铰链阻尼系数。
        base_pos : (3,) | None
            机器人基座的世界坐标位置（回合级随机化，局限于运动学可达裕度内）。
        cup_dropped : float
            杯体掉落事件标志。

        Returns
        -------
        dict
            完整的 critic observation 字典。
        """
        _zero3 = np.zeros(3, dtype=np.float64)

        privileged = {
            "door_pose": (
                np.asarray(door_pose, dtype=np.float64)
                if door_pose is not None
                else _zero7.copy()
            ),
            "door_joint_pos": np.array([door_joint_pos], dtype=np.float64),
            "door_joint_vel": np.array([door_joint_vel], dtype=np.float64),
            "cup_mass": np.array([cup_mass], dtype=np.float64),
            "door_mass": np.array([door_mass], dtype=np.float64),
            "door_damping": np.array([door_damping], dtype=np.float64),
            "base_pos": (
                np.asarray(base_pos, dtype=np.float64)
                if base_pos is not None
                else _zero3.copy()
            ),
            "cup_dropped": np.array([cup_dropped], dtype=np.float64),
        }

        return {
            "actor_obs": actor_obs,
            "privileged": privileged,
        }
