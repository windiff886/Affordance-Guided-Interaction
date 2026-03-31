"""将各子组件的输出组装为完整的 actor observation。

Actor observation 结构（参见 README.md §4）：

```
actor_obs = {
    "proprio": {
        "joint_positions":   (6,),
        "joint_velocities":  (6,),
        "joint_torques":     (6,),       # 可选
        "previous_actions":  (k, 6),
    },
    "gripper_state": {
        "position":          (3,),       # 基坐标系
        "orientation":       (4,),       # 基坐标系, quat (w,x,y,z)
        "linear_velocity":   (3,),       # 基坐标系
        "angular_velocity":  (3,),       # 基坐标系
    },
    "context": {
        "occupied":          (1,),
        "stability_level":   (1,),
    },
    "stability_proxy": { ... },          # 由 stability_proxy.py 计算
    "door_point_cloud":      (N, 3),     # 门点云
}
```

所有量使用 numpy ndarray 表示。
"""

from __future__ import annotations

import numpy as np

from .history_buffer import HistoryBuffer
from .stability_proxy import (
    StabilityProxy,
    StabilityProxyState,
    estimate_stability_proxy,
)

# Z1 机械臂单臂关节数
NUM_ARM_JOINTS = 6


class ActorObsBuilder:
    """有状态的 actor observation 构建器。

    在 episode 生命周期内维护：

    * 动作历史缓存 (``HistoryBuffer``)
    * 稳定性 proxy 的流式差分状态 (``StabilityProxyState``)

    Parameters
    ----------
    action_history_length : int
        保留最近多少步动作，默认 3。
    acc_history_length : int
        稳定性 proxy 中加速度历史窗口长度，默认 10。
    dt : float
        仿真步长（秒），默认 1/60。
    """

    def __init__(
        self,
        *,
        action_history_length: int = 3,
        acc_history_length: int = 10,
        dt: float = 1.0 / 60.0,
    ) -> None:
        self._action_history_length = action_history_length
        self._acc_history_length = acc_history_length
        self._dt = dt

        # 内部状态 —— 在 reset() 中初始化
        self._action_buffer: HistoryBuffer[np.ndarray] | None = None
        self._stability_state: StabilityProxyState | None = None
        self.reset()

    # ------------------------------------------------------------------
    # Episode 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """每个 episode 开始时调用，清空所有内部状态。"""
        zero_action = np.zeros(NUM_ARM_JOINTS, dtype=np.float64)
        self._action_buffer = HistoryBuffer(
            self._action_history_length, fill_value=zero_action
        )
        self._stability_state = StabilityProxyState()

    # ------------------------------------------------------------------
    # 每步构建
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray | None = None,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        ee_linear_velocity: np.ndarray,
        ee_angular_velocity: np.ndarray,
        occupied: float,
        stability_level: float,
        door_point_cloud: np.ndarray | None = None,
        action_taken: np.ndarray | None = None,
    ) -> dict:
        """构建单步 actor observation。

        Parameters
        ----------
        joint_positions : (6,)
            Z1 单臂 6 个旋转关节角度。
        joint_velocities : (6,)
            关节角速度。
        joint_torques : (6,) | None
            当前关节力矩，可选。
        ee_position : (3,)
            gripper 在基坐标系下的位置。
        ee_orientation : (4,)
            gripper 在基坐标系下的四元数 (w, x, y, z)。
        ee_linear_velocity : (3,)
            gripper 在基坐标系下的线速度。
        ee_angular_velocity : (3,)
            gripper 在基坐标系下的角速度。
        occupied : float
            0.0 = 空手, 1.0 = 持杯。
        stability_level : float
            稳定性要求等级标量 (0/1/2 ...)。
        door_point_cloud : (N, 3) | None
            门点云。None 表示本 step 没有新的点云。
        action_taken : (6,) | None
            本 step 策略输出的动作；如果提供，会被记录到动作历史。
            通常在 env.step() 之后、下一次 build() 之前传入。

        Returns
        -------
        dict
            完整的 actor observation 字典。
        """
        assert self._action_buffer is not None
        assert self._stability_state is not None

        # 记录动作
        if action_taken is not None:
            self._action_buffer.append(np.asarray(action_taken, dtype=np.float64))

        # -- proprio --
        proprio: dict[str, np.ndarray] = {
            "joint_positions": np.asarray(joint_positions, dtype=np.float64),
            "joint_velocities": np.asarray(joint_velocities, dtype=np.float64),
            "previous_actions": self._action_buffer.to_numpy(),
        }
        if joint_torques is not None:
            proprio["joint_torques"] = np.asarray(joint_torques, dtype=np.float64)

        # -- gripper_state --
        gripper_state = {
            "position": np.asarray(ee_position, dtype=np.float64),
            "orientation": np.asarray(ee_orientation, dtype=np.float64),
            "linear_velocity": np.asarray(ee_linear_velocity, dtype=np.float64),
            "angular_velocity": np.asarray(ee_angular_velocity, dtype=np.float64),
        }

        # -- context --
        context = {
            "occupied": np.array([occupied], dtype=np.float64),
            "stability_level": np.array([stability_level], dtype=np.float64),
        }

        # -- stability proxy --
        proxy: StabilityProxy = estimate_stability_proxy(
            quat_ee=np.asarray(ee_orientation, dtype=np.float64),
            linear_velocity=np.asarray(ee_linear_velocity, dtype=np.float64),
            angular_velocity=np.asarray(ee_angular_velocity, dtype=np.float64),
            dt=self._dt,
            state=self._stability_state,
            acc_history_length=self._acc_history_length,
        )

        # -- door point cloud --
        if door_point_cloud is not None:
            dpc = np.asarray(door_point_cloud, dtype=np.float64)
        else:
            # 没有点云时提供空数组 (0, 3)
            dpc = np.zeros((0, 3), dtype=np.float64)

        return {
            "proprio": proprio,
            "gripper_state": gripper_state,
            "context": context,
            "stability_proxy": proxy.to_dict(),
            "door_point_cloud": dpc,
        }
