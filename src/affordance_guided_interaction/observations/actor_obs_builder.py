"""将各子组件的输出组装为完整的 actor observation。

双臂平台（左臂 Z1 + 右臂 Z1）actor observation 结构（参见 README.md §4）：

持杯是 episode 开始时随机确定的事件，左右臂均可能持杯，故左右臂状态
对称提供，各维护独立的稳定性 proxy，通过 left_occupied / right_occupied
区分当前持杯情况。

```
actor_obs = {
    "proprio": {
        "left_joint_positions":    (6,),
        "left_joint_velocities":   (6,),
        "right_joint_positions":   (6,),
        "right_joint_velocities":  (6,),
        "left_joint_torques":      (6,),      # 可选
        "right_joint_torques":     (6,),      # 可选
        "previous_actions":        (k, 12),   # 双臂完整动作历史
    },
    "left_gripper_state": {
        "position":                (3,),      # 基坐标系
        "orientation":             (4,),      # quat (w,x,y,z)，基坐标系
        "linear_velocity":         (3,),
        "angular_velocity":        (3,),
    },
    "right_gripper_state": {
        "position":                (3,),
        "orientation":             (4,),
        "linear_velocity":         (3,),
        "angular_velocity":        (3,),
    },
    "context": {
        "left_occupied":           (1,),      # 左臂是否持杯 (0/1)
        "right_occupied":          (1,),      # 右臂是否持杯 (0/1)
    },
    "left_stability_proxy":  { ... },         # 左臂末端稳定性指标
    "right_stability_proxy": { ... },         # 右臂末端稳定性指标
    "door_embedding":          (768,),        # 冻结 Point-MAE 对门点云的编码
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

# 每条 Z1 机械臂的关节数
NUM_JOINTS_PER_ARM = 6
# 双臂总关节数（左臂 + 右臂）
TOTAL_ARM_JOINTS = NUM_JOINTS_PER_ARM * 2

# Point-MAE 编码器输出维度（2 × trans_dim = 2 × 384）
DOOR_EMBEDDING_DIM = 768


class ActorObsBuilder:
    """有状态的 actor observation 构建器（双臂平台）。

    持杯是随机事件，左右臂均可能持杯，因此左右臂状态对称提供，
    各维护独立的稳定性 proxy 差分状态。

    在 episode 生命周期内维护：

    * 动作历史缓存 (``HistoryBuffer``)，动作维度为 ``TOTAL_ARM_JOINTS`` (12)
    * 左臂稳定性 proxy 差分状态 (``StabilityProxyState``)
    * 右臂稳定性 proxy 差分状态 (``StabilityProxyState``)

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
        self._left_stability_state: StabilityProxyState | None = None
        self._right_stability_state: StabilityProxyState | None = None
        self.reset()

    # ------------------------------------------------------------------
    # Episode 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """每个 episode 开始时调用，清空所有内部状态。"""
        zero_action = np.zeros(TOTAL_ARM_JOINTS, dtype=np.float64)
        self._action_buffer = HistoryBuffer(
            self._action_history_length, fill_value=zero_action
        )
        self._left_stability_state = StabilityProxyState()
        self._right_stability_state = StabilityProxyState()

    # ------------------------------------------------------------------
    # 每步构建
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        left_joint_positions: np.ndarray,
        left_joint_velocities: np.ndarray,
        right_joint_positions: np.ndarray,
        right_joint_velocities: np.ndarray,
        left_joint_torques: np.ndarray | None = None,
        right_joint_torques: np.ndarray | None = None,
        left_ee_position: np.ndarray,
        left_ee_orientation: np.ndarray,
        left_ee_linear_velocity: np.ndarray,
        left_ee_angular_velocity: np.ndarray,
        right_ee_position: np.ndarray,
        right_ee_orientation: np.ndarray,
        right_ee_linear_velocity: np.ndarray,
        right_ee_angular_velocity: np.ndarray,
        left_occupied: float,
        right_occupied: float,
        door_embedding: np.ndarray | None = None,
        action_taken: np.ndarray | None = None,
    ) -> dict:
        """构建单步 actor observation（双臂平台）。

        Parameters
        ----------
        left_joint_positions : (6,)
            Z1 左臂 6 个旋转关节角度。
        left_joint_velocities : (6,)
            左臂关节角速度。
        right_joint_positions : (6,)
            Z1 右臂 6 个旋转关节角度。
        right_joint_velocities : (6,)
            右臂关节角速度。
        left_joint_torques : (6,) | None
            左臂关节力矩，可选。
        right_joint_torques : (6,) | None
            右臂关节力矩，可选。
        left_ee_position : (3,)
            左臂 gripper 在基坐标系下的位置。
        left_ee_orientation : (4,)
            左臂 gripper 在基坐标系下的四元数 (w, x, y, z)。
        left_ee_linear_velocity : (3,)
            左臂 gripper 在基坐标系下的线速度。
        left_ee_angular_velocity : (3,)
            左臂 gripper 在基坐标系下的角速度。
        right_ee_position : (3,)
            右臂 gripper 在基坐标系下的位置。
        right_ee_orientation : (4,)
            右臂 gripper 在基坐标系下的四元数 (w, x, y, z)。
        right_ee_linear_velocity : (3,)
            右臂 gripper 在基坐标系下的线速度。
        right_ee_angular_velocity : (3,)
            右臂 gripper 在基坐标系下的角速度。
        left_occupied : float
            左臂是否持杯：0.0 = 空闲, 1.0 = 持杯。
        right_occupied : float
            右臂是否持杯：0.0 = 空闲, 1.0 = 持杯。
        door_embedding : (768,) | None
            来自 door_perception/ 的冻结 Point-MAE 编码结果。
            None 时填充零向量。
        action_taken : (12,) | None
            本 step 策略输出的双臂完整动作（左臂 6 + 右臂 6）；
            如果提供，会被记录到动作历史。
            通常在 env.step() 之后、下一次 build() 之前传入。

        Returns
        -------
        dict
            完整的 actor observation 字典。
        """
        assert self._action_buffer is not None
        assert self._left_stability_state is not None
        assert self._right_stability_state is not None

        # 记录动作
        if action_taken is not None:
            self._action_buffer.append(np.asarray(action_taken, dtype=np.float64))

        # -- proprio --
        proprio: dict[str, np.ndarray] = {
            "left_joint_positions": np.asarray(left_joint_positions, dtype=np.float64),
            "left_joint_velocities": np.asarray(left_joint_velocities, dtype=np.float64),
            "right_joint_positions": np.asarray(right_joint_positions, dtype=np.float64),
            "right_joint_velocities": np.asarray(right_joint_velocities, dtype=np.float64),
            "previous_actions": self._action_buffer.to_numpy(),  # (k, 12)
        }
        if left_joint_torques is not None:
            proprio["left_joint_torques"] = np.asarray(left_joint_torques, dtype=np.float64)
        if right_joint_torques is not None:
            proprio["right_joint_torques"] = np.asarray(right_joint_torques, dtype=np.float64)

        # -- left_gripper_state --
        left_gripper_state = {
            "position": np.asarray(left_ee_position, dtype=np.float64),
            "orientation": np.asarray(left_ee_orientation, dtype=np.float64),
            "linear_velocity": np.asarray(left_ee_linear_velocity, dtype=np.float64),
            "angular_velocity": np.asarray(left_ee_angular_velocity, dtype=np.float64),
        }

        # -- right_gripper_state --
        right_gripper_state = {
            "position": np.asarray(right_ee_position, dtype=np.float64),
            "orientation": np.asarray(right_ee_orientation, dtype=np.float64),
            "linear_velocity": np.asarray(right_ee_linear_velocity, dtype=np.float64),
            "angular_velocity": np.asarray(right_ee_angular_velocity, dtype=np.float64),
        }

        # -- context --
        context = {
            "left_occupied": np.array([left_occupied], dtype=np.float64),
            "right_occupied": np.array([right_occupied], dtype=np.float64),
        }

        # -- 左臂稳定性 proxy（独立差分状态）--
        left_proxy: StabilityProxy = estimate_stability_proxy(
            quat_ee=np.asarray(left_ee_orientation, dtype=np.float64),
            linear_velocity=np.asarray(left_ee_linear_velocity, dtype=np.float64),
            angular_velocity=np.asarray(left_ee_angular_velocity, dtype=np.float64),
            dt=self._dt,
            state=self._left_stability_state,
            acc_history_length=self._acc_history_length,
        )

        # -- 右臂稳定性 proxy（独立差分状态）--
        right_proxy: StabilityProxy = estimate_stability_proxy(
            quat_ee=np.asarray(right_ee_orientation, dtype=np.float64),
            linear_velocity=np.asarray(right_ee_linear_velocity, dtype=np.float64),
            angular_velocity=np.asarray(right_ee_angular_velocity, dtype=np.float64),
            dt=self._dt,
            state=self._right_stability_state,
            acc_history_length=self._acc_history_length,
        )

        # -- door_embedding（冻结 Point-MAE 编码，来自 door_perception/）--
        if door_embedding is not None:
            emb = np.asarray(door_embedding, dtype=np.float64)
        else:
            emb = np.zeros(DOOR_EMBEDDING_DIM, dtype=np.float64)

        return {
            "proprio": proprio,
            "left_gripper_state": left_gripper_state,
            "right_gripper_state": right_gripper_state,
            "context": context,
            "left_stability_proxy": left_proxy.to_dict(),
            "right_stability_proxy": right_proxy.to_dict(),
            "door_embedding": emb,
        }
