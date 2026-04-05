"""将环境侧状态组装为目标规格 actor observation。"""

from __future__ import annotations

import numpy as np

from .history_buffer import HistoryBuffer
# 每条 Z1 机械臂的关节数
NUM_JOINTS_PER_ARM = 6
# 双臂总关节数（左臂 + 右臂）
TOTAL_ARM_JOINTS = NUM_JOINTS_PER_ARM * 2

# Point-MAE 编码器输出维度（2 × trans_dim = 2 × 384）
DOOR_EMBEDDING_DIM = 768


class ActorObsBuilder:
    """有状态的 actor observation 构建器（双臂平台）。

    持杯是随机事件，左右臂均可能持杯，因此左右臂状态对称提供，
    使用环境侧共享的稳定性 proxy。

    在 episode 生命周期内仅维护上一时刻动作缓存。

    Parameters
    ----------
    action_history_length : int
        保留最近多少步动作，默认 3。
    """

    def __init__(
        self,
        *,
        action_history_length: int = 3,
    ) -> None:
        self._action_history_length = action_history_length

        # 内部状态 —— 在 reset() 中初始化
        self._action_buffer: HistoryBuffer[np.ndarray] | None = None
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
        left_ee_linear_acceleration: np.ndarray,
        left_ee_angular_acceleration: np.ndarray,
        left_stability_proxy: dict,
        right_ee_position: np.ndarray,
        right_ee_orientation: np.ndarray,
        right_ee_linear_velocity: np.ndarray,
        right_ee_angular_velocity: np.ndarray,
        right_ee_linear_acceleration: np.ndarray,
        right_ee_angular_acceleration: np.ndarray,
        right_stability_proxy: dict,
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
        left_ee_linear_acceleration : (3,)
            左臂 gripper 在基坐标系下的线加速度。
        left_ee_angular_acceleration : (3,)
            左臂 gripper 在基坐标系下的角加速度。
        right_ee_position : (3,)
            右臂 gripper 在基坐标系下的位置。
        right_ee_orientation : (4,)
            右臂 gripper 在基坐标系下的四元数 (w, x, y, z)。
        right_ee_linear_velocity : (3,)
            右臂 gripper 在基坐标系下的线速度。
        right_ee_angular_velocity : (3,)
            右臂 gripper 在基坐标系下的角速度。
        right_ee_linear_acceleration : (3,)
            右臂 gripper 在基坐标系下的线加速度。
        right_ee_angular_acceleration : (3,)
            右臂 gripper 在基坐标系下的角加速度。
        left_stability_proxy : dict
            由环境层统一生成的左臂稳定性 proxy。
        right_stability_proxy : dict
            由环境层统一生成的右臂稳定性 proxy。
        left_occupied : float
            左臂是否持杯：0.0 = 空闲, 1.0 = 持杯。
        right_occupied : float
            右臂是否持杯：0.0 = 空闲, 1.0 = 持杯。
        door_embedding : (768,) | None
            来自 door_perception/ 的冻结 Point-MAE 编码结果。
            None 时填充零向量。
        action_taken : (12,) | None
            本 step 策略输出的双臂完整动作（左臂 6 + 右臂 6）；
            如果提供，会被记录为下一时刻的 ``prev_action``。
            通常在 env.step() 之后、下一次 build() 之前传入。

        Returns
        -------
        dict
            完整的 actor observation 字典。
        """
        assert self._action_buffer is not None

        # 记录动作
        if action_taken is not None:
            self._action_buffer.append(np.asarray(action_taken, dtype=np.float64))

        prev_action = self._action_buffer.last
        if prev_action is None:
            prev_action = np.zeros(TOTAL_ARM_JOINTS, dtype=np.float64)

        # -- proprio --
        joint_positions = np.concatenate([
            np.asarray(left_joint_positions, dtype=np.float64).ravel(),
            np.asarray(right_joint_positions, dtype=np.float64).ravel(),
        ])
        joint_velocities = np.concatenate([
            np.asarray(left_joint_velocities, dtype=np.float64).ravel(),
            np.asarray(right_joint_velocities, dtype=np.float64).ravel(),
        ])
        joint_torques = np.concatenate([
            np.asarray(
                left_joint_torques
                if left_joint_torques is not None
                else np.zeros(NUM_JOINTS_PER_ARM, dtype=np.float64),
                dtype=np.float64,
            ).ravel(),
            np.asarray(
                right_joint_torques
                if right_joint_torques is not None
                else np.zeros(NUM_JOINTS_PER_ARM, dtype=np.float64),
                dtype=np.float64,
            ).ravel(),
        ])

        proprio: dict[str, np.ndarray] = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "prev_action": np.asarray(prev_action, dtype=np.float64).ravel(),
        }

        # -- ee --
        ee = {
            "left": {
                "position": np.asarray(left_ee_position, dtype=np.float64),
                "orientation": np.asarray(left_ee_orientation, dtype=np.float64),
                "linear_velocity": np.asarray(left_ee_linear_velocity, dtype=np.float64),
                "angular_velocity": np.asarray(left_ee_angular_velocity, dtype=np.float64),
                "linear_acceleration": np.asarray(
                    left_ee_linear_acceleration, dtype=np.float64
                ),
                "angular_acceleration": np.asarray(
                    left_ee_angular_acceleration, dtype=np.float64
                ),
            },
            "right": {
                "position": np.asarray(right_ee_position, dtype=np.float64),
                "orientation": np.asarray(right_ee_orientation, dtype=np.float64),
                "linear_velocity": np.asarray(right_ee_linear_velocity, dtype=np.float64),
                "angular_velocity": np.asarray(right_ee_angular_velocity, dtype=np.float64),
                "linear_acceleration": np.asarray(
                    right_ee_linear_acceleration, dtype=np.float64
                ),
                "angular_acceleration": np.asarray(
                    right_ee_angular_acceleration, dtype=np.float64
                ),
            },
        }

        # -- context --
        context = {
            "left_occupied": np.array([left_occupied], dtype=np.float64),
            "right_occupied": np.array([right_occupied], dtype=np.float64),
        }

        # -- door_embedding（冻结 Point-MAE 编码，来自 door_perception/）--
        if door_embedding is not None:
            emb = np.asarray(door_embedding, dtype=np.float64)
        else:
            emb = np.zeros(DOOR_EMBEDDING_DIM, dtype=np.float64)

        return {
            "proprio": proprio,
            "ee": ee,
            "context": context,
            "stability": {
                "left_tilt": np.array(
                    [_extract_tilt(left_stability_proxy)], dtype=np.float64
                ),
                "right_tilt": np.array(
                    [_extract_tilt(right_stability_proxy)], dtype=np.float64
                ),
            },
            "visual": {
                "door_embedding": emb,
            },
        }


def _extract_tilt(proxy: dict) -> float:
    """从环境侧稳定性 proxy 中提取单值 tilt。"""
    return float(proxy["tilt"])
