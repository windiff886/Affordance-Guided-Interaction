"""从 gripper 速度差分估计稳定性指标。

所有 proxy 均围绕 gripper / EE frame 定义，通过有限差分从连续帧
的速度信号中估计加速度和 jerk，以贴近真实部署中的获取方式。

参考：SoFTA / Hold My Beer 中的末端稳定性奖励设计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .history_buffer import HistoryBuffer


def _norm(v: np.ndarray) -> float:
    """向量 L2 范数，返回标量。"""
    return float(np.linalg.norm(v))


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """四元数 (w, x, y, z) → 3×3 旋转矩阵。"""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def compute_tilt(quat_ee: np.ndarray) -> float:
    """计算 cup tilt-to-gravity proxy。

    将重力方向 ``g = [0, 0, -9.81]`` 投影到 EE 局部坐标系的 xy
    平面上，取投影模长作为 tilt 程度。

    Parameters
    ----------
    quat_ee : (4,) 四元数 (w, x, y, z)
        gripper 在基坐标系下的朝向。

    Returns
    -------
    float
        ``|P_xy(R_EE^T @ g)|``，越大说明杯子倾斜越严重。
    """
    g_world = np.array([0.0, 0.0, -9.81])
    R_ee = _quat_to_rotation_matrix(quat_ee)
    # 重力在 EE 局部坐标系中的表示
    g_local = R_ee.T @ g_world
    # 在 EE 局部 xy 平面上的投影模长
    tilt = float(np.linalg.norm(g_local[:2]))
    return tilt


@dataclass
class StabilityProxyState:
    """在 step 之间维护的流式状态，用于差分估计。"""

    prev_linear_vel: np.ndarray | None = None
    prev_angular_vel: np.ndarray | None = None
    prev_linear_acc: np.ndarray | None = None

    # 最近 k 帧加速度模长历史
    acc_history: HistoryBuffer | None = field(default=None)

    def _ensure_history(self, k: int) -> None:
        if self.acc_history is None:
            self.acc_history = HistoryBuffer(k, fill_value=0.0)


@dataclass
class StabilityProxy:
    """一次 step 输出的全部稳定性 proxy 值。"""

    tilt: float = 0.0
    linear_velocity_norm: float = 0.0
    linear_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    angular_velocity_norm: float = 0.0
    angular_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    jerk_proxy: float = 0.0
    recent_acc_history: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    def to_dict(self) -> dict:
        return {
            "tilt": self.tilt,
            "linear_velocity_norm": self.linear_velocity_norm,
            "linear_acceleration": self.linear_acceleration.copy(),
            "angular_velocity_norm": self.angular_velocity_norm,
            "angular_acceleration": self.angular_acceleration.copy(),
            "jerk_proxy": self.jerk_proxy,
            "recent_acc_history": self.recent_acc_history.copy(),
        }


def estimate_stability_proxy(
    *,
    quat_ee: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    dt: float,
    state: StabilityProxyState,
    acc_history_length: int = 10,
) -> StabilityProxy:
    """从 gripper 当前帧状态和前帧状态差分估计全部稳定性 proxy。

    Parameters
    ----------
    quat_ee : (4,) 四元数 (w, x, y, z)
        gripper 在基坐标系下的朝向。
    linear_velocity : (3,)
        gripper 在基坐标系下的线速度。
    angular_velocity : (3,)
        gripper 在基坐标系下的角速度。
    dt : float
        仿真步长（秒）。
    state : StabilityProxyState
        在 step 之间维护的可变状态对象。函数会原地更新它。
    acc_history_length : int
        加速度历史缓存长度 k。

    Returns
    -------
    StabilityProxy
        本 step 的全部稳定性指标。
    """
    state._ensure_history(acc_history_length)
    v = np.asarray(linear_velocity, dtype=np.float64)
    omega = np.asarray(angular_velocity, dtype=np.float64)

    # -- tilt --
    tilt = compute_tilt(quat_ee)

    # -- velocity norms --
    v_norm = _norm(v)
    omega_norm = _norm(omega)

    # -- linear acceleration (差分) --
    if state.prev_linear_vel is not None and dt > 0:
        lin_acc = (v - state.prev_linear_vel) / dt
    else:
        lin_acc = np.zeros(3)

    # -- angular acceleration (差分) --
    if state.prev_angular_vel is not None and dt > 0:
        ang_acc = (omega - state.prev_angular_vel) / dt
    else:
        ang_acc = np.zeros(3)

    # -- jerk proxy (二阶差分：加速度的变化率) --
    lin_acc_norm = _norm(lin_acc)
    if state.prev_linear_acc is not None and dt > 0:
        jerk = _norm(lin_acc - state.prev_linear_acc) / dt
    else:
        jerk = 0.0

    # -- recent acceleration history --
    assert state.acc_history is not None
    state.acc_history.append(lin_acc_norm)
    recent_acc = state.acc_history.to_numpy()

    # -- 更新流式状态 --
    state.prev_linear_vel = v.copy()
    state.prev_angular_vel = omega.copy()
    state.prev_linear_acc = lin_acc.copy()

    return StabilityProxy(
        tilt=tilt,
        linear_velocity_norm=v_norm,
        linear_acceleration=lin_acc,
        angular_velocity_norm=omega_norm,
        angular_acceleration=ang_acc,
        jerk_proxy=jerk,
        recent_acc_history=recent_acc,
    )
