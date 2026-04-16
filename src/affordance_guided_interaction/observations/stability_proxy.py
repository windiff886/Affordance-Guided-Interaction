"""环境侧共享稳定性 proxy 构建器。

所有 proxy 均围绕 gripper / EE frame 定义。线加速度与角加速度
应由环境层直接读取 Isaac Sim / Isaac Lab 的原生刚体 / link
加速度接口；本模块负责在此基础上统一补齐 tilt、jerk 和最近
加速度历史，供 observations 与 rewards 并列消费。
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


def compute_tilt_xy(quat_ee: np.ndarray) -> np.ndarray:
    """计算重力在 EE 局部坐标系中偏离 Y 轴的投影（xz 分量）。

    抓取姿态下 joint6 = ±90° 使 EE Y 轴对齐世界竖直方向，
    因此取 xz 分量度量杯子偏离竖直的程度。
    """
    g_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    r_ee = _quat_to_rotation_matrix(np.asarray(quat_ee, dtype=np.float64))
    g_local = r_ee.T @ g_world
    return g_local[[0, 2]].astype(np.float64)


def compute_tilt(quat_ee: np.ndarray) -> float:
    """计算 cup tilt-to-gravity proxy。"""
    return float(np.linalg.norm(compute_tilt_xy(quat_ee)))


@dataclass
class StabilityProxyState:
    """在 step 之间维护的流式状态，用于 jerk / 历史窗口。"""

    prev_linear_acc: np.ndarray | None = None
    acc_history: HistoryBuffer | None = field(default=None)

    def _ensure_history(self, k: int) -> None:
        if self.acc_history is None:
            self.acc_history = HistoryBuffer(k, fill_value=0.0)


@dataclass
class StabilityProxy:
    """一次 step 输出的全部稳定性 proxy 值。"""

    tilt: float = 0.0
    tilt_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    linear_velocity_norm: float = 0.0
    linear_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    angular_velocity_norm: float = 0.0
    angular_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    jerk_proxy: float = 0.0
    recent_acc_history: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    def to_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "tilt": self.tilt,
            "tilt_xy": self.tilt_xy.copy(),
            "linear_velocity_norm": self.linear_velocity_norm,
            "linear_acceleration": self.linear_acceleration.copy(),
            "angular_velocity_norm": self.angular_velocity_norm,
            "angular_acceleration": self.angular_acceleration.copy(),
            "jerk_proxy": self.jerk_proxy,
            "recent_acc_history": self.recent_acc_history.copy(),
        }


def build_stability_proxy(
    *,
    quat_ee: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    linear_acceleration: np.ndarray,
    angular_acceleration: np.ndarray,
    dt: float,
    state: StabilityProxyState,
    acc_history_length: int = 10,
) -> StabilityProxy:
    """基于环境侧原生加速度与姿态构建共享稳定性 proxy。"""
    state._ensure_history(acc_history_length)

    v = np.asarray(linear_velocity, dtype=np.float64)
    omega = np.asarray(angular_velocity, dtype=np.float64)
    lin_acc = np.asarray(linear_acceleration, dtype=np.float64)
    ang_acc = np.asarray(angular_acceleration, dtype=np.float64)

    tilt_xy = compute_tilt_xy(quat_ee)
    tilt = _norm(tilt_xy)
    v_norm = _norm(v)
    omega_norm = _norm(omega)
    lin_acc_norm = _norm(lin_acc)

    if state.prev_linear_acc is not None and dt > 0.0:
        jerk = _norm(lin_acc - state.prev_linear_acc) / dt
    else:
        jerk = 0.0

    assert state.acc_history is not None
    state.acc_history.append(lin_acc_norm)
    recent_acc = state.acc_history.to_numpy()
    state.prev_linear_acc = lin_acc.copy()

    return StabilityProxy(
        tilt=tilt,
        tilt_xy=tilt_xy,
        linear_velocity_norm=v_norm,
        linear_acceleration=lin_acc,
        angular_velocity_norm=omega_norm,
        angular_acceleration=ang_acc,
        jerk_proxy=jerk,
        recent_acc_history=recent_acc,
    )
