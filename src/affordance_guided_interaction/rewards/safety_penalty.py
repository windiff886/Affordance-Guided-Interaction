"""安全惩罚函数。

实现 README §6 的 5 个安全子项：
- §6.1 自碰撞惩罚
- §6.2 关节限位逼近惩罚
- §6.3 关节速度过大惩罚
- §6.4 原始控制力矩超限惩罚
- §6.5 杯体脱落惩罚（触发后 episode 终止）
"""
from __future__ import annotations

import numpy as np


def compute_safety_penalty(
    *,
    self_collision: bool,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_limits: np.ndarray,
    joint_vel_limits: np.ndarray,
    policy_torques: np.ndarray | None,
    torque_limits: np.ndarray | None,
    cup_dropped: bool,
    cfg: dict,
) -> tuple[float, bool, dict[str, float]]:
    """计算安全惩罚总和。

    参数:
        self_collision: 是否检测到自碰撞。
        joint_pos: (12,) 双臂关节角度 (rad)。
        joint_vel: (12,) 双臂关节角速度 (rad/s)。
        joint_limits: (12, 2) 每个关节的 [lower, upper] 极限。
        joint_vel_limits: (12,) 每个关节的最大允许角速度。
        policy_torques: (12,) policy 原始输出的控制力矩（clip 前）。
        torque_limits: (12,) 每个关节的力矩上限。
        cup_dropped: 是否检测到杯体脱落。
        cfg: 安全权重配置字典。

    返回:
        (total_penalty, should_terminate, info_dict)
        - total_penalty: 标量惩罚值（≥0，调用方在总公式中取减号）
        - should_terminate: 是否应该终止当前 episode
        - info_dict: 各子项分项日志
    """
    beta_self: float = cfg["beta_self"]
    beta_limit: float = cfg["beta_limit"]
    mu: float = cfg["mu"]
    beta_vel: float = cfg["beta_vel"]
    beta_torque: float = cfg["beta_torque"]
    w_drop: float = cfg["w_drop"]

    # ── §6.1 自碰撞惩罚 ────────────────────────────────────────────────
    r_self = beta_self if self_collision else 0.0

    # ── §6.2 关节限位逼近惩罚 ──────────────────────────────────────────
    # 计算每个关节的中心与半范围，超出 μ 比例的部分施加二次惩罚
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    center = (upper + lower) / 2.0
    half_range = (upper - lower) / 2.0

    # 避免除零：如果某关节无范围（fixed joint），其 half_range 为 0，跳过
    deviation = np.abs(joint_pos - center)
    threshold = mu * half_range
    excess = np.maximum(0.0, deviation - threshold)
    r_limit = beta_limit * float(np.sum(excess ** 2))

    # ── §6.3 关节速度过大惩罚 ──────────────────────────────────────────
    # 超出 μ·max_vel 的部分施加二次惩罚
    vel_threshold = mu * joint_vel_limits
    vel_excess = np.maximum(0.0, np.abs(joint_vel) - vel_threshold)
    r_vel = beta_vel * float(np.sum(vel_excess ** 2))

    # ── §6.4 原始控制力矩超限惩罚 ──────────────────────────────────────
    if policy_torques is None or torque_limits is None:
        r_torque = 0.0
    else:
        torque_excess = np.maximum(
            0.0,
            np.abs(policy_torques) - torque_limits,
        )
        r_torque = beta_torque * float(np.sum(torque_excess ** 2))

    # ── §6.5 杯体脱落惩罚 ──────────────────────────────────────────────
    r_drop = w_drop if cup_dropped else 0.0
    should_terminate = cup_dropped

    # ── 汇总 ───────────────────────────────────────────────────────────
    total_penalty = r_self + r_limit + r_vel + r_torque + r_drop

    info = {
        "safety/self_collision": r_self,
        "safety/joint_limit": r_limit,
        "safety/velocity": r_vel,
        "safety/torque_over_limit": r_torque,
        "safety/cup_drop": r_drop,
    }

    return total_penalty, should_terminate, info
