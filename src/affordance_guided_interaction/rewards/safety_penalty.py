"""安全惩罚函数。

实现 README §6 的 5 个安全子项：
- §6.1 无效碰撞惩罚
- §6.2 自碰撞惩罚
- §6.3 关节限位逼近惩罚
- §6.4 关节速度过大惩罚
- §6.5 杯体脱落惩罚（触发后 episode 终止）
"""
from __future__ import annotations

import numpy as np


def compute_safety_penalty(
    *,
    contact_forces: dict[str, float],
    affordance_links: set[str],
    self_collision: bool,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_limits: np.ndarray,
    joint_vel_limits: np.ndarray,
    cup_dropped: bool,
    cfg: dict,
) -> tuple[float, bool, dict[str, float]]:
    """计算安全惩罚总和。

    参数:
        contact_forces: link 名称到接触合力大小的映射。
        affordance_links: 有效 affordance 区域内的 link 名称集合。
        self_collision: 是否检测到自碰撞。
        joint_pos: (12,) 双臂关节角度 (rad)。
        joint_vel: (12,) 双臂关节角速度 (rad/s)。
        joint_limits: (12, 2) 每个关节的 [lower, upper] 极限。
        joint_vel_limits: (12,) 每个关节的最大允许角速度。
        cup_dropped: 是否检测到杯体脱落。
        cfg: 安全权重配置字典。

    返回:
        (total_penalty, should_terminate, info_dict)
        - total_penalty: 标量惩罚值（≥0，调用方在总公式中取减号）
        - should_terminate: 是否应该终止当前 episode
        - info_dict: 各子项分项日志
    """
    beta_collision: float = cfg["beta_collision"]
    beta_self: float = cfg["beta_self"]
    beta_limit: float = cfg["beta_limit"]
    mu: float = cfg["mu"]
    beta_vel: float = cfg["beta_vel"]
    w_drop: float = cfg["w_drop"]

    # ── §6.1 无效碰撞惩罚 ──────────────────────────────────────────────
    # 对未落在有效 affordance 区域内的碰撞力求和
    invalid_force_sum = 0.0
    for link_name, force in contact_forces.items():
        if link_name not in affordance_links:
            invalid_force_sum += force
    r_collision = beta_collision * invalid_force_sum

    # ── §6.2 自碰撞惩罚 ────────────────────────────────────────────────
    r_self = beta_self if self_collision else 0.0

    # ── §6.3 关节限位逼近惩罚 ──────────────────────────────────────────
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

    # ── §6.4 关节速度过大惩罚 ──────────────────────────────────────────
    # 超出 μ·max_vel 的部分施加二次惩罚
    vel_threshold = mu * joint_vel_limits
    vel_excess = np.maximum(0.0, np.abs(joint_vel) - vel_threshold)
    r_vel = beta_vel * float(np.sum(vel_excess ** 2))

    # ── §6.5 杯体脱落惩罚 ──────────────────────────────────────────────
    r_drop = w_drop if cup_dropped else 0.0
    should_terminate = cup_dropped

    # ── 汇总 ───────────────────────────────────────────────────────────
    total_penalty = r_collision + r_self + r_limit + r_vel + r_drop

    info = {
        "safety/invalid_collision": r_collision,
        "safety/self_collision": r_self,
        "safety/joint_limit": r_limit,
        "safety/velocity": r_vel,
        "safety/cup_drop": r_drop,
    }

    return total_penalty, should_terminate, info
