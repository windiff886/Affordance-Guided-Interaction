"""主任务奖励函数。

实现 README §4.2 的推门奖励：
- 分段权重函数 w(θ)：目标角度前满额，超过后线性衰减
- 成功 bonus：到达目标角度时的一次性奖励
"""
from __future__ import annotations

import math


def compute_task_reward(
    *,
    theta_t: float,
    theta_prev: float,
    already_succeeded: bool,
    cfg: dict,
) -> tuple[float, bool, dict[str, float]]:
    """计算推门任务奖励。

    参数:
        theta_t: 当前步门铰链角度 (rad)。
        theta_prev: 上一步门铰链角度 (rad)。
        already_succeeded: 是否已经在之前的步骤中触发过成功 bonus。
        cfg: 任务奖励配置字典，包含以下键：
            - w_delta: 角度增量基准奖励权重
            - alpha: 超出目标后的权重衰减下限比例
            - k_decay: 超出目标角度后的衰减速率系数
            - w_open: 任务成功一次性 bonus
            - theta_target: 目标门角度 (rad)

    返回:
        (r_task, newly_succeeded, info_dict)
        - r_task: 标量任务奖励
        - newly_succeeded: 本步是否首次触发成功
        - info_dict: 分项日志
    """
    w_delta: float = cfg["w_delta"]
    alpha: float = cfg["alpha"]
    k_decay: float = cfg["k_decay"]
    w_open: float = cfg["w_open"]
    theta_target: float = cfg["theta_target"]

    # ── §4.2 分段权重函数 w(θ) ──────────────────────────────────────────
    if theta_t <= theta_target:
        # 目标角度前：满额激励，不衰减
        weight = w_delta
    else:
        # 超出目标后：线性衰减，但不低于 α·w_δ
        weight = w_delta * max(alpha, 1.0 - k_decay * (theta_t - theta_target))

    # 角度增量奖励（稠密信号）
    delta = theta_t - theta_prev
    r_progress = weight * delta

    # ── 成功 bonus（一次性） ────────────────────────────────────────────
    newly_succeeded = False
    r_success = 0.0
    if not already_succeeded and theta_t >= theta_target:
        r_success = w_open
        newly_succeeded = True

    r_task = r_progress + r_success

    info = {
        "task/door_angle_delta": r_progress,
        "task/success_bonus": r_success,
        "task/weight": weight,
        "task/theta_t": theta_t,
    }
    return r_task, newly_succeeded, info
