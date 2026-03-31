"""持杯稳定性奖励函数。

实现 README §5.4 的 7 个子项，按 §5.5 拆分为 bonus 和 penalty 两组返回：
- Bonus (高斯核): 零线加速度奖励、零角加速度奖励
- Penalty (二次形式): 线加速度惩罚、角加速度惩罚、倾斜惩罚、力矩平滑、力矩正则
"""
from __future__ import annotations

import math

import numpy as np


def compute_stability_reward(
    *,
    lin_acc: np.ndarray,
    ang_acc: np.ndarray,
    tilt_xy: np.ndarray,
    torques: np.ndarray,
    prev_torques: np.ndarray,
    cfg: dict,
) -> tuple[float, float, dict[str, float]]:
    """计算单臂的稳定性奖励。

    参数:
        lin_acc: (3,) EE 线加速度向量 (m/s²)。
        ang_acc: (3,) EE 角加速度向量 (rad/s²)。
        tilt_xy: (2,) 重力在 EE 局部坐标系中的 xy 分量。
        torques: (N,) 当前步该臂的力矩输出。
        prev_torques: (N,) 上一步该臂的力矩输出。
        cfg: 稳定性权重配置字典。

    返回:
        (bonus, penalty, info_dict)
        - bonus: 正向激励标量（≥0）
        - penalty: 负向惩罚标量（≤0）
        - info_dict: 各子项分项日志
    """
    # ── 提取权重 ────────────────────────────────────────────────────────
    w_zero_acc: float = cfg["w_zero_acc"]
    lambda_acc: float = cfg["lambda_acc"]
    w_zero_ang: float = cfg["w_zero_ang"]
    lambda_ang: float = cfg["lambda_ang"]
    w_acc: float = cfg["w_acc"]
    w_ang: float = cfg["w_ang"]
    w_tilt: float = cfg["w_tilt"]
    w_smooth: float = cfg["w_smooth"]
    w_reg: float = cfg["w_reg"]

    # ── 预计算范数平方 ──────────────────────────────────────────────────
    lin_acc_sq = float(np.dot(lin_acc, lin_acc))
    ang_acc_sq = float(np.dot(ang_acc, ang_acc))
    tilt_sq = float(np.dot(tilt_xy, tilt_xy))
    torque_diff = torques - prev_torques
    torque_diff_sq = float(np.dot(torque_diff, torque_diff))
    torque_sq = float(np.dot(torques, torques))

    # ═══════════════════════════════════════════════════════════════════
    # (A) 正向激励项 Bonus — 高斯核，鼓励趋近绝对静止
    # ═══════════════════════════════════════════════════════════════════

    # (1) 零线加速度奖励 §5.4(1)
    r_zero_acc = w_zero_acc * math.exp(-lambda_acc * lin_acc_sq)

    # (2) 零角加速度奖励 §5.4(2)
    r_zero_ang = w_zero_ang * math.exp(-lambda_ang * ang_acc_sq)

    bonus = r_zero_acc + r_zero_ang

    # ═══════════════════════════════════════════════════════════════════
    # (B) 负向惩罚项 Penalty — 二次形式，抑制剧烈运动
    # ═══════════════════════════════════════════════════════════════════

    # (3) 线加速度惩罚 §5.4(3)
    r_acc = -w_acc * lin_acc_sq

    # (4) 角加速度惩罚 §5.4(4)
    r_ang = -w_ang * ang_acc_sq

    # (5) 重力倾斜惩罚 §5.4(5)
    r_tilt = -w_tilt * tilt_sq

    # (6) 力矩变化平滑项 §5.4(6)
    r_smooth = -w_smooth * torque_diff_sq

    # (7) 力矩幅值正则项 §5.4(7)
    r_reg = -w_reg * torque_sq

    penalty = r_acc + r_ang + r_tilt + r_smooth + r_reg

    # ── 分项日志 ────────────────────────────────────────────────────────
    info = {
        "zero_acc": r_zero_acc,
        "zero_ang": r_zero_ang,
        "acc": r_acc,
        "ang_acc": r_ang,
        "tilt": r_tilt,
        "smooth": r_smooth,
        "reg": r_reg,
        "bonus": bonus,
        "penalty": penalty,
    }

    return bonus, penalty, info
