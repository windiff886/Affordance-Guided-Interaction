"""奖励系统单元测试。

覆盖以下核心场景：
- task_reward: 分段权重、成功 bonus 一次性触发
- stability_reward: 零输入/大输入行为、bonus/penalty 符号
- safety_penalty: 各子项独立验证、杯体脱落终止
- RewardManager: 总公式聚合、mask 掩码、缩放因子退火
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from affordance_guided_interaction.rewards.task_reward import compute_task_reward
from affordance_guided_interaction.rewards.stability_reward import compute_stability_reward
from affordance_guided_interaction.rewards.safety_penalty import compute_safety_penalty
from affordance_guided_interaction.rewards.reward_manager import RewardManager


# ═══════════════════════════════════════════════════════════════════════
# 测试用默认配置
# ═══════════════════════════════════════════════════════════════════════

TASK_CFG = {
    "w_delta": 8.0,
    "alpha": 0.25,
    "k_decay": 2.0,
    "w_open": 10.0,
    "theta_target": 1.57,
}

STAB_CFG = {
    "w_zero_acc": 1.0,
    "lambda_acc": 0.25,
    "w_zero_ang": 1.5,
    "lambda_ang": 0.0044,
    "w_acc": 0.10,
    "w_ang": 0.01,
    "w_tilt": 5.0,
    "w_smooth": 0.01,
    "w_reg": 0.001,
}

SAFE_CFG = {
    "beta_collision": 0.5,
    "beta_self": 1.0,
    "beta_limit": 0.1,
    "mu": 0.9,
    "beta_vel": 0.01,
    "w_drop": 100.0,
}

SCALE_CFG = {
    "s_min": 0.1,
    "n_anneal": 10_000_000,
}

FULL_CFG = {
    "task": TASK_CFG,
    "stability": STAB_CFG,
    "safety": SAFE_CFG,
    "scaling": SCALE_CFG,
}


# ═══════════════════════════════════════════════════════════════════════
# task_reward 测试
# ═══════════════════════════════════════════════════════════════════════

class TestTaskReward:
    """§4.2 推门奖励测试。"""

    def test_positive_progress_before_target(self):
        """目标角度前，正增量应获得满额正奖励。"""
        r, _, info = compute_task_reward(
            theta_t=0.5, theta_prev=0.3, already_succeeded=False, cfg=TASK_CFG,
        )
        # 增量 0.2，权重 w_delta = 8.0 → 1.6
        assert r == pytest.approx(8.0 * 0.2, abs=1e-6)
        assert info["task/weight"] == pytest.approx(8.0)

    def test_weight_decay_after_target(self):
        """超出目标角度后，权重应衰减。"""
        theta_t = TASK_CFG["theta_target"] + 0.1
        _, _, info = compute_task_reward(
            theta_t=theta_t, theta_prev=theta_t - 0.01,
            already_succeeded=True, cfg=TASK_CFG,
        )
        expected_w = 8.0 * max(0.25, 1.0 - 2.0 * 0.1)
        assert info["task/weight"] == pytest.approx(expected_w, abs=1e-6)

    def test_success_bonus_once(self):
        """成功 bonus 应只触发一次。"""
        r1, new1, _ = compute_task_reward(
            theta_t=1.6, theta_prev=1.5, already_succeeded=False, cfg=TASK_CFG,
        )
        assert new1 is True
        assert r1 > 10.0  # 包含 w_open

        r2, new2, _ = compute_task_reward(
            theta_t=1.7, theta_prev=1.6, already_succeeded=True, cfg=TASK_CFG,
        )
        assert new2 is False

    def test_negative_progress(self):
        """门角度回退应产生负奖励。"""
        r, _, _ = compute_task_reward(
            theta_t=0.3, theta_prev=0.5, already_succeeded=False, cfg=TASK_CFG,
        )
        assert r < 0.0


# ═══════════════════════════════════════════════════════════════════════
# stability_reward 测试
# ═══════════════════════════════════════════════════════════════════════

class TestStabilityReward:
    """§5.4 稳定性奖励测试。"""

    def test_zero_input_max_bonus(self):
        """完全静止时，bonus 应达到最大值（高斯核 = 1.0）。"""
        bonus, penalty, info = compute_stability_reward(
            lin_acc=np.zeros(3),
            ang_acc=np.zeros(3),
            tilt_xy=np.zeros(2),
            torques=np.zeros(6),
            prev_torques=np.zeros(6),
            cfg=STAB_CFG,
        )
        # bonus = w_zero_acc * 1.0 + w_zero_ang * 1.0 = 1.0 + 1.5 = 2.5
        assert bonus == pytest.approx(2.5, abs=1e-6)
        # 零输入时 penalty 应为 0
        assert penalty == pytest.approx(0.0, abs=1e-6)

    def test_large_acc_reduces_bonus(self):
        """大加速度应降低 bonus（高斯核衰减）并增大 penalty 绝对值。"""
        bonus, penalty, _ = compute_stability_reward(
            lin_acc=np.array([10.0, 0.0, 0.0]),
            ang_acc=np.zeros(3),
            tilt_xy=np.zeros(2),
            torques=np.zeros(6),
            prev_torques=np.zeros(6),
            cfg=STAB_CFG,
        )
        # exp(-0.25 * 100) ≈ 0，所以 zero_acc 接近 0
        assert bonus < 2.5
        # penalty 应为负
        assert penalty < 0.0

    def test_penalty_always_nonpositive(self):
        """penalty 在任何输入下都应 ≤ 0。"""
        rng = np.random.default_rng(42)
        for _ in range(50):
            _, penalty, _ = compute_stability_reward(
                lin_acc=rng.standard_normal(3) * 5,
                ang_acc=rng.standard_normal(3) * 5,
                tilt_xy=rng.standard_normal(2) * 3,
                torques=rng.standard_normal(6),
                prev_torques=rng.standard_normal(6),
                cfg=STAB_CFG,
            )
            assert penalty <= 0.0 + 1e-12

    def test_bonus_always_nonnegative(self):
        """bonus 在任何输入下都应 ≥ 0。"""
        rng = np.random.default_rng(123)
        for _ in range(50):
            bonus, _, _ = compute_stability_reward(
                lin_acc=rng.standard_normal(3) * 10,
                ang_acc=rng.standard_normal(3) * 10,
                tilt_xy=rng.standard_normal(2),
                torques=rng.standard_normal(6),
                prev_torques=rng.standard_normal(6),
                cfg=STAB_CFG,
            )
            assert bonus >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# safety_penalty 测试
# ═══════════════════════════════════════════════════════════════════════

class TestSafetyPenalty:
    """§6 安全惩罚测试。"""

    def _default_safe_args(self, **overrides):
        """构造安全惩罚的默认参数。"""
        defaults = dict(
            contact_forces={},
            affordance_links=set(),
            self_collision=False,
            joint_pos=np.zeros(12),
            joint_vel=np.zeros(12),
            joint_limits=np.column_stack([np.full(12, -3.14), np.full(12, 3.14)]),
            joint_vel_limits=np.full(12, 10.0),
            cup_dropped=False,
            cfg=SAFE_CFG,
        )
        defaults.update(overrides)
        return defaults

    def test_all_safe_zero_penalty(self):
        """完全安全状态下，惩罚应为 0。"""
        penalty, terminate, _ = compute_safety_penalty(**self._default_safe_args())
        assert penalty == pytest.approx(0.0, abs=1e-6)
        assert terminate is False

    def test_invalid_collision(self):
        """非 affordance 区域碰撞应产生惩罚。"""
        penalty, _, info = compute_safety_penalty(**self._default_safe_args(
            contact_forces={"arm_link3": 5.0, "door_handle": 2.0},
            affordance_links={"door_handle"},
        ))
        # 只惩罚 arm_link3 的 5.0 → β_collision * 5.0 = 2.5
        assert info["safety/invalid_collision"] == pytest.approx(2.5, abs=1e-6)

    def test_self_collision(self):
        """自碰撞应产生固定惩罚。"""
        penalty, _, info = compute_safety_penalty(**self._default_safe_args(
            self_collision=True,
        ))
        assert info["safety/self_collision"] == pytest.approx(1.0, abs=1e-6)

    def test_joint_limit_penalty(self):
        """关节角度逼近极限时应产生惩罚。"""
        # 设置关节极限 [-3.14, 3.14]，center=0，half_range=3.14
        # mu=0.9 → threshold = 2.826
        # 将第0关节设为 3.0 → |3.0 - 0| = 3.0 > 2.826 → excess = 0.174
        joint_pos = np.zeros(12)
        joint_pos[0] = 3.0
        penalty, _, info = compute_safety_penalty(**self._default_safe_args(
            joint_pos=joint_pos,
        ))
        assert info["safety/joint_limit"] > 0.0

    def test_cup_drop_terminates(self):
        """杯体脱落应触发终止并施加大惩罚。"""
        penalty, terminate, info = compute_safety_penalty(**self._default_safe_args(
            cup_dropped=True,
        ))
        assert terminate is True
        assert info["safety/cup_drop"] == pytest.approx(100.0, abs=1e-6)
        assert penalty >= 100.0


# ═══════════════════════════════════════════════════════════════════════
# RewardManager 测试
# ═══════════════════════════════════════════════════════════════════════

class TestRewardManager:
    """§2 总公式 + §7 缩放因子测试。"""

    def test_scaling_factor_initial(self):
        """初始时 s_t 应等于 s_min。"""
        mgr = RewardManager(FULL_CFG)
        assert mgr.compute_scaling_factor() == pytest.approx(0.1, abs=1e-6)

    def test_scaling_factor_saturated(self):
        """超过退火窗口后 s_t 应为 1.0。"""
        mgr = RewardManager(FULL_CFG)
        mgr.global_step = 20_000_000
        assert mgr.compute_scaling_factor() == pytest.approx(1.0, abs=1e-6)

    def test_scaling_factor_midpoint(self):
        """退火中点处 s_t 应在 s_min 和 1.0 之间。"""
        mgr = RewardManager(FULL_CFG)
        mgr.global_step = 5_000_000  # 50% 进度
        s = mgr.compute_scaling_factor()
        # s = 0.1 + 0.9 * 0.5 = 0.55
        assert s == pytest.approx(0.55, abs=1e-6)

    def test_mask_zero_disables_stability(self):
        """mask = 0 时稳定性奖励应彻底归零。"""
        mgr = RewardManager(FULL_CFG)
        _, _, info = mgr.step(
            theta_t=0.5, theta_prev=0.3,
            left_occupied=False, right_occupied=False,
        )
        assert info["stability_total"] == pytest.approx(0.0, abs=1e-6)

    def test_mask_activates_stability(self):
        """mask = 1 时稳定性奖励不为零。"""
        mgr = RewardManager(FULL_CFG)
        proxy = {
            "lin_acc": np.array([1.0, 0.0, 0.0]),
            "ang_acc": np.zeros(3),
            "tilt_xy": np.zeros(2),
        }
        _, _, info = mgr.step(
            theta_t=0.5, theta_prev=0.3,
            left_occupied=True,
            left_stability_proxy=proxy,
        )
        # 有加速度输入时，bonus 下降且 penalty 出现
        assert info["stability_total"] != pytest.approx(0.0, abs=1e-6)

    def test_global_step_increments(self):
        """每次 step 应递增全局步数。"""
        mgr = RewardManager(FULL_CFG)
        assert mgr.global_step == 0
        mgr.step(theta_t=0.1, theta_prev=0.0)
        assert mgr.global_step == 1
        mgr.step(theta_t=0.2, theta_prev=0.1)
        assert mgr.global_step == 2

    def test_reset_episode_preserves_global_step(self):
        """重置 episode 状态不应影响全局步数。"""
        mgr = RewardManager(FULL_CFG)
        mgr.step(theta_t=0.1, theta_prev=0.0)
        mgr.step(theta_t=0.2, theta_prev=0.1)
        mgr.reset_episode()
        assert mgr.global_step == 2

    def test_total_reward_structure(self):
        """总奖励应包含所有必要的日志键。"""
        mgr = RewardManager(FULL_CFG)
        _, _, info = mgr.step(theta_t=0.5, theta_prev=0.3)
        required_keys = [
            "total_reward", "scaling/s_t",
            "stability_total", "safety_total",
            "task/door_angle_delta", "task/success_bonus",
        ]
        for key in required_keys:
            assert key in info, f"缺少日志键: {key}"
