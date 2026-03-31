"""observations 层单元测试。

覆盖四个子模块：
- HistoryBuffer      — FIFO 缓存行为
- stability_proxy    — 差分估计与 tilt 计算
- ActorObsBuilder    — 有状态 actor 观测构建
- CriticObsBuilder   — 无状态 critic 观测构建
"""

from pathlib import Path
import sys
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from numpy.testing import assert_allclose


# ======================================================================
# HistoryBuffer
# ======================================================================


class TestHistoryBuffer:
    """HistoryBuffer 基本行为测试。"""

    def test_keeps_recent_items(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )

        buf = HistoryBuffer(max_length=2)
        buf.append("first")
        buf.append("second")
        buf.append("third")
        assert buf.items() == ["second", "third"]

    def test_fill_value_initializes_full(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )

        buf = HistoryBuffer(max_length=3, fill_value=0.0)
        assert len(buf) == 3
        assert buf.is_full
        assert buf.items() == [0.0, 0.0, 0.0]

    def test_latest_returns_subset(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )

        buf = HistoryBuffer(max_length=5)
        for i in range(5):
            buf.append(i)
        assert buf.latest(2) == [3, 4]

    def test_last_property(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )

        buf = HistoryBuffer(max_length=3)
        assert buf.last is None
        buf.append(42)
        assert buf.last == 42

    def test_reset_clears_buffer(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )

        buf = HistoryBuffer(max_length=3, fill_value=1.0)
        assert len(buf) == 3
        buf.reset()
        assert len(buf) == 0
        buf.reset(fill_value=2.0)
        assert len(buf) == 3
        assert buf.items() == [2.0, 2.0, 2.0]

    def test_to_numpy_stacks_arrays(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )

        buf = HistoryBuffer(max_length=3, fill_value=np.zeros(2))
        buf.append(np.array([1.0, 2.0]))
        result = buf.to_numpy()
        assert result.shape == (3, 2)
        assert_allclose(result[-1], [1.0, 2.0])

    def test_invalid_max_length_raises(self) -> None:
        from affordance_guided_interaction.observations.history_buffer import (
            HistoryBuffer,
        )
        import pytest

        with pytest.raises(ValueError):
            HistoryBuffer(max_length=0)


# ======================================================================
# StabilityProxy
# ======================================================================


class TestStabilityProxy:
    """稳定性 proxy 估计测试。"""

    def test_tilt_zero_when_upright(self) -> None:
        """gripper 朝向与世界坐标系对齐时，tilt 应接近 0。"""
        from affordance_guided_interaction.observations.stability_proxy import (
            compute_tilt,
        )

        # 单位四元数 = 无旋转 => gripper z 轴与世界 z 轴平行
        quat_identity = np.array([1.0, 0.0, 0.0, 0.0])
        tilt = compute_tilt(quat_identity)
        # 重力 g=[0,0,-9.81]，R^T@g = g，投影到 xy 平面 = [0,0] => tilt=0
        assert_allclose(tilt, 0.0, atol=1e-10)

    def test_tilt_nonzero_when_tilted(self) -> None:
        """gripper 绕 x 轴旋转 45° 时，tilt 应显著 > 0。"""
        from affordance_guided_interaction.observations.stability_proxy import (
            compute_tilt,
        )

        # 绕 x 轴旋转 45°: quat = [cos(22.5°), sin(22.5°), 0, 0]
        angle = math.radians(45)
        quat_tilted = np.array(
            [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]
        )
        tilt = compute_tilt(quat_tilted)
        assert tilt > 1.0  # 9.81 * sin(45°) ≈ 6.94

    def test_estimate_returns_zero_on_first_step(self) -> None:
        """第一步没有前帧，加速度和 jerk 应为 0。"""
        from affordance_guided_interaction.observations.stability_proxy import (
            StabilityProxyState,
            estimate_stability_proxy,
        )

        state = StabilityProxyState()
        proxy = estimate_stability_proxy(
            quat_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            dt=1 / 60,
            state=state,
        )
        assert_allclose(proxy.linear_acceleration, np.zeros(3))
        assert proxy.jerk_proxy == 0.0

    def test_estimate_accumulates_over_steps(self) -> None:
        """多步运行后，差分估计应正确反映速度变化。"""
        from affordance_guided_interaction.observations.stability_proxy import (
            StabilityProxyState,
            estimate_stability_proxy,
        )

        state = StabilityProxyState()
        dt = 1 / 60

        # 第一步：速度 = 0
        estimate_stability_proxy(
            quat_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            dt=dt,
            state=state,
        )

        # 第二步：线速度 x 方向突变为 1.0
        proxy2 = estimate_stability_proxy(
            quat_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            dt=dt,
            state=state,
        )
        expected_acc = np.array([1.0, 0.0, 0.0]) / dt
        assert_allclose(proxy2.linear_acceleration, expected_acc)
        assert proxy2.linear_velocity_norm > 0

    def test_recent_acc_history_length(self) -> None:
        """加速度历史缓存应按配置长度维护。"""
        from affordance_guided_interaction.observations.stability_proxy import (
            StabilityProxyState,
            estimate_stability_proxy,
        )

        state = StabilityProxyState()
        k = 5
        for _ in range(10):
            proxy = estimate_stability_proxy(
                quat_ee=np.array([1.0, 0.0, 0.0, 0.0]),
                linear_velocity=np.random.randn(3) * 0.1,
                angular_velocity=np.zeros(3),
                dt=1 / 60,
                state=state,
                acc_history_length=k,
            )
        assert proxy.recent_acc_history.shape == (k,)

    def test_to_dict_contains_all_keys(self) -> None:
        """StabilityProxy.to_dict() 应包含 README §4 中规定的所有键。"""
        from affordance_guided_interaction.observations.stability_proxy import (
            StabilityProxy,
        )

        proxy = StabilityProxy()
        d = proxy.to_dict()
        expected_keys = {
            "tilt",
            "linear_velocity_norm",
            "linear_acceleration",
            "angular_velocity_norm",
            "angular_acceleration",
            "jerk_proxy",
            "recent_acc_history",
        }
        assert set(d.keys()) == expected_keys


# ======================================================================
# ActorObsBuilder
# ======================================================================


class TestActorObsBuilder:
    """ActorObsBuilder 集成测试。"""

    def _make_builder(self, **kwargs):
        from affordance_guided_interaction.observations.actor_obs_builder import (
            ActorObsBuilder,
        )

        return ActorObsBuilder(**kwargs)

    def _default_inputs(self, **overrides):
        defaults = dict(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            ee_position=np.array([0.3, 0.0, 0.5]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            ee_linear_velocity=np.zeros(3),
            ee_angular_velocity=np.zeros(3),
            occupied=1.0,
            stability_level=0.0,
        )
        defaults.update(overrides)
        return defaults

    def test_build_returns_all_top_level_keys(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        obs = builder.build(**self._default_inputs())
        expected = {"proprio", "gripper_state", "context", "stability_proxy", "door_point_cloud"}
        assert set(obs.keys()) == expected

    def test_proprio_contains_required_fields(self) -> None:
        builder = self._make_builder(action_history_length=3, dt=1 / 60)
        obs = builder.build(**self._default_inputs())
        proprio = obs["proprio"]
        assert "joint_positions" in proprio
        assert "joint_velocities" in proprio
        assert "previous_actions" in proprio
        assert proprio["joint_positions"].shape == (6,)
        assert proprio["previous_actions"].shape == (3, 6)

    def test_joint_torques_optional(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        # 不传 torques
        obs1 = builder.build(**self._default_inputs())
        assert "joint_torques" not in obs1["proprio"]
        # 传 torques
        builder.reset()
        obs2 = builder.build(**self._default_inputs(joint_torques=np.ones(6)))
        assert "joint_torques" in obs2["proprio"]
        assert_allclose(obs2["proprio"]["joint_torques"], np.ones(6))

    def test_gripper_state_shape(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        obs = builder.build(**self._default_inputs())
        gs = obs["gripper_state"]
        assert gs["position"].shape == (3,)
        assert gs["orientation"].shape == (4,)
        assert gs["linear_velocity"].shape == (3,)
        assert gs["angular_velocity"].shape == (3,)

    def test_context_values(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        obs = builder.build(**self._default_inputs(occupied=1.0, stability_level=2.0))
        assert_allclose(obs["context"]["occupied"], [1.0])
        assert_allclose(obs["context"]["stability_level"], [2.0])

    def test_door_point_cloud_default_empty(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        obs = builder.build(**self._default_inputs())
        assert obs["door_point_cloud"].shape == (0, 3)

    def test_door_point_cloud_passed_through(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        pc = np.random.randn(50, 3)
        obs = builder.build(**self._default_inputs(door_point_cloud=pc))
        assert obs["door_point_cloud"].shape == (50, 3)
        assert_allclose(obs["door_point_cloud"], pc)

    def test_action_history_records(self) -> None:
        builder = self._make_builder(action_history_length=2, dt=1 / 60)
        act1 = np.ones(6)
        act2 = np.ones(6) * 2
        builder.build(**self._default_inputs(action_taken=act1))
        obs = builder.build(**self._default_inputs(action_taken=act2))
        prev_actions = obs["proprio"]["previous_actions"]
        assert prev_actions.shape == (2, 6)
        assert_allclose(prev_actions[0], act1)
        assert_allclose(prev_actions[1], act2)

    def test_reset_clears_state(self) -> None:
        builder = self._make_builder(action_history_length=2, dt=1 / 60)
        builder.build(**self._default_inputs(action_taken=np.ones(6) * 5))
        builder.reset()
        obs = builder.build(**self._default_inputs())
        # reset 后动作历史应全零
        assert_allclose(obs["proprio"]["previous_actions"], np.zeros((2, 6)))

    def test_stability_proxy_in_output(self) -> None:
        builder = self._make_builder(dt=1 / 60)
        obs = builder.build(**self._default_inputs())
        sp = obs["stability_proxy"]
        assert "tilt" in sp
        assert "jerk_proxy" in sp
        assert "recent_acc_history" in sp


# ======================================================================
# CriticObsBuilder
# ======================================================================


class TestCriticObsBuilder:
    """CriticObsBuilder 测试。"""

    def test_wraps_actor_obs(self) -> None:
        from affordance_guided_interaction.observations.critic_obs_builder import (
            CriticObsBuilder,
        )

        actor_obs = {"context": {"occupied": np.array([0.0])}}
        critic_obs = CriticObsBuilder.build(actor_obs=actor_obs)
        assert critic_obs["actor_obs"] is actor_obs

    def test_privileged_contains_all_keys(self) -> None:
        from affordance_guided_interaction.observations.critic_obs_builder import (
            CriticObsBuilder,
        )

        critic_obs = CriticObsBuilder.build(
            actor_obs={},
            door_pose=np.zeros(7),
            door_joint_pos=0.1,
            door_joint_vel=0.05,
            cup_pose=np.zeros(7),
            cup_linear_vel=np.zeros(3),
            cup_angular_vel=np.zeros(3),
            cup_mass=0.3,
            cup_fill_ratio=0.5,
            door_mass=10.0,
            door_damping=5.0,
        )
        priv = critic_obs["privileged"]
        expected_keys = {
            "door_pose",
            "door_joint_pos",
            "door_joint_vel",
            "cup_pose",
            "cup_linear_vel",
            "cup_angular_vel",
            "cup_mass",
            "cup_fill_ratio",
            "door_mass",
            "door_damping",
        }
        assert set(priv.keys()) == expected_keys

    def test_privileged_shapes(self) -> None:
        from affordance_guided_interaction.observations.critic_obs_builder import (
            CriticObsBuilder,
        )

        critic_obs = CriticObsBuilder.build(
            actor_obs={},
            door_pose=np.ones(7),
            cup_pose=np.ones(7),
            cup_linear_vel=np.ones(3),
            cup_angular_vel=np.ones(3),
            cup_mass=0.5,
            door_mass=12.0,
        )
        priv = critic_obs["privileged"]
        assert priv["door_pose"].shape == (7,)
        assert priv["cup_pose"].shape == (7,)
        assert priv["cup_linear_vel"].shape == (3,)
        assert priv["cup_angular_vel"].shape == (3,)
        assert priv["cup_mass"].shape == (1,)
        assert priv["door_mass"].shape == (1,)

    def test_defaults_to_zeros_when_none(self) -> None:
        from affordance_guided_interaction.observations.critic_obs_builder import (
            CriticObsBuilder,
        )

        critic_obs = CriticObsBuilder.build(actor_obs={})
        priv = critic_obs["privileged"]
        assert_allclose(priv["door_pose"], np.zeros(7))
        assert_allclose(priv["cup_pose"], np.zeros(7))
        assert_allclose(priv["cup_linear_vel"], np.zeros(3))
        assert_allclose(priv["cup_angular_vel"], np.zeros(3))
        assert_allclose(priv["cup_mass"], [0.0])


# ======================================================================
# Package __init__ exports
# ======================================================================


def test_package_exports_all_public_symbols() -> None:
    """__init__.py 应导出 README 中提到的所有公共符号。"""
    import affordance_guided_interaction.observations as obs_pkg

    expected = [
        "ActorObsBuilder",
        "CriticObsBuilder",
        "HistoryBuffer",
        "StabilityProxy",
        "StabilityProxyState",
        "estimate_stability_proxy",
        "compute_tilt",
        "NUM_ARM_JOINTS",
    ]
    for name in expected:
        assert hasattr(obs_pkg, name), f"Missing export: {name}"
