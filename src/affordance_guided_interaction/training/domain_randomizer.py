"""域随机化器 — 回合级物理参数采样与步级动态噪声注入。

实现 training/README.md §5.2 中定义的两类随机化机制：

1. **回合级静态参数**（Episode-level）：
   每次 episode reset 时从均匀分布 U[a, b] 中采样，局内保持不变。
   包含 cup_mass, door_mass, door_damping, base_pos。

2. **步级动态噪声**（Step-level）：
   每执行一次 step() 时从高斯分布 N(0, σ²I) 中独立采样。
   包含 action_noise (ε_a) 和 observation_noise (ε_o)。

本模块只负责"摇骰子"，不直接操作仿真环境——返回纯字典/数组，
由上游调用方（RolloutCollector / envs 层）负责将参数应用到仿真中。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomizationConfig:
    """域随机化超参配置。

    默认值对齐 training/README.md §7.4。
    """

    # ── 回合级静态参数范围 ──────────────────────────────────────────
    cup_mass_range: tuple[float, float] = (0.1, 0.8)
    door_mass_range: tuple[float, float] = (5.0, 20.0)
    door_damping_range: tuple[float, float] = (0.5, 5.0)

    # 门外侧推板附近的基座采样几何
    push_plate_center_xy: tuple[float, float] = (2.98, 0.27)
    base_reference_xy: tuple[float, float] = (3.72, 0.27)
    base_height: float = 0.12
    base_radius_range: tuple[float, float] = (0.45, 0.60)
    base_sector_half_angle_deg: float = 20.0
    base_yaw_delta_deg: float = 10.0

    # ── 步级动态噪声标准差 ────────────────────────────────────────
    action_noise_std: float = 0.02    # σ_a
    observation_noise_std: float = 0.01  # σ_o


class DomainRandomizer:
    """域随机化器。

    Parameters
    ----------
    cfg : RandomizationConfig | None
        随机化超参。若为 None 则使用默认值。
    seed : int | None
        随机数种子，便于复现。
    """

    def __init__(
        self,
        cfg: RandomizationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg or RandomizationConfig()
        self._rng = np.random.default_rng(seed)

    # ═══════════════════════════════════════════════════════════════════
    # 回合级采样（Episode-level）
    # ═══════════════════════════════════════════════════════════════════

    def sample_episode_params(self) -> dict[str, float | np.ndarray]:
        """每次 episode reset 时调用，采样一组局内固定的物理参数。

        Returns
        -------
        dict 包含：
            - ``cup_mass``     : float
            - ``door_mass``    : float
            - ``door_damping`` : float
            - ``base_pos``     : (3,) ndarray — 世界坐标系下基座位置
            - ``base_yaw``     : float — 世界坐标系下基座 yaw（rad）
        """
        c = self.cfg

        # 质量与阻尼：U[min, max]
        cup_mass = float(self._rng.uniform(*c.cup_mass_range))
        door_mass = float(self._rng.uniform(*c.door_mass_range))
        door_damping = float(self._rng.uniform(*c.door_damping_range))

        base_pos = self._sample_base_position()
        base_yaw = self._sample_base_yaw(base_pos[:2])

        return {
            "cup_mass": cup_mass,
            "door_mass": door_mass,
            "door_damping": door_damping,
            "base_pos": base_pos,
            "base_yaw": base_yaw,
        }

    def _sample_base_position(self) -> np.ndarray:
        radius = self._sample_base_radius()
        angle = self._sample_base_sector_angle()
        center = np.asarray(self.cfg.push_plate_center_xy, dtype=np.float64)

        base_xy = center + radius * np.array(
            [math.cos(angle), math.sin(angle)],
            dtype=np.float64,
        )
        return np.array(
            [base_xy[0], base_xy[1], self.cfg.base_height],
            dtype=np.float64,
        )

    def _sample_base_radius(self) -> float:
        inner_radius, outer_radius = self.cfg.base_radius_range
        return float(self._rng.uniform(inner_radius, outer_radius))

    def _sample_base_sector_angle(self) -> float:
        center = np.asarray(self.cfg.push_plate_center_xy, dtype=np.float64)
        reference = np.asarray(self.cfg.base_reference_xy, dtype=np.float64)
        nominal_angle = math.atan2(reference[1] - center[1], reference[0] - center[0])
        sector_half_angle = math.radians(self.cfg.base_sector_half_angle_deg)
        return float(self._rng.uniform(
            nominal_angle - sector_half_angle,
            nominal_angle + sector_half_angle,
        ))

    def _sample_base_yaw(self, base_xy: np.ndarray) -> float:
        push_center = np.asarray(self.cfg.push_plate_center_xy, dtype=np.float64)
        nominal_yaw = math.atan2(
            push_center[1] - float(base_xy[1]),
            push_center[0] - float(base_xy[0]),
        )
        yaw_delta = math.radians(self.cfg.base_yaw_delta_deg)
        return float(self._rng.uniform(
            nominal_yaw - yaw_delta,
            nominal_yaw + yaw_delta,
        ))

    # ═══════════════════════════════════════════════════════════════════
    # 步级噪声注入（Step-level）
    # ═══════════════════════════════════════════════════════════════════

    def sample_action_noise(self, action_dim: int = 12) -> np.ndarray:
        """每步 step() 时调用，生成动作噪声 ε_a ~ N(0, σ_a² I)。

        Parameters
        ----------
        action_dim : int
            动作空间维度，默认 12（双臂各 6 关节）。

        Returns
        -------
        (action_dim,) ndarray
        """
        return self._rng.normal(0.0, self.cfg.action_noise_std, size=action_dim)

    def sample_observation_noise(self, obs_dim: int) -> np.ndarray:
        """每步 step() 时调用，生成观测噪声 ε_o ~ N(0, σ_o² I)。

        Parameters
        ----------
        obs_dim : int
            观测空间维度（与关节编码器数量一致）。

        Returns
        -------
        (obs_dim,) ndarray
        """
        return self._rng.normal(0.0, self.cfg.observation_noise_std, size=obs_dim)

    # ═══════════════════════════════════════════════════════════════════
    # 批量采样（多环境并行）
    # ═══════════════════════════════════════════════════════════════════

    def sample_batch_episode_params(
        self, n_envs: int
    ) -> list[dict[str, float | np.ndarray]]:
        """为 n_envs 个并行环境各自独立采样一套回合级参数。"""
        return [self.sample_episode_params() for _ in range(n_envs)]

    def sample_batch_action_noise(
        self, n_envs: int, action_dim: int = 12
    ) -> np.ndarray:
        """批量生成动作噪声。

        Returns
        -------
        (n_envs, action_dim) ndarray
        """
        return self._rng.normal(
            0.0, self.cfg.action_noise_std, size=(n_envs, action_dim)
        )

    def sample_batch_observation_noise(
        self, n_envs: int, obs_dim: int
    ) -> np.ndarray:
        """批量生成观测噪声。

        Returns
        -------
        (n_envs, obs_dim) ndarray
        """
        return self._rng.normal(
            0.0, self.cfg.observation_noise_std, size=(n_envs, obs_dim)
        )
