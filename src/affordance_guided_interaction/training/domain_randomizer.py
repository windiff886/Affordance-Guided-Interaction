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

    # 基座标称位置 p_0 与微扰半径 Δp（XY 平面）
    base_pos_nominal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_pos_delta: float = 0.03  # 单位：m

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
        """
        c = self.cfg

        # 质量与阻尼：U[min, max]
        cup_mass = float(self._rng.uniform(*c.cup_mass_range))
        door_mass = float(self._rng.uniform(*c.door_mass_range))
        door_damping = float(self._rng.uniform(*c.door_damping_range))

        # 基座位置：标称 p_0 + U[-Δp, Δp]（仅 XY 平面微扰，Z 保持标称）
        nominal = np.asarray(c.base_pos_nominal, dtype=np.float64)
        delta_xy = self._rng.uniform(-c.base_pos_delta, c.base_pos_delta, size=2)
        base_pos = nominal.copy()
        base_pos[0] += delta_xy[0]
        base_pos[1] += delta_xy[1]

        return {
            "cup_mass": cup_mass,
            "door_mass": door_mass,
            "door_damping": door_damping,
            "base_pos": base_pos,
        }

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
