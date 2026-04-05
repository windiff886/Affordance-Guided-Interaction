"""向量化并行环境封装 — 满足 VecEnvProtocol 协议。

将 N 个 DoorInteractionEnv 实例封装为批量接口，供
training/rollout_collector.py 调用。

接口签名严格对齐 VecEnvProtocol：
    reset()  → (actor_obs_list, critic_obs_list)
    step(actions)  → (actor_obs_list, critic_obs_list,
                      rewards(n,), dones(n,), infos)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_env import EnvConfig
from .door_env import DoorInteractionEnv


class VecDoorEnv:
    """向量化门交互环境。

    持有 n_envs 个独立的 DoorInteractionEnv 实例，
    提供批量 reset / step 接口。

    当某个子环境 done 时，在 step 内自动重置该环境，
    返回重置后的初始观测（auto-reset 策略）。

    Parameters
    ----------
    n_envs : int
        并行环境数量。
    cfg : EnvConfig | None
        所有子环境共享的配置。
    """

    def __init__(
        self,
        n_envs: int,
        cfg: EnvConfig | None = None,
    ) -> None:
        self._n_envs = n_envs
        self._cfg = cfg or EnvConfig()
        self._envs = [DoorInteractionEnv(self._cfg) for _ in range(n_envs)]

        # 当前回合级参数缓存（用于 auto-reset）
        self._door_types: list[str] = ["push"] * n_envs
        self._left_occupied: list[bool] = [False] * n_envs
        self._right_occupied: list[bool] = [False] * n_envs
        self._domain_params_list: list[dict[str, Any] | None] = [None] * n_envs

    @property
    def n_envs(self) -> int:
        """并行环境数量。"""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """单环境动作空间维度。"""
        return self._cfg.total_joints

    # ═══════════════════════════════════════════════════════════════════
    # reset
    # ═══════════════════════════════════════════════════════════════════

    def reset(
        self,
        *,
        domain_params_list: list[dict[str, Any] | None] | None = None,
        door_types: list[str] | None = None,
        left_occupied_list: list[bool] | None = None,
        right_occupied_list: list[bool] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """重置所有环境。

        Parameters
        ----------
        domain_params_list : list[dict] | None
            每个环境的域随机化参数。None 时使用默认值。
        door_types : list[str] | None
            每个环境的门类型。None 时全部使用 "push"。
        left_occupied_list : list[bool] | None
            每个环境左臂是否持杯。
        right_occupied_list : list[bool] | None
            每个环境右臂是否持杯。

        Returns
        -------
        actor_obs_list : list[dict]
        critic_obs_list : list[dict]
        """
        if domain_params_list is not None:
            self._domain_params_list = domain_params_list
        if door_types is not None:
            self._door_types = door_types
        if left_occupied_list is not None:
            self._left_occupied = left_occupied_list
        if right_occupied_list is not None:
            self._right_occupied = right_occupied_list

        actor_obs_list = []
        critic_obs_list = []

        for i, env in enumerate(self._envs):
            actor_obs, critic_obs = env.reset(
                domain_params=self._domain_params_list[i],
                door_type=self._door_types[i],
                left_occupied=self._left_occupied[i],
                right_occupied=self._right_occupied[i],
            )
            actor_obs_list.append(actor_obs)
            critic_obs_list.append(critic_obs)

        return actor_obs_list, critic_obs_list

    # ═══════════════════════════════════════════════════════════════════
    # step
    # ═══════════════════════════════════════════════════════════════════

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, list[dict]]:
        """批量执行一步。

        Parameters
        ----------
        actions : (n_envs, 12) ndarray
            每个环境的绝对力矩指令。

        Returns
        -------
        actor_obs_list : list[dict]
        critic_obs_list : list[dict]
        rewards : (n_envs,) ndarray
        dones : (n_envs,) ndarray
        infos : list[dict]
        """
        assert actions.shape == (self._n_envs, self.action_dim), (
            f"动作维度不匹配：期望 ({self._n_envs}, {self.action_dim})，"
            f"实际 {actions.shape}"
        )

        actor_obs_list = []
        critic_obs_list = []
        rewards = np.zeros(self._n_envs, dtype=np.float64)
        dones = np.zeros(self._n_envs, dtype=np.float64)
        infos: list[dict] = []

        for i, env in enumerate(self._envs):
            actor_obs, critic_obs, reward, done, info = env.step(actions[i])

            # auto-reset：done 的环境立即重置，返回新的初始观测
            if done:
                actor_obs, critic_obs = env.reset(
                    domain_params=self._domain_params_list[i],
                    door_type=self._door_types[i],
                    left_occupied=self._left_occupied[i],
                    right_occupied=self._right_occupied[i],
                )
                # 在 info 中保留终止状态，便于统计
                info["terminal_observation"] = True

            actor_obs_list.append(actor_obs)
            critic_obs_list.append(critic_obs)
            rewards[i] = reward
            dones[i] = float(done)
            infos.append(info)

        return actor_obs_list, critic_obs_list, rewards, dones, infos

    # ═══════════════════════════════════════════════════════════════════
    # 课程阶段更新
    # ═══════════════════════════════════════════════════════════════════

    def set_curriculum(
        self,
        *,
        door_types: list[str],
        left_occupied_list: list[bool],
        right_occupied_list: list[bool],
        domain_params_list: list[dict[str, Any] | None],
    ) -> None:
        """更新课程阶段配置。

        在下一次 reset（包括 auto-reset）时生效。

        Parameters
        ----------
        door_types : list[str]
            每个环境的门类型。
        left_occupied_list : list[bool]
            每个环境左臂持杯状态。
        right_occupied_list : list[bool]
            每个环境右臂持杯状态。
        domain_params_list : list[dict | None]
            每个环境的域随机化参数。
        """
        self._door_types = door_types
        self._left_occupied = left_occupied_list
        self._right_occupied = right_occupied_list
        self._domain_params_list = domain_params_list

    def get_visual_observations(self) -> list[dict[str, Any] | None]:
        """返回每个子环境当前的原始视觉观测。"""
        return [env.get_visual_observation() for env in self._envs]

    # ═══════════════════════════════════════════════════════════════════
    # 资源释放
    # ═══════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """释放所有子环境资源。"""
        for env in self._envs:
            env.close()
