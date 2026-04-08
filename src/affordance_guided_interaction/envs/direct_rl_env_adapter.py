"""DirectRLEnv → VecEnvProtocol 适配器。

将 ``DoorPushEnv`` (DirectRLEnv) 的 GPU 批量 tensor 接口
包装为 ``RolloutCollector`` 期望的 ``VecEnvProtocol`` 格式。

这是一个过渡层 — 使现有训练管线（RolloutCollector + PPO）能直接
消费新的 GPU 并行环境，而无需大规模改写训练侧代码。

长期目标是让训练侧原生支持 tensor dict 接口，届时可移除本适配器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from affordance_guided_interaction.envs.door_push_env import DoorPushEnv


# privileged 信息在 critic tensor 中的布局（最后 16 维）
_PRIV_DIM = 16

# privileged 子段的偏移和大小
_PRIV_LAYOUT: list[tuple[str, int, int]] = [
    ("door_pose",       0,  7),
    ("door_joint_pos",  7,  1),
    ("door_joint_vel",  8,  1),
    ("cup_mass",        9,  1),
    ("door_mass",      10,  1),
    ("door_damping",   11,  1),
    ("base_pos",       12,  3),
    ("cup_dropped",    15,  1),
]

# Actor obs tensor 各段切片（M6：命名常量，避免硬编码偏移）
# 对齐 DoorPushEnv._get_observations() 的拼接顺序
_S_JP   = slice(0,   12)   # joint_positions
_S_JV   = slice(12,  24)   # joint_velocities
_S_JT   = slice(24,  36)   # joint_torques
_S_PA   = slice(36,  48)   # prev_action
_S_LEP  = slice(48,  51)   # left ee position
_S_LEQ  = slice(51,  55)   # left ee orientation (quat)
_S_LLV  = slice(55,  58)   # left ee linear_velocity
_S_LAV  = slice(58,  61)   # left ee angular_velocity
_S_LLA  = slice(61,  64)   # left ee linear_acceleration
_S_LAA  = slice(64,  67)   # left ee angular_acceleration
_S_REP  = slice(67,  70)   # right ee position
_S_REQ  = slice(70,  74)   # right ee orientation (quat)
_S_RLV  = slice(74,  77)   # right ee linear_velocity
_S_RAV  = slice(77,  80)   # right ee angular_velocity
_S_RLA  = slice(80,  83)   # right ee linear_acceleration
_S_RAA  = slice(83,  86)   # right ee angular_acceleration
_S_LO   = slice(86,  87)   # left_occupied
_S_RO   = slice(87,  88)   # right_occupied
_S_LT   = slice(88,  89)   # left_tilt
_S_RT   = slice(89,  90)   # right_tilt
_S_EMB  = slice(90,  858)  # door_embedding (768D)


class DirectRLEnvAdapter:
    """将 DoorPushEnv 包装为 VecEnvProtocol 兼容接口。

    VecEnvProtocol 期望：
        - reset(**kwargs) → (list[dict], list[dict])
        - step(np.ndarray) → (list[dict], list[dict], np.ndarray, np.ndarray, list[dict])

    DoorPushEnv (DirectRLEnv) 接口：
        - reset() → (obs_dict, info_dict)   obs_dict = {"policy": Tensor, "critic": Tensor}
        - step(Tensor) → (obs_dict, reward, terminated, truncated, info_dict)

    本适配器将 tensor 观测解包为 list[dict] 格式，使下游 RolloutCollector
    中的 batch_flatten 流程可以正常消费。
    """

    def __init__(
        self,
        env: "DoorPushEnv",
        *,
        actor_obs_dim: int | None = None,
        critic_obs_dim: int | None = None,
    ) -> None:
        self._env = env
        self._actor_obs_dim = actor_obs_dim or env.cfg.num_observations
        self._critic_obs_dim = critic_obs_dim or env.cfg.num_states

        # 缓存当前 obs tensor
        self._current_actor_obs: Tensor | None = None
        self._current_critic_obs: Tensor | None = None

        # episode reset 回调（兼容旧接口）
        self._episode_reset_fn: Callable | None = None

    @dataclass(slots=True)
    class BatchStepResult:
        actor_obs: Tensor
        critic_obs: Tensor
        rewards: Tensor
        dones: Tensor
        terminated: Tensor
        truncated: Tensor
        info_dict: dict[str, Any]
        infos: list[dict[str, Any]]

    @property
    def n_envs(self) -> int:
        return self._env.num_envs

    def reset(
        self,
        *,
        domain_params_list: list[dict[str, Any] | None] | None = None,
        door_types: list[str] | None = None,
        left_occupied_list: list[bool] | None = None,
        right_occupied_list: list[bool] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """重置所有环境。"""
        actor_obs, critic_obs = self.reset_batch(
            domain_params_list=domain_params_list,
            door_types=door_types,
            left_occupied_list=left_occupied_list,
            right_occupied_list=right_occupied_list,
        )
        return self._tensor_to_obs_list({"policy": actor_obs, "critic": critic_obs})

    def reset_batch(
        self,
        *,
        domain_params_list: list[dict[str, Any] | None] | None = None,
        door_types: list[str] | None = None,
        left_occupied_list: list[bool] | None = None,
        right_occupied_list: list[bool] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """重置所有环境并保留 batched tensor 观测。"""
        del door_types
        # 将 occupancy 传递给底层 env
        if left_occupied_list is not None and right_occupied_list is not None:
            self._env.set_occupancy(
                torch.tensor(left_occupied_list, dtype=torch.bool, device=self._env.device),
                torch.tensor(right_occupied_list, dtype=torch.bool, device=self._env.device),
            )

        # 将域随机化参数传递给底层 env（如果它支持的话）
        if domain_params_list is not None and hasattr(self._env, "set_domain_params_batch"):
            self._env.set_domain_params_batch(domain_params_list)

        obs_dict, _info = self._env.reset()
        self._current_actor_obs = obs_dict["policy"]
        self._current_critic_obs = obs_dict["critic"]
        return self._current_actor_obs, self._current_critic_obs

    def step(
        self, actions: np.ndarray | Tensor
    ) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, list[dict]]:
        """执行一步。"""
        if isinstance(actions, torch.Tensor):
            actions_t = actions.to(self._env.device)
        else:
            actions_t = torch.from_numpy(actions).float().to(self._env.device)

        obs_dict, reward_t, terminated_t, truncated_t, info_dict = self._env.step(
            actions_t
        )

        self._current_actor_obs = obs_dict["policy"]
        self._current_critic_obs = obs_dict["critic"]

        # 转换为 numpy
        dones_np = (terminated_t | truncated_t).cpu().numpy().astype(np.float64)
        rewards_np = reward_t.cpu().numpy().astype(np.float64)

        # 构建 info list
        infos = self._build_info_list(
            terminated_t, truncated_t, info_dict, dones_np,
        )

        actor_obs_list, critic_obs_list = self._tensor_to_obs_list(obs_dict)
        return actor_obs_list, critic_obs_list, rewards_np, dones_np, infos

    def step_batch(self, actions: Tensor) -> BatchStepResult:
        """执行一步并保留 tensor 结果。"""
        obs_dict, reward_t, terminated_t, truncated_t, info_dict = self._env.step(
            actions
        )

        self._current_actor_obs = obs_dict["policy"]
        self._current_critic_obs = obs_dict["critic"]

        dones_np = (terminated_t | truncated_t).detach().cpu().numpy().astype(np.float64)
        infos = self._build_info_list(
            terminated_t, truncated_t, info_dict, dones_np,
        )

        return self.BatchStepResult(
            actor_obs=obs_dict["policy"],
            critic_obs=obs_dict["critic"],
            rewards=reward_t,
            dones=(terminated_t | truncated_t),
            terminated=terminated_t,
            truncated=truncated_t,
            info_dict=info_dict,
            infos=infos,
        )

    def _tensor_to_obs_list(
        self, obs_dict: dict[str, Tensor]
    ) -> tuple[list[dict], list[dict]]:
        """将 batch tensor obs 转为 per-env dict list。

        将 DoorPushEnv 产出的 858D flat actor tensor 解构为
        ``flatten_actor_obs()`` 期望的嵌套字典结构。

        Actor obs 858D tensor 布局（对齐 DoorPushEnv._get_observations）：
            [0:12)   joint_positions
            [12:24)  joint_velocities
            [24:36)  joint_torques
            [36:48)  prev_action
            [48:51)  left_ee_pos      [51:55)  left_ee_quat
            [55:58)  left_ee_lv       [58:61)  left_ee_av
            [61:64)  left_ee_la       [64:67)  left_ee_aa
            [67:70)  right_ee_pos     [70:74)  right_ee_quat
            [74:77)  right_ee_lv      [77:80)  right_ee_av
            [80:83)  right_ee_la      [83:86)  right_ee_aa
            [86:87)  left_occupied    [87:88)  right_occupied
            [88:89)  left_tilt        [89:90)  right_tilt
            [90:858) door_embedding

        critic_obs dict 结构:
            {
                "actor_obs": { ... 同 actor_obs ... },
                "privileged": {
                    "door_pose": ndarray(7,),
                    "door_joint_pos": ndarray(1,),
                    ...
                },
            }
        """
        actor_t = obs_dict["policy"]   # (N, 858)
        critic_t = obs_dict["critic"]  # (N, 874)

        actor_list: list[dict] = []
        critic_list: list[dict] = []

        for i in range(self.n_envs):
            # L13: .clone() 避免 slice view 持有整个 (N,858) batch tensor
            a = actor_t[i].clone()

            actor_obs = {
                "proprio": {
                    "joint_positions": a[_S_JP],
                    "joint_velocities": a[_S_JV],
                    "joint_torques":    a[_S_JT],
                    "prev_action":      a[_S_PA],
                },
                "ee": {
                    "left": {
                        "position":             a[_S_LEP],
                        "orientation":          a[_S_LEQ],
                        "linear_velocity":      a[_S_LLV],
                        "angular_velocity":     a[_S_LAV],
                        "linear_acceleration":  a[_S_LLA],
                        "angular_acceleration": a[_S_LAA],
                    },
                    "right": {
                        "position":             a[_S_REP],
                        "orientation":          a[_S_REQ],
                        "linear_velocity":      a[_S_RLV],
                        "angular_velocity":     a[_S_RAV],
                        "linear_acceleration":  a[_S_RLA],
                        "angular_acceleration": a[_S_RAA],
                    },
                },
                "context": {
                    "left_occupied":  a[_S_LO],
                    "right_occupied": a[_S_RO],
                },
                "stability": {
                    "left_tilt":  a[_S_LT],
                    "right_tilt": a[_S_RT],
                },
                "visual": {
                    "door_embedding": a[_S_EMB],
                },
            }
            actor_list.append(actor_obs)

            # 解构 critic tensor → actor_obs + privileged
            critic_row = critic_t[i].cpu().numpy()  # (874,)
            priv_np = critic_row[self._actor_obs_dim:]

            privileged: dict[str, np.ndarray] = {}
            for name, offset, size in _PRIV_LAYOUT:
                privileged[name] = priv_np[offset:offset + size].astype(np.float32)

            critic_list.append({
                "actor_obs": actor_obs,
                "privileged": privileged,
            })

        return actor_list, critic_list

    def _build_info_list(
        self,
        terminated: Tensor,
        truncated: Tensor,
        info_dict: dict,
        dones_np: np.ndarray,
    ) -> list[dict]:
        """构建 per-env info dict list。

        C3 修复：从 env.extras["success"] 读取 pre-reset 成功标志（通过 info_dict 传递），
        而非在 auto-reset 之后重读 door.data.joint_pos（reset 后角度已归零）。

        D7 修复：从 env.extras["episode_left_occupied"] / ["episode_right_occupied"]
        读取 pre-reset occupancy，避免 auto-reset 回调覆写后读到新 episode 的 context。
        """
        infos: list[dict[str, Any]] = []

        # C3: 读取 _get_dones() 在 auto-reset 之前写入的成功标志
        success_tensor: Tensor | None = info_dict.get("success")

        # D7: 读取 _get_dones() 在 auto-reset 之前缓存的 occupancy
        ep_left_occ: Tensor | None = info_dict.get("episode_left_occupied")
        ep_right_occ: Tensor | None = info_dict.get("episode_right_occupied")

        for i in range(self.n_envs):
            done = bool(dones_np[i])
            # success 仅在 episode 结束时有意义；pre-reset 时由 env 计算正确值
            if success_tensor is not None:
                success = bool(success_tensor[i]) and done
            else:
                success = False

            # D7: 从 pre-reset extras 读取 occupancy（done env 的 occupancy
            # 可能已被 _episode_reset_fn 覆写为新 episode 的值）
            if ep_left_occ is not None and ep_right_occ is not None:
                left_occ = bool(ep_left_occ[i])
                right_occ = bool(ep_right_occ[i])
            else:
                # 回退：仅在首次 reset()（env 尚未调用过 _get_dones()）时触发，
                # 此时 occupancy 未被 auto-reset 回调覆写，直接读取安全。
                left_occ = bool(self._env._left_occupied[i])
                right_occ = bool(self._env._right_occupied[i])
            if left_occ and right_occ:
                context = "both"
            elif left_occ:
                context = "left_only"
            elif right_occ:
                context = "right_only"
            else:
                context = "none"

            info: dict[str, Any] = {
                "terminated": bool(terminated[i]),
                "truncated": bool(truncated[i]),
                "success": success,
                "episode_context": context,
            }
            infos.append(info)
        return infos

    # ── 兼容旧接口 ──────────────────────────────────────────────────

    def set_episode_reset_fn(self, fn: Callable | None) -> None:
        """兼容接口。DirectRLEnv 的 auto-reset 由框架管理。"""
        self._episode_reset_fn = fn
        if hasattr(self._env, "set_episode_reset_fn"):
            self._env.set_episode_reset_fn(fn)

    def set_randomizer(self, randomizer: Any) -> None:
        """兼容旧接口。DirectRLEnv 的噪声在 cfg 中配置。"""
        pass

    def set_curriculum(
        self,
        *,
        door_types: list[str],
        left_occupied_list: list[bool],
        right_occupied_list: list[bool],
        domain_params_list: list[dict[str, Any] | None],
    ) -> None:
        """更新课程阶段配置。"""
        # 传递 occupancy
        self._env.set_occupancy(
            torch.tensor(left_occupied_list, dtype=torch.bool, device=self._env.device),
            torch.tensor(right_occupied_list, dtype=torch.bool, device=self._env.device),
        )

        # 传递域随机化参数（如果底层环境支持）
        if domain_params_list is not None and hasattr(self._env, "set_domain_params_batch"):
            self._env.set_domain_params_batch(domain_params_list)

    def get_visual_observations(self) -> list[dict[str, Any] | None]:
        """读取底层环境暴露的视觉观测；缺失时回退为全 None。"""
        if hasattr(self._env, "get_visual_observations"):
            return self._env.get_visual_observations()
        return [None] * self.n_envs

    def get_visual_observations_batch(self) -> dict[str, Tensor] | None:
        """读取底层环境暴露的 batched 视觉观测。"""
        if hasattr(self._env, "get_visual_observations_batch"):
            return self._env.get_visual_observations_batch()
        return None

    def close(self) -> None:
        """关闭环境。"""
        self._env.close()
