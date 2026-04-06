"""轨迹采集器 — 在并行环境中收集 on-policy 轨迹数据。

实现 training/README.md §6.1 Step 1 的轨迹收集逻辑：

在 N_env 个并行环境中推演 T 步，收集轨迹集合 D：

    D = { (o_t^actor, o_t^critic, a_t, r_t, h_t, log π_old(a_t), V_old(s_t)) }

本模块通过 Protocol 声明环境接口，不硬依赖具体的仿真实现。

接口约定：
- Actor 的观测为 **分支字典** ``{"proprio", "ee", "context", "stability", "visual"}``
  由 ``flatten_actor_obs()`` / ``batch_flatten_actor_obs()`` 产出。
- Critic 接收同样的分支字典 + 额外的 ``privileged`` 向量。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Callable

import torch
import numpy as np

from .episode_stats import compute_episode_outcome_stats


# ═══════════════════════════════════════════════════════════════════
# 环境接口协议
# ═══════════════════════════════════════════════════════════════════

@runtime_checkable
class VecEnvProtocol(Protocol):
    """并行向量化环境的接口协议。

    envs/ 层的具体实现需要满足此协议。
    """

    @property
    def n_envs(self) -> int:
        """并行环境数量。"""
        ...

    def reset(self) -> tuple[list[dict], list[dict]]:
        """重置所有环境。

        Returns
        -------
        actor_obs_list : list[dict]
            每个环境的 actor 观测原始字典
            （由 DirectRLEnvAdapter 产出）。
        critic_obs_list : list[dict]
            每个环境的 critic 观测原始字典
            （由 DirectRLEnvAdapter 产出，含 privileged）。
        """
        ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, list[dict]]:
        """执行一步。

        Parameters
        ----------
        actions : (n_envs, action_dim)

        Returns
        -------
        actor_obs_list, critic_obs_list, rewards(n_envs,),
        dones(n_envs,), infos
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# 类型别名
# ═══════════════════════════════════════════════════════════════════

# 将单个 actor_obs 原始字典展平为分支 Tensor 字典的函数类型
# 签名: (obs_dict) -> dict[str, Tensor]
ActorFlattenFn = Callable[[dict], dict[str, torch.Tensor]]

# 将多个 actor_obs 原始字典批量展平为分支 Tensor 字典的函数类型
# 签名: (list[obs_dict]) -> dict[str, Tensor]  其中 Tensor 形状 (B, dim)
BatchActorFlattenFn = Callable[[list[dict]], dict[str, torch.Tensor]]

# 将 privileged 字典展平为 1-D Tensor 的函数类型
# 签名: (privileged_dict) -> Tensor(16,)
PrivFlattenFn = Callable[[dict], torch.Tensor]


# ═══════════════════════════════════════════════════════════════════
# 轨迹采集器
# ═══════════════════════════════════════════════════════════════════

class RolloutCollector:
    """并行轨迹采集器。

    在 N_env 个环境中同时推演 T 步，将数据写入 RolloutBuffer。

    Parameters
    ----------
    actor : nn.Module
        Actor 网络（``forward(flat_obs_branches, hidden)``）。
    critic : nn.Module
        Critic 网络（``forward(actor_branches, privileged)``）。
    buffer : RolloutBuffer
        轨迹缓存。
    batch_actor_flatten_fn : BatchActorFlattenFn
        ``batch_flatten_actor_obs(obs_list, cfg)``
        的 partial 绑定版本（已绑定 cfg）。
    priv_flatten_fn : PrivFlattenFn
        ``flatten_privileged(privileged_dict)``，来自 policy/critic.py。
    device : str | torch.device
        计算设备。
    """

    def __init__(
        self,
        actor,
        critic,
        buffer,
        batch_actor_flatten_fn: BatchActorFlattenFn,
        priv_flatten_fn: PrivFlattenFn,
        perception_runtime=None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.batch_actor_flatten_fn = batch_actor_flatten_fn
        self.priv_flatten_fn = priv_flatten_fn
        self.perception_runtime = perception_runtime
        self.device = torch.device(device)

        # 隐状态缓存
        self._hidden = None

    # ═══════════════════════════════════════════════════════════════════
    # §6.1 Step 1 — 轨迹收集
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def collect(
        self,
        envs,
        n_steps: int,
        current_actor_obs: list[dict],
        current_critic_obs: list[dict],
    ) -> tuple[list[dict], list[dict], dict[str, float]]:
        """在并行环境中收集 n_steps 步轨迹数据。

        Parameters
        ----------
        envs : VecEnvProtocol
            向量化并行环境。
        n_steps : int
            采集步数 T。
        current_actor_obs : list[dict]
            当前各环境的 actor 原始观测字典。
        current_critic_obs : list[dict]
            当前各环境的 critic 原始观测字典（含 privileged）。

        Returns
        -------
        next_actor_obs : list[dict]
            采集结束后各环境的 actor 观测。
        next_critic_obs : list[dict]
            采集结束后各环境的 critic 观测。
        collect_stats : dict[str, float]
            采集期间的统计信息。
        """
        n_envs = envs.n_envs
        self.actor.eval()
        self.critic.eval()

        # 初始化隐状态
        if self._hidden is None:
            self._init_hidden(n_envs)

        # 统计
        total_rewards = 0.0
        completed_episodes = 0
        successful_episodes = 0
        total_steps = 0
        context_totals = {
            "none": 0,
            "left_only": 0,
            "right_only": 0,
            "both": 0,
        }
        context_successes = {
            "none": 0,
            "left_only": 0,
            "right_only": 0,
            "both": 0,
        }

        current_actor_obs, current_critic_obs = self._prepare_visual_batch(
            envs=envs,
            actor_obs_list=current_actor_obs,
            critic_obs_list=current_critic_obs,
            force_refresh_mask=[True] * n_envs,
        )

        for step in range(n_steps):
            # ── 1. 展平观测为分支张量字典 ────────────────────────
            # actor_branches: {"proprio": (B, d1), "ee": (B, d2), ...}
            actor_branches = self.batch_actor_flatten_fn(current_actor_obs)
            actor_branches = {
                k: v.to(self.device) for k, v in actor_branches.items()
            }

            # privileged: (B, 16)
            priv_flat = self._batch_flatten_priv(current_critic_obs)
            priv_flat = priv_flat.to(self.device)

            # ── 2. 缓存当前隐状态（用于 TBPTT 恢复）────────────
            if isinstance(self._hidden, tuple):
                # LSTM: 缓存完整 (h, c) 元组，TBPTT 恢复需要 cell state
                cached_hidden = tuple(h.detach().clone() for h in self._hidden)
            else:
                cached_hidden = self._hidden.detach().clone()

            # ── 3. Actor 前向：采样动作 ──────────────────────────
            # Actor.forward 期望 dict[str, Tensor] + hidden
            action, log_prob, _entropy, hidden_new = self.actor.forward(
                actor_branches, self._hidden
            )

            # ── 4. Critic 前向：估计价值 ─────────────────────────
            # Critic.forward 期望 dict[str, Tensor] + Tensor(B, 16)
            value = self.critic.forward(
                actor_branches, priv_flat
            ).squeeze(-1)  # (n_envs,)

            # ── 5. 环境 step ─────────────────────────────────────
            actions_np = action.cpu().numpy()
            (
                next_actor_obs,
                next_critic_obs,
                rewards_np,
                dones_np,
                infos,
            ) = envs.step(actions_np)

            rewards = torch.from_numpy(rewards_np).float().to(self.device)
            dones = torch.from_numpy(dones_np).float().to(self.device)

            # ── 6. 写入 Buffer ───────────────────────────────────
            if isinstance(cached_hidden, tuple):
                h_to_store = cached_hidden[0]
                c_to_store = cached_hidden[1]
            else:
                h_to_store = cached_hidden
                c_to_store = None

            self.buffer.add(
                step,
                actor_obs_branches=actor_branches,
                privileged_flat=priv_flat,
                actions=action,
                log_probs=log_prob,
                values=value,
                rewards=rewards,
                dones=dones,
                hidden_states=h_to_store,
                cell_states=c_to_store,
            )

            # ── 7. 更新隐状态，处理 done 环境的隐状态重置 ────────
            self._hidden = hidden_new
            self._reset_hidden_for_dones(dones)

            # 统计
            outcome_stats = compute_episode_outcome_stats(infos=infos, dones=dones_np)
            total_rewards += rewards_np.sum()
            completed_episodes += int(outcome_stats["collect/completed_episodes"])
            successful_episodes += int(outcome_stats["collect/successful_episodes"])
            total_steps += n_envs
            self._accumulate_context_stats(
                infos=infos,
                dones=dones_np,
                context_totals=context_totals,
                context_successes=context_successes,
            )

            # ── 8. 推进观测 ──────────────────────────────────────
            current_actor_obs, current_critic_obs = self._prepare_visual_batch(
                envs=envs,
                actor_obs_list=next_actor_obs,
                critic_obs_list=next_critic_obs,
                force_refresh_mask=[bool(x) for x in dones_np],
            )

        # ── 计算最终 bootstrap value ─────────────────────────────
        last_actor_branches = self.batch_actor_flatten_fn(current_actor_obs)
        last_actor_branches = {
            k: v.to(self.device) for k, v in last_actor_branches.items()
        }
        last_priv_flat = self._batch_flatten_priv(current_critic_obs).to(self.device)

        last_values = self.critic.forward(
            last_actor_branches, last_priv_flat
        ).squeeze(-1)
        last_dones = torch.from_numpy(dones_np).float().to(self.device)

        # 缓存供外部调用 buffer.compute_gae()
        self._last_values = last_values
        self._last_dones = last_dones

        collect_stats = {
            "collect/mean_reward": total_rewards / max(total_steps, 1),
            "collect/completed_episodes": float(completed_episodes),
            "collect/successful_episodes": float(successful_episodes),
            "collect/episode_success_rate": (
                successful_episodes / max(completed_episodes, 1)
            ),
            "collect/success_mixed": (
                successful_episodes / max(completed_episodes, 1)
            ),
            "collect/success_none": self._safe_rate(
                context_successes["none"], context_totals["none"]
            ),
            "collect/success_left_only": self._safe_rate(
                context_successes["left_only"], context_totals["left_only"]
            ),
            "collect/success_right_only": self._safe_rate(
                context_successes["right_only"], context_totals["right_only"]
            ),
            "collect/success_both": self._safe_rate(
                context_successes["both"], context_totals["both"]
            ),
            "collect/total_steps": float(total_steps),
        }

        return current_actor_obs, current_critic_obs, collect_stats

    @property
    def last_values(self) -> torch.Tensor:
        """最近一次 collect 结束后的 bootstrap value（用于 GAE）。"""
        return self._last_values

    @property
    def last_dones(self) -> torch.Tensor:
        """最近一次 collect 结束后的 done 标记（用于 GAE）。"""
        return self._last_dones

    # ═══════════════════════════════════════════════════════════════════
    # 内部辅助方法
    # ═══════════════════════════════════════════════════════════════════

    def _batch_flatten_priv(self, critic_obs_list: list[dict]) -> torch.Tensor:
        """将多个 critic_obs 中的 privileged 字典展平并 stack。"""
        flat_list = [
            self.priv_flatten_fn(co["privileged"]) for co in critic_obs_list
        ]
        return torch.stack(flat_list, dim=0)  # (n_envs, priv_dim)

    def _init_hidden(self, n_envs: int) -> None:
        """初始化隐状态并移至正确设备。"""
        self._hidden = self.actor.init_hidden(n_envs)
        if isinstance(self._hidden, tuple):
            self._hidden = tuple(h.to(self.device) for h in self._hidden)
        else:
            self._hidden = self._hidden.to(self.device)

    def _reset_hidden_for_dones(self, dones: torch.Tensor) -> None:
        """当某些环境 done 时，将对应环境的 RNN 隐状态清零。"""
        done_mask = dones.bool()
        if not done_mask.any():
            return

        if isinstance(self._hidden, tuple):
            h, c = self._hidden
            h[:, done_mask, :] = 0.0
            c[:, done_mask, :] = 0.0
            self._hidden = (h, c)
        else:
            self._hidden[:, done_mask, :] = 0.0

    def reset_hidden(self, n_envs: int) -> None:
        """完全重置所有环境的隐状态（外部调用接口）。"""
        self._init_hidden(n_envs)
        if self.perception_runtime is not None:
            self.perception_runtime.reset(n_envs)

    def _prepare_visual_batch(
        self,
        *,
        envs,
        actor_obs_list: list[dict],
        critic_obs_list: list[dict],
        force_refresh_mask: list[bool],
    ) -> tuple[list[dict], list[dict]]:
        if self.perception_runtime is None:
            return actor_obs_list, critic_obs_list

        visual_observations = None
        if hasattr(envs, "get_visual_observations"):
            visual_observations = envs.get_visual_observations()

        return self.perception_runtime.prepare_batch(
            actor_obs_list=actor_obs_list,
            critic_obs_list=critic_obs_list,
            visual_observations=visual_observations,
            force_refresh_mask=force_refresh_mask,
        )

    @staticmethod
    def _accumulate_context_stats(
        *,
        infos: list[dict],
        dones: np.ndarray,
        context_totals: dict[str, int],
        context_successes: dict[str, int],
    ) -> None:
        for done, info in zip(dones, infos, strict=False):
            if not bool(done):
                continue
            context_name = str(info.get("episode_context", ""))
            if context_name not in context_totals:
                continue
            context_totals[context_name] += 1
            if bool(info.get("success", False)):
                context_successes[context_name] += 1

    @staticmethod
    def _safe_rate(successes: int, totals: int) -> float:
        if totals <= 0:
            return 0.0
        return float(successes / totals)
