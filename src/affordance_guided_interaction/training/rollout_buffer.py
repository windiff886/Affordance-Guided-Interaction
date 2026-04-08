"""轨迹缓存 — 存储 rollout 数据、执行 GAE 优势估计、提供 TBPTT mini-batch。

实现 training/README.md 中的以下逻辑：

- §2.2  GAE（Generalized Advantage Estimation）：逐步反向累积 TD 残差
- §3.4  TBPTT 截断序列分割：将轨迹切分为长度 L 的片段并恢复初始隐状态
- §6.1  Step 1 的数据结构支撑

设计要点：
- 内部全部使用 ``torch.Tensor`` 以支持 GPU 加速
- 预分配 ``(n_steps, n_envs, dim)`` 形状的固定容量张量，避免动态 append
- GAE 在 rollout 结束后一次性完成，不在每步递增计算
- Actor 观测按分支字典存储，保持与 ``flatten_actor_obs()`` 输出格式一致
"""

from __future__ import annotations

import warnings
from typing import Iterator

import torch
import numpy as np


class RolloutBuffer:
    """固定容量的 on-policy 轨迹缓存。

    Parameters
    ----------
    n_envs : int
        并行环境数量。
    n_steps : int
        每轮 rollout 的采集步数 T。
    actor_branch_dims : dict[str, int]
        Actor 各分支的维度映射，例如：
        ``{"proprio": 60, "ee": 26, "context": 2, "stability": 40, "visual": 768}``
    privileged_dim : int
        展平后的 privileged 信息总维度（16）。
    action_dim : int
        动作维度（默认 12）。
    rnn_hidden_dim : int
        RNN 隐状态维度（用于 TBPTT 时恢复初始状态）。
    rnn_num_layers : int
        RNN 层数。
    device : str | torch.device
        存储设备。
    """

    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        actor_branch_dims: dict[str, int],
        privileged_dim: int,
        action_dim: int = 12,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = torch.device(device)
        self.actor_branch_dims = actor_branch_dims

        # ── Actor 观测：按分支预分配 ─────────────────────────────
        self.actor_obs_branches: dict[str, torch.Tensor] = {}
        for name, dim in actor_branch_dims.items():
            self.actor_obs_branches[name] = torch.zeros(
                n_steps, n_envs, dim, device=self.device
            )

        # ── Privileged 观测 ──────────────────────────────────────
        self.privileged_obs = torch.zeros(
            n_steps, n_envs, privileged_dim, device=self.device
        )

        # ── 动作与策略输出 ───────────────────────────────────────
        self.actions = torch.zeros(
            n_steps, n_envs, action_dim, device=self.device
        )
        self.log_probs = torch.zeros(n_steps, n_envs, device=self.device)
        self.values = torch.zeros(n_steps, n_envs, device=self.device)

        # ── 环境反馈 ─────────────────────────────────────────────
        self.rewards = torch.zeros(n_steps, n_envs, device=self.device)
        self.dones = torch.zeros(n_steps, n_envs, device=self.device)

        # ── RNN 隐状态：每步开始时缓存，用于 TBPTT 恢复 ─────────
        self.hidden_states = torch.zeros(
            n_steps, rnn_num_layers, n_envs, rnn_hidden_dim, device=self.device
        )
        # LSTM cell state（如果使用 GRU 则保持全零不影响）
        self.cell_states = torch.zeros(
            n_steps, rnn_num_layers, n_envs, rnn_hidden_dim, device=self.device
        )

        # ── GAE 计算结果 ─────────────────────────────────────────
        self.advantages = torch.zeros(n_steps, n_envs, device=self.device)
        self.returns = torch.zeros(n_steps, n_envs, device=self.device)

    # ═══════════════════════════════════════════════════════════════════
    # 数据写入
    # ═══════════════════════════════════════════════════════════════════

    def add(
        self,
        step_idx: int,
        *,
        actor_obs_branches: dict[str, torch.Tensor],
        privileged_flat: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        hidden_states: torch.Tensor,
        cell_states: torch.Tensor | None = None,
    ) -> None:
        """写入第 step_idx 步的 transition 数据。

        Parameters
        ----------
        step_idx : int
            当前步索引，范围 [0, n_steps)。
        actor_obs_branches : dict[str, Tensor]
            分支张量字典，每个值形状 ``(n_envs, branch_dim)``。
            key 必须与初始化时的 ``actor_branch_dims`` 一致。
        privileged_flat : ``(n_envs, privileged_dim)``
        actions : ``(n_envs, action_dim)``
        log_probs : ``(n_envs,)``
        values : ``(n_envs,)``
        rewards : ``(n_envs,)``
        dones : ``(n_envs,)``
        hidden_states : ``(num_layers, n_envs, hidden_dim)``
            当前步开始时的 RNN 隐状态（在 actor forward 之前缓存）。
        cell_states : ``(num_layers, n_envs, hidden_dim)`` | None
            LSTM cell state（GRU 时传 None）。
        """
        for name, tensor in actor_obs_branches.items():
            self.actor_obs_branches[name][step_idx] = tensor

        self.privileged_obs[step_idx] = privileged_flat
        self.actions[step_idx] = actions
        self.log_probs[step_idx] = log_probs
        self.values[step_idx] = values
        self.rewards[step_idx] = rewards
        self.dones[step_idx] = dones
        self.hidden_states[step_idx] = hidden_states
        if cell_states is not None:
            self.cell_states[step_idx] = cell_states

    # ═══════════════════════════════════════════════════════════════════
    # §2.2 GAE 优势估计
    # ═══════════════════════════════════════════════════════════════════

    def compute_gae(
        self,
        gamma: float,
        lam: float,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
    ) -> None:
        """基于 GAE 公式计算优势函数和回报目标。

        .. math::

            \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)

            \\hat{A}_t = \\sum_{l=0}^{T-t-1} (\\gamma \\lambda)^l \\delta_{t+l}

        Parameters
        ----------
        gamma : float
            折扣因子 γ。
        lam : float
            GAE 偏差-方差权衡系数 λ。
        last_values : ``(n_envs,)``
            rollout 结束后下一步的 V(s_{T})，用于 bootstrap。
        last_dones : ``(n_envs,)``
            rollout 结束后下一步的 done 标记。
        """
        last_gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # TD 残差 δ_t
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )

            # GAE 递推：A_t = δ_t + γλ(1-d_t) A_{t+1}
            last_gae = delta + gamma * lam * next_non_terminal * last_gae

            self.advantages[t] = last_gae

        # 回报目标 R_t = A_t + V(s_t)（原地写入，避免重新分配预分配缓存）
        torch.add(self.advantages, self.values, out=self.returns)

    # ═══════════════════════════════════════════════════════════════════
    # §3.4 TBPTT 序列分割 mini-batch 生成器
    # ═══════════════════════════════════════════════════════════════════

    def recurrent_mini_batch_generator(
        self,
        num_mini_batches: int,
        seq_length: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """生成 TBPTT 截断序列的 mini-batch 迭代器。

        将 ``(n_steps, n_envs)`` 的轨迹数据按时间维度切分为长度
        ``seq_length`` 的连续序列片段，每个片段附带从 rollout 缓存
        恢复的初始隐状态 h_{t_0}（梯度只在片段内传播）。

        Parameters
        ----------
        num_mini_batches : int
            每轮迭代划分的 mini-batch 数量。
        seq_length : int
            TBPTT 截断长度 L。

        Yields
        ------
        dict[str, Tensor]
            包含以下 key 的 mini-batch 字典：
            - 各 actor 分支:  ``{name: (batch, L, branch_dim)}``
            - ``privileged``:  ``(batch, L, priv_dim)``
            - ``actions``:     ``(batch, L, action_dim)``
            - ``log_probs``:   ``(batch, L)``
            - ``values``:      ``(batch, L)``
            - ``advantages``:  ``(batch, L)``
            - ``returns``:     ``(batch, L)``
            - ``hidden_init``: ``(num_layers, batch, hidden_dim)``
            - ``cell_init``:   ``(num_layers, batch, hidden_dim)``
        """
        # 计算可切分的序列片段数（沿时间维度）
        num_seqs_per_env = self.n_steps // seq_length

        if num_seqs_per_env == 0:
            raise ValueError(
                f"seq_length={seq_length} 超过 n_steps={self.n_steps}，"
                f"无法切分序列片段"
            )

        tail = self.n_steps % seq_length
        if tail > 0:
            warnings.warn(
                f"TBPTT: n_steps={self.n_steps} 不能被 seq_length={seq_length} "
                f"整除，末尾 {tail} 步数据将被丢弃。"
                f"建议选择 n_steps 为 seq_length 的整数倍。",
                stacklevel=2,
            )

        # 总序列片段数
        total_seqs = num_seqs_per_env * self.n_envs

        if total_seqs < num_mini_batches:
            raise ValueError(
                f"total_seqs ({total_seqs}) < num_mini_batches ({num_mini_batches})，"
                f"会产生空 mini-batch。请减小 num_mini_batches 或增大 n_envs/n_steps。"
            )

        seq_indices = self._build_seq_index_tensor(seq_length)

        # 随机打乱
        perm = torch.randperm(total_seqs)

        # 按 mini-batch 切分
        batch_size = total_seqs // num_mini_batches

        for mb_idx in range(num_mini_batches):
            start = mb_idx * batch_size
            end = start + batch_size if mb_idx < num_mini_batches - 1 else total_seqs

            mb_indices = perm[start:end]
            mb_pairs = seq_indices[mb_indices]
            t_start = mb_pairs[:, 0]
            env_idx = mb_pairs[:, 1]
            time_offsets = torch.arange(seq_length, device=self.device).unsqueeze(0)
            time_index = t_start.unsqueeze(1) + time_offsets
            env_index = env_idx.unsqueeze(1)

            result: dict[str, torch.Tensor] = {}
            for name in self.actor_branch_dims:
                result[name] = self.actor_obs_branches[name][time_index, env_index]

            result["privileged"] = self.privileged_obs[time_index, env_index]
            result["actions"] = self.actions[time_index, env_index]
            result["log_probs"] = self.log_probs[time_index, env_index]
            result["values"] = self.values[time_index, env_index]
            result["advantages"] = self.advantages[time_index, env_index]
            result["returns"] = self.returns[time_index, env_index]
            result["hidden_init"] = self.hidden_states[t_start, :, env_idx, :].permute(1, 0, 2)
            result["cell_init"] = self.cell_states[t_start, :, env_idx, :].permute(1, 0, 2)

            yield result

    def iter_minibatches_recurrent(
        self,
        *,
        seq_length: int,
        num_mini_batches: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """兼容包装：按 recurrent mini-batch 形式迭代序列。"""
        yield from self.recurrent_mini_batch_generator(
            num_mini_batches=num_mini_batches,
            seq_length=seq_length,
        )

    def _build_seq_index_tensor(self, seq_length: int) -> torch.Tensor:
        """构建所有 `(t_start, env_idx)` 序列起点索引。"""
        num_seqs_per_env = self.n_steps // seq_length
        t_starts = torch.arange(num_seqs_per_env, device=self.device) * seq_length
        env_ids = torch.arange(self.n_envs, device=self.device)
        grid_t, grid_env = torch.meshgrid(t_starts, env_ids, indexing="ij")
        return torch.stack([grid_t.reshape(-1), grid_env.reshape(-1)], dim=1)

    # ═══════════════════════════════════════════════════════════════════
    # 清空
    # ═══════════════════════════════════════════════════════════════════

    def clear(self) -> None:
        """清零所有缓存数据，准备下一轮 rollout。"""
        for tensor in self.actor_obs_branches.values():
            tensor.zero_()
        self.privileged_obs.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.hidden_states.zero_()
        self.cell_states.zero_()
        self.advantages.zero_()
        self.returns.zero_()
