"""PPO 训练器 — 策略梯度优化核心。

严格实现 training/README.md §2 中定义的 PPO 数学模型：

- §2.1  总损失函数：L = L_actor + c_v · L_critic - c_e · H[π]
- §2.3  PPO-Clip actor loss（clipped surrogate objective）
- §2.4  Critic value loss（可选 clipped value loss）
- §2.5  高斯策略熵正则化
- §6.1  Step 3 — mini-batch 梯度下降循环
- §6.2  全局梯度范数裁剪

接口约定：
- RolloutBuffer 的 mini-batch 按分支字典输出 actor 观测
  （key 为 "proprio", "ee", "context", "stability", "visual"）。
- Actor.evaluate_actions(flat_obs_branches, actions, hidden) → log_prob, entropy, hidden
- Critic.forward(actor_branches, privileged) → value
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim


# ── Actor 观测的标准分支 key 列表 ─────────────────────────────────
_ACTOR_BRANCH_KEYS = ("proprio", "ee", "context", "stability", "visual")


@dataclass
class PPOConfig:
    """PPO 超参配置。

    默认值对齐 training/README.md §7.1。
    """

    # ── 折扣与 GAE ────────────────────────────────────────────────
    gamma: float = 0.99        # 折扣因子 γ
    lam: float = 0.95          # GAE 偏差-方差权衡 λ

    # ── PPO-Clip ──────────────────────────────────────────────────
    clip_eps: float = 0.2      # 策略比率裁剪参数 ε
    value_clip_eps: float = 0.2  # Value function 裁剪参数 ε_v
    use_clipped_value_loss: bool = True

    # ── 损失权重 ──────────────────────────────────────────────────
    entropy_coef: float = 0.01  # 熵正则化系数 c_e
    value_coef: float = 0.5     # Critic 损失权重 c_v

    # ── 梯度裁剪 ──────────────────────────────────────────────────
    max_grad_norm: float = 1.0  # 全局梯度范数上限 g_max

    # ── 学习率 ────────────────────────────────────────────────────
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # ── 学习率线性衰减 ────────────────────────────────────────────
    lr_decay: bool = False            # 是否启用线性衰减至 0
    lr_total_steps: int = 10_000_000  # 衰减完成时对应的全局 env 步数

    # ── 更新循环 ──────────────────────────────────────────────────
    num_mini_batches: int = 4   # Mini-batch 数量 N_mb
    num_epochs: int = 5         # 优化轮数 K
    seq_length: int = 16        # TBPTT 截断长度 L

    # ── 优势标准化 ────────────────────────────────────────────────
    normalize_advantages: bool = True


class PPOTrainer:
    """PPO 策略梯度训练器。

    Parameters
    ----------
    actor : nn.Module
        Actor 网络，需实现：
        - ``forward(flat_obs_branches, hidden)``
        - ``evaluate_actions(flat_obs_branches, actions, hidden)``
    critic : nn.Module
        Critic 网络，需实现：
        - ``forward(actor_branches, privileged)``
    cfg : PPOConfig | None
        PPO 超参配置。
    device : str | torch.device
        计算设备。
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        cfg: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)

        # 确定 RNN 类型，避免在 _update_step 中通过 tensor 值推断
        self.use_lstm: bool = (
            getattr(getattr(actor, "cfg", None), "rnn_type", "gru") == "lstm"
        )

        # 独立 optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr
        )

    # ═══════════════════════════════════════════════════════════════════
    # §6.1 Step 3 — PPO 更新主循环
    # ═══════════════════════════════════════════════════════════════════

    def update(self, buffer) -> dict[str, float]:
        """执行一次完整的 PPO 参数更新。

        Parameters
        ----------
        buffer : RolloutBuffer
            已完成 GAE 计算的轨迹缓存。

        Returns
        -------
        dict[str, float]
            本次更新的汇总指标。
        """
        c = self.cfg
        self.actor.train()
        self.critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _epoch in range(c.num_epochs):
            for batch in buffer.recurrent_mini_batch_generator(
                num_mini_batches=c.num_mini_batches,
                seq_length=c.seq_length,
            ):
                metrics = self._update_step(batch)

                total_actor_loss += metrics["actor_loss"]
                total_critic_loss += metrics["critic_loss"]
                total_entropy += metrics["entropy"]
                total_clip_frac += metrics["clip_fraction"]
                total_approx_kl += metrics["approx_kl"]
                num_updates += 1

        n = max(num_updates, 1)
        explained_var = self._compute_explained_variance(buffer)

        return {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "clip_fraction": total_clip_frac / n,
            "approx_kl": total_approx_kl / n,
            "explained_variance": explained_var,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 单步更新
    # ═══════════════════════════════════════════════════════════════════

    def _update_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """执行单个 mini-batch 的梯度更新。

        batch 字典来自 ``RolloutBuffer.recurrent_mini_batch_generator()``，
        包含各 actor 分支 ``(B, L, dim)`` 和标量序列 ``(B, L)``。
        """
        c = self.cfg

        # ── 解包通用序列 ─────────────────────────────────────────
        act_seq = batch["actions"]            # (B, L, act_dim)
        old_log_probs = batch["log_probs"]    # (B, L)
        old_values = batch["values"]          # (B, L)
        advantages = batch["advantages"]      # (B, L)
        returns = batch["returns"]            # (B, L)
        hidden_init = batch["hidden_init"]    # (layers, B, H)
        priv_seq = batch["privileged"]        # (B, L, priv_dim)

        B, L = act_seq.shape[:2]

        # ── 优势标准化 ────────────────────────────────────────────
        if c.normalize_advantages:
            adv_flat = advantages.reshape(-1)
            adv_mean = adv_flat.mean()
            adv_std = adv_flat.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # ── TBPTT: 逐步推演 Actor ────────────────────────────────
        all_log_probs = []
        all_entropies = []

        # 重建 hidden state — 支持 GRU (bare tensor) 和 LSTM (h, c) tuple
        h_init = hidden_init.detach()
        cell_init = batch.get("cell_init")
        if self.use_lstm and cell_init is not None:
            hidden = (h_init, cell_init.detach())
        else:
            hidden = h_init

        for t in range(L):
            # 构建当前步的分支字典 {name: (B, dim)}
            obs_t = {
                name: batch[name][:, t, :]
                for name in _ACTOR_BRANCH_KEYS
                if name in batch
            }
            act_t = act_seq[:, t, :]

            log_prob_t, entropy_t, hidden = self.actor.evaluate_actions(
                obs_t, act_t, hidden
            )
            all_log_probs.append(log_prob_t)
            all_entropies.append(entropy_t)

        new_log_probs = torch.stack(all_log_probs, dim=1)  # (B, L)
        entropies = torch.stack(all_entropies, dim=1)      # (B, L)

        # ── §2.3 PPO-Clip Actor Loss ─────────────────────────────
        log_ratio = new_log_probs - old_log_probs
        # Clamp log_ratio to prevent numerical overflow in exp()
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - c.clip_eps, 1.0 + c.clip_eps
        ) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        with torch.no_grad():
            clip_fraction = (
                (torch.abs(ratio - 1.0) > c.clip_eps).float().mean().item()
            )
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

        # ── §2.5 熵正则化 ────────────────────────────────────────
        entropy_loss = entropies.mean()

        # ── §2.4 Critic Value Loss ───────────────────────────────
        # Critic 无 RNN，可批量前向：将序列展平为 (B*L, dim)
        critic_obs_flat = {
            name: batch[name].reshape(B * L, -1)
            for name in _ACTOR_BRANCH_KEYS
            if name in batch
        }
        priv_flat = priv_seq.reshape(B * L, -1)

        new_values = self.critic.forward(
            critic_obs_flat, priv_flat
        ).squeeze(-1).reshape(B, L)

        if c.use_clipped_value_loss:
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values,
                -c.value_clip_eps,
                c.value_clip_eps,
            )
            value_loss_unclipped = (new_values - returns) ** 2
            value_loss_clipped = (value_pred_clipped - returns) ** 2
            critic_loss = 0.5 * torch.max(
                value_loss_unclipped, value_loss_clipped
            ).mean()
        else:
            critic_loss = 0.5 * ((new_values - returns) ** 2).mean()

        # ── §2.1 总损失 ──────────────────────────────────────────
        total_loss = (
            actor_loss
            + c.value_coef * critic_loss
            - c.entropy_coef * entropy_loss
        )

        # ── 反向传播 + §6.2 梯度裁剪 ────────────────────────────
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_loss.item(),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_explained_variance(buffer) -> float:
        """计算 Value function 对 return 的解释方差。"""
        with torch.no_grad():
            values = buffer.values.reshape(-1)
            returns = buffer.returns.reshape(-1)

            var_returns = returns.var()
            if var_returns < 1e-8:
                return 0.0

            explained = 1.0 - (returns - values).var() / var_returns
            return float(explained.clamp(-1.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # 序列化 / 恢复
    # ═══════════════════════════════════════════════════════════════════

    def state_dict(self) -> dict:
        """导出训练器状态（用于 checkpoint 保存）。"""
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复训练器状态（用于 checkpoint 加载）。"""
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def step_lr(self, global_steps: int) -> None:
        """按全局 env 步数线性衰减学习率（每 training iteration 调用一次）。

        仅在 ``cfg.lr_decay=True`` 时生效，从初始 lr 线性衰减至 0。
        """
        if not self.cfg.lr_decay:
            return
        frac = min(1.0, global_steps / max(self.cfg.lr_total_steps, 1))
        factor = 1.0 - frac
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = self.cfg.actor_lr * factor
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.cfg.critic_lr * factor
