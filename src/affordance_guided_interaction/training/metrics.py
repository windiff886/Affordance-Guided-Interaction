"""训练指标聚合器 — 收集并汇总训练过程中的各类监控指标。

对应 training/README.md §6.1 Step 5 的指标记录：
- 分项奖励（task / stability / safety）
- 任务成功率（按 affordance 类别可拆分）
- 杯体脱落率
- 平均 episode 长度
- PPO 优化指标（actor loss, critic loss, entropy, clip fraction 等）

所有指标通过 `summarize()` 一次性输出，供外部写入 TensorBoard / WandB。
"""

from __future__ import annotations

from collections import defaultdict


class TrainingMetrics:
    """训练指标聚合器。

    使用方式：
    1. 在每个 episode 结束时调用 ``update_episode(...)``
    2. 在每次 PPO 更新后调用 ``update_ppo(...)``
    3. 在需要记录日志时调用 ``summarize()`` 获取汇总
    4. 调用 ``reset()`` 清空当前周期数据
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """清空当前统计周期的所有数据。"""
        self._episode_count: int = 0
        self._success_count: int = 0
        self._cup_drop_count: int = 0
        self._episode_lengths: list[int] = []
        self._context_totals: dict[str, int] = defaultdict(int)
        self._context_successes: dict[str, int] = defaultdict(int)

        # 分项奖励累加器
        self._reward_sums: dict[str, float] = defaultdict(float)
        self._reward_counts: dict[str, int] = defaultdict(int)

        # PPO 更新指标
        self._ppo_metrics: dict[str, list[float]] = defaultdict(list)

    # ═══════════════════════════════════════════════════════════════════
    # Episode 级更新
    # ═══════════════════════════════════════════════════════════════════

    def update_episode(
        self,
        *,
        success: bool,
        context_name: str | None = None,
        cup_dropped: bool = False,
        episode_length: int = 0,
        reward_info: dict[str, float] | None = None,
    ) -> None:
        """在每个 episode 结束时更新统计数据。

        Parameters
        ----------
        success : bool
            本回合是否完成任务。
        context_name : str | None
            本回合的上下文标识：``none`` / ``left_only`` / ``right_only`` / ``both``。
        cup_dropped : bool
            是否发生杯体脱落。
        episode_length : int
            本回合持续步数。
        reward_info : dict | None
            来自 ``DoorPushEnv._get_rewards()`` 返回的分项奖励信息字典。
        """
        self._episode_count += 1
        if success:
            self._success_count += 1
        if context_name is not None:
            self._context_totals[context_name] += 1
            if success:
                self._context_successes[context_name] += 1
        if cup_dropped:
            self._cup_drop_count += 1
        self._episode_lengths.append(episode_length)

        if reward_info:
            for key, val in reward_info.items():
                self._reward_sums[key] += val
                self._reward_counts[key] += 1

    # ═══════════════════════════════════════════════════════════════════
    # PPO 更新级指标
    # ═══════════════════════════════════════════════════════════════════

    def update_ppo(
        self,
        *,
        actor_loss: float = 0.0,
        critic_loss: float = 0.0,
        entropy: float = 0.0,
        clip_fraction: float = 0.0,
        approx_kl: float = 0.0,
        explained_variance: float = 0.0,
    ) -> None:
        """在每次 PPO 优化更新后记录指标。

        Parameters
        ----------
        actor_loss : float
            Actor 策略损失均值。
        critic_loss : float
            Critic 价值损失均值。
        entropy : float
            策略熵均值。
        clip_fraction : float
            被 clip 的样本比例（衡量策略偏移程度）。
        approx_kl : float
            近似 KL 散度。
        explained_variance : float
            Value function 对 return 的解释方差。
        """
        self._ppo_metrics["ppo/actor_loss"].append(actor_loss)
        self._ppo_metrics["ppo/critic_loss"].append(critic_loss)
        self._ppo_metrics["ppo/entropy"].append(entropy)
        self._ppo_metrics["ppo/clip_fraction"].append(clip_fraction)
        self._ppo_metrics["ppo/approx_kl"].append(approx_kl)
        self._ppo_metrics["ppo/explained_variance"].append(explained_variance)

    # ═══════════════════════════════════════════════════════════════════
    # 汇总输出
    # ═══════════════════════════════════════════════════════════════════

    def summarize(self) -> dict[str, float]:
        """汇总当前周期的所有指标。

        Returns
        -------
        dict[str, float]
            扁平化的指标字典，可直接写入 TensorBoard / WandB。
        """
        summary: dict[str, float] = {}

        # ── Episode 级 ────────────────────────────────────────────────
        n = max(self._episode_count, 1)
        summary["episode/success_rate"] = self._success_count / n
        summary["episode/success_mixed"] = self._success_count / n
        summary["episode/cup_drop_rate"] = self._cup_drop_count / n
        summary["episode/count"] = float(self._episode_count)
        summary["episode/success_none"] = _safe_rate(
            self._context_successes["none"],
            self._context_totals["none"],
        )
        summary["episode/success_left_only"] = _safe_rate(
            self._context_successes["left_only"],
            self._context_totals["left_only"],
        )
        summary["episode/success_right_only"] = _safe_rate(
            self._context_successes["right_only"],
            self._context_totals["right_only"],
        )
        summary["episode/success_both"] = _safe_rate(
            self._context_successes["both"],
            self._context_totals["both"],
        )

        if self._episode_lengths:
            summary["episode/mean_length"] = (
                sum(self._episode_lengths) / len(self._episode_lengths)
            )
        else:
            summary["episode/mean_length"] = 0.0

        # ── 分项奖励均值 ──────────────────────────────────────────────
        for key, total in self._reward_sums.items():
            count = max(self._reward_counts[key], 1)
            summary[f"reward/{key}"] = total / count

        # ── PPO 指标均值 ──────────────────────────────────────────────
        for key, values in self._ppo_metrics.items():
            if values:
                summary[key] = sum(values) / len(values)
            else:
                summary[key] = 0.0

        return summary

    @property
    def success_rate(self) -> float:
        """当前周期的成功率（便捷属性）。"""
        if self._episode_count == 0:
            return 0.0
        return self._success_count / self._episode_count


def _safe_rate(successes: int, totals: int) -> float:
    if totals <= 0:
        return 0.0
    return successes / totals
