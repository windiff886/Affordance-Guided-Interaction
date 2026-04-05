"""PPO 训练主入口 — 完整的采集-优化-课程-日志循环。

用法:
    # Isaac Lab 本地验证（4060 Laptop，少量环境快速跑通）
    python scripts/train.py --config configs/training/default.yaml --num-envs 2

    # A100 服务器无头训练（完整规模）
    python scripts/train.py --config configs/training/default.yaml --headless

    # 从 checkpoint 恢复训练
    python scripts/train.py --resume checkpoints/ckpt_iter_1000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# 确保项目 src 在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict[str, Any]:
    """加载并合并所有 YAML 配置文件。"""
    base_dir = Path(config_path).resolve().parent.parent  # configs/
    merged: dict[str, Any] = {}

    config_files = {
        "training": base_dir / "training/default.yaml",
        "env": base_dir / "env/default.yaml",
        "policy": base_dir / "policy/default.yaml",
        "task": base_dir / "task/default.yaml",
        "curriculum": base_dir / "curriculum/default.yaml",
        "reward": base_dir / "reward/default.yaml",
    }

    for key, path in config_files.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                merged[key] = yaml.safe_load(f) or {}
        else:
            print(f"⚠️ 配置文件不存在: {path}")
            merged[key] = {}

    return merged


# ═══════════════════════════════════════════════════════════════════════
# 组件工厂
# ═══════════════════════════════════════════════════════════════════════

def build_env_config(cfg: dict) -> "EnvConfig":
    """从合并配置构建 EnvConfig 实例。"""
    from affordance_guided_interaction.envs.base_env import EnvConfig

    env_cfg = cfg.get("env", {})
    task_cfg = cfg.get("task", {})
    reward_cfg = cfg.get("reward", {})

    return EnvConfig(
        physics_dt=env_cfg.get("physics_dt", 1.0 / 120.0),
        decimation=env_cfg.get("decimation", 2),
        max_episode_steps=env_cfg.get("max_episode_steps", 500),
        door_angle_target=task_cfg.get("door_angle_target", 1.57),
        cup_drop_threshold=task_cfg.get("cup_drop_threshold", 0.15),
        contact_force_threshold=env_cfg.get("contact_force_threshold", 0.1),
        joints_per_arm=env_cfg.get("joints_per_arm", 6),
        total_joints=env_cfg.get("total_joints", 12),
        reward_cfg=reward_cfg if reward_cfg else None,
        action_history_length=env_cfg.get("action_history_length", 3),
        acc_history_length=env_cfg.get("acc_history_length", 10),
    )


def build_models(cfg: dict, device: torch.device):
    """创建 Actor 和 Critic 网络实例。"""
    from affordance_guided_interaction.policy import (
        Actor, ActorConfig,
        Critic, CriticConfig,
    )

    policy_cfg = cfg.get("policy", {})
    actor_section = policy_cfg.get("actor", {})
    critic_section = policy_cfg.get("critic", {})

    actor_cfg = ActorConfig(
        rnn_hidden=actor_section.get("rnn_hidden", 512),
        rnn_layers=actor_section.get("rnn_layers", 1),
        rnn_type=actor_section.get("rnn_type", "gru"),
        action_dim=actor_section.get("action_dim", 12),
        log_std_init=actor_section.get("log_std_init", -0.5),
        action_history_length=actor_section.get("action_history_length", 3),
        acc_history_length=actor_section.get("acc_history_length", 10),
        include_torques=actor_section.get("include_torques", True),
    )

    critic_cfg = CriticConfig(
        hidden_dims=tuple(critic_section.get("hidden_dims", [512, 256, 128])),
    )

    actor = Actor(actor_cfg).to(device)
    critic = Critic(actor_cfg, critic_cfg).to(device)

    return actor, critic, actor_cfg


def build_ppo_trainer(actor, critic, cfg: dict, device: torch.device):
    """创建 PPO 训练器实例。"""
    from affordance_guided_interaction.training.ppo_trainer import (
        PPOTrainer, PPOConfig,
    )

    t_cfg = cfg.get("training", {})
    ppo_section = t_cfg.get("ppo", {})

    ppo_cfg = PPOConfig(
        gamma=ppo_section.get("gamma", 0.99),
        lam=ppo_section.get("lam", 0.95),
        clip_eps=ppo_section.get("clip_eps", 0.2),
        value_clip_eps=ppo_section.get("value_clip_eps", 0.2),
        use_clipped_value_loss=ppo_section.get("use_clipped_value_loss", True),
        entropy_coef=ppo_section.get("entropy_coef", 0.01),
        value_coef=ppo_section.get("value_coef", 0.5),
        max_grad_norm=ppo_section.get("max_grad_norm", 1.0),
        actor_lr=ppo_section.get("actor_lr", 3e-4),
        critic_lr=ppo_section.get("critic_lr", 3e-4),
        num_mini_batches=ppo_section.get("num_mini_batches", 4),
        num_epochs=ppo_section.get("num_epochs", 5),
        seq_length=ppo_section.get("seq_length", 16),
        normalize_advantages=ppo_section.get("normalize_advantages", True),
    )

    return PPOTrainer(actor, critic, ppo_cfg, device), ppo_cfg


def build_curriculum_reset_batch(
    stage_cfg: Any,
    randomizer: Any,
    n_envs: int,
) -> tuple[list[Any], list[str], list[bool], list[bool]]:
    """根据当前课程阶段生成一批 reset 参数。"""
    domain_params_list = randomizer.sample_batch_episode_params(n_envs)
    door_types = [str(np.random.choice(stage_cfg.door_types)) for _ in range(n_envs)]
    left_occupied, right_occupied = stage_cfg.sample_occupancy_batch(n_envs)
    return domain_params_list, door_types, left_occupied, right_occupied


def build_collector(
    actor, critic, actor_cfg, cfg: dict, ppo_cfg, device: torch.device
):
    """创建轨迹采集器和 RolloutBuffer 实例。"""
    from affordance_guided_interaction.training.rollout_buffer import RolloutBuffer
    from affordance_guided_interaction.training.rollout_collector import RolloutCollector
    from affordance_guided_interaction.training.perception_runtime import (
        PerceptionRuntime,
    )
    from affordance_guided_interaction.policy import (
        batch_flatten_actor_obs,
        flatten_privileged,
    )

    t_cfg = cfg.get("training", {})
    n_envs = t_cfg.get("num_envs", 4)
    n_steps = t_cfg.get("n_steps_per_rollout", 128)

    # 计算各分支维度（与 Actor 网络输入一致）
    n = 6  # joints_per_arm
    proprio_dim = (
        12                                               # q
        + 12                                             # dq
        + (12 if actor_cfg.include_torques else 0)       # tau
        + 12                                             # prev_action
    )

    actor_branch_dims = {
        "proprio": proprio_dim,
        "ee": 38,                                           # 双臂各 19
        "context": 2,                                       # left_occ + right_occ
        "stability": 2,                                     # left/right tilt
        "visual": 768,                                      # Point-MAE 编码
    }

    buffer = RolloutBuffer(
        n_envs=n_envs,
        n_steps=n_steps,
        actor_branch_dims=actor_branch_dims,
        privileged_dim=16,
        action_dim=12,
        rnn_hidden_dim=actor_cfg.rnn_hidden,
        rnn_num_layers=actor_cfg.rnn_layers,
        device=device,
    )

    # partial 绑定展平函数
    batch_flatten_fn = partial(batch_flatten_actor_obs, cfg=actor_cfg)
    visual_cfg = t_cfg.get("visual", {})
    perception_runtime = PerceptionRuntime(
        refresh_interval=visual_cfg.get("refresh_interval", 4),
        embedding_dim=actor_branch_dims["visual"],
    )

    collector = RolloutCollector(
        actor=actor,
        critic=critic,
        buffer=buffer,
        batch_actor_flatten_fn=batch_flatten_fn,
        priv_flatten_fn=flatten_privileged,
        perception_runtime=perception_runtime,
        device=device,
    )

    return collector, buffer, n_steps


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint 管理
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path: Path,
    iteration: int,
    global_steps: int,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    trainer,
    curriculum,
    best_success_rate: float,
) -> None:
    """保存训练 checkpoint，包含模型权重和训练器/课程管理器状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "global_steps": global_steps,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "trainer_state_dict": trainer.state_dict(),
        "curriculum_state_dict": curriculum.state_dict(),
        "best_success_rate": best_success_rate,
    }, str(path))
    print(f"💾 Checkpoint 已保存: {path}")


def load_checkpoint(
    path: Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    trainer,
    curriculum,
    device: torch.device,
) -> tuple[int, int, float]:
    """加载训练 checkpoint。"""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    critic.load_state_dict(ckpt["critic_state_dict"])
    trainer.load_state_dict(ckpt["trainer_state_dict"])
    if "curriculum_state_dict" in ckpt:
        curriculum.load_state_dict(ckpt["curriculum_state_dict"])
    print(f"✅ Checkpoint 已加载: {path}")
    return (
        ckpt.get("iteration", 0),
        ckpt.get("global_steps", 0),
        ckpt.get("best_success_rate", 0.0),
    )


# ═══════════════════════════════════════════════════════════════════════
# 训练主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    """训练主入口。"""
    parser = argparse.ArgumentParser(
        description="Affordance-Guided Interaction PPO 训练"
    )
    parser.add_argument(
        "--config", default="configs/training/default.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="覆盖并行环境数")
    parser.add_argument("--headless", action="store_true", help="无头模式（服务器训练）")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard 日志目录")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Checkpoint 保存目录")
    args = parser.parse_args()

    # ── 设备选择 ─────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  计算设备: {device}")

    # ── 随机种子 ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 加载配置 ─────────────────────────────────────────────
    config_path = str(_PROJECT_ROOT / args.config)
    cfg = load_config(config_path)

    # 命令行覆盖
    if args.num_envs is not None:
        cfg["training"]["num_envs"] = args.num_envs

    t_cfg = cfg.get("training", {})
    n_envs = t_cfg.get("num_envs", 4)
    total_steps = t_cfg.get("total_steps", 10_000_000)
    n_steps_per_rollout = t_cfg.get("n_steps_per_rollout", 128)
    ckpt_interval = t_cfg.get("checkpoint_interval", 50)
    log_interval = t_cfg.get("log_interval", 1)

    print(f"📋 训练配置:")
    print(f"   并行环境数: {n_envs}")
    print(f"   总步数: {total_steps:,}")
    print(f"   每轮采集步数: {n_steps_per_rollout}")
    print(f"   无头模式: {args.headless}")

    # ── 构建环境 ─────────────────────────────────────────────
    env_config = build_env_config(cfg)

    from affordance_guided_interaction.envs.vec_env import VecDoorEnv
    envs = VecDoorEnv(n_envs=n_envs, cfg=env_config)
    print(f"✅ 已创建 {n_envs} 个并行环境")

    # ── 构建模型 ─────────────────────────────────────────────
    actor, critic, actor_cfg = build_models(cfg, device)
    n_actor_params = sum(p.numel() for p in actor.parameters())
    n_critic_params = sum(p.numel() for p in critic.parameters())
    print(f"✅ Actor 参数量: {n_actor_params:,}")
    print(f"✅ Critic 参数量: {n_critic_params:,}")

    # ── 构建训练器 ────────────────────────────────────────────
    ppo_trainer, ppo_cfg = build_ppo_trainer(actor, critic, cfg, device)

    # ── 构建采集器 ────────────────────────────────────────────
    collector, buffer, n_steps = build_collector(
        actor, critic, actor_cfg, cfg, ppo_cfg, device
    )
    print(f"✅ RolloutCollector: n_steps={n_steps}, buffer={n_envs}×{n_steps}={n_envs * n_steps} transitions/iter")

    # ── 课程管理器 ────────────────────────────────────────────
    from affordance_guided_interaction.training.curriculum_manager import CurriculumManager
    from affordance_guided_interaction.training.episode_stats import (
        extract_curriculum_success_rate,
    )
    cur_cfg = cfg.get("curriculum", {})
    curriculum = CurriculumManager(
        window_size=cur_cfg.get("window_size", 50),
        threshold=cur_cfg.get("threshold", 0.8),
        initial_stage=cur_cfg.get("initial_stage"),
    )

    # ── 域随机化器 ────────────────────────────────────────────
    from affordance_guided_interaction.training.domain_randomizer import DomainRandomizer
    randomizer = DomainRandomizer(seed=args.seed)

    # ── 指标聚合器 ────────────────────────────────────────────
    from affordance_guided_interaction.training.metrics import TrainingMetrics
    metrics = TrainingMetrics()

    # ── TensorBoard ──────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=str(_PROJECT_ROOT / args.log_dir / run_name))
        print(f"📊 TensorBoard 日志: {args.log_dir}/{run_name}")
    except ImportError:
        print("⚠️ TensorBoard 不可用，跳过日志记录")

    # ── 恢复 checkpoint ──────────────────────────────────────
    start_iter = 0
    global_steps = 0
    best_success_rate = 0.0
    ckpt_dir = Path(_PROJECT_ROOT / args.ckpt_dir)

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_iter, global_steps, best_success_rate = load_checkpoint(
                resume_path, actor, critic, ppo_trainer, curriculum, device
            )
        else:
            print(f"⚠️ Checkpoint 文件不存在: {resume_path}，从头开始训练")

    # ── 初始化环境 ────────────────────────────────────────────
    initial_stage_cfg = curriculum.get_stage_config()
    print(f"✅ 初始课程阶段: {initial_stage_cfg.name} ({initial_stage_cfg.description})")
    domain_params_list, door_types, left_occupied, right_occupied = (
        build_curriculum_reset_batch(initial_stage_cfg, randomizer, n_envs)
    )

    actor_obs_list, critic_obs_list = envs.reset(
        domain_params_list=domain_params_list,
        door_types=door_types,
        left_occupied_list=left_occupied,
        right_occupied_list=right_occupied,
    )
    collector.reset_hidden(n_envs)

    # ═══════════════════════════════════════════════════════════════════
    # 训练主循环
    # ═══════════════════════════════════════════════════════════════════

    max_iterations = total_steps // (n_envs * n_steps_per_rollout) + 1
    print(f"\n🚀 开始训练（最大 {max_iterations:,} 轮迭代）\n{'─' * 70}")
    t_start = time.time()
    iteration = start_iter  # 用于 finally 块

    try:
        for iteration in range(start_iter, max_iterations):
            iter_start = time.time()

            # ── Step 1: 轨迹采集 ──────────────────────────────
            actor_obs_list, critic_obs_list, collect_stats = collector.collect(
                envs=envs,
                n_steps=n_steps,
                current_actor_obs=actor_obs_list,
                current_critic_obs=critic_obs_list,
            )

            # ── Step 2: 计算 GAE ──────────────────────────────
            buffer.compute_gae(
                gamma=ppo_cfg.gamma,
                lam=ppo_cfg.lam,
                last_values=collector.last_values,
                last_dones=collector.last_dones,
            )

            # ── Step 3: PPO 参数更新 ──────────────────────────
            update_metrics = ppo_trainer.update(buffer)

            # ── Step 4: 更新全局步数计数器 ─────────────────────
            steps_this_iter = n_envs * n_steps
            global_steps += steps_this_iter

            # ── Step 5: 指标更新 ──────────────────────────────
            metrics.update_ppo(
                actor_loss=update_metrics["actor_loss"],
                critic_loss=update_metrics["critic_loss"],
                entropy=update_metrics["entropy"],
                clip_fraction=update_metrics["clip_fraction"],
                approx_kl=update_metrics["approx_kl"],
                explained_variance=update_metrics["explained_variance"],
            )

            iter_time = time.time() - iter_start
            fps = steps_this_iter / max(iter_time, 1e-6)

            # ── Step 6: 课程管理器跃迁判定 ─────────────────────
            mean_reward = collect_stats.get("collect/mean_reward", 0.0)
            completed_eps = collect_stats.get("collect/completed_episodes", 0.0)
            successful_eps = collect_stats.get("collect/successful_episodes", 0.0)
            epoch_success_rate = extract_curriculum_success_rate(collect_stats)

            best_success_rate = max(best_success_rate, epoch_success_rate)
            stage_changed = curriculum.report_epoch(epoch_success_rate)

            if stage_changed:
                new_stage_cfg = curriculum.get_stage_config()
                print(f"\n📈 课程阶段跃迁 → {new_stage_cfg.name}: {new_stage_cfg.description}")
                # 更新环境课程参数
                domain_params_list, door_types, left_occupied, right_occupied = (
                    build_curriculum_reset_batch(new_stage_cfg, randomizer, n_envs)
                )
                envs.set_curriculum(
                    door_types=door_types,
                    left_occupied_list=left_occupied,
                    right_occupied_list=right_occupied,
                    domain_params_list=domain_params_list,
                )

            # ── Step 7: 控制台日志 ────────────────────────────
            if iteration % log_interval == 0:
                elapsed = time.time() - t_start
                eta_sec = elapsed / max(iteration - start_iter + 1, 1) * (max_iterations - iteration)
                eta_h = eta_sec / 3600

                print(
                    f"[Iter {iteration:>6d}] "
                    f"steps={global_steps:>10,} | "
                    f"fps={fps:>6.0f} | "
                    f"a_loss={update_metrics['actor_loss']:>8.4f} | "
                    f"c_loss={update_metrics['critic_loss']:>8.4f} | "
                    f"ent={update_metrics['entropy']:>7.4f} | "
                    f"clip={update_metrics['clip_fraction']:>5.3f} | "
                    f"r̄={mean_reward:>.4f} | "
                    f"succ={epoch_success_rate:>5.3f} | "
                    f"stage={curriculum.current_stage} | "
                    f"ETA={eta_h:.1f}h"
                )

            # ── Step 8: TensorBoard ───────────────────────────
            if writer is not None:
                writer.add_scalar("train/actor_loss", update_metrics["actor_loss"], global_steps)
                writer.add_scalar("train/critic_loss", update_metrics["critic_loss"], global_steps)
                writer.add_scalar("train/entropy", update_metrics["entropy"], global_steps)
                writer.add_scalar("train/clip_fraction", update_metrics["clip_fraction"], global_steps)
                writer.add_scalar("train/approx_kl", update_metrics["approx_kl"], global_steps)
                writer.add_scalar("train/explained_variance", update_metrics["explained_variance"], global_steps)
                writer.add_scalar("train/fps", fps, global_steps)
                writer.add_scalar("collect/mean_reward", mean_reward, global_steps)
                writer.add_scalar("collect/completed_episodes", completed_eps, global_steps)
                writer.add_scalar("collect/successful_episodes", successful_eps, global_steps)
                writer.add_scalar("collect/episode_success_rate", epoch_success_rate, global_steps)
                writer.add_scalar(
                    "collect/success_mixed",
                    collect_stats.get("collect/success_mixed", epoch_success_rate),
                    global_steps,
                )
                writer.add_scalar(
                    "collect/success_none",
                    collect_stats.get("collect/success_none", 0.0),
                    global_steps,
                )
                writer.add_scalar(
                    "collect/success_left_only",
                    collect_stats.get("collect/success_left_only", 0.0),
                    global_steps,
                )
                writer.add_scalar(
                    "collect/success_right_only",
                    collect_stats.get("collect/success_right_only", 0.0),
                    global_steps,
                )
                writer.add_scalar(
                    "collect/success_both",
                    collect_stats.get("collect/success_both", 0.0),
                    global_steps,
                )
                writer.add_scalar("curriculum/stage", curriculum.current_stage, global_steps)
                writer.add_scalar("curriculum/window_mean", curriculum.window_mean, global_steps)

            # ── Step 9: Checkpoint 保存 ───────────────────────
            if iteration > 0 and iteration % ckpt_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"ckpt_iter_{iteration}.pt",
                    iteration=iteration,
                    global_steps=global_steps,
                    actor=actor,
                    critic=critic,
                    trainer=ppo_trainer,
                    curriculum=curriculum,
                    best_success_rate=best_success_rate,
                )

            # ── Step 10: 清空 buffer 准备下轮采集 ─────────────
            buffer.clear()

            # 检查是否达到总步数
            if global_steps >= total_steps:
                print(f"\n✅ 达到总步数上限 {total_steps:,}，训练结束")
                break

    except KeyboardInterrupt:
        print("\n\n⏹ 用户中断训练")

    # ═══════════════════════════════════════════════════════════════════
    # 清理与总结
    # ═══════════════════════════════════════════════════════════════════

    # 保存最终 checkpoint
    save_checkpoint(
        ckpt_dir / "ckpt_final.pt",
        iteration=iteration,
        global_steps=global_steps,
        actor=actor,
        critic=critic,
        trainer=ppo_trainer,
        curriculum=curriculum,
        best_success_rate=best_success_rate,
    )

    elapsed = time.time() - t_start
    print(f"\n{'═' * 70}")
    print(f"📊 训练总结:")
    print(f"   总用时: {elapsed / 3600:.2f} 小时")
    print(f"   总步数: {global_steps:,}")
    print(f"   平均 FPS: {global_steps / max(elapsed, 1):.0f}")
    print(f"   最终课程阶段: {curriculum.current_stage_name}")
    print(f"   课程窗口成功率: {curriculum.window_mean:.3f}")
    print(f"{'═' * 70}")

    # 资源释放
    envs.close()
    if writer is not None:
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
