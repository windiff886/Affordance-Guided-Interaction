"""PPO 训练主入口 — 完整的采集-优化-课程-日志循环。

用法:
    python scripts/train.py

训练运行参数统一从 ``configs/training/default.yaml`` 读取。
如需切换本地验证 / A100 训练配置，请直接修改 YAML 中的生效值。
"""

from __future__ import annotations

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

from affordance_guided_interaction.utils.sim_runtime import (
    launch_simulation_app,
)
from affordance_guided_interaction.utils.runtime_env import resolve_headless_mode
from affordance_guided_interaction.utils.train_runtime_config import (
    resolve_train_runtime_config,
)


# ═══════════════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════════════

def load_config(configs_dir: str | Path | None = None) -> dict[str, Any]:
    """加载并合并所有 YAML 配置文件。

    默认从项目根目录下的 ``configs/`` 目录读取各子目录中的
    ``default.yaml``，无需手动指定路径。

    Parameters
    ----------
    configs_dir:
        配置根目录，默认为 ``<project_root>/configs``。
    """
    configs_dir = _resolve_configs_root(configs_dir)

    merged: dict[str, Any] = {}

    config_files = {
        "training": configs_dir / "training/default.yaml",
        "env": configs_dir / "env/default.yaml",
        "policy": configs_dir / "policy/default.yaml",
        "task": configs_dir / "task/default.yaml",
        "curriculum": configs_dir / "curriculum/default.yaml",
        "reward": configs_dir / "reward/default.yaml",
        "visualization": configs_dir / "visualization/default.yaml",
    }

    for key, path in config_files.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                merged[key] = yaml.safe_load(f) or {}
        else:
            print(f"⚠️ 配置文件不存在: {path}")
            merged[key] = {}

    return merged


def _resolve_configs_root(configs_dir: str | Path | None) -> Path:
    """将配置根解析为 ``configs/`` 目录。

    兼容两种输入：
    1. 配置根目录本身，如 ``configs/``
    2. 其中任一 YAML 文件，如 ``configs/training/default.yaml``
    """
    if configs_dir is None:
        return (_PROJECT_ROOT / "configs").resolve()

    path = Path(configs_dir).resolve()
    if path.is_file():
        return path.parents[1]

    if path.name in {"training", "env", "policy", "task", "curriculum", "reward", "visualization"}:
        default_yaml = path / "default.yaml"
        if default_yaml.exists():
            return path.parent

    return path


# ═══════════════════════════════════════════════════════════════════════
# 组件工厂
# ═══════════════════════════════════════════════════════════════════════

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
        lr_decay=ppo_section.get("lr_decay", False),
        lr_total_steps=ppo_section.get("lr_total_steps", 10_000_000),
    )

    return PPOTrainer(actor, critic, ppo_cfg, device), ppo_cfg


def build_env_cfg(
    cfg: dict[str, Any],
    n_envs: int,
    *,
    device: str | None = None,
    seed: int | None = None,
    enable_cameras: bool = True,
):
    """基于 YAML 配置构建 DoorPushEnvCfg。"""
    from affordance_guided_interaction.envs.door_push_env_cfg import DoorPushEnvCfg

    env_cfg = DoorPushEnvCfg()
    env_cfg.scene.num_envs = int(n_envs)
    env_cfg.sim.render_interval = int(env_cfg.decimation)

    if device is not None:
        env_cfg.sim.device = str(device)
    if seed is not None:
        env_cfg.seed = int(seed)

    env_cfg_yaml = cfg.get("env", {})
    if "physics_dt" in env_cfg_yaml:
        env_cfg.sim.dt = float(env_cfg_yaml["physics_dt"])
    if "decimation" in env_cfg_yaml:
        env_cfg.decimation = int(env_cfg_yaml["decimation"])
        env_cfg.sim.render_interval = env_cfg.decimation

    # NOTE: tiled_camera 已从 DoorPushSceneCfg 默认配置中移除，
    # 不再需要 env_cfg.scene.tiled_camera = None 的守卫逻辑。

    task_cfg = cfg.get("task", {})
    if "door_angle_target" in task_cfg:
        env_cfg.door_angle_target = float(task_cfg["door_angle_target"])
    if "cup_drop_threshold" in task_cfg:
        env_cfg.cup_drop_threshold = float(task_cfg["cup_drop_threshold"])

    # ── 从 reward YAML 注入奖励超参 ─────────────────────────────────
    reward_cfg = cfg.get("reward", {})
    _inject_reward_params(env_cfg, reward_cfg)

    return env_cfg


# ═══════════════════════════════════════════════════════════════════════
# 奖励参数注入
# ═══════════════════════════════════════════════════════════════════════

# YAML 键 → DoorPushEnvCfg 属性 的映射表
_REWARD_PARAM_MAP: dict[str, dict[str, str]] = {
    "task": {
        "w_delta": "rew_w_delta",
        "alpha": "rew_alpha",
        "k_decay": "rew_k_decay",
        "w_open": "rew_w_open",
        "success_angle_threshold": "success_angle_threshold",
    },
    "stability": {
        "w_zero_acc": "rew_w_zero_acc",
        "lambda_acc": "rew_lambda_acc",
        "w_zero_ang": "rew_w_zero_ang",
        "lambda_ang": "rew_lambda_ang",
        "w_acc": "rew_w_acc",
        "w_ang": "rew_w_ang",
        "w_tilt": "rew_w_tilt",
        "w_smooth": "rew_w_smooth",
        "w_reg": "rew_w_reg",
    },
    "safety": {
        "beta_self": "rew_beta_self",
        "beta_limit": "rew_beta_limit",
        "mu": "rew_mu",
        "beta_vel": "rew_beta_vel",
        "beta_torque": "rew_beta_torque",
        "w_drop": "rew_w_drop",
    },
}


def _inject_reward_params(env_cfg: Any, reward_cfg: dict[str, Any]) -> None:
    """将 reward YAML 中的参数注入到 DoorPushEnvCfg 实例。"""
    for section, mapping in _REWARD_PARAM_MAP.items():
        section_cfg = reward_cfg.get(section, {})
        for yaml_key, cfg_attr in mapping.items():
            if yaml_key in section_cfg:
                setattr(env_cfg, cfg_attr, float(section_cfg[yaml_key]))


_REWARD_SUMMARY_KEYS = frozenset({"total", "task", "stab_left", "stab_right", "safe"})


def _iter_reward_scalar_tags(collect_stats: dict[str, float]):
    """将 reward 统计路由到 TensorBoard 标签。

    `reward/` 仅保留总项，细分子项统一写入 `reward_terms/`。
    """
    for key, val in collect_stats.items():
        if not key.startswith("reward/"):
            continue
        suffix = key[len("reward/"):]
        if suffix in _REWARD_SUMMARY_KEYS:
            yield key, val
        else:
            yield f"reward_terms/{suffix}", val


def build_curriculum_reset_batch(
    stage_cfg: Any,
    randomizer: Any,
    n_envs: int,
) -> tuple[list[Any], list[str], list[bool], list[bool]]:
    """根据当前课程阶段生成一批 reset 参数。"""
    domain_params_list = randomizer.sample_batch_episode_params(n_envs)
    door_types = [str(randomizer._rng.choice(stage_cfg.door_types)) for _ in range(n_envs)]
    left_occupied, right_occupied = stage_cfg.sample_occupancy_batch(n_envs)
    return domain_params_list, door_types, left_occupied, right_occupied


def build_collector(
    actor, critic, actor_cfg, cfg: dict, ppo_cfg, device: torch.device
):
    """创建轨迹采集器和 RolloutBuffer 实例。"""
    from affordance_guided_interaction.training.rollout_buffer import RolloutBuffer
    from affordance_guided_interaction.training.rollout_collector import RolloutCollector
    from affordance_guided_interaction.policy import (
        batch_flatten_actor_obs,
        flatten_privileged,
    )

    t_cfg = cfg.get("training", {})
    n_envs = t_cfg.get("num_envs", 4)
    n_steps = t_cfg.get("n_steps_per_rollout", 128)

    # 计算各分支维度（与 Actor 网络输入一致）
    proprio_dim = (
        12                                               # q (TOTAL_ARM_JOINTS)
        + 12                                             # dq
        + (12 if actor_cfg.include_torques else 0)       # tau
        + 12                                             # prev_action
    )

    actor_branch_dims = {
        "proprio": proprio_dim,
        "ee": 38,                                           # 双臂各 19
        "context": 2,                                       # left_occ + right_occ
        "stability": 2,                                     # left/right tilt
        "door_geometry": 6,                                 # center(3) + normal(3)
    }

    buffer = RolloutBuffer(
        n_envs=n_envs,
        n_steps=n_steps,
        actor_branch_dims=actor_branch_dims,
        privileged_dim=13,
        action_dim=12,
        rnn_hidden_dim=actor_cfg.rnn_hidden,
        rnn_num_layers=actor_cfg.rnn_layers,
        device=device,
    )

    # partial 绑定展平函数
    batch_flatten_fn = partial(batch_flatten_actor_obs, cfg=actor_cfg)

    collector = RolloutCollector(
        actor=actor,
        critic=critic,
        buffer=buffer,
        batch_actor_flatten_fn=batch_flatten_fn,
        priv_flatten_fn=flatten_privileged,
        perception_runtime=None,
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
    cfg = load_config()
    runtime_cfg = resolve_train_runtime_config(cfg, project_root=_PROJECT_ROOT)

    # ── 设备选择 ─────────────────────────────────────────────
    if runtime_cfg.device:
        device = torch.device(runtime_cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  计算设备: {device}")

    # ── 随机种子 ─────────────────────────────────────────────
    torch.manual_seed(runtime_cfg.seed)
    np.random.seed(runtime_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime_cfg.seed)

    t_cfg = cfg.get("training", {})
    debug_cfg = t_cfg.get("debug", {})

    n_envs = runtime_cfg.num_envs or t_cfg.get("num_envs", 4)
    total_steps = t_cfg.get("total_steps", 10_000_000)
    n_steps_per_rollout = t_cfg.get("n_steps_per_rollout", 128)
    ckpt_interval = t_cfg.get("checkpoint_interval", 50)
    log_interval = t_cfg.get("log_interval", 1)
    requested_headless = runtime_cfg.headless or ("--headless" in sys.argv[1:])
    headless = resolve_headless_mode(requested_headless, os.environ)
    enable_cameras = False  # door geometry replaces visual embedding; cameras not needed

    print(f"📋 训练配置:")
    print(f"   并行环境数: {n_envs}")
    print(f"   总步数: {total_steps:,}")
    print(f"   每轮采集步数: {n_steps_per_rollout}")
    print(f"   无头模式: {headless}")
    print(f"   随机种子: {runtime_cfg.seed}")

    simulation_app = launch_simulation_app(
        headless=headless,
        enable_cameras=enable_cameras,
        import_error_message=(
            "未检测到 isaacsim 运行时，train.py 将继续使用当前 Python 环境。"
        ),
    )
    if simulation_app is not None:
        mode = "headless" if headless else "windowed"
        print(f"✅ Isaac Sim 已启动 ({mode})")

    # ── 构建环境 ─────────────────────────────────────────────
    # GPU 批量并行环境（DirectRLEnv）— 唯一训练路径
    from affordance_guided_interaction.envs.door_push_env import DoorPushEnv
    from affordance_guided_interaction.envs.direct_rl_env_adapter import DirectRLEnvAdapter

    env_cfg = build_env_cfg(
        cfg,
        n_envs,
        device=str(device),
        seed=runtime_cfg.seed,
        enable_cameras=enable_cameras,
    )
    _direct_env = DoorPushEnv(cfg=env_cfg)
    envs = DirectRLEnvAdapter(_direct_env)
    print(f"✅ 已创建 {n_envs} 个 GPU 批量并行环境 (DirectRLEnv)")

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
    randomizer = DomainRandomizer(seed=runtime_cfg.seed)

    # ── 注入 episode 重采样回调 ─────────────────────────────────
    # 让 auto-reset 时每个 done 的环境都能独立重新采样域参数和课程上下文，
    # 而不是复用旧缓存（修复 episode 级域随机化）。
    def _episode_reset_fn(env_idx: int):
        stage_cfg = curriculum.get_stage_config()
        domain_params = randomizer.sample_episode_params()
        door_type = str(randomizer._rng.choice(stage_cfg.door_types))
        left_occ, right_occ = stage_cfg.sample_occupancy_batch(1)
        return domain_params, door_type, left_occ[0], right_occ[0]

    envs.set_episode_reset_fn(_episode_reset_fn)

    # ── 兼容旧接口（no-op）─────────────────────────────────────
    # DoorPushEnv 内部已内置 action_noise_std 和 obs_noise_std，
    # 不依赖外部 randomizer 注入步级噪声。保留调用以兼容接口。
    envs.set_randomizer(randomizer)

    # ── 指标聚合器 ────────────────────────────────────────────
    from affordance_guided_interaction.training.metrics import TrainingMetrics
    metrics = TrainingMetrics()

    # ── TensorBoard ──────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=str(runtime_cfg.log_dir / run_name))
        print(f"📊 TensorBoard 日志: {runtime_cfg.log_dir / run_name}")
    except ImportError:
        print("⚠️ TensorBoard 不可用，跳过日志记录")

    # ── 恢复 checkpoint ──────────────────────────────────────
    start_iter = 0
    global_steps = 0
    best_success_rate = 0.0
    ckpt_dir = runtime_cfg.ckpt_dir

    if runtime_cfg.resume is not None:
        resume_path = runtime_cfg.resume
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

    if hasattr(envs, "reset_batch"):
        actor_obs_list, critic_obs_list = envs.reset_batch(
            domain_params_list=domain_params_list,
            door_types=door_types,
            left_occupied_list=left_occupied,
            right_occupied_list=right_occupied,
        )
    else:
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
            collect_start = time.time()
            actor_obs_list, critic_obs_list, collect_stats = collector.collect(
                envs=envs,
                n_steps=n_steps,
                current_actor_obs=actor_obs_list,
                current_critic_obs=critic_obs_list,
            )
            rollout_s = time.time() - collect_start

            # ── Step 2: 计算 GAE ──────────────────────────────
            buffer.compute_gae(
                gamma=ppo_cfg.gamma,
                lam=ppo_cfg.lam,
                last_values=collector.last_values,
                last_dones=collector.last_dones,
            )

            # ── Step 3: PPO 参数更新 ──────────────────────────
            update_start = time.time()
            update_metrics = ppo_trainer.update(buffer)
            update_s = time.time() - update_start

            # ── Step 4: 更新全局步数计数器 & 学习率衰减 ──────────
            steps_this_iter = n_envs * n_steps
            global_steps += steps_this_iter
            ppo_trainer.step_lr(global_steps)

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
                # 新阶段参数通过 _episode_reset_fn 在各 env 下一次 reset 时逐 env 注入，
                # 无需立即覆盖正在运行的 episode（H3 修复）

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
                    f"rollout={rollout_s:>5.2f}s | "
                    f"update={update_s:>5.2f}s | "
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
                writer.add_scalar("timing/rollout_s", rollout_s, global_steps)
                writer.add_scalar("timing/update_s", update_s, global_steps)
                writer.add_scalar(
                    "timing/env_steps_per_s",
                    steps_this_iter / max(rollout_s, 1e-6),
                    global_steps,
                )
                writer.add_scalar("collect/mean_reward", mean_reward, global_steps)
                for tag, val in _iter_reward_scalar_tags(collect_stats):
                    writer.add_scalar(tag, val, global_steps)
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

                # 写入聚合指标
                summary = metrics.summarize()
                for key, val in summary.items():
                    writer.add_scalar(f"metrics/{key}", val, global_steps)

            # metrics 无论是否有 writer 都必须重置，否则列表无限增长（H4 修复）
            metrics.reset()

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
    if simulation_app is not None:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
