"""RL-Games training entrypoint aligned to the IsaacLab official flow.

Chain:
    YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher -> gym.make
    -> RlGamesVecEnvWrapper -> rl_games.Runner.load(...) -> runner.run(...)
"""

from __future__ import annotations

import argparse
import collections
import importlib
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from affordance_guided_interaction.utils.runtime_env import resolve_headless_mode
from affordance_guided_interaction.utils.rl_games_config import (
    build_rl_games_wrapper_kwargs,
    ensure_central_value_config,
)
from affordance_guided_interaction.utils.train_runtime_config import TrainRuntimeConfig, resolve_train_runtime_config

DEFAULT_TASK_NAME = "Affordance-DoorPush-Direct-v0"
DEFAULT_AGENT_ENTRY_POINT = "rl_games_cfg_entry_point"
_DEFAULT_LOG_ROOT = "rl_games"


def load_config(configs_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the active YAML configuration files for the rl_games-only workflow."""
    configs_dir = _resolve_configs_root(configs_dir)
    merged: dict[str, Any] = {}
    config_files = {
        "training": configs_dir / "training/default.yaml",
        "env": configs_dir / "env/default.yaml",
        "task": configs_dir / "task/default.yaml",
        "reward": configs_dir / "reward/default.yaml",
    }

    for key, path in config_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required config file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            merged[key] = yaml.safe_load(f) or {}

    return merged


def _resolve_configs_root(configs_dir: str | Path | None) -> Path:
    if configs_dir is None:
        return (_PROJECT_ROOT / "configs").resolve()

    path = Path(configs_dir).resolve()
    if path.is_file():
        return path.parents[1]

    if path.name in {"training", "env", "task", "reward"}:
        default_yaml = path / "default.yaml"
        if default_yaml.exists():
            return path.parent

    return path


def build_env_cfg(
    cfg: dict[str, Any],
    n_envs: int,
    *,
    device: str | None = None,
    seed: int | None = None,
    enable_cameras: bool = True,
    task_name: str | None = None,
    for_training: bool = False,
):
    """Build DoorPushEnvCfg from the registered default config plus YAML overrides."""
    del enable_cameras

    resolved_task_name = task_name or _resolve_task_name(cfg)
    env_cfg = _load_cfg_from_registry(resolved_task_name, "env_cfg_entry_point")
    if isinstance(env_cfg, dict):
        raise RuntimeError(f"Task '{resolved_task_name}' returned a dict env config; a config class is required.")

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

    control_cfg = env_cfg_yaml.get("control", {})
    if "action_type" in control_cfg:
        env_cfg.control_action_type = str(control_cfg["action_type"])
    if "arm_pd_stiffness" in control_cfg:
        env_cfg.arm_pd_stiffness = float(control_cfg["arm_pd_stiffness"])
    if "arm_pd_damping" in control_cfg:
        env_cfg.arm_pd_damping = float(control_cfg["arm_pd_damping"])
    if "position_target_noise_std" in control_cfg:
        env_cfg.position_target_noise_std = float(control_cfg["position_target_noise_std"])
    if "base_max_lin_vel_x" in control_cfg:
        env_cfg.base_max_lin_vel_x = float(control_cfg["base_max_lin_vel_x"])
    if "base_max_lin_vel_y" in control_cfg:
        env_cfg.base_max_lin_vel_y = float(control_cfg["base_max_lin_vel_y"])
    if "base_max_ang_vel_z" in control_cfg:
        env_cfg.base_max_ang_vel_z = float(control_cfg["base_max_ang_vel_z"])
    if "wheel_radius" in control_cfg:
        env_cfg.wheel_radius = float(control_cfg["wheel_radius"])
    if "wheel_base_half_length" in control_cfg:
        env_cfg.wheel_base_half_length = float(control_cfg["wheel_base_half_length"])
    if "wheel_base_half_width" in control_cfg:
        env_cfg.wheel_base_half_width = float(control_cfg["wheel_base_half_width"])
    if "wheel_velocity_limit" in control_cfg:
        env_cfg.wheel_velocity_limit = float(control_cfg["wheel_velocity_limit"])
    _apply_arm_control_to_actuators(env_cfg)

    task_cfg = cfg.get("task", {})
    if "door_angle_target" in task_cfg:
        env_cfg.door_angle_target = float(task_cfg["door_angle_target"])
    if "cup_drop_threshold" in task_cfg:
        env_cfg.cup_drop_threshold = float(task_cfg["cup_drop_threshold"])

    _inject_reward_params(env_cfg, cfg.get("reward", {}))
    if for_training:
        _apply_training_env_simplifications(env_cfg)
    return env_cfg


def build_rl_games_agent_cfg(
    cfg: dict[str, Any],
    runtime_cfg: TrainRuntimeConfig,
    *,
    task_name: str | None = None,
    agent_entry_point: str = DEFAULT_AGENT_ENTRY_POINT,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the rl_games agent config from the registered default YAML plus project YAML overrides."""
    resolved_task_name = task_name or _resolve_task_name(cfg)
    default_agent_cfg = _load_cfg_from_registry(resolved_task_name, agent_entry_point)
    if not isinstance(default_agent_cfg, dict):
        raise RuntimeError(f"Task '{resolved_task_name}' returned a non-dict rl_games agent config.")

    agent_cfg = deepcopy(default_agent_cfg)
    training_cfg = cfg.get("training", {})
    ppo_cfg = training_cfg.get("ppo", {})

    rl_device = _normalize_device(device or runtime_cfg.device)
    horizon_length = int(training_cfg.get("n_steps_per_rollout", agent_cfg["params"]["config"]["horizon_length"]))
    num_envs = int(training_cfg.get("num_envs", runtime_cfg.num_envs or 1))
    num_mini_batches = int(ppo_cfg.get("num_mini_batches", 1))
    total_batch_size = num_envs * horizon_length
    if total_batch_size % num_mini_batches != 0:
        raise ValueError(
            "The effective batch size must be divisible by num_mini_batches for rl_games. "
            f"Got total_batch_size={total_batch_size}, num_mini_batches={num_mini_batches}."
        )

    actor_lr = float(ppo_cfg.get("actor_lr", agent_cfg["params"]["config"]["learning_rate"]))
    critic_lr = float(ppo_cfg.get("critic_lr", actor_lr))
    if not math.isclose(actor_lr, critic_lr, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "The rl_games-only path uses a single optimizer. "
            f"Expected actor_lr == critic_lr, got {actor_lr} vs {critic_lr}."
        )

    total_steps = int(training_cfg.get("total_steps", 0))
    if total_steps <= 0:
        raise ValueError(f"training.total_steps must be positive, got {total_steps}.")
    max_epochs = math.ceil(total_steps / max(1, total_batch_size))

    params = agent_cfg.setdefault("params", {})
    env_section = params.setdefault("env", {})
    config_section = params.setdefault("config", {})

    params["seed"] = int(runtime_cfg.seed)
    env_section["clip_actions"] = float(training_cfg.get("clip_actions", env_section.get("clip_actions", 100.0)))
    env_section["clip_observations"] = float(
        training_cfg.get("clip_observations", env_section.get("clip_observations", 100.0))
    )

    config_section["name"] = str(training_cfg.get("experiment_name", config_section.get("name", "door_push_direct")))
    config_section["device"] = rl_device
    config_section["device_name"] = rl_device
    config_section["horizon_length"] = horizon_length
    config_section["minibatch_size"] = total_batch_size // num_mini_batches
    config_section["mini_epochs"] = int(ppo_cfg.get("num_epochs", config_section.get("mini_epochs", 1)))
    config_section["seq_length"] = int(ppo_cfg.get("seq_length", config_section.get("seq_length", 4)))
    config_section["normalize_advantage"] = bool(
        ppo_cfg.get("normalize_advantages", config_section.get("normalize_advantage", True))
    )
    config_section["gamma"] = float(ppo_cfg.get("gamma", config_section.get("gamma", 0.99)))
    config_section["tau"] = float(ppo_cfg.get("lam", config_section.get("tau", 0.95)))
    config_section["learning_rate"] = actor_lr
    if "lr_schedule" in ppo_cfg:
        config_section["lr_schedule"] = ppo_cfg["lr_schedule"]
    if "kl_threshold" in ppo_cfg:
        config_section["kl_threshold"] = float(ppo_cfg["kl_threshold"])
    if "adaptive_lr_max" in ppo_cfg:
        config_section["adaptive_lr_max"] = float(ppo_cfg["adaptive_lr_max"])
    if "adaptive_lr_min" in ppo_cfg:
        config_section["adaptive_lr_min"] = float(ppo_cfg["adaptive_lr_min"])
    config_section["entropy_coef"] = float(ppo_cfg.get("entropy_coef", config_section.get("entropy_coef", 0.0)))
    config_section["critic_coef"] = float(ppo_cfg.get("value_coef", config_section.get("critic_coef", 1.0)))
    config_section["grad_norm"] = float(ppo_cfg.get("max_grad_norm", config_section.get("grad_norm", 1.0)))
    config_section["e_clip"] = float(ppo_cfg.get("clip_eps", config_section.get("e_clip", 0.2)))
    config_section["clip_value"] = bool(
        ppo_cfg.get("use_clipped_value_loss", config_section.get("clip_value", True))
    )
    config_section["max_epochs"] = max_epochs
    config_section["save_frequency"] = int(
        training_cfg.get("checkpoint_interval", config_section.get("save_frequency", 100))
    )
    config_section["save_best_after"] = min(config_section["save_frequency"], max_epochs)
    config_section["use_diagnostics"] = bool(training_cfg.get("use_diagnostics", True))

    reward_shaper = config_section.setdefault("reward_shaper", {})
    reward_shaper["scale_value"] = float(training_cfg.get("reward_scale", reward_shaper.get("scale_value", 1.0)))

    continuous_space = params.setdefault("network", {}).setdefault("space", {}).setdefault("continuous", {})
    sigma_init = continuous_space.setdefault("sigma_init", {})
    if "sigma_init_logstd" in ppo_cfg:
        sigma_init["val"] = float(ppo_cfg["sigma_init_logstd"])
    elif "sigma_init_val" in ppo_cfg:
        sigma_init["val"] = float(ppo_cfg["sigma_init_val"])

    ensure_central_value_config(agent_cfg)

    resume_path = Path(checkpoint_path) if checkpoint_path is not None else runtime_cfg.resume
    params["load_checkpoint"] = resume_path is not None
    params["load_path"] = str(resume_path) if resume_path is not None else ""
    return agent_cfg


def _configure_rl_games_adaptive_scheduler_bounds(config_section: dict[str, Any]) -> None:
    """Patch rl_games' adaptive LR scheduler to honor project-level bounds.

    rl_games hard-codes AdaptiveScheduler bounds to [1e-6, 1e-2].  The
    project config keeps those defaults unless adaptive_lr_min/max are set.
    """
    adaptive_lr_min = config_section.get("adaptive_lr_min")
    adaptive_lr_max = config_section.get("adaptive_lr_max")
    if adaptive_lr_min is None and adaptive_lr_max is None:
        return

    min_lr = float(adaptive_lr_min) if adaptive_lr_min is not None else None
    max_lr = float(adaptive_lr_max) if adaptive_lr_max is not None else None
    if min_lr is not None and min_lr <= 0.0:
        raise ValueError(f"adaptive_lr_min must be positive, got {min_lr}.")
    if max_lr is not None and max_lr <= 0.0:
        raise ValueError(f"adaptive_lr_max must be positive, got {max_lr}.")
    if min_lr is not None and max_lr is not None and min_lr > max_lr:
        raise ValueError(f"adaptive_lr_min must be <= adaptive_lr_max, got {min_lr} > {max_lr}.")

    from rl_games.common import schedulers

    scheduler_cls = schedulers.AdaptiveScheduler
    if not hasattr(scheduler_cls, "_door_push_original_init"):
        scheduler_cls._door_push_original_init = scheduler_cls.__init__

        def _bounded_init(self, kl_threshold: float = 0.008):
            scheduler_cls._door_push_original_init(self, kl_threshold)
            patched_min = getattr(scheduler_cls, "_door_push_min_lr", None)
            patched_max = getattr(scheduler_cls, "_door_push_max_lr", None)
            if patched_min is not None:
                self.min_lr = patched_min
            if patched_max is not None:
                self.max_lr = patched_max

        scheduler_cls.__init__ = _bounded_init

    scheduler_cls._door_push_min_lr = min_lr
    scheduler_cls._door_push_max_lr = max_lr


_REWARD_PARAM_MAP: dict[str, dict[str, str]] = {
    "task": {
        "w_delta": "rew_w_delta",
        "alpha": "rew_alpha",
        "k_decay": "rew_k_decay",
        "w_open": "rew_w_open",
        "w_approach": "rew_w_approach",
        "approach_eps": "rew_approach_eps",
        "approach_stop_angle": "rew_approach_stop_angle",
        "w_base_approach": "rew_w_base_approach",
        "base_approach_open_gate": "rew_base_approach_open_gate",
        "w_base_cross": "rew_w_base_cross",
        "base_cross_open_gate": "rew_base_cross_open_gate",
    },
    "stability": {
        "w_zero_acc": "rew_w_zero_acc",
        "lambda_acc": "rew_lambda_acc",
        "w_zero_ang": "rew_w_zero_ang",
        "lambda_ang": "rew_lambda_ang",
        "w_acc": "rew_w_acc",
        "w_ang": "rew_w_ang",
        "w_tilt": "rew_w_tilt",
    },
    "safety": {
        "mu": "rew_mu",
        "beta_vel": "rew_beta_vel",
        "beta_target": "rew_beta_target",
        "target_margin_ratio": "rew_target_margin_ratio",
        "beta_joint_move": "rew_beta_joint_move",
        "beta_cup_door_prox": "rew_beta_cup_door_prox",
        "cup_door_prox_threshold": "rew_cup_door_prox_threshold",
        "w_drop": "rew_w_drop",
        "w_base_zero_speed": "rew_w_base_zero_speed",
        "lambda_base_speed": "rew_lambda_base_speed",
        "w_base_speed": "rew_w_base_speed",
        "beta_base_cmd": "rew_beta_base_cmd",
        "beta_base_heading": "rew_beta_base_heading",
        "beta_base_corridor": "rew_beta_base_corridor",
        "base_align_tolerance_deg": "rew_base_align_tolerance_deg",
    },
}


def _apply_arm_control_to_actuators(env_cfg: Any) -> None:
    if getattr(env_cfg, "control_action_type", "joint_position") != "joint_position":
        raise ValueError(f"Unsupported control action type: {env_cfg.control_action_type}")

    robot_actuators = env_cfg.scene.robot.actuators
    shoulder = robot_actuators["shoulder_joints"]
    arm = robot_actuators["arm_joints"]
    wheel = robot_actuators.get("wheel_joints")

    shoulder.stiffness = float(env_cfg.arm_pd_stiffness)
    shoulder.damping = float(env_cfg.arm_pd_damping)
    arm.stiffness = float(env_cfg.arm_pd_stiffness)
    arm.damping = float(env_cfg.arm_pd_damping)

    effort_limits = tuple(getattr(env_cfg, "effort_limits", ()))
    if len(effort_limits) >= 8:
        shoulder_indices = (1, 7)
        shoulder_limits = [effort_limits[i] for i in shoulder_indices]
        arm_limits = [limit for idx, limit in enumerate(effort_limits) if idx not in shoulder_indices]
        if shoulder_limits:
            shoulder.effort_limit = float(max(shoulder_limits))
        if arm_limits:
            arm.effort_limit = float(max(arm_limits))
    if wheel is not None:
        wheel.velocity_limit = float(getattr(env_cfg, "wheel_velocity_limit", wheel.velocity_limit))


def _apply_training_env_simplifications(env_cfg: Any) -> None:
    env_cfg.base_control_backend = "planar_joint_velocity"
    env_cfg.training_planar_base_only = True
    env_cfg.emit_wheel_debug_state = False


def _inject_reward_params(env_cfg: Any, reward_cfg: dict[str, Any]) -> None:
    for section, mapping in _REWARD_PARAM_MAP.items():
        section_cfg = reward_cfg.get(section, {})
        for yaml_key, cfg_attr in mapping.items():
            if yaml_key in section_cfg:
                setattr(env_cfg, cfg_attr, float(section_cfg[yaml_key]))


def main(argv: list[str] | None = None) -> int:
    """Train DoorPush with the IsaacLab rl_games downstream flow."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_arg_parser()
    args_cli = parser.parse_args(argv_list)

    cfg = load_config(args_cli.configs_dir)
    runtime_cfg = resolve_train_runtime_config(cfg, project_root=_PROJECT_ROOT)

    task_name = args_cli.task or _resolve_task_name(cfg)
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else _resolve_num_envs(cfg, runtime_cfg)
    seed = args_cli.seed if args_cli.seed is not None else runtime_cfg.seed
    cli_device_explicit = getattr(args_cli, "device_explicit", False) or _cli_option_present(argv_list, "--device")
    device = _normalize_device(args_cli.device if cli_device_explicit else runtime_cfg.device)
    checkpoint_path = args_cli.checkpoint or runtime_cfg.resume

    if not _cli_option_present(argv_list, "--headless"):
        args_cli.headless = resolve_headless_mode(runtime_cfg.headless, os.environ)
    else:
        args_cli.headless = resolve_headless_mode(bool(args_cli.headless), os.environ)
    if not _cli_option_present(argv_list, "--device"):
        args_cli.device = device
    if args_cli.video:
        args_cli.enable_cameras = True

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env_cfg = build_env_cfg(
        cfg,
        n_envs=num_envs,
        device=device,
        seed=seed,
        task_name=task_name,
        for_training=True,
    )
    runtime_cfg = TrainRuntimeConfig(
        headless=args_cli.headless,
        device=device,
        seed=seed,
        resume=Path(checkpoint_path) if checkpoint_path is not None else None,
        log_dir=runtime_cfg.log_dir,
        num_envs=num_envs,
    )
    agent_cfg = build_rl_games_agent_cfg(
        cfg,
        runtime_cfg,
        task_name=task_name,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    try:
        import gymnasium as gym
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner

        from affordance_guided_interaction.utils.rl_games_observer import DoorPushTensorboardObserver
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        _ensure_tasks_registered()

        config_name = agent_cfg["params"]["config"]["name"]
        log_root_path = (runtime_cfg.log_dir / _DEFAULT_LOG_ROOT / config_name).resolve()
        run_name = agent_cfg["params"]["config"].get("full_experiment_name") or datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        run_dir = log_root_path / run_name

        agent_cfg["params"]["config"]["train_dir"] = str(log_root_path)
        agent_cfg["params"]["config"]["full_experiment_name"] = run_name
        env_cfg.log_dir = str(run_dir)

        dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
        dump_yaml(str(run_dir / "params" / "project.yaml"), cfg)

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

        env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            video_kwargs = {
                "video_folder": str(run_dir / "videos" / "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()
        env = RlGamesVecEnvWrapper(
            env,
            rl_device,
            clip_obs,
            clip_actions,
            **build_rl_games_wrapper_kwargs(),
        )
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        runner = Runner(DoorPushTensorboardObserver())
        _configure_rl_games_adaptive_scheduler_bounds(agent_cfg["params"]["config"])
        runner.load(agent_cfg)
        runner.reset()

        run_kwargs = {"train": True, "play": False, "sigma": None}
        if checkpoint_path is not None:
            run_kwargs["checkpoint"] = str(checkpoint_path)
        runner.run(run_kwargs)

        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        env.close()
        return 0
    finally:
        simulation_app.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train DoorPush with IsaacLab rl_games flow.")
    parser.add_argument("--configs-dir", type=str, default=None, help="Path to the project configs root or one YAML.")
    parser.add_argument("--task", type=str, default=None, help="Registered task name.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override the number of environments.")
    parser.add_argument("--seed", type=int, default=None, help="Override the training seed.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a checkpoint path.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Step interval between video captures.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict[str, Any] | object:
    import gymnasium as gym

    _ensure_tasks_registered()
    spec = gym.spec(task_name.split(":")[-1])
    cfg_entry_point = spec.kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        agents = collections.defaultdict(list)
        for key in spec.kwargs:
            if key.endswith("_cfg_entry_point") and key != "env_cfg_entry_point":
                parts = key.replace("_cfg_entry_point", "").replace("rl_games", "rl-games").split("_")
                agent = parts[0].replace("-", "_")
                algorithms = [item.upper() for item in (parts[1:] if len(parts) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)
        details = ""
        if agents:
            details = "\nExisting RL config entry points:"
            for agent, algorithms in agents.items():
                details += f"\n  |-- {agent}: {', '.join(algorithms)}"
        raise ValueError(
            f"Could not find configuration for task '{task_name}' and entry point '{entry_point_key}'.{details}"
        )

    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            config_file = cfg_entry_point
        else:
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            config_file = os.path.join(mod_path, file_name)
        with open(config_file, encoding="utf-8") as f:
            return yaml.full_load(f)

    if callable(cfg_entry_point):
        cfg_cls = cfg_entry_point()
    elif isinstance(cfg_entry_point, str):
        mod_name, attr_name = cfg_entry_point.split(":")
        mod = importlib.import_module(mod_name)
        cfg_cls = getattr(mod, attr_name)
    else:
        cfg_cls = cfg_entry_point

    if callable(cfg_cls):
        return cfg_cls()
    return cfg_cls


def _ensure_tasks_registered() -> None:
    import affordance_guided_interaction.tasks  # noqa: F401


def _resolve_task_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("training", {}).get("task", DEFAULT_TASK_NAME))


def _resolve_num_envs(cfg: dict[str, Any], runtime_cfg: TrainRuntimeConfig) -> int:
    if runtime_cfg.num_envs is not None:
        return int(runtime_cfg.num_envs)
    if "num_envs" in cfg.get("training", {}):
        return int(cfg["training"]["num_envs"])
    raise ValueError("No num_envs configured in training YAML.")


def _normalize_device(device: str | None) -> str:
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    text = str(device).strip()
    if text == "cuda":
        return "cuda:0"
    return text


def _cli_option_present(argv: list[str], option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in argv)


if __name__ == "__main__":
    raise SystemExit(main())
