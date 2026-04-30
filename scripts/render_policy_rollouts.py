"""Headless checkpoint rollout renderer for the DoorPush task.

This script follows the official IsaacLab RL-Games play flow:

YAML -> build_env_cfg/build_rl_games_agent_cfg -> AppLauncher ->
gym.make(render_mode="rgb_array") -> RecordVideo -> RlGamesVecEnvWrapper ->
Runner.load/create_player -> obs_to_torch/get_action/env.step

The project-specific additions are:

- parameters come from the local YAML tree
- one clip is exported per configured mode
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import (
    DEFAULT_TASK_NAME,
    _ensure_tasks_registered,
    build_env_cfg,
    build_rl_games_agent_cfg,
    load_config,
)
from affordance_guided_interaction.utils.rl_games_config import build_rl_games_wrapper_kwargs
from affordance_guided_interaction.utils.train_runtime_config import TrainRuntimeConfig


DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference" / "default.yaml"
DEFAULT_VIDEO_LENGTH = 200
_ROLLOUT_PROGRESS_INTERVAL = 100
_EPISODE_END_REASON_PRIORITY = (
    ("success", "success"),
    ("fail_timeout", "timeout"),
    ("hard_collision", "hard_collision_at_end"),
    ("reverse_open", "reverse_open_at_end"),
    ("fail_not_passed", "not_passed"),
)


@dataclass(frozen=True)
class RolloutRenderConfig:
    headless: bool
    device: str
    seed: int
    checkpoint: str
    task_name: str
    modes: tuple[str, ...]
    output_root: str
    folder_name: str
    deterministic: bool
    video_length: int
    camera_eye: tuple[float, float, float] | None = None
    camera_lookat: tuple[float, float, float] | None = None


def load_inference_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_optional_vector3(value: Any, *, key_path: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key_path} must be a 3-element list/tuple, got {value!r}.")
    return tuple(float(component) for component in value)


def resolve_rollout_config(
    inference_cfg: dict[str, Any],
    *,
    checkpoint_override: str | None,
    device_override: str | None,
    task_override: str | None,
) -> RolloutRenderConfig:
    runtime_cfg = inference_cfg.get("runtime", {})
    policy_cfg = inference_cfg.get("policy", {})
    rollout_cfg = inference_cfg.get("rollout", {})
    video_cfg = inference_cfg.get("video", {})
    camera_cfg = video_cfg.get("camera", {})

    checkpoint = str(checkpoint_override or policy_cfg.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("Inference config must provide policy.checkpoint or --checkpoint.")

    raw_modes = rollout_cfg.get("modes", ("default",))
    modes = tuple(str(mode).strip().lower() for mode in raw_modes)
    if not modes:
        raise ValueError("Inference config must provide at least one rollout mode.")

    video_length = int(video_cfg.get("length", DEFAULT_VIDEO_LENGTH))
    if video_length <= 0:
        raise ValueError(f"video.length must be positive, got {video_length}.")

    return RolloutRenderConfig(
        headless=bool(runtime_cfg.get("headless", True)),
        device=str(device_override or runtime_cfg.get("device", "cuda")).strip(),
        seed=int(runtime_cfg.get("seed", 42)),
        checkpoint=checkpoint,
        task_name=str(task_override or rollout_cfg.get("task_name", DEFAULT_TASK_NAME)).strip(),
        modes=modes,
        output_root=str(video_cfg.get("output_root", "runs/inference")).strip(),
        folder_name=str(video_cfg.get("folder_name", "policy_rollouts")).strip(),
        deterministic=bool(policy_cfg.get("deterministic", True)),
        video_length=video_length,
        camera_eye=_parse_optional_vector3(camera_cfg.get("eye"), key_path="video.camera.eye"),
        camera_lookat=_parse_optional_vector3(camera_cfg.get("lookat"), key_path="video.camera.lookat"),
    )


def _mode_output_dir(output_root: Path, mode: str) -> Path:
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    return mode_dir


def _should_log_rollout_step(*, step: int, interval: int = _ROLLOUT_PROGRESS_INTERVAL) -> bool:
    return step == 1 or (step > 0 and step % interval == 0)


def _format_rollout_event(*, mode: str, event: str, step: int | None = None) -> str:
    message = f"[ROLLOUT] mode={mode} event={event}"
    if step is not None:
        message += f" step={step}"
    return message


def _format_rollout_progress(*, mode: str, event: str, step: int, elapsed_s: float) -> str:
    return f"[ROLLOUT] mode={mode} event={event} step={step} elapsed={elapsed_s:.1f}s"


def _log_rollout_message(message: str) -> None:
    print(message, flush=True)


def _build_record_video_kwargs(mode_dir: Path, *, video_length: int) -> dict[str, Any]:
    return {
        "video_folder": str(mode_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "disable_logger": True,
    }


def _as_any_true(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(torch.any(value.to(dtype=torch.bool)).item())
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return False
    return bool(torch.any(tensor.to(dtype=torch.bool)).item())


def _resolve_episode_end_reason(infos: Any) -> str:
    if not isinstance(infos, dict):
        return "episode_done"
    for info_key, reason in _EPISODE_END_REASON_PRIORITY:
        if _as_any_true(infos.get(info_key)):
            return reason
    return "episode_done"


def _sanitize_episode_end_reason(reason: str) -> str:
    safe_reason = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in reason.strip().lower())
    safe_reason = safe_reason.strip("_")
    return safe_reason or "unknown"


def _create_episode_end_reason_marker(mode_dir: Path, reason: str) -> Path:
    mode_dir.mkdir(parents=True, exist_ok=True)
    marker_path = mode_dir / f"{_sanitize_episode_end_reason(reason)}.txt"
    marker_path.write_text("", encoding="utf-8")
    return marker_path


def _extract_policy_obs(obs: Any) -> Any:
    if isinstance(obs, dict) and "obs" in obs:
        return obs["obs"]
    return obs


def _episode_finished(dones: Any) -> bool:
    if isinstance(dones, torch.Tensor):
        return bool(torch.any(dones).item())
    return bool(torch.any(torch.as_tensor(dones)).item())


def _play_mode_rollout(
    *,
    simulation_app: Any,
    env: Any,
    agent: Any,
    mode: str,
    deterministic: bool,
    video_length: int,
    reason_marker_dir: Path | None = None,
) -> int:
    _log_rollout_message(_format_rollout_event(mode=mode, event="start"))

    agent.reset()

    obs = env.reset()
    obs = _extract_policy_obs(obs)

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    start_time = time.perf_counter()
    timestep = 0
    end_reason = "video_length"
    while simulation_app.is_running() and timestep < video_length:
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=deterministic)
            obs, _, dones, infos = env.step(actions)

            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        timestep += 1
        if _should_log_rollout_step(step=timestep):
            elapsed_s = time.perf_counter() - start_time
            _log_rollout_message(_format_rollout_progress(mode=mode, event="step", step=timestep, elapsed_s=elapsed_s))
        if _episode_finished(dones):
            end_reason = _resolve_episode_end_reason(infos)
            break
    if not simulation_app.is_running() and timestep < video_length:
        end_reason = "simulation_stopped"

    elapsed_s = time.perf_counter() - start_time
    _log_rollout_message(_format_rollout_progress(mode=mode, event="finished", step=timestep, elapsed_s=elapsed_s))
    if reason_marker_dir is not None:
        marker_path = _create_episode_end_reason_marker(reason_marker_dir, end_reason)
        _log_rollout_message(f"{_format_rollout_event(mode=mode, event='end_reason')} reason={end_reason} marker={marker_path}")
    return timestep


def _register_rl_games_env(env: Any) -> None:
    from rl_games.common import env_configurations, vecenv

    from isaaclab_rl.rl_games import RlGamesGpuEnv

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


def _stop_recording_if_needed(env: Any) -> None:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        stop_recording = getattr(current, "stop_recording", None)
        if callable(stop_recording) and getattr(current, "recording", False):
            stop_recording()
            return
        current = getattr(current, "env", None)


def _terminate_worker_process(exit_code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def _build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Render checkpoint rollouts for DoorPush.")
    parser.add_argument("--configs-dir", type=str, default=None, help="Path to the project configs root or one YAML.")
    parser.add_argument(
        "--inference-config",
        type=str,
        default=str(DEFAULT_INFERENCE_CONFIG),
        help="Path to the inference YAML file.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override.")
    parser.add_argument("--task", type=str, default=None, help="Optional task-name override.")
    parser.add_argument("--single-mode", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, default=None, help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _resolve_run_dir(rollout_cfg: RolloutRenderConfig) -> Path:
    root = Path(rollout_cfg.output_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / rollout_cfg.folder_name / run_name
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    return run_dir


def _configure_app_launcher_args(args_cli: argparse.Namespace, rollout_cfg: RolloutRenderConfig) -> None:
    args_cli.headless = bool(rollout_cfg.headless)
    args_cli.enable_cameras = True
    args_cli.device = rollout_cfg.device


def _build_mode_command(script_args: list[str], *, mode: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *script_args,
        "--single-mode",
        mode,
        "--run-dir",
        str(run_dir),
    ]


def _apply_rollout_camera(env_cfg: Any, rollout_cfg: RolloutRenderConfig) -> None:
    viewer = getattr(env_cfg, "viewer", None)
    if viewer is None:
        return
    if rollout_cfg.camera_eye is not None:
        viewer.eye = rollout_cfg.camera_eye
    if rollout_cfg.camera_lookat is not None:
        viewer.lookat = rollout_cfg.camera_lookat


def _build_mode_env(
    *,
    cfg: dict[str, Any],
    rollout_cfg: RolloutRenderConfig,
    mode_dir: Path,
    log_dir: Path,
    rl_device: str,
    clip_obs: float,
    clip_actions: float,
) -> Any:
    import gymnasium as gym
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    env_cfg = build_env_cfg(
        cfg,
        n_envs=1,
        device=rollout_cfg.device,
        seed=rollout_cfg.seed,
        task_name=rollout_cfg.task_name,
    )
    env_cfg.log_dir = str(log_dir)
    env_cfg.events = None
    _apply_rollout_camera(env_cfg, rollout_cfg)

    env = gym.make(rollout_cfg.task_name, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, **_build_record_video_kwargs(mode_dir, video_length=rollout_cfg.video_length))
    return RlGamesVecEnvWrapper(
        env,
        rl_device,
        clip_obs,
        clip_actions,
        **build_rl_games_wrapper_kwargs(),
    )


def _build_player(
    *,
    base_agent_cfg: dict[str, Any],
    checkpoint_path: str,
    env: Any,
) -> Any:
    from rl_games.torch_runner import Runner

    agent_cfg = copy.deepcopy(base_agent_cfg)
    params = agent_cfg.setdefault("params", {})
    config_section = params.setdefault("config", {})
    config_section["num_actors"] = env.unwrapped.num_envs
    params["load_checkpoint"] = True
    params["load_path"] = checkpoint_path

    _register_rl_games_env(env)

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(checkpoint_path)
    return player


def _render_all_modes_official(
    *,
    simulation_app: Any,
    cfg: dict[str, Any],
    rollout_cfg: RolloutRenderConfig,
    base_agent_cfg: dict[str, Any],
    checkpoint_path: str,
    output_root: Path,
    log_dir: Path,
) -> None:
    rl_device = base_agent_cfg["params"]["config"]["device"]
    clip_obs = base_agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = base_agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    for mode in rollout_cfg.modes:
        mode_dir = _mode_output_dir(output_root, mode)
        _log_rollout_message(f"{_format_rollout_event(mode=mode, event='env_setup')} dir={mode_dir}")
        env = _build_mode_env(
            cfg=cfg,
            rollout_cfg=rollout_cfg,
            mode_dir=mode_dir,
            log_dir=log_dir,
            rl_device=rl_device,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
        )
        try:
            player = _build_player(
                base_agent_cfg=base_agent_cfg,
                checkpoint_path=checkpoint_path,
                env=env,
            )
            _play_mode_rollout(
                simulation_app=simulation_app,
                env=env,
                agent=player,
                mode=mode,
                deterministic=rollout_cfg.deterministic,
                video_length=rollout_cfg.video_length,
                reason_marker_dir=mode_dir,
            )
        finally:
            _stop_recording_if_needed(env)


def _run_single_mode_worker(
    *,
    args_cli: argparse.Namespace,
    rollout_cfg: RolloutRenderConfig,
    cfg: dict[str, Any],
    checkpoint_path: Path,
    run_dir: Path,
    mode: str,
) -> int:
    worker_rollout_cfg = replace(rollout_cfg, modes=(mode,))
    _configure_app_launcher_args(args_cli, worker_rollout_cfg)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    exit_code = 0

    try:
        _ensure_tasks_registered()
        agent_runtime_cfg = TrainRuntimeConfig(
            headless=worker_rollout_cfg.headless,
            device=worker_rollout_cfg.device,
            seed=worker_rollout_cfg.seed,
            resume=None,
            log_dir=run_dir,
            num_envs=1,
        )
        base_agent_cfg = build_rl_games_agent_cfg(
            cfg,
            agent_runtime_cfg,
            task_name=worker_rollout_cfg.task_name,
            device=worker_rollout_cfg.device,
            checkpoint_path=None,
        )
        _render_all_modes_official(
            simulation_app=simulation_app,
            cfg=cfg,
            rollout_cfg=worker_rollout_cfg,
            base_agent_cfg=base_agent_cfg,
            checkpoint_path=str(checkpoint_path),
            output_root=run_dir / "videos",
            log_dir=run_dir,
        )
    except Exception:
        exit_code = 1
        traceback.print_exc()

    _log_rollout_message(f"{_format_rollout_event(mode=mode, event='worker_exit')} returncode={exit_code}")
    _terminate_worker_process(exit_code)
    return exit_code


def _run_driver(
    *,
    script_args: list[str],
    rollout_cfg: RolloutRenderConfig,
    run_dir: Path,
) -> int:
    for mode in rollout_cfg.modes:
        _log_rollout_message(f"{_format_rollout_event(mode=mode, event='spawn')} dir={run_dir / 'videos' / mode}")
        command = _build_mode_command(script_args, mode=mode, run_dir=run_dir)
        result = subprocess.run(command, check=False)
        _log_rollout_message(
            f"{_format_rollout_event(mode=mode, event='child_exit')} returncode={result.returncode}"
        )
        if result.returncode != 0:
            return result.returncode
        _log_rollout_message(f"{_format_rollout_event(mode=mode, event='mode_complete')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    script_args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_arg_parser()
    args_cli = parser.parse_args(script_args)

    inference_cfg = load_inference_config(args_cli.inference_config)
    rollout_cfg = resolve_rollout_config(
        inference_cfg,
        checkpoint_override=args_cli.checkpoint,
        device_override=args_cli.device,
        task_override=args_cli.task,
    )

    checkpoint_path = Path(rollout_cfg.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    cfg = load_config(args_cli.configs_dir)
    run_dir = Path(args_cli.run_dir).expanduser().resolve() if args_cli.run_dir else _resolve_run_dir(rollout_cfg)

    if args_cli.single_mode is None:
        return _run_driver(script_args=script_args, rollout_cfg=rollout_cfg, run_dir=run_dir)

    mode = str(args_cli.single_mode).strip().lower()
    return _run_single_mode_worker(
        args_cli=args_cli,
        rollout_cfg=rollout_cfg,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        mode=mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
