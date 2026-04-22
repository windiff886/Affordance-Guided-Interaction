"""Headless reset probe for deterministic DoorPush scene debugging.

This script reuses the same env construction path as training, but disables reset-time
events and drives occupancy/reset deterministically so reset state can be inspected
without the manual UI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import DEFAULT_TASK_NAME, _ensure_tasks_registered, build_env_cfg, load_config
from scripts.load_scene import (
    ACTION_DIM,
    _configure_env_cfg_for_manual_ui,
    resolve_occupancy_mode,
)
from affordance_guided_interaction.utils.runtime_env import configure_omniverse_client_environment
from affordance_guided_interaction.utils.train_runtime_config import resolve_train_runtime_config


def _build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Headless deterministic reset probe for DoorPush.")
    parser.add_argument("--configs-dir", type=str, default=None, help="Path to the project configs root or one YAML.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_NAME, help="Registered task name.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("empty", "left", "right", "both"),
        help="Reset occupancy modes to probe.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the environment seed.")
    parser.add_argument("--post-reset-steps", type=int, default=0, help="Zero-action steps after each reset.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _build_probe_snapshot(
    *,
    mode: str,
    state: dict[str, Any],
    cup_left_pos_w: torch.Tensor,
    cup_right_pos_w: torch.Tensor,
) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "left_occupied": bool(state["left_occupied"][0].item()),
        "right_occupied": bool(state["right_occupied"][0].item()),
        "cup_dropped": bool(state["cup_dropped"][0].item()),
        "episode_success": bool(state["episode_success"][0].item()),
        "door_angle": float(state["door_angle"][0].item()),
        "base_pos_w": tuple(float(v) for v in state["base_pos_w"][0].detach().cpu().tolist()),
        "planar_joint_positions": tuple(
            float(v) for v in state["planar_joint_positions"][0].detach().cpu().tolist()
        ),
        "cup_left_pos_w": tuple(float(v) for v in cup_left_pos_w[0].detach().cpu().tolist()),
        "cup_right_pos_w": tuple(float(v) for v in cup_right_pos_w[0].detach().cpu().tolist()),
    }


def _format_probe_snapshot(snapshot: dict[str, Any]) -> str:
    base_pos = snapshot["base_pos_w"]
    planar = snapshot["planar_joint_positions"]
    cup_left = snapshot["cup_left_pos_w"]
    cup_right = snapshot["cup_right_pos_w"]
    lines = [
        f"mode: {snapshot['mode']}",
        f"left_occupied: {snapshot['left_occupied']}",
        f"right_occupied: {snapshot['right_occupied']}",
        f"cup_dropped: {snapshot['cup_dropped']}",
        f"episode_success: {snapshot['episode_success']}",
        f"door_angle: {snapshot['door_angle']:+.4f}",
        f"base_pos_w: ({base_pos[0]:+.3f}, {base_pos[1]:+.3f}, {base_pos[2]:+.3f})",
        f"planar_joint_pos: ({planar[0]:+.3f}, {planar[1]:+.3f}, {planar[2]:+.3f})",
        f"cup_left_pos_w: ({cup_left[0]:+.3f}, {cup_left[1]:+.3f}, {cup_left[2]:+.3f})",
        f"cup_right_pos_w: ({cup_right[0]:+.3f}, {cup_right[1]:+.3f}, {cup_right[2]:+.3f})",
    ]
    return "\n".join(lines)


def _collect_mode_snapshot(env: Any, *, mode: str, post_reset_steps: int) -> dict[str, Any]:
    left_occ, right_occ = resolve_occupancy_mode(mode)
    base_env = env.unwrapped
    base_env.set_occupancy(
        torch.tensor([left_occ], dtype=torch.bool, device=base_env.device),
        torch.tensor([right_occ], dtype=torch.bool, device=base_env.device),
    )
    env.reset()

    if post_reset_steps > 0:
        zero_action = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=base_env.device)
        for _ in range(post_reset_steps):
            env.step(zero_action)

    state = base_env.get_debug_state()
    cup_left_pos_w = base_env.scene["cup_left"].data.root_pos_w.clone()
    cup_right_pos_w = base_env.scene["cup_right"].data.root_pos_w.clone()
    return _build_probe_snapshot(
        mode=mode,
        state=state,
        cup_left_pos_w=cup_left_pos_w,
        cup_right_pos_w=cup_right_pos_w,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args_cli = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    for mode in args_cli.modes:
        resolve_occupancy_mode(mode)

    cfg = load_config(args_cli.configs_dir)
    runtime_cfg = resolve_train_runtime_config(cfg, project_root=PROJECT_ROOT)
    seed = args_cli.seed if args_cli.seed is not None else runtime_cfg.seed
    device = args_cli.device or runtime_cfg.device or "cuda:0"

    configure_omniverse_client_environment(os.environ)
    args_cli.headless = True
    args_cli.enable_cameras = False
    args_cli.livestream = 0
    args_cli.device = device

    from isaaclab.app import AppLauncher
    import gymnasium as gym

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = None
    try:
        _ensure_tasks_registered()
        env_cfg = build_env_cfg(
            cfg,
            n_envs=1,
            device=device,
            seed=seed,
            task_name=args_cli.task,
        )
        _configure_env_cfg_for_manual_ui(env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

        for mode in args_cli.modes:
            snapshot = _collect_mode_snapshot(env, mode=mode, post_reset_steps=int(args_cli.post_reset_steps))
            print(_format_probe_snapshot(snapshot), flush=True)
            print("---", flush=True)

        env.close()
        env = None
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
