"""Manual single-environment scene loader for DoorPush validation.

This script reuses the same environment construction path as training:

YAML -> build_env_cfg(...) -> gym.make(...) -> DoorPushEnv

The only difference is that the policy output is replaced by a persistent
12D action vector driven from an omni.ui control panel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train import DEFAULT_TASK_NAME, _ensure_tasks_registered, build_env_cfg, load_config
from affordance_guided_interaction.utils.runtime_env import resolve_headless_mode
from affordance_guided_interaction.utils.train_runtime_config import resolve_train_runtime_config

ACTION_DIM = 12
RESET_MODES = ("empty", "left", "right", "both")


def resolve_occupancy_mode(mode: str) -> tuple[bool, bool]:
    """Map a UI reset mode to left/right cup occupancy flags."""
    normalized = str(mode).strip().lower()
    if normalized == "empty":
        return False, False
    if normalized == "left":
        return True, False
    if normalized == "right":
        return False, True
    if normalized == "both":
        return True, True
    raise ValueError(f"Unsupported occupancy mode: {mode!r}")


def build_action_tensor(action_values: Sequence[float], *, device: str | torch.device) -> torch.Tensor:
    """Convert the current slider values into a single-env action batch."""
    if len(action_values) != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} action values, got {len(action_values)}.")
    return torch.tensor([list(action_values)], dtype=torch.float32, device=device)


class UiJointTargetDevice:
    """UI-backed manual device that mirrors IsaacLab teleop device semantics."""

    def __init__(self, *, sim_device: str | torch.device) -> None:
        self._sim_device = str(sim_device)
        self._action_values = [0.0] * ACTION_DIM
        self._callbacks: dict[str, Any] = {}

    @property
    def action_values(self) -> list[float]:
        return list(self._action_values)

    def reset(self) -> None:
        self._action_values[:] = [0.0] * ACTION_DIM

    def add_callback(self, key: str, func: Any) -> None:
        self._callbacks[key] = func

    def set_action_value(self, index: int, value: float) -> None:
        self._action_values[index] = float(value)

    def advance(self) -> torch.Tensor:
        return build_action_tensor(self._action_values, device=self._sim_device).squeeze(0)


class ManualDoorPushController:
    """Bridge between a manual device and the training environment."""

    def __init__(self, env: Any, simulation_app: Any, device: UiJointTargetDevice) -> None:
        self.env = env
        self.base_env = env.unwrapped
        self.simulation_app = simulation_app
        self.device = device
        self.paused = False
        self.pending_reset_mode: str | None = None
        self.last_step_info: dict[str, Any] = {}

    def request_reset(self, mode: str) -> None:
        resolve_occupancy_mode(mode)
        self.pending_reset_mode = mode

    def zero_actions(self) -> None:
        self.device.reset()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def tick(self) -> None:
        if self.pending_reset_mode is not None:
            self._apply_pending_reset()
            self.refresh_debug_state()
            return

        if self.paused:
            self.simulation_app.update()
            self.refresh_debug_state()
            return

        action = self.device.advance().unsqueeze(0)
        _, _, _, _, info = self.env.step(action)
        self.last_step_info = info if isinstance(info, dict) else {}
        self.refresh_debug_state()

    def initialize(self) -> None:
        self.request_reset("empty")
        self.tick()

    def refresh_debug_state(self) -> dict[str, Any]:
        self._debug_state = self.base_env.get_debug_state()
        return self._debug_state

    def get_debug_state(self) -> dict[str, Any]:
        return getattr(self, "_debug_state", self.refresh_debug_state())

    def _apply_pending_reset(self) -> None:
        assert self.pending_reset_mode is not None
        left_occ, right_occ = resolve_occupancy_mode(self.pending_reset_mode)
        self.base_env.set_occupancy(
            torch.tensor([left_occ], dtype=torch.bool, device=self.base_env.device),
            torch.tensor([right_occ], dtype=torch.bool, device=self.base_env.device),
        )
        self.env.reset()
        self.pending_reset_mode = None


def _build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Manual single-env DoorPush scene loader.")
    parser.add_argument("--configs-dir", type=str, default=None, help="Path to the project configs root or one YAML.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_NAME, help="Registered task name.")
    parser.add_argument("--device", type=str, default=None, help="Override the simulation device.")
    parser.add_argument("--seed", type=int, default=None, help="Override the environment seed.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _make_slider_row(
    *,
    ui: Any,
    label_text: str,
    model: Any,
    range_min: float,
    range_max: float,
    step: float,
    value_format: str = ".3f",
) -> Any:
    value_label = None
    with ui.HStack(height=28, spacing=4):
        ui.Label(label_text, width=150)

        def on_minus() -> None:
            model.set_value(max(range_min, model.get_value_as_float() - step))

        def on_plus() -> None:
            model.set_value(min(range_max, model.get_value_as_float() + step))

        ui.Button("-", width=24, clicked_fn=on_minus)
        ui.FloatSlider(model=model, min=range_min, max=range_max)
        ui.Button("+", width=24, clicked_fn=on_plus)
        value_label = ui.Label(f"{model.get_value_as_float():{value_format}}", width=70)
    return value_label


def format_joint_tracking_lines(
    *,
    joint_names: Sequence[str],
    target_values: Sequence[float],
    actual_values: Sequence[float],
) -> list[str]:
    """Format per-joint target / actual / delta lines for the status panel."""
    if not (len(joint_names) == len(target_values) == len(actual_values)):
        raise ValueError("joint_names, target_values, and actual_values must have the same length.")

    lines: list[str] = []
    for joint_name, target, actual in zip(joint_names, target_values, actual_values, strict=True):
        delta = float(actual) - float(target)
        lines.append(
            f"{_joint_slider_label(joint_name)}: "
            f"target={float(target):+.3f} actual={float(actual):+.3f} delta={delta:+.3f}"
        )
    return lines


def _format_status_text(controller: ManualDoorPushController) -> str:
    state = controller.get_debug_state()
    robot = controller.base_env.scene["robot"]
    arm_joint_names = [robot.joint_names[idx] for idx in controller.base_env._arm_joint_ids]
    arm_q = state["arm_joint_positions"][0].detach().cpu().tolist()
    arm_targets = state["arm_joint_targets"][0].detach().cpu().tolist()
    tracking_lines = format_joint_tracking_lines(
        joint_names=arm_joint_names,
        target_values=arm_targets,
        actual_values=arm_q,
    )
    lines = [
        f"paused: {controller.paused}",
        f"left_occupied: {bool(state['left_occupied'][0].item())}",
        f"right_occupied: {bool(state['right_occupied'][0].item())}",
        f"door_angle: {float(state['door_angle'][0].item()):.4f}",
        f"cup_dropped: {bool(state['cup_dropped'][0].item())}",
        f"episode_success: {bool(state['episode_success'][0].item())}",
        "",
        "joint_tracking:",
        *tracking_lines,
        "",
        "ui_action:",
        ", ".join(f"{value:+.3f}" for value in controller.device.action_values),
    ]
    return "\n".join(lines)


def _joint_slider_label(name: str) -> str:
    return (
        name.replace("left_", "L ")
        .replace("right_", "R ")
        .replace("joint", "J")
        .replace("Gripper", "Grip")
    )


def _build_ui(controller: ManualDoorPushController) -> tuple[list[Any], Any, Any]:
    import omni.ui as ui

    robot = controller.base_env.scene["robot"]
    arm_joint_ids = controller.base_env._arm_joint_ids
    arm_joint_names = [robot.joint_names[idx] for idx in arm_joint_ids]
    joint_limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids].detach().cpu()

    window = ui.Window("DoorPush Manual Validation", width=520, height=960)
    slider_models: list[Any] = []
    slider_labels: list[Any] = []
    status_label = None
    pause_button = None

    with window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=6):
                ui.Label("Reset / Cup Init", height=22)
                with ui.HStack(height=32, spacing=6):
                    ui.Button("Reset Empty", clicked_fn=lambda: controller.request_reset("empty"))
                    ui.Button("Reset Left Cup", clicked_fn=lambda: controller.request_reset("left"))
                with ui.HStack(height=32, spacing=6):
                    ui.Button("Reset Right Cup", clicked_fn=lambda: controller.request_reset("right"))
                    ui.Button("Reset Both Cup", clicked_fn=lambda: controller.request_reset("both"))

                ui.Spacer(height=6)

                with ui.HStack(height=32, spacing=6):
                    pause_button = ui.Button("Pause", clicked_fn=controller.toggle_pause)
                    ui.Button("Zero Actions", clicked_fn=controller.zero_actions)

                ui.Spacer(height=6)
                ui.Label("12D Action Sliders", height=22)

                for idx, joint_name in enumerate(arm_joint_names):
                    model = ui.SimpleFloatModel(0.0)
                    slider_models.append(model)
                    model.add_value_changed_fn(
                        lambda m, action_index=idx: controller.device.set_action_value(
                            action_index, m.get_value_as_float()
                        )
                    )
                    slider_labels.append(
                        _make_slider_row(
                            ui=ui,
                            label_text=_joint_slider_label(joint_name),
                            model=model,
                            range_min=float(joint_limits[idx, 0].item()),
                            range_max=float(joint_limits[idx, 1].item()),
                            step=0.02,
                            value_format=".3f",
                        )
                    )

                ui.Spacer(height=8)
                ui.Label("Status", height=22)
                status_label = ui.Label("", height=360, word_wrap=True)

    assert status_label is not None
    assert pause_button is not None
    return slider_models, slider_labels, status_label, pause_button


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args_cli = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    cfg = load_config(args_cli.configs_dir)
    runtime_cfg = resolve_train_runtime_config(cfg, project_root=PROJECT_ROOT)
    seed = args_cli.seed if args_cli.seed is not None else runtime_cfg.seed
    device = args_cli.device or runtime_cfg.device or "cuda:0"

    if "--headless" not in (argv or sys.argv[1:]):
        args_cli.headless = resolve_headless_mode(False, os.environ)
    else:
        args_cli.headless = resolve_headless_mode(bool(args_cli.headless), os.environ)
    args_cli.device = device
    args_cli.enable_cameras = False

    env_cfg = build_env_cfg(
        cfg,
        n_envs=1,
        device=device,
        seed=seed,
        task_name=args_cli.task,
    )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = None
    try:
        import gymnasium as gym

        _ensure_tasks_registered()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        device = UiJointTargetDevice(sim_device=env.unwrapped.device)
        controller = ManualDoorPushController(env, simulation_app, device)
        controller.initialize()

        slider_models, slider_labels, status_label, pause_button = _build_ui(controller)

        while simulation_app.is_running():
            for idx, model in enumerate(slider_models):
                value = model.get_value_as_float()
                slider_labels[idx].text = f"{value:+.3f}"

            controller.tick()
            pause_button.text = "Resume" if controller.paused else "Pause"
            status_label.text = _format_status_text(controller)

        env.close()
        env = None
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
