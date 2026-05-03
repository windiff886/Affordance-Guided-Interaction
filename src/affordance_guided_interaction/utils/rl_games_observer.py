from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver


class DoorPushTensorboardObserver(IsaacAlgoObserver):
    """Augment rl_games logging with push-door episode statistics.

    Aligned with ``docs/training_pipeline_detailed.md``:
    target success/reward/task/randomization tags.
    """

    def after_init(self, algo):
        super().after_init(algo)
        self._reset_custom_stats()

    def after_clear_stats(self):
        super().after_clear_stats()
        self._reset_custom_stats()

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)

        if not isinstance(infos, dict):
            return

        done_env_ids = self._normalize_done_indices(done_indices)
        if done_env_ids.numel() == 0:
            return

        reward_info = infos.get("episode_reward_info")
        if reward_info is None:
            reward_info = infos.get("reward_info")
        if isinstance(reward_info, dict):
            for key, value in reward_info.items():
                selected = self._select_done_values(value, done_env_ids)
                if selected is None:
                    continue
                self._reward_sums[key] += selected.sum().item()
                self._reward_counts[key] += int(selected.numel())

        # Success metrics
        success = self._select_done_values(infos.get("success"), done_env_ids, cast_float=False)
        if success is not None:
            success = success.to(dtype=torch.bool)
            self._success_counts["rate"] += int(success.sum().item())
            self._success_totals["rate"] += int(success.numel())

        # Opened enough / passed through / no collision rates
        opened_enough = self._select_done_values(infos.get("opened_enough"), done_env_ids, cast_float=False)
        if opened_enough is not None:
            opened_enough = opened_enough.to(dtype=torch.bool)
            self._success_counts["opened_enough_rate"] += int(opened_enough.sum().item())
            self._success_totals["opened_enough_rate"] += int(opened_enough.numel())

        passed_through = self._select_done_values(infos.get("passed_through"), done_env_ids, cast_float=False)
        if passed_through is not None:
            passed_through = passed_through.to(dtype=torch.bool)
            self._success_counts["passed_through_rate"] += int(passed_through.sum().item())
            self._success_totals["passed_through_rate"] += int(passed_through.numel())

        no_collision = self._select_done_values(infos.get("no_collision"), done_env_ids, cast_float=False)
        if no_collision is not None:
            no_collision = no_collision.to(dtype=torch.bool)
            self._success_counts["no_collision_rate"] += int(no_collision.sum().item())
            self._success_totals["no_collision_rate"] += int(no_collision.numel())

        # Task state
        door_angle = self._select_done_values(infos.get("door_angle"), done_env_ids)
        if door_angle is not None:
            self._door_angle_sum += door_angle.sum().item()
            self._door_angle_count += int(door_angle.numel())

        base_cross_progress = self._select_done_values(infos.get("base_cross_progress"), done_env_ids)
        if base_cross_progress is not None:
            self._cross_progress_sum += base_cross_progress.sum().item()
            self._cross_progress_count += int(base_cross_progress.numel())

        door_angular_velocity = self._select_done_values(infos.get("door_angular_velocity"), done_env_ids)
        if door_angular_velocity is not None:
            self._door_angular_velocity_sum += door_angular_velocity.sum().item()
            self._door_angular_velocity_count += int(door_angular_velocity.numel())

        stage = self._select_done_values(infos.get("stage"), done_env_ids)
        if stage is not None:
            passing = stage > 0.5
            self._stage_passing_count += int(passing.sum().item())
            self._stage_open_count += int((~passing).sum().item())
            self._stage_total += int(stage.numel())

        lateral_error = self._select_done_values(infos.get("lateral_error"), done_env_ids)
        if lateral_error is not None:
            self._lateral_error_sum += lateral_error.sum().item()
            self._lateral_error_count += int(lateral_error.numel())

        heading_error = self._select_done_values(infos.get("heading_error"), done_env_ids)
        if heading_error is not None:
            self._heading_error_sum += heading_error.sum().item()
            self._heading_error_count += int(heading_error.numel())

        # Randomization diagnostics
        for rand_key in ("random/door_mass", "random/hinge_resistance", "random/reset_x",
                         "random/reset_y", "random/reset_yaw"):
            val = self._select_done_values(infos.get(rand_key), done_env_ids)
            if val is not None:
                self._random_sums[rand_key] += val.sum().item()
                self._random_counts[rand_key] += int(val.numel())

        # Episode outcome diagnostics. Timeout is the only failure class.
        for reason in ("hard_collision", "reverse_open", "fail_timeout"):
            val = self._select_done_values(infos.get(reason), done_env_ids, cast_float=False)
            if val is not None:
                val = val.to(dtype=torch.bool)
                self._fail_counts[reason] += int(val.sum().item())
                self._fail_totals[reason] += int(val.numel())

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)

        # Reward tags (target: reward/total, reward/opening, etc.)
        _REWARD_TAG_MAP = {
            "total": "reward/total",
            "opening": "reward/opening",
            "opening/open_door_target": "reward/open_door_target",
            "passing": "reward/passing",
            "shaping": "reward/shaping",
            "shaping/min_arm_motion": "reward/min_arm_motion",
            "shaping/stretched_arm": "reward/penalize_stretched_arm",
            "shaping/end_effector_to_panel": "reward/end_effector_to_panel",
            "shaping/command_limit": "reward/penalize_command_limit",
            "shaping/collision": "reward/penalize_collision",
        }
        for key, tag in _REWARD_TAG_MAP.items():
            value = self._mean_reward(key)
            if value is not None:
                self.writer.add_scalar(tag, value, frame)

        # Success tags
        for mode in ("rate", "opened_enough_rate", "passed_through_rate", "no_collision_rate"):
            total = self._success_totals.get(mode, 0)
            if total > 0:
                rate = self._success_counts[mode] / total
                self.writer.add_scalar(f"success/{mode}", rate, frame)

        # Episode length
        self.writer.add_scalar("episode/length", self._mean_reward("_step_count") or 0.0, frame)

        # Task state tags
        if self._door_angle_count > 0:
            mean_door_angle = self._door_angle_sum / self._door_angle_count
            self.writer.add_scalar(
                "task/door_angle_mean",
                mean_door_angle,
                frame,
            )
            self.writer.add_scalar(
                "task/door_angle_final",
                mean_door_angle,
                frame,
            )
        if self._cross_progress_count > 0:
            self.writer.add_scalar(
                "task/base_cross_progress",
                self._cross_progress_sum / self._cross_progress_count,
                frame,
            )
        if self._door_angular_velocity_count > 0:
            self.writer.add_scalar(
                "task/door_angular_velocity",
                self._door_angular_velocity_sum / self._door_angular_velocity_count,
                frame,
            )

        # progress_reward = mean cumulative passing reward per episode
        passing_reward = self._mean_reward("passing")
        if passing_reward is not None:
            self.writer.add_scalar("task/progress_reward", passing_reward, frame)

        if self._stage_total > 0:
            self.writer.add_scalar("task/open_stage_rate", self._stage_open_count / self._stage_total, frame)
            self.writer.add_scalar("task/passing_stage_rate", self._stage_passing_count / self._stage_total, frame)

        if self._lateral_error_count > 0:
            self.writer.add_scalar("task/base_lateral_error", self._lateral_error_sum / self._lateral_error_count, frame)
        if self._heading_error_count > 0:
            self.writer.add_scalar("task/base_heading_error", self._heading_error_sum / self._heading_error_count, frame)

        # Episode outcome diagnostics. Timeout is the only failure class.
        for reason in ("hard_collision", "reverse_open", "fail_timeout"):
            total = self._fail_totals.get(reason, 0)
            if total > 0:
                fail_rate = self._fail_counts[reason] / total
                self.writer.add_scalar(f"task/{reason}_rate", fail_rate, frame)

        # Randomization diagnostics
        for rand_key in sorted(self._random_sums):
            count = self._random_counts.get(rand_key, 0)
            if count > 0:
                self.writer.add_scalar(rand_key, self._random_sums[rand_key] / count, frame)

        self._write_direct_std_stats(frame)
        self._reset_custom_stats()

    def _reset_custom_stats(self) -> None:
        self._reward_sums: defaultdict[str, float] = defaultdict(float)
        self._reward_counts: defaultdict[str, int] = defaultdict(int)
        self._success_counts: defaultdict[str, int] = defaultdict(int)
        self._success_totals: defaultdict[str, int] = defaultdict(int)
        self._fail_counts: defaultdict[str, int] = defaultdict(int)
        self._fail_totals: defaultdict[str, int] = defaultdict(int)
        self._door_angle_sum = 0.0
        self._door_angle_count = 0
        self._cross_progress_sum = 0.0
        self._cross_progress_count = 0
        self._door_angular_velocity_sum = 0.0
        self._door_angular_velocity_count = 0
        self._stage_open_count = 0
        self._stage_passing_count = 0
        self._stage_total = 0
        self._lateral_error_sum = 0.0
        self._lateral_error_count = 0
        self._heading_error_sum = 0.0
        self._heading_error_count = 0
        self._random_sums: defaultdict[str, float] = defaultdict(float)
        self._random_counts: defaultdict[str, int] = defaultdict(int)

    def _normalize_done_indices(self, done_indices: Any) -> torch.Tensor:
        if done_indices is None:
            return torch.empty(0, dtype=torch.long, device=self.algo.device)
        done_tensor = torch.as_tensor(done_indices, dtype=torch.long, device=self.algo.device)
        return done_tensor.reshape(-1)

    def _select_done_values(
        self,
        value: Any,
        done_env_ids: torch.Tensor,
        *,
        cast_float: bool = True,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.algo.device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        if done_env_ids.numel() == 0:
            selected = tensor[:0]
        elif tensor.shape[0] > int(done_env_ids.max().item()):
            selected = tensor[done_env_ids]
        else:
            selected = tensor
        if cast_float:
            selected = selected.to(dtype=torch.float32)
        return selected

    def _mean_reward(self, key: str) -> float | None:
        count = self._reward_counts.get(key, 0)
        if count <= 0:
            return None
        return self._reward_sums[key] / count

    def _write_direct_std_stats(self, frame: int) -> None:
        model = getattr(self.algo, "model", None)
        net = getattr(model, "a2c_network", None)
        sigma = getattr(net, "sigma", None)
        direct_std_config = getattr(net, "direct_std_config", None)
        if sigma is None or not isinstance(direct_std_config, dict) or not direct_std_config.get("enabled", False):
            return

        std = sigma.detach()
        self.writer.add_scalar("policy/std_min", float(std.min().item()), frame)
        self.writer.add_scalar("policy/std_max", float(std.max().item()), frame)
        self.writer.add_scalar("policy/std_mean", float(std.mean().item()), frame)

        arm_dim = int(direct_std_config.get("arm_action_dim", 0))
        base_dim = int(direct_std_config.get("base_action_dim", 0))
        if arm_dim > 0 and std.numel() >= arm_dim:
            self.writer.add_scalar("policy/std_arm_mean", float(std[:arm_dim].mean().item()), frame)
        if base_dim > 0 and std.numel() >= arm_dim + base_dim:
            self.writer.add_scalar("policy/std_base_mean", float(std[arm_dim:arm_dim + base_dim].mean().item()), frame)
