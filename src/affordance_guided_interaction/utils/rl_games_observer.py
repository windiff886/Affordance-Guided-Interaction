from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver

_BIG_REWARD_KEYS = ("task", "stab_left", "stab_right", "safe", "total")
_SUCCESS_MODES = ("all", "empty", "left", "right", "both")


class DoorPushTensorboardObserver(IsaacAlgoObserver):
    """Augment rl_games logging with project-specific episode statistics."""

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

        state_info = infos.get("episode_state_info")
        if isinstance(state_info, dict):
            for key, value in state_info.items():
                selected = self._select_done_values(value, done_env_ids)
                if selected is None:
                    continue
                selected = selected[torch.isfinite(selected)]
                if selected.numel() == 0:
                    continue
                self._state_sums[key] += selected.sum().item()
                self._state_counts[key] += int(selected.numel())

        reward_stage_index = self._select_done_values(infos.get("reward_stage_index"), done_env_ids)
        if reward_stage_index is not None:
            reward_stage_index = reward_stage_index[torch.isfinite(reward_stage_index)]
            if reward_stage_index.numel() > 0:
                self._reward_stage_index_sum += reward_stage_index.sum().item()
                self._reward_stage_index_count += int(reward_stage_index.numel())

        door_angle = self._select_done_values(infos.get("door_angle"), done_env_ids)
        if door_angle is not None:
            self._door_angle_sum += door_angle.sum().item()
            self._door_angle_count += int(door_angle.numel())

        success = self._select_done_values(infos.get("success"), done_env_ids, cast_float=False)
        base_crossed = self._select_done_values(infos.get("base_crossed"), done_env_ids, cast_float=False)
        door_open_met = self._select_done_values(infos.get("door_open_met"), done_env_ids, cast_float=False)
        left_occupied = self._select_done_values(
            infos.get("episode_left_occupied"),
            done_env_ids,
            cast_float=False,
        )
        right_occupied = self._select_done_values(
            infos.get("episode_right_occupied"),
            done_env_ids,
            cast_float=False,
        )
        if success is None or left_occupied is None or right_occupied is None:
            return

        success = success.to(dtype=torch.bool)
        if base_crossed is not None:
            base_crossed = base_crossed.to(dtype=torch.bool)
            self._task_state_counts["base_crossed"] += int(base_crossed.sum().item())
            self._task_state_totals["base_crossed"] += int(base_crossed.numel())
        if door_open_met is not None:
            door_open_met = door_open_met.to(dtype=torch.bool)
            self._task_state_counts["door_open_met"] += int(door_open_met.sum().item())
            self._task_state_totals["door_open_met"] += int(door_open_met.numel())
        door_open_but_not_crossed = self._select_done_values(
            infos.get("door_open_but_not_crossed"),
            done_env_ids,
            cast_float=False,
        )
        if door_open_but_not_crossed is not None:
            door_open_but_not_crossed = door_open_but_not_crossed.to(dtype=torch.bool)
            self._task_state_counts["door_open_but_not_crossed"] += int(door_open_but_not_crossed.sum().item())
            self._task_state_totals["door_open_but_not_crossed"] += int(door_open_but_not_crossed.numel())
        left_occupied = left_occupied.to(dtype=torch.bool)
        right_occupied = right_occupied.to(dtype=torch.bool)

        self._accumulate_success("all", success)
        self._accumulate_mode_success("empty", success, (~left_occupied) & (~right_occupied))
        self._accumulate_mode_success("left", success, left_occupied & (~right_occupied))
        self._accumulate_mode_success("right", success, (~left_occupied) & right_occupied)
        self._accumulate_mode_success("both", success, left_occupied & right_occupied)

        cup_drop = self._select_done_values(infos.get("fail_cup_drop"), done_env_ids, cast_float=False)
        timeout = self._select_done_values(infos.get("fail_timeout"), done_env_ids, cast_float=False)
        not_crossed = self._select_done_values(infos.get("fail_not_crossed"), done_env_ids, cast_float=False)
        if cup_drop is not None:
            cup_drop = cup_drop.to(dtype=torch.bool)
            self._fail_counts["cup_drop"] += int(cup_drop.sum().item())
            self._fail_totals["cup_drop"] += int(cup_drop.numel())
        if timeout is not None:
            timeout = timeout.to(dtype=torch.bool)
            self._fail_counts["timeout"] += int(timeout.sum().item())
            self._fail_totals["timeout"] += int(timeout.numel())
        if not_crossed is not None:
            not_crossed = not_crossed.to(dtype=torch.bool)
            self._fail_counts["not_crossed"] += int(not_crossed.sum().item())
            self._fail_totals["not_crossed"] += int(not_crossed.numel())

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)

        for key in _BIG_REWARD_KEYS:
            value = self._mean_reward(key)
            if value is not None:
                self.writer.add_scalar(f"reward/{key}", value, frame)

        for key in sorted(self._reward_sums):
            if key in _BIG_REWARD_KEYS:
                continue
            value = self._mean_reward(key)
            if value is not None:
                self.writer.add_scalar(f"reward_detail/{key}", value, frame)

        for key in sorted(self._state_sums):
            value = self._mean_state(key)
            if value is not None:
                self.writer.add_scalar(f"state/{key}", value, frame)

        if self._reward_stage_index_count > 0:
            stage_index_value = self._reward_stage_index_sum / self._reward_stage_index_count
            self.writer.add_scalar("stage/current_stage_index", stage_index_value, frame)

        if self._door_angle_count > 0:
            self.writer.add_scalar(
                "task_state/door_angle_final",
                self._door_angle_sum / self._door_angle_count,
                frame,
            )
        for key in ("base_crossed", "door_open_met", "door_open_but_not_crossed"):
            total = self._task_state_totals.get(key, 0)
            if total > 0:
                rate = self._task_state_counts[key] / total
                self.writer.add_scalar(f"task_state/{key}_rate", rate, frame)

        for mode in _SUCCESS_MODES:
            if self._success_totals[mode] <= 0:
                continue
            success_rate = self._success_counts[mode] / self._success_totals[mode]
            self.writer.add_scalar(f"success/{mode}", success_rate, frame)

        for reason in ("cup_drop", "timeout", "not_crossed"):
            total = self._fail_totals.get(reason, 0)
            if total > 0:
                fail_rate = self._fail_counts[reason] / total
                self.writer.add_scalar(f"fail_reason/{reason}", fail_rate, frame)

        self._reset_custom_stats()

    def _reset_custom_stats(self) -> None:
        self._reward_sums: defaultdict[str, float] = defaultdict(float)
        self._reward_counts: defaultdict[str, int] = defaultdict(int)
        self._state_sums: defaultdict[str, float] = defaultdict(float)
        self._state_counts: defaultdict[str, int] = defaultdict(int)
        self._success_counts: defaultdict[str, int] = defaultdict(int)
        self._success_totals: defaultdict[str, int] = defaultdict(int)
        self._task_state_counts: defaultdict[str, int] = defaultdict(int)
        self._task_state_totals: defaultdict[str, int] = defaultdict(int)
        self._fail_counts: defaultdict[str, int] = defaultdict(int)
        self._fail_totals: defaultdict[str, int] = defaultdict(int)
        self._door_angle_sum = 0.0
        self._door_angle_count = 0
        self._reward_stage_index_sum = 0.0
        self._reward_stage_index_count = 0

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

    def _accumulate_success(self, mode: str, success: torch.Tensor) -> None:
        self._success_counts[mode] += int(success.sum().item())
        self._success_totals[mode] += int(success.numel())

    def _accumulate_mode_success(self, mode: str, success: torch.Tensor, mask: torch.Tensor) -> None:
        count = int(mask.sum().item())
        if count <= 0:
            return
        self._success_counts[mode] += int(success[mask].sum().item())
        self._success_totals[mode] += count

    def _mean_reward(self, key: str) -> float | None:
        count = self._reward_counts.get(key, 0)
        if count <= 0:
            return None
        return self._reward_sums[key] / count

    def _mean_state(self, key: str) -> float | None:
        count = self._state_counts.get(key, 0)
        if count <= 0:
            return None
        return self._state_sums[key] / count
