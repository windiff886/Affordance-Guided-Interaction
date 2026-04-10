"""训练侧在线视觉缓存运行时。

.. deprecated::
    当前默认训练路径使用 ground-truth ``door_geometry(6D)`` 替代视觉 embedding，
    本模块不再被默认入口引用。仅保留为历史实验与后续感知研究参考。
    若未来需恢复视觉感知实验，应在独立实验配置或分支中接入。

负责按固定频率刷新 ``door_embedding``，并把最近一次缓存结果注入
actor / critic observation。当前实现支持两种模式：

1. 若环境提供 ``RGB-D`` 观测且视觉前端依赖可用，则执行真实编码
2. 若环境尚未提供视觉观测，则退化为零向量缓存，但接口保持一致
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass
class VisualCacheEntry:
    """单环境视觉缓存条目。"""

    door_embedding: np.ndarray
    visual_valid: bool = False
    stale_steps: int = 0


class PerceptionRuntime:
    """按固定频率维护每个环境的视觉 embedding 缓存。"""

    def __init__(
        self,
        *,
        refresh_interval: int = 4,
        embedding_dim: int = 768,
        pipeline: Any | None = None,
        visualize_detections: bool = False,
        strict_mode: bool = False,
    ) -> None:
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        self.refresh_interval = refresh_interval
        self.embedding_dim = embedding_dim
        self._pipeline = pipeline
        self._pipeline_initialized = pipeline is not None
        self._visualize_detections = visualize_detections
        self._strict_mode = strict_mode
        self._cache: list[VisualCacheEntry] = []
        self._last_error_reason: str | None = None
        self._door_embeddings: torch.Tensor | None = None
        self._visual_valid_mask: torch.Tensor | None = None
        self._stale_steps_tensor: torch.Tensor | None = None
        self._stage_timings: dict[str, float] = {}

    def reset(self, n_envs: int) -> None:
        """完全重置所有环境的视觉缓存。"""
        zero = np.zeros(self.embedding_dim, dtype=np.float32)
        self._cache = [
            VisualCacheEntry(
                door_embedding=zero.copy(),
                visual_valid=False,
                stale_steps=self.refresh_interval - 1,
            )
            for _ in range(n_envs)
        ]
        self._door_embeddings = None
        self._visual_valid_mask = None
        self._stale_steps_tensor = None
        self._stage_timings.clear()

    def prepare_batch(
        self,
        *,
        actor_obs_list: list[dict],
        critic_obs_list: list[dict],
        visual_observations: Sequence[dict[str, Any] | None] | None = None,
        force_refresh_mask: Sequence[bool] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """为一批 observation 注入最近一次视觉缓存。"""
        n_envs = len(actor_obs_list)
        refresh_mask = self.get_list_refresh_mask(
            n_envs=n_envs,
            force_refresh_mask=force_refresh_mask,
        )

        if visual_observations is None:
            visual_observations = [None] * n_envs

        for idx in range(n_envs):
            entry = self._cache[idx]
            should_refresh = refresh_mask[idx]

            if should_refresh:
                embedding, visual_valid = self._encode(visual_observations[idx])
                if not visual_valid:
                    reason = self._last_error_reason or "未知视觉错误"
                    raise RuntimeError(
                        f"visual_valid=0，视觉模块异常，env_id={idx}，原因：{reason}"
                    )
                entry = VisualCacheEntry(
                    door_embedding=embedding,
                    visual_valid=visual_valid,
                    stale_steps=0,
                )
                self._cache[idx] = entry
            else:
                entry.stale_steps += 1

            self._inject_embedding(actor_obs_list[idx], critic_obs_list[idx], entry)

        return actor_obs_list, critic_obs_list

    def _ensure_cache_size(self, n_envs: int) -> None:
        if len(self._cache) != n_envs:
            self.reset(n_envs)

    def _ensure_tensor_cache(self, n_envs: int, device: torch.device) -> None:
        if (
            self._door_embeddings is not None
            and self._visual_valid_mask is not None
            and self._stale_steps_tensor is not None
            and self._door_embeddings.shape[0] == n_envs
            and self._door_embeddings.device == device
        ):
            return

        self._door_embeddings = torch.zeros(
            n_envs, self.embedding_dim, dtype=torch.float32, device=device
        )
        self._visual_valid_mask = torch.zeros(
            n_envs, dtype=torch.bool, device=device
        )
        self._stale_steps_tensor = torch.full(
            (n_envs,),
            self.refresh_interval - 1,
            dtype=torch.int64,
            device=device,
        )

    def get_list_refresh_mask(
        self,
        *,
        n_envs: int,
        force_refresh_mask: Sequence[bool] | None = None,
    ) -> list[bool]:
        """返回 list-observation 路径中实际需要视觉刷新的 env mask。"""
        self._ensure_cache_size(n_envs)
        if force_refresh_mask is None:
            force_refresh_mask = [False] * n_envs
        if len(force_refresh_mask) != n_envs:
            raise ValueError(
                f"force_refresh_mask 长度错误: got {len(force_refresh_mask)}, expected {n_envs}"
            )

        refresh_mask: list[bool] = []
        for idx, entry in enumerate(self._cache):
            refresh_mask.append(
                bool(force_refresh_mask[idx])
                or (not entry.visual_valid)
                or entry.stale_steps >= (self.refresh_interval - 1)
            )
        return refresh_mask

    def get_tensor_refresh_mask(
        self,
        *,
        n_envs: int,
        device: torch.device,
        force_refresh_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 tensor-observation 路径中实际需要视觉刷新的 env mask。"""
        self._ensure_tensor_cache(n_envs, device)
        assert self._visual_valid_mask is not None
        assert self._stale_steps_tensor is not None

        refresh_mask = force_refresh_mask.to(device=device, dtype=torch.bool)
        stale_mask = self._stale_steps_tensor >= (self.refresh_interval - 1)
        return refresh_mask | (~self._visual_valid_mask) | stale_mask

    def consume_stage_timings(self) -> dict[str, float]:
        """取出并清空最近累积的视觉子阶段 timing。"""
        timings = dict(self._stage_timings)
        self._stage_timings.clear()
        return timings

    def _accumulate_stage_timings(self, timings: dict[str, float] | None) -> None:
        if not timings:
            return
        for name, value in timings.items():
            self._stage_timings[name] = self._stage_timings.get(name, 0.0) + float(value)

    def prepare_batch_tensors(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        *,
        visual_observations: dict[str, torch.Tensor] | None,
        force_refresh_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """为 batched tensor observation 注入最近一次视觉缓存。"""
        n_envs = actor_obs.shape[0]
        refresh_mask = self.get_tensor_refresh_mask(
            n_envs=n_envs,
            device=actor_obs.device,
            force_refresh_mask=force_refresh_mask,
        )
        assert self._door_embeddings is not None
        assert self._visual_valid_mask is not None
        assert self._stale_steps_tensor is not None

        if refresh_mask.any():
            if visual_observations is None:
                raise RuntimeError("visual_observations 缺失")
            masked_obs = {
                key: value[refresh_mask] for key, value in visual_observations.items()
            }
            encoded = self._encode_batch(masked_obs)
            self._door_embeddings[refresh_mask] = encoded
            self._visual_valid_mask[refresh_mask] = True
            self._stale_steps_tensor[refresh_mask] = 0

        self._stale_steps_tensor[~refresh_mask] += 1
        actor_obs[:, 90:858] = self._door_embeddings
        critic_obs[:, 90:858] = self._door_embeddings

        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "visual_embeddings": self._door_embeddings.clone(),
        }

    def _encode(self, observation: dict[str, Any] | None) -> tuple[np.ndarray, bool]:
        zero = np.zeros(self.embedding_dim, dtype=np.float32)
        self._last_error_reason = None
        if observation is None:
            self._last_error_reason = "visual_observation 缺失"
            return zero, False

        pipeline = self._get_pipeline()
        if pipeline is None:
            self._last_error_reason = "视觉管线初始化失败"
            return zero, False

        try:
            if hasattr(pipeline, "encode_with_timings"):
                embedding, stage_timings = pipeline.encode_with_timings(
                    observation=observation,
                    task_goal="push",
                )
                self._accumulate_stage_timings(stage_timings)
            else:
                embedding = pipeline.encode(observation=observation, task_goal="push")
        except Exception as exc:
            self._last_error_reason = f"视觉编码异常: {exc}"
            return zero, False

        arr = np.asarray(embedding, dtype=np.float32).ravel()
        if arr.shape != (self.embedding_dim,):
            self._last_error_reason = (
                f"embedding 维度错误: got {arr.shape}, expected {(self.embedding_dim,)}"
            )
            return zero, False
        return arr, True

    def _encode_batch(
        self,
        visual_observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pipeline = self._get_pipeline()
        if pipeline is None:
            raise RuntimeError("视觉管线初始化失败")

        if hasattr(pipeline, "encode_batch_with_timings"):
            embedding, stage_timings = pipeline.encode_batch_with_timings(
                observations=visual_observations,
                task_goal="push",
            )
            self._accumulate_stage_timings(stage_timings)
        elif hasattr(pipeline, "encode_batch"):
            embedding = pipeline.encode_batch(
                observations=visual_observations,
                task_goal="push",
            )
        else:
            chunks = []
            batch = int(visual_observations["rgb"].shape[0])
            for idx in range(batch):
                single = {
                    key: value[idx].detach().cpu().numpy()
                    for key, value in visual_observations.items()
                }
                if hasattr(pipeline, "encode_with_timings"):
                    embedding_i, stage_timings = pipeline.encode_with_timings(
                        observation=single,
                        task_goal="push",
                    )
                    self._accumulate_stage_timings(stage_timings)
                else:
                    embedding_i = pipeline.encode(observation=single, task_goal="push")
                chunks.append(embedding_i)
            embedding = np.stack(chunks, axis=0)

        tensor = torch.as_tensor(
            embedding,
            dtype=torch.float32,
            device=visual_observations["rgb"].device,
        )
        if tensor.ndim != 2 or tensor.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"embedding 维度错误: got {tuple(tensor.shape)}, "
                f"expected (*, {self.embedding_dim})"
            )
        return tensor

    def _get_pipeline(self) -> Any | None:
        if self._pipeline_initialized:
            return self._pipeline

        self._pipeline_initialized = True
        try:
            from affordance_guided_interaction.door_perception import (
                AffordancePipeline,
                AffordancePipelineConfig,
            )

            config = AffordancePipelineConfig(
                visualize_detections=self._visualize_detections,
            )
            self._pipeline = AffordancePipeline(config)
        except Exception as exc:
            if self._strict_mode:
                raise RuntimeError(
                    "strict_mode=true 但视觉管线初始化失败。"
                    "请确保 LangSAM / Point-MAE 等依赖已正确安装。"
                    "若需要在无视觉依赖环境下开发调试，请设置 "
                    "training.debug.strict_mode: false"
                ) from exc
            self._pipeline = None
        return self._pipeline

    @staticmethod
    def _inject_embedding(
        actor_obs: dict,
        critic_obs: dict,
        entry: VisualCacheEntry,
    ) -> None:
        actor_obs.setdefault("visual", {})
        actor_obs["visual"]["door_embedding"] = entry.door_embedding.copy()

        critic_actor_obs = critic_obs.get("actor_obs")
        if isinstance(critic_actor_obs, dict):
            critic_actor_obs.setdefault("visual", {})
            critic_actor_obs["visual"]["door_embedding"] = entry.door_embedding.copy()
