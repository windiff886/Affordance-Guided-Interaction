"""训练侧在线视觉缓存运行时。

负责按固定频率刷新 ``door_embedding``，并把最近一次缓存结果注入
actor / critic observation。当前实现支持两种模式：

1. 若环境提供 ``RGB-D`` 观测且视觉前端依赖可用，则执行真实编码
2. 若环境尚未提供视觉观测，则退化为零向量缓存，但接口保持一致
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


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
        self._ensure_cache_size(n_envs)

        if visual_observations is None:
            visual_observations = [None] * n_envs
        if force_refresh_mask is None:
            force_refresh_mask = [False] * n_envs

        for idx in range(n_envs):
            entry = self._cache[idx]
            should_refresh = (
                bool(force_refresh_mask[idx])
                or (not entry.visual_valid)
                or entry.stale_steps >= (self.refresh_interval - 1)
            )

            if should_refresh:
                embedding, visual_valid = self._encode(visual_observations[idx])
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

    def _encode(self, observation: dict[str, Any] | None) -> tuple[np.ndarray, bool]:
        zero = np.zeros(self.embedding_dim, dtype=np.float32)
        if observation is None:
            return zero, False

        pipeline = self._get_pipeline()
        if pipeline is None:
            return zero, False

        try:
            embedding = pipeline.encode(observation=observation, task_goal="push")
        except Exception:
            return zero, False

        arr = np.asarray(embedding, dtype=np.float32).ravel()
        if arr.shape != (self.embedding_dim,):
            return zero, False
        return arr, True

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
        actor_obs["visual"]["visual_valid"] = np.array(
            [1.0 if entry.visual_valid else 0.0], dtype=np.float32
        )

        critic_actor_obs = critic_obs.get("actor_obs")
        if isinstance(critic_actor_obs, dict):
            critic_actor_obs.setdefault("visual", {})
            critic_actor_obs["visual"]["door_embedding"] = entry.door_embedding.copy()
            critic_actor_obs["visual"]["visual_valid"] = np.array(
                [1.0 if entry.visual_valid else 0.0], dtype=np.float32
            )
