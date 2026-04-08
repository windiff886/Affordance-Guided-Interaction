"""端到端 Affordance 管线：RGB-D -> Point-MAE Embedding。

核心流程：
    1. LangSAM / Grounded-SAM 2 开集分割 → 门/把手/按钮 mask
    2. 深度反投影 → 局部点云
    3. Voxel 降采样 + 点数对齐
    4. Point-MAE 冻结编码器 → 高维 embedding (door_embedding)

管线不进行任何手工几何特征提取（无 RANSAC、无包围盒、无距离计算）。
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np
import torch

from affordance_guided_interaction.door_perception.config import (
    AffordancePipelineConfig,
)
from affordance_guided_interaction.door_perception.depth_projection import (
    backproject_depth,
    backproject_depth_batch,
    sample_or_pad,
    sample_or_pad_batch,
    voxel_downsample,
)
from affordance_guided_interaction.door_perception.frozen_encoder import (
    PointMAEEncoder,
)
from affordance_guided_interaction.door_perception.segmentation import (
    OpenVocabSegmentor,
    SegmentResult,
)

logger = logging.getLogger(__name__)

# 文本提示词 → 内部 key 的映射
_PROMPT_TO_KEY: dict[str, str] = {
    "door": "door",
    "door handle": "handle",
    "button": "button",
    "push bar": "handle",
}


class AffordancePipeline:
    """端到端 Affordance 管线：RGB-D → door_embedding。

    管线不关心夹爪具体坐标，不计算几何距离特征，
    所有空间关系由下游 Policy 隐式学习。

    Expected ``observation`` dict keys:

    * ``rgb`` : np.ndarray (H, W, 3) uint8
    * ``depth`` : np.ndarray (H, W) float，单位：米
    * ``extrinsic`` : np.ndarray (4, 4) 相机到世界变换（可选）
    """

    def __init__(self, config: AffordancePipelineConfig | None = None) -> None:
        self._config = config or AffordancePipelineConfig()
        self._segmentor = OpenVocabSegmentor(self._config.segmentation)
        self._encoder = PointMAEEncoder(self._config.encoder)

        # ── debug 可视化（仅在启用时导入 cv2）──────────────────
        self._visualizer = None
        if self._config.visualize_detections:
            try:
                from affordance_guided_interaction.door_perception.debug_visualizer import (
                    DetectionVisualizer,
                )
                self._visualizer = DetectionVisualizer()
            except ImportError:
                logger.warning(
                    "visualize_detections=True 但 opencv-python 未安装，"
                    "跳过可视化。安装方式: pip install opencv-python"
                )

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def encode(
        self,
        *,
        observation: dict[str, Any],
        task_goal: str | None = None,
    ) -> np.ndarray:
        """执行完整管线，返回 door_embedding。"""
        embedding, _timings = self.encode_with_timings(
            observation=observation,
            task_goal=task_goal,
        )
        return embedding

    def encode_with_timings(
        self,
        *,
        observation: dict[str, Any],
        task_goal: str | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """执行单帧管线并返回阶段 timing。"""
        del task_goal
        rgb: np.ndarray = observation["rgb"]
        depth: np.ndarray = observation["depth"]
        extrinsic: np.ndarray | None = observation.get("extrinsic")
        timings: dict[str, float] = {}

        seg_results = self._measure_stage(
            timings,
            "segmentation_s",
            lambda: self._segmentor.segment(rgb),
        )
        masks = self._unpack_masks(seg_results)

        def build_points() -> np.ndarray:
            all_points: list[np.ndarray] = []
            debug_point_clouds: dict[str, np.ndarray] = {}
            for key in ("door",):
                mask = masks.get(key)
                if mask is not None and mask.any():
                    pts = backproject_depth(
                        depth, self._config.camera, mask=mask, extrinsic=extrinsic
                    )
                    if len(pts) > 0:
                        all_points.append(pts)
                        if self._visualizer is not None:
                            debug_point_clouds[key] = pts

            if self._visualizer is not None:
                self._visualizer.show(
                    rgb=rgb,
                    seg_results=seg_results,
                    point_clouds=debug_point_clouds,
                    intrinsics=self._config.camera,
                    extrinsic=extrinsic,
                )

            if all_points:
                merged_points = np.concatenate(all_points, axis=0)
            else:
                merged_points = np.zeros((0, 3), dtype=np.float64)

            if len(merged_points) > 0:
                merged_points = voxel_downsample(
                    merged_points, self._config.point_cloud.voxel_size
                )
            return sample_or_pad(
                merged_points, self._config.point_cloud.max_points
            )

        aligned_points = self._measure_stage(timings, "pointcloud_s", build_points)
        door_embedding = self._measure_stage(
            timings,
            "encoder_s",
            lambda: self._encoder.encode(aligned_points),
        )
        return door_embedding, timings

    def encode_batch(
        self,
        *,
        observations: dict[str, torch.Tensor],
        task_goal: str | None = None,
    ) -> torch.Tensor:
        """执行 batched RGB-D -> door_embedding 管线。"""
        embedding, _timings = self.encode_batch_with_timings(
            observations=observations,
            task_goal=task_goal,
        )
        return embedding

    def encode_batch_with_timings(
        self,
        *,
        observations: dict[str, torch.Tensor],
        task_goal: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """执行 batched RGB-D -> door_embedding 管线并返回阶段 timing。"""
        del task_goal
        timings: dict[str, float] = {}
        seg = self._measure_stage(
            timings,
            "segmentation_s",
            lambda: self._segmentor.segment_batch(observations["rgb"]),
        )
        points = self._measure_stage(
            timings,
            "pointcloud_s",
            lambda: self._build_points_batch(observations, seg),
        )
        embedding = self._measure_stage(
            timings,
            "encoder_s",
            lambda: self._encoder.encode_batch(points),
        )
        return embedding, timings

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_masks(
        results: list[SegmentResult],
    ) -> dict[str, np.ndarray]:
        """将分割结果列表转换为按部件名索引的 mask 字典。"""
        masks: dict[str, np.ndarray] = {}
        for r in results:
            key = _PROMPT_TO_KEY.get(r.prompt, r.prompt)
            # 取置信度最高的那个（如果同一 key 出现多次）
            if key not in masks or r.confidence > 0:
                masks[key] = r.mask
        return masks

    @staticmethod
    def _measure_stage(
        timings: dict[str, float],
        name: str,
        fn,
    ):
        start = perf_counter()
        result = fn()
        timings[name] = timings.get(name, 0.0) + (perf_counter() - start)
        return result

    def _build_points_batch(
        self,
        observations: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        depth = observations["depth"]
        extrinsic = observations.get("extrinsic")

        point_batches: list[torch.Tensor] = []
        valid_batches: list[torch.Tensor] = []
        for key in ("door",):
            mask = masks.get(key)
            if mask is None:
                continue
            points, valid = backproject_depth_batch(
                depth,
                self._config.camera,
                mask=mask,
                extrinsic=extrinsic,
                max_points=self._config.point_cloud.max_points,
            )
            point_batches.append(points)
            valid_batches.append(valid)

        if not point_batches:
            batch = int(depth.shape[0])
            return torch.zeros(
                batch,
                self._config.point_cloud.max_points,
                3,
                dtype=depth.dtype,
                device=depth.device,
            )

        merged_points = torch.cat(point_batches, dim=1)
        merged_valid = torch.cat(valid_batches, dim=1)
        aligned_points, _ = sample_or_pad_batch(
            merged_points,
            merged_valid,
            self._config.point_cloud.max_points,
        )
        return aligned_points
