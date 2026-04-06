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
from typing import Any

import numpy as np

from affordance_guided_interaction.door_perception.config import (
    AffordancePipelineConfig,
)
from affordance_guided_interaction.door_perception.depth_projection import (
    backproject_depth,
    sample_or_pad,
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
        """执行完整管线，返回 door_embedding。

        Parameters
        ----------
        observation : dict
            包含 rgb, depth 等键值的观测字典。
        task_goal : str | None
            兼容参数。当前版本不再输出 z_prog，故该参数不会参与计算。

        Returns
        -------
        door_embedding : np.ndarray
            (embed_dim,) Point-MAE 输出的高维 embedding 向量。
        """
        rgb: np.ndarray = observation["rgb"]
        depth: np.ndarray = observation["depth"]
        extrinsic: np.ndarray | None = observation.get("extrinsic")

        # --- 1. 开集分割 ---
        seg_results = self._segmentor.segment(rgb)
        masks = self._unpack_masks(seg_results)

        # --- 2. 深度反投影：mask → 局部点云 ---
        all_points: list[np.ndarray] = []
        debug_point_clouds: dict[str, np.ndarray] = {}
        for key in ("door", "handle", "button"):
            mask = masks.get(key)
            if mask is not None and mask.any():
                pts = backproject_depth(
                    depth, self._config.camera, mask=mask, extrinsic=extrinsic
                )
                if len(pts) > 0:
                    all_points.append(pts)
                    if self._visualizer is not None:
                        debug_point_clouds[key] = pts

        # --- debug 可视化（在降采样前使用原始点云）---
        if self._visualizer is not None:
            self._visualizer.show(
                rgb=rgb,
                seg_results=seg_results,
                point_clouds=debug_point_clouds,
                intrinsics=self._config.camera,
                extrinsic=extrinsic,
            )

        # 合并所有部件的点云为一个整体
        if all_points:
            merged_points = np.concatenate(all_points, axis=0)
        else:
            merged_points = np.zeros((0, 3), dtype=np.float64)

        # --- 3. 基础降采样 + 点数对齐 ---
        if len(merged_points) > 0:
            merged_points = voxel_downsample(
                merged_points, self._config.point_cloud.voxel_size
            )
        aligned_points = sample_or_pad(
            merged_points, self._config.point_cloud.max_points
        )

        # --- 4. Point-MAE 编码 → door_embedding ---
        door_embedding = self._encoder.encode(aligned_points)

        return door_embedding

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
