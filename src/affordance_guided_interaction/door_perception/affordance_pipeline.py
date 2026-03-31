"""端到端 Affordance 管线：RGB-D -> Point-MAE Embedding + z_prog。

核心流程：
    1. LangSAM / Grounded-SAM 2 开集分割 → 门/把手/按钮 mask
    2. 深度反投影 → 局部点云
    3. Voxel 降采样 + 点数对齐
    4. Point-MAE 冻结编码器 → 高维 embedding (z_aff)
    5. 仿真状态 → 任务进展向量 (z_prog)

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
    """端到端 Affordance 管线：RGB-D → (z_aff embedding, z_prog dict)。

    管线不关心夹爪具体坐标，不计算几何距离特征，
    所有空间关系由下游 Policy 隐式学习。

    Expected ``observation`` dict keys:

    * ``rgb`` : np.ndarray (H, W, 3) uint8
    * ``depth`` : np.ndarray (H, W) float，单位：米
    * ``extrinsic`` : np.ndarray (4, 4) 相机到世界变换（可选）
    * ``door_angle`` : float  (可选，用于 z_prog)
    * ``button_pressed`` : bool (可选，用于 z_prog)
    * ``handle_triggered`` : bool (可选，用于 z_prog)
    """

    def __init__(self, config: AffordancePipelineConfig | None = None) -> None:
        self._config = config or AffordancePipelineConfig()
        self._segmentor = OpenVocabSegmentor(self._config.segmentation)
        self._encoder = PointMAEEncoder(self._config.encoder)

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def encode(
        self,
        *,
        observation: dict[str, Any],
        task_goal: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """执行完整管线，返回 (z_aff_embedding, z_prog_dict)。

        Parameters
        ----------
        observation : dict
            包含 rgb, depth 等键值的观测字典。
        task_goal : str
            当前任务目标（如 "push", "press", "handle"）。

        Returns
        -------
        z_aff : np.ndarray
            (embed_dim,) Point-MAE 输出的高维 embedding 向量。
        z_prog : dict[str, Any]
            任务进展信息，包含 "vector" 键。
        """
        rgb: np.ndarray = observation["rgb"]
        depth: np.ndarray = observation["depth"]
        extrinsic: np.ndarray | None = observation.get("extrinsic")

        # --- 1. 开集分割 ---
        seg_results = self._segmentor.segment(rgb)
        masks = self._unpack_masks(seg_results)

        # --- 2. 深度反投影：mask → 局部点云 ---
        all_points: list[np.ndarray] = []
        for key in ("door", "handle", "button"):
            mask = masks.get(key)
            if mask is not None and mask.any():
                pts = backproject_depth(
                    depth, self._config.camera, mask=mask, extrinsic=extrinsic
                )
                if len(pts) > 0:
                    all_points.append(pts)

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

        # --- 4. Point-MAE 编码 → z_aff ---
        z_aff = self._encoder.encode(aligned_points)

        # --- 5. 任务进展 → z_prog ---
        z_prog = self._build_z_prog(observation, task_goal)

        return z_aff, z_prog

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
    def _build_z_prog(
        observation: dict[str, Any],
        task_goal: str,
    ) -> dict[str, Any]:
        """从仿真状态构造任务进展表示。

        当前版本直接使用 simulator 提供的特权状态信息。
        """
        door_angle = float(observation.get("door_angle", 0.0))
        button_pressed = float(observation.get("button_pressed", False))
        handle_triggered = float(observation.get("handle_triggered", False))

        # 归一化进度标量（基于启发式规则）
        goal = task_goal.lower()
        if goal in ("push", "handle"):
            progress = min(door_angle / 1.57, 1.0)  # ~90 度
        elif goal == "press":
            progress = button_pressed
        elif goal == "sequential":
            progress = 0.5 * button_pressed + 0.5 * min(door_angle / 1.57, 1.0)
        else:
            progress = 0.0

        z_prog_vec = np.array(
            [door_angle, button_pressed, handle_triggered, progress],
            dtype=np.float64,
        )

        return {
            "vector": z_prog_vec,
            "door_angle": door_angle,
            "button_pressed": button_pressed,
            "handle_triggered": handle_triggered,
            "progress": progress,
        }
