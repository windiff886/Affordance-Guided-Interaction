"""Debug 可视化工具：在 RGB 图像上叠加分割 mask、bbox 和点云回投影。

仅在 ``AffordancePipelineConfig.visualize_detections = True`` 时使用。
``cv2`` 仅在本文件中导入，关闭可视化时不会产生任何额外依赖。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from affordance_guided_interaction.door_perception.config import CameraIntrinsics

logger = logging.getLogger(__name__)

# 部件名 → BGR 颜色
_COLORS: dict[str, tuple[int, int, int]] = {
    "door": (255, 120, 50),     # 蓝橙
    "handle": (50, 220, 50),    # 绿
    "button": (50, 50, 255),    # 红
}

_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)

# 文本提示 → 内部 key（与 affordance_pipeline._PROMPT_TO_KEY 一致）
_PROMPT_TO_KEY: dict[str, str] = {
    "door": "door",
    "door handle": "handle",
    "button": "button",
    "push bar": "handle",
}


class DetectionVisualizer:
    """OpenCV 弹窗可视化：分割 mask + bbox + 点云回投影。

    Parameters
    ----------
    window_name : str
        OpenCV 窗口标题。
    mask_alpha : float
        mask 叠加半透明度（0 = 完全透明，1 = 不透明）。
    point_radius : int
        点云回投影的圆点半径（像素）。
    """

    def __init__(
        self,
        window_name: str = "Affordance Debug",
        mask_alpha: float = 0.4,
        point_radius: int = 2,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "debug visualize_detections 需要 opencv-python。"
                "安装方式: pip install opencv-python"
            ) from exc

        self._cv2 = cv2
        self._window_name = window_name
        self._mask_alpha = mask_alpha
        self._point_radius = point_radius

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def show(
        self,
        rgb: np.ndarray,
        seg_results: Sequence[Any],
        point_clouds: dict[str, np.ndarray],
        intrinsics: CameraIntrinsics,
        extrinsic: np.ndarray | None = None,
    ) -> None:
        """在 RGB 上叠加可视化内容并弹窗显示。

        Parameters
        ----------
        rgb : (H, W, 3) uint8
            原始 RGB 图像。
        seg_results : list[SegmentResult]
            分割结果列表（含 prompt, mask, confidence, bbox）。
        point_clouds : dict[str, (N, 3) ndarray]
            部件名 → 3D 点云（可能在相机系或世界系）。
        intrinsics : CameraIntrinsics
            相机内参。
        extrinsic : (4, 4) ndarray | None
            相机到世界变换。若不为 None 则点云在世界系，需要逆变换回相机系。
        """
        cv2 = self._cv2
        canvas = rgb.copy()

        # 1. 叠加分割 mask + bbox
        for result in seg_results:
            key = _PROMPT_TO_KEY.get(result.prompt, result.prompt)
            color = _COLORS.get(key, _DEFAULT_COLOR)

            if result.mask is not None and result.mask.any():
                self._overlay_mask(canvas, result.mask, color)

            if result.bbox is not None and result.confidence > 0:
                self._draw_bbox(canvas, result.bbox, key, result.confidence, color)

        # 2. 点云回投影
        for key, pts in point_clouds.items():
            if len(pts) == 0:
                continue
            color = _COLORS.get(key, _DEFAULT_COLOR)
            pixels = self._project_to_2d(pts, intrinsics, extrinsic)
            self._draw_points(canvas, pixels, color, intrinsics.width, intrinsics.height)

        # 3. 非阻塞显示
        # OpenCV 使用 BGR 格式
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, canvas_bgr)
        cv2.waitKey(1)

    def close(self) -> None:
        """销毁 OpenCV 窗口。"""
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 内部绘制方法
    # ------------------------------------------------------------------

    def _overlay_mask(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        """在 canvas 上半透明叠加彩色 mask。"""
        overlay = canvas.copy()
        overlay[mask] = color
        alpha = self._mask_alpha
        canvas[mask] = (
            (1 - alpha) * canvas[mask].astype(np.float32)
            + alpha * overlay[mask].astype(np.float32)
        ).astype(np.uint8)

    def _draw_bbox(
        self,
        canvas: np.ndarray,
        bbox: np.ndarray,
        label: str,
        confidence: float,
        color: tuple[int, int, int],
    ) -> None:
        """画 bbox 矩形 + 标签文字。"""
        cv2 = self._cv2
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # 文字背景
        cv2.rectangle(
            canvas,
            (x1, y1 - th - baseline - 4),
            (x1 + tw, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas, text, (x1, y1 - baseline - 2),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    @staticmethod
    def _project_to_2d(
        points: np.ndarray,
        intrinsics: CameraIntrinsics,
        extrinsic: np.ndarray | None,
    ) -> np.ndarray:
        """将 3D 点云投影回 2D 像素坐标。

        Returns
        -------
        (M, 2) ndarray
            像素坐标 (u, v)。仅返回 z > 0 的有效点。
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return np.zeros((0, 2), dtype=np.int32)

        # 如果点云在世界系，先变换回相机系
        if extrinsic is not None:
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            # extrinsic = cam_to_world, 逆变换 = world_to_cam
            pts = (R.T @ (pts - t).T).T

        # 过滤 z <= 0 的点（在相机后方）
        valid = pts[:, 2] > 0
        pts = pts[valid]
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.int32)

        u = (intrinsics.fx * pts[:, 0] / pts[:, 2] + intrinsics.cx).astype(np.int32)
        v = (intrinsics.fy * pts[:, 1] / pts[:, 2] + intrinsics.cy).astype(np.int32)

        return np.stack([u, v], axis=-1)

    def _draw_points(
        self,
        canvas: np.ndarray,
        pixels: np.ndarray,
        color: tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        """在 canvas 上绘制投影后的点云。"""
        cv2 = self._cv2
        for u, v in pixels:
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(canvas, (int(u), int(v)), self._point_radius, color, -1)
