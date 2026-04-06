"""配置数据类：Door Perception 端到端管线。

所有视觉模型均为开箱即用的冻结预训练大模型，整条管线不包含任何训练步骤。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CameraIntrinsics:
    """针孔相机内参。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(slots=True)
class SegmentationConfig:
    """开集语义分割配置（LangSAM / Grounded-SAM 2）。"""

    model_type: str = "lang_sam"  # "lang_sam" | "grounded_sam2"
    text_prompts: list[str] = field(
        default_factory=lambda: ["door", "door handle", "button"]
    )
    confidence_threshold: float = 0.3
    device: str = "cuda"


@dataclass(slots=True)
class PointCloudConfig:
    """点云基础降采样配置。

    仅用于控制送入 Point-MAE 的点数上限，不做繁复的离群点剔除或平面拟合。
    """

    voxel_size: float = 0.005  # 体素边长（米）
    max_points: int = 1024  # 对齐到固定点数后送入 Encoder


@dataclass(slots=True)
class PointMAEEncoderConfig:
    """冻结的 Point-MAE 编码器配置。

    网络组件直接内联于 frozen_encoder.py，不依赖外部第三方 Point-MAE 包。
    权重文件使用 Point-MAE 官方发布的 pretrain.pth。
    """

    checkpoint_path: str = "model/pretrain.pth"
    # Point-MAE 网络超参（需与预训练权重一致）
    trans_dim: int = 384
    encoder_dims: int = 384
    depth: int = 12
    num_heads: int = 6
    group_size: int = 32
    num_group: int = 64
    device: str = "cuda"

    @property
    def embed_dim(self) -> int:
        """输出 embedding 维度（mean_pool + max_pool 拼接）。"""
        return self.trans_dim * 2


@dataclass(slots=True)
class AffordancePipelineConfig:
    """顶层配置，聚合所有子配置。"""

    camera: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(
            fx=606.0, fy=606.0, cx=320.0, cy=240.0, width=640, height=480
        )
    )
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    encoder: PointMAEEncoderConfig = field(default_factory=PointMAEEncoderConfig)

    # ── 调试选项 ──────────────────────────────────────────────────────
    visualize_detections: bool = False
