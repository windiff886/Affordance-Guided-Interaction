"""Configuration dataclasses for the door perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CameraIntrinsics:
    """Pin-hole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(slots=True)
class SegmentationConfig:
    """Settings for open-vocabulary segmentation."""

    model_type: str = "lang_sam"  # "lang_sam" | "grounded_sam2"
    text_prompts: list[str] = field(
        default_factory=lambda: ["door", "door handle", "button"]
    )
    confidence_threshold: float = 0.3
    device: str = "cuda"


@dataclass(slots=True)
class PointCloudConfig:
    """Settings for point cloud post-processing."""

    voxel_size: float = 0.005  # metres
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    radius_filter_radius: float = 0.02
    radius_filter_min_neighbors: int = 5
    max_points: int = 4096  # cap for frozen encoder input


@dataclass(slots=True)
class FrozenEncoderConfig:
    """Settings for the optional frozen point-cloud encoder."""

    enabled: bool = False
    model_type: str = "point_mae"  # "point_mae" | "ulip2"
    checkpoint_path: str = ""
    embed_dim: int = 256
    num_input_points: int = 1024
    device: str = "cuda"


@dataclass(slots=True)
class AffordancePipelineConfig:
    """Top-level configuration aggregating all sub-configs."""

    camera: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(
            fx=606.0, fy=606.0, cx=320.0, cy=240.0, width=640, height=480
        )
    )
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    frozen_encoder: FrozenEncoderConfig = field(default_factory=FrozenEncoderConfig)
