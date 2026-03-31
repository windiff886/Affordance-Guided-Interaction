"""Door Perception Pipeline：RGB-D → Point-MAE Embedding (z_aff)。"""

from affordance_guided_interaction.door_perception.config import (
    AffordancePipelineConfig,
    CameraIntrinsics,
    PointCloudConfig,
    PointMAEEncoderConfig,
    SegmentationConfig,
)
from affordance_guided_interaction.door_perception.affordance_pipeline import (
    AffordancePipeline,
)

__all__ = [
    "AffordancePipeline",
    "AffordancePipelineConfig",
    "CameraIntrinsics",
    "PointCloudConfig",
    "PointMAEEncoderConfig",
    "SegmentationConfig",
]
