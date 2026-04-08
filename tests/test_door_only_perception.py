from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from affordance_guided_interaction.door_perception.affordance_pipeline import (
    AffordancePipeline,
)
from affordance_guided_interaction.door_perception.config import (
    AffordancePipelineConfig,
    SegmentationConfig,
)


def test_segmentation_defaults_to_door_only_prompt():
    cfg = SegmentationConfig()
    assert cfg.text_prompts == ["door"]


def test_build_points_batch_ignores_non_door_masks():
    cfg = AffordancePipelineConfig()
    cfg.point_cloud.max_points = 8
    pipeline = AffordancePipeline(cfg)

    observations = {
        "depth": torch.ones(1, 2, 2, dtype=torch.float32),
    }
    masks = {
        "handle": torch.ones(1, 2, 2, dtype=torch.bool),
        "button": torch.ones(1, 2, 2, dtype=torch.bool),
    }

    points = pipeline._build_points_batch(observations, masks)
    assert points.shape == (1, 8, 3)
    assert torch.count_nonzero(points) == 0
