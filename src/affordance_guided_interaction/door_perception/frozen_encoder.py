"""Optional frozen point-cloud encoder (Point-MAE / ULIP-2).

All parameters are frozen — no training is performed.  This module is an
*optional enhancement* on top of the geometric summary.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from affordance_guided_interaction.door_perception.config import FrozenEncoderConfig
from affordance_guided_interaction.door_perception.point_cloud_processing import (
    random_sample,
)

logger = logging.getLogger(__name__)


class BaseFrozenEncoder(ABC):
    """Interface for frozen point-cloud encoders."""

    def __init__(self, config: FrozenEncoderConfig) -> None:
        self._config = config

    @abstractmethod
    def encode(self, points: np.ndarray) -> np.ndarray:
        """Encode a point cloud into a fixed-size embedding.

        Parameters
        ----------
        points : np.ndarray (N, 3)
            Input point cloud (will be resampled to ``config.num_input_points``).

        Returns
        -------
        np.ndarray (embed_dim,)
        """
        ...


class PointMAEEncoder(BaseFrozenEncoder):
    """Frozen Point-MAE encoder."""

    def __init__(self, config: FrozenEncoderConfig) -> None:
        super().__init__(config)
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch

        try:
            from point_mae import PointMAE  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "point-mae is required for PointMAEEncoder. "
                "See https://github.com/Pang-Yatian/Point-MAE"
            ) from exc

        logger.info("Loading Point-MAE checkpoint from %s", self._config.checkpoint_path)
        model = PointMAE()
        if self._config.checkpoint_path:
            state = torch.load(self._config.checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.eval()
        model.to(self._config.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

    def encode(self, points: np.ndarray) -> np.ndarray:
        import torch

        self._ensure_model()
        pts = random_sample(points, self._config.num_input_points)
        tensor = torch.from_numpy(pts).float().unsqueeze(0).to(self._config.device)
        with torch.no_grad():
            embedding = self._model.forward_features(tensor)  # type: ignore[union-attr]
        return embedding.squeeze(0).cpu().numpy()


class ULIP2Encoder(BaseFrozenEncoder):
    """Frozen ULIP-2 point-cloud encoder."""

    def __init__(self, config: FrozenEncoderConfig) -> None:
        super().__init__(config)
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch

        try:
            from ulip2 import ULIP2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ulip2 is required for ULIP2Encoder. "
                "See https://github.com/salesforce/ULIP"
            ) from exc

        logger.info("Loading ULIP-2 checkpoint from %s", self._config.checkpoint_path)
        model = ULIP2()
        if self._config.checkpoint_path:
            state = torch.load(self._config.checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.eval()
        model.to(self._config.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

    def encode(self, points: np.ndarray) -> np.ndarray:
        import torch

        self._ensure_model()
        pts = random_sample(points, self._config.num_input_points)
        tensor = torch.from_numpy(pts).float().unsqueeze(0).to(self._config.device)
        with torch.no_grad():
            embedding = self._model.encode_point_cloud(tensor)  # type: ignore[union-attr]
        return embedding.squeeze(0).cpu().numpy()


def build_frozen_encoder(config: FrozenEncoderConfig) -> BaseFrozenEncoder | None:
    """Factory: build the appropriate frozen encoder or return None if disabled."""
    if not config.enabled:
        return None
    if config.model_type == "point_mae":
        return PointMAEEncoder(config)
    if config.model_type == "ulip2":
        return ULIP2Encoder(config)
    raise ValueError(f"Unknown frozen encoder type: {config.model_type}")
