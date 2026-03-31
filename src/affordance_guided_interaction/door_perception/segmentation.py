"""Open-vocabulary segmentation wrapper (LangSAM / Grounded-SAM 2).

Provides a unified interface that takes an RGB image and a list of text
prompts, and returns per-prompt binary masks with confidence scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from affordance_guided_interaction.door_perception.config import SegmentationConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SegmentResult:
    """Single segmentation result for one text prompt."""

    prompt: str
    mask: np.ndarray        # (H, W) bool
    confidence: float       # average detection confidence
    bbox: np.ndarray | None = None  # (4,) xyxy, optional


class OpenVocabSegmentor:
    """Thin wrapper around LangSAM / Grounded-SAM 2.

    Usage::

        seg = OpenVocabSegmentor(SegmentationConfig())
        results = seg.segment(rgb, ["door", "door handle", "button"])
    """

    def __init__(self, config: SegmentationConfig) -> None:
        self._config = config
        self._model = None  # lazy init

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if self._config.model_type == "lang_sam":
            self._model = self._load_lang_sam()
        elif self._config.model_type == "grounded_sam2":
            self._model = self._load_grounded_sam2()
        else:
            raise ValueError(
                f"Unknown segmentation model type: {self._config.model_type}"
            )

    def _load_lang_sam(self) -> object:
        try:
            from lang_sam import LangSAM  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "lang-sam is required. Install with: pip install lang-sam"
            ) from exc

        logger.info("Loading LangSAM model on %s ...", self._config.device)
        model = LangSAM()
        return model

    def _load_grounded_sam2(self) -> object:
        try:
            from grounded_sam2.grounded_sam2 import GroundedSAM2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "grounded-sam-2 is required. See https://github.com/IDEA-Research/Grounded-SAM-2"
            ) from exc

        logger.info("Loading Grounded-SAM 2 model on %s ...", self._config.device)
        model = GroundedSAM2()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        rgb: np.ndarray,
        prompts: Sequence[str] | None = None,
    ) -> list[SegmentResult]:
        """Run open-vocab segmentation on *rgb* for each text prompt.

        Parameters
        ----------
        rgb : np.ndarray
            (H, W, 3) uint8 RGB image.
        prompts : list[str] | None
            Text prompts. Defaults to ``config.text_prompts``.

        Returns
        -------
        list[SegmentResult]
            One result per prompt.  If a prompt is not detected the mask is
            all-False and confidence is 0.
        """
        self._ensure_model()
        if prompts is None:
            prompts = self._config.text_prompts

        h, w = rgb.shape[:2]
        results: list[SegmentResult] = []

        for prompt in prompts:
            mask, confidence, bbox = self._predict_single(rgb, prompt)
            if mask is None:
                mask = np.zeros((h, w), dtype=bool)
                confidence = 0.0
                bbox = None
            results.append(
                SegmentResult(
                    prompt=prompt,
                    mask=mask,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Per-model prediction
    # ------------------------------------------------------------------

    def _predict_single(
        self,
        rgb: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        """Return (mask, confidence, bbox) or (None, 0, None)."""
        if self._config.model_type == "lang_sam":
            return self._predict_lang_sam(rgb, prompt)
        elif self._config.model_type == "grounded_sam2":
            return self._predict_grounded_sam2(rgb, prompt)
        return None, 0.0, None

    def _predict_lang_sam(
        self,
        rgb: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        from PIL import Image  # type: ignore[import-untyped]

        image = Image.fromarray(rgb)
        results = self._model.predict([image], [prompt])  # type: ignore[union-attr]

        # LangSAM returns a list of results per image
        if not results or len(results) == 0:
            return None, 0.0, None

        result = results[0]
        masks = result.get("masks")
        scores = result.get("scores")
        boxes = result.get("boxes")

        if masks is None or len(masks) == 0:
            return None, 0.0, None

        # Pick the highest-confidence detection above threshold
        scores_np = np.asarray(scores)
        valid = scores_np >= self._config.confidence_threshold
        if not np.any(valid):
            return None, 0.0, None

        best_idx = int(np.argmax(scores_np * valid))
        mask = np.asarray(masks[best_idx], dtype=bool)
        conf = float(scores_np[best_idx])
        bbox = np.asarray(boxes[best_idx]) if boxes is not None else None
        return mask, conf, bbox

    def _predict_grounded_sam2(
        self,
        rgb: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        # Grounded-SAM 2 API may vary; adapt to actual package interface
        results = self._model.predict(rgb, prompt)  # type: ignore[union-attr]
        if results is None or len(results.get("masks", [])) == 0:
            return None, 0.0, None

        masks = results["masks"]
        scores = np.asarray(results["scores"])
        valid = scores >= self._config.confidence_threshold
        if not np.any(valid):
            return None, 0.0, None

        best_idx = int(np.argmax(scores * valid))
        mask = np.asarray(masks[best_idx], dtype=bool)
        conf = float(scores[best_idx])
        bbox = (
            np.asarray(results["boxes"][best_idx])
            if "boxes" in results
            else None
        )
        return mask, conf, bbox
