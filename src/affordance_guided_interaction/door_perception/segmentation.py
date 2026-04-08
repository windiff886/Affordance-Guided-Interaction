"""Open-vocabulary segmentation wrapper (LangSAM / Grounded-SAM 2).

Provides a unified interface that takes an RGB image and a list of text
prompts, and returns per-prompt binary masks with confidence scores.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from affordance_guided_interaction.door_perception.config import SegmentationConfig

logger = logging.getLogger(__name__)

_PROMPT_ALIAS = {
    "door": "door",
    "door handle": "handle",
    "button": "button",
    "push bar": "handle",
}


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
        self._lang_sam_effective_batch_size: int | None = None

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
        model = LangSAM(device=self._config.device)
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

    def segment_batch(
        self,
        rgb_batch: torch.Tensor | np.ndarray,
        prompts: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run open-vocab segmentation on a batch of RGB images."""
        self._ensure_model()
        if prompts is None:
            prompts = self._config.text_prompts

        masks = self._predict_batch(rgb_batch, list(prompts))
        device = rgb_batch.device if isinstance(rgb_batch, torch.Tensor) else None
        return {
            key: (
                value.to(device=device, dtype=torch.bool)
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value, dtype=torch.bool, device=device)
            )
            for key, value in masks.items()
        }

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

    def _predict_batch(
        self,
        rgb_batch: torch.Tensor | np.ndarray,
        prompts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        rgb_np = self._as_numpy_rgb_batch(rgb_batch)
        if self._config.model_type == "lang_sam":
            return self._predict_lang_sam_batch(rgb_np, prompts)
        if self._config.model_type == "grounded_sam2":
            return self._predict_grounded_sam2_batch(rgb_np, prompts)
        raise ValueError(f"Unknown segmentation model type: {self._config.model_type}")

    @staticmethod
    def _as_numpy_rgb_batch(rgb_batch: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(rgb_batch, torch.Tensor):
            return rgb_batch.detach().cpu().numpy().astype(np.uint8, copy=False)
        return np.asarray(rgb_batch, dtype=np.uint8)

    def _predict_lang_sam_batch(
        self,
        rgb_batch: np.ndarray,
        prompts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        from PIL import Image  # type: ignore[import-untyped]

        batch, height, width = rgb_batch.shape[:3]
        images = [Image.fromarray(rgb_batch[idx]) for idx in range(batch)]
        outputs = {
            _PROMPT_ALIAS.get(prompt, prompt): np.zeros((batch, height, width), dtype=bool)
            for prompt in prompts
        }

        for prompt in prompts:
            results = self._predict_lang_sam_prompt_chunked(images, prompt)
            key = _PROMPT_ALIAS.get(prompt, prompt)
            for idx, result in enumerate(results):
                mask, _conf, _bbox = self._extract_lang_sam_result(result, height, width)
                outputs[key][idx] |= mask

        return outputs

    def _predict_lang_sam_prompt_chunked(
        self,
        images: list[object],
        prompt: str,
    ) -> list[dict | None]:
        total = len(images)
        chunk_size = self._get_lang_sam_chunk_size(total)

        while True:
            try:
                results: list[dict | None] = []
                for start in range(0, total, chunk_size):
                    chunk_images = images[start : start + chunk_size]
                    chunk_prompts = [prompt] * len(chunk_images)
                    with torch.inference_mode(), self._lang_sam_autocast_context():
                        chunk_results = self._model.predict(  # type: ignore[union-attr]
                            chunk_images,
                            chunk_prompts,
                        )
                    if chunk_results is None:
                        chunk_results = [None] * len(chunk_images)
                    results.extend(chunk_results)
                self._lang_sam_effective_batch_size = chunk_size
                return results
            except torch.OutOfMemoryError:
                if chunk_size <= 1:
                    raise
                next_chunk_size = max(1, chunk_size // 2)
                logger.warning(
                    "LangSAM batch OOM for prompt %s at chunk_size=%d; retrying with chunk_size=%d",
                    prompt,
                    chunk_size,
                    next_chunk_size,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = next_chunk_size

    def _get_lang_sam_chunk_size(self, total_batch: int) -> int:
        chunk_size = total_batch
        if self._config.lang_sam_max_batch_size is not None:
            chunk_size = min(chunk_size, self._config.lang_sam_max_batch_size)
        if self._lang_sam_effective_batch_size is not None:
            chunk_size = min(chunk_size, self._lang_sam_effective_batch_size)
        return max(1, int(chunk_size))

    def _lang_sam_autocast_context(self):
        device = str(self._config.device)
        if not device.startswith("cuda"):
            return nullcontext()
        dtype_name = self._config.lang_sam_autocast_dtype.lower()
        autocast_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    def _predict_grounded_sam2_batch(
        self,
        rgb_batch: np.ndarray,
        prompts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        batch, height, width = rgb_batch.shape[:3]
        outputs = {
            _PROMPT_ALIAS.get(prompt, prompt): np.zeros((batch, height, width), dtype=bool)
            for prompt in prompts
        }

        for idx in range(batch):
            for prompt in prompts:
                mask, _conf, _bbox = self._predict_grounded_sam2(rgb_batch[idx], prompt)
                if mask is not None:
                    outputs[_PROMPT_ALIAS.get(prompt, prompt)][idx] |= mask

        return outputs

    def _extract_lang_sam_result(
        self,
        result: dict | None,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        if not result:
            return np.zeros((height, width), dtype=bool), 0.0, None

        masks = result.get("masks")
        scores = result.get("scores")
        boxes = result.get("boxes")

        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=bool), 0.0, None

        scores_np = np.asarray(scores)
        valid = scores_np >= self._config.confidence_threshold
        if not np.any(valid):
            return np.zeros((height, width), dtype=bool), 0.0, None

        best_idx = int(np.argmax(scores_np * valid))
        mask = np.asarray(masks[best_idx], dtype=bool)
        conf = float(scores_np[best_idx])
        bbox = np.asarray(boxes[best_idx]) if boxes is not None else None
        return mask, conf, bbox
