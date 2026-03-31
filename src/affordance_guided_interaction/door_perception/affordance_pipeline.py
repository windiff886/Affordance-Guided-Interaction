"""End-to-end affordance pipeline: RGB-D -> (z_aff, z_prog).

Implements the ``AffordanceEncoder`` protocol defined in
``perception.interfaces``.
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
)
from affordance_guided_interaction.door_perception.frozen_encoder import (
    BaseFrozenEncoder,
    build_frozen_encoder,
)
from affordance_guided_interaction.door_perception.geometric_summary import (
    AFFORDANCE_HANDLE,
    AFFORDANCE_PRESS,
    AFFORDANCE_PUSH,
    AFFORDANCE_SEQUENTIAL,
    Z_AFF_DIM,
    compute_z_aff,
)
from affordance_guided_interaction.door_perception.point_cloud_processing import (
    clean_point_cloud,
)
from affordance_guided_interaction.door_perception.segmentation import (
    OpenVocabSegmentor,
    SegmentResult,
)

logger = logging.getLogger(__name__)

# Mapping from task_goal text to affordance type index
_GOAL_TO_AFFORDANCE: dict[str, int] = {
    "push": AFFORDANCE_PUSH,
    "press": AFFORDANCE_PRESS,
    "handle": AFFORDANCE_HANDLE,
    "sequential": AFFORDANCE_SEQUENTIAL,
}

# Mapping from text prompt to dict key used internally
_PROMPT_TO_KEY: dict[str, str] = {
    "door": "door",
    "door handle": "handle",
    "button": "button",
    "push bar": "handle",
}


class AffordancePipeline:
    """Full pipeline: RGB-D observation -> z_aff + z_prog.

    Satisfies the ``AffordanceEncoder`` protocol from
    ``perception.interfaces``.

    Expected ``observation`` dict keys:

    * ``rgb`` : np.ndarray (H, W, 3) uint8
    * ``depth`` : np.ndarray (H, W) float metres
    * ``gripper_pos`` : np.ndarray (3,) gripper position in world frame
    * ``extrinsic`` : np.ndarray (4, 4) camera-to-world (optional)
    * ``door_angle`` : float  (optional, for z_prog)
    * ``button_pressed`` : bool (optional, for z_prog)
    * ``handle_triggered`` : bool (optional, for z_prog)
    """

    def __init__(self, config: AffordancePipelineConfig | None = None) -> None:
        self._config = config or AffordancePipelineConfig()
        self._segmentor = OpenVocabSegmentor(self._config.segmentation)
        self._frozen_encoder: BaseFrozenEncoder | None = build_frozen_encoder(
            self._config.frozen_encoder
        )

    # ------------------------------------------------------------------
    # AffordanceEncoder protocol
    # ------------------------------------------------------------------

    def encode(
        self,
        *,
        observation: dict[str, Any],
        task_goal: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the full pipeline and return ``(z_aff_dict, z_prog_dict)``.

        Returns dicts so that the caller can inspect individual fields.
        The ``"vector"`` key in each dict holds the flat numpy array
        suitable for concatenation into actor observations.
        """
        rgb: np.ndarray = observation["rgb"]
        depth: np.ndarray = observation["depth"]
        gripper_pos: np.ndarray = np.asarray(observation["gripper_pos"], dtype=np.float64)
        extrinsic: np.ndarray | None = observation.get("extrinsic")

        # --- 1. Open-vocabulary segmentation ---
        seg_results = self._segmentor.segment(rgb)
        masks, confidences = self._unpack_seg_results(seg_results)

        # --- 2. Depth back-projection per part ---
        point_clouds: dict[str, np.ndarray] = {}
        for key in ("door", "handle", "button"):
            mask = masks.get(key)
            if mask is not None and mask.any():
                pts = backproject_depth(
                    depth, self._config.camera, mask=mask, extrinsic=extrinsic
                )
            else:
                pts = np.zeros((0, 3), dtype=np.float64)
            point_clouds[key] = pts

        # --- 3. Point cloud cleaning ---
        cleaned: dict[str, np.ndarray] = {
            key: clean_point_cloud(pts, self._config.point_cloud)
            for key, pts in point_clouds.items()
        }

        # --- 4. Geometric summary -> z_aff ---
        affordance_type = _GOAL_TO_AFFORDANCE.get(task_goal.lower(), AFFORDANCE_PUSH)
        z_aff_vec = compute_z_aff(
            door_points=cleaned["door"],
            handle_points=cleaned["handle"],
            button_points=cleaned["button"],
            gripper_pos=gripper_pos,
            affordance_type=affordance_type,
            door_confidence=confidences.get("door", 0.0),
            handle_confidence=confidences.get("handle", 0.0),
            button_confidence=confidences.get("button", 0.0),
        )

        # --- 5. Optional frozen encoder ---
        pc_embed: np.ndarray | None = None
        if self._frozen_encoder is not None:
            # Encode the door panel cloud (primary interaction surface)
            door_pts = cleaned["door"]
            if len(door_pts) > 0:
                pc_embed = self._frozen_encoder.encode(door_pts)
            else:
                pc_embed = np.zeros(self._config.frozen_encoder.embed_dim)

        # --- 6. Assemble z_aff output ---
        z_aff_dict: dict[str, Any] = {
            "vector": z_aff_vec,
            "point_clouds": cleaned,
            "affordance_type": affordance_type,
        }
        if pc_embed is not None:
            z_aff_full = np.concatenate([z_aff_vec, pc_embed])
            z_aff_dict["vector_with_embed"] = z_aff_full
            z_aff_dict["pc_embed"] = pc_embed

        # --- 7. Task progress z_prog ---
        z_prog_dict = self._build_z_prog(observation, task_goal)

        return z_aff_dict, z_prog_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_seg_results(
        results: list[SegmentResult],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Convert list of SegmentResults into keyed dicts."""
        masks: dict[str, np.ndarray] = {}
        confs: dict[str, float] = {}
        for r in results:
            key = _PROMPT_TO_KEY.get(r.prompt, r.prompt)
            masks[key] = r.mask
            confs[key] = r.confidence
        return masks, confs

    @staticmethod
    def _build_z_prog(
        observation: dict[str, Any],
        task_goal: str,
    ) -> dict[str, Any]:
        """Construct task-progress representation from simulator state.

        Current version uses privileged sim state (door_angle,
        button_pressed, handle_triggered).  A future version can replace
        this with vision/geometry-based heuristics.
        """
        door_angle = float(observation.get("door_angle", 0.0))
        button_pressed = float(observation.get("button_pressed", False))
        handle_triggered = float(observation.get("handle_triggered", False))

        # Normalised progress scalar (heuristic)
        if task_goal.lower() in ("push", "handle"):
            progress = min(door_angle / 1.57, 1.0)  # ~90 deg
        elif task_goal.lower() == "press":
            progress = button_pressed
        elif task_goal.lower() == "sequential":
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
