from __future__ import annotations

import numpy as np


def row_major_rotation_to_column_major(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float64).T


def relative_pose_in_parent_frame(
    parent_pos: np.ndarray,
    parent_rot: np.ndarray,
    child_pos: np.ndarray,
    child_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_pos = np.asarray(parent_pos, dtype=np.float64)
    parent_rot = np.asarray(parent_rot, dtype=np.float64)
    child_pos = np.asarray(child_pos, dtype=np.float64)
    child_rot = np.asarray(child_rot, dtype=np.float64)
    local_pos = parent_rot.T @ (child_pos - parent_pos)
    local_rot = parent_rot.T @ child_rot
    return local_pos, local_rot


def calibrate_pose_alignment(
    model_pos: np.ndarray,
    model_rot: np.ndarray,
    sim_pos: np.ndarray,
    sim_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_pos = np.asarray(model_pos, dtype=np.float64)
    model_rot = np.asarray(model_rot, dtype=np.float64)
    sim_pos = np.asarray(sim_pos, dtype=np.float64)
    sim_rot = np.asarray(sim_rot, dtype=np.float64)
    align_rot = sim_rot @ model_rot.T
    align_pos = sim_pos - align_rot @ model_pos
    return align_pos, align_rot


def model_pose_to_sim_frame(
    model_pos: np.ndarray,
    model_rot: np.ndarray,
    align_pos: np.ndarray,
    align_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_pos = np.asarray(model_pos, dtype=np.float64)
    model_rot = np.asarray(model_rot, dtype=np.float64)
    align_pos = np.asarray(align_pos, dtype=np.float64)
    align_rot = np.asarray(align_rot, dtype=np.float64)
    sim_pos = align_rot @ model_pos + align_pos
    sim_rot = align_rot @ model_rot
    return sim_pos, sim_rot


def sim_pose_to_model_frame(
    sim_pos: np.ndarray,
    sim_rot: np.ndarray,
    align_pos: np.ndarray,
    align_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sim_pos = np.asarray(sim_pos, dtype=np.float64)
    sim_rot = np.asarray(sim_rot, dtype=np.float64)
    align_pos = np.asarray(align_pos, dtype=np.float64)
    align_rot = np.asarray(align_rot, dtype=np.float64)
    align_rot_inv = align_rot.T
    model_pos = align_rot_inv @ (sim_pos - align_pos)
    model_rot = align_rot_inv @ sim_rot
    return model_pos, model_rot


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_delta = np.asarray(rot_a, dtype=np.float64) @ np.asarray(rot_b, dtype=np.float64).T
    trace_term = (np.trace(rot_delta) - 1.0) * 0.5
    trace_term = float(np.clip(trace_term, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_term)))
