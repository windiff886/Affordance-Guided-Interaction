"""类型化的 rollout 可视化配置解析与验证。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_CONTEXTS = frozenset({"none", "left_only", "right_only", "both"})


@dataclass(frozen=True)
class RolloutVisualizationConfig:
    """Rollout 可视化的运行时配置，从 YAML 解析而来。"""

    checkpoint: Path | None
    device: str
    seed: int
    headless: bool
    deterministic: bool
    contexts: tuple[str, ...]
    episodes_per_context: int
    save_video: bool
    save_frames: bool
    frame_stride: int
    video_fps: int
    viewer_eye: tuple[float, float, float]
    viewer_lookat: tuple[float, float, float]
    output_root: Path
    video_name_template: str
    frames_dir_template: str


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    """将可能是相对路径的值解析为绝对路径。"""
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _parse_xyz_triplet(value: Any, *, field_name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """解析长度为 3 的 XYZ 向量。"""
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} 必须是长度为 3 的数字列表，当前值: {value!r}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须只包含数字，当前值: {value!r}") from exc


def resolve_visualization_config(
    vis_dict: dict[str, Any],
    *,
    project_root: Path,
) -> RolloutVisualizationConfig:
    """从 YAML 字典解析并验证可视化配置。"""
    raw_contexts = vis_dict.get("contexts", ["none"])
    if not raw_contexts:
        raise ValueError(
            "contexts 列表不能为空。可选值: " + ", ".join(sorted(VALID_CONTEXTS))
        )
    invalid = [c for c in raw_contexts if c not in VALID_CONTEXTS]
    if invalid:
        raise ValueError(
            f"无效的 context 值: {invalid}。"
            f"可选值: {', '.join(sorted(VALID_CONTEXTS))}"
        )
    contexts = tuple(raw_contexts)

    episodes = int(vis_dict.get("episodes_per_context", 1))
    if episodes <= 0:
        raise ValueError(
            f"episodes_per_context 必须为正整数，当前值: {episodes}"
        )

    stride = int(vis_dict.get("frame_stride", 1))
    if stride <= 0:
        raise ValueError(
            f"frame_stride 必须为正整数，当前值: {stride}"
        )

    fps = int(vis_dict.get("video_fps", 30))
    if fps <= 0:
        raise ValueError(
            f"video_fps 必须为正整数，当前值: {fps}"
        )

    checkpoint = _resolve_path(vis_dict.get("checkpoint"), project_root)
    output_root = _resolve_path(
        vis_dict.get("output_root", "artifacts/vis"), project_root
    )
    if output_root is None:
        output_root = (project_root / "artifacts" / "vis").resolve()

    viewer_eye = _parse_xyz_triplet(
        vis_dict.get("viewer_eye"),
        field_name="viewer_eye",
        default=(6.0, 3.0, 3.2),
    )
    viewer_lookat = _parse_xyz_triplet(
        vis_dict.get("viewer_lookat"),
        field_name="viewer_lookat",
        default=(3.0, 0.3, 1.0),
    )

    return RolloutVisualizationConfig(
        checkpoint=checkpoint,
        device=str(vis_dict.get("device", "cpu")),
        seed=int(vis_dict.get("seed", 42)),
        headless=bool(vis_dict.get("headless", True)),
        deterministic=bool(vis_dict.get("deterministic", True)),
        contexts=contexts,
        episodes_per_context=episodes,
        save_video=bool(vis_dict.get("save_video", False)),
        save_frames=bool(vis_dict.get("save_frames", False)),
        frame_stride=stride,
        video_fps=fps,
        viewer_eye=viewer_eye,
        viewer_lookat=viewer_lookat,
        output_root=output_root,
        video_name_template=str(
            vis_dict.get("video_name_template", "{checkpoint_stem}/{context}.mp4")
        ),
        frames_dir_template=str(
            vis_dict.get("frames_dir_template", "{checkpoint_stem}/{context}_frames")
        ),
    )
