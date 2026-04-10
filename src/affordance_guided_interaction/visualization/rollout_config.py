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
    output_root: Path
    video_name_template: str
    frames_dir_template: str


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    """将可能是相对路径的值解析为绝对路径。

    - ``None`` → ``None``
    - 已是绝对路径 → 原样返回
    - 相对路径 → ``project_root / value``
    """
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def resolve_visualization_config(
    vis_dict: dict[str, Any],
    *,
    project_root: Path,
) -> RolloutVisualizationConfig:
    """从 YAML 字典解析并验证可视化配置。

    Parameters
    ----------
    vis_dict:
        ``cfg["visualization"]`` 字典，通常由 ``load_config()`` 加载。
    project_root:
        项目根目录，用于解析相对路径。

    Returns
    -------
    RolloutVisualizationConfig
        不可变的类型化配置对象。

    Raises
    ------
    ValueError
        配置值不合法时抛出，附带可操作的错误信息。
    """
    # ── 验证 contexts ────────────────────────────────────────────
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

    # ── 验证 episodes_per_context ────────────────────────────────
    episodes = int(vis_dict.get("episodes_per_context", 1))
    if episodes <= 0:
        raise ValueError(
            f"episodes_per_context 必须为正整数，当前值: {episodes}"
        )

    # ── 验证 frame_stride ────────────────────────────────────────
    stride = int(vis_dict.get("frame_stride", 1))
    if stride <= 0:
        raise ValueError(
            f"frame_stride 必须为正整数，当前值: {stride}"
        )

    # ── 验证 video_fps ───────────────────────────────────────────
    fps = int(vis_dict.get("video_fps", 30))
    if fps <= 0:
        raise ValueError(
            f"video_fps 必须为正整数，当前值: {fps}"
        )

    # ── 解析路径 ─────────────────────────────────────────────────
    checkpoint = _resolve_path(vis_dict.get("checkpoint"), project_root)
    output_root = _resolve_path(
        vis_dict.get("output_root", "artifacts/vis"), project_root
    )
    if output_root is None:
        output_root = (project_root / "artifacts" / "vis").resolve()

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
        output_root=output_root,
        video_name_template=str(
            vis_dict.get("video_name_template", "{checkpoint_stem}/{context}.mp4")
        ),
        frames_dir_template=str(
            vis_dict.get("frames_dir_template", "{checkpoint_stem}/{context}_frames")
        ),
    )
