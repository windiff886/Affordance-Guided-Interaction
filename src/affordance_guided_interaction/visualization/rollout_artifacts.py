"""Rollout 可视化 artifact 管理：帧捕获、路径构建、视频组装。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 帧捕获控制
# ═══════════════════════════════════════════════════════════════════════


def should_capture_step(step: int, frame_stride: int) -> bool:
    """判断当前步是否应捕获帧。"""
    return step % frame_stride == 0


# ═══════════════════════════════════════════════════════════════════════
# Artifact 路径构建
# ═══════════════════════════════════════════════════════════════════════


def build_video_path(
    output_root: Path,
    checkpoint_stem: str,
    context: str,
    video_name_template: str,
) -> Path:
    """构建视频文件的输出路径。"""
    relative = video_name_template.format(
        checkpoint_stem=checkpoint_stem,
        context=context,
    )
    return output_root / relative


def build_frames_dir(
    output_root: Path,
    checkpoint_stem: str,
    context: str,
    frames_dir_template: str,
) -> Path:
    """构建帧图片的输出目录路径。"""
    relative = frames_dir_template.format(
        checkpoint_stem=checkpoint_stem,
        context=context,
    )
    return output_root / relative


# ═══════════════════════════════════════════════════════════════════════
# 帧捕获
# ═══════════════════════════════════════════════════════════════════════

_capture_warned_once = False


def _normalize_rgb_frame(frame) -> np.ndarray | None:
    """将 render/visual_observation 返回值标准化为 RGB uint8。"""
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.size == 0:
        return None
    if array.ndim != 3:
        return None
    if array.shape[-1] >= 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def capture_frame(env) -> np.ndarray | None:
    """从环境获取当前帧的 RGB 数组。

    优先使用 Isaac Lab 官方 `render()` 路径；若不可用，再回退到旧的
    `get_visual_observation()` 接口。
    """
    global _capture_warned_once
    try:
        render_fn = getattr(env, "render", None)
        if callable(render_fn):
            frame = _normalize_rgb_frame(render_fn())
            if frame is not None:
                _capture_warned_once = False
                return frame

        if not hasattr(env, "get_visual_observation"):
            if not _capture_warned_once:
                _logger.warning(
                    "capture_frame: 环境既没有 render() 也没有 get_visual_observation()，"
                    "帧捕获不可用。请确认 rollout 环境以 rgb_array 模式创建。"
                )
                _capture_warned_once = True
            return None

        vis_obs = env.get_visual_observation()
        frame = None if vis_obs is None else _normalize_rgb_frame(vis_obs.get("rgb"))
        if frame is None:
            if not _capture_warned_once:
                _logger.warning(
                    "capture_frame: render()/get_visual_observation() 都没有返回有效 RGB 帧，"
                    "请检查 rollout 的离屏渲染配置。"
                )
                _capture_warned_once = True
            return None

        _capture_warned_once = False
        return frame
    except Exception as exc:
        if not _capture_warned_once:
            _logger.warning("capture_frame: 帧捕获异常: %s", exc)
            _capture_warned_once = True
        return None


# ═══════════════════════════════════════════════════════════════════════
# 帧保存
# ═══════════════════════════════════════════════════════════════════════


def save_frame_image(rgb: np.ndarray, path: Path) -> Path | None:
    """将 RGB 帧保存为图片文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(rgb, dtype=np.uint8)

    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), rgb)
        return path
    except ImportError:
        pass
    except Exception:
        pass

    ppm_path = path.with_suffix(".ppm")
    try:
        _save_ppm(ppm_path, rgb)
        return ppm_path
    except Exception:
        return None


def _save_ppm(path: Path, rgb: np.ndarray) -> None:
    """保存 RGB 帧为 PPM 格式（无依赖回退方案）。"""
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())


# ═══════════════════════════════════════════════════════════════════════
# 视频组装
# ═══════════════════════════════════════════════════════════════════════


def assemble_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 30,
) -> bool:
    """将帧列表组装为视频文件。"""
    if not frames:
        print("  无可用帧，跳过视频生成")
        return False

    try:
        import imageio.v3 as iio

        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(output_path), frames, fps=fps)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  视频已保存: {output_path} ({size_mb:.2f} MB, {len(frames)} 帧)")
        return True
    except ImportError:
        print(
            "  imageio[ffmpeg] 不可用，无法生成视频。"
            "请安装: pip install imageio[ffmpeg]"
        )
        return False
    except Exception as e:
        print(f"  视频生成失败: {e}")
        return False
