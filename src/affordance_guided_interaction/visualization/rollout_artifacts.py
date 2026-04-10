"""Rollout 可视化 artifact 管理：帧捕获、路径构建、视频组装。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 帧捕获控制
# ═══════════════════════════════════════════════════════════════════════


def should_capture_step(step: int, frame_stride: int) -> bool:
    """判断当前步是否应捕获帧。

    Parameters
    ----------
    step:
        当前环境步索引（从 0 开始）。
    frame_stride:
        帧步幅，1 表示每步都捕获，2 表示隔一步捕获一次。
    """
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
    """构建视频文件的输出路径。

    Parameters
    ----------
    output_root:
        artifact 输出根目录（绝对路径）。
    checkpoint_stem:
        checkpoint 文件名去掉扩展名，如 ``"iter_010000"``；无 checkpoint 时为 ``"random_init"``。
    context:
        当前上下文名称，如 ``"none"``、``"both"``。
    video_name_template:
        视频路径模板，支持 ``{checkpoint_stem}`` 和 ``{context}`` 占位符。
    """
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
    """构建帧图片的输出目录路径。

    参数同 :func:`build_video_path`，使用 ``frames_dir_template`` 模板。
    """
    relative = frames_dir_template.format(
        checkpoint_stem=checkpoint_stem,
        context=context,
    )
    return output_root / relative


# ═══════════════════════════════════════════════════════════════════════
# 帧捕获
# ═══════════════════════════════════════════════════════════════════════


def capture_frame(env) -> np.ndarray | None:
    """从环境获取当前帧的 RGB 数组。

    安全方法：任何异常都会返回 ``None`` 而不传播。
    """
    try:
        vis_obs = env.get_visual_observation()
        if vis_obs is None or "rgb" not in vis_obs:
            return None
        return vis_obs["rgb"]  # (H, W, 3)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# 帧保存
# ═══════════════════════════════════════════════════════════════════════


def save_frame_image(rgb: np.ndarray, path: Path) -> Path | None:
    """将 RGB 帧保存为图片文件。

    优先使用 imageio 保存 PNG；不可用时回退到 PPM 格式。

    Returns
    -------
    Path | None
        实际写入的文件路径，失败时返回 ``None``。
    """
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

    # 回退到 PPM
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
    """将帧列表组装为视频文件。

    Parameters
    ----------
    frames:
        RGB 帧列表，每个元素为 ``(H, W, 3)`` uint8 数组。
    output_path:
        输出视频文件路径。
    fps:
        输出帧率。

    Returns
    -------
    bool
        成功返回 ``True``；帧为空或 imageio[ffmpeg] 不可用时返回 ``False``。
    """
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
