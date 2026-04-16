"""Rollout 演示 — 加载训练好的策略执行可视化 rollout。

用法:
    # 使用 YAML 默认配置运行（推荐）
    python scripts/rollout_demo.py

    # 指定自定义配置根目录
    python scripts/rollout_demo.py --configs-dir /path/to/configs/

所有运行参数（checkpoint、contexts、帧率等）从 ``configs/visualization/default.yaml`` 读取，
不再需要长命令行参数。如需修改配置，请直接编辑 YAML 文件。

服务器/VS Code Remote 使用:
    1. 确保配置中 ``headless: true``
    2. 运行 ``python scripts/rollout_demo.py``
    3. 在 ``artifacts/vis/`` 下查看生成的 mp4 或帧目录
    4. 若 mp4 不可用（无 imageio[ffmpeg]），设 ``save_frames: true`` 获取逐帧图片
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ═══════════════════════════════════════════════════════════════════════
# 上下文映射
# ═══════════════════════════════════════════════════════════════════════

VALID_CONTEXTS = {
    "none": (False, False),
    "left_only": (True, False),
    "right_only": (False, True),
    "both": (True, True),
}


def _move_hidden_to_device(hidden, device: torch.device):
    """将 RNN 隐状态移动到指定设备。"""
    if isinstance(hidden, tuple):
        return tuple(h.to(device) for h in hidden)
    return hidden.to(device)


# ═══════════════════════════════════════════════════════════════════════
# 单 Episode Rollout
# ═══════════════════════════════════════════════════════════════════════


def run_episode(
    envs,
    actor,
    actor_cfg,
    *,
    device: torch.device,
    deterministic: bool = True,
    verbose: bool = True,
    frame_stride: int = 1,
    save_frames: bool = False,
    frames_dir: Path | None = None,
    episode_idx: int = 0,
    context_name: str = "none",
) -> dict:
    """执行单个 episode rollout。

    Parameters
    ----------
    envs:
        DirectRLEnvAdapter 实例（num_envs=1）。
    frame_stride:
        每隔 N 步捕获一帧。
    save_frames:
        是否将帧写入磁盘。
    frames_dir:
        帧图片保存目录。

    Returns
    -------
    dict
        包含 success, total_reward, steps, termination_reason, frames 的结果字典。
    """
    from affordance_guided_interaction.policy import batch_flatten_actor_obs
    from affordance_guided_interaction.visualization.rollout_artifacts import (
        should_capture_step,
        capture_frame,
        save_frame_image,
    )

    left_occupied, right_occupied = VALID_CONTEXTS[context_name]

    actor_obs_list, critic_obs_list = envs.reset(
        left_occupied_list=[left_occupied],
        right_occupied_list=[right_occupied],
    )
    actor_obs = actor_obs_list[0]

    hidden = _move_hidden_to_device(actor.init_hidden(1), device)
    done = False
    step = 0
    total_reward = 0.0
    last_info: dict = {}
    frames: list[np.ndarray] = []

    while not done:
        # ── 帧捕获 ──────────────────────────────────────────────
        if should_capture_step(step, frame_stride):
            frame = capture_frame(envs)
            if frame is not None:
                frames.append(frame)
                if save_frames and frames_dir is not None:
                    save_frame_image(
                        frame,
                        frames_dir / f"ep{episode_idx:03d}_step{step:05d}.png",
                    )

        # 构建 batch 观测
        actor_branches = batch_flatten_actor_obs([actor_obs], cfg=actor_cfg)
        actor_branches = {k: v.to(device) for k, v in actor_branches.items()}

        # 推理
        with torch.no_grad():
            action, hidden = actor.act(
                actor_branches, hidden=hidden, deterministic=deterministic
            )

        # 环境步进（向量化接口）
        actor_obs_list, critic_obs_list, rewards, dones, infos = envs.step(
            action.cpu().numpy()
        )
        done = bool(dones[0])
        last_info = infos[0]
        total_reward += float(rewards[0])
        step += 1

        actor_obs = actor_obs_list[0]

        # 逐步诊断输出
        if verbose and step % 20 == 0:
            door_angle = last_info.get("door_angle", 0.0)
            print(
                f"    [Step {step:>4d}] "
                f"r={float(rewards[0]):>7.4f}  "
                f"R={total_reward:>8.4f}  "
                f"angle={door_angle:>5.3f}  "
                f"action_norm={float(action.norm()):>6.3f}"
            )

    success = bool(last_info.get("success", False))
    # 优先从 info 读取终止原因（由 env → adapter 透传）；回退到字段推导
    reason = last_info.get("termination_reason")
    if not reason:
        if success:
            reason = "door_opened"
        elif last_info.get("terminated", False):
            reason = "cup_dropped"
        elif last_info.get("truncated", False):
            reason = "max_steps"
        else:
            reason = "UNKNOWN"
    reason = str(reason)

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step,
        "termination_reason": reason,
        "context": context_name,
        "frames": frames,
    }


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    """Rollout 演示主入口 — YAML-first。"""
    parser = argparse.ArgumentParser(
        description="加载训练策略执行可视化 rollout 演示"
    )
    parser.add_argument(
        "--configs-dir",
        default=None,
        help="可选：指定配置根目录；默认读取项目根目录下的 configs/",
    )
    args = parser.parse_args()

    from train import build_env_cfg, load_config, build_models
    from affordance_guided_interaction.visualization.rollout_config import (
        resolve_visualization_config,
    )
    from affordance_guided_interaction.visualization.rollout_artifacts import (
        build_video_path,
        build_frames_dir,
        assemble_video,
    )
    from affordance_guided_interaction.utils.runtime_env import resolve_headless_mode
    from affordance_guided_interaction.utils.sim_runtime import (
        launch_simulation_app,
        close_simulation_app,
    )
    from affordance_guided_interaction.utils.train_runtime_config import (
        resolve_train_runtime_config,
    )

    # ── 从 YAML 加载配置 ─────────────────────────────────────
    cfg = load_config(args.configs_dir)
    runtime_cfg = resolve_train_runtime_config(cfg, project_root=_PROJECT_ROOT)
    vis_cfg = resolve_visualization_config(
        cfg["visualization"], project_root=_PROJECT_ROOT
    )

    # ── 初始化 ──────────────────────────────────────────────
    np.random.seed(vis_cfg.seed)
    torch.manual_seed(vis_cfg.seed)

    device = torch.device(vis_cfg.device)

    # 仅在需要帧捕获时启用渲染管线。
    # 注意：enable_cameras=True 会启用 AppLauncher 完整渲染管线，
    # 显著降低仿真速度（~5-10x）。仅当确实需要录制视频时才开启。
    needs_rendering = vis_cfg.save_video or vis_cfg.save_frames
    if needs_rendering:
        print("提示: save_video/save_frames 已启用，将开启完整渲染管线，仿真速度会降低。")

    # SimulationApp 必须在 import isaaclab 之前启动，
    # 否则 omni 模块不可用 → ModuleNotFoundError
    simulation_app = launch_simulation_app(
        headless=resolve_headless_mode(vis_cfg.headless, os.environ),
        enable_cameras=needs_rendering,
        import_error_message=(
            "未检测到 isaacsim 运行时，rollout_demo.py 需要在 Isaac Sim / Isaac Lab 运行时下执行。"
        ),
    )

    # isaaclab 依赖 omni 运行时，必须在 SimulationApp 启动后导入
    from affordance_guided_interaction.envs.door_push_env import DoorPushEnv
    from affordance_guided_interaction.envs.direct_rl_env_adapter import DirectRLEnvAdapter

    actor, _critic, actor_cfg = build_models(cfg, device)
    actor.eval()

    # ── 加载 checkpoint（可选）─────────────────────────────
    if vis_cfg.checkpoint is not None:
        ckpt_path = vis_cfg.checkpoint
        if not ckpt_path.exists():
            print(f"错误: checkpoint 文件不存在: {ckpt_path}")
            return 1
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        iteration = checkpoint.get("iteration", "?")
        checkpoint_stem = ckpt_path.stem
        print(f"已加载 checkpoint: {ckpt_path} (iter={iteration})")
    else:
        checkpoint_stem = "random_init"
        print("未指定 checkpoint，使用随机初始化策略")

    # ── 创建环境 ────────────────────────────────────────────
    env_cfg = build_env_cfg(
        cfg,
        n_envs=1,
        device=str(device),
        seed=vis_cfg.seed,
        enable_cameras=needs_rendering,
    )
    env_cfg.viewer.eye = vis_cfg.viewer_eye
    env_cfg.viewer.lookat = vis_cfg.viewer_lookat
    render_mode = "rgb_array" if needs_rendering else None
    _direct_env = DoorPushEnv(cfg=env_cfg, render_mode=render_mode)
    envs = DirectRLEnvAdapter(_direct_env)

    verbose = True

    print(f"\n{'═' * 60}")
    print(f"Rollout 可视化配置 (from YAML):")
    print(f"  Checkpoint: {vis_cfg.checkpoint or 'random_init'}")
    print(f"  上下文: {', '.join(vis_cfg.contexts)}")
    print(f"  每上下文 episode: {vis_cfg.episodes_per_context}")
    print(f"  动作模式: {'确定性' if vis_cfg.deterministic else '随机采样'}")
    print(f"  设备: {device}")
    print(f"  保存视频: {vis_cfg.save_video}")
    print(f"  保存帧: {vis_cfg.save_frames}")
    print(f"  帧步幅: {vis_cfg.frame_stride}")
    print(f"  输出目录: {vis_cfg.output_root}")
    print(f"  Viewer eye: {vis_cfg.viewer_eye}")
    print(f"  Viewer lookat: {vis_cfg.viewer_lookat}")
    print(f"{'═' * 60}\n")

    # ── 跨上下文 Rollout ────────────────────────────────────
    all_results: list[dict] = []
    all_frames_by_context: dict[str, list[np.ndarray]] = {}
    video_skipped = False
    generated_files: list[str] = []

    try:
        for context_name in vis_cfg.contexts:
            context_frames: list[np.ndarray] = []

            # 构建 artifact 路径
            if vis_cfg.save_video:
                video_path = build_video_path(
                    vis_cfg.output_root,
                    checkpoint_stem,
                    context_name,
                    vis_cfg.video_name_template,
                )
            else:
                video_path = None

            if vis_cfg.save_frames:
                frames_dir = build_frames_dir(
                    vis_cfg.output_root,
                    checkpoint_stem,
                    context_name,
                    vis_cfg.frames_dir_template,
                )
            else:
                frames_dir = None

            for ep in range(vis_cfg.episodes_per_context):
                print(
                    f"── {context_name} | Episode {ep + 1}/{vis_cfg.episodes_per_context} ──"
                )
                t0 = time.time()

                result = run_episode(
                    envs=envs,
                    actor=actor,
                    actor_cfg=actor_cfg,
                    device=device,
                    deterministic=vis_cfg.deterministic,
                    verbose=verbose,
                    frame_stride=vis_cfg.frame_stride,
                    save_frames=vis_cfg.save_frames,
                    frames_dir=frames_dir,
                    episode_idx=ep,
                    context_name=context_name,
                )
                all_results.append(result)
                context_frames.extend(result["frames"])

                elapsed = time.time() - t0
                status = "SUCCESS" if result["success"] else "FAILURE"
                print(
                    f"  [{status}] "
                    f"steps={result['steps']}  "
                    f"reward={result['total_reward']:.4f}  "
                    f"reason={result['termination_reason']}  "
                    f"time={elapsed:.2f}s\n"
                )

            all_frames_by_context[context_name] = context_frames

            # ── 视频组装（按上下文） ──────────────────────────
            if vis_cfg.save_video and video_path is not None and context_frames:
                ok = assemble_video(context_frames, video_path, fps=vis_cfg.video_fps)
                if ok:
                    generated_files.append(str(video_path))
                else:
                    video_skipped = True

            if vis_cfg.save_frames and frames_dir is not None and context_frames:
                generated_files.append(str(frames_dir))

    finally:
        envs.close()
        close_simulation_app(simulation_app, wait_for_replicator=False)

    # ── 总结 ────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if r["success"])
    avg_reward = sum(r["total_reward"] for r in all_results) / max(len(all_results), 1)
    avg_steps = sum(r["steps"] for r in all_results) / max(len(all_results), 1)
    total_frames = sum(len(f) for f in all_frames_by_context.values())

    print(f"{'═' * 60}")
    print(f"Rollout 可视化总结:")
    print(f"  Checkpoint: {vis_cfg.checkpoint or 'random_init'}")
    print(f"  渲染上下文: {', '.join(vis_cfg.contexts)}")
    print(f"  成功率: {n_success}/{len(all_results)} ({n_success / max(len(all_results), 1):.1%})")
    print(f"  平均奖励: {avg_reward:.4f}")
    print(f"  平均步数: {avg_steps:.1f}")
    print(f"  总捕获帧数: {total_frames}")
    if generated_files:
        print(f"  生成的文件:")
        for f in generated_files:
            print(f"    - {f}")
    if video_skipped:
        print("  注意: mp4 生成被跳过（imageio[ffmpeg] 不可用）")
        print("  请安装: pip install imageio[ffmpeg]")
    print(f"{'═' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
