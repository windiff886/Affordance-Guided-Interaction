"""Rollout 演示 — 加载训练好的策略执行可视化 rollout。

用法:
    # 使用随机策略运行一个 episode
    python scripts/rollout_demo.py

    # 加载 checkpoint 运行确定性策略
    python scripts/rollout_demo.py --checkpoint checkpoints/ckpt_final.pt

    # 指定上下文并保存视频
    python scripts/rollout_demo.py --checkpoint checkpoints/ckpt_final.pt \
        --context both --episodes 3 --save-video rollout.mp4

    # 保存逐帧图片
    python scripts/rollout_demo.py --checkpoint checkpoints/ckpt_final.pt \
        --save-frames frames/
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


def _try_save_frame(env, frame_dir: Path | None, episode: int, step: int) -> np.ndarray | None:
    """尝试从环境获取渲染帧并保存到磁盘。"""
    if frame_dir is None:
        return None

    try:
        vis_obs = env.get_visual_observation()
        if vis_obs is None or "rgb" not in vis_obs:
            return None

        rgb = vis_obs["rgb"]  # (H, W, 3)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"ep{episode:03d}_step{step:05d}.png"

        try:
            import imageio.v3 as iio
            iio.imwrite(str(frame_path), rgb.astype(np.uint8))
        except ImportError:
            # 回退到简易 PPM 格式
            _save_ppm(frame_path.with_suffix(".ppm"), rgb.astype(np.uint8))

        return rgb
    except Exception:
        return None


def _save_ppm(path: Path, rgb: np.ndarray) -> None:
    """保存 RGB 帧为 PPM 格式（无依赖回退方案）。"""
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())


def _assemble_video(frames: list[np.ndarray], output_path: Path, fps: int = 30) -> bool:
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
        print("  imageio[ffmpeg] 不可用，无法生成视频。请安装: pip install imageio[ffmpeg]")
        return False
    except Exception as e:
        print(f"  视频生成失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# 单 Episode Rollout
# ═══════════════════════════════════════════════════════════════════════

def run_episode(
    envs,
    actor,
    actor_cfg,
    runtime,
    *,
    device: torch.device,
    deterministic: bool = True,
    verbose: bool = True,
    frame_dir: Path | None = None,
    episode_idx: int = 0,
    context_name: str = "none",
) -> dict:
    """执行单个 episode rollout。

    Parameters
    ----------
    envs:
        DirectRLEnvAdapter 实例（num_envs=1）。

    Returns
    -------
    dict
        包含 success, total_reward, steps, termination_reason, frames 的结果字典。
    """
    from affordance_guided_interaction.policy import batch_flatten_actor_obs

    left_occupied, right_occupied = VALID_CONTEXTS[context_name]

    actor_obs_list, critic_obs_list = envs.reset(
        door_types=["push"],
        left_occupied_list=[left_occupied],
        right_occupied_list=[right_occupied],
    )

    runtime.reset(1)
    actor_obs_list, critic_obs_list = runtime.prepare_batch(
        actor_obs_list=actor_obs_list,
        critic_obs_list=critic_obs_list,
        visual_observations=envs.get_visual_observations(),
        force_refresh_mask=[True],
    )
    actor_obs = actor_obs_list[0]

    hidden = _move_hidden_to_device(actor.init_hidden(1), device)
    done = False
    step = 0
    total_reward = 0.0
    last_info: dict = {}
    frames: list[np.ndarray] = []

    while not done:
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

        # 视觉 embedding 更新
        actor_obs_list, critic_obs_list = runtime.prepare_batch(
            actor_obs_list=actor_obs_list,
            critic_obs_list=critic_obs_list,
            visual_observations=envs.get_visual_observations(),
            force_refresh_mask=[done],
        )
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
    reason = str(last_info.get("termination_reason", "UNKNOWN"))

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
    """Rollout 演示主入口。"""
    parser = argparse.ArgumentParser(
        description="加载训练策略执行可视化 rollout 演示"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="模型 checkpoint 路径（不指定则使用随机初始化策略）",
    )
    parser.add_argument(
        "--config", default="configs/training/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="运行 episode 数量 (默认: 1)",
    )
    parser.add_argument(
        "--context",
        choices=list(VALID_CONTEXTS.keys()),
        default="none",
        help="杯子占用上下文 (默认: none)",
    )
    parser.add_argument(
        "--save-frames", default=None,
        help="帧图片保存目录（可选）",
    )
    parser.add_argument(
        "--save-video", default=None,
        help="输出视频文件路径（可选，需要 imageio[ffmpeg]）",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="推理设备 (默认: cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="使用采样动作而非确定性动作",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="减少逐步输出",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="以无窗口模式启动 Isaac Sim",
    )
    args = parser.parse_args()

    from train import build_env_cfg, load_config, build_models
    from affordance_guided_interaction.utils.runtime_env import resolve_headless_mode
    from affordance_guided_interaction.utils.sim_runtime import launch_simulation_app
    from affordance_guided_interaction.envs.door_push_env import DoorPushEnv
    from affordance_guided_interaction.envs.direct_rl_env_adapter import DirectRLEnvAdapter
    from affordance_guided_interaction.training.perception_runtime import PerceptionRuntime

    # ── 初始化 ──────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    config_path = str(_PROJECT_ROOT / args.config)
    cfg = load_config(config_path)
    simulation_app = launch_simulation_app(
        headless=resolve_headless_mode(args.headless, os.environ),
        enable_cameras=True,
        import_error_message=(
            "未检测到 isaacsim 运行时，rollout_demo.py 需要在 Isaac Sim / Isaac Lab 运行时下执行。"
        ),
    )

    actor, _critic, actor_cfg = build_models(cfg, device)
    actor.eval()

    # ── 加载 checkpoint（可选）─────────────────────────────
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"错误: checkpoint 文件不存在: {ckpt_path}")
            return 1
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        iteration = checkpoint.get("iteration", "?")
        print(f"已加载 checkpoint: {ckpt_path} (iter={iteration})")
    else:
        print("未指定 checkpoint，使用随机初始化策略")

    # ── 创建环境和视觉运行时 ────────────────────────────────
    runtime = PerceptionRuntime(refresh_interval=4, embedding_dim=768)

    # 使用 GPU 环境 (num_envs=1 用于演示)
    env_cfg = build_env_cfg(cfg, n_envs=1, device=str(device), seed=args.seed)
    _direct_env = DoorPushEnv(cfg=env_cfg)
    envs = DirectRLEnvAdapter(_direct_env)

    frame_dir = Path(args.save_frames) if args.save_frames else None
    deterministic = not args.stochastic
    verbose = not args.quiet

    print(f"\n{'═' * 60}")
    print(f"Rollout 演示配置:")
    print(f"  上下文: {args.context}")
    print(f"  Episode 数: {args.episodes}")
    print(f"  动作模式: {'确定性' if deterministic else '随机采样'}")
    print(f"  设备: {device}")
    if frame_dir:
        print(f"  帧保存目录: {frame_dir}")
    if args.save_video:
        print(f"  视频输出: {args.save_video}")
    print(f"{'═' * 60}\n")

    # ── 运行 Rollout ────────────────────────────────────────
    all_frames: list[np.ndarray] = []
    results: list[dict] = []

    try:
        for ep in range(args.episodes):
            print(f"── Episode {ep + 1}/{args.episodes} ({args.context}) ──")
            t0 = time.time()

            result = run_episode(
                envs=envs,
                actor=actor,
                actor_cfg=actor_cfg,
                runtime=runtime,
                device=device,
                deterministic=deterministic,
                verbose=verbose,
                frame_dir=frame_dir,
                episode_idx=ep,
                context_name=args.context,
            )
            results.append(result)
            all_frames.extend(result["frames"])

            elapsed = time.time() - t0
            status = "SUCCESS" if result["success"] else "FAILURE"
            print(
                f"  [{status}] "
                f"steps={result['steps']}  "
                f"reward={result['total_reward']:.4f}  "
                f"reason={result['termination_reason']}  "
                f"time={elapsed:.2f}s\n"
            )
    finally:
        envs.close()
        if simulation_app is not None:
            simulation_app.close()

    # ── 视频组装 ────────────────────────────────────────────
    if args.save_video and all_frames:
        _assemble_video(all_frames, Path(args.save_video))

    # ── 总结 ────────────────────────────────────────────────
    n_success = sum(1 for r in results if r["success"])
    avg_reward = sum(r["total_reward"] for r in results) / max(len(results), 1)
    avg_steps = sum(r["steps"] for r in results) / max(len(results), 1)

    print(f"{'═' * 60}")
    print(f"Rollout 总结:")
    print(f"  成功率: {n_success}/{len(results)} ({n_success / max(len(results), 1):.1%})")
    print(f"  平均奖励: {avg_reward:.4f}")
    print(f"  平均步数: {avg_steps:.1f}")
    if all_frames:
        print(f"  捕获帧数: {len(all_frames)}")
    print(f"{'═' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
