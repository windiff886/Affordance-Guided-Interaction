from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _move_hidden_to_device(hidden, device: torch.device):
    if isinstance(hidden, tuple):
        return tuple(h.to(device) for h in hidden)
    return hidden.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="混合上下文推门评估入口")
    parser.add_argument(
        "--config",
        default="configs/training/default.yaml",
        help="评估配置文件路径",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="待评估 checkpoint 路径")
    parser.add_argument("--episodes-per-context", type=int, default=5, help="每个上下文评估回合数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="以无窗口模式启动 Isaac Sim",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="若指定则使用采样动作；默认使用确定性动作",
    )
    args = parser.parse_args()

    from train import build_env_cfg, build_models, load_config
    from affordance_guided_interaction.utils.runtime_env import resolve_headless_mode
    from affordance_guided_interaction.utils.train_runtime_config import (
        resolve_train_runtime_config,
    )
    from affordance_guided_interaction.utils.sim_runtime import launch_simulation_app
    from affordance_guided_interaction.envs.door_push_env import DoorPushEnv
    from affordance_guided_interaction.envs.direct_rl_env_adapter import DirectRLEnvAdapter
    from affordance_guided_interaction.policy import batch_flatten_actor_obs
    from affordance_guided_interaction.training.evaluation import (
        summarize_evaluation_outcomes,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = load_config(args.config)
    runtime_cfg = resolve_train_runtime_config(cfg, project_root=_PROJECT_ROOT)
    simulation_app = launch_simulation_app(
        headless=resolve_headless_mode(args.headless, os.environ),
        enable_cameras=False,
        import_error_message=(
            "未检测到 isaacsim 运行时，evaluate.py 需要在 Isaac Sim / Isaac Lab 运行时下执行。"
        ),
    )
    actor, _critic, actor_cfg = build_models(cfg, device)
    actor.eval()

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])

    # 使用 GPU 环境 (num_envs=1 用于评估)
    env_cfg = build_env_cfg(
        cfg,
        n_envs=1,
        device=str(device),
        seed=args.seed,
        enable_cameras=False,
        variant=runtime_cfg.env_variant,
    )
    _direct_env = DoorPushEnv(cfg=env_cfg)
    envs = DirectRLEnvAdapter(_direct_env)

    contexts = {
        "none": (False, False),
        "left_only": (True, False),
        "right_only": (False, True),
        "both": (True, True),
    }
    outcomes: list[dict[str, object]] = []

    try:
        for context_name, (left_occupied, right_occupied) in contexts.items():
            for _ in range(args.episodes_per_context):
                actor_obs_list, critic_obs_list = envs.reset(
                    door_types=["push"],
                    left_occupied_list=[left_occupied],
                    right_occupied_list=[right_occupied],
                )
                actor_obs = actor_obs_list[0]

                hidden = _move_hidden_to_device(actor.init_hidden(1), device)
                done = False
                last_info: dict[str, object] = {}

                while not done:
                    actor_branches = batch_flatten_actor_obs([actor_obs], cfg=actor_cfg)
                    actor_branches = {
                        key: value.to(device) for key, value in actor_branches.items()
                    }

                    with torch.no_grad():
                        action, hidden = actor.act(
                            actor_branches,
                            hidden=hidden,
                            deterministic=not args.stochastic,
                        )

                    actor_obs_list, critic_obs_list, _rewards, dones, infos = envs.step(
                        action.cpu().numpy()
                    )
                    done = bool(dones[0])
                    last_info = infos[0]
                    actor_obs = actor_obs_list[0]

                outcomes.append(
                    {
                        "success": bool(last_info.get("success", False)),
                        "episode_context": context_name,
                        "termination_reason": str(
                            last_info.get("termination_reason", "UNKNOWN")
                        ),
                    }
                )
    finally:
        envs.close()
        if simulation_app is not None:
            simulation_app.close()

    summary = summarize_evaluation_outcomes(outcomes)
    print(f"[evaluate] checkpoint={args.checkpoint or 'random_init'}")
    print(f"[evaluate] episodes_per_context={args.episodes_per_context}")
    for key in [
        "evaluation/success_mixed",
        "evaluation/success_none",
        "evaluation/success_left_only",
        "evaluation/success_right_only",
        "evaluation/success_both",
    ]:
        print(f"{key}={summary[key]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
