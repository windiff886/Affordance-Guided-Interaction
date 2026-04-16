"""策略模型导出 — 将训练好的 Actor 导出为 ONNX 或 TorchScript 格式。

导出后的模型接受单个扁平张量输入 + RNN 隐状态，输出动作 + 更新后的隐状态，
便于在 C++ / ONNX Runtime / Isaac Lab 推理管线中直接部署。

用法:
    # 导出 ONNX 格式
    python scripts/export_policy.py --checkpoint checkpoints/ckpt_final.pt --format onnx

    # 导出 TorchScript 格式
    python scripts/export_policy.py --checkpoint checkpoints/ckpt_final.pt --format torchscript

    # 同时导出两种格式
    python scripts/export_policy.py --checkpoint checkpoints/ckpt_final.pt --format both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ═══════════════════════════════════════════════════════════════════════
# 导出包装器
# ═══════════════════════════════════════════════════════════════════════

# 分支维度常量（与 actor.py 保持一致）
_EE_DIM = 38
_CONTEXT_DIM = 2
_STAB_DIM = 2
_DOOR_GEOMETRY_DIM = 6


def _compute_proprio_dim() -> int:
    """计算 proprio 分支的输入维度。"""
    total_joints = 12
    dim = total_joints * 2  # q + dq
    dim += total_joints  # prev_joint_target
    return dim


class ActorExportWrapper(nn.Module):
    """将 Actor 网络包装为适合 ONNX / TorchScript 导出的接口。

    原始 Actor 接受 ``dict[str, Tensor]`` 分支输入，不适合直接导出。
    本包装器将所有分支拼接为一个 ``(batch, flat_dim)`` 的扁平张量输入，
    在内部切分后还原为分支字典再送入 Actor。

    输入张量拼接顺序::

        [proprio | ee | context | stability | door_geometry]

    Parameters
    ----------
    actor : Actor
        训练好的 Actor 网络实例。
    """

    def __init__(self, actor: "Actor") -> None:
        super().__init__()
        self.actor = actor

        self.proprio_dim = _compute_proprio_dim()
        self.ee_dim = _EE_DIM
        self.context_dim = _CONTEXT_DIM
        self.stab_dim = _STAB_DIM
        self.door_geometry_dim = _DOOR_GEOMETRY_DIM

        self.flat_dim = (
            self.proprio_dim
            + self.ee_dim
            + self.context_dim
            + self.stab_dim
            + self.door_geometry_dim
        )

        # 记录切分索引
        self._splits = [
            self.proprio_dim,
            self.ee_dim,
            self.context_dim,
            self.stab_dim,
            self.door_geometry_dim,
        ]

    def forward(
        self,
        flat_input: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向推理。

        Parameters
        ----------
        flat_input : ``(batch, flat_dim)``
            所有分支按 [proprio, ee, context, stability, door_geometry] 顺序拼接。
        h_in : ``(num_layers, batch, hidden_dim)``
            RNN 隐状态 h。
        c_in : ``(num_layers, batch, hidden_dim)``
            RNN 隐状态 c。对 GRU 无意义，传入全零张量即可。

        Returns
        -------
        action : ``(batch, 12)``
        h_out : ``(num_layers, batch, hidden_dim)``
        c_out : ``(num_layers, batch, hidden_dim)``
            对 GRU，c_out 始终为全零。
        """
        # 切分为各分支
        proprio, ee, context, stability, door_geometry = torch.split(
            flat_input, self._splits, dim=-1
        )

        flat_obs = {
            "proprio": proprio,
            "ee": ee,
            "context": context,
            "stability": stability,
            "door_geometry": door_geometry,
        }

        # 构造隐状态
        is_lstm = self.actor.cfg.rnn_type == "lstm"
        hidden = (h_in, c_in) if is_lstm else h_in

        action, hidden_new = self.actor.act(
            flat_obs, hidden=hidden, deterministic=True
        )

        if is_lstm:
            h_out, c_out = hidden_new
        else:
            h_out = hidden_new
            c_out = torch.zeros_like(h_out)

        return action, h_out, c_out


# ═══════════════════════════════════════════════════════════════════════
# 导出逻辑
# ═══════════════════════════════════════════════════════════════════════

def _export_onnx(
    wrapper: ActorExportWrapper,
    output_path: Path,
    device: torch.device,
) -> None:
    """导出为 ONNX 格式。"""
    batch = 1
    rnn_layers = wrapper.actor.cfg.rnn_layers
    rnn_hidden = wrapper.actor.cfg.rnn_hidden

    dummy_input = torch.randn(batch, wrapper.flat_dim, device=device)
    dummy_h = torch.zeros(rnn_layers, batch, rnn_hidden, device=device)
    dummy_c = torch.zeros(rnn_layers, batch, rnn_hidden, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_h, dummy_c),
        str(output_path),
        opset_version=17,
        input_names=["flat_input", "h_in", "c_in"],
        output_names=["action", "h_out", "c_out"],
        dynamic_axes={
            "flat_input": {0: "batch"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "action": {0: "batch"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"},
        },
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX 导出完成: {output_path} ({size_mb:.2f} MB)")
    print(f"  输入: flat_input=(B, {wrapper.flat_dim}), h_in=({rnn_layers}, B, {rnn_hidden}), c_in=({rnn_layers}, B, {rnn_hidden})")
    print(f"  输出: action=(B, 12), h_out=({rnn_layers}, B, {rnn_hidden}), c_out=({rnn_layers}, B, {rnn_hidden})")


def _export_torchscript(
    wrapper: ActorExportWrapper,
    output_path: Path,
    device: torch.device,
) -> None:
    """导出为 TorchScript 格式（traced）。"""
    batch = 1
    rnn_layers = wrapper.actor.cfg.rnn_layers
    rnn_hidden = wrapper.actor.cfg.rnn_hidden

    dummy_input = torch.randn(batch, wrapper.flat_dim, device=device)
    dummy_h = torch.zeros(rnn_layers, batch, rnn_hidden, device=device)
    dummy_c = torch.zeros(rnn_layers, batch, rnn_hidden, device=device)

    traced = torch.jit.trace(wrapper, (dummy_input, dummy_h, dummy_c))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  TorchScript 导出完成: {output_path} ({size_mb:.2f} MB)")
    print(f"  输入: flat_input=(B, {wrapper.flat_dim}), h_in=({rnn_layers}, B, {rnn_hidden}), c_in=({rnn_layers}, B, {rnn_hidden})")
    print(f"  输出: action=(B, 12), h_out=({rnn_layers}, B, {rnn_hidden}), c_out=({rnn_layers}, B, {rnn_hidden})")


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    """策略导出主入口。"""
    parser = argparse.ArgumentParser(
        description="将训练好的 Actor 网络导出为 ONNX / TorchScript 格式"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="训练 checkpoint 路径（.pt 文件）",
    )
    parser.add_argument(
        "--config", default="configs/training/default.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript", "both"],
        default="onnx",
        help="导出格式 (默认: onnx)",
    )
    parser.add_argument(
        "--output-dir", default="exports",
        help="导出文件输出目录 (默认: exports/)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="推理设备 (默认: cpu)",
    )
    args = parser.parse_args()

    from train import load_config, build_models

    # ── 加载配置和模型 ──────────────────────────────────────
    device = torch.device(args.device)
    config_path = str(_PROJECT_ROOT / args.config)
    cfg = load_config(config_path)

    actor, _critic, actor_cfg = build_models(cfg, device)
    actor.eval()

    # ── 加载 checkpoint 权重 ────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"错误: checkpoint 文件不存在: {ckpt_path}")
        return 1

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    print(f"已加载 checkpoint: {ckpt_path}")

    iteration = checkpoint.get("iteration", "?")
    global_steps = checkpoint.get("global_steps", "?")
    print(f"  训练轮次: {iteration}, 全局步数: {global_steps}")

    # ── 创建导出包装器 ──────────────────────────────────────
    wrapper = ActorExportWrapper(actor)
    wrapper.eval()
    print(f"  扁平输入维度: {wrapper.flat_dim}")
    print(f"  RNN 类型: {actor_cfg.rnn_type}, 隐层: {actor_cfg.rnn_hidden}, 层数: {actor_cfg.rnn_layers}")

    # ── 执行导出 ────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    fmt = args.format

    print(f"\n导出格式: {fmt}")
    print(f"输出目录: {output_dir}")
    print("─" * 50)

    if fmt in ("onnx", "both"):
        _export_onnx(wrapper, output_dir / "actor.onnx", device)

    if fmt in ("torchscript", "both"):
        _export_torchscript(wrapper, output_dir / "actor.pt", device)

    print("\n导出完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
