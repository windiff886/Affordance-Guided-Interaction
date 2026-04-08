"""循环主干网络：GRU / LSTM 封装。

接收各分支 encoder 拼接后的特征向量，输出经过时序建模的隐状态特征。
隐状态在 episode 内持续流转，用于隐式辨识隐藏环境参数
（如杯体质量、门阻尼）以及编码历史交互模式。
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _should_retry_without_cudnn(exc: RuntimeError) -> bool:
    """Return whether a cuDNN-backed RNN call should be retried without cuDNN."""
    msg = str(exc)
    if "cuDNN error" in msg:
        return True
    if "CUDNN_STATUS_" in msg:
        return True
    if "non-contiguous input" in msg:
        return True
    return False


class RecurrentBackbone(nn.Module):
    """GRU / LSTM 循环主干网络。

    Parameters
    ----------
    input_dim : int
        各分支特征拼接后的总维度。
    hidden_dim : int
        循环层隐状态维度，默认 512。
    num_layers : int
        循环层堆叠数，默认 1。
    rnn_type : ``"gru"`` | ``"lstm"``
        循环单元类型，默认 GRU。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        rnn_type: Literal["gru", "lstm"] = "gru",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self._warned_cudnn_fallback = False

        # 构建循环层
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    # ------------------------------------------------------------------
    # 初始化隐状态
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """创建全零初始隐状态，用于 episode 开始时调用。

        Returns
        -------
        GRU  → ``(num_layers, batch_size, hidden_dim)``
        LSTM → tuple of ``(h_0, c_0)``，每个形状同上。
        """
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """循环前向计算。

        Parameters
        ----------
        x : ``(batch, feature_dim)`` 或 ``(batch, seq_len, feature_dim)``
            当输入为 2-D 时，自动在 seq 维度上扩展为 ``(batch, 1, feature_dim)``。
        hidden : 隐状态
            来自上一步的隐状态（GRU 为 Tensor，LSTM 为 tuple）。
            传入 ``None`` 时自动初始化全零。

        Returns
        -------
        output : ``(batch, hidden_dim)``
            当前步的循环层输出特征。
        hidden_new : 形状同输入 hidden
            更新后的隐状态，需由调用方保存并在下一步传回。
        """
        # 2-D 输入 → 3-D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F)
        x = x.contiguous()

        batch_size = x.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size)
        elif isinstance(hidden, tuple):
            hidden = tuple(h.contiguous() for h in hidden)
        else:
            hidden = hidden.contiguous()

        # RNN 前向
        try:
            rnn_out, hidden_new = self.rnn(x, hidden)  # rnn_out: (B, 1, H)
        except RuntimeError as exc:
            if not _should_retry_without_cudnn(exc):
                raise

            if not self._warned_cudnn_fallback:
                logger.warning(
                    "cuDNN RNN path rejected the current tensor layout. "
                    "Falling back to the non-cuDNN RNN kernel for this call. "
                    "input_shape=%s rnn_type=%s",
                    tuple(x.shape),
                    self.rnn_type,
                )
                self._warned_cudnn_fallback = True

            with torch.backends.cudnn.flags(enabled=False):
                rnn_out, hidden_new = self.rnn(x, hidden)

        if return_sequence:
            return rnn_out, hidden_new

        # 取最后一个时间步的输出
        output = rnn_out[:, -1, :]  # (B, H)
        return output, hidden_new
