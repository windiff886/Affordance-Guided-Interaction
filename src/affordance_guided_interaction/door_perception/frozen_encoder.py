"""冻结的 Point-MAE 编码器：内联网络结构，开箱加载预训练权重。

权重完全冻结（requires_grad=False + eval 模式），不参与任何训练。
网络组件直接内联自 Point-MAE 官方仓库 (https://github.com/Pang-Yatian/Point-MAE)，
避免对外部 knn_cuda / point_mae 等第三方 CUDA 扩展包的依赖。

使用方式:
    encoder = PointMAEEncoder(config)
    embedding = encoder.encode(points)  # (embed_dim,) numpy array
"""

from __future__ import annotations

import logging

import numpy as np

from affordance_guided_interaction.door_perception.config import PointMAEEncoderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内联 Point-MAE 网络组件（纯 PyTorch，无 CUDA 扩展依赖）
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from timm.models.layers import DropPath, trunc_normal_  # type: ignore[import-untyped]

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _check_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch 和 timm 是 Point-MAE 编码器的必要依赖。"
            "请安装: pip install torch timm"
        )


# ---- 最远点采样（FPS）：纯 PyTorch 实现，替代 CUDA 扩展 ----

def _farthest_point_sample(xyz: "torch.Tensor", npoint: int) -> "torch.Tensor":
    """最远点采样 (Farthest Point Sampling)。

    Args:
        xyz: (B, N, 3) 输入点云坐标。
        npoint: 需要采样的中心点数。

    Returns:
        (B, npoint, 3) 采样后的中心点坐标。
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids_idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_arange = torch.arange(B, device=device)

    for i in range(npoint):
        centroids_idx[:, i] = farthest
        centroid = xyz[batch_arange, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)  # (B,)

    # 收集中心点坐标
    idx_expand = centroids_idx.unsqueeze(-1).expand(-1, -1, 3)  # (B, npoint, 3)
    centers = torch.gather(xyz, 1, idx_expand)
    return centers


# ---- K 近邻搜索：基于 torch.cdist，替代 knn_cuda ----

def _knn(xyz: "torch.Tensor", center: "torch.Tensor", k: int) -> "torch.Tensor":
    """K 近邻搜索。

    Args:
        xyz: (B, N, 3) 全部点。
        center: (B, G, 3) 中心点。
        k: 近邻数。

    Returns:
        (B, G, K) 近邻索引。
    """
    dist = torch.cdist(center, xyz)  # (B, G, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, G, K)
    return idx


# ---- Point-MAE 局部点云编码器（Conv1d 提取局部特征） ----

if _HAS_TORCH:

    class _PointEncoder(nn.Module):
        """逐组点云编码器：将每个局部点组 (G, S, 3) 编码为 (G, C) 特征。

        结构与 Point-MAE 官方 Encoder 类完全一致。
        """

        def __init__(self, encoder_channel: int) -> None:
            super().__init__()
            self.encoder_channel = encoder_channel
            self.first_conv = nn.Sequential(
                nn.Conv1d(3, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1),
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, self.encoder_channel, 1),
            )

        def forward(self, point_groups: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                point_groups: (B, G, S, 3) 分组后的局部点云。
            Returns:
                (B, G, C) 每组的全局特征。
            """
            bs, g, s, _ = point_groups.shape
            x = point_groups.reshape(bs * g, s, 3)
            # 局部编码
            feat = self.first_conv(x.transpose(2, 1))  # (BG, 256, S)
            feat_global = torch.max(feat, dim=2, keepdim=True)[0]  # (BG, 256, 1)
            feat = torch.cat(
                [feat_global.expand(-1, -1, s), feat], dim=1
            )  # (BG, 512, S)
            feat = self.second_conv(feat)  # (BG, C, S)
            feat_global = torch.max(feat, dim=2, keepdim=False)[0]  # (BG, C)
            return feat_global.reshape(bs, g, self.encoder_channel)

    class _Mlp(nn.Module):
        """前馈网络（与 Point-MAE 官方一致）。"""

        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            drop: float = 0.0,
        ) -> None:
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class _Attention(nn.Module):
        """多头自注意力（与 Point-MAE 官方一致）。"""

        def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
        ) -> None:
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim**-0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class _Block(nn.Module):
        """Transformer 编码块（与 Point-MAE 官方一致）。"""

        def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
            self.attn = _Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    class _TransformerEncoder(nn.Module):
        """Transformer 编码器堆叠（与 Point-MAE 官方一致）。"""

        def __init__(
            self,
            embed_dim: int = 384,
            depth: int = 12,
            num_heads: int = 6,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
        ) -> None:
            super().__init__()
            dpr = (
                [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
                if isinstance(drop_path_rate, float)
                else drop_path_rate
            )
            self.blocks = nn.ModuleList(
                [
                    _Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                    )
                    for i in range(depth)
                ]
            )

        def forward(
            self, x: "torch.Tensor", pos: "torch.Tensor"
        ) -> "torch.Tensor":
            for block in self.blocks:
                x = block(x + pos)
            return x

    class _PointMAEFeatureExtractor(nn.Module):
        """Point-MAE 特征提取器。

        仅包含编码器部分（不含 MAE 解码器和分类头）。
        输出为 mean_pool + max_pool 的拼接特征，维度 = 2 * trans_dim。
        不使用 cls_token（避免引入未经预训练的随机参数）。
        """

        def __init__(self, config: PointMAEEncoderConfig) -> None:
            super().__init__()
            self.num_group = config.num_group
            self.group_size = config.group_size
            self.trans_dim = config.trans_dim

            # 局部点云编码器
            self.encoder = _PointEncoder(config.encoder_dims)

            # 位置编码
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, config.trans_dim),
            )

            # Transformer 编码堆叠
            drop_path_rate = 0.1
            dpr = [
                x.item()
                for x in torch.linspace(0, drop_path_rate, config.depth)
            ]
            self.blocks = _TransformerEncoder(
                embed_dim=config.trans_dim,
                depth=config.depth,
                drop_path_rate=dpr,
                num_heads=config.num_heads,
            )

            # Layer Norm
            self.norm = nn.LayerNorm(config.trans_dim)

        def forward(self, pts: "torch.Tensor") -> "torch.Tensor":
            """提取点云特征。

            Args:
                pts: (B, N, 3) 输入点云。

            Returns:
                (B, 2 * trans_dim) 特征向量。
            """
            B, N, _ = pts.shape
            device = pts.device

            # 1. FPS 选取中心点
            center = _farthest_point_sample(pts, self.num_group)  # (B, G, 3)

            # 2. KNN 构建局部邻域
            idx = _knn(pts, center, self.group_size)  # (B, G, K)
            idx_base = (
                torch.arange(0, B, device=device).view(-1, 1, 1) * N
            )
            idx_flat = (idx + idx_base).view(-1)
            neighborhood = pts.view(B * N, -1)[idx_flat, :]
            neighborhood = neighborhood.view(
                B, self.num_group, self.group_size, 3
            ).contiguous()
            # 归一化：减去中心点坐标
            neighborhood = neighborhood - center.unsqueeze(2)

            # 3. 局部编码
            group_tokens = self.encoder(neighborhood)  # (B, G, C)

            # 4. 位置编码
            pos = self.pos_embed(center)  # (B, G, C)

            # 5. Transformer 编码
            x = self.blocks(group_tokens, pos)  # (B, G, C)
            x = self.norm(x)

            # 6. 全局池化：mean + max 拼接
            mean_f = x.mean(dim=1)  # (B, C)
            max_f = x.max(dim=1)[0]  # (B, C)
            return torch.cat([mean_f, max_f], dim=-1)  # (B, 2C)


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


class PointMAEEncoder:
    """冻结的 Point-MAE 编码器（对外唯一接口）。

    懒加载模型：首次调用 encode() 时才会实际加载权重和构建网络。
    所有权重均为冻结状态 (eval + requires_grad=False)。
    """

    def __init__(self, config: PointMAEEncoderConfig) -> None:
        self._config = config
        self._model: "_PointMAEFeatureExtractor | None" = None

    def _ensure_model(self) -> None:
        """懒加载：首次调用时构建并冻结模型。"""
        if self._model is not None:
            return

        _check_torch()
        import torch as _torch

        model = _PointMAEFeatureExtractor(self._config)

        # 加载预训练权重
        if self._config.checkpoint_path:
            self._load_pretrained(model)
        else:
            logger.warning(
                "未指定 Point-MAE 预训练权重路径 (checkpoint_path 为空)，"
                "编码器将使用随机初始化权重。"
            )

        # 冻结所有参数
        model.eval()
        model.to(self._config.device)
        for p in model.parameters():
            p.requires_grad_(False)

        self._model = model
        logger.info(
            "Point-MAE 编码器已加载并冻结。输出维度: %d，设备: %s",
            self._config.embed_dim,
            self._config.device,
        )

    def _load_pretrained(self, model: "_PointMAEFeatureExtractor") -> None:
        """加载 Point-MAE 官方预训练权重，执行必要的 key 映射。

        官方 pretrain.pth 的 state_dict 格式：
            base_model.MAE_encoder.encoder.xxx  → 映射到 encoder.xxx
            base_model.MAE_encoder.blocks.xxx   → 映射到 blocks.xxx
            base_model.MAE_encoder.norm.xxx     → 映射到 norm.xxx
            base_model.MAE_encoder.pos_embed.xxx → 映射到 pos_embed.xxx
        """
        import torch as _torch

        # 支持相对路径：相对于项目根目录解析
        from pathlib import Path as _Path
        ckpt_path = _Path(self._config.checkpoint_path)
        if not ckpt_path.is_absolute():
            _project_root = _Path(__file__).resolve().parents[3]  # door_perception → src → project root
            ckpt_path = _project_root / ckpt_path
        logger.info("正在加载 Point-MAE 权重: %s", ckpt_path)
        ckpt = _torch.load(
            str(ckpt_path), map_location="cpu", weights_only=False
        )

        # 提取 state_dict（兼容多种存储格式）
        if "base_model" in ckpt:
            raw_state = ckpt["base_model"]
        elif "model" in ckpt:
            raw_state = ckpt["model"]
        elif "state_dict" in ckpt:
            raw_state = ckpt["state_dict"]
        else:
            raw_state = ckpt

        # key 映射：去除 module. 前缀，剥离 MAE_encoder. 前缀
        mapped_state: dict[str, "torch.Tensor"] = {}
        for k, v in raw_state.items():
            new_key = k.replace("module.", "")
            if new_key.startswith("MAE_encoder."):
                new_key = new_key[len("MAE_encoder."):]
            elif new_key.startswith("base_model."):
                new_key = new_key[len("base_model."):]
            mapped_state[new_key] = v

        # 只保留模型中存在的 key
        model_keys = set(model.state_dict().keys())
        filtered_state = {k: v for k, v in mapped_state.items() if k in model_keys}

        incompatible = model.load_state_dict(filtered_state, strict=False)
        if incompatible.missing_keys:
            logger.debug("权重加载 - 缺失的 key: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            logger.debug("权重加载 - 多余的 key: %s", incompatible.unexpected_keys)
        logger.info(
            "权重加载完成。匹配: %d / %d",
            len(filtered_state),
            len(model_keys),
        )

    def encode(self, points: np.ndarray) -> np.ndarray:
        """将点云编码为固定维度的 embedding 向量。

        Args:
            points: (N, 3) numpy 点云。会被自动采样/填充到固定点数。

        Returns:
            (embed_dim,) numpy embedding 向量。
        """
        _check_torch()
        import torch as _torch

        self._ensure_model()
        assert self._model is not None

        # 采样/填充到固定点数
        pts = _sample_or_pad(points, self._config.num_group * self._config.group_size)
        tensor = (
            _torch.from_numpy(pts)
            .float()
            .unsqueeze(0)
            .to(self._config.device)
        )  # (1, N, 3)

        with _torch.no_grad():
            embedding = self._model(tensor)  # (1, embed_dim)

        return embedding.squeeze(0).cpu().numpy()

    @property
    def embed_dim(self) -> int:
        """输出 embedding 的维度。"""
        return self._config.embed_dim


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _sample_or_pad(points: np.ndarray, n: int) -> np.ndarray:
    """将点云对齐到恰好 n 个点（随机采样或零填充）。"""
    if len(points) == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if len(points) >= n:
        idx = np.random.choice(len(points), size=n, replace=False)
    else:
        idx = np.random.choice(len(points), size=n, replace=True)
    return points[idx]
