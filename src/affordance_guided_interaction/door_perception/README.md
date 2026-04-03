# Door Perception Pipeline

从单帧 RGB-D 中提取门相关点云，并通过**开箱即用、完全冻结的视觉点云编码器（Vision Encoder）**，直接将其编码为 policy 可用的高维 affordance latent（`z_aff`）。
全流程严格坚持**现成预训练大模型直出，绝对不包含任何训练感知/分割模型的步骤**。

## 管线概览 (端到端新范式)

```
RGB-D + task_goal
  │
  ├─ 1. 开集分割 (LangSAM / Grounded-SAM 2) 【开箱即用】
  │     输入文本提示 "door" / "button" / "handle"
  │     输出特定交互区域的 binary mask
  │
  ├─ 2. 深度反投影 & 基础降采样
  │     截取 mask 区域的 Depth 生成局部点云
  │     仅做 Voxel downsampling (单纯为了控制送入网络的最大点数，不需要繁复滤波)
  │
  ├─ 3. 直入视觉 Encoder (Point-MAE) 【权重完全冻结】
  │     将局部点云直接喂给预训练开箱即用的大模型
  │     输出高维的深层语义 Embeddings
  │
  └─ 4. 传输给下游 Policy
        Embeddings 作为 `z_aff` 进入控制策略网络
```

## 为什么要砍掉过往的手工几何特征？

基于现代 End-to-End 的思想，我们移除了此前的“RANSAC 提取门板法向”、“手工计算包围盒大小”、“手工测算夹爪到门板绝对距离”等 25 维度的几何特征工程。因为：

1. **Policy 自己能学懂空间关系**：机械臂带有本体状态（Proprioception 的位姿），结合视觉 Encoder 吐出的高维点云特征，Policy 本身就应该拥有隐式估算门板法向与距离的泛化能力。
2. **纯粹、干净、直出**：消除大量不稳定且需要调参的传统视觉空间解算流程，避免因为错误平面拟合带来的误差级联崩溃。
3. **贯彻零训练 (Zero-Training) 与开箱即用**：不论是 2D 分割 还是 3D 编码，全部复用大型预训练基座模型。整个 perception 管线变成了一个纯粹的正向推理黑盒通道。

## 文件结构规划

| 文件 | 职责 |
|---|---|
| `config.py` | 配置参数表：相机内参、SAM 提示词配置、冻结点云编码器的基座权重路径等 |
| `segmentation.py` | `OpenVocabSegmentor` — 开箱封装 LangSAM / Grounded-SAM 2 模型 |
| `depth_projection.py` | `backproject_depth()` — 从 Mask 生成局部点云并进行点数对齐截断/降采样 |
| `frozen_encoder.py` | 加载 3D 基础模型 Point-MAE，**并施加 `requires_grad=False` 与 `.eval()` 冻结所有权重** |
| `affordance_pipeline.py` | `AffordancePipeline` — 管线入口，串联从 RGB-D 到特征 Embedding 直出的执行链 |

> **注：** 此前用于提取 25D 人工结构化参数的 `geometric_summary.py` 以及用于离群点剔除/平面求解的 `point_cloud_processing.py` 已被删除。

## 抽象接口期望

```python
from affordance_guided_interaction.door_perception import (
    AffordancePipeline,
    AffordancePipelineConfig,
    CameraIntrinsics,
)

config = AffordancePipelineConfig(
    camera=CameraIntrinsics(fx=606, fy=606, cx=320, cy=240, width=640, height=480),
    encoder_type="point_mae",   # 使用能开箱即用的预训练模型
)
pipeline = AffordancePipeline(config)

observation = {
    "rgb": rgb_image,           # (H, W, 3) uint8
    "depth": depth_map,         # (H, W) float, metres
    # ... 其余信息留给低层 Policy 处理，Pipeline 内不主动关心夹爪具体坐标距离 ...
}

# 真正的一步直出：SAM 开集分割 -> 局部点云化 -> 预训练 3D 网络推理 -> 输出 Embeddings
z_aff_embedding = pipeline.encode(observation=observation, task_goal="push")

# Policy 将直接接收这份深维隐含特征
z_aff_feature = z_aff_embedding  # 比如 (768,) 维度的 torch tensor
```

## 接口边界：`z_prog` 不属于本模块

按当前确认的目标架构，`door_perception` 只负责输出门相关视觉 affordance 表征，不负责输出任务进展向量。

这意味着：

1. 对外目标接口只保留 `z_aff`
2. `door_angle`、`button_pressed`、`handle_triggered`、`progress` 等任务进展量不属于 perception 职责
3. 若系统仍需要任务进展信号，应由 `envs/`、`task_manager.py` 或 `rewards/` 等非感知层维护

说明：若在历史分支或旧文档中看到 `z_prog`，应将其视为已废弃的历史残留，而不是本 README 定义的目标接口。

## 外部依赖标准

为了保证“一切开箱即用”的要求，感知大模块**只引入带预训练权重的标准库**，杜绝冷启动训练环境：
1. **2D 感知**：`lang-sam` 或 `grounded-sam-2`
2. **3D 编码**：基础模型权重库与 `torch`（纯推理模式）
