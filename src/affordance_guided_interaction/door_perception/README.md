# Door Perception Pipeline

从单帧 RGB-D 中提取门相关点云，并编码为 policy 可用的 affordance observation（`z_aff` + `z_prog`）。
全流程使用现成预训练模型，无需自行训练。

## 管线概览

```
RGB-D + task_goal
  │
  ├─ 1. 开集分割 (LangSAM / Grounded-SAM 2)
  │     输入文本提示 "door" / "door handle" / "button"
  │     输出 per-part binary mask + confidence
  │
  ├─ 2. 深度反投影
  │     mask 区域像素 × 相机内参 → 3D 点云 (相机/世界坐标系)
  │
  ├─ 3. 点云清理
  │     voxel 下采样 → 统计离群移除 → 半径滤波
  │     可选: RANSAC 平面拟合 (估计门板法向)
  │
  ├─ 4. 几何摘要 → z_aff (25-D)
  │     门中心/法向/包围盒 + 把手中心 + 按钮中心
  │     + gripper 距离 + affordance 类型 one-hot + 置信度
  │
  ├─ 5. [可选] 冻结点云编码器 (Point-MAE / ULIP-2)
  │     拼接 embedding 到 z_aff 后方
  │
  └─ 6. 任务进展 → z_prog (4-D)
        门角度 + 按钮状态 + 把手状态 + 归一化进度
```

## 文件结构

| 文件 | 职责 |
|---|---|
| `config.py` | 配置 dataclass：相机内参、分割参数、点云处理参数、冻结编码器参数 |
| `segmentation.py` | `OpenVocabSegmentor` — 封装 LangSAM / Grounded-SAM 2 |
| `depth_projection.py` | `backproject_depth()` — mask + depth → N×3 点云 |
| `point_cloud_processing.py` | 体素下采样、离群移除、RANSAC 平面拟合（Open3D / numpy 双后端） |
| `geometric_summary.py` | `compute_z_aff()` — 从点云计算 25 维 affordance 向量 |
| `frozen_encoder.py` | 可选冻结 Point-MAE / ULIP-2 编码器（参数全部冻结） |
| `affordance_pipeline.py` | `AffordancePipeline` — 端到端管线，实现 `AffordanceEncoder` Protocol |

## 快速使用

```python
from affordance_guided_interaction.door_perception import (
    AffordancePipeline,
    AffordancePipelineConfig,
    CameraIntrinsics,
)

config = AffordancePipelineConfig(
    camera=CameraIntrinsics(fx=606, fy=606, cx=320, cy=240, width=640, height=480),
)
pipeline = AffordancePipeline(config)

observation = {
    "rgb": rgb_image,           # (H, W, 3) uint8
    "depth": depth_map,         # (H, W) float, metres
    "gripper_pos": gripper_xyz, # (3,) world frame
    "extrinsic": cam2world,     # (4, 4) optional
    # 仿真状态 (用于 z_prog)
    "door_angle": 0.0,
    "button_pressed": False,
    "handle_triggered": False,
}

z_aff_dict, z_prog_dict = pipeline.encode(observation=observation, task_goal="push")

z_aff_vector = z_aff_dict["vector"]    # (25,) numpy array
z_prog_vector = z_prog_dict["vector"]  # (4,)  numpy array
```

## z_aff 向量布局 (25-D)

| 索引 | 字段 | 维度 | 说明 |
|------|------|------|------|
| 0–2 | `c_door` | 3 | 门板质心 |
| 3–5 | `n_door` | 3 | 门板平面法向 |
| 6–8 | `b_door` | 3 | 门板包围盒尺寸 (dx, dy, dz) |
| 9–11 | `c_handle` | 3 | 把手质心 |
| 12–14 | `c_button` | 3 | 按钮质心 |
| 15 | `d_g_door` | 1 | gripper 到门平面的有符号距离 |
| 16 | `d_g_handle` | 1 | gripper 到把手的欧式距离 |
| 17 | `d_g_button` | 1 | gripper 到按钮的欧式距离 |
| 18–21 | `affordance_type` | 4 | one-hot [push, press, handle, sequential] |
| 22–24 | `confidence` | 3 | 门/把手/按钮的分割置信度 |

启用冻结编码器后，`z_aff_dict["vector_with_embed"]` 为 `(25 + embed_dim,)` 拼接向量。

## z_prog 向量布局 (4-D)

| 索引 | 字段 | 说明 |
|------|------|------|
| 0 | `door_angle` | 门当前开合角度 (rad) |
| 1 | `button_pressed` | 按钮是否按下 (0/1) |
| 2 | `handle_triggered` | 把手是否触发 (0/1) |
| 3 | `progress` | 归一化任务进度 [0, 1] |

## 依赖

**必需：**
- `numpy`

**分割模型 (二选一)：**
- `lang-sam` — `pip install lang-sam`（推荐，封装最简单）
- `grounded-sam-2` — 参考 [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)

**点云处理 (可选，有 numpy fallback)：**
- `open3d` — `pip install open3d`
- `scipy` — numpy fallback 使用 KDTree

**冻结编码器 (可选)：**
- `torch`
- Point-MAE 或 ULIP-2 预训练权重

## 设计说明

1. **不训练任何模型** — 分割使用 LangSAM 现成权重，编码使用冻结 Point-MAE/ULIP
2. **几何摘要优先，冻结编码器可选** — 默认只输出 25 维几何摘要，足够 RL 策略使用
3. **Open3D 可选** — 未安装时自动回退 numpy/scipy 实现，方便测试和轻量环境部署
4. **实现 `AffordanceEncoder` Protocol** — 可直接替换 `perception/` 中的占位类
