# TensorBoard 训练监控指南

## 启动方式

### 1. 服务器端（在 tmux 中）

```bash
# 确保 setuptools 版本兼容
pip install "setuptools<71"

# 启动 TensorBoard
tensorboard --logdir=/root/Affordance-Guided-Interaction/runs --bind_all --port 6006
```

### 2. 本地端口转发

```bash
ssh -L 6006:localhost:6006 root@39.105.12.60 -p 6026
```

### 3. 浏览器访问

打开 `http://localhost:6006`

---

## TensorBoard 各参数含义

### `train/` — 训练核心指标

| 指标 | 含义 | 如何判断好坏 |
|------|------|-------------|
| **actor_loss** | 策略网络（Actor）的 PPO 损失，包含策略梯度和熵正则项 | 训练初期会波动，长期应趋于稳定，不需要趋近 0 |
| **critic_loss** | 价值网络（Critic）的预测误差，通常是 MSE 或 Huber loss | 应逐步下降，表示价值估计越来越准 |
| **entropy** | 策略输出动作分布的信息熵，衡量探索程度 | 训练初期较高（充分探索），随训练缓慢下降。如果快速降到接近 0 说明策略过早收敛（模式崩溃） |
| **clip_fraction** | PPO clip 触发比例，即策略更新幅度超过 ε=0.2 的样本占比 | 正常应 < 0.1~0.2。如果持续很高说明更新步太大，训练不稳定 |
| **approx_kl** | 新旧策略间的近似 KL 散度，衡量策略变化幅度 | 正常应 < 0.02~0.05。太大说明更新太激进 |
| **explained_variance** | Critic 对 returns 的解释方差（R²），衡量价值函数质量 | 范围 [-∞, 1]，越接近 1 越好。负值说明 Critic 预测还不如均值 |
| **fps** | 每秒处理的环境交互步数，按完整迭代（rollout + GAE + update）总耗时计算 | 越高越好，反映整体训练吞吐量 |

### `timing/` — 各阶段耗时

这些参数记录的是**每一轮迭代中各阶段的耗时**。一轮迭代的流程：

```
┌─────────────────────────────────────────────────────────────┐
│  一轮 iteration                                              │
│                                                              │
│  ┌──────────────────────┐  ┌───────────┐  ┌──────────────┐  │
│  │   rollout_s          │  │ update_s  │  │ GAE + 其他   │  │
│  │  (轨迹采集)           │  │ (PPO更新) │  │              │  │
│  │                      │  │           │  │              │  │
│  │  ┌────────────────┐  │  │           │  │              │  │
│  │  │   vision_s     │  │  │           │  │              │  │
│  │  │  (视觉感知总和) │  │  │           │  │              │  │
│  │  │                │  │  │           │  │              │  │
│  │  │ camera_fetch_s │  │  │           │  │              │  │
│  │  │ segmentation_s │  │  │           │  │              │  │
│  │  │ pointcloud_s   │  │  │           │  │              │  │
│  │  │ encoder_s      │  │  │           │  │              │  │
│  │  └────────────────┘  │  │           │  │              │  │
│  └──────────────────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### **rollout_s** — 轨迹采集总耗时

从 `collector.collect()` 开始到结束的完整耗时。包含：
- `n_steps_per_rollout`（默认 128）步环境交互
- Actor/Critic 前向推理
- 视觉感知（嵌入其中）
- 数据写入 Buffer

这是一轮迭代中**最耗时的部分**。

#### **vision_s** — 视觉感知管线总耗时

是 `rollout_s` 的**子集**。包含完整的视觉感知流程：
- 从环境获取相机图像（RGB + 深度）
- 语义分割（生成门区域 mask）
- 深度投影（生成 3D 点云）
- Point-MAE 编码器前向推理（生成 768 维 visual embedding）
- 将 embedding 注入观测数据

这个值是**累积的**——`n_steps_per_rollout` 步中每步都会调用视觉感知，`vision_s` 是所有步的总和。

`vision_s` 可进一步分解为以下子计时（它们的总和 ≈ `vision_s`）：

| 子指标 | 含义 |
|--------|------|
| **camera_fetch_s** | 获取相机图像（RGB + 深度）的耗时 |
| **segmentation_s** | 语义分割模型推理耗时（LangSAM / Grounded-SAM2） |
| **pointcloud_s** | 深度投影 → 3D 点云构建 + FPS 下采样的耗时 |
| **encoder_s** | Point-MAE 编码器前向推理耗时 |

**如果 `vision_s` 接近 `rollout_s`，说明视觉感知是性能瓶颈，可查看子计时定位具体瓶颈环节。**

#### **update_s** — PPO 参数更新耗时

从 `ppo_trainer.update(buffer)` 开始到结束的耗时。包含：
- 从 Buffer 中抽取 mini-batch
- TBPTT（截断反向传播）序列处理
- Actor/Critic 反向传播 + 梯度裁剪
- 优化器更新参数（共 4 mini-batches × 5 epochs = 20 次更新）

这部分通常在**秒级**，一般不是瓶颈。

#### **env_steps_per_s** — 纯环境交互速率

计算方式：

```
env_steps_per_s = steps_this_iter / rollout_s
              = (n_envs × n_steps_per_rollout) / rollout_s
```

即每秒能完成多少个环境步。这个值**越高越好**，反映环境 + 策略推理的吞吐量。

注意：此指标仅按 `rollout_s` 计算，不包含 PPO 更新耗时。`train/fps` 则按完整迭代时间计算，两者互补。

#### 瓶颈定位

| 如果 | 说明 |
|------|------|
| `vision_s` ≈ `rollout_s`（比如都是 900+ 秒） | 视觉感知是瓶颈，查看子计时确定是分割、点云还是编码器 |
| `segmentation_s` 占 `vision_s` 大头 | 语义分割模型推理太慢，考虑换更快的模型（如 Grounded-SAM2） |
| `encoder_s` 占 `vision_s` 大头 | Point-MAE 编码器推理太慢，考虑 AMP 或减少点数 |
| `vision_s` 很小，`rollout_s` 很大 | 环境仿真本身慢（Isaac Sim 渲染/物理） |
| `update_s` 很大 | PPO 更新有问题（不太可能） |

### `collect/` — 环境交互指标

| 指标 | 含义 |
|------|------|
| **mean_reward** | 本轮所有环境所有步的平均即时奖励 |
| **completed_episodes** | 本轮完成的 episode 总数 |
| **successful_episodes** | 本轮成功的 episode 总数 |
| **episode_success_rate** | 成功 episode / 完成 episode（**最核心指标**） |

以下指标按**持杯占位上下文**（affordance context）拆分成功率。机器人在推门的同时手部可能持有杯子，"左/右"指左右手的持杯状态：

| 指标 | 含义 |
|------|------|
| **success_none** | 上下文为"无杯"时的成功率（左右手都没持杯） |
| **success_left_only** | 上下文为"仅左手持杯"时的成功率 |
| **success_right_only** | 上下文为"仅右手持杯"时的成功率 |
| **success_both** | 上下文为"双手持杯"时的成功率（最终目标） |
| **success_mixed** | 所有上下文混合的整体成功率，等于 `episode_success_rate` |

### `curriculum/` — 课程学习

| 指标 | 含义 |
|------|------|
| **stage** | 当前课程难度阶段编号（整数），越大越难 |
| **window_mean** | 最近 N 个 episode 的滑动窗口平均成功率 |

课程学习分三个阶段，逐步增加持杯难度：
- **Stage 1** — 无杯推门（学习基础视觉引导）
- **Stage 2** — 单臂持杯推门（left_only / right_only 各 50%）
- **Stage 3** — 混合分布（none / left_only / right_only / both 各 25%）

课程管理器根据 `window_mean` 是否超过阈值（默认 0.8）来自动提升阶段。`stage` 在图上呈阶梯状上升。

### `metrics/` — 聚合统计

`TrainingMetrics` 聚合器输出的长期均值统计量，用于观察更长周期内的趋势。包含以下子标签：

#### `metrics/episode/` — Episode 级统计

| 指标 | 含义 |
|------|------|
| **success_rate** | 当前聚合周期的整体成功率 |
| **success_mixed** | 同 `success_rate`（兼容别名） |
| **cup_drop_rate** | 杯体脱落率（推门过程中杯子从手中脱落的比例） |
| **count** | 聚合周期内的 episode 数量 |
| **mean_length** | 平均 episode 长度（步数） |
| **success_none** | "无杯"上下文的成功率 |
| **success_left_only** | "仅左手持杯"上下文的成功率 |
| **success_right_only** | "仅右手持杯"上下文的成功率 |
| **success_both** | "双手持杯"上下文的成功率 |

#### `metrics/reward/` — 分项奖励均值

动态标签，取决于 `DoorPushEnv._get_rewards()` 返回的 `reward_info` 字典内容。常见标签如 `metrics/reward/task`、`metrics/reward/stability`、`metrics/reward/safety` 等。

#### `metrics/ppo/` — PPO 指标均值

| 指标 | 含义 |
|------|------|
| **ppo/actor_loss** | 聚合周期内的 Actor loss 均值 |
| **ppo/critic_loss** | 聚合周期内的 Critic loss 均值 |
| **ppo/entropy** | 聚合周期内的策略熵均值 |
| **ppo/clip_fraction** | 聚合周期内的 clip 比例均值 |
| **ppo/approx_kl** | 聚合周期内的 KL 散度均值 |
| **ppo/explained_variance** | 聚合周期内的解释方差均值 |

---

## 快速判断训练效果

只需关注以下四条曲线：

1. **`collect/episode_success_rate`** — 成功率是否在上升
2. **`train/entropy`** — 是否在缓慢下降（不是暴跌）
3. **`curriculum/stage`** — 是否在逐步推进
4. **`train/fps`** — 训练速度是否稳定

---

## 常见问题排查

| 现象 | 可能原因 | 建议 |
|------|---------|------|
| success_rate 始终为 0 | 奖励设计或环境初始化问题 | 检查环境 reset 和奖励函数 |
| entropy 快速降到 0 | 策略过早收敛 | 增大 `entropy_coef`（如 0.02~0.05） |
| clip_fraction 持续 > 0.3 | 更新步太大 | 减小 `actor_lr` 或增大 `clip_eps` |
| approx_kl > 0.1 | 策略更新过于激进 | 减小学习率 |
| fps 突然下降 | GPU 显存不足或环境异常 | 减小 `num_envs` |
| stage 始终不提升 | 成功率未达到课程阈值 | 检查阈值设置或延长训练时间 |
