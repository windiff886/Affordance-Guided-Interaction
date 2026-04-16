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

> **日志频率**：TensorBoard 指标在每轮迭代结束时写入（即每 `n_steps_per_rollout` 步）。控制台日志受 `log_interval` 控制（默认每 5 轮打印一次）。

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
│  └──────────────────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### **rollout_s** — 轨迹采集总耗时

从 `collector.collect()` 开始到结束的完整耗时。包含：
- `n_steps_per_rollout` 步环境交互（当前配置为 16）
- Actor/Critic 前向推理
- 观测数据准备（door geometry ground truth）
- 数据写入 Buffer

这是一轮迭代中**最耗时的部分**。

#### **update_s** — PPO 参数更新耗时

从 `ppo_trainer.update(buffer)` 开始到结束的耗时。包含：
- 从 Buffer 中抽取 mini-batch
- TBPTT（截断反向传播）序列处理
- Actor/Critic 反向传播 + 梯度裁剪
- 优化器更新参数（共 `num_mini_batches` × `num_epochs` 次更新，当前配置为 16 × 3 = 48 次）

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
| `rollout_s` 很大 | 环境仿真本身慢（Isaac Sim 渲染/物理），或策略推理开销大 |
| `update_s` 很大 | PPO 更新有问题（不太可能） |
| `train/fps` 远小于 `timing/env_steps_per_s` | PPO 更新占了大量时间 |

> **注**：视觉感知管线的子计时（`vision_s`、`camera_fetch_s` 等）为历史实验代码残留，仅在显式启用 `perception_runtime` 时有效。当前默认训练路径使用仿真 ground truth 的 door geometry（6 维），不涉及任何视觉管线计时。

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

### `reward/` — 奖励总项

`reward/` 页面只保留总奖励和四个一级分量，方便先看整体趋势。所有这些曲线都记录的是**本轮 rollout 内所有环境、所有步的平均值**。

| 指标 | 含义 |
|------|------|
| **total** | 本轮平均总奖励，等于 `r_task + r_stab^L + r_stab^R - r_safe` |
| **task** | 主任务奖励总量 `r_task` |
| **stab_left** | 左臂稳定性奖励总量 `r_stab^L`，已乘左臂 occupancy mask |
| **stab_right** | 右臂稳定性奖励总量 `r_stab^R`，已乘右臂 occupancy mask |
| **safe** | 安全惩罚总量 `r_safe`，按正惩罚量记录，进入总奖励时会被减掉 |

### `reward_terms/` — 奖励子项

`reward_terms/` 页面只放细分子项，用来排查到底是哪一项在主导 reward 变化。

- `reward_terms/task/*`：`delta`、`open_bonus`、`approach`、`approach_raw`
- `reward_terms/stab_left/*` 与 `reward_terms/stab_right/*`：`zero_acc`、`zero_ang`、`acc`、`ang`、`tilt`
- `reward_terms/safe/*`：`joint_vel`、`target_limit`、`cup_drop`

**解释时要注意符号：**
- `reward/task`、`reward/stab_left`、`reward/stab_right` 和对应的 `reward_terms/stab_*/*` / `reward_terms/task/*` 都按进入总奖励的有符号贡献记录。
- `reward/safe` 与 `reward_terms/safe/*` 都按正惩罚量记录，最终总奖励通过减去它们得到。
- 在 `stage_1`（无杯）中，`reward/stab_left`、`reward/stab_right` 以及 `reward_terms/stab_left/*`、`reward_terms/stab_right/*` 理论上都应接近 0；如果不为 0，优先检查 occupancy mask 或日志链路。
- 当前默认路径已经删除 `smooth`、`reg`、`joint_limit`、`torque_limit` 这些旧标签；如果日志里仍出现它们，通常说明代码与文档版本不一致。

### `curriculum/` — 课程学习

| 指标 | 含义 |
|------|------|
| **stage** | 当前课程难度阶段编号（整数），越大越难 |
| **window_mean** | 最近 N 个 episode 的滑动窗口平均成功率 |

课程学习分三个阶段，逐步增加持杯难度：
- **Stage 1** — 无杯推门（学习基础推门接触）
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
| **mean_length** | 平均 episode 长度（步数） |
| **success_none** | "无杯"上下文的成功率 |
| **success_left_only** | "仅左手持杯"上下文的成功率 |
| **success_right_only** | "仅右手持杯"上下文的成功率 |
| **success_both** | "双手持杯"上下文的成功率 |

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
