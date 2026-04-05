# Affordance-Guided Interaction

基于 **Isaac Lab** 仿真器与 **PPO** 强化学习算法，训练双臂机器人（Unitree Z1 × 2 + Dingo 底座）在持杯约束下完成推门任务。策略通过 affordance 视觉编码提取门相关特征，并在当前的三阶段 push-only 课程中逐步引入持杯约束。

---

## 快速开始

### 环境依赖

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Isaac Lab | ≥ 1.0 | 仿真引擎（含 Isaac Sim 4.x） |
| Python | ≥ 3.10 | Isaac Lab 自带的 Python 环境 |
| PyTorch | ≥ 2.0 | GPU 加速训练 |
| TensorBoard | 可选 | 训练指标可视化 |

### 安装

```bash
# 克隆仓库
git clone https://github.com/<your-org>/Affordance-Guided-Interaction.git
cd Affordance-Guided-Interaction

# 确保 Isaac Lab 的 Python 环境已激活
# 如果使用 conda：
conda activate isaaclab

# 将项目 src 加入 PYTHONPATH（或在已有 conda 环境中安装）
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 验证安装
python -c "from affordance_guided_interaction.training import PPOTrainer; print('✅ OK')"
```

---

## 训练

### 本地验证（4060 Laptop）

在正式训练前，使用少量环境快速验证训练管线是否能跑通：

```bash
python scripts/train.py \
    --config configs/training/default.yaml \
    --num-envs 2 \
    --seed 42
```

> 预期行为：控制台每轮迭代打印 `actor_loss / critic_loss / entropy / reward / stage`，无报错即管线正常。

### 服务器训练（A100）

无头模式启动完整规模训练：

```bash
python scripts/train.py \
    --config configs/training/default.yaml \
    --headless \
    --num-envs 64
```

### 断点续训

从已有 checkpoint 恢复训练：

```bash
python scripts/train.py \
    --resume checkpoints/ckpt_iter_1000.pt \
    --headless
```

### 完整命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/training/default.yaml` | 训练配置文件路径 |
| `--num-envs` | 配置文件中的值 (64) | 覆盖并行环境数量 |
| `--headless` | `false` | 无头模式（无 GUI 渲染，服务器训练） |
| `--resume` | 无 | 恢复训练的 checkpoint 文件路径 |
| `--device` | 自动检测 | 指定计算设备 (`cuda` / `cpu`) |
| `--seed` | `42` | 随机种子 |
| `--log-dir` | `runs/` | TensorBoard 日志目录 |
| `--ckpt-dir` | `checkpoints/` | Checkpoint 保存目录 |

### 推荐环境数配置

| 硬件 | `--num-envs` | 用途 |
|---|---|---|
| RTX 4060 Laptop | 2 – 4 | 本地功能验证、debug |
| RTX 4090 | 16 – 32 | 中等规模训练 |
| A100 80GB | 64 – 128 | 正式训练 |

---

## 训练监控

### TensorBoard

```bash
tensorboard --logdir runs/
```

记录的核心指标：

| 分类 | 指标 | 含义 |
|---|---|---|
| **损失** | `train/actor_loss` | 策略梯度损失 |
| | `train/critic_loss` | 价值函数损失 |
| | `train/entropy` | 策略熵（探索度） |
| **优化** | `train/clip_fraction` | PPO 裁剪比例 |
| | `train/approx_kl` | 近似 KL 散度 |
| | `train/explained_variance` | Critic 拟合质量 |
| **环境** | `collect/mean_reward` | 平均回合奖励 |
| | `collect/completed_episodes` | 完成的 episode 数 |
| **课程** | `curriculum/stage` | 当前课程阶段 (1-3) |
| | `curriculum/window_mean` | 滑动窗口平均成功率 |

### Checkpoint

Checkpoint 默认每 50 轮迭代自动保存至 `checkpoints/` 目录，训练中断 (`Ctrl+C`) 时也会自动保存最终状态。

每个 checkpoint 包含：Actor / Critic 权重、PPO 优化器状态、课程管理器进度。

---

## 配置体系

所有训练超参均通过 `configs/` 目录下的 YAML 文件管理：

```
configs/
├── training/default.yaml    # PPO 超参、总步数、mini-batch 配置
├── env/default.yaml         # 物理步长、机器人规格、USD 资产路径
├── policy/default.yaml      # Actor/Critic 网络架构参数
├── task/default.yaml        # 任务成功/失败判定条件
├── curriculum/default.yaml  # 课程跃迁窗口与阈值
└── reward/default.yaml      # SoFTA 奖励权重分配
```

修改配置后无需改动代码，`train.py` 会自动合并所有配置文件。

### 关键超参速查

```yaml
# configs/training/default.yaml
num_envs: 64               # 并行环境数
total_steps: 10_000_000    # 总训练步数
n_steps_per_rollout: 128   # 每轮采集步数

ppo:
  gamma: 0.99              # 折扣因子
  lam: 0.95                # GAE λ
  clip_eps: 0.2            # PPO 裁剪参数
  actor_lr: 3.0e-4         # Actor 学习率
  critic_lr: 3.0e-4        # Critic 学习率
  num_epochs: 5            # 优化轮数
  seq_length: 16           # TBPTT 截断长度
```

---

## 课程学习

训练采用三阶段课程自动跃迁，当滑动窗口（50 epoch）平均成功率 ≥ 80% 时自动进入下一阶段：

| 阶段 | 门类型 | 持杯概率 | 学习目标 |
|---|---|---|---|
| Stage 1 | push | 0% | 基础视觉引导接触 |
| Stage 2 | push | 100% | 力控与稳定性约束 |
| Stage 3 | push | 50% | 有杯 / 无杯混合场景下的稳定推门 |

> 当前课程只针对 push 任务调度训练难度；阶段变化只影响持杯采样概率，门类型保持为 `push`。

---

## 项目结构

```
Affordance-Guided-Interaction/
├── scripts/
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估（开发中）
│   ├── load_scene.py           # Isaac Sim 场景加载/调试
│   └── rollout_demo.py         # Rollout 可视化
├── src/affordance_guided_interaction/
│   ├── envs/                   # Isaac Lab 环境层
│   ├── observations/           # 观测构建（Actor/Critic 分支）
│   ├── policy/                 # Actor-Critic 网络
│   ├── rewards/                # SoFTA 奖励体系
│   ├── training/               # PPO 训练组件
│   ├── door_perception/        # 视觉感知模块
│   └── utils/                  # 工具函数
├── src/teleop_cup_grasp/       # 杯体抓取遥操作（独立模块）
├── configs/                    # YAML 配置文件
├── assets/                     # USD 仿真资产
└── checkpoints/                # 训练 checkpoint（自动生成）
```

---

## 技术架构

```
                    ┌─────────────────────────────┐
                    │     CurriculumManager       │
                    │   (三阶段课程跃迁)           │
                    └──────────┬──────────────────┘
                               │ 阶段配置
                    ┌──────────▼──────────────────┐
                    │     RolloutCollector         │
                    │   (并行轨迹采集)             │
                    └──────────┬──────────────────┘
                               │ 观测/动作/奖励
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────┐  ┌───────▼──────┐  ┌──────▼───────┐
    │  VecDoorEnv  │  │    Actor     │  │   Critic     │
    │ (N并行环境)  │  │ (GRU+分支)   │  │ (非对称MLP)  │
    └─────────┬────┘  └──────────────┘  └──────────────┘
              │
    ┌─────────▼────────────────────────┐
    │     DoorInteractionEnv           │
    │  SceneFactory + ContactMonitor   │
    │  TaskManager + RewardManager     │
    └──────────────────────────────────┘
              │
    ┌─────────▼────────────────────────┐
    │     Isaac Lab SimulationContext  │
    │        (GPU 物理仿真)            │
    └──────────────────────────────────┘
```

---

## License

[MIT](LICENSE)
