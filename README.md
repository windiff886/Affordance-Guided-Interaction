# Affordance-Guided Interaction

基于 **Isaac Lab** 仿真器与 **PPO** 强化学习算法，训练双臂机器人（Unitree Z1 × 2 + Dingo 底座）在持杯约束下完成推门任务。策略通过仿真 ground truth 计算的 `door_geometry`（6 维门几何信号）作为唯一门相关输入，并在当前的三阶段 push-only 课程中逐步引入持杯约束。

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

训练入口现在固定为：

```bash
python scripts/train.py
```

所有训练运行参数统一在 [configs/training/default.yaml](configs/training/default.yaml) 中调整，包括：

- `headless`
- `device`
- `seed`
- `resume`
- `log_dir`
- `ckpt_dir`
- `num_envs`
- `total_steps`

### 本地验证（4060 Laptop）

将 [configs/training/default.yaml](configs/training/default.yaml) 中的当前生效值切到文档内注释给出的 `4060 Laptop` 建议配置后运行：

```bash
python scripts/train.py
```

> 预期行为：控制台每轮迭代打印 `actor_loss / critic_loss / entropy / reward / stage`，无报错即管线正常。

### 服务器训练（A100）

将 [configs/training/default.yaml](configs/training/default.yaml) 中的当前生效值切到文档内注释给出的 `A100` 建议配置后运行：

```bash
python scripts/train.py
```

### 推荐环境数配置

| 硬件 | YAML 中的 `num_envs` 建议值 | 用途 |
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
└── curriculum/default.yaml  # 课程跃迁窗口与阈值
```

> 奖励权重已内联至 `DoorPushEnvCfg`（`door_push_env_cfg.py`），数学设计文档见 [`envs/Reward.md`](src/affordance_guided_interaction/envs/Reward.md)。

修改配置后无需改动代码，`train.py` 会自动合并所有配置文件，并直接读取训练 YAML 中的运行时参数。

### 关键超参速查

```yaml
# configs/training/default.yaml
headless: false            # 是否无窗口运行
num_envs: 2                # 并行环境数（当前生效值）
total_steps: 50_000        # 总训练步数（当前生效值）
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
| Stage 1 | push | 0% | 基础推门接触 |
| Stage 2 | push | 100% | 力控与稳定性约束 |
| Stage 3 | push | 50% | 有杯 / 无杯混合场景下的稳定推门 |

> 当前课程只针对 push 任务调度训练难度；阶段变化只影响持杯采样概率，门类型保持为 `push`。

---

## 项目结构

```
Affordance-Guided-Interaction/
├── scripts/
│   └── train.py                  # rl_games 训练入口
├── src/affordance_guided_interaction/
│   ├── envs/                     # DoorPush DirectRLEnv 及配置
│   ├── tasks/                    # Gym task registry + rl_games agent 配置
│   └── utils/
│       ├── runtime_env.py        # 运行环境与 headless 解析
│       └── train_runtime_config.py # 训练运行时配置解析
├── src/teleop_cup_grasp/         # 杯体抓取遥操作（独立模块）
├── configs/                      # 训练链路 YAML 配置
├── assets/                       # USD 仿真资产
├── model/                        # 预训练权重
├── docs/                         # 当前流程文档
```
└── checkpoints/                  # 训练 checkpoint（自动生成）
```

---

## 技术架构

```
             ┌──────────────────────────────────────────────────────┐
             │  train.py                                           │
             │  配置加载 → 环境创建 → 采集-优化-课程循环            │
             └────────────────────┬─────────────────────────────────┘
                                  │
             ┌────────────────────▼─────────────────────────────────┐
             │  DirectRLEnvAdapter                                  │
             │  DoorPushEnv (tensor) → VecEnvProtocol (list[dict]) │
             │  训练管线桥接层                                      │
             └────────────────────┬─────────────────────────────────┘
                                  │
             ┌────────────────────▼─────────────────────────────────┐
             │  DoorPushEnv  (DirectRLEnv)                         │
             │  ┌──────────────────────────────────────────────┐   │
             │  │ 自包含 GPU 批量环境                           │   │
             │  │                                              │   │
             │  │  • _get_observations()  非对称 Actor/Critic  │   │
             │  │  • _get_rewards()       12 项 tensor 奖励    │   │
             │  │  • _get_dones()         终止判定              │   │
             │  │  • _reset_idx()         批量杯体抓取初始化    │   │
             │  └──────────────────────────────────────────────┘   │
             └────────────────────┬─────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼──────────┐ ┌─────────▼──────────┐ ┌──────────▼─────────┐
│ DoorPushEnvCfg     │ │ DoorPushSceneCfg   │ │ Isaac Lab Cloner   │
│ 环境参数+奖励权重  │ │ 声明式场景定义     │ │ 自动 N 环境复制    │
│ (configclass)      │ │ Robot/Door/Cup/    │ │ GPU 批量仿真       │
│                    │ │ ContactSensor      │ │                    │
└────────────────────┘ └────────────────────┘ └────────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
             ┌────────────────────▼─────────────────────────────────┐
             │  Isaac Lab SimulationContext + PhysX GPU             │
             │  GPU 物理仿真 (ArticulationView / RigidBodyView)    │
             └──────────────────────────────────────────────────────┘
```

### 核心设计

- **无 Python 循环**：所有 per-env 状态为 `(num_envs, ...)` 形状的 torch tensor，观测 / 奖励 / 终止判定均为纯 tensor 操作
- **Cloner 自动复制**：`DoorPushSceneCfg` 声明式定义场景（机器人、门、杯体、接触传感器），Isaac Lab Cloner 自动为 N 个并行环境复制完整场景子树
- **自包含环境**：`DoorPushEnv` 内置 12 项奖励计算（任务进展 + 稳定性约束 + 安全惩罚）、非对称 Actor/Critic 观测构建、批量杯体抓取初始化，无需外部 Manager
- **非对称观测**：Actor 含传感器噪声 + door_geometry(6D)；Critic 含无噪声物理状态 + 门关节角/速度/质量等 privileged 信息
- **适配器桥接**：`DirectRLEnvAdapter` 将 tensor 接口转换为训练管线 (`RolloutCollector`) 期望的 `VecEnvProtocol`

---

## License

[MIT](LICENSE)
