# Affordance-Guided Interaction

基于 **Isaac Lab** 与 **PPO** 训练双臂移动底盘机器人推开无把手门并穿门。79 维观测、15 维 raw Gaussian action、单 teacher PPO 策略。

---

## 项目结构

```
Affordance-Guided-Interaction/
├── scripts/
│   ├── train.py                    # 训练入口
│   ├── load_scene.py               # 场景加载/可视化
│   └── render_policy_rollouts.py   # 策略推理回放
├── configs/
│   ├── training/default.yaml       # 训练 profile、环境数、总步数
│   ├── env/default.yaml            # 物理步长、PD 增益、底盘参数
│   ├── task/default.yaml           # 门角阈值、成功条件
│   ├── reward/default.yaml         # 奖励权重与缩放
│   └── inference/default.yaml      # 推理回放配置
├── src/affordance_guided_interaction/
│   ├── envs/
│   │   ├── door_push_env.py        # DirectRLEnv 核心（观测/奖励/终止/reset）
│   │   ├── door_push_env_cfg.py    # 场景与环境参数 configclass
│   │   ├── door_reward_math.py     # 各 reward 项的 tensor 计算
│   │   ├── joint_target_math.py    # torque-proxy 关节目标映射
│   │   ├── base_control_math.py    # 底盘速度命令映射（planar/wheel/force）
│   │   ├── batch_math.py           # 批量四元数/坐标变换/基座采样
│   │   ├── physx_mass_ops.py       # 门质量随机化写入 PhysX
│   │   ├── gripper_hold.py         # Gripper 闭合保持
│   │   └── doorway_geometry.py     # 门洞几何常量
│   ├── tasks/door_push_direct/
│   │   ├── __init__.py             # Gym task 注册
│   │   └── agents/
│   │       └── rl_games_ppo_cfg.yaml  # rl_games PPO agent 配置
│   └── utils/
│       ├── rl_games_observer.py    # TensorBoard 自定义 observer
│       ├── rl_games_config.py      # rl_games 配置构建
│       ├── runtime_env.py          # 运行环境检测
│       └── train_runtime_config.py # 训练配置解析与 profile 选择
├── assets/
│   ├── robot/usd/                  # 机器人 USD 资产
│   └── minimal_push_door/          # 门板与侧墙 USD 资产
├── tests/                          # 单元测试
├── docs/
│   └── training_pipeline_detailed.md  # 训练流水线数学详解
└── RoboDuet/                       # 参考论文代码（独立模块）
```

---

## 快速开始

### 依赖

- Isaac Lab >= 1.0（含 Isaac Sim 4.x）
- Python >= 3.10
- PyTorch >= 2.0

### 训练

```bash
python scripts/train.py
```

默认使用 `env_256` profile（256 并行环境）。可在 `configs/training/default.yaml` 最后一行切换 profile。

### 场景可视化(还没维护)

```bash
python scripts/load_scene.py
```

### 策略推理回放(还没维护)

```bash
python scripts/render_policy_rollouts.py
```

---

## 训练监控

```bash
tensorboard --logdir runs/
```

核心指标：

| 分类 | Tag | 含义 |
| --- | --- | --- |
| 奖励 | `reward/total` | episode 总奖励均值 |
| | `reward/opening` | 开门奖励 |
| | `reward/passing` | 穿门奖励 |
| | `reward/shaping` | 正则化奖励 |
| 成功率 | `success/rate` | 成功穿门比例 |
| | `success/opened_enough_rate` | 门角 >= 30° 比例 |
| 诊断 | `task/door_angle_mean` | done env 门角均值 |
| | `task/base_cross_progress` | 底盘穿门进度 |
| | `losses/a_loss` | PPO actor loss |
| | `losses/c_loss` | PPO critic loss |
| | `losses/entropy` | 策略熵 |

---

## 配置体系

所有参数通过 `configs/` 下的 YAML 文件管理，修改后无需改动代码。

### 训练 profile（`configs/training/default.yaml`）

最后一行控制生效的 profile：

```yaml
<<: [*runtime_defaults, *profile_env256]   # 修改此处切换 profile
```

可选 profile：

| Profile | `num_envs` | `total_steps` | 适用硬件 |
| --- | --- | --- | --- |
| `env_256` | 256 | 5 亿 | RTX 4060 / debug |
| `env_512` | 512 | 4 亿 | RTX 4070 |
| `env_1024` | 1024 | 3 亿 | RTX 4090 |
| `env_2048` | 2048 | 2 亿 | RTX 4090 |
| `env_4096` | 4096 | 3 亿 | A100 |
| `env_6144` | 6144 | 30 亿 | A100 x2 |

运行时参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `headless` | `true` | 无窗口模式 |
| `device` | `cuda:0` | 训练 GPU |
| `seed` | `42` | 随机种子 |
| `log_dir` | `runs` | 日志目录 |

PPO 超参数（`ppo_common` 段）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `gamma` | `0.99` | 折扣因子 |
| `lam` | `0.95` | GAE λ |
| `clip_eps` | `0.2` | PPO clip 系数 |
| `entropy_coef` | `0.01` | 熵正则系数 |
| `actor_lr` | `1e-3` | 学习率 |
| `num_epochs` | `5` | 每 rollout 更新轮数 |
| `num_mini_batches` | `4` | mini-batch 数 |

### 环境参数（`configs/env/default.yaml`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `physics_dt` | `0.008333` | 物理步长 (~120 Hz) |
| `decimation` | `2` | 控制频率 = 60 Hz |
| `arm_action_scale_rad` | `0.25` | raw action → 关节偏移缩放 |
| `arm_pd_stiffness` | `50.0` | 手臂 PD 刚度 |
| `arm_pd_damping` | `4.5` | 手臂 PD 阻尼 |
| `base_max_lin_vel_x/y` | `0.5` | 底盘最大线速度 |
| `base_max_ang_vel_z` | `1.0` | 底盘最大角速度 |

### 奖励权重（`configs/reward/default.yaml`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `opening.scale` | `3.0` | 开门奖励缩放 |
| `opening.theta_hat` | `75.0` | 开门目标角 (°) |
| `opening.theta_pass` | `70.0` | 进入 passing 阶段的角度 (°) |
| `shaping.w_min_arm_motion` | `0.3` | 最小手臂运动惩罚权重 |
| `shaping.w_stretched_arm` | `1.0` | 手臂过伸惩罚权重 |
| `shaping.w_end_effector_to_panel` | `1.0` | 末端靠近门板奖励权重 |
| `shaping.w_command_limit` | `0.1` | raw action 过大惩罚权重 |
| `shaping.w_collision` | `2.0` | 硬碰撞惩罚权重 |

### 任务阈值（`configs/task/default.yaml`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `theta_open` | `0.5236` (π/6) | opened_enough 判定角 |
| `theta_pass` | `1.2217` (70°) | passing stage 切换角 |
| `theta_hat` | `1.3090` (75°) | 开门奖励目标角 |
| `reverse_angle_limit` | `-0.05` | 反向开门阈值 |

### 网络结构（`rl_games_ppo_cfg.yaml`）

位于 `src/.../agents/rl_games_ppo_cfg.yaml`，控制策略网络：

| 参数 | 默认值 |
| --- | --- |
| MLP 层 | `[512, 256, 128]` |
| 激活函数 | `elu` |
| `fixed_sigma` | `True` |
| `sigma_init` | `-2.0` |

---

## 域随机化

每个 episode reset 时自动采样：

| 参数 | 范围 | 说明 |
| --- | --- | --- |
| 门板质量 | $U[15, 75]$ kg | 写入 PhysX |
| 铰链阻力 | $U[0, 30]$ Nm，20% 概率置零 | 静态摩擦扭矩 |
| 空气阻尼 | $U[0, 4]$ Nm·s² | 二次阻尼 |
| Closer 阻尼 | $\alpha \cdot \tau_{res}$，$\alpha \sim U[1.5, 3]$，40% 概率置零 | 线性阻尼 |
| 基座距离 | $U[1, 2]$ m | 距门距离 |
| 基座横向偏移 | $U[-2, 2]$ m | 门洞法向偏移 |
| 基座初始 yaw | $U[-\pi, \pi]$ | 相对门洞方向 |
| 基座初始速度 | $U[-0.5, 0.5]$ m/s | xy 方向 |

随机化参数在 `DoorPushEnvCfg` 中配置（`door_push_env_cfg.py`）。

---

## 技术参考

本项目的实现参考了三个来源，分别对应训练管线骨架、奖励设计、PPO 训练框架：

### 训练管线骨架 — Isaac Lab 官方示例

`scripts/train.py` 的整体结构沿袭自 Isaac Lab 官方训练脚本 `IsaacLab/scripts/reinforcement_learning/rl_games/train.py`：

- **配置加载**：YAML 多文件合并 → task registry 查找 env/agent cfg → `AppLauncher` 初始化
- **环境创建**：`gym.make(task_name, cfg=env_cfg)` → `RlGamesVecEnvWrapper` 适配 rl_games 接口
- **训练循环**：由 rl_games `Runner` 驱动，环境侧为 Isaac Lab `DirectRLEnv`

主要差异在于我们把官方脚本中的硬编码路径改为了 `configs/` 下可配置的 profile 体系，并加入了自定义 TensorBoard observer。

### 奖励与任务设计 — RoboDuet 论文 (2409.04882)

奖励的 opening/passing/shaping 三阶段结构和域随机化参数范围直接来自论文 *RoboDuet* 的 Appendix A/B：

- **Opening reward** $r_{od}$：$1 - |\theta - \hat\theta|/\hat\theta$，目标角 75°，缩放 3.0
- **Passing reward** $r_p$：底盘速度在 progress 方向的投影 / 最大速度，clip [0, 1]
- **Stage 切换**：门角 > 70° 从 opening 进入 passing，opening reward 取满值
- **域随机化**：门质量 [15,75] kg、铰链阻力 [0,30] Nm（20% 置零）、空气阻尼 [0,4]、closer 阻尼 = α·τ_res（α~[1.5,3]，40% 置零）、基座距离 [1,2]m、横向偏移 [-2,2]m、yaw [-π,π]

与论文的不同之处：
- 论文包含门把手操作（handle grasp/turn）、学生策略蒸馏、RNN student、多门类型估计，我们当前任务是 **handle-free push door**，不涉及这些
- 论文随机化手臂 PD 增益（$K_p \sim U[40,60]$, $K_d \sim U[3,6]$），我们当前使用固定值
- 论文有门几何尺寸随机化，我们使用单一门模型

`RoboDuet/` 目录下保留了论文的原始实现供参考。

### PPO 训练框架 — rl_games

PPO 算法本身使用 [rl_games](https://github.com/Denys88/rl_games) 库，与 RoboDuet 论文代码的选择一致：

- Actor-Critic 共享 backbone（`network.separate: False`），MLP [512, 256, 128]
- Clipped surrogate objective + clipped value loss + entropy bonus
- 自适应学习率（基于 KL 散度阈值 0.01）
- Mixed precision（bfloat16）

RoboDuet 的 PPO 实现（`RoboDuet/go1_gym_learn/ppo_cse_unified/ppo.py`）是独立手写的双头 PPO（dog/arm 分离 value head + β 交叉 advantage），而我们的项目直接使用 rl_games 库的单头 PPO，配置在 `src/.../agents/rl_games_ppo_cfg.yaml` 中。

---

## 详细文档

训练流水线的完整数学建模（MDP 定义、动作映射、观测构造、奖励公式、PPO loss、GAE、域随机化）见 [`docs/training_pipeline_detailed.md`](docs/training_pipeline_detailed.md)。

## License

[MIT](LICENSE)
