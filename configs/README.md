# configs/ — 配置参数总览与审计报告

> **审计日期**：2026-04-06
>
> 本文档对 6 个 YAML 配置文件的全部参数逐一追溯到源代码消费点，说明其含义与作用。
> 旧路径（VecDoorEnv）已删除，仅保留 GPU 批量并行路径（DoorPushEnv + DirectRLEnvAdapter）。
> 训练入口默认会加载 `reward/default.yaml`，并通过 `train.py` 的 `_inject_reward_params()` 将奖励超参数覆盖写入 `DoorPushEnvCfg`；`DoorPushEnvCfg` 中保留同名默认值作为回退。

---

## 1. 配置文件清单

| 文件 | 参数数 | 消费率 |
|------|--------|-------|
| `training/default.yaml` | 28 | 100% |
| `env/default.yaml` | 2 | 100% |
| `policy/default.yaml` | 8 | 100% |
| `curriculum/default.yaml` | 3 | 100% |
| `task/default.yaml` | 2 | 100% |
| `reward/default.yaml` | 22 | 100% |

**整体**：65 个参数，全部被代码消费。

> **已恢复**: `reward/default.yaml`（22 参数）— 奖励超参数从 `DoorPushEnvCfg` 迁移到独立 YAML 配置文件，由 `train.py` 的 `_inject_reward_params()` 注入。
> 数学公式参考见 `envs/Reward.md`。

---

## 2. 配置加载机制

`scripts/train.py` 中的 `load_config()` 从 `configs/` 目录加载 6 个 YAML 文件，返回字典：

```python
cfg = {
    "training":   ...,  # configs/training/default.yaml
    "env":        ...,  # configs/env/default.yaml
    "policy":     ...,  # configs/policy/default.yaml
    "task":       ...,  # configs/task/default.yaml
    "curriculum": ...,  # configs/curriculum/default.yaml
    "reward":     ...,  # configs/reward/default.yaml
}
```

其中 `training/default.yaml` 的运行时字段（`headless`、`device`、`seed`、`resume`、`log_dir`、`ckpt_dir`）由 `resolve_train_runtime_config()` 单独提取为 `TrainRuntimeConfig` dataclass。其余字段通过 `cfg["training"].get(...)` 在 `train.py` 主函数中直接读取。

---

## 3. 各文件参数详细审计

### 3.1 `training/default.yaml` — 训练管线配置（28 参数）

#### 运行时参数

这组参数由 `resolve_train_runtime_config()` 提取，控制训练的运行环境。

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `headless` | `false` | `train_runtime_config.py` → `train.py:332` | 是否以无窗口模式启动 Isaac Sim 仿真 |
| `device` | `null` | `train_runtime_config.py` → `train.py:304` | 计算设备选择；`null` 表示自动检测 CUDA |
| `seed` | `42` | `train_runtime_config.py` → `train.py:313` | 全局随机种子，设置 PyTorch 和 NumPy |
| `resume` | `null` | `train_runtime_config.py` → `train.py:417` | checkpoint 恢复路径；`null` 表示从头训练 |
| `log_dir` | `runs` | `train_runtime_config.py` → `train.py:406` | TensorBoard 日志输出目录 |
| `ckpt_dir` | `checkpoints` | `train_runtime_config.py` → `train.py:415` | checkpoint 文件保存目录 |

#### 基本训练参数

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `num_envs` | `2` | `train.py` → `DoorPushEnvCfg.scene.num_envs` | 并行环境数量 $N$ |
| `total_steps` | `50_000` | `train.py:320` → 主循环终止条件 | 训练总环境交互步数上限 |
| `n_steps_per_rollout` | `128` | `train.py:321` → `RolloutCollector` | 每轮 rollout 中每个环境采集的步数 $T$ |

#### PPO 算法超参

| YAML 键 | 默认值 | 消费位置 | 数学符号 | 含义 |
|---------|--------|---------|---------|------|
| `ppo.gamma` | `0.99` | `PPOConfig` → `RolloutBuffer.compute_gae()` | $\gamma$ | 折扣因子 |
| `ppo.lam` | `0.95` | `PPOConfig` → `RolloutBuffer.compute_gae()` | $\lambda$ | GAE 偏差-方差权衡系数 |
| `ppo.clip_eps` | `0.2` | `PPOConfig` → `PPOTrainer.update()` | $\epsilon$ | PPO-Clip 策略比率裁剪范围 |
| `ppo.value_clip_eps` | `0.2` | `PPOConfig` → `PPOTrainer.update()` | $\epsilon_v$ | Value function 裁剪范围 |
| `ppo.use_clipped_value_loss` | `true` | `PPOConfig` → `PPOTrainer.update()` | — | 是否对 Critic 损失启用 clipped value loss |
| `ppo.entropy_coef` | `0.01` | `PPOConfig` → `PPOTrainer.update()` | $c_e$ | 策略熵正则化系数 |
| `ppo.value_coef` | `0.5` | `PPOConfig` → `PPOTrainer.update()` | $c_v$ | Critic 损失权重 |
| `ppo.max_grad_norm` | `1.0` | `PPOConfig` → `PPOTrainer.update()` | $g_{\max}$ | 全局梯度范数裁剪上限 |
| `ppo.actor_lr` | `3.0e-4` | `PPOConfig` → Actor Adam 优化器 | $\alpha_\theta$ | Actor 学习率 |
| `ppo.critic_lr` | `3.0e-4` | `PPOConfig` → Critic Adam 优化器 | $\alpha_\phi$ | Critic 学习率 |
| `ppo.num_mini_batches` | `4` | `PPOConfig` → `RolloutBuffer` 切分 | $N_{\text{mb}}$ | Mini-batch 数量 |
| `ppo.num_epochs` | `5` | `PPOConfig` → `PPOTrainer.update()` 外层循环 | $K$ | 每轮数据的重复优化次数 |
| `ppo.seq_length` | `16` | `PPOConfig` → `RolloutBuffer` TBPTT 切分 | $L$ | 截断反向传播的序列长度 |
| `ppo.normalize_advantages` | `true` | `PPOConfig` → `PPOTrainer.update()` | — | 是否对每个 mini-batch 的优势值做零均值单位方差标准化 |

#### 日志与 checkpoint

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `log_interval` | `1` | `train.py:514` | 每多少轮迭代打印一次控制台日志 |
| `checkpoint_interval` | `50` | `train.py:575` | 每多少轮迭代保存一次中间 checkpoint |

#### 调试选项

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `debug.visualize_detections` | `false` | `train.py` → `PerceptionRuntime` → `AffordancePipelineConfig` → `AffordancePipeline` | 启用后在每次视觉编码时弹出 OpenCV 窗口，叠加显示分割 mask、bbox 和点云回投影 |
| `debug.strict_mode` | `false` | `train.py`（物理引擎检查）、`PerceptionRuntime`（视觉管线检查） | 启用后，缺少 Isaac Lab 或视觉依赖时直接报错退出，而非静默退化为 placeholder。正式训练应设为 `true` |

---

### 3.2 `env/default.yaml` — 环境配置（2 参数）

旧路径的环境参数（`joints_per_arm`、`total_joints`、`contact_force_threshold` 等）已删除。
其他环境参数现在定义在 `DoorPushEnvCfg`（`door_push_env_cfg.py`）中。

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `physics_dt` | `0.008333` | `train.py` → `create_simulation_context()` | 物理引擎步长，$\Delta t = 1/120$ 秒 |
| `decimation` | `2` | `train.py` → `create_simulation_context()` | 每个策略步执行多少次物理步进。控制频率 = $1 / (\Delta t \times \text{decimation}) = 60\text{Hz}$ |

---

### 3.3 `policy/default.yaml` — 策略网络配置（8 参数）

分支 encoder 的维度参数（`proprio_hidden`、`ee_hidden` 等）由 `ActorConfig` dataclass 默认值控制，不在 YAML 中暴露。如需修改这些维度，直接编辑 `actor.py` 中的 `ActorConfig` dataclass。

#### Actor 网络

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `actor.rnn_hidden` | `512` | `train.py:115` → `ActorConfig` → `RecurrentBackbone` | GRU 隐状态维度 |
| `actor.rnn_layers` | `1` | `train.py:116` → `ActorConfig` → `RecurrentBackbone` | GRU 层数 |
| `actor.rnn_type` | `gru` | `train.py:117` → `ActorConfig` → `RecurrentBackbone` | 循环单元类型（`gru` 或 `lstm`） |
| `actor.action_dim` | `12` | `train.py:118` → `ActorConfig` → `ActionHead` | 输出动作维度（双臂 12 个关节力矩） |
| `actor.log_std_init` | `-0.5` | `train.py:119` → `ActorConfig` → `ActionHead` | 高斯策略初始 $\log\sigma$（对应 $\sigma \approx 0.61$） |
| `actor.include_torques` | `true` | `train.py:120` → `ActorConfig` → `Actor`、`Critic` | 是否在本体感觉输入中包含关节力矩。为 `true` 时 proprio 维度为 48，为 `false` 时为 36 |

#### Critic 网络

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `critic.hidden_dims` | `[512, 256, 128]` | `train.py:126` → `CriticConfig` → `Critic` MLP | Critic MLP 各层维度 |

注：Critic 使用与 Actor 相同结构的分支 encoder（独立权重），其维度由 `ActorConfig` 控制。

---

### 3.4 `curriculum/default.yaml` — 课程学习配置（3 参数）

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `initial_stage` | `stage_1` | `train.py:374` → `CurriculumManager.__init__()` | 训练的起始课程阶段 |
| `window_size` | `50` | `train.py:372` → `CurriculumManager.__init__()` | 滑动窗口长度，用于计算最近若干轮迭代的平均成功率 |
| `threshold` | `0.8` | `train.py:373` → `CurriculumManager.report_epoch()` | 课程跃迁阈值：滑动窗口平均成功率达到此值时跃迁到下一阶段 |

三阶段的具体定义（上下文分布、门类型）硬编码在 `curriculum_manager.py` 的 `STAGE_CONFIGS` 中：

| 阶段 | 上下文分布 | 门类型 | 学习目标 |
|------|-----------|-------|---------|
| Stage 1 | `none: 1.0` | push | 无持杯条件下学会基础视觉引导接触 |
| Stage 2 | `left_only: 0.5, right_only: 0.5` | push | 单臂持杯约束下学会稳定推门 |
| Stage 3 | `none/left_only/right_only/both` 各 `0.25` | push | 最终混合分布，统一覆盖所有持杯组合 |

---

### 3.5 `task/default.yaml` — 任务定义配置（2 参数）

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `door_angle_target` | `1.57` | `DoorPushEnvCfg.door_angle_target` | episode 成功终止的门角度阈值（rad，约 90°）。门角度达到此值时 episode 判定为成功并终止 |
| `cup_drop_threshold` | `0.15` | `DoorPushEnvCfg.cup_drop_threshold` | 杯体脱落判定距离阈值（m）。杯体偏离手部超过此距离即判定为脱落 |

---

## 4. 域随机化参数

域随机化参数当前硬编码在 `training/domain_randomizer.py` 的 `RandomizationConfig` dataclass 中，未外置到 YAML。

| 参数 | 符号 | 默认值 | 时间尺度 | 含义 |
|------|------|--------|---------|------|
| `cup_mass_range` | $m_{\text{cup}}$ | `(0.1, 0.8)` kg | 回合级 | 杯体质量均匀分布范围 |
| `door_mass_range` | $m_{\text{door}}$ | `(5.0, 20.0)` kg | 回合级 | 门板质量均匀分布范围 |
| `door_damping_range` | $d_{\text{hinge}}$ | `(0.5, 5.0)` N·m·s/rad | 回合级 | 门铰链阻尼均匀分布范围 |
| `push_plate_center_xy` | $c_{\text{push}}$ | `(2.98, 0.27)` m | 回合级 | 推板中心的世界坐标投影，作为基座采样圆心 |
| `base_reference_xy` | $p_{\text{ref}}$ | `(3.72, 0.27)` m | 回合级 | 门外侧参考基座点，用于确定采样扇区中心方向 |
| `base_height` | $z_{\text{base}}$ | `0.12` m | 回合级 | 机器人 root 的固定高度 |
| `base_radius_range` | $[r_{\min}, r_{\max}]$ | `(0.45, 0.60)` m | 回合级 | 基座到推板中心的允许半径范围 |
| `base_sector_half_angle_deg` | $\Delta \theta$ | `20.0°` | 回合级 | 门外侧扇形环的半角 |
| `base_yaw_delta_deg` | $\Delta \psi$ | `10.0°` | 回合级 | 名义朝向推板中心后的 yaw 扰动范围 |
| `action_noise_std` | $\sigma_a$ | `0.02` | 步级 | 动作噪声标准差 |
| `observation_noise_std` | $\sigma_o$ | `0.01` | 步级 | 观测噪声标准差（仅注入 Actor 的 proprio 分支） |

---

## 5. 配置文件间的一致性约束

以下参数在多个文件中出现或存在隐含依赖，修改时必须同步：

| 约束 | 相关参数 | 当前值 | 说明 |
|------|---------|--------|------|
| 门角度阈值区分 | `DoorPushEnvCfg.success_angle_threshold` 与 `DoorPushEnvCfg.door_angle_target` | `1.2` vs `1.57` | 前者是奖励 bonus 触发角度，后者是 episode 成功终止角度，两者故意不同 |

---

## 6. 关键数学公式与配置参数的对应

### 6.1 PPO 总损失

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{actor}}(\theta) + c_v \cdot \mathcal{L}_{\text{critic}}(\theta) - c_e \cdot \mathcal{H}[\pi_\theta]$$

- $c_v$ = `ppo.value_coef` = 0.5
- $c_e$ = `ppo.entropy_coef` = 0.01
- $\epsilon$ = `ppo.clip_eps` = 0.2（PPO-Clip 裁剪范围）

### 6.2 GAE 优势估计

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$$

- $\gamma$ = `ppo.gamma` = 0.99
- $\lambda$ = `ppo.lam` = 0.95

### 6.3 课程跃迁判据

$$\frac{1}{M} \sum_{e=E-M+1}^{E} \eta_e \geq \eta_{\text{thresh}}$$

- $M$ = `curriculum.window_size` = 50
- $\eta_{\text{thresh}}$ = `curriculum.threshold` = 0.8

### 6.4 持杯稳定性项的激活条件

稳定性奖励不再使用时间退火。当前总式中的稳定性项只由 occupancy mask 控制：

$$
r_{\text{stab}} = m_L \cdot \left(r_{\text{stab\_bonus}}^L + r_{\text{stab\_penalty}}^L\right) + m_R \cdot \left(r_{\text{stab\_bonus}}^R + r_{\text{stab\_penalty}}^R\right)
$$

- Stage 1: `m_L = m_R = 0`，稳定性项整体关闭
- Stage 2/3: 持杯侧稳定性项从 episode 起点开始全量生效
