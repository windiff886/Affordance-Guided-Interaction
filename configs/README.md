# configs/ — 配置参数总览与审计报告

> **审计日期**：2026-04-09
>
> 本文档对 7 个 YAML 配置文件的全部参数逐一追溯到源代码消费点，说明其含义与作用。
> 旧路径（VecDoorEnv）已删除，仅保留 GPU 批量并行路径（DoorPushEnv + DirectRLEnvAdapter）。
> 训练入口默认会加载 `reward/default.yaml`，并通过 `train.py` 的 `_inject_reward_params()` 将奖励超参数覆盖写入 `DoorPushEnvCfg`；`DoorPushEnvCfg` 中保留同名默认值作为回退。
> 可视化入口从 `visualization/default.yaml` 读取运行参数，生成 MP4 视频和/或逐帧图片。

---

## 1. 配置文件清单

| 文件 | 参数数 | 消费率 |
|------|--------|-------|
| `training/default.yaml` | 28 | 100% |
| `env/default.yaml` | 6 | 100% |
| `policy/default.yaml` | 8 | 100% |
| `curriculum/default.yaml` | 3 | 100% |
| `task/default.yaml` | 2 | 100% |
| `reward/default.yaml` | 25 | 100% |
| `visualization/default.yaml` | 14 | 100% |

**整体**：86 个参数，全部被代码消费。

> **已恢复**: `reward/default.yaml`（25 参数）— 奖励超参数从 `DoorPushEnvCfg` 迁移到独立 YAML 配置文件，由 `train.py` 的 `_inject_reward_params()` 注入。
> 数学公式参考见 `envs/Reward.md`。
> 当前 `task` 奖励侧新增 `w_approach`、`approach_eps`、`approach_stop_angle` 三个键，用于“接近门板大表面”的 shaping。

---

## 2. 配置加载机制

`scripts/train.py` 中的 `load_config()` 从 `configs/` 目录加载 7 个 YAML 文件，返回字典：

```python
cfg = {
    "training":      ...,  # configs/training/default.yaml
    "env":           ...,  # configs/env/default.yaml
    "policy":        ...,  # configs/policy/default.yaml
    "task":          ...,  # configs/task/default.yaml
    "curriculum":    ...,  # configs/curriculum/default.yaml
    "reward":        ...,  # configs/reward/default.yaml
    "visualization": ...,  # configs/visualization/default.yaml
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
| `ckpt_dir` | `checkpoints` | `train_runtime_config.py` → `train.py` | checkpoint 根目录；单次训练实际输出到 `checkpoints/checkpoints_<timestamp>/` |

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

训练会沿用 `runs/ppo_<timestamp>/` 的命名方式，同时把 checkpoint 统一写到 `checkpoints/checkpoints_<timestamp>/`。除固定间隔 checkpoint 外，课程从 `stage_1 -> stage_2`、`stage_2 -> stage_3` 跃迁时会额外保存 `ckpt_stage_<new_stage_name>.pt`。

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `log_interval` | `1` | `train.py:514` | 每多少轮迭代打印一次控制台日志 |
| `checkpoint_interval` | `50` | `train.py` | 每多少轮迭代保存一次中间 checkpoint；stage 跃迁时还会额外保存 `ckpt_stage_<new_stage_name>.pt` |

#### 调试选项

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `debug.strict_mode` | `true` | `train.py`（物理引擎检查） | 启用后，缺少 Isaac Lab 时直接报错退出，而非静默退化为 placeholder |

> **已移除**：`debug.visualize_detections`（视觉调试开关）和 `debug.strict_mode` 中的视觉管线检查部分。当前默认训练不使用视觉感知，这些配置项已清理。

---

### 3.2 `env/default.yaml` — 环境配置（6 参数）

旧路径的环境参数（`joints_per_arm`、`total_joints`、`contact_force_threshold` 等）已删除。
其他环境参数现在定义在 `DoorPushEnvCfg`（`door_push_env_cfg.py`）中。

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `physics_dt` | `0.008333` | `train.py` → `DoorPushEnvCfg.sim.dt` | 物理引擎步长，$\Delta t = 1/120$ 秒 |
| `decimation` | `2` | `train.py` → `DoorPushEnvCfg.decimation` | 每个策略步执行多少次物理步进。控制频率 = $1 / (\Delta t \times \text{decimation}) = 60\text{Hz}$ |
| `control.action_type` | `joint_position` | `train.py` → `env_cfg.control_action_type` | 默认动作语义：双臂 12 维关节位置目标（rad） |
| `control.arm_pd_stiffness` | `1000.0` | `train.py` → `scene.robot.actuators[*].stiffness` | arm actuator 的 PD 刚度 |
| `control.arm_pd_damping` | `100.0` | `train.py` → `scene.robot.actuators[*].damping` | arm actuator 的 PD 阻尼 |
| `control.position_target_noise_std` | `0.0` | `train.py` → `env_cfg.position_target_noise_std` | 位置目标噪声标准差 |

---

### 3.3 `policy/default.yaml` — 策略网络配置（8 参数）

分支 encoder 的维度参数（`proprio_hidden`、`ee_hidden` 等）由 `ActorConfig` dataclass 默认值控制，不在 YAML 中暴露。如需修改这些维度，直接编辑 `actor.py` 中的 `ActorConfig` dataclass。

#### Actor 网络

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `actor.rnn_hidden` | `512` | `train.py:115` → `ActorConfig` → `RecurrentBackbone` | GRU 隐状态维度 |
| `actor.rnn_layers` | `1` | `train.py:116` → `ActorConfig` → `RecurrentBackbone` | GRU 层数 |
| `actor.rnn_type` | `gru` | `train.py:117` → `ActorConfig` → `RecurrentBackbone` | 循环单元类型（`gru` 或 `lstm`） |
| `actor.action_dim` | `12` | `train.py:118` → `ActorConfig` → `ActionHead` | 输出动作维度（双臂 12 个关节位置目标 rad） |
| `actor.log_std_init` | `-0.5` | `train.py:119` → `ActorConfig` → `ActionHead` | 高斯策略初始 $\log\sigma$（对应 $\sigma \approx 0.61$） |

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
| Stage 1 | `none: 1.0` | push | 无持杯条件下学会基础推门接触 |
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

默认路径的 episode 级随机化范围保留在 `DoorPushEnvCfg` 中；步级位置目标噪声由 `configs/env/default.yaml` 暴露；Actor 观测噪声由 `DoorPushEnvCfg.obs_noise_std` 控制。

| 参数 | 位置 | 默认值 | 时间尺度 | 含义 |
|------|------|--------|---------|------|
| `cup_mass_range` | `DoorPushEnvCfg` | `(0.1, 0.8)` kg | 回合级 | 杯体质量均匀分布范围 |
| `door_mass_range` | `DoorPushEnvCfg` | `(5.0, 20.0)` kg | 回合级 | 门板质量均匀分布范围 |
| `door_damping_range` | `DoorPushEnvCfg` | `(0.5, 5.0)` N·m·s/rad | 回合级 | 门铰链阻尼均匀分布范围 |
| `push_plate_center_xy` | `DoorPushEnvCfg` | `(2.98, 0.27)` m | 回合级 | 推板中心的世界坐标投影，作为基座采样圆心 |
| `base_reference_xy` | `DoorPushEnvCfg` | `(3.72, 0.27)` m | 回合级 | 门外侧参考基座点，用于确定采样扇区中心方向 |
| `base_height` | `DoorPushEnvCfg` | `0.12` m | 回合级 | 机器人 root 的固定高度 |
| `base_radius_range` | `DoorPushEnvCfg` | `(0.45, 0.60)` m | 回合级 | 基座到推板中心的允许半径范围 |
| `base_sector_half_angle_deg` | `DoorPushEnvCfg` | `20.0°` | 回合级 | 门外侧扇形环的半角 |
| `base_yaw_delta_deg` | `DoorPushEnvCfg` | `10.0°` | 回合级 | 名义朝向推板中心后的 yaw 扰动范围 |
| `control.position_target_noise_std` | `env/default.yaml` | `0.0` | 步级 | 位置目标噪声标准差 |
| `obs_noise_std` | `DoorPushEnvCfg` | `0.01` | 步级 | Actor 观测噪声标准差（仅注入 q/dq） |

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

---

### 3.7 `visualization/default.yaml` — Rollout 可视化配置（16 参数）

> **已新增**: rollout 可视化配置，用于 `scripts/rollout_demo.py` 入口脚本。
> 所有可视化运行参数统一从本文件读取，不再依赖命令行覆盖。

#### 运行时参数

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `checkpoint` | `null` | `rollout_config.py` → `rollout_demo.py` | 模型 checkpoint 路径（相对项目根）；`null` 使用随机初始化策略 |
| `device` | `cpu` | `rollout_config.py` → `rollout_demo.py` | 推理设备 |
| `seed` | `42` | `rollout_config.py` → `rollout_demo.py` | 随机种子 |
| `headless` | `true` | `rollout_config.py` → `rollout_demo.py` | 是否以无窗口模式启动 Isaac Sim |
| `deterministic` | `true` | `rollout_config.py` → `rollout_demo.py` | 确定性/采样动作模式 |

#### 上下文与回合

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `contexts` | `[none, left_only, right_only, both]` | `rollout_config.py` → `rollout_demo.py` | 要渲染的杯子占用上下文列表 |
| `episodes_per_context` | `1` | `rollout_config.py` → `rollout_demo.py` | 每个上下文运行的 episode 数 |

#### 帧捕获与视频

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `save_video` | `true` | `rollout_config.py` → `rollout_artifacts.py` | 是否生成 MP4 视频（需要 imageio[ffmpeg]） |
| `save_frames` | `false` | `rollout_config.py` → `rollout_artifacts.py` | 是否保存逐帧图片 |
| `frame_stride` | `1` | `rollout_config.py` → `rollout_artifacts.py` | 每隔 N 步捕获一帧 |
| `video_fps` | `30` | `rollout_config.py` → `rollout_artifacts.py` | 输出视频帧率 |
| `viewer_eye` | `[6.0, 3.0, 3.2]` | `rollout_config.py` → `rollout_demo.py` | rollout viewer 相机位置 |
| `viewer_lookat` | `[3.0, 0.3, 1.0]` | `rollout_config.py` → `rollout_demo.py` | rollout viewer 相机目标点 |

#### Artifact 输出

| YAML 键 | 默认值 | 消费位置 | 含义 |
|---------|--------|---------|------|
| `output_root` | `artifacts/vis` | `rollout_config.py` → `rollout_artifacts.py` | artifact 输出根目录（相对项目根） |
| `video_name_template` | `{checkpoint_stem}/{context}.mp4` | `rollout_artifacts.py` | 视频文件路径模板 |
| `frames_dir_template` | `{checkpoint_stem}/{context}_frames` | `rollout_artifacts.py` | 帧目录路径模板 |

#### Artifact 输出结构

```
artifacts/vis/
  iter_010000/           ← checkpoint_stem
    none.mp4             ← 单个 context 的视频
    both.mp4
    none_frames/         ← 单个 context 的帧目录
      ep000_step00000.png
      ep000_step00001.png
    both_frames/
      ...
```
