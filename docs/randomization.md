# 项目中的随机化量与参数

这份文档只描述当前默认训练路径实际生效的随机化与噪声，不再沿用旧的 torque-action 说明。

主要对应：

- `src/affordance_guided_interaction/envs/door_push_env.py`
- `src/affordance_guided_interaction/envs/door_push_env_cfg.py`
- `scripts/train.py`
- `configs/env/default.yaml`
- `configs/curriculum/default.yaml`

## 1. 回合级随机化

这些量在每次 episode reset 时重新采样，并在该 episode 内保持不变。

### 1.1 杯体质量

`m_cup ~ Uniform(0.1, 0.8)`，单位 `kg`

### 1.2 门板质量

`m_door ~ Uniform(5.0, 20.0)`，单位 `kg`

### 1.3 门铰链阻尼

`d_hinge ~ Uniform(0.5, 5.0)`，单位 `N·m·s/rad`

### 1.4 基座位姿

基座在门外侧扇形环内采样：

- `base_radius_range = (0.45, 0.60)`
- `base_sector_half_angle_deg = 20.0`
- `base_yaw_delta_deg = 10.0`
- `base_height = 0.12`

采样逻辑位于 `sample_base_poses()` 与 `DoorPushEnv._reset_idx()`。

## 2. 步级噪声

### 2.1 位置目标噪声

当前动作侧噪声不是 torque noise，而是 **位置目标噪声**。

环境在 `_pre_physics_step()` 中按以下顺序执行：

`q_target_cmd = clip(q_target_raw, q_min, q_max)`

`q_target_cmd = clip(q_target_cmd + epsilon_q, q_min, q_max)`

其中：

`epsilon_q ~ Normal(0, sigma_q^2 I)`

默认值来自 `configs/env/default.yaml`：

- `control.position_target_noise_std = 0.0`

### 2.2 观测噪声

观测噪声只注入 Actor 观测中的：

- `joint_positions`
- `joint_velocities`

即 `proprio` 前 24 维。critic 观测始终使用无噪声真值。

定义为：

`epsilon_o ~ Normal(0, sigma_o^2 I)`

当前默认值：

- `obs_noise_std = 0.01`

该值目前保留在 `DoorPushEnvCfg` 中。

### 2.3 旧版 `action_noise_std`

`DoorPushEnvCfg` 和 `training/domain_randomizer.py` 中仍保留 `action_noise_std` 这个历史字段，但它已经不驱动默认训练主线。当前默认动作侧噪声入口应以 `control.position_target_noise_std` 为准。

## 3. 课程中的上下文随机化

训练课程在 episode 级别采样离散上下文：

`c_episode ∈ {none, left_only, right_only, both}`

并映射为：

- `none -> (0, 0)`
- `left_only -> (1, 0)`
- `right_only -> (0, 1)`
- `both -> (1, 1)`

当前默认三阶段分布为：

| 阶段 | 上下文分布 |
|---|---|
| Stage 1 | `none: 1.0` |
| Stage 2 | `left_only: 0.5, right_only: 0.5` |
| Stage 3 | `none: 0.25, left_only: 0.25, right_only: 0.25, both: 0.25` |

## 4. 门类型

训练接口允许门类型分布存在，但当前默认配置下始终只有：

`g = push`

因此默认路径没有实际的门类型随机化。

## 5. 当前随机化参数总表

| 类别 | 量 | 默认参数 | 生效位置 |
|---|---|---|---|
| 回合级 | 杯体质量 | `cup_mass_range=(0.1, 0.8)` | `DoorPushEnvCfg` |
| 回合级 | 门板质量 | `door_mass_range=(5.0, 20.0)` | `DoorPushEnvCfg` |
| 回合级 | 门阻尼 | `door_damping_range=(0.5, 5.0)` | `DoorPushEnvCfg` |
| 回合级 | 基座半径 | `base_radius_range=(0.45, 0.60)` | `DoorPushEnvCfg` |
| 回合级 | 基座扇形角 | `base_sector_half_angle_deg=20.0` | `DoorPushEnvCfg` |
| 回合级 | 基座 yaw 扰动 | `base_yaw_delta_deg=10.0` | `DoorPushEnvCfg` |
| 步级 | 位置目标噪声 | `control.position_target_noise_std=0.0` | `configs/env/default.yaml` |
| 步级 | Actor 观测噪声 | `obs_noise_std=0.01` | `DoorPushEnvCfg` |
| 上下文 | episode 持杯上下文 | Stage 1/2/3 离散分布 | `configs/curriculum/default.yaml` |
