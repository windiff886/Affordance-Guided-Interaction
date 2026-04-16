# Affordance-Guided Interaction 训练管线技术文档

本文档描述当前默认训练路径。当前实现已经完成从 `torque action` 到 `joint position target + PD` 的迁移，文中所有动作相关描述均以此为准。

## 1. 任务定义

项目目标是在 Isaac Lab GPU 并行仿真中训练一套统一策略，使双臂机器人在四种 occupancy 上下文下完成推门，同时在持杯侧保持稳定：

- `none`
- `left_only`
- `right_only`
- `both`

当前策略输出为：

`a_t = q_target_raw ∈ R^12`

即双臂 12 个 arm joints 的绝对关节目标角，单位为 `rad`。

环境会将该目标角交给机器人 actuator 的隐式 PD 控制器执行，而不是把它当作 joint torque 直接注入仿真。

## 2. 系统主线

默认训练闭环为：

`configs + assets -> DoorPushEnvCfg -> DoorPushEnv -> DirectRLEnvAdapter -> Actor/Critic -> RolloutCollector -> RolloutBuffer -> PPOTrainer`

关键模块：

| 模块 | 作用 |
|---|---|
| `scripts/train.py` | 装配训练运行时与各组件 |
| `envs/door_push_env.py` | 批量仿真、动作执行、观测、奖励、终止 |
| `envs/direct_rl_env_adapter.py` | 将 batched tensor 观测转换为训练侧结构 |
| `policy/actor.py` | 输出 joint position targets |
| `policy/critic.py` | 消费 actor_obs + privileged 做 value estimation |
| `training/*` | rollout、buffer、GAE、PPO、课程 |

## 3. 动作与控制链路

当前执行链路是：

`raw joint target -> joint-limit clip -> optional position-target noise -> re-clip -> set_joint_position_target() -> PD torque generation -> effort_limit clip`

详细分工：

1. Actor 输出 `q_target_raw`
2. `DoorPushEnv._pre_physics_step()` 缓存 raw target，供 reward 中 `safe/target_limit` 使用
3. 环境按 `robot.data.soft_joint_pos_limits` 进行位置限位
4. 如 `position_target_noise_std > 0`，在限位后注入位置目标噪声并再次限位
5. 通过 `robot.set_joint_position_target()` 将目标角写入仿真
6. Actuator 隐式 PD 生成执行力矩
7. 执行力矩受 actuator `effort_limit` 约束

默认控制参数入口在 [configs/env/default.yaml](./../configs/env/default.yaml)：

```yaml
physics_dt: 0.008333
decimation: 2

control:
  action_type: joint_position
  arm_pd_stiffness: 1000.0
  arm_pd_damping: 100.0
  position_target_noise_std: 0.0
```

其中 `arm_pd_stiffness` 和 `arm_pd_damping` 会在 `scripts/train.py` 中回写到 `scene.robot.actuators["shoulder_joints"]` 与 `["arm_joints"]`。

## 4. 观测接口

### 4.1 Actor 观测

Actor 观测维度为 `84`：

`actor_obs = proprio(36) + ee(38) + context(2) + stability(2) + door_geometry(6)`

各分段如下：

| 分支 | 内容 | 维度 |
|---|---|---|
| `proprio` | `joint_positions(12) + joint_velocities(12) + prev_joint_target(12)` | 36 |
| `ee` | 左右末端位姿、速度、加速度 | 38 |
| `context` | `left_occupied + right_occupied` | 2 |
| `stability` | `left_tilt + right_tilt` | 2 |
| `door_geometry` | `door_center_in_base + door_normal_in_base` | 6 |

当前 actor 观测已经删除：

- `joint_torques`
- `prev_action`

### 4.2 Critic 观测

Critic 观测维度为 `97`：

`critic_obs = actor_obs_clean(84) + privileged(13)`

`privileged` 包括：

- `door_pose(7)`
- `door_joint_pos(1)`
- `door_joint_vel(1)`
- `cup_mass(1)`
- `door_mass(1)`
- `door_damping(1)`
- `cup_dropped(1)`

### 4.3 噪声

当前默认路径下：

- actor 只对 `joint_positions` / `joint_velocities` 注入观测噪声 `obs_noise_std`
- critic 读取无噪声真值
- 执行动作噪声不再是默认路径的一部分；默认动作侧噪声是 `position_target_noise_std`

## 5. Reward

总奖励：

`r_total = r_task + r_stab_left + r_stab_right - r_safe`

### 5.1 Task

当前任务奖励包含：

- `task/delta`
- `task/open_bonus`
- `task/approach`

其中 `task/approach_raw` 只作为日志辅助项记录。

### 5.2 Stability

每侧稳定性奖励现在只有 5 项：

- `zero_acc`
- `zero_ang`
- `acc`
- `ang`
- `tilt`

旧版 torque 语义下的：

- `smooth`
- `reg`

已经删除。

### 5.3 Safety

当前安全惩罚只有 3 项：

- `safe/joint_vel`
- `safe/target_limit`
- `safe/cup_drop`

其中：

- `safe/target_limit` 惩罚的是 **policy 原始目标角越界**
- 仿真层仍会对 joint target 做 clip
- actuator PD 输出 torque 的饱和裁剪只负责执行安全，不进入 reward

已经删除：

- `safe/joint_limit`
- `safe/torque_limit`

更完整的数学定义见 [envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)。

## 6. Reward 默认参数

默认训练会从 [configs/reward/default.yaml](../configs/reward/default.yaml) 注入有效奖励超参。

### 6.1 Task

- `w_delta = 5000.0`
- `alpha = 0.3`
- `k_decay = 0.5`
- `w_open = 25000.0`
- `success_angle_threshold = 1.2`
- `w_approach = 200.0`
- `approach_eps = 1.0e-6`
- `approach_stop_angle = 0.10`

### 6.2 Stability

- `w_zero_acc = 0.0`
- `lambda_acc = 2.0`
- `w_zero_ang = 0.0`
- `lambda_ang = 1.0`
- `w_acc = 0.05`
- `w_ang = 0.001`
- `w_tilt = 0.03`

### 6.3 Safety

- `mu = 0.9`
- `beta_vel = 0.5`
- `beta_target = 1.0`
- `w_drop = 100.0`

## 7. Reset 与域随机化

默认路径的 episode 级随机化由 `DoorPushEnv._reset_idx()` 直接采样并落到 PhysX：

- `cup_mass_range = (0.1, 0.8)`
- `door_mass_range = (5.0, 20.0)`
- `door_damping_range = (0.5, 5.0)`
- `base_radius_range = (0.45, 0.60)`
- `base_sector_half_angle_deg = 20.0`
- `base_yaw_delta_deg = 10.0`

每次 selective reset 还会：

1. 依据课程或外部 override 设置 occupancy
2. 重置机器人与门的状态
3. 对持杯 env 执行杯体抓取初始化
4. 清零 `_prev_joint_target`、`_prev_door_angle`、加速度缓存与成功标志

## 8. PPO 与课程

训练侧仍然是 PPO + asymmetric actor-critic + recurrent actor：

- actor 默认 `GRU(512, 1 layer)`
- critic 为纯 MLP
- GAE：`gamma = 0.99`，`lam = 0.95`
- PPO：`clip_eps = 0.2`

课程仍为三阶段：

| 阶段 | 上下文分布 |
|---|---|
| `stage_1` | `none: 1.0` |
| `stage_2` | `left_only: 0.5, right_only: 0.5` |
| `stage_3` | `none/left_only/right_only/both` 各 `0.25` |

跃迁条件仍为滑动窗口平均成功率达到 `0.8`。

## 9. TensorBoard 关键标签

当前默认训练路径关注以下标签：

- `collect/episode_success_rate`
- `reward/total`
- `reward/task`
- `reward/stab_left`
- `reward/stab_right`
- `reward/safe`
- `reward_terms/task/*`
- `reward_terms/stab_left/*`
- `reward_terms/stab_right/*`
- `reward_terms/safe/joint_vel`
- `reward_terms/safe/target_limit`
- `reward_terms/safe/cup_drop`
- `curriculum/stage`
- `train/entropy`
- `train/approx_kl`
- `train/explained_variance`

## 10. 当前默认路径与历史路径边界

当前默认路径已经不再使用：

- raw torque action
- actor 侧 `joint_torques`
- stability 中的 `smooth/reg`
- safety 中的 `joint_limit/torque_limit`
- 视觉门 embedding 作为默认门观测输入

当前训练主线以：

- 结构化门几何 `door_geometry(6)`
- joint-position action
- actuator PD 执行

作为唯一默认实现。

## 11. 推荐阅读顺序

1. [src/affordance_guided_interaction/envs/README.md](../src/affordance_guided_interaction/envs/README.md)
2. [src/affordance_guided_interaction/envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)
3. [src/affordance_guided_interaction/observations/README.md](../src/affordance_guided_interaction/observations/README.md)
4. [src/affordance_guided_interaction/policy/README.md](../src/affordance_guided_interaction/policy/README.md)
5. [src/affordance_guided_interaction/training/README.md](../src/affordance_guided_interaction/training/README.md)
6. [docs/randomization.md](./randomization.md)
7. [docs/tensorboard_guide.md](./tensorboard_guide.md)
