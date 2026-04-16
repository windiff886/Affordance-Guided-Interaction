# envs — GPU 批量并行仿真环境

本文档只描述当前默认实现，不再保留旧的 torque 直驱语义。

默认环境由 [door_push_env.py](./door_push_env.py) 和 [door_push_env_cfg.py](./door_push_env_cfg.py) 组成，训练侧通过 [direct_rl_env_adapter.py](./direct_rl_env_adapter.py) 读取它输出的 batched tensor。

## 1. 层职责

`envs/` 是项目中唯一直接与 Isaac Lab 物理引擎交互的层，负责：

- 场景装配与 reset
- joint-position 动作执行
- actor / critic 观测构建
- reward 计算
- episode 终止判定

默认路径中，观测与奖励都直接在 `DoorPushEnv` 内部完成，不再存在旧版“环境层输出原始状态，再由独立 observations/rewards 模块二次加工”的执行链。

## 2. 当前动作语义

策略输出为 12 维双臂关节位置目标：

`a_t = q_target_raw ∈ R^12`

- 单位：`rad`
- 语义：双臂 12 个臂关节的绝对目标角
- gripper 不由策略控制，gripper 保持目标由环境内部持续刷新

`_pre_physics_step()` 的执行链路是：

`raw joint target -> joint-limit clip -> optional position-target noise -> re-clip -> set_joint_position_target() -> actuator PD -> effort_limit clip`

具体语义：

1. `DoorPushEnv` 先缓存 policy 的原始输出 `self._cached_raw_joint_target`
2. 按 `robot.data.soft_joint_pos_limits` 对目标角做仿真层限位
3. 若 `position_target_noise_std > 0`，对裁剪后的目标角注入噪声，并再次限位
4. 将结果写入 `robot.set_joint_position_target()`
5. 机器人 actuator 的隐式 PD 控制器生成实际执行力矩
6. PD 输出力矩再受 actuator `effort_limit` 硬截断

当前默认 arm actuator 配置：

- `joint2`（肩关节）组：`effort_limit = 60 N·m`
- 其他 arm joints：`effort_limit = 30 N·m`
- 默认 PD：`stiffness = 1000.0`，`damping = 100.0`

其中 `arm_pd_stiffness`、`arm_pd_damping` 与 `position_target_noise_std` 由 `configs/env/default.yaml -> control` 注入，再由 `scripts/train.py` 回写到 `scene.robot.actuators`。

## 3. 观测空间

### 3.1 Actor 观测

Actor 观测维度为 `84`：

| 分段 | 内容 | 维度 |
|---|---|---|
| `proprio` | `joint_positions(12) + joint_velocities(12) + prev_joint_target(12)` | 36 |
| `ee` | 左右末端 `position(3) + orientation(4) + linear_velocity(3) + angular_velocity(3) + linear_acceleration(3) + angular_acceleration(3)` | 38 |
| `context` | `left_occupied + right_occupied` | 2 |
| `stability` | `left_tilt + right_tilt` | 2 |
| `door_geometry` | `door_center_in_base(3) + door_normal_in_base(3)` | 6 |

总计：`36 + 38 + 2 + 2 + 6 = 84`

当前 actor 观测不再包含：

- `joint_torques`
- `prev_action`

其中 `prev_joint_target` 指的是“上一控制步送入仿真的裁剪后位置目标”，不是 raw policy 输出。

### 3.2 Critic 观测

Critic 观测维度为 `97`：

`critic_obs = actor_obs_clean(84) + privileged(13)`

`privileged` 包含：

- `door_pose(7)`
- `door_joint_pos(1)`
- `door_joint_vel(1)`
- `cup_mass(1)`
- `door_mass(1)`
- `door_damping(1)`
- `cup_dropped(1)`

### 3.3 噪声注入

默认路径下，actor 观测噪声只注入：

- `joint_positions`
- `joint_velocities`

也就是 `proprio` 前 24 维。`prev_joint_target`、`ee`、`context`、`stability`、`door_geometry` 都不注入这一路噪声。critic 使用无噪声真值。

## 4. 奖励结构

当前总奖励为：

`r_total = r_task + r_stab_left + r_stab_right - r_safe`

### 4.1 任务奖励

任务奖励由 3 项组成：

- `task/delta`
- `task/open_bonus`
- `task/approach`

其中 `task/approach_raw` 只作为日志中间量记录。

### 4.2 稳定性奖励

每侧稳定性奖励由 5 项组成，并受 occupancy mask 控制：

- `stab_{side}/zero_acc`
- `stab_{side}/zero_ang`
- `stab_{side}/acc`
- `stab_{side}/ang`
- `stab_{side}/tilt`

已经删除旧版 torque 语义下的：

- `smooth`
- `reg`

### 4.3 安全惩罚

当前安全项只有 3 项：

- `safe/joint_vel`
- `safe/target_limit`
- `safe/cup_drop`

重要语义：

- `safe/target_limit` 基于 **policy 原始输出的目标角** 计算
- 仿真层仍会先对目标角做 joint-limit clip，再交给 PD 执行
- PD 输出力矩的 effort clipping 只发生在执行层，不进入 reward

已经删除旧版安全项：

- `safe/joint_limit`
- `safe/torque_limit`

奖励的完整数学形式见 [Reward.md](./Reward.md)。

## 5. Reset 与域随机化

`_reset_idx(env_ids)` 负责选择性 reset，并在其中完成：

1. 采样 `cup_mass`、`door_mass`、`door_damping`
2. 采样门外扇形环上的机器人 base pose
3. 重置机器人与门的状态
4. 根据 occupancy 对左右杯体执行 teleport / 抓取初始化
5. 将随机化参数写回 PhysX
6. 清零缓存：`_prev_joint_target`、`_prev_door_angle`、加速度缓存、成功标记等

默认 episode 级随机化范围：

- `cup_mass_range = (0.1, 0.8)`
- `door_mass_range = (5.0, 20.0)`
- `door_damping_range = (0.5, 5.0)`
- `base_radius_range = (0.45, 0.60)`
- `base_sector_half_angle_deg = 20.0`
- `base_yaw_delta_deg = 10.0`

## 6. 关键文件

| 文件 | 作用 |
|---|---|
| [door_push_env.py](./door_push_env.py) | 环境主体，执行动作、构建观测、计算奖励与 done |
| [door_push_env_cfg.py](./door_push_env_cfg.py) | 场景、动作空间、观测维度与默认参数 |
| [direct_rl_env_adapter.py](./direct_rl_env_adapter.py) | 将 batched tensor 观测解包为训练侧使用的嵌套字典 |
| [batch_math.py](./batch_math.py) | base 系坐标变换、四元数运算、基座采样 |
| [Reward.md](./Reward.md) | 当前 reward 的数学定义 |

## 7. 当前默认参数入口

运行时最常调的参数入口是：

- `configs/env/default.yaml`
- `configs/task/default.yaml`
- `configs/reward/default.yaml`

其中：

- `configs/env/default.yaml` 负责 `physics_dt`、`decimation`、`control.action_type`、`control.arm_pd_stiffness`、`control.arm_pd_damping`、`control.position_target_noise_std`
- `configs/task/default.yaml` 负责 `door_angle_target`、`cup_drop_threshold`
- `configs/reward/default.yaml` 负责 task / stability / safety 的有效 reward 权重
