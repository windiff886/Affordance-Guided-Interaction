# DoorPushEnv Reward 说明

本文档描述当前默认实现的 reward。旧版 torque 直驱相关项已经移除，以下公式以当前位置控制版本为准。

## 1. 总式

总奖励为：

`r_total = r_task + r_stab_left + r_stab_right - r_safe`

其中：

- `r_task`：推门主任务奖励
- `r_stab_left / r_stab_right`：左右持杯稳定性奖励
- `r_safe`：正惩罚量，进入总奖励时统一相减

## 2. 任务奖励

### 2.1 角度增量项

设当前门角度为 `theta_t`，上一控制步门角度为 `theta_{t-1}`：

`delta_t = theta_t - theta_{t-1}`

增量项采用分段权重：

- 当 `theta_t <= success_angle_threshold` 时，权重为 `w_delta`
- 当 `theta_t > success_angle_threshold` 时，权重按 `1 - k_decay * (theta_t - success_angle_threshold)` 线性衰减，并以 `alpha` 为下限

即：

`r_task_delta = weight(theta_t) * delta_t`

### 2.2 一次性成功 bonus

当门角度首次跨过 `success_angle_threshold` 时触发：

`r_task_open_bonus = w_open * 1[newly_succeeded]`

该 bonus 每个 episode 只给一次。

### 2.3 接近门板奖励

接近奖励在门刚开始推动时激活，用于帮助策略建立接触：

`r_task_approach = 1[theta_t < approach_stop_angle] * w_approach * approach_raw`

其中：

`approach_raw = max(1 - current_dist^2 / (initial_dist^2 + eps), 0)`

`current_dist` 由左右 EE 到门板推门侧大表面的最近距离计算而来，`initial_dist` 是 episode 开始时的对应距离。

### 2.4 任务奖励汇总

`r_task = r_task_delta + r_task_open_bonus + r_task_approach`

## 3. 稳定性奖励

稳定性项仅对持杯侧生效。若某一侧当前未持杯，则该侧稳定性奖励为零。

对每一侧 `s ∈ {left, right}`：

`r_stab_s = zero_acc + zero_ang + acc + ang + tilt`

### 3.1 零线加速度奖励

`r_zero_acc = w_zero_acc * exp(-lambda_acc * ||a||^2)`

### 3.2 零角加速度奖励

`r_zero_ang = w_zero_ang * exp(-lambda_ang * ||alpha||^2)`

### 3.3 线加速度惩罚

`r_acc = -w_acc * ||a||^2`

### 3.4 角加速度惩罚

`r_ang = -w_ang * ||alpha||^2`

### 3.5 倾斜惩罚

`r_tilt = -w_tilt * ||tilt_perp||^2`

其中 `tilt_perp` 是重力方向在末端局部坐标系中偏离局部 `Y` 轴的 `xz` 分量。

### 3.6 已删除的旧项

以下 torque 语义下的旧稳定性项已经删除，不再进入当前 reward：

- `smooth`
- `reg`

原因是当前 policy 输出的是关节位置目标，不再直接输出 joint torque。

## 4. 安全惩罚

安全惩罚始终激活，当前只保留 3 项：

`r_safe = r_safe_joint_vel + r_safe_target_limit + r_safe_cup_drop`

### 4.1 关节速度超限惩罚

设每个关节速度上限为 `dq_max`，触发比例为 `mu`：

`vel_threshold = mu * dq_max`

`vel_excess = max(|dq| - vel_threshold, 0)`

`r_safe_joint_vel = beta_vel * sum(vel_excess^2)`

### 4.2 目标角越界惩罚

这是当前版本最关键的安全项。

设 policy 原始输出为 `q_target_raw`，关节上下限为 `q_min, q_max`。环境执行前会先做仿真层 clip：

`q_target_cmd = clip(q_target_raw, q_min, q_max)`

但 reward 惩罚的不是 `q_target_cmd`，而是 raw policy 输出的越界部分：

`target_excess_high = max(q_target_raw - q_max, 0)`

`target_excess_low = max(q_min - q_target_raw, 0)`

`target_excess = target_excess_high + target_excess_low`

`r_safe_target_limit = beta_target * sum(target_excess^2)`

这保证了两件事：

- 执行层永远安全，仿真只接收 clip 后的位置目标
- 策略若持续输出越界目标角，仍会在 reward 上收到清晰惩罚

### 4.3 杯体脱落惩罚

若持杯侧 cup 与对应 EE 的距离超过 `cup_drop_threshold`：

`r_safe_cup_drop = w_drop`

并且 episode 会终止。

### 4.4 已删除的旧项

以下旧安全项已经删除：

- `safe/joint_limit`
- `safe/torque_limit`

另外，PD 控制器输出力矩的饱和裁剪仍然存在，但只属于执行层安全约束，不进入 reward。

## 5. TensorBoard 标签

当前 `reward_info` 中的主要 key 为：

- `task`
- `task/delta`
- `task/open_bonus`
- `task/approach`
- `task/approach_raw`
- `stab_left`
- `stab_left/zero_acc`
- `stab_left/zero_ang`
- `stab_left/acc`
- `stab_left/ang`
- `stab_left/tilt`
- `stab_right`
- `stab_right/zero_acc`
- `stab_right/zero_ang`
- `stab_right/acc`
- `stab_right/ang`
- `stab_right/tilt`
- `safe`
- `safe/joint_vel`
- `safe/target_limit`
- `safe/cup_drop`
- `total`

## 6. 默认参数来源

当前默认训练路径的有效 reward 权重来自 [configs/reward/default.yaml](../../../configs/reward/default.yaml)，不是 `DoorPushEnvCfg` 中保留的回退默认值。

当前 YAML 默认值为：

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
