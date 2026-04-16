# Joint Position Control Migration

本文档记录“从 joint torque policy 迁移到 joint position target + PD”的结果摘要。当前仓库默认路径已经完成迁移，以下内容描述的是已经落地的接口与保留下来的约束。

## 1. 已完成的接口变化

### 1.1 Action

- policy 输出从 `raw joint torque` 改为 `raw joint position target`
- 输出维度仍为 `12`
- 单位改为 `rad`

### 1.2 执行链路

当前执行链路为：

`raw target -> joint-limit clip -> optional target noise -> re-clip -> set_joint_position_target() -> actuator PD -> effort_limit clip`

### 1.3 Observation

actor / critic 的 actor-side 观测已删除：

- `joint_torques`
- `prev_action`

并改为：

- `prev_joint_target`

当前维度：

- actor obs: `84`
- critic obs: `97`

### 1.4 Reward

已删除的 reward 项：

- `stab_*/smooth`
- `stab_*/reg`
- `safe/joint_limit`
- `safe/torque_limit`

新增并保留的关键项：

- `safe/target_limit`

它惩罚的是 raw policy target 的越界部分，而不是 clip 后的执行目标。

### 1.5 执行层安全

虽然 reward 不再关注 torque 超限，但执行层仍然保留：

- actuator `effort_limit`
- PD 输出力矩的硬截断

该部分只负责执行安全，不直接形成 reward 项。

## 2. 当前配置入口

控制器相关的默认参数在 [configs/env/default.yaml](../configs/env/default.yaml)：

```yaml
control:
  action_type: joint_position
  arm_pd_stiffness: 1000.0
  arm_pd_damping: 100.0
  position_target_noise_std: 0.0
```

reward 有效参数在 [configs/reward/default.yaml](../configs/reward/default.yaml)。

## 3. 当前验证要点

如需确认迁移结果，可检查：

1. `DoorPushEnv._pre_physics_step()` 是否写入 `set_joint_position_target()`
2. actor `proprio` 是否为 `q + dq + prev_joint_target`
3. reward 日志里是否只剩 `safe/joint_vel`、`safe/target_limit`、`safe/cup_drop`
4. TensorBoard 中是否不再出现 `smooth`、`reg`、`joint_limit`、`torque_limit`

## 4. 参考文档

- [docs/training_pipeline_detailed.md](./training_pipeline_detailed.md)
- [src/affordance_guided_interaction/envs/README.md](../src/affordance_guided_interaction/envs/README.md)
- [src/affordance_guided_interaction/envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)
