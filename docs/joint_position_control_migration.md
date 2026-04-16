# Policy Action Migration: Torque to Joint Position Target

## 0. 文档定位

本文档是对当前训练系统的一份修改设计文档，用于指导项目从

- `policy 输出 12 维关节力矩`

迁移到

- `policy 输出 12 维绝对关节位置目标（rad）`
- `环境层将位置目标交给 PD 控制器执行`

的完整改造。

本文档不描述“当前代码已经实现了什么”，而是明确：

1. 当前实现的问题在哪里
2. 目标控制链路应该是什么
3. reward / observation / config / export / logging 应如何同步修改
4. 需要改哪些文件
5. 推荐的实施顺序和验证方法

本文档与 [docs/training_pipeline_detailed.md](./training_pipeline_detailed.md) 的关系是：

- `training_pipeline_detailed.md` 仍是当前主训练管线现状文档
- 本文档描述的是下一步要落实的 joint-position control 迁移设计

---

## 1. 当前实现审计

当前仓库中的默认训练路径仍然是 torque 直驱。

### 1.1 当前动作语义

当前 `policy` 输出 12 维连续动作，语义是双臂 12 个关节的 raw torque。

关键代码位置：

- [src/affordance_guided_interaction/policy/README.md](../src/affordance_guided_interaction/policy/README.md)
- [src/affordance_guided_interaction/policy/action_head.py](../src/affordance_guided_interaction/policy/action_head.py)
- [src/affordance_guided_interaction/envs/door_push_env.py](../src/affordance_guided_interaction/envs/door_push_env.py)

其中 [door_push_env.py](../src/affordance_guided_interaction/envs/door_push_env.py) 的 `_pre_physics_step()` 当前逻辑是：

1. 缓存 raw action
2. 按 `effort_limits` 对 action 做力矩裁剪
3. 注入动作噪声
4. 再次裁剪
5. 通过 `robot.set_joint_effort_target()` 写入 arm effort target

### 1.2 当前 actuator 配置

当前双臂关节 actuator 在
[src/affordance_guided_interaction/envs/door_push_env_cfg.py](../src/affordance_guided_interaction/envs/door_push_env_cfg.py)
中配置为 torque 直驱：

- arm joints: `stiffness = 0.0`
- arm joints: `damping = 0.0`

这意味着当前没有用于 arm joints 的 PD 闭环。

### 1.3 当前 reward 中与 torque 强耦合的部分

当前 reward 设计中，以下部分直接依赖 torque 动作语义：

- 稳定性项中的 `smooth`
- 稳定性项中的 `reg`
- 安全项中的 `torque_limit`

对应文档与代码位置：

- [src/affordance_guided_interaction/envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)
- [src/affordance_guided_interaction/envs/door_push_env.py](../src/affordance_guided_interaction/envs/door_push_env.py)
- [configs/reward/default.yaml](../configs/reward/default.yaml)

### 1.4 当前 observation 中与 torque 强耦合的部分

当前 `proprio` 里包含：

- `joint_positions`
- `joint_velocities`
- `joint_torques`
- `prev_action`

其中：

- `joint_torques` 是执行结果中的力矩反馈
- `prev_action` 当前语义是“上一时刻已执行的 torque action”

这同样与目标的 joint-position control 语义不一致。

---

## 2. 本次迁移已确认的设计决策

以下内容已经明确，不再作为开放问题：

### 2.1 动作语义

- `policy` 直接输出真实关节角度目标
- 单位为 `rad`
- 输出含义是双臂 12 个 arm joints 的绝对关节位置目标

不采用：

- 归一化到 `[-1, 1]` 的动作
- 关节位置增量动作
- raw torque 动作

### 2.2 控制执行链路

目标执行链路为：

`raw joint target -> 仿真层按 joint limits 做位置限位 -> set_joint_position_target() -> PD 生成 torque -> effort limit 截断`

### 2.3 PD 参数管理

- 默认参考值暂定：`stiffness = 1000`，`damping = 100`
- 这些参数不应继续硬编码在 `DoorPushEnvCfg`
- 应放入 `configs/env/default.yaml` 的 `control` 段

### 2.4 Observation 设计

保留：

- `joint_positions`
- `joint_velocities`
- 上一时刻的位置目标

删除：

- `joint_torques`

### 2.5 Reward 设计

保留：

- 任务奖励
- 稳定性项中的 `zero_acc / zero_ang / acc / ang / tilt`
- 安全项中的 `joint_vel / cup_drop`

删除：

- `smooth`
- `reg`
- `joint_limit`
- `torque_limit`

新增：

- `policy 原始目标关节角越界惩罚`

### 2.6 PD torque saturation 的处理原则

- PD 控制器输出的 torque 仍然必须按 effort limits 做硬截断
- 该截断不进入 reward
- 该截断只用于仿真安全和监控

---

## 3. 目标控制链路

### 3.1 变量定义

记：

- policy 原始输出为 `q_target_raw ∈ R^12`
- 关节物理限位为 `[q_min, q_max]`
- 仿真层实际下发的位置目标为 `q_target_cmd ∈ R^12`
- 当前关节位置为 `q ∈ R^12`
- 当前关节速度为 `dq ∈ R^12`
- PD 输出 torque 为 `tau_pd ∈ R^12`
- 经过 effort limit 截断后的执行 torque 为 `tau_applied ∈ R^12`

### 3.2 位置限位

仿真执行前，环境必须先对 policy 原始目标角做位置限位：

```math
q_{\text{target\_cmd}} = \operatorname{clip}(q_{\text{target\_raw}}, q_{\min}, q_{\max})
```

这里的 `clip` 是执行层的安全要求，不等于“reward 已经原谅越界行为”。

也就是说：

- 执行层必须安全
- reward 仍然要对 `q_target_raw` 的越界部分做惩罚

### 3.3 PD 控制律

若采用目标速度为零的标准 PD 控制，则执行含义可写为：

```math
\tau_{pd} = K_p (q_{\text{target\_cmd}} - q) + K_d (0 - \dot{q})
```

其中：

- `Kp = stiffness`
- `Kd = damping`

在 Isaac Lab 的 implicit actuator 语义下，不一定需要在 Python 里显式手写该公式；
只要 actuator 配置切换为 position control 语义，并通过 `set_joint_position_target()` 写入目标角，物理引擎会按该 PD 规律求解。

### 3.4 Torque 限幅

PD 输出的 torque 仍需受每个关节的 effort limit 约束：

```math
\tau_{\text{applied}} = \operatorname{clip}(\tau_{pd}, -\tau_{\max}, \tau_{\max})
```

该步骤属于执行层安全边界，不进入 reward 惩罚。

### 3.5 推荐的 `_pre_physics_step()` 新语义

迁移后的 `_pre_physics_step(actions)` 推荐流程：

1. 缓存 `q_target_raw`
2. 按 joint limits 计算 `q_target_cmd`
3. 可选地对 `q_target_cmd` 注入 position-target 级噪声
4. 再做一次位置限位，确保噪声后仍在 joint range 内
5. 构造 full-joint target tensor
6. 调用 `robot.set_joint_position_target()`
7. 保持 gripper hold targets
8. 缓存 `prev_joint_target`

---

## 4. 配置体系改造

### 4.1 配置归位原则

控制器参数属于环境执行层，应放入 `configs/env/default.yaml`，不新开 `configs/control/default.yaml`。

原因：

- 与 `physics_dt`、`decimation` 同属环境执行层
- `build_env_cfg()` 已经负责把 `configs/env/default.yaml` 注入到 `DoorPushEnvCfg`
- 可以避免把配置文件从 7 份扩展到 8 份

### 4.2 推荐的 `configs/env/default.yaml` 结构

建议新增 `control` 段，例如：

```yaml
physics_dt: 0.008333
decimation: 2

control:
  action_type: joint_position
  arm_pd_stiffness: 1000.0
  arm_pd_damping: 100.0
  position_target_noise_std: 0.0
  monitor_pd_torque_clip: true
```

### 4.3 `DoorPushEnvCfg` 建议新增字段

建议在 [src/affordance_guided_interaction/envs/door_push_env_cfg.py](../src/affordance_guided_interaction/envs/door_push_env_cfg.py)
中增加以下环境级字段：

- `control_action_type: str = "joint_position"`
- `arm_pd_stiffness: float = 1000.0`
- `arm_pd_damping: float = 100.0`
- `position_target_noise_std: float = 0.0`
- `monitor_pd_torque_clip: bool = True`

### 4.4 Policy 配置建议

[configs/policy/default.yaml](../configs/policy/default.yaml) 中的以下字段需要同步重写语义：

- `actor.action_dim: 12` 保持不变
- `actor.log_std_init` 保持为高斯位置目标策略的初始化标准差
- `actor.include_torques` 应删除

原因：

- observation 中将不再包含 `joint_torques`
- 继续保留 `include_torques` 只会制造文档与代码歧义

### 4.5 Reward 配置建议

[configs/reward/default.yaml](../configs/reward/default.yaml) 需要：

- 删除 `stability.w_smooth`
- 删除 `stability.w_reg`
- 删除 `safety.beta_limit`
- 删除 `safety.beta_torque`
- 新增 `safety.beta_target` 或等价命名，用于目标角越界惩罚

推荐命名：

- `safety.beta_target`

这样可以明确表示该项惩罚的是 `policy target overflow`，而不是实际 joint state 或执行 torque。

---

## 5. 环境层修改设计

### 5.1 actuator 配置修改

当前 arm joints 为 torque 直驱：

- `stiffness = 0`
- `damping = 0`

迁移后应改成 position target + PD：

- `stiffness = cfg.arm_pd_stiffness`
- `damping = cfg.arm_pd_damping`

对应改动位置：

- [src/affordance_guided_interaction/envs/door_push_env_cfg.py](../src/affordance_guided_interaction/envs/door_push_env_cfg.py)

### 5.2 `_pre_physics_step()` 需要替换的缓存

当前缓存：

- `_cached_raw_action`
- `_prev_action`
- `_prev_prev_action`

迁移后建议改为：

- `_cached_raw_joint_target`
- `_prev_joint_target`

`_prev_prev_action` 可以删除，因为 `smooth/reg` 已确认删除。

### 5.3 关节限位来源

位置目标的执行限位与越界惩罚都应基于物理引擎给出的 joint limits，而不是重新手写一套关节范围常量。

推荐继续使用：

- `robot.data.soft_joint_pos_limits`

分别用于：

- 执行前 clip
- reward 中的目标角越界超额计算

### 5.4 observation 维度变化

当前 actor observation：

- proprio: `q(12) + dq(12) + tau(12) + prev_action(12) = 48`
- total: `96`

迁移后 actor observation 推荐改为：

- proprio: `q(12) + dq(12) + prev_joint_target(12) = 36`
- ee: `38`
- context: `2`
- stability: `2`
- door_geometry: `6`

即：

```math
36 + 38 + 2 + 2 + 6 = 84
```

因此：

- actor obs: `(N, 84)`
- critic obs: `(N, 97)`，因为 `84 + privileged(13) = 97`

### 5.5 `DirectRLEnvAdapter` 需要同步修改

[src/affordance_guided_interaction/envs/direct_rl_env_adapter.py](../src/affordance_guided_interaction/envs/direct_rl_env_adapter.py)
中的 flat tensor slice 需要整体重排。

重点变化：

- 删除 `joint_torques` slice
- 将 `prev_action` 重命名为 `prev_joint_target`
- actor obs 总维度从 `96` 改成 `84`
- critic obs 总维度从 `109` 改成 `97`

推荐不要继续沿用 `prev_action` 这个 key 名称。

推荐新名称：

- `prev_joint_target`

原因是位置控制语义下继续使用 `prev_action` 会模糊“动作”和“位置目标”的边界。

---

## 6. Policy / Observation / Export 修改设计

### 6.1 Actor 输入修改

[src/affordance_guided_interaction/policy/actor.py](../src/affordance_guided_interaction/policy/actor.py)
中的 `proprio` 展平逻辑需要从：

- `q`
- `dq`
- `joint_torques`
- `prev_action`

改成：

- `q`
- `dq`
- `prev_joint_target`

### 6.2 `ActorConfig.include_torques` 删除

以下位置均需要删除 `include_torques` 分支逻辑：

- `policy/actor.py`
- `policy/critic.py`
- `scripts/train.py`
- `scripts/export_policy.py`
- `configs/policy/default.yaml`
- `configs/README.md`

### 6.3 ActionHead 的文档语义修改

[src/affordance_guided_interaction/policy/action_head.py](../src/affordance_guided_interaction/policy/action_head.py)
本身仍可保留“12 维高斯连续动作头”的实现形式，但文档和注释必须改成：

- 输出 12 维绝对关节位置目标
- 单位 `rad`

而不是：

- 输出 12 维 raw torque

### 6.4 Export 接口变化

[scripts/export_policy.py](../scripts/export_policy.py) 需要同步修改：

- `proprio_dim` 从 `48/36` 的条件分支简化为固定 `36`
- 导出描述中的输出语义改为 `joint position target`
- 若包装器里仍使用 `prev_action` 命名，应同步改为 `prev_joint_target`

### 6.5 Critic 不应继续消费 torque 观测

Critic 的 actor-like 分支输入必须与 actor 保持一致，因此：

- Critic 也不再消费 `joint_torques`

这里不应出现：

- actor 不看 torque，但 critic 的 actor-like branch 仍看 torque

否则会破坏 observation contract 的一致性。

---

## 7. Reward 与 Safety 修改设计

## 7.1 新的总奖励结构

迁移后推荐总奖励形式为：

```math
r_t = r_{\text{task}} + m_L \cdot r_{\text{stab}}^L + m_R \cdot r_{\text{stab}}^R - r_{\text{safe}}
```

其中：

```math
r_{\text{stab}}^{(\cdot)} = r_{\text{zero-acc}} + r_{\text{zero-ang}} + r_{\text{acc}} + r_{\text{ang}} + r_{\text{tilt}}
```

```math
r_{\text{safe}} = r_{\text{joint\_vel}} + r_{\text{target\_limit}} + r_{\text{cup\_drop}}
```

### 7.2 删除的 reward 项

以下项应从 reward 设计中完全移除：

- `stab_left/smooth`
- `stab_left/reg`
- `stab_right/smooth`
- `stab_right/reg`
- `safe/joint_limit`
- `safe/torque_limit`

删除原因如下。

#### 7.2.1 删除 `smooth/reg`

它们原本约束的是 torque-action 语义下的：

- 力矩变化率
- 力矩幅值

但在新方案中：

- policy 不再输出 torque
- 继续用 PD 内部执行 torque 来构造这两项会把控制器内部变量重新拉回 reward 主体

因此本次迁移中直接删除，不做同义替换。

#### 7.2.2 删除 `safe/joint_limit`

该项惩罚的是“实际 joint state 接近 joint limit”。

但本次已确认改为：

- policy 原始目标角越界时才罚
- 实际状态接近边界不再单独罚

因此 `safe/joint_limit` 删除。

#### 7.2.3 删除 `safe/torque_limit`

该项原本惩罚的是 raw torque action 越界。

由于 policy 动作语义已经改成位置目标，这一项不再成立，应删除。

### 7.3 新增 `target_limit` 惩罚

新的安全项需要针对 policy 原始输出的目标角越界进行惩罚。

定义每个关节的越界超额：

```math
e_i^{\text{target}} =
\max(0, q^{\text{raw}}_{target,i} - q^{\max}_i)
+ \max(0, q^{\min}_i - q^{\text{raw}}_{target,i})
```

则目标角越界惩罚可写为：

```math
r_{\text{target\_limit}} = \beta_{\text{target}} \sum_{i=1}^{12} \left(e_i^{\text{target}}\right)^2
```

这里使用 raw target，而不是 clip 之后的 target。

原因：

- 执行层必须安全
- reward 仍要明确告诉策略“你给出了越界目标”

### 7.4 保留 `joint_vel`

`safe/joint_vel` 应保留。

原因：

- 这项约束的是系统实际动态状态
- 即使动作语义变成位置目标，过大的 joint velocity 仍是重要安全风险

### 7.5 保留 `cup_drop`

`safe/cup_drop` 应保持不变。

### 7.6 PD torque saturation 的日志策略

虽然 `tau_pd` 的截断不进入 reward，但建议新增监控量：

- `control/pd_torque_clip_count`
- `control/pd_torque_clip_fraction`
- `control/pd_torque_clip_max_excess`

这些量可用于判断：

- `Kp/Kd` 是否过大
- 位置目标是否过激
- 是否频繁打到 effort limit

但不应直接并入 `reward/safe/*`。

---

## 8. TensorBoard 与日志修改设计

### 8.1 `reward_terms/*` 需要删除的标签

以下标签应从训练日志中移除：

- `reward_terms/stab_left/smooth`
- `reward_terms/stab_left/reg`
- `reward_terms/stab_right/smooth`
- `reward_terms/stab_right/reg`
- `reward_terms/safe/joint_limit`
- `reward_terms/safe/torque_limit`

### 8.2 `reward_terms/*` 需要新增的标签

应新增：

- `reward_terms/safe/target_limit`

### 8.3 `docs/tensorboard_guide.md` 需要同步重写

[docs/tensorboard_guide.md](./tensorboard_guide.md) 中当前对 `reward_terms/*` 的解释仍是 torque 版本，必须同步改成位置控制版本。

推荐的 reward 子项集合应变为：

- `reward_terms/task/*`: `delta`, `open_bonus`, `approach`, `approach_raw`
- `reward_terms/stab_left/*` 与 `reward_terms/stab_right/*`: `zero_acc`, `zero_ang`, `acc`, `ang`, `tilt`
- `reward_terms/safe/*`: `joint_vel`, `target_limit`, `cup_drop`

### 8.4 控制链路监控

除 reward 外，建议新增单独的 `control/*` 命名空间，用于记录：

- 下发位置目标越界率
- PD torque saturation 频率
- 实际 applied torque 的统计量

这些监控有助于调 `Kp/Kd`，但不应和 reward 概念混合。

---

## 9. 文档同步范围

以下文档在代码实现完成后都需要同步更新：

- [docs/training_pipeline_detailed.md](./training_pipeline_detailed.md)
- [docs/tensorboard_guide.md](./tensorboard_guide.md)
- [src/affordance_guided_interaction/envs/README.md](../src/affordance_guided_interaction/envs/README.md)
- [src/affordance_guided_interaction/envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)
- [src/affordance_guided_interaction/observations/README.md](../src/affordance_guided_interaction/observations/README.md)
- [src/affordance_guided_interaction/policy/README.md](../src/affordance_guided_interaction/policy/README.md)
- [src/affordance_guided_interaction/training/README.md](../src/affordance_guided_interaction/training/README.md)
- [configs/README.md](../configs/README.md)

推荐的同步原则是：

- 本文档先作为迁移设计基准
- 代码实现完成后，再把现状文档改成“已实现的 position-control 版本”

---

## 10. 代码修改清单

以下文件预计需要改动。

### 10.1 环境与配置

- `configs/env/default.yaml`
  - 新增 `control` 段
- `configs/reward/default.yaml`
  - 删除 `w_smooth`、`w_reg`、`beta_limit`、`beta_torque`
  - 新增 `beta_target`
- `configs/policy/default.yaml`
  - 删除 `include_torques`
- `scripts/train.py`
  - 注入 env control 参数
  - 删除 `include_torques` 路径
- `src/affordance_guided_interaction/envs/door_push_env_cfg.py`
  - arm actuator 改成 PD 位置控制语义
  - 增加 env control 字段
- `src/affordance_guided_interaction/envs/door_push_env.py`
  - `_pre_physics_step()` 从 effort target 改成 joint position target
  - observation 维度改为 84 / 97
  - reward 删除 torque-based 项并新增 `target_limit`

### 10.2 适配器与策略

- `src/affordance_guided_interaction/envs/direct_rl_env_adapter.py`
  - slice 布局重写
- `src/affordance_guided_interaction/policy/actor.py`
  - proprio 输入维度改为 36
  - 删除 torque 分支
- `src/affordance_guided_interaction/policy/critic.py`
  - 同步删除 torque 分支
- `src/affordance_guided_interaction/policy/action_head.py`
  - 注释与文档语义改为 joint position target
- `scripts/export_policy.py`
  - flat input 维度同步修改

### 10.3 训练缓冲与 rollout

- `src/affordance_guided_interaction/training/rollout_buffer.py`
  - actor branch dims 更新
- `src/affordance_guided_interaction/training/rollout_collector.py`
  - 一般不需要大改
  - 但观测 contract 和日志字段需要同步新命名

---

## 11. 推荐实施顺序

建议按以下顺序改，避免一次性大爆炸修改。

### 阶段 A: 控制链路切换

1. 在 `configs/env/default.yaml` 中加入 `control` 段
2. 在 `DoorPushEnvCfg` 中增加 control 字段
3. 将 arm actuator 改成 position target + PD
4. 将 `_pre_physics_step()` 改成写入 `set_joint_position_target()`
5. 增加 raw target cache 与 clip 后 target cache

### 阶段 B: reward 迁移

1. 删除 `smooth/reg`
2. 删除 `joint_limit/torque_limit`
3. 新增 `target_limit`
4. 保留 `joint_vel/cup_drop`
5. 更新 `reward_info` 和 TensorBoard 标签

### 阶段 C: observation / policy / export 迁移

1. 删除 `joint_torques`
2. 将 `prev_action` 改为 `prev_joint_target`
3. 调整 actor/critic 输入维度
4. 更新 adapter、export、config 文档

### 阶段 D: 文档与监控收尾

1. 更新现状文档
2. 更新 `tensorboard_guide`
3. 增加 `control/*` 监控

---

## 12. 验证建议

### 12.1 单步控制验证

先不跑完整训练，先验证以下事实：

1. policy 输出的位置目标会被正确写入 `set_joint_position_target()`
2. arm joints 的 actuator 确实工作在 PD 位置控制模式
3. joint target 超出物理范围时，执行层能正确 clip
4. PD 输出 torque 即使打到 effort limit，仿真仍稳定

### 12.2 reward 验证

应重点检查：

1. `target_limit` 只对 raw target 越界部分生效
2. `joint_limit` 与 `torque_limit` 已彻底消失
3. `smooth/reg` 已彻底消失
4. `joint_vel` 与 `cup_drop` 仍正常记录

### 12.3 observation contract 验证

应检查：

1. actor obs 维度从 `96` 变为 `84`
2. critic obs 维度从 `109` 变为 `97`
3. `joint_torques` 不再出现在 adapter 输出字典中
4. `prev_joint_target` 与当前控制链路语义一致

### 12.4 训练稳定性验证

建议先做小规模 smoke test：

- 小 `num_envs`
- 短 rollout
- 短 total_steps

重点观察：

- 仿真是否抖振
- 门接触是否显著恶化
- `control/pd_torque_clip_fraction` 是否过高
- `reward_terms/safe/target_limit` 是否长期过大

---

## 13. 风险与注意事项

### 13.1 直接输出真实 `rad` 的探索风险

当前设计已确认 policy 直接输出真实关节角度 `rad`，这意味着：

- 高斯策略天然是无界的
- 训练初期更容易频繁输出越界目标

因此必须同时做好两件事：

1. 执行层 joint-target clip
2. reward 中对 raw target overflow 的明确惩罚

否则训练初期会出现：

- 要么仿真不安全
- 要么越界被执行层静默吞掉，策略学不到边界

### 13.2 高 `Kp` 带来的饱和风险

`stiffness=1000, damping=100` 只是当前参考起点，不保证就是最终最优值。

风险包括：

- 频繁打满 torque limit
- 接触瞬间抖振增强
- 门接触变硬，影响探索

因此必须提供：

- 配置层可调
- `control/*` 监控

### 13.3 不要混淆三种“边界”

迁移后必须明确区分三类边界：

1. `raw target` 是否越界
   - 对应 reward 惩罚
2. `cmd target` 是否已 clip 到 joint range
   - 对应执行层安全
3. `PD torque` 是否打到 effort limit
   - 对应执行层安全和监控

这三者不能再混写成一个“torque limit”概念。

---

## 14. 结论

本次迁移的核心不是简单把 `set_joint_effort_target()` 改成
`set_joint_position_target()`，而是要把整条训练链路的语义一起改干净：

- action 改成 `joint position target`
- actuator 改成 `PD`
- observation 删除 `joint_torques`
- reward 删除 torque-based 设计并新增 `target_limit`
- logging / export / docs 同步改成位置控制语义

如果只改控制 API，不改 reward、observation 和日志命名，系统会长期处于“代码语义、文档语义、训练信号语义互相打架”的状态。

本文档建议以“控制链路先切换、reward 再迁移、observation 与文档最后清理”的顺序实施，以降低回归风险。
