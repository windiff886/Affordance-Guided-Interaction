# policy — 当前策略网络接口

本文档描述当前代码中的 policy 实现，不再使用旧的 torque 输出语义。

## 1. Actor

Actor 位于 [actor.py](./actor.py)，采用：

`多分支 encoder -> RNN backbone -> Gaussian action head`

输入分支为：

- `proprio`
- `ee`
- `context`
- `stability`
- `door_geometry`

### 1.1 Actor 输入语义

`proprio` 当前只包含：

- `joint_positions(12)`
- `joint_velocities(12)`
- `prev_joint_target(12)`

已经不再包含：

- `joint_torques`
- `prev_action`

### 1.2 Actor 输出语义

Actor 输出 `12` 维连续动作，语义是：

`q_target_raw ∈ R^12`

- 单位：`rad`
- 含义：双臂 12 个 arm joints 的绝对关节目标角

环境不会把这 12 维解释为 torque。执行链路是：

`raw position target -> joint-limit clip -> set_joint_position_target() -> actuator PD`

也就是说：

- `policy/` 负责产生目标角
- `envs/` 负责执行层裁剪与 PD 控制
- PD 输出力矩的 effort clipping 不属于 actor 输出语义

## 2. Critic

Critic 位于 [critic.py](./critic.py)，采用非对称 actor-critic 结构：

`actor_obs_clean + privileged -> encoder/MLP -> value`

当前 privileged 维度为 `13`，包含：

- `door_pose`
- `door_joint_pos`
- `door_joint_vel`
- `cup_mass`
- `door_mass`
- `door_damping`
- `cup_dropped`

Critic 不参与部署，只在训练时提供 value estimate。

## 3. 当前维度

| 项 | 维度 |
|---|---|
| Actor obs | 84 |
| Critic obs | 97 |
| Actor action | 12 |
| Critic privileged | 13 |

Actor 分支维度：

- `proprio = 36`
- `ee = 38`
- `context = 2`
- `stability = 2`
- `door_geometry = 6`

## 4. RNN 语义

Actor 默认使用：

- `rnn_type = gru`
- `rnn_hidden = 512`
- `rnn_layers = 1`

RNN 的作用是处理 POMDP：actor 看不到门真实铰链状态，也看不到域随机化出的隐藏动力学参数，因此要通过交互历史做隐式辨识。

## 5. 导出接口

[scripts/export_policy.py](../../../scripts/export_policy.py) 导出的扁平输入顺序是：

`[proprio | ee | context | stability | door_geometry]`

其中 `proprio` 已经按当前实现改为：

`joint_positions + joint_velocities + prev_joint_target`

导出模型输出仍是 `12` 维 joint position targets。

## 6. 关键边界

当前默认路径中：

- actor 不消费 `joint_torques`
- actor 不输出 torque
- critic 也不再从 actor 侧消费 torque 观测
- policy 不负责 target clipping、PD 控制或 effort clipping

这些职责全部留在环境执行层。
