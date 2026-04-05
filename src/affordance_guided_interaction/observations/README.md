# observations — 从环境状态到策略输入

> 文档状态：目标设计导向
>
> 本文档描述 `observations/` 层的目标形态。
> 它以 [training_pipeline_detailed.md](/home/windiff/Code/Affordance-Guided-Interaction/training_pipeline_detailed.md) 和 [policy/README.md](/home/windiff/Code/Affordance-Guided-Interaction/src/affordance_guided_interaction/policy/README.md) 为准。

---

## 1. 本层做什么

`observations/` 位于 `envs/` 与 `policy/` 之间，只负责一件事：

> 把环境给出的原始物理状态和视觉结果，整理成 actor / critic 的正式输入。

这里有两条硬边界：

- actor 只看部署时原则上可获得的信息
- critic 在 actor 基础上额外看训练期 privileged information

本层不负责：

- 不做 reward 计算
- 不做 PPO 更新
- 不直接处理原始 `RGB-D`
- 不决定课程阶段
- 不给 actor 泄漏仿真 oracle

---

## 2. 原始输入来源

### 2.1 机器人本体状态

来自双臂 12 个关节的原始状态：

- 左臂关节位置 `q_left ∈ R^6`
- 左臂关节速度 `dq_left ∈ R^6`
- 左臂当前或最近测得力矩 `tau_left ∈ R^6`
- 右臂关节位置 `q_right ∈ R^6`
- 右臂关节速度 `dq_right ∈ R^6`
- 右臂当前或最近测得力矩 `tau_right ∈ R^6`

这些量构成 actor 的 `proprio` 主体。

### 2.2 末端执行器状态

左右末端都以固定的 `left / right` 顺序提供，不在数据结构层面改名为“执行臂”或“持杯臂”。

每侧至少包含：

- 位置 `p ∈ R^3`
- 姿态 `quat ∈ R^4`
- 线速度 `v ∈ R^3`
- 角速度 `omega ∈ R^3`
- 线加速度 `a ∈ R^3`
- 角加速度 `alpha ∈ R^3`

所有几何量统一表达在 `base_link` 坐标系下，而不是世界坐标系。

### 2.3 在线视觉结果

视觉前端的正式输出只有一个：

- `door_embedding ∈ R^768`

其来源链路固定为：

```text
RGB-D -> 2D视觉识别 -> 3D点云提取 -> 冻结 Point-MAE 编码 -> door_embedding
```

`observations/` 不处理原始图像、mask 或原始点云，只消费已经编码好的视觉结果。

### 2.4 持杯上下文

每个 episode 在 reset 时采样上下文，并在回合内固定：

- `left_occupied ∈ {0, 1}`
- `right_occupied ∈ {0, 1}`

它们共同对应四种正式上下文：

- `none`
- `left_only`
- `right_only`
- `both`

### 2.5 Critic 专用隐藏量

以下信息只给 critic：

- 门板精确位姿 `door_pose`
- 门铰链精确角度 `door_joint_pos`
- 门铰链精确角速度 `door_joint_vel`
- 关键域随机化参数：`cup_mass`、`door_mass`、`door_damping`、`base_pos`
- 掉杯事件标志 `cup_dropped ∈ {0, 1}`

目标设计中，critic **不需要**杯体精确位姿、线速度或角速度。

---

## 3. 视觉缓存与时间语义

在线视觉采用固定频率更新，而不是每个控制步完整重跑一次。

因此每个 environment 都维护最近一次有效视觉缓存：

```text
VisualCachePerEnv:
  door_embedding: R^768
  is_initialized: bool
```

其时间语义如下：

1. episode reset 后必须先完成一次 warm start，得到有效 `door_embedding`
2. 到达视觉刷新步时，更新缓存
3. 非刷新步直接复用最近一次缓存

这意味着 actor 在每个控制步都能拿到 `door_embedding`，但它并不要求每一步都来自新的视觉前向。

---

## 4. Actor observation

actor 的正式输入定义为：

```text
o_actor,t = {
  proprio,
  ee,
  context,
  stability,
  visual
}
```

### 4.1 `proprio`

`proprio` 至少包含：

- 双臂 12 个关节位置
- 双臂 12 个关节速度
- 双臂 12 个当前执行力矩或最近测得力矩
- 最近一步动作 `a_(t-1)`

如果后续实验确认有收益，也允许保留很短的动作历史窗口，但正式接口不依赖长 history stacking。

### 4.2 `ee`

`ee` 承载左右末端在 `base_link` 坐标系下的状态：

- 左末端位置、姿态、线速度、角速度、线加速度、角加速度
- 右末端位置、姿态、线速度、角速度、线加速度、角加速度

这里要特别注意：

- 线加速度和角加速度属于 `ee`
- 它们不再重复塞进 `stability`

### 4.3 `context`

`context` 只描述持杯上下文：

- `left_occupied`
- `right_occupied`

它告诉策略当前哪一侧处在“需要兼顾持杯稳定性”的模式下，但不直接规定动作角色分配。

### 4.4 `stability`

`stability` 只保留左右两侧最核心的持杯稳定性信号：

- 左侧倾斜程度
- 右侧倾斜程度

其职责很单一：

> 告诉 actor 当前末端姿态是否在破坏“杯口朝上”的稳定性。

与运动相关的速度、加速度等量统一归入 `ee`，不在 `stability` 中重复表达。

### 4.5 `visual`

`visual` 只包含：

```text
door_embedding ∈ R^768
```

它来自当前环境最近一次视觉缓存，而不是原始 `RGB-D`。

### 4.6 推荐结构

```python
actor_obs = {
    "proprio": {
        "joint_positions": (12,),
        "joint_velocities": (12,),
        "joint_torques": (12,),
        "prev_action": (12,),
    },
    "ee": {
        "left": {
            "position": (3,),
            "orientation": (4,),
            "linear_velocity": (3,),
            "angular_velocity": (3,),
            "linear_acceleration": (3,),
            "angular_acceleration": (3,),
        },
        "right": {
            "position": (3,),
            "orientation": (4,),
            "linear_velocity": (3,),
            "angular_velocity": (3,),
            "linear_acceleration": (3,),
            "angular_acceleration": (3,),
        },
    },
    "context": {
        "left_occupied": (1,),
        "right_occupied": (1,),
    },
    "stability": {
        "left_tilt": (1,),
        "right_tilt": (1,),
    },
    "visual": {
        "door_embedding": (768,),
    },
}
```

---

## 5. Critic observation

critic 的正式输入定义为：

```text
o_critic,t = {
  actor_obs,
  privileged
}
```

其中 `privileged` 至少包含：

- `door_pose`
- `door_joint_pos`
- `door_joint_vel`
- `cup_mass`
- `door_mass`
- `door_damping`
- `base_pos`
- `cup_dropped`

推荐结构如下：

```python
critic_obs = {
    "actor_obs": {...},
    "privileged": {
        "door_pose": (7,),
        "door_joint_pos": (1,),
        "door_joint_vel": (1,),
        "cup_mass": (1,),
        "door_mass": (1,),
        "door_damping": (1,),
        "base_pos": (3,),
        "cup_dropped": (1,),
    },
}
```

这里的关键约束是：

- critic 可以看到掉杯事件
- critic 不要求看到杯体连续刚体状态

---

## 6. 信息流总览

```text
envs/
  ├── joint states ------------------------------┐
  ├── ee states ---------------------------------┤
  ├── context -----------------------------------┤
  ├── privileged oracle -------------------------┤
  └── RGB-D -------------------------------------┘
                    │
                    ▼
perception runtime
  fixed-frequency update
  -> cached door_embedding
                    │
                    ▼
observations/
  raw state + cached embedding
  -> actor_obs
  -> critic_obs
                    │
                    ▼
policy/
  actor / critic
```

---

## 7. 关键设计决策

### 为什么始终保持 left / right 对称结构

因为正式目标场景同时包含无持杯、单臂持杯和双臂持杯。固定 `left / right` 命名最稳定，不会因上下文变化而频繁改字段语义。

### 为什么 `stability` 只保留 tilt

因为 actor 需要的是最直接的持杯稳定性信号，而不是把所有末端动力学量都塞进一个“稳定性大杂烩”分支。速度和加速度已经在 `ee` 中表达，再重复一次只会让接口变得混乱。

### 为什么 critic 不看杯体精确位姿和速度

目标设计里假设杯体在未掉落前被稳定抓持。对 value estimate 真正关键的是门真实推进状态、隐藏动力学参数，以及是否已经发生掉杯，而不是连续的杯体刚体状态。

### 为什么视觉必须通过缓存接入 observations

因为训练系统采用固定频率视觉更新。`observations/` 需要消费的是“当前可用的视觉结果”，而不是假设视觉每一步都新鲜计算完成。
