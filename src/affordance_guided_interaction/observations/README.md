# observations — 从环境状态到策略输入

> 文档状态：目标设计导向
>
> 本文档描述观测层的设计规范。
> 参见 [training_pipeline_detailed.md](../../../docs/training_pipeline_detailed.md) 和 [policy/README.md](../policy/README.md)。
>
> **实现说明**: 观测构建现在直接在 `DoorPushEnv._get_observations()` 中以 PyTorch tensor
> 操作完成，不再通过独立的 `ActorObsBuilder`/`CriticObsBuilder` 类。
> 本模块保留 `HistoryBuffer` 和 `StabilityProxy` 作为通用工具类。

---

## 1. 本层做什么

`observations/` 位于 `envs/` 与 `policy/` 之间，只负责一件事：

> 把环境给出的原始物理状态和门几何信息，整理成 actor / critic 的正式输入。

这里有两条硬边界：

- actor 只看部署时原则上可获得的信息
- critic 在 actor 基础上额外看训练期 privileged information

本层不负责：

- 不做 reward 计算
- 不做 PPO 更新
- 不直接处理原始 `RGB-D` 或感知流水线
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

### 2.3 门几何信号

门几何观测的正式输出为 6 维信号：

- `door_center_in_base ∈ R^3` -- 门叶中心在 base_link 坐标系下的位置
- `door_normal_in_base ∈ R^3` -- 门叶法向量在 base_link 坐标系下的方向

其来源为仿真 ground truth：

```text
仿真门叶刚体位姿 (door leaf body pose)
  + 固定的 DoorLeaf 本体坐标系局部偏移
  -> 变换到 base_link 坐标系
  -> door_center_in_base(3) + door_normal_in_base(3)
```

该信号直接从仿真物理引擎获取，不经过相机或感知流水线。感知流水线（camera -> segmentation -> point cloud -> Point-MAE）不再是默认观测路径的一部分。

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
- 关键域随机化参数：`cup_mass`、`door_mass`、`door_damping`
  注：当前 episode 级随机化还会生成 `base_yaw`，但它尚未进入 critic privileged 向量。
  注：`base_pos` 已从 critic privileged 向量中移除。
- 掉杯事件标志 `cup_dropped ∈ {0, 1}`

目标设计中，critic **不需要**杯体精确位姿、线速度或角速度。

---

## 3. 门几何信号的数据流

门几何信号直接从仿真 ground truth 计算，无需缓存或异步更新：

```text
DoorGeometryPerEnv:
  door_center_in_base: R^3
  door_normal_in_base: R^3
```

其计算流程如下：

1. episode reset 后，门叶刚体位姿由仿真器直接提供
2. 每个控制步，从门叶 body pose 中提取位置与朝向
3. 利用固定的 DoorLeaf 本体坐标系局部偏移，计算门叶中心与法向量
4. 将结果变换到 base_link 坐标系，得到 `door_center_in_base` 和 `door_normal_in_base`

由于该信号来自仿真 ground truth，不存在视觉缓存中的延迟或刷新频率问题。每个控制步的计算都是即时完成的。

---

## 4. Actor observation

actor 的正式输入定义为：

```text
o_actor,t = {
  proprio,
  ee,
  context,
  stability,
  door_geometry
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

### 4.5 `door_geometry`

`door_geometry` 包含：

```text
door_center_in_base ∈ R^3
door_normal_in_base ∈ R^3
```

它来自仿真 ground truth（门叶刚体位姿 + 固定局部偏移），经过 base_link 坐标系变换后直接提供。

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
    "door_geometry": {
        "door_center_in_base": (3,),
        "door_normal_in_base": (3,),
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
DoorPushEnv (GPU batched)
  ├── ArticulationView -> joint states ─────────┐
  ├── body state view -> ee states ─────────────┤
  ├── occupancy flags -> context ───────────────┤  DoorPushEnv._get_observations()
  ├── simulation oracle -> privileged ──────────┤  直接构建 actor_obs(96D)
  └── door leaf body pose -> door_geometry ─────┘  和 critic_obs(109D) tensor
                    │
                    ▼
DirectRLEnvAdapter
  tensor -> list[dict] 转换
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

### 为什么门几何信号直接从仿真 ground truth 计算

门几何信号（`door_center_in_base` 和 `door_normal_in_base`）从仿真 ground truth 直接获取，而非经过相机/感知流水线。这样做的原因是：

1. 训练阶段不需要引入感知噪声，保持策略学习的数据干净
2. 感知流水线（camera -> segmentation -> point cloud -> Point-MAE）不再是默认观测路径的一部分，减少了训练时的计算开销和异步复杂度
3. 如需引入感知，可以在后续阶段作为独立模块叠加，而不会影响核心策略训练
