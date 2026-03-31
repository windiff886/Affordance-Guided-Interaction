# observations — 从仿真到策略的观测构建

## 1. 本层做什么

observations 层处在仿真环境 (`envs/`) 与策略网络 (`policy/`) 之间，职责只有一个：

> **从仿真环境的原始物理状态出发，整理出 actor 和 critic 各自需要的结构化输入。**

这里有一条关键的设计边界：

- **actor 的观测只包含部署时可获得的信息**——不依赖仿真 oracle、不依赖隐藏物理参数
- **critic 的观测在 actor 基础上额外包含 privileged information**——只在训练时使用

---

## 2. 从 env 获取的原始信息清单

observations 层本身不读取仿真，它接收从 envs 层传入的原始状态量。以下是完整的信息来源：

### 2.1 机器人本体状态 → 来自 `ArticulationView`

| 量 | 维度 | 说明 |
|---|---|---|
| 关节位置 `q_t` | (6,) | Z1 单臂 6 个旋转关节角度 |
| 关节速度 `dq_t` | (6,) | 关节角速度 |
| 关节力矩 `tau_t` | (6,) | 当前施加力矩（可选） |

这些量在真实机器人上通过编码器和力矩传感器直接获得。

### 2.2 Gripper 刚体状态 → 来自 `RigidPrimView`

| 量 | 维度 | 说明 |
|---|---|---|
| gripper 位置 `p_EE` | (3,) | 在 `base_link` 坐标系下 |
| gripper 朝向 `quat_EE` | (4,) | 四元数 (w,x,y,z)，在 `base_link` 坐标系下 |
| gripper 线速度 `v_EE` | (3,) | 在 `base_link` 坐标系下 |
| gripper 角速度 `ω_EE` | (3,) | 在 `base_link` 坐标系下 |

**为什么用 gripper frame 而不是 wrist frame？** 因为杯子直接被 gripper 夹持，稳定性关注的核心是杯体末端。gripper frame 是最直接反映杯体状态的参考系。

**为什么参考 base_link 而不是世界坐标系？** 对移动机器人而言，base_link 下的表示更有利于控制规划和 sim-to-real 泛化。

### 2.3 任务上下文 → 来自 `TaskManager`

| 量 | 维度 | 说明 |
|---|---|---|
| `occupied` | (1,) | 0 = 空手，1 = 持杯 |
| `stability_level` | (1,) | 稳定性要求等级 (0/1/2) |

这两个量在 episode 开始时确定，episode 内不变。真实部署中由操作员或上层规划器设定。

### 2.4 门点云 → 来自 `door_perception/`

| 量 | 维度 | 说明 |
|---|---|---|
| `door_point_cloud` | (N, 3) | 经 RGB-D 分割与反投影得到的门表面点云 |

上层感知模块负责从原始视觉输入中提取门点云，observations 层只接收结果。

### 2.5 Critic 专用量 → 来自仿真 oracle

以下信息**永远不会给 actor**，只给 critic（训练时）：

| 量 | 来源 | 说明 |
|---|---|---|
| 门板精确位姿 | `RigidPrimView.get_world_poses()` | pos(3) + quat(4) |
| 门铰链角度/角速度 | `ArticulationView` | 精确关节状态 |
| 杯体精确位姿 | `RigidPrimView.get_world_poses()` | pos(3) + quat(4) |
| 杯体线速度/角速度 | `RigidPrimView` | 精确动力学状态 |
| `cup_mass` | domain randomizer | 训练时随机化的隐藏参数 |
| `cup_fill_ratio` | domain randomizer | 训练时随机化的隐藏参数 |
| `door_mass` | domain randomizer | 训练时随机化的隐藏参数 |
| `door_damping` | domain randomizer | 训练时随机化的隐藏参数 |

actor 不知道杯子多重、液体填了多少、门有多沉。它必须通过历史交互经验隐式适应。

---

## 3. 从原始量到 actor observation 的加工过程

env 传进来的是零散的物理量，但 actor 真正需要的不仅仅是"当前帧的数值"——它还需要**时间上的对比信息**（加速度）和**物理几何推理**（倾斜方向）。这些加工逻辑是 observations 层存在的核心理由。

### 3.1 关节状态 → 直接转发

关节位置、速度、力矩不做任何加工，直接装入 `proprio` 字段。这些量本身已经是良好的状态表示。

### 3.2 Gripper 刚体状态 → 直接转发

gripper 位置、朝向、线速度、角速度同样直接装入 `gripper_state` 字段。但它们同时也是下面"稳定性 proxy"计算的原料。

### 3.3 Gripper 速度 → 差分估计加速度、jerk（核心加工）

这是 observations 层最重要的加工逻辑。仿真中虽然可以直接读取加速度，但**真实部署中通常没有精确加速度传感器**，因此我们刻意通过速度差分来估计——保持 sim-to-real 一致性。

**差分链条**：

```
v_EE(t), v_EE(t-1)
    │
    ├──→ 线加速度 a(t) = (v(t) - v(t-1)) / dt        ... 一阶差分
    │
    ├──→ a(t), a(t-1)
    │       │
    │       └──→ jerk = ‖a(t) - a(t-1)‖ / dt          ... 二阶差分
    │
    └──→ 最近 k 帧的 ‖a‖ 序列                          ... 滑动窗口历史

ω_EE(t), ω_EE(t-1)
    │
    └──→ 角加速度 α(t) = (ω(t) - ω(t-1)) / dt         ... 一阶差分
```

这要求 observations 层**记住上一帧的速度和上一帧的加速度**——这就是为什么它内部维护了一个流式状态。每个 episode 开始时这个状态被清空，第一步的加速度和 jerk 输出为零。

### 3.4 Gripper 朝向 → 几何推理得到倾斜度

杯子倒不倒水，本质上取决于杯口相对重力方向的倾斜程度。这个量从 gripper 朝向通过一步几何变换得到：

```
g_world = [0, 0, -9.81]               # 重力方向

R_EE = quat_to_matrix(quat_EE)        # gripper 旋转矩阵

g_local = R_EE^T @ g_world            # 重力在 EE 局部坐标系中的表达

tilt = ‖g_local[x, y]‖               # 投影到 EE 的 xy 平面取模长
```

直觉：如果 gripper 保持竖直（杯口朝上），重力在 EE 局部坐标系中完全沿 z 轴，xy 分量为零 → tilt = 0。gripper 越倾斜，xy 分量越大 → tilt 越大。

### 3.5 动作历史 → 由层内部维护

最近 k 步策略输出的力矩被保存在一个固定长度的 FIFO 缓存中。这不需要从 env 获取，而是 observations 层在每次 `build()` 调用时接收当前步的动作并追加到缓存。

episode 开始时缓存被全零填满，避免冷启动时长度不一致。

### 3.6 任务上下文和门点云 → 直接转发

`occupied`、`stability_level` 直接包装为 `(1,)` 数组。门点云直接转发，如果本步没有点云输入则使用空数组 `(0, 3)`。

---

## 4. 最终 actor observation 结构

经过上述加工，actor 拿到的是一个嵌套字典：

```python
actor_obs = {
    # ---- 直接转发 ----
    "proprio": {
        "joint_positions":       (6,),     # 原始关节角度
        "joint_velocities":      (6,),     # 原始关节角速度
        "joint_torques":         (6,),     # 原始关节力矩（可选）
        "previous_actions":      (k, 6),   # 层内缓存的动作历史
    },
    "gripper_state": {
        "position":              (3,),     # 原始 EE 位置
        "orientation":           (4,),     # 原始 EE 四元数
        "linear_velocity":       (3,),     # 原始 EE 线速度
        "angular_velocity":      (3,),     # 原始 EE 角速度
    },
    "context": {
        "occupied":              (1,),     # 直接转发
        "stability_level":       (1,),     # 直接转发
    },

    # ---- 加工得到 ----
    "stability_proxy": {
        "tilt":                  float,    # 从 quat_EE 几何推理
        "linear_velocity_norm":  float,    # ‖v_EE‖
        "linear_acceleration":   (3,),     # 速度一阶差分
        "angular_velocity_norm": float,    # ‖ω_EE‖
        "angular_acceleration":  (3,),     # 角速度一阶差分
        "jerk_proxy":            float,    # 加速度二阶差分模长
        "recent_acc_history":    (k,),     # 滑动窗口加速度模长
    },

    # ---- 直接转发 ----
    "door_point_cloud":          (N, 3),   # 上层感知输出
}
```

可以这样理解 actor 看到的信息：
- **"我的身体在做什么"** → proprio + gripper_state
- **"我现在稳不稳"** → stability_proxy
- **"任务要求是什么"** → context
- **"门在哪里"** → door_point_cloud

---

## 5. Critic observation：在 actor 基础上追加仿真 oracle

Critic 的输入是 actor_obs 的超集。它额外获得的全部是**部署时不可能知道的信息**：

```python
critic_obs = {
    "actor_obs": { ... },                  # 完整的 actor_obs

    "privileged": {
        # 精确物体状态（actor 只有点云，不知道门的精确位姿）
        "door_pose":             (7,),     # 门板 world pose
        "door_joint_pos":        (1,),     # 铰链精确角度
        "door_joint_vel":        (1,),     # 铰链精确角速度
        "cup_pose":              (7,),     # 杯体 world pose
        "cup_linear_vel":        (3,),     # 杯体线速度
        "cup_angular_vel":       (3,),     # 杯体角速度

        # 隐藏物理参数（actor 完全不知道）
        "cup_mass":              (1,),     # 杯体质量
        "cup_fill_ratio":        (1,),     # 液体填充率
        "door_mass":             (1,),     # 门板质量
        "door_damping":          (1,),     # 铰链阻尼系数
    },
}
```

**为什么这样设计？** 因为在 asymmetric actor-critic 框架中，critic 能看到更多信息可以更准确地估计 value function，但 actor 必须只依赖真实可得的信息。训练结束后 critic 被丢弃，只部署 actor。

---

## 6. 信息流总览

```
仿真层 envs/
  │
  ├── ArticulationView ──────────────────────────┐
  │   q(6), dq(6), tau(6)                        │  直接转发
  │                                               │
  ├── RigidPrimView ─────────────────────────────┤
  │   p_EE(3), quat_EE(4), v_EE(3), ω_EE(3)    │  直接转发
  │   (已转换到 base_link 坐标系)                 │  + 差分加工 → stability_proxy
  │                                               │
  ├── TaskManager ───────────────────────────────┤  直接转发
  │   occupied(1), stability_level(1)            │
  │                                               ├──→ actor_obs
  ├── door_perception/ ──────────────────────────┤  直接转发
  │   door_point_cloud(N, 3)                     │
  │                                               │
  ├── 层内维护 ───────────────────────────────────┤  内部缓存
  │   动作历史(k, 6)                              │
  │   前帧速度/加速度(差分状态)                    │
  │                                               │
  ╔═══════════════════════════════════════════════╗
  ║  以下仅训练时传入，构建 critic_obs             ║
  ╚═══════════════════════════════════════════════╝
  │                                               │
  ├── RigidPrimView (门/杯体) ───────────────────┤
  │   door_pose(7), cup_pose(7)                  │
  │   cup_vel(3), cup_ω(3)                       ├──→ critic_obs["privileged"]
  │                                               │
  ├── ArticulationView (门铰链) ─────────────────┤
  │   door_joint_pos(1), door_joint_vel(1)       │
  │                                               │
  └── DomainRandomizer ──────────────────────────┘
      cup_mass, cup_fill_ratio, door_mass, door_damping
```

---

## 7. 关键设计决策

### 为什么加速度不从仿真直接读取？

仿真可以给出精确加速度，但真实机器人通常只有 IMU（噪声大）或编码器二阶差分。用速度差分估计加速度，训练和部署用同一套逻辑，减少 sim-to-real gap。

### 为什么维护动作历史？

过去几步的动作为 recurrent policy 提供了额外的时间上下文。策略可以从"我最近输出了什么力矩"中推断环境的隐藏动力学参数（比如门有多重）。

### 为什么隐藏参数只给 critic？

`cup_mass`、`door_damping` 等参数在训练中被随机化。如果 actor 也能看到它们，它就会学到一个依赖这些精确值的策略，部署时无法获得这些值就会失效。让 actor 只看间接信号（速度、加速度、历史），迫使它通过 recurrent backbone 隐式适应参数变化。

### tilt 为什么不直接用欧拉角？

欧拉角有万向节锁问题且不连续。四元数→旋转矩阵→重力投影的方式在数学上更稳健，且物理含义更直观——直接度量的是"重力方向偏离杯口法线的程度"。
