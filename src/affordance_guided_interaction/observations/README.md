# observations — 从仿真到策略的观测构建

## 1. 本层做什么

observations 层处在仿真环境 (`envs/`) 与策略网络 (`policy/`) 之间，职责只有一个：

> **从仿真环境的原始物理状态出发，整理出 actor 和 critic 各自需要的结构化输入。**

这里有一条关键的设计边界：

- **actor 的观测只包含部署时可获得的信息**——不依赖仿真 oracle、不依赖隐藏物理参数
- **critic 的观测在 actor 基础上额外包含 privileged information**——只在训练时使用

当前系统为**双臂 Z1 + RealSense 深度相机**平台，不含移动底座与云台。持杯是 episode 开始时随机确定的事件，存在三种情况：

- **无臂持杯**：两条臂均空闲，可自由分配执行任务
- **一臂持杯**（左臂或右臂）：一侧末端被占用，需约束该侧稳定性
- **双臂持杯**：两侧末端均持杯，稳定性约束同时作用于两侧

因此，左右臂状态对称提供，各自带独立的持杯标记与稳定性 proxy，策略根据两个 occupied 标记自行决定角色分配。

---

## 2. 从 env 获取的原始信息清单

observations 层本身不读取仿真，它接收从 envs 层传入的原始状态量。以下是完整的信息来源：

### 2.1 机器人本体状态 → 来自 `ArticulationView`

双臂平台共 12 个旋转关节（左臂 + 右臂各 6 个）：

| 量 | 维度 | 说明 |
|---|---|---|
| 左臂关节位置 `q_left` | (6,) | Z1 左臂 6 个旋转关节角度 |
| 左臂关节速度 `dq_left` | (6,) | 左臂关节角速度 |
| 左臂关节力矩 `tau_left` | (6,) | 左臂当前施加力矩（可选） |
| 右臂关节位置 `q_right` | (6,) | Z1 右臂 6 个旋转关节角度 |
| 右臂关节速度 `dq_right` | (6,) | 右臂关节角速度 |
| 右臂关节力矩 `tau_right` | (6,) | 右臂当前施加力矩（可选） |

这些量在真实机器人上通过编码器和力矩传感器直接获得。

### 2.2 末端执行器刚体状态 → 来自 `RigidPrimView`

左右臂 gripper 状态**对称、始终提供**，不预先区分"执行臂"与"持杯臂"——因为持杯是随机事件，哪侧持杯在 episode 开始时才确定：

**左臂 gripper**

| 量 | 维度 | 说明 |
|---|---|---|
| 位置 `p_left` | (3,) | 在 `base_link` 坐标系下 |
| 朝向 `quat_left` | (4,) | 四元数 (w,x,y,z)，在 `base_link` 坐标系下 |
| 线速度 `v_left` | (3,) | 在 `base_link` 坐标系下 |
| 角速度 `ω_left` | (3,) | 在 `base_link` 坐标系下 |

**右臂 gripper**

| 量 | 维度 | 说明 |
|---|---|---|
| 位置 `p_right` | (3,) | 在 `base_link` 坐标系下 |
| 朝向 `quat_right` | (4,) | 四元数 (w,x,y,z)，在 `base_link` 坐标系下 |
| 线速度 `v_right` | (3,) | 在 `base_link` 坐标系下 |
| 角速度 `ω_right` | (3,) | 在 `base_link` 坐标系下 |

**为什么不预先区分"执行臂"与"持杯臂"？** 持杯是随机事件，可能无臂持杯、左臂持杯、右臂持杯或双臂持杯。若在数据结构层面预先区分角色，就必须在 episode 内动态重新命名字段，增加实现复杂度，也不利于策略学习对称性。采用固定的 left / right 命名，配合 `left_occupied` / `right_occupied` 标记，策略可自行根据上下文决定哪侧承担交互、哪侧需要稳定。

**为什么统一参考 `base_link` 而不是世界坐标系？** 训练中会对机器人基座位置做回合级随机化，若 actor / critic 直接消费世界系坐标，就会把底盘绝对放置误差混进状态表示。统一转换到 `base_link` 后，策略看到的是与底盘自身相关的局部几何关系，表示更稳定，也更符合部署时的控制接口。

### 2.3 Affordance 模块输出 → 来自上层 `door_perception/`

上层 `AffordancePipeline` 每帧对 RGB-D 做端到端处理，observations 层直接接收其输出：

| 量 | 维度 | 说明 |
|---|---|---|
| `z_aff` | (768,) | **冻结 Point-MAE 编码器输出的高维 affordance latent**（mean_pool + max_pool 拼接，2 × trans_dim = 768） |

**产生过程**（完全在 `door_perception/` 内完成，observations 层不感知细节）：
1. LangSAM / Grounded-SAM 2 对 RGB 图做开集分割，得到 door / handle / button 的 binary mask
2. 将 mask 区域的 depth 像素反投影到三维空间，合并为局部点云
3. Voxel 降采样后对齐到固定点数（1024 点）
4. 送入**权重完全冻结**的 Point-MAE，输出 768 维 embedding

**不包含任何手工几何特征**：此前版本中的 RANSAC 平面拟合、包围盒尺寸、到 gripper 距离等 25 维结构化特征已全部移除。空间关系由下游 policy 隐式学习。不单独区分任务进展（z_prog）与 affordance 表示（z_aff），统一以 `z_aff` 作为对门点云的编码结果。

### 2.4 任务上下文 → 来自 `TaskManager`

| 量 | 维度 | 说明 |
|---|---|---|
| `left_occupied` | (1,) | 左臂是否持杯：0 = 空闲，1 = 持杯 |
| `right_occupied` | (1,) | 右臂是否持杯：0 = 空闲，1 = 持杯 |

这两个量在 episode 开始时随机确定，episode 内不变。`left_occupied` 与 `right_occupied` 相互独立，覆盖无杯、单臂持杯（左或右）、双臂持杯四种组合。真实部署中由操作员或上层规划器设定。

### 2.5 Critic 专用量 → 来自仿真 oracle

以下信息**永远不会给 actor**，只给 critic（训练时）：

| 量 | 来源 | 说明 |
|---|---|---|
| 门板精确位姿 | `RigidPrimView.get_world_poses()` | 转换到 `base_link` 后按 pos(3) + quat(4) 提供 |
| 门铰链角度/角速度 | `ArticulationView` | 精确关节状态 |
| 杯体精确位姿 | `RigidPrimView.get_world_poses()` | 转换到 `base_link` 后按 pos(3) + quat(4) 提供 |
| 杯体线速度/角速度 | `RigidPrimView` | 转换到 `base_link` 后提供精确动力学状态 |
| `cup_mass` | domain randomizer | 训练时随机化的隐藏参数（通过质量变化模拟杯中装有不同量液体的效果） |
| `door_mass` | domain randomizer | 训练时随机化的隐藏参数 |
| `door_damping` | domain randomizer | 训练时随机化的隐藏参数 |
| `base_pos` | domain randomizer | 机器人基座的初始世界坐标位置（每回合在标称位置附近微扰采样，范围局限于运动学可达裕度内） |

actor 不知道杯子多重、门有多沉、自己被放到了哪个精确坐标。它必须通过历史交互经验隐式适应。本实验不在杯中放置液体，通过随机化 `cup_mass` 来模拟不同装载量对惯性和稳定性的影响。

---

## 3. 从原始量到 actor observation 的加工过程

env 传进来的是零散的物理量，但 actor 真正需要的是已经按任务语义整理好的输入分支。对于稳定性相关量，环境层应统一读取 Isaac Sim / Isaac Lab 原生加速度与姿态，并组装为共享的 `stability_proxy`；observations 层负责消费并打包这些量，而不是再次自行推导奖励所需物理量。

### 3.1 关节状态 → 直接转发

左臂和右臂的关节位置、速度、力矩分别不做任何加工，直接装入 `proprio` 字段，拆分存储为 `left_joint_*` 和 `right_joint_*`。

### 3.2 末端执行器状态 → 直接转发（左右臂对称）

左臂末端信息装入 `left_gripper_state`，右臂末端信息装入 `right_gripper_state`。这些量由 actor 直接消费；与之并行的 `stability_proxy` 则由环境层基于同一批末端状态统一构造。

### 3.3 环境侧稳定性 proxy → 直接消费（核心接口）

左右臂**各自独立**维护一套稳定性 proxy，分别生成 `left_stability_proxy` 和 `right_stability_proxy`。该 proxy 应由环境层统一构造，策略再根据 `left_occupied` / `right_occupied` 决定以哪侧（或两侧）稳定性为约束重点：

- 无臂持杯：两套 proxy 均存在但稳定性约束权重为零
- 一臂持杯：对应侧的 proxy 作为稳定性约束信号
- 双臂持杯：两套 proxy 均作为稳定性约束信号

**推荐数据链条**：

```text
Isaac Sim / Isaac Lab 原生接口
    │
    ├──→ EE 线加速度 a(t)
    ├──→ EE 角加速度 α(t)
    ├──→ EE 姿态 quat_EE
    │       │
    │       └──→ tilt = ‖P_xy(R_EE^T g)‖
    │
    └──→ 环境侧共享 stability_proxy
              │
              ├──→ observations 打包进 actor_obs
              └──→ rewards 直接消费
```

若需要 `jerk_proxy` 或最近加速度历史，也应由环境侧共享 proxy 在同一条链路中统一维护，而不是由 observations 与 rewards 各自重复维护状态。

### 3.4 末端朝向 → 几何推理得到倾斜度

倾斜度对左右臂**各自独立**计算，逻辑相同：

```
g_world = [0, 0, -9.81]               # 重力方向

R_EE = quat_to_matrix(quat_EE)        # gripper 旋转矩阵

g_local = R_EE^T @ g_world            # 重力在 EE 局部坐标系中的表达

tilt = ‖g_local[x, y]‖               # 投影到 EE 的 xy 平面取模长
```

### 3.5 动作历史 → 由层内部维护

双臂平台动作空间为 12 维（左臂 6 + 右臂 6），动作历史缓存形状为 `(k, 12)`。最近 k 步策略输出的完整双臂力矩被保存在 FIFO 缓存中，episode 开始时全零填满。

### 3.6 门点云 Affordance Latent → 直接转发

`z_aff`（768 维 Point-MAE affordance latent）来自上层 `door_perception/`，observations 层直接打包，不做任何额外处理。若本步无输入则填充零向量。

### 3.7 任务上下文 → 直接转发

`occupied` 直接包装为 `(1,)` 数组。

---

## 4. 最终 actor observation 结构

经过上述加工，actor 拿到的是一个嵌套字典：

```python
actor_obs = {
    # ---- 直接转发 ----
    "proprio": {
        "left_joint_positions":    (6,),      # 左臂关节角度
        "left_joint_velocities":   (6,),      # 左臂关节角速度
        "right_joint_positions":   (6,),      # 右臂关节角度
        "right_joint_velocities":  (6,),      # 右臂关节角速度
        "left_joint_torques":      (6,),      # 左臂关节力矩（可选）
        "right_joint_torques":     (6,),      # 右臂关节力矩（可选）
        "previous_actions":        (k, 12),   # 双臂完整动作历史
    },
    "left_gripper_state": {                   # 左臂末端（始终提供）
        "position":                (3,),
        "orientation":             (4,),      # quat (w,x,y,z)
        "linear_velocity":         (3,),
        "angular_velocity":        (3,),
    },
    "right_gripper_state": {                  # 右臂末端（始终提供）
        "position":                (3,),
        "orientation":             (4,),      # quat (w,x,y,z)
        "linear_velocity":         (3,),
        "angular_velocity":        (3,),
    },
    "context": {
        "left_occupied":           (1,),      # 左臂是否持杯 (0/1)
        "right_occupied":          (1,),      # 右臂是否持杯 (0/1)
    },

    # ---- 加工得到（左右臂各自独立计算）----
    "left_stability_proxy": {
        "tilt":                    float,     # 从左臂 quat_EE 几何推理
        "linear_velocity_norm":    float,     # ‖v_EE‖
        "linear_acceleration":     (3,),      # 环境侧直接读取的 EE 线加速度
        "angular_velocity_norm":   float,     # ‖ω_EE‖
        "angular_acceleration":    (3,),      # 环境侧直接读取的 EE 角加速度
        "jerk_proxy":              float,     # 由环境侧共享 proxy 维护
        "recent_acc_history":      (k,),      # 环境侧共享历史窗口
    },
    "right_stability_proxy": {
        "tilt":                    float,     # 从右臂 quat_EE 几何推理
        "linear_velocity_norm":    float,
        "linear_acceleration":     (3,),
        "angular_velocity_norm":   float,
        "angular_acceleration":    (3,),
        "jerk_proxy":              float,
        "recent_acc_history":      (k,),
    },

    # ---- 直接转发（来自上层 door_perception/ 模块）----
    "z_aff": (768,),           # 冻结 Point-MAE 输出的高维 affordance latent (2 × trans_dim)
}
```

可以这样理解 actor 看到的信息：
- **"我的双臂在做什么"** → proprio（左臂 + 右臂）
- **"两个末端在哪里、做什么"** → left_gripper_state + right_gripper_state
- **"哪侧持杯、稳不稳"** → context（left/right_occupied）+ left/right_stability_proxy
- **"门在哪里、长什么样"** → z_aff（Point-MAE 对门点云的 768 维编码）

---

## 5. Critic observation：在 actor 基础上追加仿真 oracle

Critic 的输入是 actor_obs 的超集。它额外获得的全部是**部署时不可能知道的信息**：

```python
critic_obs = {
    "actor_obs": { ... },                  # 完整的 actor_obs

    "privileged": {
        # 精确物体状态（actor 只有 z_aff，不知道门的精确位姿）
        "door_pose":             (7,),     # 门板相对 base_link 的 pose: pos(3)+quat(4)
        "door_joint_pos":        (1,),     # 铰链精确角度
        "door_joint_vel":        (1,),     # 铰链精确角速度
        "cup_pose":              (7,),     # 杯体相对 base_link 的 pose: pos(3)+quat(4)
        "cup_linear_vel":        (3,),     # 杯体相对 base_link 的线速度
        "cup_angular_vel":       (3,),     # 杯体相对 base_link 的角速度

        # 隐藏物理参数（actor 完全不知道）
        "cup_mass":              (1,),     # 杯体质量（随机化以模拟不同装载量）
        "door_mass":             (1,),     # 门板质量
        "door_damping":          (1,),     # 铰链阻尼系数
        "base_pos":              (3,),     # 机器人基座世界坐标位置（回合级随机化）
    },
}
```

**为什么这样设计？** 因为在 asymmetric actor-critic 框架中，critic 能看到更多信息可以更准确地估计 value function，但 actor 必须只依赖真实可得的信息。训练结束后 critic 被丢弃，只部署 actor。

---

## 6. 信息流总览

```
仿真层 envs/
  │
  ├── ArticulationView (左臂) ─────────────────────┐
  │   q_left(6), dq_left(6), tau_left(6)           │  直接转发
  │                                                 │
  ├── ArticulationView (右臂) ─────────────────────┤
  │   q_right(6), dq_right(6), tau_right(6)        │  直接转发
  │                                                 │
  ├── RigidPrimView (左臂 gripper) ────────────────┤
  │   p_left(3), quat_left(4),                      │  直接转发
  │   v_left(3), ω_left(3), a_left(3), α_left(3)   │  + 环境侧 proxy → left_stability_proxy
  │   (base_link 坐标系)                             │
  │                                                 │
  ├── RigidPrimView (右臂 gripper) ────────────────┤
  │   p_right(3), quat_right(4),                    │  直接转发
  │   v_right(3), ω_right(3), a_right(3), α_right(3) │ + 环境侧 proxy → right_stability_proxy
  │                                                 │
  ├── TaskManager ──────────────────────────────────┤  直接转发
  │   left_occupied(1), right_occupied(1),          │
  │                                                 ├──→ actor_obs
  ├── door_perception/ ─────────────────────────────┤  直接转发
  │   z_aff(768)                                    │
  │                                                 │
  ├── 环境侧共享 proxy / observations 层缓存 ────────┤  内部缓存
  │   动作历史(k, 12)                               │
  │   最近加速度历史 / jerk 所需状态（如启用）        │
  │                                                 │
  ╔═══════════════════════════════════════════════════╗
  ║  以下仅训练时传入，构建 critic_obs               ║
  ╚═══════════════════════════════════════════════════╝
  │                                                 │
  ├── RigidPrimView (门/杯体) ──────────────────────┤
  │   door_pose(7), cup_pose(7)                     │
  │   cup_vel(3), cup_ω(3)                          ├──→ critic_obs["privileged"]
  │                                                 │
  ├── ArticulationView (门铰链) ────────────────────┤
  │   door_joint_pos(1), door_joint_vel(1)          │
  │                                                 │
  └── DomainRandomizer ─────────────────────────────┘
      cup_mass, door_mass, door_damping, base_pos
```

---

## 7. 关键设计决策

### 为什么把双臂关节分拆为 left / right 而不是拼接成 (12,)？

拆分存储保留了臂的语义标签，策略可以结合 `left_occupied` / `right_occupied` 区分"哪条臂在持杯"、"哪条臂在执行任务"。若直接拼接为 (12,)，策略网络必须自行从位置推断臂的身份，增加学习难度。

### 为什么使用 left / right 对称结构，而不是 active / cup 结构？

持杯是随机事件，可能无臂持杯、一臂持杯（左或右）或双臂同时持杯。若预先区分"执行臂"与"持杯臂"，当双臂均持杯时该命名将失去意义；若 episode 内动态切换角色命名，会引入不必要的实现复杂度。采用固定的 left / right 命名加独立的 occupied 标记，结构始终一致，策略根据上下文自行决定角色分配。

### 为什么对左右臂各维护独立的稳定性 proxy？

双臂持杯时，两侧末端均可能承载杯体，需要独立监控各自的运动状态。独立维护两套环境侧稳定性 proxy，确保左右臂的稳定性信号互不干扰。策略可根据 `left_occupied` / `right_occupied` 决定以哪侧（或两侧）的 proxy 作为稳定性约束依据。

### 为什么 actor 消费 Point-MAE latent 而非原始点云或手工几何特征？

`door_perception/` 管线采用端到端设计：原始点云经过冻结的 Point-MAE 编码为 768 维 embedding，维度固定、语义丰富，无需手工设计特征。此前的 RANSAC 平面拟合、包围盒、到 gripper 距离等 25 维手工几何特征已全部移除，policy 结合本体状态（proprioception）本身就有能力隐式学习空间关系，手工特征反而容易因误差级联而崩溃。不单独维护 z_prog（任务进展）与 z_aff（affordance 表示）的区分，统一为对门点云的编码输出，接口更简洁。

### 为什么加速度应从仿真直接读取？

Isaac Sim / Isaac Lab 已提供刚体 / link 级的原生加速度接口，环境层可以直接读取末端线加速度与角加速度。这比在业务层重复做速度差分更清晰，也更符合当前奖励函数对物理语义的要求。若后续为了 sim-to-real 需要引入噪声模型或观测退化，应作为环境侧观测建模策略显式处理，而不是让 rewards 或 observations 各自隐式差分。

### 为什么维护动作历史？

过去几步的完整双臂动作为 recurrent policy 提供了额外的时间上下文。策略可以从"我最近两条臂输出了什么力矩"中推断环境的隐藏动力学参数（比如门有多重、杯子多满）。

### 为什么隐藏参数只给 critic？

`cup_mass`、`door_damping`、`base_pos` 等参数在训练中被随机化。如果 actor 也能看到它们，它就会学到一个依赖这些精确值的策略，部署时无法获得这些值就会失效。让 actor 只看间接信号（速度、加速度、历史），迫使它通过 recurrent backbone 隐式适应参数变化。其中 `base_pos` 的隐藏尤为关键——若 actor 直接获得基座绝对坐标，它会在固定坐标上过拟合出死板的轨迹规划，丧失对部署环境安装误差的鲁棒性。

### tilt 为什么不直接用欧拉角？

欧拉角有万向节锁问题且不连续。四元数→旋转矩阵→重力投影的方式在数学上更稳健，且物理含义更直观——直接度量的是"重力方向偏离杯口法线的程度"。
