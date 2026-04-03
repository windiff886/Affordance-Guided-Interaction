# envs — 仿真环境封装与任务管理

## 1. 本层做什么

envs 层直接对接 Isaac Sim 物理引擎，职责是把底层复杂的仿真推演过程包装成强化学习标准的 `reset()` / `step()` 接口。它是整个系统中**唯一直接与物理引擎交互的层**。

```
                training/ (PPO 训练循环)
                     │ action
                     ▼
           ┌──────────────────────────────────────┐
           │  envs/                                │  ◄── 本层
           │                                       │
           │  scene_factory.py   装配场景资产        │
           │  door_env.py        环境主体生命周期    │
           │  task_manager.py    任务进度状态机       │
           │  contact_monitor.py 接触事件汇总        │
           └──────────┬────────────┬────────────────┘
                      │            │
               原始物理状态    接触事件 + done
                      │            │
                      ▼            ▼
              observations/    rewards/
```

这里有几条关键的设计边界：

- **envs 负责忠实地采集物理真值，并统一组装共享 stability proxy**（原生加速度、姿态、tilt、jerk/history 等与稳定性建模直接相关的量）
- **envs 只负责报告发生了什么事**（谁碰谁了、门转了多少度、杯子掉没掉），不对这些事件赋予奖惩价值（这是 `rewards/` 的职责）
- **envs 负责接收域随机化参数并落实到物理引擎中**，但参数的采样逻辑在 `training/domain_randomizer.py`

---

## 2. 环境对外暴露的数据

envs 层在每一步 `step()` 之后，向外输出以下数据供下游模块消费：

### 2.1 原始物理状态 → 供 `observations/` 与共享 proxy 构建消费

这是 observations 层构建 actor_obs 和 critic_obs 的全部数据来源。

**机器人本体**

| 量 | 维度 | 来源 API | 说明 |
|---|---|---|---|
| 左臂关节位置 `q_left` | (6,) | `ArticulationView.get_joint_positions()` | 6 个旋转关节角度 |
| 左臂关节速度 `dq_left` | (6,) | `ArticulationView.get_joint_velocities()` | 角速度 |
| 左臂关节力矩 `tau_left` | (6,) | `ArticulationView.get_applied_joint_efforts()` | 当前施加力矩 |
| 右臂关节位置 `q_right` | (6,) | 同上 | |
| 右臂关节速度 `dq_right` | (6,) | 同上 | |
| 右臂关节力矩 `tau_right` | (6,) | 同上 | |

**末端执行器刚体状态（左/右臂对称提供）**

| 量 | 维度 | 来源 API | 说明 |
|---|---|---|---|
| 位置 `p` | (3,) | `RigidPrimView.get_world_poses()` | base_link 坐标系下 |
| 朝向 `quat` | (4,) | 同上 | 四元数 (w,x,y,z) |
| 线速度 `v` | (3,) | `RigidPrimView.get_velocities()` | base_link 坐标系下 |
| 角速度 `ω` | (3,) | 同上 | |
| 线加速度 `a` | (3,) | Isaac Sim / Isaac Lab 原生 body / link acceleration 接口 | base_link 坐标系下 |
| 角加速度 `α` | (3,) | 同上 | |

**交互对象（仅训练时提供给 critic）**

| 量 | 维度 | 说明 |
|---|---|---|
| 门板位姿 | (7,) | pos(3) + quat(4)，`base_link` 坐标系 |
| 门铰链角度 / 角速度 | (1,) + (1,) | 铰链关节的精确状态 |
| 杯体位姿 | (7,) | pos(3) + quat(4)，`base_link` 坐标系 |
| 杯体线速度 / 角速度 | (3,) + (3,) | `base_link` 坐标系下的完整动力学状态 |

### 2.2 接触事件 → 供 `rewards/` 计算安全惩罚

`ContactMonitor` 从 Isaac Sim 的 Contact Sensor 接口中提取每步发生的接触信息，整理后提供给 rewards 层：

| 数据 | 说明 | 下游消费者 |
|---|---|---|
| 各 link 接触力向量 | 机器人每个 link 受到的净接触力 `(3,)`，主要用于阈值过滤和诊断统计 | `ContactMonitor` 内部聚合 / 调试监控 |
| 自碰撞标志 | 机器人任意两个 link 之间是否发生接触 | `safety_penalty.py` 中的自碰撞惩罚 $r_{\text{self}}$ |
| 杯体脱落标志 | 杯体与末端距离是否超过安全阈值 | `safety_penalty.py` 中的脱落惩罚 $r_{\text{drop}}$，同时触发 episode 终止 |

**为什么不让 rewards 层自己去读 Contact Sensor？** 因为 Isaac Sim 的原始接触数据频率极高、噪声大，需要先做阈值过滤和事件聚合。这些物理层面的预处理属于环境封装的职责，而非奖励计算的职责。当前奖励层只消费清洗后的布尔安全事件，不再直接惩罚接触力。

### 2.3 任务状态信号 → 供 `rewards/` 和 `training/`

| 信号 | 类型 | 说明 |
|---|---|---|
| `done` | bool | 当前 episode 是否结束（成功 / 失败 / 超时） |
| `success` | bool | 是否因为任务完成而终止 |
| `door_angle` | float | 门铰链当前角度 $\theta_t$，供 rewards 计算进展奖励 |
| `door_angle_prev` | float | 门铰链上一步角度 $\theta_{t-1}$，供 rewards 计算角度增量 |

这里需要明确区分两个阈值语义：按当前收敛目标，env / TaskManager 负责维护的是 **episode 成功终止阈值**，取 `1.57 rad`；reward 层自己的 success bonus 阈值则单独定义为 `1.2 rad`，不由 env 文档中的 `success` 字段代指。

### 2.4 上下文标记 → 供 `observations/` 和 `rewards/`

| 量 | 维度 | 说明 |
|---|---|---|
| `left_occupied` | (1,) | 左臂是否持杯（episode 开始时随机确定，局内不变） |
| `right_occupied` | (1,) | 右臂是否持杯 |

这两个标记同时供两个下游消费：observations 层将其打包进 `actor_obs["context"]`，rewards 层用它们决定是否激活稳定性惩罚/奖励的 mask $m_L$、$m_R$。

### 2.5 环境侧共享 stability proxy → 同时供 `observations/` 和 `rewards/`

在当前实现中，envs 不再把“稳定性量的语义解释”留给下游各自完成，而是统一构造左右臂共享的 `stability_proxy`：

| 字段 | 说明 |
|---|---|
| `tilt_xy` / `tilt` | 基于末端姿态与重力方向的几何投影 |
| `linear_velocity_norm` | 末端线速度模长 |
| `linear_acceleration` | 末端原生线加速度 |
| `angular_velocity_norm` | 末端角速度模长 |
| `angular_acceleration` | 末端原生角加速度 |
| `jerk_proxy` | 连续两步线加速度变化率 |
| `recent_acc_history` | 最近若干步线加速度模长窗口 |

其中：

- `observations/` 负责把这份 proxy 打包进 `actor_obs`
- `rewards/` 直接消费同一份 proxy 计算稳定性奖励
- 两者并列依赖 envs，而不是互相依赖

---

## 3. 环境从外部接收的数据

### 3.1 策略动作力矩 → 来自 `policy/`

在新版控制架构中，策略网络（Actor）输出的指令 $\mathbf{a}_t \in \mathbb{R}^{12}$ 被直接设定为物理级的绝对控制力矩（单位：$\text{N}\cdot\text{m}$）。

环境层在接收该指令后，执行**力矩直通（Direct Pass-through）**规则：

$$
\boldsymbol{\tau}_{\text{cmd}} = \mathbf{a}_t \quad (\text{s.t.} \quad \| \tau_{\text{cmd}}^{(i)} \| \le \tau_{\text{limit}}^{(i)})
$$

系统取消了在环境层计算重力补偿与量纲解算的逻辑，把抵抗刚体重力与摩擦扰动的隐式工作全部交由循环神经网络（RNN）去推理。envs 仅在数据注入 Isaac Sim 执行器之前，负责依据各关节的出厂物理极值（effort limit）对力矩幅值做安全硬截断，防止突破硬件承载能力。

### 3.2 域随机化参数 → 来自 `training/domain_randomizer.py`

每次 `reset()` 时，training 层会传入一组在该回合生效的物理参数：

| 参数 | 在 envs 中的落地方式 |
|---|---|
| `cup_mass` | 修改杯体刚体的质量属性 |
| `door_mass` | 修改门板刚体的质量属性 |
| `door_damping` | 修改门铰链的阻尼系数 |
| `base_pos` | 将机器人基座传送至新坐标 |

这些参数一旦注入，会实质性地改变当前回合的物理动力学——门变重了推起来更费劲，杯子变重了惯性更大更容易晃。从策略的角度看，每一局的环境都略有不同。

> **重要**：这些域随机化参数在注入物理引擎的同时，也必须存入环境状态供后续读取。`observations/critic_obs_builder.py` 会读取它们作为 critic 的 privileged information。actor 则完全不知道这些数值。

---

## 4. 内部组件分工

### 4.1 `scene_factory.py` — 场景装配

负责在 episode 开始时搭建 Isaac Sim 场景：

- 加载机器人模型（Unitree Z1 双臂 + Dingo 底座）
- 根据当前课程阶段（curriculum stage）决定生成哪种类型的门（当前仅支持 push）
- 根据 `occupied` 上下文决定是否在末端生成杯体
- 应用域随机化参数（质量、阻尼、基座位置等）

### 4.2 `door_env.py` — 环境主体

继承 `BaseEnv`，实现核心的 `reset()` / `step()` 循环：

- `reset()`：调用 SceneFactory 重建场景，初始化所有状态，返回初始观测
- `step(action)`：将动作映射为力矩，推进物理仿真一个 $\Delta t$，收集新状态并组装共享 `stability_proxy`，调用 ContactMonitor 和 TaskManager，返回观测、奖励数据源、done 信号

### 4.3 `task_manager.py` — 任务进度状态机

维护单个 episode 内的任务进度判定逻辑，决定 done 信号的触发：

按当前目标定义，TaskManager 使用的是**episode 成功终止阈值**，而不是奖励 bonus 阈值。

**成功终止**：门铰链角度达到终止目标

$$
\theta_d \geq \theta_{\text{episode\_success}}
$$

**失败终止（杯体脱落）**：杯体与持握末端的欧氏距离超过容差

$$
\| \mathbf{p}_{\text{cup}} - \mathbf{p}_{\text{ee}} \|_2 > \epsilon_{\text{drop}}
$$

**超时终止**：episode 步数达到上限 $T_{\max}$。

### 4.4 `contact_monitor.py` — 接触事件汇总

每步从 Isaac Sim 的 Contact Sensor 或 Rigid Body View 拉取原始接触数据，做阈值过滤后输出结构化摘要：

- 过滤低于力阈值 $f_{\text{thresh}}$ 的微小接触噪声
- 按 link 归类接触力大小
- 检测自碰撞（同一 articulation 内的 link pair）
- 检测杯体是否脱离夹爪

---

## 5. 并行化：VecEnv 封装

训练时需要同时运行上千个环境实例来加速数据收集。envs 层需要提供向量化接口，使 `training/rollout_collector.py` 能够批量调用：

| 接口 | 输入 | 输出 |
|---|---|---|
| `reset()` | 域随机化参数 | `actor_obs_list`, `critic_obs_list` |
| `step(actions)` | `(n_envs, 12)` 动作矩阵 | `actor_obs_list`, `critic_obs_list`, `rewards (n_envs,)`, `dones (n_envs,)`, `infos` |

其中每个 `_obs_list` 是长度为 `n_envs` 的列表，每个元素是单个环境的原始观测字典（observations 层从中提取并构建最终的 actor/critic 张量）。

Isaac Sim 原生支持基于 GPU 的批量仿真（`Cloner` + `ArticulationView`），可以在单个 GPU 上并行运行数千环境实例而无需多进程。

---

## 6. 环境参数一览

### 6.1 仿真参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| $\Delta t$ | 物理仿真步长 | 1/120 s |
| 控制频率 | 策略决策频率（每 N 个物理步执行一次策略） | 1/60 s（decimation=2） |
| $T_{\max}$ | 单 episode 最大步数 | 500 |

### 6.2 任务判定参数（目标定义）

| 参数 | 含义 | 默认值 |
|---|---|---|
| $\theta_{\text{episode\_success}}$ | episode 成功终止角度阈值 | 1.57 rad（约 90°） |
| $\epsilon_{\text{drop}}$ | 杯体脱落检测距离 | 0.15 m |
| $f_{\text{thresh}}$ | 接触力过滤阈值 | 0.1 N |

说明：`1.2 rad` 不再作为 env 终止判定阈值记录在本层，它属于 reward 层的 success bonus 触发角度，应在 `rewards/README.md` 中单独维护。

### 6.3 动作控制参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| $\tau_{\text{limit}}$ | 关节绝对力矩安全限幅（硬截断阈值） | 从机械臂 URDF $effort\_limit$ 参数动态读取 |

---

## 7. 关键设计决策

### 为什么 envs 现在要统一构建 stability proxy？

因为稳定性相关量已经被 `observations/` 与 `rewards/` 共同消费。如果仍让两边各自推导，就会再次出现“reward 吃到的是速度伪装成加速度、observations 吃到的是另一套语义”的分裂。把 proxy 的语义源头收敛到 envs，可以保证：

- 加速度直接来自 Isaac Sim / Isaac Lab 原生接口
- `tilt` / `tilt_xy` 的几何定义在系统内只有一份
- `jerk_proxy` 与历史窗口只维护一套状态
- `observations/` 与 `rewards/` 并列消费同一份数据

这类最小必要的共享建模属于环境封装的一部分，而不是策略特征工程。

### 为什么 envs 不直接计算奖励？

同样是为了解耦。奖励函数的权重调整、动态缩放（$s_t$ 退火）和分项组合是实验迭代最频繁的部分。如果奖励逻辑嵌入环境，每一次调权重都要改环境代码，风险高、排查难。envs 只负责提供精确的物理真值，rewards 层自己决定怎么打分。

### 为什么杯体脱落既是 contact_monitor 的事也是 task_manager 的事？

ContactMonitor 负责检测脱落**事实**（距离超限），TaskManager 负责将这个事实转化为 episode 级的**终止决策**和 done 信号。前者是物理层面的测量，后者是逻辑层面的裁定。

### 为什么域随机化参数由 envs 落实而非由 training 直接注入物理引擎？

training 层不应该知道 Isaac Sim 的具体 API。它只知道"我要采样一个杯子质量 0.35 kg"，至于怎么在仿真器里改这个值、改完之后需不需要重新初始化惯性张量，那是 envs 层的封装职责。这样如果未来换仿真器（比如换成 MuJoCo），只需要重写 envs 层，training 层完全不用动。
