# Affordance-Guided Interaction 训练管线技术文档

---

## 0. 文档定位与适用范围

本文档是 Affordance-Guided Interaction 项目的训练系统技术规格说明书（Technical Specification），面向需要系统性理解项目整体架构、训练闭环机制与各层协作关系的工程与研究人员。

文档以训练管线为主线，按"全局架构 → 模块职责 → 数学形式化 → 数据流与接口契约"的层次组织。各节在阐述模块功能的同时，给出关键的数学定义与算法伪代码，使读者能够建立对系统行为的精确理解。子模块的实现细节（逐行注释、API 参考等）由各模块 README 承担；本文聚焦于跨模块的协作方式与系统级行为。

**关联文档**：
- 奖励函数完整推导：[envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)
- 训练算法形式化：[training/README.md](../src/affordance_guided_interaction/training/README.md)
- 环境层实现：[envs/README.md](../src/affordance_guided_interaction/envs/README.md)
- 观测层语义规范：[observations/README.md](../src/affordance_guided_interaction/observations/README.md)
- 策略网络架构：[policy/README.md](../src/affordance_guided_interaction/policy/README.md)
- 域随机化参数表：[docs/randomization.md](./randomization.md)
- TensorBoard 指标释义：[docs/tensorboard_guide.md](./tensorboard_guide.md)

除非特别注明，本文以当前默认实现（default training path）为准。

---i

## 1. 项目概述与任务定义

### 1.1 任务描述

本项目研究的核心任务是在 Isaac Lab GPU 并行仿真环境中，训练一台双臂移动机器人（Dingo 底座 + 双 Unitree Z1 机械臂）在持杯约束下完成推门动作。形式化地，任务可表述为受限双臂操作问题：

给定机器人状态 $\mathbf{s}_t \in \mathcal{S}$、门体状态 $\mathbf{g}_t$、持杯上下文 $c \in \{\text{none}, \text{left\_only}, \text{right\_only}, \text{both}\}$，策略 $\pi_\theta$ 需输出连续力矩 $\mathbf{a}_t \in \mathbb{R}^{12}$，使得门铰链角度 $\theta_t$ 单调递增至目标值 $\theta^* = \frac{\pi}{2}$，同时满足持杯稳定性约束。

### 1.2 问题特征

该任务具有以下关键性质：

**接触丰富的时序控制问题**。门体动力学未知（质量、阻尼由域随机化采样），策略必须通过交互历史进行在线辨识。设门铰链动力学为 $\dot{\theta} = f(\tau_{\text{ext}}, m_{\text{door}}, d_{\text{hinge}}, \theta)$，其中 $f$ 对策略不可直接观测，必须通过 RNN 隐状态 $h_t$ 间接推断。

**约束驱动的双臂协调问题**。在四种正式上下文 $c \in \mathcal{C}$ 下，策略需在统一动作空间 $\mathbb{R}^{12}$ 中学会差异化的动作模式，而非依赖硬编码角色分配。持杯侧（$m_{\text{occupied}} = 1$）需同时满足稳定性约束，自由侧（$m_{\text{occupied}} = 0$）负责主动施力。

**部分可观测性（POMDP）**。策略无法直接观测门板质量 $m_{\text{door}} \sim \mathcal{U}(5.0, 20.0)$、门铰链阻尼 $d_{\text{hinge}} \sim \mathcal{U}(0.5, 5.0)$、杯体质量 $m_{\text{cup}} \sim \mathcal{U}(0.1, 0.8)$ 等隐藏参数，构成典型的 POMDP 结构。此性质决定了 Actor 必须采用循环结构以积累时序信息。

**Sim-to-Real 鲁棒性要求**。域随机化覆盖质量、阻尼、基座位姿、动作噪声与观测噪声五个维度，策略需在参数扰动下保持行为稳定。

### 1.3 默认路径的设计取舍

当前默认训练路径采用 6 维结构化门几何信号 $\mathbf{d}_t \in \mathbb{R}^6$（`base_link` 坐标系下的门叶中心坐标 $\mathbf{p}_c \in \mathbb{R}^3$ 与门叶法向量 $\mathbf{n} \in \mathbb{R}^3$）作为门相关输入，由仿真 ground truth 直接计算，不经过视觉感知模型。此取舍将问题定义为"基于结构化低维几何信号的双臂受约束推门控制"，而非"端到端视觉控制"。

### 1.4 系统目标

项目训练系统需满足以下工程目标：

1. 在统一策略下覆盖四种正式上下文 $c \in \mathcal{C}$。
2. 在域随机化扰动下保持训练稳定性与执行鲁棒性。
3. 维持 Actor-Critic 信息边界：训练期使用 privileged information，部署期不依赖仿真 oracle。
4. 提供完整的监控、日志、checkpoint、导出与可视化工程链路。

---

## 2. 仓库结构与模块职责

```
Affordance-Guided-Interaction/
├── scripts/          # 训练、评估、导出、场景调试入口脚本
├── configs/          # YAML 配置体系（7 份配置文件）
├── assets/           # USD 资产（机器人、门、杯体、房间）
├── src/affordance_guided_interaction/
│   ├── envs/         # GPU 批量并行仿真环境（DirectRLEnv）
│   ├── observations/ # 观测层语义规范与通用工具
│   ├── policy/       # Actor-Critic 网络与动作头
│   ├── training/     # Rollout、Buffer、PPO、课程、随机化
│   ├── visualization/# Rollout artifact 配置与输出
│   ├── utils/        # 运行时启动、路径、环境检测
│   └── door_perception/ # [历史] 视觉感知实验模块
├── docs/             # 设计文档
├── tests/            # 回归测试
└── configs/          # 参数配置
```

### 模块职责矩阵

| 模块 | 职责 | 默认路径参与度 |
|------|------|---------------|
| `scripts/train.py` | 装配并驱动完整训练循环 | 核心入口 |
| `configs/` | 统一管理训练、环境、策略、任务、课程、奖励与可视化参数 | 配置来源 |
| `assets/` | USD 物理资产声明 | 底层资源 |
| `envs/` | GPU batched 仿真环境：物理交互、观测构建、奖励计算、终止判定 | 核心 |
| `observations/` | 观测层结构边界与语义规范定义 | 接口规范层 |
| `policy/` | Actor-Critic 网络定义与前向推理 | 核心 |
| `training/` | 轨迹采集、GAE 估计、PPO 更新、课程推进、随机化调度 | 核心 |
| `visualization/` | Rollout artifact（视频、帧图）生成 | 输出链路 |
| `door_perception/` | [历史] RGB-D → Point-MAE 视觉感知管线 | 非默认路径 |

---

## 3. 系统架构

系统的六层架构遵循单向依赖原则：上层消费下层提供的能力，下层不反向依赖上层的训练细节。

```text
┌──────────────────────────────────────────────────────┐
│                 监控与输出层                           │
│  TensorBoard · 控制台日志 · Checkpoint · 导出脚本       │
├──────────────────────────────────────────────────────┤
│                 训练优化层                             │
│  PPOTrainer · RolloutBuffer · RolloutCollector        │
│  CurriculumManager · DomainRandomizer                 │
├──────────────────────────────────────────────────────┤
│                 策略网络层                             │
│  Actor (多分支编码器 + RNN + ActionHead)               │
│  Critic (多分支编码器 + privileged编码器 + MLP)         │
├──────────────────────────────────────────────────────┤
│                 观测与状态表示层                        │
│  Actor Obs (96D) · Critic Obs (109D) · privileged (13D)│
├──────────────────────────────────────────────────────┤
│                 仿真环境层                             │
│  DoorPushEnv (DirectRLEnv) · DirectRLEnvAdapter       │
├──────────────────────────────────────────────────────┤
│                 资源与配置层                           │
│  DoorPushEnvCfg · DoorPushSceneCfg · 7×YAML · USD     │
└──────────────────────────────────────────────────────┘
```

**数据流方向**：

```text
configs/ + assets/
      │
      ▼
DoorPushEnvCfg ──► DoorPushEnv (DirectRLEnv)
      │                 │
      │    ┌────────────┼──────────────┐
      │    │            │              │
      │    ▼            ▼              ▼
      │  观测构建    动作执行       奖励/done
      │  (96D/109D)  (12D torque)   (标量+分项)
      │    │            │              │
      ▼    ▼            ▼              ▼
DirectRLEnvAdapter ──► RolloutCollector
                         │
                         ▼
                    RolloutBuffer
                         │
                         ▼
                    PPOTrainer ──► Actor/Critic 参数更新
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       TensorBoard/日志          Checkpoint/导出
```

---

## 4. 资源与配置体系

### 4.1 物理资产

默认训练路径依赖以下物理资产，均位于 `assets/` 目录下，由 `DoorPushSceneCfg` 以声明式 `@configclass` 引用：

| 资产 | USD 路径 | 关键属性 |
|------|---------|---------|
| 双臂机器人 | `assets/robot/usd/uni_dingo_dual_arm.usd` | 12 个臂关节 + 4 个轮关节 + 2 个云台关节；臂关节力矩上限 joint2=60 N·m，其余=30 N·m |
| 轻量机器人 | `assets/robot/usd/uni_dingo_lite.usd` | 移除轮子、云台、支架；固定底座；保留自碰撞 |
| 推门 | `assets/minimal_push_door/solid_push_door.usda` | 单铰链关节；默认阻尼 2.0 N·m·s/rad |
| 杯体 | `assets/grasp_objects/cup/carry_cup.usda` | 默认质量 0.3 kg（域随机化覆盖范围 0.1–0.8 kg） |
| 房间 | `assets/minimal_push_door/room_shell.usda` | 碰撞几何边界（当前统一配置保留字段 `room=None`，默认不实例化） |

环境配置类为 `DoorPushEnvCfg`，作为唯一的场景配置入口。当前统一配置采用轻量机器人场景：移除 wheels / pan-tilt、固定 root、保留自碰撞，并保留 `room` 配置字段但默认设为 `None`；基座位姿在 reset 时按扇形环随机采样。

### 4.2 配置文件体系

训练过程统一加载 7 份 YAML 配置文件：

| 配置文件 | 核心参数 | 消费者 |
|---------|---------|--------|
| `configs/training/default.yaml` | `total_steps`, `n_steps_per_rollout`, PPO 超参数, profile 系统 | `train.py`, `PPOTrainer` |
| `configs/env/default.yaml` | `physics_dt = 1/120$ s, `decimation = 2` | `build_env_cfg()` |
| `configs/policy/default.yaml` | `rnn_hidden = 512`, `rnn_layers = 1`, `log_std_init = -0.5` | `build_models()` |
| `configs/task/default.yaml` | `door_angle_target = 1.57` rad, `cup_drop_threshold = 0.15` m | `DoorPushEnvCfg` |
| `configs/curriculum/default.yaml` | `initial_stage`, `window_size = 50`, `threshold = 0.8` | `CurriculumManager` |
| `configs/reward/default.yaml` | 奖励权重 20 项（详见 §8.3） | `_inject_reward_params()` → `DoorPushEnvCfg` |
| `configs/visualization/default.yaml` | rollout 可视化参数 | `rollout_demo.py` |

**训练 profile 系统**：`training/default.yaml` 内置 5 档可扩展 profile，通过 `num_envs` 选择：

| Profile | `num_envs` | `n_steps` | `actor_lr` | `mini_batches` | `epochs` | `seq_length` |
|---------|-----------|-----------|-----------|----------------|----------|-------------|
| `env_256` | 256 | 64 | 3×10⁻⁴ | 8 | 4 | 16 |
| `env_512` | 512 | 64 | 3×10⁻⁴ | 16 | 4 | 16 |
| `env_1024` | 1024 | 64 | 2.5×10⁻⁴ | 16 | 3 | 16 |
| `env_2048` | 2048 | 64 | 2×10⁻⁴ | 32 | 3 | 16 |
| `env_4096` | 4096 | 16 | 2×10⁻⁴ | 16 | 3 | 16 |

### 4.3 运行时配置解析

`training/default.yaml` 中的字段分为两类：

1. **运行时设置**：由 `resolve_train_runtime_config()` 提取为 `TrainRuntimeConfig` 数据类，包含 `headless`、`device`、`seed`、`resume`、`log_dir`、`ckpt_dir`、`num_envs`。

2. **训练过程参数**：保留在合并后的 `cfg` 字典中，由 `train.py` 直接读取，包含 `total_steps`、`n_steps_per_rollout`、`ppo.*`、`log_interval`、`checkpoint_interval` 等。

### 4.4 奖励参数注入机制

奖励超参数的默认来源为独立的 `configs/reward/default.yaml`。`train.py` 在构建环境配置时通过 `_inject_reward_params()` 将 `task`、`stability` 和 `safety` 三组参数覆盖写入 `DoorPushEnvCfg`。此设计实现了环境逻辑（通过 `cfg.rew_*` 字段计算奖励）与奖励设计（通过 YAML 管理）的解耦。

---

## 5. 仿真环境层

`envs/` 是唯一与 Isaac Lab 物理引擎直接交互的层。默认环境实现为 `DoorPushEnv`，以 Isaac Lab 的 `DirectRLEnv` 为基类，在单块 GPU 上批量并行运行 $N$ 个环境实例。

### 5.1 场景声明与 Cloner 机制

场景由 `DoorPushSceneCfg` 以声明式定义。机器人、门、杯体、地面、照明等对象注册为 `InteractiveSceneCfg` 的配置条目，通过 `{ENV_REGEX_NS}` 占位符交由 Isaac Lab Cloner 自动复制至 `/World/envs/env_0`、`env_1`、...、`env_{N-1}` 子树。

Cloner 在 GPU 批量语义下统一完成场景实例化，无需显式编写逐环境加载逻辑。因此 `door_push_env.py` 中的 `_setup_scene()` 几乎为空——场景拓扑已由配置声明提前固定。

### 5.2 DoorPushEnv 核心接口

`DoorPushEnv` 作为 `DirectRLEnv` 的子类，实现以下六个核心接口方法：

| 方法 | 职责 | 输入 → 输出 |
|------|------|-------------|
| `_setup_scene()` | 注册场景 prim 路径索引 | — |
| `_reset_idx(env_ids)` | 选择性环境重置、域随机化采样、持杯初始化 | `env_ids: Tensor` → 更新仿真状态 |
| `_pre_physics_step(actions)` | 力矩裁剪、噪声注入、写入关节目标 | `actions: (N, 12)` → 物理引擎 |
| `_get_observations()` | 构造 Actor/Critic 观测张量 | — → `obs_dict: {actor: (N,96), critic: (N,109)}` |
| `_get_rewards()` | 计算任务奖励、稳定性奖励、安全惩罚 | — → `rewards: (N,)`, `extras["reward_info"]` |
| `_get_dones()` | 判定成功、掉杯、超时 | — → `terminated: (N,)`, `truncated: (N,)`, `extras` |

**关键工程约束**：上述所有方法均操作 `(N, ...)` 批量张量，不存在逐环境的 Python 循环。中间计算结果（末端加速度、tilt proxy、`cup_dropped`）在 `_get_observations()` 中缓存，供后续 `_get_rewards()` 和 `_get_dones()` 复用，避免重复计算。

### 5.3 动作执行与噪声注入

`_pre_physics_step()` 中的动作处理流程为：

$$\tilde{\mathbf{a}}_t = \text{clip}(\mathbf{a}_t,\; -\tau_{\max},\; \tau_{\max})$$

$$\hat{\mathbf{a}}_t = \text{clip}\!\left(\tilde{\mathbf{a}}_t + \boldsymbol{\epsilon}_a,\; -\tau_{\max},\; \tau_{\max}\right), \quad \boldsymbol{\epsilon}_a \sim \mathcal{N}(\mathbf{0},\; \sigma_a^2 \mathbf{I}_{12})$$

其中 $\boldsymbol{\tau}_{\max} \in \mathbb{R}^{12}$ 为 per-joint 力矩上限向量，$\sigma_a = 0.02$ 为动作噪声标准差。力矩上限来自 Z1 URDF 硬件规格，按 `ARM_JOINT_NAMES` 顺序排列：

$$\boldsymbol{\tau}_{\max} = (30, 60, 30, 30, 30, 30,\; 30, 60, 30, 30, 30, 30) \text{ N·m}$$

其中 joint2（肩关节）上限为 60 N·m，其余 5 个关节上限均为 30 N·m，双臂对称。`DoorPushSceneCfg` 中 actuator 配置按此拆分为 `shoulder_joints`（60 N·m）和 `arm_joints`（30 N·m）两组。裁剪执行两次：第一次移除策略输出的超限分量，第二次保证噪声注入后仍处于物理安全范围。

### 5.4 观测构建的数学过程

#### 5.4.1 坐标系变换

所有几何量均在 `base_link` 坐标系下表达，以消除对世界坐标的依赖。设机器人基座在世界系下的位姿为 $(\mathbf{p}_b, \mathbf{q}_b)$，对应旋转矩阵为 $\mathbf{R}_b \in SO(3)$，则任意世界系向量 $\mathbf{v}^W$ 到 base 系的变换为：

$$\mathbf{v}^B = \mathbf{R}_b^\top (\mathbf{v}^W - \mathbf{p}_b)$$

此变换应用于末端执行器位置、速度和门几何信息。

#### 5.4.2 末端加速度的数值微分

末端线加速度与角加速度通过一阶后向差分计算：

$$\mathbf{a}_t = \frac{\mathbf{v}_t - \mathbf{v}_{t-1}}{\Delta t_{\text{ctrl}}}, \quad \boldsymbol{\alpha}_t = \frac{\boldsymbol{\omega}_t - \boldsymbol{\omega}_{t-1}}{\Delta t_{\text{ctrl}}}$$

其中 $\Delta t_{\text{ctrl}} = \text{physics\_dt} \times \text{decimation} = \frac{1}{120} \times 2 = \frac{1}{60}$ s 为控制步长。

#### 5.4.3 Tilt Proxy 计算

持杯稳定性信号 tilt 的计算方式为：将世界系重力向量 $\mathbf{g}^W = (0, 0, -9.81)^\top$ 变换至末端执行器局部坐标系，取其水平分量之范数：

$$\mathbf{g}^{\text{local}} = \mathbf{R}_{\text{EE}}^\top \mathbf{g}^W$$

$$\text{tilt} = \left\| P_{xy}(\mathbf{g}^{\text{local}}) \right\| = \sqrt{(g_x^{\text{local}})^2 + (g_y^{\text{local}})^2}$$

物理含义：当末端执行器保持水平时，$\mathbf{g}^{\text{local}} = (0, 0, -g)^\top$，tilt $= 0$；倾斜越大，tilt 值越大。

#### 5.4.4 门几何观测计算

6 维 `door_geometry` 信号的计算过程为：

1. 获取门叶 body 在世界系下的位姿 $(\mathbf{p}_d^W, \mathbf{q}_d^W)$，构造旋转矩阵 $\mathbf{R}_d$。
2. 计算门叶中心在世界系下的坐标：$\mathbf{c}^W = \mathbf{p}_d^W + \mathbf{R}_d \cdot \mathbf{o}_{\text{center}}$，其中 $\mathbf{o}_{\text{center}}$ 为门叶几何中心在门叶局部系下的固定偏移。
3. 计算门叶法向量在世界系下：$\mathbf{n}^W = \mathbf{R}_d \cdot \mathbf{n}_{\text{local}}$。
4. 变换至 base 系：$\mathbf{c}^B = \mathbf{R}_b^\top(\mathbf{c}^W - \mathbf{p}_b)$，$\mathbf{n}^B = \mathbf{R}_b^\top \mathbf{n}^W$。

最终输出 $\mathbf{d}_t = [\mathbf{c}^B;\; \mathbf{n}^B] \in \mathbb{R}^6$。

### 5.5 DirectRLEnvAdapter

`DirectRLEnvAdapter` 将 `DoorPushEnv` 的 batch tensor 输出包装为训练侧可稳定消费的接口。其核心功能是将 pre-reset 的 `extras` 重建为 per-env `info` 字典，使训练层能获取"上一 episode 真实结束状态"而非已 reset 的新 episode 初始状态。

### 5.6 批量环境中的 Reset 语义

默认训练路径在场景创建时预生成所有资产（含左右杯体），运行时不反复创建或删除。Reset 语义为"固定拓扑下的状态重写"：

- 不需要持杯的环境：杯体 teleport 至远处 $(100, 0, 0)$。
- 需要持杯的环境：`_batch_cup_grasp_init()` 直接将关节写入预设抓取姿态，并将杯体 teleport 至夹爪对应位置。

此纯状态写入方式避免在部分环境 reset 时调用 `sim.step()` 影响其他并行环境，是 GPU batched 设计中的关键工程约束。

---

## 6. 观测层：状态表示契约

观测层定义了环境与策略之间的正式数据契约。虽然观测构建逻辑在当前实现中已直接并入 `DoorPushEnv._get_observations()`，但观测层的结构边界和语义规范仍然是系统设计的核心组成部分。

### 6.1 Actor 观测

当前 Actor 观测总维度为 $\dim(\mathbf{o}_t^{\text{actor}}) = 96$，由五个分支拼接而成：

$$\mathbf{o}_t^{\text{actor}} = [\mathbf{o}_t^{\text{proprio}};\; \mathbf{o}_t^{\text{ee}};\; \mathbf{o}_t^{\text{ctx}};\; \mathbf{o}_t^{\text{stab}};\; \mathbf{o}_t^{\text{geom}}]$$

| 分支 | 维度 | 组成 | 数学表示 |
|------|------|------|---------|
| `proprio` | 48 | 关节位置 $\mathbf{q} \in \mathbb{R}^{12}$、关节速度 $\dot{\mathbf{q}} \in \mathbb{R}^{12}$、关节力矩 $\boldsymbol{\tau} \in \mathbb{R}^{12}$、上一步动作 $\mathbf{a}_{t-1} \in \mathbb{R}^{12}$ | $[\mathbf{q};\; \dot{\mathbf{q}};\; \boldsymbol{\tau};\; \mathbf{a}_{t-1}]$ |
| `ee` | 38 | 左右末端位置 $\mathbf{p} \in \mathbb{R}^3$、姿态 $\mathbf{q}_{\text{quat}} \in \mathbb{R}^4$（仅左臂）、线速度 $\mathbf{v} \in \mathbb{R}^3$、角速度 $\boldsymbol{\omega} \in \mathbb{R}^3$、线加速度 $\mathbf{a} \in \mathbb{R}^3$、角加速度 $\boldsymbol{\alpha} \in \mathbb{R}^3$ | $[\mathbf{p}_L;\; \mathbf{q}_L;\; \mathbf{v}_L;\; \boldsymbol{\omega}_L;\; \mathbf{a}_L;\; \boldsymbol{\alpha}_L;\; \mathbf{p}_R;\; \mathbf{v}_R;\; \boldsymbol{\omega}_R;\; \mathbf{a}_R;\; \boldsymbol{\alpha}_R]$ |
| `context` | 2 | 左臂占用标志 $m_L$、右臂占用标志 $m_R$ | $[m_L;\; m_R] \in \{0,1\}^2$ |
| `stability` | 2 | 左侧 tilt $\text{tilt}_L$、右侧 tilt $\text{tilt}_R$ | $[\text{tilt}_L;\; \text{tilt}_R]$ |
| `door_geometry` | 6 | `base_link` 系下门叶中心 $\mathbf{c}^B \in \mathbb{R}^3$、门叶法向量 $\mathbf{n}^B \in \mathbb{R}^3$ | $[\mathbf{c}^B;\; \mathbf{n}^B]$ |

**观测噪声**：仅注入至 `proprio` 分支中的关节位置和关节速度（共 24 维）：

$$\hat{\mathbf{q}} = \mathbf{q} + \boldsymbol{\epsilon}_q, \quad \hat{\dot{\mathbf{q}}} = \dot{\mathbf{q}} + \boldsymbol{\epsilon}_{\dot{q}}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\; \sigma_o^2 \mathbf{I})$$

其中 $\sigma_o = 0.01$。Critic 观测使用无噪声的真实状态（非对称 Actor-Critic 设计）。

**设计要点**：

1. **Base-relative 表达**：所有几何量在 `base_link` 坐标系下表达，策略看到的是相对几何关系而非绝对世界坐标，提升对基座位姿扰动的鲁棒性。
2. **Tilt 独立性**：`stability` 分支仅包含 tilt 标量，不与速度/加速度混杂；后者已在 `ee` 分支中完整表达。
3. **门信号纯几何化**：`door_geometry` 来自仿真 ground truth 的几何计算，不经过视觉感知模型，不依赖相机更新频率。

### 6.2 Critic 观测与 Privileged Information

当前 Critic 观测总维度为 $\dim(\mathbf{o}_t^{\text{critic}}) = 109$，结构为：

$$\mathbf{o}_t^{\text{critic}} = [\mathbf{o}_t^{\text{actor,\,clean}};\; \boldsymbol{\psi}_t]$$

其中 $\mathbf{o}_t^{\text{actor,\,clean}}$ 为 Actor 观测的无噪声版本（$\sigma_o = 0$），$\boldsymbol{\psi}_t \in \mathbb{R}^{13}$ 为 privileged 信息：

| Privileged 信号 | 维度 | 含义 |
|----------------|------|------|
| `door_pose` | 7 | 门叶在世界系下的位姿 $(\mathbf{p}, \mathbf{q}_{\text{quat}})$ |
| `door_joint_pos` | 1 | 门铰链角度 $\theta_t$ |
| `door_joint_vel` | 1 | 门铰链角速度 $\dot{\theta}_t$ |
| `cup_mass` | 1 | 当前 episode 的杯体质量 $m_{\text{cup}}$ |
| `door_mass` | 1 | 当前 episode 的门板质量 $m_{\text{door}}$ |
| `door_damping` | 1 | 当前 episode 的门铰链阻尼 $d_{\text{hinge}}$ |
| `cup_dropped` | 1 | 掉杯事件标志 $\mathbb{1}[\text{cup\_dropped}]$ |

**信息边界的设计意义**：Actor 无法直接观测隐藏动力学参数 $(m_{\text{door}}, d_{\text{hinge}}, m_{\text{cup}})$ 和门的真实铰链状态 $(\theta_t, \dot{\theta}_t)$，必须通过交互历史间接推断。Critic 则利用这些额外信息降低价值估计的方差，加速训练收敛。此非对称设计确保部署时不依赖仿真 oracle。

---

## 7. 策略网络层

### 7.1 Actor 架构

Actor 采用多分支编码器 + 循环主干 + 高斯动作头的结构。形式化地，给定观测 $\mathbf{o}_t^{\text{actor}}$ 的五个分支，前向过程为：

**Step 1：分支编码**

$$\mathbf{f}_t^{\text{proprio}} = E_p(\mathbf{o}_t^{\text{proprio}}) \in \mathbb{R}^{d_p}, \quad \mathbf{f}_t^{\text{ee}} = E_e(\mathbf{o}_t^{\text{ee}}) \in \mathbb{R}^{d_e}$$

$$\mathbf{f}_t^{\text{stab}} = E_s(\mathbf{o}_t^{\text{stab}}) \in \mathbb{R}^{d_s}, \quad \mathbf{f}_t^{\text{geom}} = E_g(\mathbf{o}_t^{\text{geom}}) \in \mathbb{R}^{d_g}$$

其中各 $E_*$ 为轻量 MLP 编码器。`context` 分支因维度极低（2D）而直接拼接，不经过编码器。

**Step 2：特征融合**

$$\mathbf{f}_t = [\mathbf{f}_t^{\text{proprio}};\; \mathbf{f}_t^{\text{ee}};\; \mathbf{o}_t^{\text{ctx}};\; \mathbf{f}_t^{\text{stab}};\; \mathbf{f}_t^{\text{geom}}] \in \mathbb{R}^{d_f}$$

**Step 3：循环时序建模**

$$\mathbf{h}_t = \text{GRU}(\mathbf{f}_t,\; \mathbf{h}_{t-1})$$

默认配置下，`rnn_hidden = 512`，`rnn_layers = 1`，`rnn_type = gru`（可切换为 LSTM）。RNN 在此系统中的角色不是性能装饰，而是应对部分可观测性的必要结构：由于 Actor 无法直接观测隐藏参数，必须通过 GRU 隐状态 $\mathbf{h}_t \in \mathbb{R}^{512}$ 积累交互历史以间接推断。

**Step 4：动作采样**

$$\boldsymbol{\mu}_t, \boldsymbol{\sigma}_t = \text{Head}(\mathbf{h}_t)$$

$$\mathbf{a}_t \sim \boldsymbol{\pi}_\theta(\cdot | \mathbf{o}_t, \mathbf{h}_t) = \mathcal{N}(\boldsymbol{\mu}_t,\; \text{diag}(\boldsymbol{\sigma}_t^2))$$

其中 $\boldsymbol{\mu}_t \in \mathbb{R}^{12}$ 为均值向量，$\boldsymbol{\sigma}_t = \exp(\boldsymbol{\ell}) \in \mathbb{R}^{12}$ 为逐维标准差（$\boldsymbol{\ell}$ 为可学习参数，初始化为 $\ell_0 = -0.5$），$\mathbf{a}_t \in \mathbb{R}^{12}$ 为双臂 12 维连续力矩。

### 7.2 Critic 架构

Critic 使用与 Actor 对应的独立分支结构（不共享参数），并额外增加一个 `privileged` 编码器：

$$\mathbf{g}_t^{\text{actor}} = [E_p^c(\mathbf{o}_t^{\text{proprio}});\; E_e^c(\mathbf{o}_t^{\text{ee}});\; \mathbf{o}_t^{\text{ctx}};\; E_s^c(\mathbf{o}_t^{\text{stab}});\; E_g^c(\mathbf{o}_t^{\text{geom}})]$$

$$\mathbf{g}_t^{\text{priv}} = E_{\psi}(\boldsymbol{\psi}_t)$$

$$V_t = \text{MLP}([\mathbf{g}_t^{\text{actor}};\; \mathbf{g}_t^{\text{priv}}]) \in \mathbb{R}$$

Critic 默认不使用循环结构（因其已拥有完整状态信息），MLP 层配置为 $\{512, 256, 128, 1\}$。

### 7.3 动作语义与安全分工

策略输出始终定义为 12 维 raw torque $\mathbf{a}_t \in \mathbb{R}^{12}$。环境在执行前做硬裁剪至 $[-\tau_{\max}, \tau_{\max}]$。安全惩罚中的"力矩超限"以 raw action 为依据。此分工意味着：

- `policy/` 负责表达控制意图。
- `envs/` 负责执行约束和物理安全边界。
- 奖励函数负责将超限控制意图作为训练信号显式反馈给策略。

当前设计明确不使用动作 mask：即使某一侧处于持杯状态，该侧仍允许输出动作。策略必须在统一动作空间中通过 reward 和上下文约束学会差异化的动作模式。

### 7.4 时序语义

每个并行环境维护独立的 Actor 隐状态 $\mathbf{h}_t^{(i)}$：

- **Episode 内部**：隐状态连续传递 $\mathbf{h}_0 \to \mathbf{h}_1 \to \cdots \to \mathbf{h}_T$。
- **Episode 结束**：对应环境的隐状态被清零 $\mathbf{h}_0^{(i)} = \mathbf{0}$。

`RolloutCollector` 在采样阶段缓存每步的 hidden state；`RolloutBuffer` 在更新阶段将缓存切分为 TBPTT 所需的序列片段。

---

## 8. 奖励函数

本节给出奖励函数的完整数学定义。总奖励为任务奖励、双侧稳定性奖励与安全惩罚的组合：

$$r_t = r_t^{\text{task}} + m_L \cdot r_t^{\text{stab},L} + m_R \cdot r_t^{\text{stab},R} - r_t^{\text{safe}}$$

其中 $m_L, m_R \in \{0, 1\}$ 为左右臂持杯占用标志（occupancy mask），仅对持杯侧计算稳定性奖励。

### 8.1 任务奖励

任务奖励由角度增量奖励、一次性成功 bonus 与接近门板大表面的 shaping 项组成：

$$r_t^{\text{task}} = w(\theta_t) \cdot \Delta\theta_t + w_{\text{open}} \cdot \mathbb{1}[\theta_t \geq \theta_{\text{bonus}} \;\wedge\; \neg\text{already\_succeeded}] + \mathbb{1}[\theta_t < \theta_{\text{stop}}] \cdot w_{\text{approach}} \cdot r_{\text{approach}, t}$$

其中 $\Delta\theta_t = \theta_t - \theta_{t-1}$ 为门角度增量，接近门奖励定义为：

$$r_{\text{approach}, t} = \max\!\left(1 - \frac{a_t^2}{b^2 + \varepsilon},\; 0\right)$$

$$a_t = \min_{\mathbf{x} \in \mathcal{A}_t,\; \mathbf{y} \in \mathcal{D}_t} \|\mathbf{x} - \mathbf{y}\|_2, \qquad b = \min_{\mathbf{x} \in \mathcal{A}_0,\; \mathbf{y} \in \mathcal{D}_0} \|\mathbf{x} - \mathbf{y}\|_2$$

当前最小实现中，$\mathcal{A}_t$ 取左右 EE 控制点，$\mathcal{D}_t$ 取门板 `Panel` 的推门侧大矩形表面。

权重函数 $w(\theta_t)$ 的设计目的是在门接近完全打开后逐步衰减增量奖励，避免策略过度优化已完成的子目标：

$$w(\theta_t) = \begin{cases} w_\delta & \text{if } \theta_t \leq \theta_{\text{bonus}} \\ w_\delta \cdot \max\!\left(\alpha,\; 1 - k_{\text{decay}} \cdot (\theta_t - \theta_{\text{bonus}})\right) & \text{if } \theta_t > \theta_{\text{bonus}} \end{cases}$$

默认参数：$w_\delta = 10.0$，$\alpha = 0.3$，$k_{\text{decay}} = 0.5$，$w_{\text{approach}} = 200.0$，$\varepsilon = 10^{-6}$，$\theta_{\text{stop}} = 0.10$ rad。

### 8.2 稳定性奖励（每侧 7 项）

对每侧 $s \in \{L, R\}$，稳定性奖励由 7 个分量组成：

$$r_t^{\text{stab},s} = r_{\text{zero-acc}} + r_{\text{zero-ang}} + r_{\text{acc}} + r_{\text{ang}} + r_{\text{tilt}} + r_{\text{smooth}} + r_{\text{reg}}$$

**分量 1：零线性加速度奖励**（高斯核，鼓励加速度趋零）

$$r_{\text{zero-acc}} = w_{\text{zero-acc}} \cdot \exp\!\left(-\lambda_{\text{acc}} \cdot \|\mathbf{a}_t\|^2\right)$$

参数：$w_{\text{zero-acc}} = 1.0$，$\lambda_{\text{acc}} = 2.0$。

**分量 2：零角加速度奖励**（高斯核，鼓励角加速度趋零）

$$r_{\text{zero-ang}} = w_{\text{zero-ang}} \cdot \exp\!\left(-\lambda_{\text{ang}} \cdot \|\boldsymbol{\alpha}_t\|^2\right)$$

参数：$w_{\text{zero-ang}} = 0.5$，$\lambda_{\text{ang}} = 1.0$。

**分量 3：线性加速度惩罚**（二次型）

$$r_{\text{acc}} = -w_{\text{acc}} \cdot \|\mathbf{a}_t\|^2$$

参数：$w_{\text{acc}} = 0.5$。

**分量 4：角加速度惩罚**（二次型）

$$r_{\text{ang}} = -w_{\text{ang}} \cdot \|\boldsymbol{\alpha}_t\|^2$$

参数：$w_{\text{ang}} = 0.3$。

**分量 5：倾斜惩罚**（基于重力在末端局部系的水平投影）

$$r_{\text{tilt}} = -w_{\text{tilt}} \cdot \left\|P_{xy}\!\left(\mathbf{R}_{\text{EE}}^\top \mathbf{g}\right)\right\|^2$$

参数：$w_{\text{tilt}} = 0.3$。

**分量 6：平滑度惩罚**（力矩变化率的二次惩罚）

$$r_{\text{smooth}} = -w_{\text{smooth}} \cdot \|\boldsymbol{\tau}_t - \boldsymbol{\tau}_{t-1}\|^2$$

参数：$w_{\text{smooth}} = 0.1$。

**分量 7：力矩正则化**（鼓励小力矩输出）

$$r_{\text{reg}} = -w_{\text{reg}} \cdot \|\boldsymbol{\tau}_t\|^2$$

参数：$w_{\text{reg}} = 0.01$。

**设计说明**：分量 1–2 采用高斯核 $\exp(-\lambda \|\cdot\|^2)$，在零附近提供稠密梯度信号；分量 3–4 采用二次惩罚，在整个值域提供一致梯度；分量 5 直接惩罚杯体倾斜；分量 6–7 鼓励平滑、小幅度的力矩输出。

### 8.3 安全惩罚

安全惩罚由四项组成：

$$r_t^{\text{safe}} = r_{\text{joint\_limit}} + r_{\text{joint\_vel}} + r_{\text{torque\_limit}} + r_{\text{cup\_drop}}$$

**关节限位惩罚**：

$$r_{\text{joint\_limit}} = \beta_1 \sum_{i=1}^{12} \max\!\left(0,\; |q_i - q_i^c| - \mu \cdot \delta_i\right)^2$$

其中 $q_i^c = \frac{q_i^{\min} + q_i^{\max}}{2}$ 为关节限位中心，$\delta_i = \frac{q_i^{\max} - q_i^{\min}}{2}$ 为半范围，$\mu = 0.9$ 为触发比例。参数：$\beta_1 = 1.0$。

**关节速度惩罚**：

$$r_{\text{joint\_vel}} = \beta_2 \sum_{i=1}^{12} \max\!\left(0,\; |\dot{q}_i| - \mu \cdot \dot{q}_i^{\max}\right)^2$$

参数：$\beta_2 = 0.5$。

**力矩超限惩罚**（以 raw action 为依据，per-joint limits）：

$$r_{\text{torque\_limit}} = \beta_3 \sum_{i=1}^{12} \max\!\left(0,\; |a_i^{\text{raw}}| - \tau_{\max,i}\right)^2$$

其中 $\tau_{\max,i}$ 为第 $i$ 个关节的力矩上限（来自 Z1 URDF：joint2 为 60 N·m，其余为 30 N·m）。参数：$\beta_3 = 0.01$。

**掉杯惩罚**：

$$r_{\text{cup\_drop}} = w_{\text{drop}} \cdot \mathbb{1}[\text{cup\_dropped}]$$

参数：$w_{\text{drop}} = 100.0$。掉杯判定条件为杯体相对持杯末端偏移超过 `cup_drop_threshold = 0.15` m。

### 8.4 奖励参数总表

| 类别 | 参数 | 符号 | 默认值 |
|------|------|------|--------|
| 任务 | 角度增量权重 | $w_\delta$ | 10.0 |
| 任务 | 衰减下限比例 | $\alpha$ | 0.3 |
| 任务 | 衰减速率 | $k_{\text{decay}}$ | 0.5 |
| 任务 | 成功 bonus 权重 | $w_{\text{open}}$ | 50.0 |
| 任务 | bonus 触发角度 | $\theta_{\text{bonus}}$ | 1.2 rad |
| 稳定性 | 零加速度权重 | $w_{\text{zero-acc}}$ | 1.0 |
| 稳定性 | 加速度高斯衰减率 | $\lambda_{\text{acc}}$ | 2.0 |
| 稳定性 | 零角加速度权重 | $w_{\text{zero-ang}}$ | 0.5 |
| 稳定性 | 角加速度高斯衰减率 | $\lambda_{\text{ang}}$ | 1.0 |
| 稳定性 | 加速度惩罚系数 | $w_{\text{acc}}$ | 0.5 |
| 稳定性 | 角加速度惩罚系数 | $w_{\text{ang}}$ | 0.3 |
| 稳定性 | 倾斜惩罚系数 | $w_{\text{tilt}}$ | 0.3 |
| 稳定性 | 平滑度系数 | $w_{\text{smooth}}$ | 0.1 |
| 稳定性 | 正则化系数 | $w_{\text{reg}}$ | 0.01 |
| 安全 | 关节限位系数 | $\beta_1$ | 1.0 |
| 安全 | 触发比例 | $\mu$ | 0.9 |
| 安全 | 速度限位系数 | $\beta_2$ | 0.5 |
| 安全 | 力矩限位系数 | $\beta_3$ | 0.01 |
| 安全 | 掉杯惩罚 | $w_{\text{drop}}$ | 100.0 |

---

## 9. 训练优化与调度层

### 9.1 RolloutCollector

`RolloutCollector` 是训练循环与环境交互之间的主要驱动器。在每一轮 iteration 中，它在 $N$ 个并行环境中连续推进 $T$ 步（$T = \text{n\_steps\_per\_rollout}$），执行以下流程：

```
for t = 1, ..., T:
    1. 将当前观测转换成策略前向所需的张量分支
    2. Actor 前向：采样动作 a_t, 计算旧 log_prob π_old(a_t|o_t, h_t)
    3. Critic 前向：估计价值 V(s_t)
    4. 环境步进：将 a_t 送入环境，接收 o_{t+1}, r_t, d_t, info_t
    5. 写入 Buffer：存储 (o_t, a_t, log_prob_t, V_t, r_t, d_t, h_t)
    6. 对 done 环境清零 hidden state
    7. 聚合统计量（mean_reward, 成功率, 上下文成功率, reward 分项）

# Rollout 结束后
last_values = Critic(o_{T+1})   # Bootstrap value
```

**统计聚合**：collector 在 rollout 内对以下指标求均值并生成 `collect_stats`：

- `collect/mean_reward`：全部环境全部步的即时奖励均值。
- `collect/completed_episodes`、`collect/successful_episodes`：完成和成功的 episode 计数。
- `collect/episode_success_rate`：成功率 $\eta = \frac{N_{\text{success}}}{N_{\text{completed}}}$。
- `collect/success_none`、`success_left_only`、`success_right_only`、`success_both`：按上下文拆分的成功率。
- `reward/*`：一级奖励分量均值。

### 9.2 RolloutBuffer

`RolloutBuffer` 是固定容量的 on-policy 轨迹缓存。它预先分配形状为 $(T, N, \text{dim})$ 的张量，存储：

- Actor 各分支观测、privileged 观测
- 动作 $\mathbf{a}_t$、旧 log prob $\log \pi_{\theta_{\text{old}}}(\mathbf{a}_t)$、旧价值 $V_{\text{old}}(s_t)$
- 奖励 $r_t$、终止标志 $d_t$
- Hidden state $\mathbf{h}_t$（和 LSTM 的 cell state $\mathbf{c}_t$）

两个核心职责：

#### 9.2.1 GAE 优势估计

Generalized Advantage Estimation (GAE) 的反向递推计算：

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1}$$

其中 $\gamma = 0.99$ 为折扣因子，$\lambda = 0.95$ 为 GAE 偏差-方差权衡参数。Returns 通过优势与价值估计之和计算：

$$\hat{R}_t = \hat{A}_t + V(s_t)$$

对于 rollout 最后一帧（$t = T$），$V(s_{T+1})$ 使用 collector 提供的 bootstrap value `last_values`，$d_T$ 使用 `last_dones`。

#### 9.2.2 TBPTT 序列切分

为支持截断反向传播（Truncated Backpropagation Through Time），buffer 将完整轨迹按固定序列长度 $L$ 切分：

$$\text{num\_seqs\_per\_env} = \lfloor T / L \rfloor, \quad \text{total\_seqs} = \text{num\_seqs\_per\_env} \times N$$

每个序列片段包含：
- 起始步的 hidden state 作为 RNN 初始状态：$\mathbf{h}_0^{\text{seq}} \in \mathbb{R}^{1 \times B \times 512}$
- 长度为 $L$ 的观测、动作、优势、returns 序列

梯度仅在长度为 $L$ 的片段内传播，实现有界的内存消耗与稳定的长序列训练。

### 9.3 PPOTrainer

`PPOTrainer` 持有 Actor 和 Critic 各自独立的 Adam 优化器，在每一轮数据收集之后执行 $K$ epoch $\times$ $M$ mini-batch 的 PPO 更新。当前默认配置下 $K = 3$，$M = 16$。

#### 9.3.1 PPO 总损失函数

$$\mathcal{L}(\theta, \phi) = \mathcal{L}^{\text{clip}}(\theta) + c_v \cdot \mathcal{L}^{\text{value}}(\phi) - c_e \cdot \bar{\mathcal{H}}[\pi_\theta]$$

其中 $c_v = 0.5$ 为 Critic 损失权重，$c_e = 0.01$ 为熵正则化系数。

#### 9.3.2 Actor Loss（PPO-Clip）

概率比：

$$\rho_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t | \mathbf{o}_t, \mathbf{h}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t | \mathbf{o}_t, \mathbf{h}_t)} = \exp\!\left(\log \pi_\theta(\mathbf{a}_t) - \log \pi_{\theta_{\text{old}}}(\mathbf{a}_t)\right)$$

实现中对 log ratio 做数值裁剪 $\text{clamp}(\cdot, -20, 20)$ 以防止溢出。

Clipped surrogate 目标：

$$\mathcal{L}^{\text{clip}}(\theta) = -\frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \min\!\left(\rho_t \hat{A}_t,\; \text{clip}(\rho_t,\; 1-\epsilon,\; 1+\epsilon) \hat{A}_t\right)$$

其中 $\epsilon = 0.2$ 为裁剪参数。

**优势归一化**：更新前对 mini-batch 内的优势做标准化：

$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \bar{A}}{\text{std}(A) + 10^{-8}}$$

**诊断指标**：
- Clip fraction：$\frac{1}{|\mathcal{B}|} \sum_t \mathbb{1}[|\rho_t - 1| > \epsilon]$，正常应 $< 0.1{\sim}0.2$。
- Approximate KL：$\widehat{D}_{\text{KL}} = \mathbb{E}[(\rho_t - 1) - \log \rho_t]$，正常应 $< 0.02{\sim}0.05$。

#### 9.3.3 Critic Loss

标准形式为 MSE：

$$\mathcal{L}^{\text{value}}(\phi) = \frac{1}{2|\mathcal{B}|} \sum_{t \in \mathcal{B}} \left(V_\phi(s_t) - \hat{R}_t\right)^2$$

可选的 clipped value loss（默认启用）：

$$\bar{V}_\phi(s_t) = V_{\phi_{\text{old}}}(s_t) + \text{clip}\!\left(V_\phi(s_t) - V_{\phi_{\text{old}}}(s_t),\; -\epsilon_v,\; \epsilon_v\right)$$

$$\mathcal{L}^{\text{value, clip}}(\phi) = \frac{1}{2|\mathcal{B}|} \sum_{t \in \mathcal{B}} \max\!\left((V_\phi(s_t) - \hat{R}_t)^2,\; (\bar{V}_\phi(s_t) - \hat{R}_t)^2\right)$$

其中 $\epsilon_v = 0.2$。

#### 9.3.4 熵正则化

对于 12 维对角高斯策略，每个维度的微分熵为 $\frac{1}{2}\ln(2\pi e \sigma_i^2)$，总熵为：

$$\mathcal{H}[\pi_\theta] = \frac{1}{2} \sum_{i=1}^{12} \ln(2\pi e \sigma_i^2)$$

熵正则化项 $-c_e \cdot \bar{\mathcal{H}}$ 鼓励策略保持探索，防止过早收敛。

#### 9.3.5 梯度裁剪与学习率调度

全局梯度裁剪（按范数）：

$$\hat{\mathbf{g}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq g_{\max} \\ g_{\max} \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > g_{\max} \end{cases}$$

其中 $g_{\max} = 1.0$。Actor 和 Critic 各自独立执行梯度裁剪。

可选线性学习率衰减（从初始 $\eta_0$ 到 $0$）：

$$\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T_{\text{total}}}\right)$$

#### 9.3.6 解释方差

Critic 质量的核心诊断指标：

$$\text{EV} = 1 - \frac{\text{Var}(\hat{R} - V)}{\text{Var}(\hat{R})}$$

值域为 $(-\infty, 1]$。接近 1 表示 Critic 预测准确；负值表示预测不如均值。

### 9.4 CurriculumManager

`CurriculumManager` 管理三阶段 push-only 课程，各阶段的上下文分布为：

| 阶段 | 上下文分布 | 训练目标 |
|------|-----------|---------|
| Stage 1 | $P(c) = \{\text{none}: 1.0\}$ | 学习基础推门接触 |
| Stage 2 | $P(c) = \{\text{left\_only}: 0.5,\; \text{right\_only}: 0.5\}$ | 单臂持杯约束下推门 |
| Stage 3 | $P(c) = \{\text{none}: 0.25,\; \text{left\_only}: 0.25,\; \text{right\_only}: 0.25,\; \text{both}: 0.25\}$ | 最终混合分布 |

**阶段跃迁判据**：维护长度为 $M = 50$ 的滑动窗口，记录近 $M$ 个 epoch 的成功率 $\eta_e$：

$$\bar{\eta} = \frac{1}{M}\sum_{e=E-M+1}^{E} \eta_e \geq \eta_{\text{thresh}} = 0.8$$

满足条件时推进至下一阶段。

**跃迁的惰性生效**：`curriculum.report_epoch()` 只改变当前阶段状态，不立即重置所有环境。新阶段的上下文样本通过 `_episode_reset_fn` 在各环境下一次 auto-reset 时逐步注入，避免在同一 iteration 内硬重配所有并行环境。

### 9.5 DomainRandomizer

`DomainRandomizer` 负责采样（而非应用）episode 级物理参数和步级噪声。环境在 `_reset_idx()` 或 `_pre_physics_step()` 中将采样结果写入仿真状态。

**Episode 级随机化**：

| 参数 | 符号 | 分布 | 默认范围 |
|------|------|------|---------|
| 杯体质量 | $m_{\text{cup}}$ | $\mathcal{U}(0.1, 0.8)$ | 0.1 – 0.8 kg |
| 门板质量 | $m_{\text{door}}$ | $\mathcal{U}(5.0, 20.0)$ | 5.0 – 20.0 kg |
| 门铰链阻尼 | $d_{\text{hinge}}$ | $\mathcal{U}(0.5, 5.0)$ | 0.5 – 5.0 N·m·s/rad |
| 基座径向距离 | $r$ | $\mathcal{U}(0.45, 0.60)$ | 0.45 – 0.60 m |
| 基座扇形角度 | $\theta_{\text{sector}}$ | $\mathcal{U}(-20°, +20°)$ | $\pm 20°$ |
| 基座 yaw 扰动 | $\delta_{\text{yaw}}$ | $\mathcal{U}(-10°, +10°)$ | $\pm 10°$ |

**步级噪声**：

| 参数 | 符号 | 分布 | 标准差 |
|------|------|------|--------|
| 动作噪声 | $\boldsymbol{\epsilon}_a$ | $\mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I}_{12})$ | $\sigma_a = 0.02$ |
| 观测噪声 | $\boldsymbol{\epsilon}_o$ | $\mathcal{N}(\mathbf{0}, \sigma_o^2 \mathbf{I}_{24})$ | $\sigma_o = 0.01$ |

观测噪声仅注入至 Actor 的 proprio 分支（关节位置 + 关节速度，共 24 维），Critic 观测使用无噪声的真实状态。

### 9.6 训练层统计组件的职责分化

| 组件 | 职责 | 默认路径中的角色 |
|------|------|-----------------|
| `RolloutCollector._accumulate_*()` | Rollout 内奖励/成功率均值 | `collect/*` 和 `reward/*` 的主来源 |
| `episode_stats.py` | 从 info 提取成功率 | 课程学习的输入 |
| `TrainingMetrics` | 长期均值聚合 | `metrics/ppo/*` 指标；`metrics/episode/*` 辅助 |

---

## 10. 入口脚本与运行方式

### 10.1 `scripts/train.py`

`train.py` 是默认训练入口，执行以下装配流程：

```
1. load_config()          → 合并 7 份 YAML 为 cfg 字典
2. resolve_train_runtime_config() → 提取 TrainRuntimeConfig
3. launch_simulation_app() → 启动 Isaac Sim 运行时
4. build_env_cfg()        → 选择环境配置变体，注入参数
5. 构建组件序列：
   DoorPushEnv + DirectRLEnvAdapter
   → Actor + Critic
   → PPOTrainer
   → RolloutCollector + RolloutBuffer
   → CurriculumManager
   → DomainRandomizer
   → TrainingMetrics
6. 注册 _episode_reset_fn
7. 首次批量 reset（build_curriculum_reset_batch → reset_batch）
8. collector.reset_hidden(n_envs)
9. 进入主循环
```

### 10.2 其他脚本

| 脚本 | 职责 |
|------|------|
| `scripts/evaluate.py` | 评估训练策略的跨上下文表现 |
| `scripts/export_policy.py` | 将 Actor 导出为 ONNX/TorchScript 用于部署 |
| `scripts/load_scene.py` | Isaac Sim 中加载和调试场景资产 |
| `scripts/rollout_demo.py` | 结合 visualization 模块生成 rollout artifact（视频、帧图） |

---

## 11. 端到端训练闭环

### 11.1 单轮 Iteration 流程

每轮 iteration 包括以下步骤：

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: Rollout 采集                                            │
│   collector.collect() → 推进 N×T 步环境交互                     │
│   输出: buffer 数据 + collect_stats + last_values              │
├────────────────────────────────────────────────────────────────┤
│ Step 2: GAE 计算                                                │
│   buffer.compute_gae(last_values, last_dones)                  │
│   反向递推: δ_t → Â_t → R̂_t                                   │
├────────────────────────────────────────────────────────────────┤
│ Step 3: PPO 参数更新                                            │
│   ppo_trainer.update(buffer)                                   │
│   K epochs × M mini-batches × TBPTT sequences                  │
├────────────────────────────────────────────────────────────────┤
│ Step 4: 调度更新                                                │
│   global_steps += N × T                                        │
│   学习率衰减（若启用）                                           │
│   curriculum.report_epoch(success_rate) → 检查阶段跃迁           │
├────────────────────────────────────────────────────────────────┤
│ Step 5: 监控与持久化                                            │
│   控制台日志（log_interval 控制频率）                             │
│   TensorBoard 写入                                              │
│   Checkpoint 保存（checkpoint_interval 控制频率）                │
├────────────────────────────────────────────────────────────────┤
│ Step 6: Buffer 清空                                             │
│   buffer.clear() → 准备下一轮                                   │
└────────────────────────────────────────────────────────────────┘
```

训练在 `global_steps >= total_steps` 时结束，或用户中断时保存最终 checkpoint 并释放资源。

### 11.2 Episode 生命周期

从单个并行环境实例的视角，执行循环为：

**单步执行顺序**：

1. `_pre_physics_step(actions)`：缓存 raw action（用于安全惩罚计算）；执行硬裁剪；注入动作噪声；二次裁剪；写入 arm effort target。
2. 物理引擎推进 `decimation = 2` 个物理步（总时长 $\frac{2}{120} = \frac{1}{60}$ s）。
3. `_get_observations()`：读取关节和刚体状态；执行世界系→base 系变换；数值微分计算加速度；计算 tilt proxy；构造 Actor/Critic 观测。**同步缓存**后续阶段需复用的中间量。
4. `_get_rewards()`：基于门角度增量、缓存的加速度/tilt/上一步力矩/安全状态，计算任务奖励、稳定性奖励、安全惩罚；分项写入 `extras["reward_info"]`。
5. `_get_dones()`：根据门角度达标、杯体脱落和步数超限生成 `terminated` 与 `truncated`；把 pre-reset 成功标志和 occupancy 写入 `extras`。

**终止条件**：

| 条件 | 判据 | 类型 |
|------|------|------|
| 成功终止 | $\theta_t \geq \theta^* = 1.57$ rad | `terminated = True` |
| 失败终止 | $\|\mathbf{p}_{\text{cup}} - \mathbf{p}_{\text{ee}}\| > 0.15$ m | `terminated = True` |
| 截断终止 | $\text{step\_count} \geq 900$（15 s × 60 Hz） | `truncated = True` |

需区分两个角度阈值：
- `success_angle_threshold = 1.2` rad：仅触发任务奖励中的一次性成功 bonus。
- `door_angle_target = 1.57` rad：触发 episode 成功终止。

### 11.3 选择性 Reset 流程

当部分环境完成或失败后，Isaac Lab 自动调用 `_reset_idx(env_ids)`，仅重置 `env_ids` 指定的子集：

```
_reset_idx(env_ids):
  1. 采样默认域随机化参数（cup_mass, door_mass, door_damping）
  2. 应用 pending override（来自 set_domain_params_batch）
  3. 对未显式覆写的环境，调用 _episode_reset_fn
     → 按当前课程阶段注入 occupancy 和 domain params
  4. 写入机器人 root state 与 joint state（含新 base pose）
  5. 将门关节角度归零
  6. 不需要持杯的环境: teleport 杯体至远处 (100, y, 0)
  7. 需要持杯的环境: _batch_cup_grasp_init()
     → 直接写关节至预设抓取姿态，teleport 杯体至夹爪位置
  8. 写入新物理参数至仿真（门板质量、杯体质量、阻尼）
  9. 清零 per-env 状态（step_count, prev_action, 速度缓存, _already_succeeded）
```

**关键约束**：此过程中无资产重建，无 `sim.step()` 调用，不影响未完成环境的物理状态。

---

## 12. 跨层数据流

### 12.1 课程学习数据流

```text
环境层                              训练层
_get_dones()                      RolloutCollector
  │ 写 success + occupancy           │
  │ 到 extras                        │ compute_episode_outcome_stats()
  ▼                                  ▼
DirectRLEnvAdapter                extract_curriculum_success_rate()
  │ 重构 info dict                    │
  ▼                                  ▼
info["success"]               CurriculumManager.report_epoch(η)
  │                                   │ 更新滑动窗口
  │                                   │ 检查跃迁条件
  │                                   ▼
  │                              阶段变化? → 更新 current_stage
  │                                   │
  ▼                                   ▼
_episode_reset_fn              下次 auto-reset 时
  │ 按新阶段采样                   注入新上下文分布
  ▼
环境 reset
```

课程机制在训练层判定，在环境层惰性生效。

### 12.2 奖励统计链路

```text
DoorPushEnv._get_rewards()
  │ 构造 reward_info dict
  │ 写入 extras["reward_info"]
  ▼
DirectRLEnvAdapter
  │ 还原为 per-env 标量字典
  ▼
RolloutCollector._accumulate_reward_stats()
  │ rollout 内求均值
  ▼
collect_stats["reward/*"]
  │
  ├─ reward/total, reward/task, reward/stab_left, reward/stab_right, reward/safe
  └─ reward_terms/* (细分子项)
  ▼
TensorBoard
```

### 12.3 域随机化跨层流

```text
DomainRandomizer                  DoorPushEnv
  │ 采样 episode 级参数               │
  │ (m_cup, m_door, d_hinge,         │
  │  base_pos, base_yaw)              │
  │                                   │
  │ 步级噪声:                         │
  │ ε_a → _pre_physics_step()        │ → 注入到 action
  │ ε_o → _get_observations()        │ → 注入到 proprio obs
  │                                   │
  ▼                                   ▼
  Critic privileged vector         PhysX 仿真
  [cup_mass, door_mass,            质量/阻尼写入
   door_damping]                   仿真状态
```

---

## 13. 监控、输出与训练工件

### 13.1 控制台日志

按 `log_interval`（默认 5 轮）输出 iteration 级概览：iteration 编号、累计 `global_steps`、FPS、actor_loss、critic_loss、entropy、clip_fraction、平均奖励 $\bar{r}$、当前 iteration 成功率、rollout_s/update_s 耗时、当前课程阶段、ETA。

### 13.2 TensorBoard 指标体系

| 命名空间 | 内容 | 详细参考 |
|---------|------|---------|
| `train/*` | actor_loss, critic_loss, entropy, clip_fraction, approx_kl, explained_variance, fps | §9.3 |
| `timing/*` | rollout_s, update_s, env_steps_per_s | §11.1 |
| `collect/*` | mean_reward, completed/successful_episodes, episode_success_rate, 上下文成功率 | §9.1 |
| `reward/*` | total, task, stab_left, stab_right, safe | §8 |
| `reward_terms/*` | task/delta, task/open_bonus, task/approach, task/approach_raw, zero_acc, zero_ang, acc, ang, tilt, smooth, reg, joint_limit, joint_vel, torque_limit, cup_drop | §8 |
| `curriculum/*` | stage, window_mean | §9.4 |
| `metrics/ppo/*` | 聚合周期内的 PPO 指标均值 | §9.6 |

### 13.3 Checkpoint 结构

默认训练会为每次运行创建独立目录 `checkpoints/checkpoints_<timestamp>/`，并沿用 `runs/ppo_<timestamp>/` 的时间戳。固定间隔 checkpoint 仍由 `checkpoint_interval` 控制；当课程发生 `stage_1 -> stage_2` 或 `stage_2 -> stage_3` 跃迁时，会额外立即保存 `ckpt_stage_<new_stage_name>.pt`。训练结束或用户中断时保存 `ckpt_final.pt`。每个 checkpoint 包含：

```python
{
    "iteration": int,
    "global_steps": int,
    "actor_state_dict": dict,
    "critic_state_dict": dict,
    "trainer_state_dict": dict,      # 含 optimizer state
    "curriculum_state_dict": dict,   # 含阶段与滑动窗口
    "best_success_rate": float,
}
```

训练恢复包括模型权重、优化器状态和课程学习进度三部分。

---

## 14. 关键接口与数据契约

### 14.1 核心张量形状

| 对象 | 形状 | 含义 |
|------|------|------|
| Actor 观测 | $(N, 96)$ | 5 分支拼接 |
| Critic 观测 | $(N, 109)$ | Actor 观测无噪声版 + 13D privileged |
| Privileged | $(N, 13)$ | door_pose(7) + door_joint_pos(1) + door_joint_vel(1) + cup_mass(1) + door_mass(1) + door_damping(1) + cup_dropped(1) |
| 动作 | $(N, 12)$ | 双臂 raw torque |
| 奖励 | $(N,)$ | 标量即时奖励 |
| `reward_info` | `dict[str, Tensor(N,)]` | 分项奖励字典 |
| Hidden state | $(L, N, 512)$ | RNN 隐状态（$L$ = num_layers） |
| 优势 | $(T, N)$ | GAE 优势估计 |
| Returns | $(T, N)$ | 累计折扣回报 |
| Mini-batch 序列 | $(B, L_{\text{seq}}, \text{dim})$ | TBPTT 片段 |

### 14.2 `reward_info` 键体系

```
reward_info/
├── task
│   ├── task/delta
│   ├── task/open_bonus
│   ├── task/approach
│   └── task/approach_raw
├── stab_left
│   ├── stab_left/zero_acc
│   ├── stab_left/zero_ang
│   ├── stab_left/acc
│   ├── stab_left/ang
│   ├── stab_left/tilt
│   ├── stab_left/smooth
│   └── stab_left/reg
├── stab_right
│   └── (同 stab_left 结构)
├── safe
│   ├── safe/joint_limit
│   ├── safe/joint_vel
│   ├── safe/torque_limit
│   └── safe/cup_drop
└── total
```

### 14.3 `collect_stats` 核心字段

```
collect/
├── mean_reward
├── completed_episodes
├── successful_episodes
├── episode_success_rate
├── success_mixed (= episode_success_rate)
├── success_none
├── success_left_only
├── success_right_only
└── success_both
reward/
├── total
├── task
├── stab_left
├── stab_right
└── safe
```

### 14.4 配置系统共享约束

以下约束在修改配置时必须保持一致：

1. `success_angle_threshold`（1.2 rad，奖励用途）与 `door_angle_target`（1.57 rad，终止用途）不可混用。
2. `reward/default.yaml` 通过注入进入环境配置，修改奖励权重需同时理解配置侧和环境侧。
3. 课程阶段仅决定上下文分布与门类型集合，不直接改变策略输入结构。

---

## 15. 默认路径与历史路径的边界

### 15.1 默认训练主线特征

- 门相关输入采用 6D `door_geometry`（仿真 ground truth），不经过视觉感知。
- `DoorPushEnv` 是环境、观测、奖励和终止判定的统一实现位置。
- 训练使用 PPO + asymmetric Actor-Critic + recurrent Actor（GRU）。
- 课程学习采用三阶段 push-only 分布，滑动窗口成功率触发跃迁。
- 域随机化覆盖 episode 级物理参数（3 项）+ 基座位姿（3 项）+ 步级噪声（2 项）。
- TensorBoard、checkpoint 与导出工具形成完整训练输出链路。

### 15.2 历史视觉感知路径

`src/affordance_guided_interaction/door_perception/` 是历史实验模块，描述从 RGB-D经开集分割、深度反投影和 Point-MAE 编码得到高维 `door_embedding` 的视觉路径。当前保留用于记录过去的感知研究方向与未来视觉实验参考。`training/perception_runtime.py` 同属兼容保留模块。

### 15.3 其他非默认模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `src/teleop_cup_grasp/` | 独立辅助 | 杯体抓取遥操作与实验工具 |
| `visualization/` | 正式输出链路 | Rollout artifact 生成，非历史残留 |

---

## 16. PPO 算法超参数总表

| 参数 | 符号 | 默认值（env_4096 profile） |
|------|------|--------------------------|
| 并行环境数 | $N$ | 4096 |
| 每轮步数 | $T$ | 16 |
| 总训练步数 | — | 300M |
| Actor 学习率 | $\eta_{\text{actor}}$ | 2 × 10⁻⁴ |
| Critic 学习率 | $\eta_{\text{critic}}$ | 2 × 10⁻⁴ |
| PPO epoch 数 | $K$ | 3 |
| Mini-batch 数 | $M$ | 16 |
| TBPTT 序列长度 | $L$ | 16 |
| 折扣因子 | $\gamma$ | 0.99 |
| GAE 参数 | $\lambda$ | 0.95 |
| PPO 裁剪参数 | $\epsilon$ | 0.2 |
| Value 裁剪参数 | $\epsilon_v$ | 0.2 |
| 熵正则化系数 | $c_e$ | 0.01 |
| Critic 损失权重 | $c_v$ | 0.5 |
| 梯度裁剪范数 | $g_{\max}$ | 1.0 |
| 优势归一化 | — | 启用 |
| Clipped value loss | — | 启用 |
| Actor RNN 隐状态维度 | — | 512 |
| Actor RNN 层数 | — | 1 |
| Actor RNN 类型 | — | GRU |
| Log std 初始值 | $\ell_0$ | −0.5 |
| Critic MLP 层 | — | [512, 256, 128, 1] |
| 物理步长 | $\Delta t_{\text{phys}}$ | 1/120 s |
| Decimation | — | 2（控制频率 60 Hz） |
| Episode 最大时长 | — | 15 s (900 步) |
| 门角度终止目标 | $\theta^*$ | 1.57 rad (≈ 90°) |
| 成功奖励触发角度 | $\theta_{\text{bonus}}$ | 1.2 rad |
| 掉杯判定阈值 | — | 0.15 m |
| 臂关节力矩上限 | $\boldsymbol{\tau}_{\max}$ | joint2: 60 N·m, 其余: 30 N·m (per-joint, 来自 Z1 URDF) |
| 课程窗口长度 | $M_{\text{cur}}$ | 50 |
| 课程跃迁阈值 | $\eta_{\text{thresh}}$ | 0.8 |

---

## 17. 推荐阅读顺序

1. **本文档**：建立全项目架构、训练闭环与模块边界的系统级认知。
2. [README.md](../README.md)：快速开始方式与外部使用视角。
3. [envs/Reward.md](../src/affordance_guided_interaction/envs/Reward.md)：奖励函数完整数学推导与设计动机。
4. [training/README.md](../src/affordance_guided_interaction/training/README.md)：PPO 算法、TBPTT、课程学习与域随机化的形式化描述。
5. [envs/README.md](../src/affordance_guided_interaction/envs/README.md)：环境层在 Isaac Lab 中的实现细节。
6. [observations/README.md](../src/affordance_guided_interaction/observations/README.md)：观测层语义规范与工具类。
7. [policy/README.md](../src/affordance_guided_interaction/policy/README.md)：策略网络架构、动作语义与时序设计。
8. [configs/README.md](../configs/README.md)：7 份 YAML 的消费路径与参数含义。
9. [docs/randomization.md](./randomization.md) 与 [docs/tensorboard_guide.md](./tensorboard_guide.md)：随机化参数详表与监控指标释义。
10. 最后进入 `scripts/`、`envs/`、`policy/`、`training/` 的具体实现文件进行代码级阅读。

---

## 18. 总结

本项目的训练系统由资源、环境、观测、策略、训练、监控与输出六层协同构成。环境层提供 GPU 批量并行的交互世界；观测层将仿真状态整理为结构化输入；策略层在部分可观测条件下输出连续控制信号；训练层通过 PPO + GAE + TBPTT 将交互数据转化为稳定的参数更新；监控与输出层保证整个过程可解释、可恢复、可分析。

默认训练路径以 6D `door_geometry` 为门相关输入，以 asymmetric Actor-Critic + GRU 为策略架构，以三阶段课程和 8 维域随机化为泛化机制，以 20 项奖励超参数和显式 stage checkpoint 为训练信号与运行保障，构成当前项目的主干实现。理解这套系统的关键在于把握各层之间的职责划分与数据流向，从而准确判断改动的影响范围与正确的实现位置。
