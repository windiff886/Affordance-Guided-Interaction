# Policy 层：约束感知执行策略

## 1. 本层在系统中的位置

```text
 door_perception/ (Frozen Point-MAE)       observations/ (actor_obs / critic_obs)
        │                                         │
        ▼                                         ▼
  门点云特征 (door_embedding)                  双臂状态与稳定性 proxy
        │                                         │
        └────────────────────┬────────────────────┘
                             ▼
             ┌─────────────────────────────────┐
             │  policy/                         │  ◄── 本层
             │  约束感知执行层                   │
             │                                   │
             │  actor.py        → 输出动作      │
             │  critic.py       → 输出价值估计  │
             │  recurrent_backbone.py           │
             │  action_head.py                  │
             └────────────┬────────────────────┘
                          │  关节力矩 τ ∈ R^12 (双臂)
                          ▼
                     仿真执行层 (envs/)
```

policy 层是整个框架的决策核心。它接收来自 `door_perception/` 的环境高维视觉 affordance 特征，并结合 `observations/` 输出的双臂物理状态，在持杯等身体约束存在的情况下，学习如何完成门相关的交互任务（door-related interaction）。

**本层不做的事情**：
- 不处理原始图像或原始点云，视觉降维工作全部交由上游冻结编码器完成。
- 不依赖 `cup_mass`、`door_mass` 等部署阶段不可得的物理隐参数。
- 不预设“持杯时必须用哪个 link 接触门”的刚性规则。

---

## 2. 策略形式化

根据最新 v5 架构，控制策略形式化为：

$$a_t = \pi_\theta(o_t,\; h_t,\; c_t,\; z_{\text{aff}, t})$$

其中各变量的含义如下：

| 符号 | 说明 |
|---|---|
| $o_t$ | 当前机器人本体观测，包括双臂关节状态、双臂末端位姿与速度，以及双臂独立的稳定性 proxy。 |
| $h_t$ | 循环网络隐状态（GRU / LSTM cell），用于隐式编码历史交互信息并辨识隐藏物理参数。 |
| $c_t$ | 任务上下文，包括左/右臂的 `occupied` 标志。 |
| $z_{\text{aff}, t}$ | 统一视觉 affordance 表征，即当前帧门点云对应的 `door_embedding` $(768,)$。 |
| $a_t$ | 输出动作：双臂总计 12 维的关节力矩向量 $\tau \in \mathbb{R}^{12}$。 |

策略依然采用 **PPO + recurrent actor**。使用循环层（RNN）的核心目的是：使策略在随机化的仿真环境中，通过局部交互反馈潜移默化地推断出当前的重量、阻尼等环境特征。

---

## 3. 网络结构

### 3.1 总体数据流

```text
actor_obs
  │
  ├── proprio (q, dq, tau, past_actions)         → MLP encoder → f_proprio
  ├── gripper_states (左/右臂位姿、速度)            → MLP encoder → f_ee
  ├── context (left/right_occupied, stability)   → 直接拼接
  ├── stability_proxies (左/右臂 acc, tilt...)      → MLP encoder → f_stab
  └── door_embedding (768维冻结特征)               → MLP encoder → f_vis
                                                        │
                      ┌─────────────────────────────────┘
                      │  concat([f_proprio, f_ee, context, f_stab, f_vis])
                      ▼
             RecurrentBackbone (GRU / LSTM)
                      │  隐状态 h_t
                      ▼
               ActionHead (Gaussian policy)
                      │
                      ▼
              τ_t ∈ R^12  (双臂关节力矩)
```

### 3.2 子模块职责

#### `recurrent_backbone.py` — 循环主干网络
- 可选用 **GRU 或 LSTM**，负责处理所有观测分支拼接后的高维特征向量。
- 隐状态 $h_t$ 在单个 episode 期间持续流转，每次 reset 时由环境发信号进行清零。
- 用于记忆历史的力矩与动作规律，从而弥补单帧观测中缺失的环境动力学信息。

#### `actor.py` — Actor 网络
- 管理所有的特征编码分支（encoder），最后流经 Backbone 与 ActionHead。
- 输出用于 PPO 采样的高斯分布均值与对数标准差。
- 注意：由于不再输入原始点云，原先内部包含的 PointNet 现已移除，变为接收 768 维特征的简单多层感知机（MLP）。

#### `critic.py` — 非对称评价网络 (Asymmetric Critic)
- 输入为完整 `critic_obs`（即 `actor_obs` 加上 `privileged` 信息）。
- **不包含任何循环结构**，直接通过 MLP 拟合状态价值 $V(s)$。
- 凭借 privileged information 提供的精确杯具状态与门铰链参数，能在训练时提供更准的梯度方向。

#### `action_head.py` — 动作映射与裁剪
- 将高斯特征解码为物理需要的 **12 维关节力矩**（左臂 6 维 + 右臂 6 维）。
- 执行力矩截断操作（clip），防范越界力矩撕裂物理仿真。

---

## 4. 视觉 Affordance 特征接口

在 v5 版本中，为了解耦视觉特征抽取与强化学习训练，Policy 层将点云处理逻辑剥离至 `door_perception/` 模块。

- **输入形式**：当前 `actor_obs["door_embedding"]` 是一条大小为 $(768,)$ 的一维向量。
- **上游来源**：该特征由 `frozen_encoder.py` 中冻结权重的 Point-MAE 推理得出，融合了 mean_pooling 和 max_pooling 以涵盖全局与局部空间语义。
- **优势**：消除了训练初期因视觉表征不稳定导致的梯度爆炸问题，且不再需要昂贵的在线点卷积运算。

---

## 5. 动作空间与控制

本框架采用**关节力矩控制（Joint Torque Control）**。

相比基于位置的解算控制，采用力矩控制的理由在于：
- 它天生适合解决与环境频繁发生碰撞和物理摩擦的任务。
- 难以准确预估门推力时，力矩控制允许机器人呈现柔顺接触。
- 配合我们施加的力矩平滑度正则化惩罚项，能够更好地避免持杯臂产生破坏稳定性的突变冲击力。

---

## 6. 上下文调制与行为分化

当前设定为双臂系统。环境中，每个臂都有独立的持杯标志 (`left_occupied` / `right_occupied`) 和独立的末端平稳度监测器 (`stability_proxy`)。这种设计具有以下深度影响：

1. **输入调制**：上下文 `context` 直连至特征空间，使策略网络能够“知道”自身双手的占用状态。
2. **奖励遮罩 (Reward Mask)**：策略会收到由 `occupied` 决定的对象稳定惩罚项。只有当某臂的 `occupied = 1` 时，对应的加速度突切、重力倾斜项等高额惩罚才计入。

**最终效果表现**：同一套权重（$ \theta $），当机械臂空手时会表现出激进且快速的尝试；当机械臂持满水杯时，则会立刻转换出谨慎施力、维持末端水平的柔顺交互风格。系统通过强化学习自发选出最适合承担推/按压等操作的身体部位，无需刻意编写人工操作分配规则。

---

## 7. 关键设计决策摘要

| 决策点 | 原因解析 |
|---|---|
| **移除内建 PointNet** | 改用分离的静态 Point-MAE 特征输入。大大提升训练速度，分离任务进展识别和动作探索两部分的不稳定因素。 |
| **异步不对称 Actor-Critic** | Actor 通过 LSTM 暗中推断动力学参数，Critic 借用超维物理真值评估状态，共同提升复杂交互的收敛性。 |
| **参考系对齐至 Gripper** | 取消 Wrist 坐标为主导，改用末端夹爪 (Gripper) 定义稳定性。因为“水杯”的倾角直接绑定抓取器，该调整对重力方向惩罚（Gravity tilt）更加精确。 |
| **双臂解耦代理** | 两条独立处理的臂膀共享同一个强化学习大脑，方便后期拓展单边、双边不同的 affordance 组合任务。 |
