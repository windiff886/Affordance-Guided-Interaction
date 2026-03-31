# Policy 层：约束感知执行策略

## 1. 本层在系统中的位置

```
observations/ (actor_obs / critic_obs)
       │
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
              │  关节力矩 τ ∈ R^6
              ▼
         仿真执行层 (envs/)
```

policy 层是整个框架的决策核心。它接收来自 `observations/` 层的结构化输入，在身体约束存在的情况下，学习如何完成 door-related interaction。

**本层不做的事情**：
- 不直接读取原始图像
- 不依赖 `cup_mass`、`door_mass` 等部署时不可得参数
- 不预设"持杯时必须用哪个 link 接触"的显式规则
- 不对门点云做语义标注或分割

---

## 2. 策略形式化

按照 v11 方案，策略可写为：

$$a_t = \pi_\theta(o_t,\; h_t,\; c_t,\; \text{point\_cloud}_t)$$

其中：

| 符号 | 说明 |
|---|---|
| $o_t$ | 当前本体观测（关节状态 + EE 状态 + 稳定性 proxy） |
| $h_t$ | 循环网络隐状态（GRU / LSTM cell），用于隐式编码历史与参数不确定性 |
| $c_t$ | 任务上下文（`occupied`、`stability_level`） |
| $\text{point\_cloud}_t$ | 当前帧门点云 $(N, 3)$，来自 `door_perception/` |
| $a_t$ | 输出：6 维关节力矩向量 $\tau \in \mathbb{R}^6$ |

策略使用 **PPO + recurrent actor**（GRU/LSTM）。循环结构的作用不是明确地记忆"动作序列"，而是让策略通过历史观测隐式辨识当前的隐藏环境参数（如杯体质量、门阻尼），从而适应域随机化带来的多样参数分布。

---

## 3. 网络结构

### 3.1 总体数据流

```
actor_obs
  │
  ├── proprio (q, dq, tau, past_actions)   → MLP encoder → f_proprio
  ├── gripper_state (pos, ori, vel, ω)     → MLP encoder → f_ee
  ├── context (occupied, stability_level)  → 直接拼接
  ├── stability_proxy (tilt, acc, jerk...) → MLP encoder → f_stab
  └── door_point_cloud (N, 3)             → PointNet encoder → f_pc
                                                    │
                  ┌─────────────────────────────────┘
                  │  concat([f_proprio, f_ee, context, f_stab, f_pc])
                  ▼
         RecurrentBackbone (GRU / LSTM)
                  │  隐状态 h_t
                  ▼
           ActionHead (Gaussian policy)
                  │
                  ▼
          τ_t ∈ R^6  (关节力矩)
```

### 3.2 各子模块说明

#### `recurrent_backbone.py` — 循环骨干网络

- 使用 **GRU 或 LSTM**，输入为上述拼接后的特征向量
- 隐状态 $h_t$ 在 episode 内持续传递，episode 开始时清零
- 用途：隐式辨识隐藏环境参数；对历史运动模式建模（如是否正在接触门）

推荐配置：
```
input_dim  = dim(f_proprio) + dim(f_ee) + 2 + dim(f_stab) + dim(f_pc)
hidden_dim = 512
num_layers = 1 或 2
```

#### `actor.py` — Actor 网络

- 接收完整 `actor_obs`，依次经过各分支 encoder → RecurrentBackbone → ActionHead
- 输出高斯分布的均值与对数标准差，用于 PPO 采样
- **在 episode 内维护 hidden state**，外部调用时需传入并接收更新后的隐状态

推荐分支 encoder 结构：
```
proprio encoder:    Linear(6+6+6 + k*6, 128) → LayerNorm → ReLU → Linear(128, 64)
ee encoder:         Linear(3+4+3+3, 64)       → LayerNorm → ReLU → Linear(64, 32)
stability encoder:  Linear(stability_dim, 64) → LayerNorm → ReLU → Linear(64, 32)
point cloud enc:    PointNet(N, 3) → global max pool → Linear(1024, 128)
```

#### `critic.py` — Asymmetric Critic

- 接收完整 `critic_obs`（= `actor_obs` + `privileged`）
- **不使用循环结构**，直接用 MLP 估计状态价值
- Privileged information 额外带来精确对象状态和隐藏物理参数

推荐结构：
```
actor_obs_features → MLP encoder (与 actor 共享或独立)
privileged_features → MLP encoder
concat → MLP(512, 256, 128) → Linear(128, 1) → V(s)
```

#### `action_head.py` — 动作参数化与输出

- 输出 **6 维关节力矩**（对应 Z1 机械臂 6 个旋转关节）
- 参数化为对角高斯分布：$a \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2)$
- 动作在输出前需进行**关节力矩裁剪**（clip to joint torque limits）

力矩限制参考（Z1 机械臂）：
```
joint 1~6 的最大力矩限制分别需参考 Z1 规格文档设定
```

---

## 4. 门点云编码器

门点云是本项目感知输入的核心，由 `door_perception/` 的完整管线（分割 → 反投影 → 清理）得到。

Policy 层需要一个点云编码器将变长点云 $(N, 3)$ 压缩为固定维度特征。推荐使用 **PointNet** 作为默认编码器：

```
输入: (N, 3)
  → shared MLP(3, 64, 128, 1024) [per-point]
  → global max pool
  → (1024,)
  → MLP(1024, 512, 128)
  → f_pc: (128,)
```

**设计原则**：
- PointNet 对点顺序不变，适合从深度反投影得到的无序点云
- global max pool 保留最显著的局部几何特征
- 不使用 PointNet++ 或 transformer，避免引入不必要的复杂度
- 编码器**端到端随策略联合训练**（不冻结），因为任务相关的几何特征与门交互紧密耦合

可选：若已有冻结的 Point-MAE / ULIP 预训练权重（来自 `door_perception/frozen_encoder.py`），可将其输出的 embedding 与 PointNet 特征拼接，作为增强版 $f_{pc}$。

---

## 5. 动作空间

**控制方式**：关节力矩控制（joint torque control）

选择力矩控制而非位置控制的原因：
- 力矩控制可以直接表达"施加多大的力"，适合与门、按钮、把手的物理接触
- 位置控制在接触过程中容易产生大力，难以精细调节接触力
- SoFTA / Hold My Beer 风格的稳定性奖励也更自然地与力矩幅值正则项配合

**动作维度**：$a_t \in \mathbb{R}^6$，对应 Z1 单臂 6 个旋转关节的力矩

---

## 6. 训练设置

### 6.1 算法

| 项目 | 选择 |
|---|---|
| 强化学习算法 | **PPO**（clip ratio 0.2） |
| Actor 结构 | Recurrent（GRU/LSTM） |
| Critic 结构 | Asymmetric MLP（接收 privileged info） |
| 采样 | 多环境并行 rollout |
| 优化器 | Adam |

### 6.2 训练态 vs 部署态边界

| | Actor | Critic |
|---|---|---|
| **训练态** | 看现实可得观测 | 额外看 privileged info（精确对象状态 + 隐藏参数） |
| **部署态** | 直接使用，hidden state 持续维护 | 不参与推理 |

Critic 的 privileged information 包括：

```
door_pose, door_joint_pos/vel,
cup_pose, cup_linear_vel, cup_angular_vel,
cup_mass, cup_fill_ratio,
door_mass, door_damping
```

这些量在训练时帮助 Critic 更精确地估计 value，但 actor 对其不可见，只能通过循环隐状态和历史观测隐式适应。

### 6.3 `occupied` 对策略行为的影响

`occupied` 不会改变网络结构，而是通过两条路径影响策略：

1. **作为显式输入**：直接出现在 `context` 中，让策略感知当前是否持杯
2. **通过奖励函数**：`occupied=1` 时激活持杯稳定奖励项 $m_{\text{occ}} \cdot r_{\text{carry-stability}}$

这样，同一个策略参数 $\theta$ 在 `occupied=0` 和 `occupied=1` 两种上下文下，会自然学出不同的行为模式：
- **空手**：可以更自由、直接地完成门交互，不受末端稳定约束
- **持杯**：倾向于低加速度、低冲击、末端平稳的动作风格

---

## 7. 文件职责对照

| 文件 | 职责 |
|---|---|
| `actor.py` | 完整 actor 前向：分支编码 → RecurrentBackbone → ActionHead → 采样动作 |
| `critic.py` | Asymmetric critic 前向：`actor_obs` + `privileged` → MLP → 标量价值 |
| `recurrent_backbone.py` | GRU / LSTM 封装；管理 hidden state 的传入与输出 |
| `action_head.py` | 高斯分布参数化；输出 $(\mu, \log\sigma)$；施加力矩 clip |

---

## 8. 关键设计决策

1. **Recurrent actor + MLP critic（非对称结构）**
   循环结构让 actor 能通过历史观测隐式辨识隐藏参数（cup_mass、door_damping 等），而 MLP critic 直接访问这些参数从而更准确地估计价值。

2. **门点云端到端训练，不冻结**
   门点云编码器与策略联合训练，使 PointNet 学到与门交互任务相关的几何特征，而非通用的形状特征。

3. **力矩控制**
   比位置控制更适合接触丰富的 door-related interaction，与稳定性奖励的力矩正则项配合更自然。

4. **不预设接触部位**
   policy 层不预先规定"持杯时必须用哪个 link 接触门"。body affordance 通过 RL 优化过程在策略隐状态中隐式学出。

5. **`occupied` 的双重作用**
   既作为显式上下文输入，又通过奖励 mask 间接调节行为，使同一套参数在持杯/空手两种场景下呈现分化的行为风格。
