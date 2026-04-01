# Training — 策略训练与课程编排

## 1. 本层在系统中的位置

training 层是整个系统的**优化与调度中心**，处于观测构建 (`observations/`)、奖励计算 (`rewards/`) 和策略网络 (`policy/`) 之上，负责：

1. **策略优化**：使用 PPO（Proximal Policy Optimization）更新网络参数
2. **轨迹收集**：管理多环境的高度并行 Rollout，协调 actor 与 critic 的 asymmetric 观测
3. **课程推进**：管理并自动切换从简单交互到复杂混合持杯任务的训练阶段
4. **域随机化**：在每个 episode reset 时注入随机性，防止策略过拟合到单一环境实例

```
    observations/                rewards/
    actor_obs, critic_obs        r_total
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────┐
│  training/                                    │  ◄── 本层
│                                                │
│  PPO Trainer      ← 策略梯度优化              │
│  Rollout Collector ← 并行轨迹采样            │
│  Curriculum Manager← 阶段自动跃迁            │
│  Domain Randomizer ← 物理参数随机化           │
└──────────────────────┬───────────────────────┘
                       │  ∇θ → policy/ 参数更新
                       ▼
                  policy/ (actor + critic)
```

**核心设计原则**：

- training 层不处理具体的物理接触或视觉编码，只关注"如何高效收敛"
- 严格维持 actor / critic 的信息不对称——actor 只看部署可得信号，critic 额外看仿真 oracle
- 所有超参数外置于配置文件，代码保持数学纯粹性

---

## 2. PPO 算法数学模型

### 2.1 目标函数总览

PPO（Proximal Policy Optimization）通过限制策略更新幅度来保证训练稳定性。每次更新使用从并行环境收集的 $N$ 步轨迹数据，执行 $K$ 个 epoch 的 mini-batch 梯度下降。

总损失函数为：

$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{actor}}(\theta) + c_v \cdot \mathcal{L}_{\text{critic}}(\theta) - c_e \cdot \mathcal{H}[\pi_\theta]
$$

其中 $c_v$ 为 value loss 权重，$c_e$ 为熵正则化系数，$\mathcal{H}[\pi_\theta]$ 为策略输出分布的熵。

### 2.2 广义优势估计（GAE）

优势函数采用 GAE（Generalized Advantage Estimation），在偏差与方差之间提供可调平衡。设时间步 $t$ 的 TD 残差为：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

则 GAE 优势为指数加权的 TD 残差累积和：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \, \delta_{t+l}
$$

其中：

| 参数 | 含义 |
|------|------|
| $\gamma$ | 折扣因子，控制未来奖励的衰减速度 |
| $\lambda$ | GAE 偏差-方差权衡系数。$\lambda = 0$ 退化为单步 TD，$\lambda = 1$ 退化为蒙特卡洛 |
| $V_\phi(s_t)$ | Critic 网络估计的状态价值 |

### 2.3 Actor 损失（PPO-Clip）

定义策略比率：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

PPO-Clip 的 actor 损失为：

$$
\mathcal{L}_{\text{actor}}(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{t \in \mathcal{B}} \min\!\left( \rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right)
$$

其中 $\epsilon$ 为 clipping 参数（默认 0.2），$\mathcal{B}$ 为当前 mini-batch。clip 操作将策略比率限制在 $[1-\epsilon, 1+\epsilon]$ 区间内，阻止单次更新过大偏移。

### 2.4 Critic 损失

Critic 网络拟合 GAE 回报目标 $\hat{R}_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$：

$$
\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{|\mathcal{B}|}\sum_{t \in \mathcal{B}} \left( V_\phi(s_t) - \hat{R}_t \right)^2
$$

可选 clipped value loss 以进一步稳定训练：

$$
\mathcal{L}_{\text{critic}}^{\text{clip}}(\phi) = \frac{1}{|\mathcal{B}|}\sum_{t \in \mathcal{B}} \max\!\left( (V_\phi - \hat{R}_t)^2,\; (\bar{V}_\phi - \hat{R}_t)^2 \right)
$$

其中 $\bar{V}_\phi = \text{clip}(V_\phi,\; V_{\phi_{\text{old}}} - \epsilon_v,\; V_{\phi_{\text{old}}} + \epsilon_v)$。

### 2.5 熵正则化

为防止策略过早收敛到确定性分布，对高斯策略的输出进行熵正则化：

$$
\mathcal{H}[\pi_\theta] = \frac{1}{2} \sum_{i=1}^{12} \left( \ln(2\pi e\, \sigma_i^2) \right)
$$

其中 $\sigma_i$ 为动作空间第 $i$ 维的标准差。12 维对应双臂各 6 个关节的力矩输出。

---

## 3. Asymmetric Actor-Critic 架构

### 3.1 信息不对称的数学形式化

本系统采用非对称 Actor-Critic 架构。Actor 与 Critic 接收不同的状态表示：

$$
\begin{aligned}
a_t &\sim \pi_\theta\!\left(o_t^{\text{actor}},\; h_t\right) \\
V_\phi(s_t) &= f_\phi\!\left(o_t^{\text{critic}}\right)
\end{aligned}
$$

其中 $o_t^{\text{critic}} \supset o_t^{\text{actor}}$：

| 角色 | 输入 | 说明 |
|------|------|------|
| **Actor** | 本体状态、双臂 gripper pose、稳定性 proxy、$z_{\text{aff}}$、context | 仅部署时可得的信号 |
| **Critic** | actor 观测 $\cup$ 精确物体状态、隐藏物理参数 | 额外获得仿真 oracle 提供的 privileged information |

### 3.2 信息不对称的收敛意义

设真实环境状态为 $s_t$，actor 观测到的部分状态为 $o_t \subset s_t$。Critic 直接拟合 $V(s_t)$ 而非 $V(o_t)$，其优势在于：

$$
\text{Var}[\hat{A}_t \mid s_t] \leq \text{Var}[\hat{A}_t \mid o_t]
$$

完整状态信息下估计的优势函数具有更低方差，使梯度方向更准确，加速训练收敛。训练结束后 Critic 被丢弃，只部署 Actor。

### 3.3 循环隐状态与 POMDP

由于 actor 看不到 `cup_mass`、`door_damping` 等隐藏参数，环境对 actor 而言是部分可观测马尔可夫决策过程（POMDP）。Actor backbone 中的循环网络（GRU / LSTM）通过隐状态 $h_t$ 进行在线参数辨识：

$$
h_t = \text{GRU}(h_{t-1},\; e_t)
$$

其中 $e_t$ 为所有 encoder 输出拼接后的特征向量。隐状态在每个 episode 开始时清零，在 episode 内部持续流转，使策略能够通过交互历史隐式推断环境动力学。

### 3.4 截断时间反向传播（TBPTT）

对包含 RNN 的 actor 进行梯度更新时，采用截断时间反向传播以控制计算开销与梯度爆炸风险。设截断长度为 $L$，mini-batch 采样单元为连续 $L$ 步的序列片段：

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{|\mathcal{B}|} \sum_{(t_0, t_0+L) \in \mathcal{B}} \sum_{t=t_0}^{t_0+L-1} \nabla_\theta \log \pi_\theta(a_t \mid o_t, h_t) \cdot \hat{A}_t
$$

每个序列片段的初始隐状态 $h_{t_0}$ 从 rollout 阶段缓存中恢复，梯度只在片段内传播。

---

## 4. 课程学习设计

### 4.1 设计动机

直接在全约束环境下训练会导致策略陷入局部最优——例如为了规避持杯加速度惩罚而选择完全不运动。课程学习通过逐步引入任务复杂度和约束强度，引导策略在**先学会任务本身，再学会在约束下完成任务**的路径上收敛。

### 4.2 五阶段自动跃迁

课程管理器实现 5 阶段训练，每个阶段定义了不同的任务复杂度与约束组合：

| 阶段 | 持杯概率 | 门类型 | 核心学习目标 |
|------|----------|--------|------------|
| **Stage 1** | $P(\text{occupied}) = 0$ | 单一 Push 门 | 基础视觉引导接触，跑通网络闭环 |
| **Stage 2** | $P(\text{occupied}) = 1$ | 单一 Push 门 | 在稳定性约束 $r_{\text{stab}}$ 与 $s_t$ 下学会力控 |
| **Stage 3** | $P(\text{occupied}) \sim \text{Bernoulli}(0.5)$ | Push + Press 混合 | 视觉区分 affordance 类型，调整接触策略 |
| **Stage 4** | $P(\text{occupied}) \sim \text{Bernoulli}(0.5)$ | Button+Door, Handle+Door | 学习时序子任务组合（先按再推），依靠 RNN 跨越 reward delay |
| **Stage 5** | $P(\text{occupied}) \sim \text{Bernoulli}(p)$ | 全类型混合 | 高强度域随机化下的全域泛化 |

### 4.3 跃迁条件的数学判据

阶段跃迁由滑动窗口平均成功率触发。设 $\eta_e$ 为第 $e$ 个 epoch 的成功率，跃迁判据为：

$$
\frac{1}{M} \sum_{e=E-M+1}^{E} \eta_e \geq \eta_{\text{thresh}}
$$

即最近连续 $M$ 个 epoch 的平均成功率超过阈值 $\eta_{\text{thresh}}$ 时，系统自动推进到下一阶段。

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $M$ | 滑动窗口长度（epoch 数） | 50 |
| $\eta_{\text{thresh}}$ | 成功率跃迁阈值 | 0.8 |

### 4.4 课程与奖励缩放的耦合

课程阶段的推进与 rewards 层的动态缩放因子 $s_t$（见 `rewards/README.md` §7）协同工作：

- **阶段 1**（无持杯）：$s_t$ 不影响策略（无稳定性惩罚项被激活）
- **阶段 2**（引入持杯）：$s_t$ 从低值开始线性退火，策略先探索推门方式，再逐步被要求动作平滑
- **阶段 3-5**：$s_t$ 持续增长至 $1.0$，全额惩罚促使策略打磨出顺滑的操作风格

两套机制互为补充：课程控制**任务复杂度的离散跃迁**，$s_t$ 控制**约束强度的连续增长**。

---

## 5. 域随机化

### 5.1 设计目的

域随机化在每次 episode reset 时对环境物理参数进行采样，目的是：
- 防止策略过拟合到单一环境配置
- 在 sim-to-real 迁移前提升策略的鲁棒性
- 迫使 actor 通过 RNN 隐状态隐式辨识环境动力学

### 5.2 随机化参数与噪声注入

系统将随机化分为**回合级（Episode-level）静态参数**与**步级（Step-level）动态噪声**两类：

**(1) 回合级静态参数**（每次 episode reset 时从均匀分布 $\mathcal{U}$ 中采样，局内保持这套常数不变）：

| 参数 | 符号 | 分布 | 说明 |
|------|------|------|------|
| 杯体质量 | $m_{\text{cup}}$ | $\mathcal{U}[m_{\min}, m_{\max}]$ | 模拟不同液体装载量对惯性的影响 |
| 门板质量 | $m_{\text{door}}$ | $\mathcal{U}[m_{\min}^d, m_{\max}^d]$ | 影响推门所需力矩 |
| 门铰链阻尼 | $d_{\text{hinge}}$ | $\mathcal{U}[d_{\min}, d_{\max}]$ | 控制门的运动阻力 |
| 基座初始位置 | $p_{\text{base}}$ | $\mathcal{U}[p_0 - \Delta p, p_0 + \Delta p]$ | 机器人基座的平面偏移（局限在机械臂运动学可达的工作空间裕度内，防止过拟合绝对坐标） |

**(2) 步级动态噪声**（每执行一次 `step()` 时从高斯分布 $\mathcal{N}$ 中独立采样，持续不断地模拟传感器与执行器的高频抖动）：

| 噪声 | 符号 | 分布 | 说明 |
|------|------|------|------|
| 动作噪声 | $\epsilon_a$ | $\mathcal{N}(0, \sigma_a^2 I)$ | 实时叠加在策略发出的关节控制指令上 |
| 观测噪声 | $\epsilon_o$ | $\mathcal{N}(0, \sigma_o^2 I)$ | 实时叠加在物理环境算出的关节编码器真值上 |

其中 $\mathcal{U}[a, b]$ 为均匀分布，$\mathcal{N}(\mu, \Sigma)$ 为高斯分布。

### 5.3 随机化与 privileged information 的关系

域随机化采样的隐藏参数（$m_{\text{cup}}, m_{\text{door}}, d_{\text{hinge}}$ 等）同时提供给 Critic 作为 privileged information，但 Actor 完全不可见。这构成了 asymmetric actor-critic 框架的核心信息差：

$$
o_t^{\text{critic}} = o_t^{\text{actor}} \cup \{m_{\text{cup}},\, m_{\text{door}},\, d_{\text{hinge}},\, p_{\text{base}}\}
$$

Actor 必须通过交互反馈（力矩响应、速度变化等）和 RNN 隐状态来隐式推断这些参数。

---

## 6. 训练流程形式化

### 6.1 单轮更新过程

一次 PPO 更新 epoch 的流程可形式化为：

**Step 1 — 轨迹收集**：在 $N_{\text{env}}$ 个并行环境中推演 $T$ 步，收集轨迹集合：

$$
\mathcal{D} = \left\{ \left(o_t^{\text{actor}}, o_t^{\text{critic}}, a_t, r_t, h_t, \log\pi_{\theta_{\text{old}}}(a_t), V_{\phi_{\text{old}}}(s_t) \right) \right\}_{t=1}^{N_{\text{env}} \times T}
$$

**Step 2 — 优势计算**：对每条轨迹执行 GAE（§2.2），得到 $\hat{A}_t$ 和回报目标 $\hat{R}_t$。

**Step 3 — 策略更新**：将 $\mathcal{D}$ 按序列片段（长度 $L$）划分 mini-batch，执行 $K$ 轮梯度下降：

$$
\theta \leftarrow \theta - \alpha_\theta \nabla_\theta \mathcal{L}(\theta), \quad \phi \leftarrow \phi - \alpha_\phi \nabla_\phi \mathcal{L}_{\text{critic}}(\phi)
$$

**Step 4 — 课程判定**：统计本轮成功率 $\eta$，更新滑动窗口，判断是否满足阶段跃迁条件（§4.3）。

**Step 5 — 指标记录**：分项奖励、成功率（按 affordance 类别）、杯体脱落率、平均 $s_t$ 值等关键指标送入 TensorBoard / WandB。

### 6.2 梯度裁剪

为防止训练不稳定，对 actor 和 critic 的梯度执行全局范数裁剪：

$$
\hat{g} = \begin{cases}
g & \text{if } \|g\| \leq g_{\max} \\
g_{\max} \cdot \frac{g}{\|g\|} & \text{if } \|g\| > g_{\max}
\end{cases}
$$

默认 $g_{\max} = 1.0$。

---

## 7. 超参数一览

### 7.1 PPO 核心参数

| 参数 | 符号 | 含义 | 默认值 |
|------|------|------|--------|
| 折扣因子 | $\gamma$ | 未来奖励衰减 | 0.99 |
| GAE 参数 | $\lambda$ | 偏差-方差权衡 | 0.95 |
| Clip 参数 | $\epsilon$ | 策略比率裁剪范围 | 0.2 |
| Value clip | $\epsilon_v$ | Value function 裁剪范围 | 0.2 |
| 熵系数 | $c_e$ | 探索正则化强度 | 0.01 |
| Value loss 权重 | $c_v$ | Critic 损失权重 | 0.5 |
| 梯度裁剪 | $g_{\max}$ | 全局梯度范数上限 | 1.0 |
| 学习率（actor） | $\alpha_\theta$ | Actor 优化步长 | 3e-4 |
| 学习率（critic） | $\alpha_\phi$ | Critic 优化步长 | 3e-4 |
| Mini-batch 数量 | $N_{\text{mb}}$ | 每轮更新的 mini-batch 数 | 4 |
| 更新 epoch 数 | $K$ | 每次收集后的优化轮数 | 5 |

### 7.2 Rollout 参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $N_{\text{env}}$ | 并行环境数量 | 1024 |
| $T$ | 每轮收集的步数 | 24 |
| $L$ | TBPTT 截断长度 | 16 |

### 7.3 课程学习参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $M$ | 跃迁判据滑动窗口（epoch 数） | 50 |
| $\eta_{\text{thresh}}$ | 成功率跃迁阈值 | 0.8 |
| 总阶段数 | 课程阶段总数 | 5 |

### 7.4 域随机化范围

| 参数 | 范围 |
|------|------|
| $m_{\text{cup}}$ | $[0.1, 0.8]$ kg |
| $m_{\text{door}}$ | $[5.0, 20.0]$ kg |
| $d_{\text{hinge}}$ | $[0.5, 5.0]$ N·m·s/rad |
| $\Delta p$（基座位移半径） | $[0.02, 0.05]$ m（XY 平面，局限于运动学可达裕度内） |
| $\sigma_a$（动作噪声标准差） | 0.02 |
| $\sigma_o$（观测噪声标准差） | 0.01 |
