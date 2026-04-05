# rewards — 奖励函数设计

## 1. 本层在系统中的位置

rewards 层独立于环境与策略实现，在每个仿真步结束后、PPO 更新之前，从环境状态中计算标量奖励信号。它的唯一职责是把"任务做得好不好"和"持杯稳不稳"这两件事翻译成训练信号。

```
envs/ ──→ next_state, contact_events
  │
  ├── 任务进展（门角度、回合结果事件 ...）
  ├── 接触事件（冲击力、自碰撞 ...）
  └── 精确物理状态（杯体位姿、双臂关节状态 ...）
          │
          ▼
    ┌────────────────────────┐
    │  rewards/              │  ◄── 本层
    │  task_reward.py        │  主任务进展奖励
    │  stability_reward.py   │  持杯稳定性奖励
    │  safety_penalty.py     │  安全惩罚
    │  reward_manager.py     │  聚合 + 日志
    └────────────────────────┘
             │  r_total, reward_info dict
             ▼
       training/ (PPO update)
```

**设计核心原则**：

- 奖励"在约束下正确交互"，而不是"用某个特定身体部位"
- 所有稳定性约束通过持杯 mask 条件化激活，不持杯时不施加末端约束
- 训练时直接使用仿真地面真值计算奖励，不依赖 actor 观测
- 稳定性 penalty 在分支内部采用负向二次形式；安全惩罚则统一作为正惩罚量汇总进 `r_safe`，再由总式统一扣除

---

## 2. 总体奖励函数

每步的原始分项奖励分为正向激励（Bonus）和负向惩罚（Penalty）。为了防止训练初期策略陷入局部最优，只有稳定性分支中的负向惩罚项会经过动态缩放因子 $s_t$（见 §7）的调制后再送入 PPO；安全惩罚始终以原值计入总奖励：

$$
r_t = r_{\text{task}} + m_L \cdot \left( r_{\text{stab\_bonus}}^L + s_t \cdot r_{\text{stab\_penalty}}^L \right) + m_R \cdot \left( r_{\text{stab\_bonus}}^R + s_t \cdot r_{\text{stab\_penalty}}^R \right) - r_{\text{safe}}
$$

关键要点：

- $m_L, m_R \in \{0, 1\}$ 为左右臂持杯 mask，由 `left_occupied` / `right_occupied` 决定
- 稳定性被拆分为正向高斯激励项 $r_{\text{stab\_bonus}}$ 与负向二次惩罚项 $r_{\text{stab\_penalty}}$（本身值为负）
- $s_t \in (0, 1]$ 只作用于稳定性分支中的负向项 $r_{\text{stab\_penalty}}$
- 安全惩罚 $r_{\text{safe}}$ 不受 $s_t$ 影响，始终按原值从总奖励中扣除
- `r_safe` 内部的各个子项统一按“正惩罚量”定义，避免符号歧义

---

## 3. 持杯 mask 与稳定性调制

### 3.1 双臂 occupancy mask

持杯是随机事件，有四种情况：

| $m_L$ | $m_R$ | 含义 |
|-------|-------|------|
| 0 | 0 | 双臂空闲，无稳定性约束 |
| 1 | 0 | 左臂持杯，约束左臂末端 |
| 0 | 1 | 右臂持杯，约束右臂末端 |
| 1 | 1 | 双臂持杯，同时约束两侧 |



---

## 4. 主任务奖励 $r_{\text{task}}$

### 4.1 设计思路

现阶段主任务为**推门**。奖励函数奖励门角度的进展本身而不是特定的接触方式，确保策略自行发现有效的推门策略。

奖励由**基于进展增量的稠密奖励**加**成功 bonus** 构成，避免稀疏奖励带来的探索困难。

### 4.2 推门奖励

进展度量为门铰链角度增量，权重 $w(\theta_t)$ 是关于当前角度的**分段递减函数**。按当前目标定义，reward 层单独维护 success bonus 阈值 $\theta_{\text{reward\_success}} = 1.2\ \text{rad}$；它**不等同于** env / TaskManager 使用的 episode 结束阈值 `1.57 rad`。在到达奖励目标角度前保持满额激励（不衰减），超过奖励目标角度后开始逐步衰减，防止过度推门：

$$
r_{\text{task}} = w(\theta_t) \cdot (\theta_t - \theta_{t-1}) + w_{\text{open}} \cdot \mathbf{1}[\theta_t \geq \theta_{\text{reward\_success}}]
$$

其中：

$$
w(\theta_t) = \begin{cases}
 w_{\delta} & \text{if } \theta_t \leq \theta_{\text{reward\_success}} \\
 w_{\delta} \cdot \max\!\left(\alpha,\; 1 - k_{\text{decay}}(\theta_t - \theta_{\text{reward\_success}})\right) & \text{if } \theta_t > \theta_{\text{reward\_success}}
\end{cases}
$$

- $\theta_t \leq \theta_{\text{reward\_success}}$ 时：$w = w_\delta$（满额激励，不衰减）
- $\theta_t > \theta_{\text{reward\_success}}$ 时：权重由于超出奖励目标角度而线性衰减，直到降至下限 $\alpha \cdot w_\delta$。$k_{\text{decay}}$ 控制衰减速率。

参数说明：

- $w_\delta$：基准角度增量奖励
- $\alpha \in (0, 1)$：衰减下限比例，控制到达目标后的权重保留比例
- $k_{\text{decay}}$：超出目标角度后的衰减速率系数
- $\theta_{\text{reward\_success}}$：reward 层 success bonus 触发角度，目标值为 `1.2 rad`
- $w_{\text{open}}$：完成 bonus（一次性）

env / TaskManager 的 episode 结束角度由环境层单独维护，目标值为 `1.57 rad`。reward 文档中的成功阈值不再与 episode 结束阈值共用同一名字，以避免再次混淆。

---

## 5. 持杯稳定性奖励 $r_{\text{stab}}$

### 5.1 设计思路

参考 SoFTA 框架上半身智能体的末端稳定性奖励设计。核心思想是：杯体的稳定性由末端执行器（EE）的运动状态决定，通过**高斯核正奖励**鼓励趋近零状态，**二次惩罚**抑制过大运动量。正奖励与惩罚构成互补的双向梯度场。

所有指标基于 **gripper frame**（末端执行器坐标系）定义。`lin_acc` / `ang_acc` 由环境层直接读取 Isaac Sim / Isaac Lab 提供的末端原生加速度数据，`tilt` 由末端姿态与重力方向几何推理得到。

对左右臂的计算方式完全对称，以下以单臂为例写出公式，左臂取 $(\cdot)^L$，右臂取 $(\cdot)^R$。

### 5.2 稳定性 proxy 中的关键量

| 量 | 定义 |
|----|------|
| $\mathbf{a}_t$ | EE 线加速度，由环境层直接读取 Isaac Sim / Isaac Lab 原生刚体 / link 加速度接口 |
| $\boldsymbol{\alpha}_t$ | EE 角加速度，由环境层直接读取 Isaac Sim / Isaac Lab 原生刚体 / link 加速度接口 |
| $j_t$ | Jerk proxy，可由环境侧共享稳定性 proxy 基于连续两步加速度进一步构造 |
| $\text{tilt}$ | 杯体倾斜度，见 §5.3 |

### 5.3 杯体倾斜度的几何推理

杯体是否倾倒取决于杯口法线相对重力方向的偏转角。设重力方向为 $\mathbf{g} = [0, 0, -9.81]^\top$，gripper 旋转矩阵为 $R_{EE}$，则重力在 EE 局部坐标系中的表达为：

$$
\mathbf{g}_{\text{local}} = R_{EE}^\top \mathbf{g}
$$

取其 $xy$ 分量的模长即为倾斜度：

$$
\text{tilt} = \| P_{xy}(\mathbf{g}_{\text{local}}) \| = \sqrt{g_x^2 + g_y^2}
$$

直觉：若 gripper 保持竖直（杯口朝上），重力在 EE 坐标系中完全沿 $-z$ 轴，$xy$ 分量为零，tilt = 0；gripper 越倾斜，tilt 越大。

### 5.4 稳定性奖励子项

参考 SoFTA 上半身智能体的奖励设计。

#### (1) 零线加速度奖励

鼓励末端三维线加速度趋近于 0（$\lambda_{acc} = 0.25$）：

$$
r_{\text{zero-acc}} = w_{\text{zero-acc}} \cdot \exp\!\left( -\lambda_{acc} \cdot \|\mathbf{a}_t\|^2 \right)
$$

#### (2) 零角加速度奖励

鼓励末端三维角加速度趋近于 0（$\lambda_{ang} = 0.0044$）：

$$
r_{\text{zero-ang}} = w_{\text{zero-ang}} \cdot \exp\!\left( -\lambda_{ang} \cdot \|\boldsymbol{\alpha}_t\|^2 \right)
$$

#### (3) 线加速度惩罚

二次惩罚抑制末端高加速度冲击：

$$
r_{\text{acc}} = -w_{\text{acc}} \cdot \|\mathbf{a}_t\|^2
$$

#### (4) 角加速度惩罚

$$
r_{\text{ang}} = -w_{\text{ang}} \cdot \|\boldsymbol{\alpha}_t\|^2
$$

#### (5) 重力倾斜惩罚

**极度惩罚 EE 倾斜**，确保液体不洒：

$$
r_{\text{tilt}} = -w_{\text{tilt}} \cdot \|P_{xy}(R_{EE}^\top \mathbf{g})\|^2
$$

#### (6) 力矩变化平滑项

相邻两步力矩的变化量反映控制信号的平滑程度，抑制抖动和高频振荡：

$$
r_{\text{smooth}} = -w_{\text{smooth}} \cdot \|\boldsymbol{\tau}_t - \boldsymbol{\tau}_{t-1}\|^2
$$

其中 $\boldsymbol{\tau}_t \in \mathbb{R}^{12}$ 为双臂完整力矩输出。

#### (7) 力矩幅值正则项

防止策略输出过大的力矩，起隐式能量约束的作用：

$$
r_{\text{reg}} = -w_{\text{reg}} \cdot \|\boldsymbol{\tau}_t\|^2
$$

### 5.5 单臂稳定性奖励分类汇总（对应第 2 章）

在第 2 章的总奖励公式中，单臂的稳定性项被明确划分为不受衰减的正向激励 $r_{\text{stab\_bonus}}$，以及受动态缩放因子 $s_t$ 调制的负向惩罚 $r_{\text{stab\_penalty}}$。

根据上述 7 个子项的物理性质，它们的对应划分关系如下：

**(A) 正向激励项 (Bonus)**
包含所有采用高斯核的项，鼓励末端运动状态趋近于完美的绝对静止：
$$
r_{\text{stab\_bonus}}^{(\cdot)} = r_{\text{zero-acc}} + r_{\text{zero-ang}}
$$

**(B) 负向惩罚项 (Penalty)**
包含所有采用二次形式的项，用于抑制大范围的剧烈运动、高耗能抖动与不良姿态：
$$
r_{\text{stab\_penalty}}^{(\cdot)} = r_{\text{acc}} + r_{\text{ang}} + r_{\text{tilt}} + r_{\text{smooth}} + r_{\text{reg}}
$$

> **说明**：子项 (1)(3) 即 $r_{\text{zero-acc}}$ 与 $r_{\text{acc}}$，以及子项 (2)(4) 即 $r_{\text{zero-ang}}$ 与 $r_{\text{ang}}$，构成了典型的**惩罚-奖励对**。惩罚项在大运动时主导并强力抑制冲击，奖励项在小运动时主导并激励趋近绝对平稳，两者共同构造出了以 0 为稳态的双向梯度场。

### 5.6 双臂稳定性奖励与 occupancy mask

由于单臂部分的稳定性约束被拆分为了 Bonus 和 Penalty 两组，在聚合为最终的双臂奖励时，系统会使用持杯 mask ($m_L$, $m_R$) 对这二者分别进行掩码（即第 2 章的总体联合公式）。

这带来的直接效果是：
- **双臂持杯时**：两套 proxy 的 Bonus 和 Penalty 项均满额激活，策略必须同时维持两只杯子的平稳。
- **单臂持杯时**：仅持杯侧的相关惩罚和激励生效，空闲臂可以不受平稳性约束地自由移动。
- **完全空手时**：$m_L = m_R = 0$，$r_{\text{stab\_bonus}}$ 与 $r_{\text{stab\_penalty}}$ 在两侧全局关闭。策略不会受到任何关于末端加速度和倾角限制的惩罚（但安全的底线惩罚 $r_{\text{safe}}$ 仍在），此时它可以展现出极高速度或大爆发力的交互尝试。

---

## 6. 安全惩罚 $r_{\text{safe}}$

安全惩罚项**始终激活**，不受持杯状态影响。奖励函数不鼓励危险行为，无论手中是否持杯。

本节统一采用以下约定：

- `r_safe` 是正惩罚量的总和
- 各个安全子项都按正数幅值定义
- 最终总奖励通过 `- r_safe` 统一扣除

当前任务是显式接触任务，末端与门板/把手发生接触属于完成任务的必要条件。因此安全项不再直接惩罚接触力，而只保留与设备保护和持杯安全直接相关的约束。需要特别区分：仿真执行前的力矩 clip 由 env / 仿真层负责，而“力矩超限惩罚”基于 policy 原始输出的控制力矩计算，用于惩罚超限控制意图。

### 6.1 自碰撞惩罚

机器人任意两个 link 之间发生接触时给予固定惩罚：

$$
r_{\text{self}} = \beta_1 \cdot \mathbf{1}[\text{self-collision detected}]
$$

### 6.2 关节限位逼近惩罚

当关节角度逼近物理限位时施加递增惩罚，防止机构损伤。

**限位信息来源**：
每个关节的物理极限区间 $[q_i^{\min}, q_i^{\max}]$ 并非硬编码在奖励函数中，而是直接从机器人的底层资产文件中解析获得：
* URDF 源码定义：[`assets/robot/urdf/uni_dingo_dual_arm.urdf`](../../../assets/robot/urdf/uni_dingo_dual_arm.urdf) 中的 `<limit lower="..." upper="..."/>`
* USD 仿真模型：[`assets/robot/uni_dingo_dual_arm.usd`](../../../assets/robot/uni_dingo_dual_arm.usd)

在 Isaac Sim 仿真运行时，奖励模块会通过只读的物理抽象层（如 `ArticulationView.get_dof_limits()`）一次性获取这些真值。

设第 $i$ 关节的角度范围为 $[q_i^{\min}, q_i^{\max}]$，中心为 $q_i^c$，半范围为 $\delta_i = (q_i^{\max} - q_i^{\min}) / 2$，惩罚项的计算公式为：

$$
r_{\text{limit}} = \beta_2 \cdot \sum_{i=1}^{12} \max\!\left(0,\ \left| q_i - q_i^c \right| - \mu \cdot \delta_i \right)^2
$$

其中 $\mu \in (0, 1)$ 为触发比例（如 0.9，即偏移量超过允许半范围的 90% 时，才开始产生向上生长的二次幂惩罚）。

### 6.3 关节速度过大惩罚

与角度极限类似，防止过快运动超出关节减速器与电机的额定承载能力。

**限速信息来源**：
每个关节的物理最大转速 $\dot{q}_i^{\max}$ 同样并非随意选取，而是提取自底层的机器人资产：
* URDF 源码定义：[`assets/robot/urdf/uni_dingo_dual_arm.urdf`](../../../assets/robot/urdf/uni_dingo_dual_arm.urdf) 中的 `<limit ... velocity="..."/>` 所填写的最大速率。
* USD 仿真模型：[`assets/robot/uni_dingo_dual_arm.usd`](../../../assets/robot/uni_dingo_dual_arm.usd) 中绑定的 `maxJointVelocity` 属性。

在运行时，系统通过物理引擎接口（如 `ArticulationView.get_dof_max_velocities()`）直接获取这组速度上限，并以此设置速度阈值 $v_{\text{thresh}}^{(i)}$ （通常取物理极限的一个安全系数比如 0.9）。

$$
r_{\text{vel}} = \beta_3 \cdot \sum_{i=1}^{12} \max\!\left(0,\ |\dot{q}_i| - \mu \cdot \dot{q}_i^{\max} \right)^2
$$

超过该动态截断阈值的部分以平方惩罚，阈值内不惩罚。这样当更换其他具有不同传动比的机械臂配置时，无需修改训练超参。

### 6.4 原始控制力矩超限惩罚

设 policy 在当前步输出的原始控制力矩为 $\boldsymbol{\tau}^{\text{raw}}_t$，每个关节的力矩上限为 $\boldsymbol{\tau}^{\max}$。env 会在送入物理引擎前执行：

$$
\boldsymbol{\tau}^{\text{applied}}_t = \text{clip}\!\left(\boldsymbol{\tau}^{\text{raw}}_t,\; -\boldsymbol{\tau}^{\max},\; \boldsymbol{\tau}^{\max}\right)
$$

但安全检测不会使用 $\boldsymbol{\tau}^{\text{applied}}_t$ 掩盖超限行为，而是直接对原始输出中的超限部分施加惩罚：

$$
r_{\text{torque}} = \beta_4 \cdot \sum_{i=1}^{12} \max\!\left(0,\ |\tau^{\text{raw}}_i| - \tau^{\max}_i \right)^2
$$

这样可以同时满足两点：

- 物理仿真始终安全，只执行 clip 后的力矩；
- 策略如果持续输出越界力矩，仍会在奖励层受到明确惩罚。

### 6.5 杯体脱落惩罚

如果机械臂或环境的交互动作导致杯子不可逆地脱离了末端原有的持握状态（例如掉落、被外力撞飞），系统会给出极其严厉的单次截断性惩罚，以保证模型学会安全稳定的接触操作：

$$
r_{\text{drop}} = w_{\text{drop}} \cdot \mathbf{1}[\text{cup\_dropped}]
$$

脱落触发后当前 episode 直接终止。

---

## 7. 动态奖励缩放机制（课程学习）

参考 SoFTA 框架的动态奖励缩放策略。训练初期如果全额施加惩罚，策略容易陷入局部最优（如干脆不动以避免加速度惩罚）。为此系统引入了**全局动态缩放因子 $s_t$**，专用于对负向惩罚项施加从轻到重的动态课程学习：

### 7.1 缩放规则

如第 2 章的总公式所示，缩放因子 $s_t$ 仅作用于负向惩罚逻辑（$r_{\text{stab\_penalty}}$）：

$$
r_{t}^{\text{stab\_total}} = r_{\text{stab\_bonus}} + s_t \cdot r_{\text{stab\_penalty}}
$$

在这里，所有的正向稳定激励（Bonus）始终被施加系数 1 的无损保留，而各种高耗能或剧烈动作带来的惩罚（Penalty）则全部统一接受 $s_t$ 的削弱或增强控制。

### 7.2 $s_t$ 的自适应调节（基于全局训练步数的退火）

由于复杂的物理交互任务往往具有**不定长**的特征（早期频繁因碰撞终止，后期策略熟练后交互可能极其迅速），依然依靠单回合的绝对存活时长去动态增减边界惩罚，极易导致策略的训练目标发生剧烈震荡。因此，我们全面采用了更为鲁棒的**基于全局训练步数（Global Environment Steps）的线性退火策略**。

系统不再关注某一局智能体活了多久，而是读取底层环境的总累积推演步数 $N_{\text{step}}$。我们预设一个较长的退火窗口期 $N_{\text{anneal}}$（例如设定为前 $1.0 \times 10^7$ 步），此时 $s_t$ 的调节公式将纯粹依附时间轴展开：

$$
s_t = s_{\min} + (1.0 - s_{\min}) \cdot \min\!\left(1.0,\; \frac{N_{\text{step}}}{N_{\text{anneal}}}\right)
$$

其中，**初始惩罚基数** $s_{\min}$ 通常设为极低的数值（例如 0.1 或 0.0）：
1. **探索发力期（极低惩罚）**：在 $N_{\text{step}} < N_{\text{anneal}}$ 的漫长窗口中，$s_t$ 不受任何回合中途崩溃扰动地、呈线性平缓地向 $1.0$ 爬坡。在这个阶段，机器人在面临微乎其微的运动学约束的条件下，拥有极大的自由度去探索“如何发力、“如何将距离迫近门把手”以及“如何推门”等**强目标导向（Task-Oriented）**的动作策略。
2. **平滑收敛期（全额惩罚）**：当 $N_{\text{step}} \geq N_{\text{anneal}}$ 后，$s_t$ 固定在峰值 $1.0$ 封顶。此时模型已经彻底学会了怎么完成这件主任务，系统转而开始严抓**操作平稳度**，迫使它抹平因野蛮发力带来的任何一丝高频抖动和非受控的冲击，进而打磨出具备极强抗干扰能力的顺滑动作。

---

## 8. 奖励数据来源

rewards 层从以下来源获取输入，自身不维护跨步状态：

| 数据 | 来源 | 消费者 |
|------|------|--------|
| $\theta_t, \theta_{t-1}$，以及成功/终止相关事件 | `envs/` 仿真 oracle | `task_reward.py` |
| $\mathbf{a}_t, \boldsymbol{\alpha}_t, \text{tilt}$ | `envs/` 环境侧稳定性 proxy（基于 Isaac Sim / Isaac Lab 原生加速度 + 姿态） | `stability_reward.py` |
| $m_L, m_R$ | `envs/` 持杯状态标记（`left_occupied` / `right_occupied`） | `reward_manager.py` |
| $\boldsymbol{\tau}^{\text{raw}}_t$ | `policy/` 原始控制力矩输出（clip 前） | `safety_penalty.py` |
| $\boldsymbol{\tau}^{\text{applied}}_t, \boldsymbol{\tau}^{\text{applied}}_{t-1}$ | `envs/` 裁剪后实际执行的力矩 | `stability_reward.py` |
| $\boldsymbol{\tau}^{\max}$ | `envs/` 关节力矩上限（effort limits） | `safety_penalty.py` |
| $\mathbf{q}, \dot{\mathbf{q}}$（双臂 12 维） | `envs/` 仿真 oracle | `safety_penalty.py` |
| 自碰撞标志、杯体脱落标志 | `envs/` 仿真 `ContactMonitor` | `safety_penalty.py` |
| Episode Length | `envs/` 仿真计时器 | `reward_manager.py`（动态缩放） |

注意：rewards 层消费仿真地面真值（不依赖 critic 的 privileged 信息），这是因为奖励计算在训练时由仿真环境直接调用，可以访问精确状态。

---

## 9. 权重参数一览

### 9.1 主任务权重

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $\theta_{\text{reward\_success}}$ | success bonus 触发角度 | 1.2 rad |
| $w_\delta$ | 角度增量基准奖励 | 8.0 |
| $\alpha$ | 权重衰减下限比例 | 0.25 |
| $k_{\text{decay}}$ | 超出目标角度后的衰减速率 | 2.0 |
| $w_{\text{open}}$ | 任务成功 bonus | 10.0 |

### 9.2 稳定性权重（SoFTA 对齐）

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $w_{\text{zero-acc}}$ | 零线加速度奖励权重 | 1.0 |
| $\lambda_{acc}$ | 零线加速度高斯核衰减率 | 0.25 |
| $w_{\text{zero-ang}}$ | 零角加速度奖励权重 | 1.5 |
| $\lambda_{ang}$ | 零角加速度高斯核衰减率 | 0.0044 |
| $w_{\text{acc}}$ | 线加速度惩罚系数 | 0.10 |
| $w_{\text{ang}}$ | 角加速度惩罚系数 | 0.01 |
| $w_{\text{tilt}}$ | 重力倾斜惩罚系数 | 5.0 |
| $w_{\text{smooth}}$ | 力矩平滑惩罚系数 | 0.01 |
| $w_{\text{reg}}$ | 力矩正则惩罚系数 | 0.001 |

### 9.3 安全惩罚权重

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $\beta_1$ | 自碰撞固定惩罚 | 1.0 |
| $\beta_2$ | 关节限位系数 | 0.1 |
| $\mu$ | 关节限位触发比例 | 0.9 |
| $\beta_3$ | 关节速度系数 | 0.01 |
| $\beta_4$ | 原始控制力矩超限系数 | 0.01 |
| $w_{\text{drop}}$ | 杯体脱落惩罚 | 100.0 |

### 9.4 动态缩放参数 (全局退火)

| 参数 | 符号 | 含义 | 默认值 |
|------|------|------|--------|
| 退火窗口期 | $N_{\text{anneal}}$ | 惩罚因子达到最大值所需的总训练步数 | $1.0 \times 10^7$ |
| 初始惩罚基准 | $s_{\min}$ | 训练初期的最小稳定性惩罚乘数（探索期） | 0.1 |
| 峰值惩罚乘数 | $s_{\max}$ | 训练后期的最大惩罚乘数（收敛期） | 1.0 |

---

## 10. 分项监控

`RewardManager` 每步输出完整的分项字典供 TensorBoard / WandB 监控：

```
total_reward
├── task_progress
│   ├── task/door_angle_delta
│   ├── task/success_bonus
│   ├── task/weight
│   └── task/theta_t
├── stability_total
│   ├── stability/left_zero_acc
│   ├── stability/left_zero_ang_acc
│   ├── stability/left_acc
│   ├── stability/left_ang_acc
│   ├── stability/left_tilt
│   ├── stability/right_zero_acc
│   ├── stability/right_zero_ang_acc
│   ├── stability/right_acc
│   ├── stability/right_ang_acc
│   ├── stability/right_tilt
│   └── stability/torque_smooth + torque_reg   ← 双臂共用
├── safety_total
│   ├── safety/self_collision
│   ├── safety/joint_limit
│   ├── safety/velocity
│   ├── safety/torque_over_limit
│   └── safety/cup_drop
└── scaling/s_t                               ← 动态缩放因子当前值
```

分项日志使得每个子项的贡献在训练过程中可观测，便于诊断策略行为的偏差来源。
