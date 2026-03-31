# rewards — 奖励函数设计

## 1. 本层在系统中的位置

rewards 层独立于环境与策略实现，在每个仿真步结束后、PPO 更新之前，从环境状态中计算标量奖励信号。它的唯一职责是把"任务做得好不好"和"持杯稳不稳"这两件事翻译成训练信号。

```
envs/ ──→ next_state, contact_events
  │
  ├── 任务进展（门角度、按钮状态 ...）
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
- 参考 SoFTA 框架，正奖励采用 $\exp(-\frac{1}{\sigma^2}\|\cdot\|^2)$ 的高斯核形式，惩罚项采用 $-\|\cdot\|^2$ 的二次形式

---

## 2. 总体奖励函数

每步的原始分项奖励分为正向激励（Bonus）和负向惩罚（Penalty）。为了防止训练初期策略陷入局部最优，所有的惩罚项都会经过动态缩放因子 $s_t$（见 §7）的调制再送入 PPO：

$$
r_t = r_{\text{task}} + m_L \cdot \left( r_{\text{stab\_bonus}}^L + s_t \cdot r_{\text{stab\_penalty}}^L \right) + m_R \cdot \left( r_{\text{stab\_bonus}}^R + s_t \cdot r_{\text{stab\_penalty}}^R \right) - r_{\text{safe}}
$$

关键要点：

- $m_L, m_R \in \{0, 1\}$ 为左右臂持杯 mask，由 `left_occupied` / `right_occupied` 决定
- 稳定性被拆分为正向高斯激励项 $r_{\text{stab\_bonus}}$ 与负向二次惩罚项 $r_{\text{stab\_penalty}}$（本身值为负）
- 所有的惩罚信号（包括稳定性惩罚和安全惩罚 $r_{\text{safe}}$）都会被 $s_t \in (0, 1]$ 当前的课程系数缩放，鼓励智能体先关注主任务，再优化平滑与安全

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

进展度量为门铰链角度增量，权重 $w(\theta_t)$ 是关于当前角度的**分段递减函数**——在到达目标角度前保持满额激励（不衰减），超过目标角度后开始逐步衰减，防止过度推门：

$$
r_{\text{task}} = w(\theta_t) \cdot (\theta_t - \theta_{t-1}) + w_{\text{open}} \cdot \mathbf{1}[\theta_t \geq \theta_{\text{target}}]
$$

其中：

$$
w(\theta_t) = \begin{cases}
w_{\delta} & \text{if } \theta_t \leq \theta_{\text{target}} \\
w_{\delta} \cdot \max\!\left(\alpha,\; 1 - k_{\text{decay}}(\theta_t - \theta_{\text{target}})\right) & \text{if } \theta_t > \theta_{\text{target}}
\end{cases}
$$

- $\theta_t \leq \theta_{\text{target}}$ 时：$w = w_\delta$（满额激励，不衰减）
- $\theta_t > \theta_{\text{target}}$ 时：权重由于超出目标角度而线性衰减，直到降至下限 $\alpha \cdot w_\delta$。$k_{\text{decay}}$ 控制衰减速率。

参数说明：

- $w_\delta$：基准角度增量奖励
- $\alpha \in (0, 1)$：衰减下限比例，控制到达目标后的权重保留比例
- $k_{\text{decay}}$：超出目标角度后的衰减速率系数
- $\theta_{\text{target}}$：任务完成目标角度
- $w_{\text{open}}$：完成 bonus（一次性）

---

## 5. 持杯稳定性奖励 $r_{\text{stab}}$

### 5.1 设计思路

参考 SoFTA 框架上半身智能体的末端稳定性奖励设计。核心思想是：杯体的稳定性由末端执行器（EE）的运动状态决定，通过**高斯核正奖励**鼓励趋近零状态，**二次惩罚**抑制过大运动量。正奖励与惩罚构成互补的双向梯度场。

所有指标基于 **gripper frame**（末端执行器坐标系）定义，通过速度差分估计加速度（与真实部署保持一致）。

对左右臂的计算方式完全对称，以下以单臂为例写出公式，左臂取 $(\cdot)^L$，右臂取 $(\cdot)^R$。

### 5.2 稳定性 proxy 中的关键量

| 量 | 定义 |
|----|------|
| $\mathbf{a}_t$ | EE 线加速度，差分估计：$\mathbf{a}_t = (\mathbf{v}_t - \mathbf{v}_{t-1}) / \Delta t$ |
| $\boldsymbol{\alpha}_t$ | EE 角加速度，差分估计：$\boldsymbol{\alpha}_t = (\boldsymbol{\omega}_t - \boldsymbol{\omega}_{t-1}) / \Delta t$ |
| $j_t$ | Jerk proxy：$j_t = \|\mathbf{a}_t - \mathbf{a}_{t-1}\| / \Delta t$ |
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

### 6.1 无效碰撞惩罚

对未落在有效 affordance 区域内的碰撞给予惩罚，鼓励策略只在任务相关区域产生接触：

$$
r_{\text{collision}} = \beta_1 \cdot \sum_{\text{link} \notin \text{affordance}} f_{\text{link}}
$$

其中 $f_{\text{link}}$ 为该 link 受到的接触力大小。

### 6.2 自碰撞惩罚

机器人任意两个 link 之间发生接触时给予固定惩罚：

$$
r_{\text{self}} = \beta_2 \cdot \mathbf{1}[\text{self-collision detected}]
$$

### 6.3 关节限位逼近惩罚

当关节角度逼近物理限位时施加递增惩罚，防止机构损伤。

**限位信息来源**：
每个关节的物理极限区间 $[q_i^{\min}, q_i^{\max}]$ 并非硬编码在奖励函数中，而是直接从机器人的底层资产文件中解析获得：
* URDF 源码定义：[`assets/robot/urdf/uni_dingo_dual_arm.urdf`](../../../assets/robot/urdf/uni_dingo_dual_arm.urdf) 中的 `<limit lower="..." upper="..."/>`
* USD 仿真模型：[`assets/robot/uni_dingo_dual_arm.usd`](../../../assets/robot/uni_dingo_dual_arm.usd)

在 Isaac Sim 仿真运行时，奖励模块会通过只读的物理抽象层（如 `ArticulationView.get_dof_limits()`）一次性获取这些真值。

设第 $i$ 关节的角度范围为 $[q_i^{\min}, q_i^{\max}]$，中心为 $q_i^c$，半范围为 $\delta_i = (q_i^{\max} - q_i^{\min}) / 2$，惩罚项的计算公式为：

$$
r_{\text{limit}} = \beta_3 \cdot \sum_{i=1}^{12} \max\!\left(0,\ \left| q_i - q_i^c \right| - \mu \cdot \delta_i \right)^2
$$

其中 $\mu \in (0, 1)$ 为触发比例（如 0.9，即偏移量超过允许半范围的 90% 时，才开始产生向上生长的二次幂惩罚）。

### 6.4 关节速度过大惩罚

与角度极限类似，防止过快运动超出关节减速器与电机的额定承载能力。

**限速信息来源**：
每个关节的物理最大转速 $\dot{q}_i^{\max}$ 同样并非随意选取，而是提取自底层的机器人资产：
* URDF 源码定义：[`assets/robot/urdf/uni_dingo_dual_arm.urdf`](../../../assets/robot/urdf/uni_dingo_dual_arm.urdf) 中的 `<limit ... velocity="..."/>` 所填写的最大速率。
* USD 仿真模型：[`assets/robot/uni_dingo_dual_arm.usd`](../../../assets/robot/uni_dingo_dual_arm.usd) 中绑定的 `maxJointVelocity` 属性。

在运行时，系统通过物理引擎接口（如 `ArticulationView.get_dof_max_velocities()`）直接获取这组速度上限，并以此设置速度阈值 $v_{\text{thresh}}^{(i)}$ （通常取物理极限的一个安全系数比如 0.9）。

$$
r_{\text{vel}} = \beta_4 \cdot \sum_{i=1}^{12} \max\!\left(0,\ |\dot{q}_i| - \mu \cdot \dot{q}_i^{\max} \right)^2
$$

超过该动态截断阈值的部分以平方惩罚，阈值内不惩罚。这样当更换其他具有不同传动比的机械臂配置时，无需修改训练超参。

### 6.5 杯体脱落惩罚

如果机械臂或环境的交互动作导致杯子不可逆地脱离了末端原有的持握状态（例如掉落、被外力撞飞），系统会给出极其严厉的单次截断性惩罚，以保证模型学会安全稳定的接触操作：

$$
r_{\text{drop}} = -w_{\text{drop}} \cdot \mathbf{1}[\text{cup\_dropped}]
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
| $\theta_t, \theta_{t-1}$，按钮/把手状态 | `envs/` 仿真 oracle | `task_reward.py` |
| $\mathbf{a}_t, \boldsymbol{\alpha}_t, \text{tilt}$ | `actor_obs["left/right_stability_proxy"]` | `stability_reward.py` |
| $m_L, m_R$ | `actor_obs["context"]` | `reward_manager.py` |
| $\boldsymbol{\tau}_t, \boldsymbol{\tau}_{t-1}$ | `policy/` 策略输出（调用方传入） | `stability_reward.py` |
| $\mathbf{q}, \dot{\mathbf{q}}$（双臂 12 维） | `envs/` 仿真 oracle | `safety_penalty.py` |
| 接触事件、自碰撞标志 | `envs/` 仿真 `ContactMonitor` | `safety_penalty.py` |
| Episode Length | `envs/` 仿真计时器 | `reward_manager.py`（动态缩放） |

注意：rewards 层消费仿真地面真值（不依赖 critic 的 privileged 信息），这是因为奖励计算在训练时由仿真环境直接调用，可以访问精确状态。

---

## 9. 权重参数一览

### 9.1 主任务权重

| 参数 | 含义 | 默认值 |
|------|------|--------|
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
| $w_{\text{term}}$ | 终止惩罚 | 100.0 |

### 9.3 安全惩罚权重

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $\beta_1$ | 无效碰撞系数 | 0.5 |
| $\beta_2$ | 自碰撞固定惩罚 | 1.0 |
| $\beta_3$ | 关节限位系数 | 0.1 |
| $\mu$ | 关节限位触发比例 | 0.9 |
| $\beta_4$ | 关节速度系数 | 0.01 |
| $v_{\text{thresh}}$ | 速度惩罚阈值（rad/s） | 5.0 |
| $\beta_5$ | gripper 冲击系数 | 0.5 |

### 9.4 动态缩放参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $s_{\text{init}}$ | 惩罚缩放初始值 | 0.5 |
| $\rho_{\text{decay}}$ | 短 episode 衰减率 | 0.9999 |
| $\rho_{\text{grow}}$ | 长 episode 增长率 | 1.0001 |
| $T_{\text{short}}$ | 短 episode 阈值 | 0.4 s |
| $T_{\text{long}}$ | 长 episode 阈值 | 2.1 s |

---

## 10. 分项监控

`RewardManager` 每步输出完整的分项字典供 TensorBoard / WandB 监控：

```
total_reward
├── task_progress
│   ├── task/door_angle_delta
│   ├── task/contact_bonus
│   └── task/success_bonus
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
│   ├── safety/invalid_collision
│   ├── safety/self_collision
│   ├── safety/joint_limit
│   ├── safety/velocity
│   └── safety/gripper_impact
└── scaling/s_current                          ← 动态缩放因子当前值
```

分项日志使得每个子项的贡献在训练过程中可观测，便于诊断策略行为的偏差来源。
