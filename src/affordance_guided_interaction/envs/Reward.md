# 奖励函数设计 — 数学参考文档

> **注意**: 本文档是奖励函数的数学参考。实际的奖励计算代码位于
> `DoorPushEnv._get_rewards()`（`door_push_env.py`）中，以 PyTorch tensor 操作实现。
> 训练入口默认从 `configs/reward/default.yaml` 读取奖励超参数，并由 `train.py` 注入到 `DoorPushEnvCfg`。
> `DoorPushEnvCfg` 中仍保留同名字段作为回退默认值；本文档不包含具体参数数值，仅描述数学公式与物理含义。

## 1. 本层在系统中的位置

奖励计算在每个仿真步结束后、PPO 更新之前执行，从环境状态中计算标量奖励信号。所有奖励逻辑集中在 `DoorPushEnv._get_rewards()` 方法（`door_push_env.py`）中，以纯 PyTorch tensor 操作实现，无独立模块文件。训练脚本运行时会把 `configs/reward/default.yaml` 中的值覆盖写入 `DoorPushEnvCfg`；若未经过这条配置注入路径，则环境回退到 `DoorPushEnvCfg` 内的默认值。

```
envs/ ──→ next_state, contact_events
  │
  ├── 任务进展（门角度、回合结果事件 ...）
  ├── 接触相关事件（如杯体脱落）
  └── 精确物理状态（杯体位姿、双臂关节状态 ...）
          │
          ▼
    ┌──────────────────────────────────┐
    │  DoorPushEnv._get_rewards()     │  ◄── 奖励计算入口
    │    ├─ §4  任务奖励 r_task       │
    │    ├─ §5  稳定性奖励 r_stab     │
    │    └─ §6  安全惩罚   r_safe     │
    └──────────────────────────────────┘
             │  r_total = r_task + r_stab - r_safe
             ▼
       training/ (PPO update)
```

**设计核心原则**：

- 奖励"在约束下正确交互"，而不是"用某个特定身体部位"
- 所有稳定性约束通过持杯 mask 条件化激活，不持杯时不施加末端约束
- 训练时直接使用仿真地面真值计算奖励，不依赖 actor 观测
- 稳定性分支内部同时包含高斯正项和二次负项，但在文档与日志中统一视为单侧稳定性奖励 `r_stab^L / r_stab^R` 的子项
- 安全惩罚统一作为正惩罚量汇总进 `r_safe`，再由总式统一扣除

---

## 2. 总体奖励函数

每步奖励由主任务项、左右臂稳定性项和安全惩罚构成。当前课程设计中，Stage 1 完全不激活稳定性项；进入持杯阶段后，仅持杯侧的稳定性项生效：

$$
r_t = r_{\text{task}} + m_L \cdot r_{\text{stab}}^L + m_R \cdot r_{\text{stab}}^R - r_{\text{safe}}
$$

其中单臂稳定性奖励定义为：

$$
r_{\text{stab}}^{(\cdot)} = r_{\text{zero-acc}} + r_{\text{zero-ang}} + r_{\text{acc}} + r_{\text{ang}} + r_{\text{tilt}} + r_{\text{smooth}} + r_{\text{reg}}
$$

关键要点：

- $m_L, m_R \in \{0, 1\}$ 为左右臂持杯 mask，由 `left_occupied` / `right_occupied` 决定
- 单臂稳定性奖励内部同时包含高斯正项与二次负项，但统一作为单一 `r_stab^L / r_stab^R` 记录与分析
- Stage 1 中 $m_L = m_R = 0$，稳定性分支整体关闭；因此 `r_stab^L`、`r_stab^R` 及其全部子项在 TensorBoard 中都应为 0
- 安全惩罚 $r_{\text{safe}}$ 始终按原值从总奖励中扣除
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

奖励由**基于进展增量的稠密奖励**、**一次性成功 bonus** 和 **接近门板大表面的 shaping 项**构成，既保留推门进展信号，也在早期探索阶段鼓励机械臂先靠近正确的交互区域。

### 4.2 推门奖励

进展度量为门铰链角度增量，权重 $w(\theta_t)$ 是关于当前角度的**分段递减函数**。按当前目标定义，reward 层单独维护 success bonus 阈值 $\theta_{\text{reward\_success}} = 1.2\ \text{rad}$；它**不等同于** env / TaskManager 使用的 episode 结束阈值 `1.57 rad`。同时新增一个仅在推门前期激活的接近门奖励：

$$
r_{\text{task}} = w(\theta_t) \cdot (\theta_t - \theta_{t-1}) + w_{\text{open}} \cdot \mathbf{1}[\theta_t \geq \theta_{\text{reward\_success}}] + \mathbf{1}[\theta_t < \theta_{\text{stop}}] \cdot w_{\text{approach}} \cdot r_{\text{approach}, t}
$$

其中：

$$
w(\theta_t) = \begin{cases}
 w_{\delta} & \text{if } \theta_t \leq \theta_{\text{reward\_success}} \\
 w_{\delta} \cdot \max\!\left(\alpha,\; 1 - k_{\text{decay}}(\theta_t - \theta_{\text{reward\_success}})\right) & \text{if } \theta_t > \theta_{\text{reward\_success}}
\end{cases}
$$

$$
r_{\text{approach}, t} = \max\!\left(1 - \frac{a_t^2}{b^2 + \varepsilon},\; 0\right)
$$

$$
a_t = \min_{\mathbf{x} \in \mathcal{A}_t,\; \mathbf{y} \in \mathcal{D}_t} \|\mathbf{x} - \mathbf{y}\|_2,
\qquad
b = \min_{\mathbf{x} \in \mathcal{A}_0,\; \mathbf{y} \in \mathcal{D}_0} \|\mathbf{x} - \mathbf{y}\|_2
$$

当前最小实现中，$\mathcal{A}_t = \{\mathbf{p}_{\mathrm{ee}}^L(t), \mathbf{p}_{\mathrm{ee}}^R(t)\}$，即左右末端执行器控制点；$\mathcal{D}_t$ 为门板 `Panel` 推门侧的大矩形表面，而不是 `PushPlate` 的单点近似。

- $\theta_t \leq \theta_{\text{reward\_success}}$ 时：$w = w_\delta$（满额激励，不衰减）
- $\theta_t > \theta_{\text{reward\_success}}$ 时：权重由于超出奖励目标角度而线性衰减，直到降至下限 $\alpha \cdot w_\delta$
- $\theta_t \geq \theta_{\text{stop}}$ 时：关闭接近奖励，避免学成“贴着门站着不推”

参数说明：

- $w_\delta$：基准角度增量奖励
- $\alpha \in (0, 1)$：衰减下限比例，控制到达目标后的权重保留比例
- $k_{\text{decay}}$：超出目标角度后的衰减速率系数
- $\theta_{\text{reward\_success}}$：reward 层 success bonus 触发角度，目标值为 `1.2 rad`
- $w_{\text{open}}$：完成 bonus（一次性）
- $w_{\text{approach}}$：接近门板大表面的奖励权重
- $\varepsilon$：归一化平方距离公式的稳定项
- $\theta_{\text{stop}}$：关闭接近奖励的门角度阈值

env / TaskManager 的 episode 结束角度由环境层单独维护，目标值为 `1.57 rad`。reward 文档中的成功阈值不再与 episode 结束阈值共用同一名字，以避免再次混淆。


---

## 5. 持杯稳定性奖励 $r_{\text{stab}}$

### 5.1 设计思路

参考 SoFTA 框架上半身智能体的末端稳定性奖励设计。核心思想是：杯体的稳定性由末端执行器（EE）的运动状态决定，通过**高斯核正奖励**鼓励趋近零状态，**二次惩罚**抑制过大运动量。正奖励与惩罚构成互补的双向梯度场。

所有指标基于 **gripper frame**（末端执行器坐标系）定义。`lin_acc` / `ang_acc` 通过连续两步 EE 速度的数值微分计算，`tilt` 由末端姿态与重力方向几何推理得到。

对左右臂的计算方式完全对称，以下以单臂为例写出公式，左臂取 $(\cdot)^L$，右臂取 $(\cdot)^R$。

### 5.2 稳定性 proxy 中的关键量

| 量 | 定义 |
|----|------|
| $\mathbf{a}_t$ | EE 线加速度，通过连续两步 EE 线速度的数值微分 $\mathbf{a}_t = (\mathbf{v}_t - \mathbf{v}_{t-1}) / \Delta t$ 计算 |
| $\boldsymbol{\alpha}_t$ | EE 角加速度，通过连续两步 EE 角速度的数值微分 $\boldsymbol{\alpha}_t = (\boldsymbol{\omega}_t - \boldsymbol{\omega}_{t-1}) / \Delta t$ 计算 |
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

鼓励末端三维线加速度趋近于 0：

$$
r_{\text{zero-acc}} = w_{\text{zero-acc}} \cdot \exp\!\left( -\lambda_{acc} \cdot \|\mathbf{a}_t\|^2 \right)
$$

#### (2) 零角加速度奖励

鼓励末端三维角加速度趋近于 0：

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

其中 $\boldsymbol{\tau}_t \in \mathbb{R}^6$ 为该侧臂经 clip + action noise 后的控制力矩。力矩来源为 `_prev_action`（当前步）和 `_prev_prev_action`（上一步），均为经过力矩裁剪和动作噪声注入后的值。

#### (7) 力矩幅值正则项

防止策略输出过大的力矩，起隐式能量约束的作用：

$$
r_{\text{reg}} = -w_{\text{reg}} \cdot \|\boldsymbol{\tau}_t\|^2
$$

其中 $\boldsymbol{\tau}_t \in \mathbb{R}^6$ 为该侧臂经 clip + action noise 后的控制力矩，与平滑项使用相同的力矩来源。

### 5.5 单臂稳定性奖励汇总（对应第 2 章）

在第 2 章的总奖励公式中，单臂稳定性项不再拆分为 bonus 和 penalty 两组，而是统一写作单一奖励：

$$
r_{\text{stab}}^{(\cdot)} = r_{\text{zero-acc}} + r_{\text{zero-ang}} + r_{\text{acc}} + r_{\text{ang}} + r_{\text{tilt}} + r_{\text{smooth}} + r_{\text{reg}}
$$

这个写法与代码实现保持一致：高斯正项和二次负项都作为 `r_stab^(·)` 的子项参与求和，然后在左右臂层面分别应用 occupancy mask。

> **说明**：子项 $r_{\text{zero-acc}}$ / $r_{\text{acc}}$ 与 $r_{\text{zero-ang}}$ / $r_{\text{ang}}$ 仍然构成典型的“稳态吸引”组合，但在日志和分析层面统一归入单侧稳定性奖励，不再单独维护 `stab_bonus` / `stab_penalty` 概念。

### 5.6 双臂稳定性奖励与 occupancy mask

最终的双臂稳定性奖励写作：

$$
r_{\text{stab}} = m_L \cdot r_{\text{stab}}^L + m_R \cdot r_{\text{stab}}^R
$$

这带来的直接效果是：
- **双臂持杯时**：左右两侧的 7 个稳定性子项都满额激活，策略必须同时维持两只杯子的平稳。
- **单臂持杯时**：仅持杯侧的相关稳定性子项生效，空闲臂不受末端稳定性约束。
- **完全空手时**：$m_L = m_R = 0$，$r_{\text{stab}}^L$ 与 $r_{\text{stab}}^R$ 在两侧全局关闭。策略不会受到任何关于末端加速度和倾角限制的持杯稳定性约束（但安全底线惩罚 $r_{\text{safe}}$ 仍在）。

当前 TensorBoard 中按“生效后的实际贡献”记录 `r_stab^L`、`r_stab^R` 及各自子项，因此在 Stage 1 下这些曲线应保持在 0 附近。

---

## 6. 安全惩罚 $r_{\text{safe}}$

安全惩罚项**始终激活**，不受持杯状态影响。奖励函数不鼓励危险行为，无论手中是否持杯。

本节统一采用以下约定：

- `r_safe` 是正惩罚量的总和
- 各个安全子项都按正数幅值定义
- 最终总奖励通过 `- r_safe` 统一扣除

当前任务是显式接触任务，末端与门板/把手发生接触属于完成任务的必要条件。因此安全项不再直接惩罚接触力，而只保留与设备保护和持杯安全直接相关的约束。需要特别区分：仿真执行前的力矩 clip 由 env / 仿真层负责，而“力矩超限惩罚”基于 policy 原始输出的控制力矩计算，用于惩罚超限控制意图。

### 6.1 关节限位逼近惩罚

当关节角度逼近物理限位时施加递增惩罚，防止机构损伤。

**限位信息来源**：
每个关节的物理极限区间 $[q_i^{\min}, q_i^{\max}]$ 并非硬编码在奖励函数中，而是直接从机器人的底层资产文件中解析获得：
* URDF 源码定义：[`assets/robot/urdf/uni_dingo_dual_arm.urdf`](../../../assets/robot/urdf/uni_dingo_dual_arm.urdf) 中的 `<limit lower="..." upper="..."/>`
* USD 仿真模型：[`assets/robot/uni_dingo_dual_arm.usd`](../../../assets/robot/uni_dingo_dual_arm.usd)

在 Isaac Sim 仿真运行时，奖励模块会通过只读的物理抽象层（如 `ArticulationView.get_dof_limits()`）一次性获取这些真值。

设第 $i$ 关节的角度范围为 $[q_i^{\min}, q_i^{\max}]$，中心为 $q_i^c$，半范围为 $\delta_i = (q_i^{\max} - q_i^{\min}) / 2$，惩罚项的计算公式为：

$$
r_{\text{limit}} = \beta_1 \cdot \sum_{i=1}^{12} \max\!\left(0,\ \left| q_i - q_i^c \right| - \mu \cdot \delta_i \right)^2
$$

其中 $\mu \in (0, 1)$ 为触发比例（如 0.9，即偏移量超过允许半范围的 90% 时，才开始产生向上生长的二次幂惩罚）。

### 6.2 关节速度过大惩罚

与角度极限类似，防止过快运动超出关节减速器与电机的额定承载能力。

**限速信息来源**：
每个关节的物理最大转速 $\dot{q}_i^{\max}$ 同样并非随意选取，而是提取自底层的机器人资产：
* URDF 源码定义：[`assets/robot/urdf/uni_dingo_dual_arm.urdf`](../../../assets/robot/urdf/uni_dingo_dual_arm.urdf) 中的 `<limit ... velocity="..."/>` 所填写的最大速率。
* USD 仿真模型：[`assets/robot/uni_dingo_dual_arm.usd`](../../../assets/robot/uni_dingo_dual_arm.usd) 中绑定的 `maxJointVelocity` 属性。

在运行时，系统通过物理引擎接口（如 `ArticulationView.get_dof_max_velocities()`）直接获取这组速度上限，并以此设置速度阈值 $v_{\text{thresh}}^{(i)}$ （通常取物理极限的一个安全系数比如 0.9）。

$$
r_{\text{vel}} = \beta_2 \cdot \sum_{i=1}^{12} \max\!\left(0,\ |\dot{q}_i| - \mu \cdot \dot{q}_i^{\max} \right)^2
$$

超过该动态截断阈值的部分以平方惩罚，阈值内不惩罚。这样当更换其他具有不同传动比的机械臂配置时，无需修改训练超参。

### 6.3 原始控制力矩超限惩罚

设 policy 在当前步输出的原始控制力矩为 $\boldsymbol{\tau}^{\text{raw}}_t$，每个关节的力矩上限为 $\boldsymbol{\tau}^{\max}$。env 会在送入物理引擎前执行：

$$
\boldsymbol{\tau}^{\text{applied}}_t = \text{clip}\!\left(\boldsymbol{\tau}^{\text{raw}}_t,\; -\boldsymbol{\tau}^{\max},\; \boldsymbol{\tau}^{\max}\right)
$$

但安全检测不会使用 $\boldsymbol{\tau}^{\text{applied}}_t$ 掩盖超限行为，而是直接对原始输出中的超限部分施加惩罚：

$$
r_{\text{torque}} = \beta_3 \cdot \sum_{i=1}^{12} \max\!\left(0,\ |\tau^{\text{raw}}_i| - \tau^{\max}_i \right)^2
$$

这样可以同时满足两点：

- 物理仿真始终安全，只执行 clip 后的力矩；
- 策略如果持续输出越界力矩，仍会在奖励层受到明确惩罚。

### 6.4 杯体脱落惩罚

如果机械臂或环境的交互动作导致杯子不可逆地脱离了末端原有的持握状态（例如掉落、被外力撞飞），系统会给出极其严厉的单次截断性惩罚，以保证模型学会安全稳定的接触操作：

$$
r_{\text{drop}} = w_{\text{drop}} \cdot \mathbf{1}[\text{cup\_dropped}]
$$

脱落触发后当前 episode 直接终止。

---

## 7. 课程与稳定性项的关系

当前奖励设计不再使用时间退火。原因是课程本身已经把“先学会推门，再学会持杯稳定推门”拆成了两个阶段：

1. **Stage 1（无持杯）**：`m_L = m_R = 0`，稳定性分支整体关闭。策略只学习推门和基本安全约束，不会因为稳定性惩罚而学成“不动”。
2. **Stage 2/3（进入持杯）**：对持杯侧立即启用完整稳定性项。此时策略已经具备基础推门能力，奖励不再需要用时间轴逐步放大稳定性惩罚。

---

## 8. 奖励数据来源

`_get_rewards()` 从以下来源获取输入：

| 数据 | 来源 | 用途 |
|------|------|------|
| $\theta_t, \theta_{t-1}$ | `door.data.joint_pos` + `_prev_door_angle` 缓存 | 任务奖励 |
| $a_t, b$ | `_cached_approach_dist` + `_initial_approach_dist` + 门板大表面几何缓存 | approach reward |
| $\mathbf{a}_t, \boldsymbol{\alpha}_t$ | 连续两步 EE 速度的数值微分（`_get_observations` 中缓存） | 稳定性奖励 |
| $\text{tilt}$ | EE 姿态四元数 + 重力方向几何推理（`_compute_tilt`） | 稳定性奖励 |
| $m_L, m_R$ | `_left_occupied` / `_right_occupied` 持杯状态标记 | 稳定性 mask |
| $\boldsymbol{\tau}^{\text{raw}}_t$ | `_cached_raw_action`（clip 前原始输出） | 力矩超限惩罚 |
| $\boldsymbol{\tau}^{\text{applied}}_t, \boldsymbol{\tau}^{\text{applied}}_{t-1}$ | `_prev_action` / `_prev_prev_action`（clip + noise 后的力矩） | 力矩平滑/正则 |
| $\boldsymbol{\tau}^{\max}$ | `cfg.effort_limit` | 力矩超限阈值 |
| $\mathbf{q}, \dot{\mathbf{q}}$ | `robot.data.joint_pos` / `joint_vel` | 关节限位/速度惩罚 |
| 关节限位 $[q_i^{\min}, q_i^{\max}]$ | `robot.data.soft_joint_pos_limits`（物理引擎提供） | 关节限位惩罚 |
| 关节速度上限 $\dot{q}_i^{\max}$ | `robot.data.soft_joint_vel_limits`（物理引擎提供） | 速度惩罚 |
| 杯体脱落标志 | `_check_cup_dropped()`（EE-杯体距离阈值检测） | 杯体脱落惩罚 |

注意：奖励计算使用仿真地面真值（不依赖 critic 的 privileged 信息）。

---

## 9. 权重参数一览

训练脚本默认从 `configs/reward/default.yaml` 读取奖励超参数，并通过 `_inject_reward_params()` 覆盖写入 `DoorPushEnvCfg`；`DoorPushEnvCfg` 自身仍保留同名默认值作为未注入场景下的回退。以下仅列出参数的数学符号与物理含义，具体数值参见该 YAML 文件。

### 9.1 主任务权重

| 参数 | YAML 键 | 含义 |
|------|---------|------|
| $\theta_{\text{reward\_success}}$ | `task.success_angle_threshold` | success bonus 触发角度 |
| $w_\delta$ | `task.w_delta` | 角度增量基准奖励 |
| $\alpha$ | `task.alpha` | 权重衰减下限比例 |
| $k_{\text{decay}}$ | `task.k_decay` | 超出目标角度后的衰减速率 |
| $w_{\text{open}}$ | `task.w_open` | 任务成功 bonus |
| $w_{\text{approach}}$ | `task.w_approach` | 接近门板大表面奖励权重 |
| $\varepsilon$ | `task.approach_eps` | 归一化距离公式稳定项 |
| $\theta_{\text{stop}}$ | `task.approach_stop_angle` | 关闭接近奖励的门角度阈值 |

### 9.2 稳定性权重

| 参数 | YAML 键 | 含义 |
|------|---------|------|
| $w_{\text{zero-acc}}$ | `stability.w_zero_acc` | 零线加速度奖励权重 |
| $\lambda_{acc}$ | `stability.lambda_acc` | 零线加速度高斯核衰减率 |
| $w_{\text{zero-ang}}$ | `stability.w_zero_ang` | 零角加速度奖励权重 |
| $\lambda_{ang}$ | `stability.lambda_ang` | 零角加速度高斯核衰减率 |
| $w_{\text{acc}}$ | `stability.w_acc` | 线加速度惩罚系数 |
| $w_{\text{ang}}$ | `stability.w_ang` | 角加速度惩罚系数 |
| $w_{\text{tilt}}$ | `stability.w_tilt` | 重力倾斜惩罚系数 |
| $w_{\text{smooth}}$ | `stability.w_smooth` | 力矩平滑惩罚系数 |
| $w_{\text{reg}}$ | `stability.w_reg` | 力矩正则惩罚系数 |

### 9.3 安全惩罚权重

| 参数 | YAML 键 | 含义 |
|------|---------|------|
| $\beta_1$ | `safety.beta_limit` | 关节限位系数 |
| $\mu$ | `safety.mu` | 关节限位/速度触发比例 |
| $\beta_2$ | `safety.beta_vel` | 关节速度系数 |
| $\beta_3$ | `safety.beta_torque` | 原始控制力矩超限系数 |
| $w_{\text{drop}}$ | `safety.w_drop` | 杯体脱落惩罚 |

### 9.4 课程阶段下的稳定性项开关

| 阶段 | 持杯 mask | 稳定性项 |
|------|-----------|----------|
| Stage 1 | `m_L = m_R = 0` | 全部关闭 |
| Stage 2 | 单臂持杯 | 持杯侧 7 个稳定性子项全量生效 |
| Stage 3 | 混合分布 | 按具体上下文对持杯侧 7 个稳定性子项全量生效 |

---

## 10. 分项监控

当前实现会在 `_get_rewards()` 中同步构建 `reward_info`，并通过 `DirectRLEnvAdapter`、`RolloutCollector`、`train.py` 写入 TensorBoard。记录口径为“本轮 rollout 内所有环境、所有步的平均子项贡献”。

当前可观测的主标签包括：

- `reward/total`、`reward/task`、`reward/stab_left`、`reward/stab_right`、`reward/safe`
- `reward_terms/task/delta`、`reward_terms/task/open_bonus`、`reward_terms/task/approach`、`reward_terms/task/approach_raw`
- `reward_terms/stab_left/*`、`reward_terms/stab_right/*`
- `reward_terms/safe/joint_limit`、`reward_terms/safe/joint_vel`、`reward_terms/safe/torque_limit`、`reward_terms/safe/cup_drop`

其中 `task`、`stab_left`、`stab_right` 采用进入总奖励时的有符号贡献；`safe` 与 `safe/*` 采用正惩罚量记录，在总奖励中统一以减号扣除。

这一组分项日志使得训练过程中每个奖励来源都可直接观测，便于定位“任务进展不足”还是“安全/稳定性惩罚过强”。更详细的标签说明见 `docs/tensorboard_guide.md`。
