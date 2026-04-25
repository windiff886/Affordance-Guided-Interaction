# 训练流水线详解（数学建模版）

本文只讨论当前项目默认训练入口对应的真实训练任务，即 `Affordance-DoorPush-Direct-v0`。文档目标不是复述代码调用关系，而是把“这个项目到底在优化什么、环境如何被数学化、奖励如何构造、PPO 如何训练、随机化如何进入模型、TensorBoard 究竟记录了什么”完整写清楚。

需要先说明一件非常关键的事实：当前默认训练不是靠 `train.py` 自定义回调来切换 occupancy，而是直接通过 Isaac Lab 的 `EventTerm(mode="reset")` 在每次 episode reset 时按 `empty / left / right / both = 25% / 25% / 25% / 25%` 采样 occupancy。因此环境里持杯初始化、稳定性奖励和掉杯终止分支在默认训练中都会被激活，但它们只会对实际持杯的那部分 episode 生效。

## 目录

- [0. 符号、时间尺度与当前默认常数](#sec-0)
- [1. 任务介绍：项目在训练什么，解决什么问题](#sec-1)
- [2. 环境构建：资产、坐标系与仿真初始化](#sec-2)
- [3. Observation、Policy、Reward](#sec-3)
- [4. Policy 是如何训练的：完整 PPO 推导版](#sec-4)
- [5. 随机化：哪些量被随机化，如何进入训练](#sec-5)
- [6. TensorBoard：当前记录了哪些参数，它们具体表示什么](#sec-6)
- [7. 一句话总结](#sec-7)

<a id="sec-0"></a>
## 0. 符号、时间尺度与当前默认常数

| 记号 | 含义 | 当前默认值 |
| --- | --- | --- |
| $N$ | 并行环境数 | $6144$ |
| $\Delta t_{\text{phys}}$ | 物理积分步长 | $1/120 \,\text{s}\approx 0.008333$ |
| $d$ | decimation，两个控制步之间包含的物理步数 | $2$ |
| $\Delta t$ | 策略控制步长 | $d\Delta t_{\text{phys}}=1/60\,\text{s}$ |
| $T_{\text{ep}}$ | 单回合最大时长 | $15\,\text{s}$ |
| $H$ | 单次 rollout horizon | $64$ |
| $T$ | 单回合最大控制步数 | $15\times 60=900$ |
| $d_a$ | 动作维度 | $15$ |
| $d_\pi$ | actor 观测维度 | $90$ |
| $d_V$ | critic 观测维度 | $103$ |
| $\gamma$ | 折扣因子 | $0.99$ |
| $\lambda$ | GAE 参数 | $0.95$ |
| $\varepsilon$ | PPO clip 系数 | $0.2$ |

本文中：

- $t$ 表示控制步，而不是物理积分子步。
- $i$ 表示第 $i$ 个并行环境。
- $W$ 表示世界坐标系，$B$ 表示机器人 `base_link` 坐标系。
- $\theta_t$ 表示门铰链角度。
- $q_t,\dot q_t\in\mathbb R^{12}$ 表示双臂 12 个受控关节的位置和速度。
- $u_t^{\text{base}}=(v_{x,t},v_{y,t},\omega_{z,t})\in\mathbb R^3$ 表示底盘在 `base_link` 坐标系下的速度命令。
- $a_t\in[-1,1]^{15}$ 表示策略输出的归一化动作，其中前 12 维控制双臂，后 3 维控制底盘速度，而不是直接控制轮速。
- $o_t^\pi\in\mathbb R^{90}$ 表示 actor 输入，$o_t^V\in\mathbb R^{103}$ 表示 critic 输入。

除非特别注明，文中“当前默认值”均指默认训练入口实际使用的配置值，也就是 `configs/env/default.yaml`、`configs/task/default.yaml`、`configs/reward/default.yaml` 以及 rl_games agent YAML 注入后的结果，而不是 `DoorPushEnvCfg` 中保留的回退默认值。

<a id="sec-1"></a>
## 1. 任务介绍：项目在训练什么，解决什么问题

### 1.1 任务本质

这个项目训练的是一个双臂移动底盘机器人策略，使其在门前随机初始化后，通过双臂关节位置控制与底盘速度控制把门推开。环境代码还支持“左右手持杯推门”的更难版本，此时策略不仅要推门，还要尽量让杯体保持稳定、不要掉落。

如果把任务抽象成马尔可夫决策过程，可写为：

$$
\mathcal M=(\mathcal S,\mathcal A,P,r,\rho_0,\gamma).
$$

其中：

- 状态 $s_t\in\mathcal S$ 包含仿真器中的全部物理量。
- 动作 $a_t\in\mathcal A=[-1,1]^{15}$ 由双臂 12 维归一化位置命令和底盘 3 维归一化速度命令组成。
- 转移 $P(s_{t+1}\mid s_t,a_t)$ 由 Isaac Lab + PhysX 的刚体/关节动力学决定。
- 奖励 $r_t=r(s_t,a_t,s_{t+1})$ 同时鼓励开门、接近门板、保持稳定、避免越界和掉杯。
- 初始状态分布 $\rho_0$ 来自回合 reset 时的基座位姿和物理参数随机化。

策略的优化目标是最大化期望折扣回报：

$$
J(\theta)=\mathbb E_{\tau\sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
$$

### 1.2 完整状态的物理组成

从建模上看，当前环境的完整状态至少包括：

$$
s_t=
\Big(
q_t,\dot q_t,\;
x_t^{\text{door}},\dot x_t^{\text{door}},\theta_t,\dot\theta_t,\;
x_t^{\text{cup},L},\dot x_t^{\text{cup},L},\;
x_t^{\text{cup},R},\dot x_t^{\text{cup},R},\;
\xi
\Big),
$$

其中 $\xi$ 是回合级随机变量，包含：

$$
\xi=
\left(
m_{\text{cup}},\;
m_{\text{door}},\;
c_{\text{door}},\;
p_{\text{base}},\psi_{\text{base}},\;
m_t^L,m_t^R
\right).
$$

这里：

- $m_{\text{cup}}$ 是杯体质量。
- $m_{\text{door}}$ 是门板质量。
- $c_{\text{door}}$ 是门铰链阻尼。
- $p_{\text{base}},\psi_{\text{base}}$ 是机器人基座位置与 yaw。
- $m_t^L,m_t^R\in\{0,1\}$ 是左右手 occupancy 掩码，表示该侧是否持杯。

### 1.3 成功、失败和时间截断

当前环境的终止逻辑可以写成：

- 成功判定：
  $$
  \theta_t \ge \theta_{\text{target}},\qquad
  \theta_{\text{target}}=1.2\ \text{rad},\qquad
  \text{base\_link\_crossed}_t = 1,\qquad
  \text{cup\_dropped}_t = 0.
  $$
- 失败事件：
  $$
  \text{cup\_dropped}_t = 1.
  $$
- 时间截断：
  $$
  t\ge 900.
  $$

当前代码里，episode 的 success 标记要求三件事同时成立：

$$
\text{success}_t=
\mathbf 1[\theta_t\ge 1.2]\cdot
\mathbf 1[\text{cup\_dropped}_t=0]\cdot
\mathbf 1[\text{base\_link\_crossed}_t=1].
$$

其中 `base_link_crossed` 表示机器人 `base_link` 从门外侧穿过门洞下沿所在的门洞平面，且穿越时横向位置仍落在门洞内沿宽度范围内。该量在环境里是一个回合级 latch，一旦过门成功便保持为 1。

因此需要注意：

- 仅把门推到 $1.2$ rad 而没有让 `base_link` 过门，不算成功。
- 即使 `base_link` 先穿过门洞，只要门角尚未达到 $1.2$ rad，当前代码也不会把该回合记为成功。
- 即使门角达到阈值且底盘过门，只要杯子掉落，该回合仍然不算成功。

其中掉杯事件由杯体与对应末端执行器距离是否超过阈值决定：

$$
\text{cup\_dropped}_t=
\mathbf 1\!\left[
m_t^L\|p_t^{\text{cup},L}-p_t^{\text{ee},L}\|_2>\!0.15
\;\;\text{or}\;\;
m_t^R\|p_t^{\text{cup},R}-p_t^{\text{ee},R}\|_2>\!0.15
\right].
$$

### 1.4 当前默认训练的 occupancy 分布

当前 `DoorPushEnvCfg.events` 会在每次 episode reset 时通过 Isaac Lab 的 reset event 独立采样 occupancy 模式：

$$
\Pr(\text{empty})=\Pr(\text{left})=\Pr(\text{right})=\Pr(\text{both})=0.25.
$$

把模式映射到左右手 occupancy，可写成：

$$
\text{empty}\mapsto (0,0),\qquad
\text{left}\mapsto (1,0),\qquad
\text{right}\mapsto (0,1),\qquad
\text{both}\mapsto (1,1).
$$

因此默认训练不再退化成单一的空手任务，而是一个四模式混合任务：

- `empty` 回合里，稳定性项和掉杯项为 0。
- `left` / `right` 回合里，单侧持杯分支被激活。
- `both` 回合里，双侧持杯分支同时被激活。

所以当前主训练任务的真实语义是：

$$
\text{非视觉、双臂绝对位置控制、特权 critic 的混合 occupancy 推门 PPO 训练。}
$$

<a id="sec-2"></a>
## 2. 环境构建：资产、坐标系与仿真初始化

### 2.1 资产构成

每个并行环境都包含如下资产：

| 资产 | 角色 | 说明 |
| --- | --- | --- |
| 双臂机器人 | 受控主体 | 移动底盘，12 个双臂策略关节，3 个底盘速度动作维度，2 个 gripper 关节 |
| 门 | 被操作对象 | 单铰链 articulation，铰链无主动力矩 |
| 左杯体 | 可选被抓持物体 | 仅在左 occupancy 为 1 时参与任务 |
| 右杯体 | 可选被抓持物体 | 仅在右 occupancy 为 1 时参与任务 |
| 地面 | 接触平面 | 大平面刚体 |
| 环境光 | 渲染辅助 | 不影响控制建模 |

当前默认场景中有两个明确的“未启用项”：

- `room=None`，因此房间外壳不参与训练。
- 默认无相机，策略不接收图像，只接收低维状态和门几何量。

### 2.2 并行环境复制

环境通过场景克隆生成 $N=3072$ 份并行实例。数学上可以看作同时采样 $N$ 个独立 MDP 副本：

$$
\left\{\mathcal M^{(i)}\right\}_{i=1}^{N}.
$$

这些副本共享同一套动力学方程和奖励函数，但每个副本有自己的随机初值和回合轨迹。

### 2.3 坐标系定义

项目中大量观测量不是直接在世界系 $W$ 下给出，而是转到机器人基座系 $B$。

对任意世界系向量 $v^W$，其在基座系下的表达为：

$$
v^B = R_{BW}v^W = (R_{WB})^\top v^W.
$$

对任意世界系位置 $p^W$，其在基座系下的表达为：

$$
p^B = R_{BW}(p^W-p_{\text{base}}^W).
$$

对任意世界系四元数 $q_{\text{obj}}^W$，其基座相对姿态为：

$$
q_{\text{obj}}^B = (q_{\text{base}}^W)^{-1}\otimes q_{\text{obj}}^W.
$$

这意味着 actor 和 critic 看到的末端位置、门位置、门法向量，本质上都是“相对于机器人底座”的量，而不是绝对世界坐标。

### 2.4 时间离散化与控制频率

物理积分步长是：

$$
\Delta t_{\text{phys}}=\frac{1}{120}\text{s}.
$$

每个控制步包含 $d=2$ 个物理子步，因此策略作用频率是：

$$
\Delta t=d\Delta t_{\text{phys}}=\frac{1}{60}\text{s}.
$$

这直接决定了：

- 加速度和角加速度用 $\Delta t$ 做差分。
- GAE、回报累计和回合长度都以控制步为单位。

### 2.5 动作如何转成物理控制

策略输出 12 维向量，表示双臂 12 个关节的**归一化目标命令**（不是关节增量）：

$$
a_t\in[-1,1]^{12}.
$$

从策略输出到物理力矩，经过以下两个阶段：

#### 阶段一：命令重标定（将归一化动作映射到关节可行范围）

训练时 rl_games 先对采样动作做 `clip_actions=1.0`，随后环境把它仿射映射到各关节的 soft limits：

$$
\bar q_t^\star
=
\frac{q_{\max}+q_{\min}}{2}
+
a_t\odot\frac{q_{\max}-q_{\min}}{2}.
$$

接口中还带有位置目标噪声注入步骤：

$$
\tilde q_t^\star = \operatorname{clip}(\bar q_t^\star + \epsilon_t^{\text{target}},\; q_{\min},\; q_{\max}),
\qquad
\epsilon_t^{\text{target}}\sim\mathcal N(0,\sigma_{\text{target}}^2I).
$$

当前默认训练中 $\sigma_{\text{target}}=0.01$。也就是说，命令链路实际是“先把 `[-1,1]` 动作映射到 joint limits，再叠加小幅高斯目标扰动，最后再做一次 clipping”。

#### 阶段二：隐式 PD 控制器（将目标角转为关节力矩）

仿真器内置 PD 控制器，根据位置误差和速度阻尼计算关节力矩：

$$
\tau_t = \operatorname{clip}\Big(
K_p\,(\tilde q_t^\star - q_t) \;-\; K_d\,\dot q_t,
\quad -\tau_{\max},\;\tau_{\max}
\Big).
$$

直观理解：
- $K_p(\tilde q_t^\star - q_t)$：弹簧项，把关节拉向目标位置，误差越大力矩越大
- $K_d\,\dot q_t$：阻尼项，抑制关节运动速度，防止过冲振荡

当前默认 PD 增益：

$$
K_p = 1000,\qquad K_d = 100.
$$

力矩上限按关节分段设置：

- 双侧 `joint2`（肩关节）上限 $60\,N·m$
- 其余臂关节上限 $30\,N·m$

### 2.6 Gripper 初始化与持杯初始化

gripper 不是策略输出的一部分，而是环境内部保持控制的一部分。

若某侧 occupancy 为 1，则该侧 gripper 保持在抓持角：

$$
q_{\text{gripper}}^{\text{hold}}=-32^\circ.
$$

若 occupancy 为 0，则该侧 gripper 保持在完全闭合角：

$$
q_{\text{gripper}}^{\text{hold}}=0^\circ.
$$

当某个回合需要持杯时，初始化过程不是“慢慢闭合抓取”，而是直接做一个解析式重置：

1. 机械臂直接写到预设抓持姿态：
   $$
   q_{\text{left,grasp}}=(0,0,0,0,0,90^\circ),
   $$
   $$
   q_{\text{right,grasp}}=(0,0,0,0,0,-90^\circ).
   $$
2. 对应 gripper 直接设到 $-32^\circ$。
3. 杯体直接 teleport 到基座系下的预设相对位置，再绕 base yaw 旋转到世界系：
   $$
   p_{\text{cup},L}^{B}=(0.29,\ 0.1111,\ 0.6814),
   $$
   $$
   p_{\text{cup},R}^{B}=(0.29,\ -0.1111,\ 0.6814).
   $$

也就是说，持杯初始化不是通过一段抓取轨迹生成的，而是通过“机械臂姿态 + 杯体 pose”的直接写状态完成的。

### 2.7 Reset 时的几何初始化

每个回合 reset 时，机器人基座初始位置的采样应以整扇门的几何中心为锚点，而不是以前表述中的门前推板中心。记整扇门中心为：

$$
c_{\text{door}}=(2.95,\ 0.00).
$$

参考基座点记为：

$$
c_{\text{ref}}=(3.72,\ 0.00).
$$

先计算标称方位角：

$$
\phi_0=\operatorname{atan2}\!\big((c_{\text{ref}})_y-(c_{\text{door}})_y,\ (c_{\text{ref}})_x-(c_{\text{door}})_x\big).
$$

然后采样：

$$
\rho \sim \mathcal U(0.45,0.60),
$$
$$
\phi \sim \mathcal U(\phi_0-20^\circ,\ \phi_0+20^\circ).
$$

于是基座位置为：

$$
p_{\text{base}}^W=
\begin{bmatrix}
(c_{\text{door}})_x+\rho\cos\phi\\
(c_{\text{door}})_y+\rho\sin\phi\\
0.12
\end{bmatrix}.
$$

其中推板中心只用于接触区域相关描述，不应用作 reset 几何采样中心。基座 yaw 的标称值也应取“朝向整扇门中心”的方向：

$$
\psi_{\text{nom}}=
\operatorname{atan2}\!\big((c_{\text{door}})_y-p_y,\ (c_{\text{door}})_x-p_x\big),
$$

然后再采样扰动：

$$
\psi_{\text{base}}\sim \mathcal U(\psi_{\text{nom}}-10^\circ,\ \psi_{\text{nom}}+10^\circ).
$$

### 2.8 Reset 时的动力学初始化

回合重置时还会发生以下状态写入：

- 机器人 root pose 重置到新采样的 $p_{\text{base}}^W,\psi_{\text{base}}$。
- 机器人关节位置和速度重置到默认值。
- 门铰链角和角速度重置为 0。
- 不需要的杯体 teleport 到远处停放位：
  $$
  p_{\text{park},L}\approx (100,0,0)+\text{env\_origin},
  $$
  $$
  p_{\text{park},R}\approx (100,1,0)+\text{env\_origin}.
  $$

更完整地说，`_reset_idx(env_ids)` 当前执行顺序是：

1. 采样 `cup_mass`、`door_mass`、`door_damping`。
2. 采样门外扇形环上的机器人基座位置与 yaw。
3. 写入新的机器人 root pose，并把机器人关节状态重置到默认值。
4. 把门关节位置和速度清零。
5. 根据 occupancy 保留或停放杯体；若需要持杯，则执行批量抓持初始化。
6. 把回合级随机化参数写回 PhysX。
7. 清零 `_prev_joint_target`、`_prev_door_angle`、成功标记、加速度缓存和 episode reward 累积项。

<a id="sec-3"></a>
## 3. Observation、Policy、Reward

### 3.1 Actor 观测 $o_t^\pi\in\mathbb R^{90}$

actor 输入由七部分拼接而成：

$$
o_t^\pi=
\Big[
o_t^{\text{prop}},
o_t^{L},
o_t^{R},
o_t^{\text{ctx}},
o_t^{\text{stab}},
o_t^{\text{door}},
o_t^{\text{base}}
\Big].
$$

#### 3.1.1 本体项

$$
o_t^{\text{prop}}=
\big[
\tilde q_t,\ \tilde{\dot q}_t,\ q_{t-1}^\star
\big]\in\mathbb R^{36}.
$$

其中：

- $\tilde q_t,\tilde{\dot q}_t$ 是加噪后的关节位置和速度。
- $q_{t-1}^\star$ 是上一个控制步真正发送到执行器的目标位置。

当前噪声只加在 $q_t,\dot q_t$ 上：

$$
\tilde q_t = q_t + \epsilon_t^q,\qquad
\tilde{\dot q}_t = \dot q_t + \epsilon_t^{\dot q},
$$
$$
\epsilon_t^q,\epsilon_t^{\dot q}\sim \mathcal N(0,\sigma_o^2I),\qquad \sigma_o=0.01.
$$

#### 3.1.2 左右末端项

对任一侧 $k\in\{L,R\}$，末端块定义为：

$$
o_t^k=
\big[
p_t^{\text{ee},k,B},
q_t^{\text{ee},k,B},
v_t^{\text{ee},k,W\to B},
\omega_t^{\text{ee},k,W\to B},
a_t^{\text{ee},k,W\to B},
\alpha_t^{\text{ee},k,W\to B}
\big]\in\mathbb R^{19}.
$$

其中位置和姿态在 `base_link` 坐标系下表示，而线速度、角速度为世界系下的速度旋转到当前 `base_link` 坐标系表达：

$$
v_t^{\text{ee},k,W\to B}=\operatorname{Rot}(W\rightarrow B_t)\,v_t^{\text{ee},k,W},
$$

$$
\omega_t^{\text{ee},k,W\to B}=\operatorname{Rot}(W\rightarrow B_t)\,\omega_t^{\text{ee},k,W},
$$

线加速度和角加速度由世界系速度做数值差分后旋转到 base 系：

$$
a_t^{\text{ee},k,W\to B}=
\operatorname{Rot}(W\rightarrow B_t)\,
\frac{v_t^{\text{ee},k,W}-v_{t-1}^{\text{ee},k,W}}{\Delta t},
$$
$$
\alpha_t^{\text{ee},k,W\to B}=
\operatorname{Rot}(W\rightarrow B_t)\,
\frac{\omega_t^{\text{ee},k,W}-\omega_{t-1}^{\text{ee},k,W}}{\Delta t}.
$$

#### 3.1.3 上下文项

$$
o_t^{\text{ctx}}=
\big[m_t^L,\ m_t^R\big]\in\mathbb R^2.
$$

在当前默认训练中，这两维由每个 episode 的 occupancy 采样结果决定，不再恒等于 0。

#### 3.1.4 稳定性观测项

环境定义了一个“杯体竖直度 proxy”。设末端姿态对应旋转矩阵为 $R_t^{\text{ee},k,W}$，则把世界重力直接投到末端局部系：

$$
g_{\text{local}}=
\left(R_t^{\text{ee},k,W}\right)^\top
\begin{bmatrix}
0\\0\\-9.81
\end{bmatrix}.
$$

抓取姿态下，环境认为末端局部 $Y$ 轴应近似与世界竖直方向对齐，因此取重力在末端局部 $x\!-\!z$ 平面上的分量：

$$
u_t^k = \big[g_{\text{local},x},\ g_{\text{local},z}\big]\in\mathbb R^2.
$$

稳定性观测标量是：

$$
\text{tilt}_t^k=\|u_t^k\|_2.
$$

因此：

$$
o_t^{\text{stab}}=
\big[
\text{tilt}_t^L,\ \text{tilt}_t^R
\big]\in\mathbb R^2.
$$

#### 3.1.5 门几何项

环境不使用图像，而是直接给策略一个 6 维门几何量：

$$
o_t^{\text{door}}=
\big[
p_t^{\text{door-center},B},\ n_t^{\text{door},B}
\big]\in\mathbb R^6.
$$

其中门板局部系中的常量为：

$$
d_{\text{center}}^{\text{leaf}}=(0.02,0.45,1.0),
$$
$$
d_{\text{face}}^{\text{leaf}}=(0.04,0.45,1.0),
$$
$$
n^{\text{leaf}}=(1,0,0).
$$

也就是说，策略能直接知道“门板中心在基座系下的位置”和“可推门面法向量在基座系下的方向”。

除此之外，策略现在还接收门洞内沿四角在 `base_link` 系下的坐标：

$$
o_t^{\text{frame}}=
\big[
p_t^{\text{LL},B},
p_t^{\text{LR},B},
p_t^{\text{UL},B},
p_t^{\text{UR},B}
\big]\in\mathbb R^{12},
$$

其中四个点分别对应门洞内沿的左下、右下、左上、右上角点。当前资产对应的门洞局部常量为：

$$
(0,-0.51,0),\ (0,0.51,0),\ (0,-0.51,2.05),\ (0,0.51,2.05).
$$

这组 12 维量直接把可通行门洞的几何开口暴露给策略，而不再只给门板几何。

#### 3.1.6 底盘本体项

移动底盘版本还会给策略一个 6 维底盘块：

$$
o_t^{\text{base}}=
\big[
v_{x,t}^{W\to B},\ v_{y,t}^{W\to B},\ \omega_{z,t}^{W\to B},\ u_t^{\text{base}}
\big]\in\mathbb R^6.
$$

其中前三维是底盘在世界系下的线速度和角速度，旋转到 `base_link` 坐标系下表达：

$$
v^{W\to B}=\operatorname{Rot}(W\rightarrow B_t)\,v^{\text{base},W},
\quad
\omega^{W\to B}=\operatorname{Rot}(W\rightarrow B_t)\,\omega^{\text{base},W}.
$$

后三维是最近一次写入执行器的底盘速度命令。policy 控制的是这 3 维底盘速度命令；环境内部再把它们映射成四个轮子的速度目标。

### 3.2 Critic 观测 $o_t^V\in\mathbb R^{115}$

critic 使用非对称特权观测：

$$
o_t^V=
\big[
o_t^{\pi,\text{clean}},\ z_t^{\text{priv}}
\big].
$$

其中 $o_t^{\pi,\text{clean}}$ 与 actor 观测结构相同，但不对 $q_t,\dot q_t$ 加噪声。额外的 13 维特权量为：

$$
z_t^{\text{priv}}=
\big[
x_t^{\text{door-root},B},
\theta_t,\dot\theta_t,
m_{\text{cup}},
m_{\text{door}},
c_{\text{door}},
\text{cup\_dropped}_t
\big]\in\mathbb R^{13}.
$$

更细地写：

- 门 root pose：7 维
- 门关节角和关节速度：2 维
- 杯质量、门板质量、门阻尼：3 维
- 掉杯标记：1 维

### 3.3 Policy 参数化

当前策略是连续动作高斯策略：

$$
a_t \sim \pi_\theta(\cdot\mid o_t^\pi)
=
\mathcal N\!\Big(\mu_\theta(o_t^\pi),\ \operatorname{diag}(\sigma^2)\Big).
$$

其中：

- $\mu_\theta(\cdot)$ 由 ELU MLP 产生，隐藏层宽度为 $[512,256,128]$。
- $\sigma=\exp(\sigma_0)$，其中 $\sigma_0$ 不是状态相关函数，而是 state-independent 的固定可学习参数向量。

因此当前策略学习的是：

1. 一个把 $90$ 维 actor 观测映射到 $15$ 维均值的函数 $\mu_\theta$。
2. 一个不随观测变化的对角协方差 log-std 参数。

需要强调三点：

- 动作语义是“12 维双臂 joint command + 3 维底盘速度命令”，不是绝对目标角，也不是轮速，更不是 $\Delta q$。
- 动作先经过 rl_games 的 `clip_actions=1.0`，随后环境再把前 12 维映射到 joint limits，并在需要时叠加位置目标噪声；后 3 维则重标定为 $(v_x, v_y, \omega_z)$，再转换成轮速目标。
- gripper 不受策略控制，策略只控制双臂 12 个关节和底盘 3 个速度自由度。

### 3.4 奖励函数

#### 3.4.1 总体结构

整个奖励被拆成 3 个模块：任务推进、持杯稳定性和安全约束。统一写成：

$$
r_t
=
r_t^{\text{task}}
+
r_t^{\text{stab},L}
+
r_t^{\text{stab},R}
-
r_t^{\text{safe}}.
$$

其中任务模块负责把策略从“接近门”推进到“开门并穿门”；稳定性模块只在持杯侧激活，用来抑制末端大幅晃动；安全模块负责压制关节越界、杯体碰门、底盘过猛运动以及底盘几何走形。

任务模块写成：

$$
r_t^{\text{task}}
=
r_t^{\Delta\theta}
+
r_t^{\text{open}}
+
r_t^{\text{approach}}
+
r_t^{\text{base\_approach}}
+
r_t^{\text{base\_cross}}.
$$

对每一侧 $k\in\{L,R\}$，稳定性模块写成：

$$
r_t^{\text{stab},k}
=
m_t^k\Big(
r_t^{\text{zero\_acc},k}
+
r_t^{\text{zero\_ang},k}
-
r_t^{\text{acc},k}
-
r_t^{\text{ang},k}
-
r_t^{\text{tilt},k}
\Big),
$$

其中 $m_t^k\in\{0,1\}$ 是该侧 occupancy mask。

安全模块统一写成：

$$
r_t^{\text{safe}}
=
r_t^{\text{vel}}
+
r_t^{\text{target}}
+
r_t^{\text{joint\_move}}
+
r_t^{\text{cup\_door\_prox}}
+
r_t^{\text{base\_speed}}
-
r_t^{\text{base\_zero\_speed}}
+
r_t^{\text{base\_cmd\_delta}}
+
r_t^{\text{drop}}
+
r_t^{\text{base\_heading}}
+
r_t^{\text{base\_corridor}}.
$$

这里底盘速度约束被拆成两个可单独记录的子项：一个是鼓励低速静稳的 `base_zero_speed`，一个是直接惩罚速度平方和的 `base_speed`。这样后续在 TensorBoard 里可以分别观察“底盘有没有获得零速度偏好奖励”和“底盘是否因为速度过大被罚”。同时，本文不再额外保留独立的 `base_motion` 项，以避免和这两个速度项在功能上重复。
这里最后 2 项底盘几何约束在语义上仍然属于 `safe/*`，所以本文直接把它们并入安全模块统一描述；当前代码若尚未实现这些项，只意味着它们当前不会出现在训练日志里，不改变它们在奖励设计中的位置。

#### 3.4.2 任务模块

任务模块的作用是把策略从“碰到门”推进到“持续把门打开”，再推进到底盘穿过门洞。

1. `task/delta`

定义门角增量：

$$
\Delta\theta_t=\theta_t-\theta_{t-1}.
$$

当门角尚未到达成功阈值 $\theta_{\text{succ}}$ 时，门角增量奖励使用常数权重；当门角超过该阈值后，增量奖励的权重按衰减函数下降到一个保底比例：

$$
w(\theta_t)=
\begin{cases}
w_\Delta, & \theta_t\le \theta_{\text{succ}},\\
w_\Delta\cdot \max\!\big(1-k_{\text{decay}}(\theta_t-\theta_{\text{succ}}),\ \alpha\big), & \theta_t>\theta_{\text{succ}}.
\end{cases}
$$

因此：

$$
r_t^{\Delta\theta}=w(\theta_t)\Delta\theta_t.
$$

2. `task/open_bonus`

当门角第一次超过成功阈值 $\theta_{\text{succ}}$ 时，给予一次性开门 bonus：

$$
r_t^{\text{open}}
=
w_{\text{open}}\cdot
\mathbf 1[\theta_t\ge \theta_{\text{succ}}]\cdot
\mathbf 1[\text{first-crossing at }t].
$$

3. `task/approach`

环境把门的可推侧建模为一个矩形面片。对任意末端点 $p$，先转到门面局部系：

$$
u = R_{\text{face}}^\top (p-c_{\text{face}})
=
\begin{bmatrix}
u_x\\u_y\\u_z
\end{bmatrix}.
$$

门面矩形在局部坐标中满足：

$$
u_y\in[-h_y,h_y],\qquad u_z\in[-h_z,h_z].
$$

于是点到矩形面的距离定义为：

$$
d(p,\mathcal F)=
\sqrt{
u_x^2
+
\big[\max(|u_y|-h_y,0)\big]^2
+
\big[\max(|u_z|-h_z,0)\big]^2
}.
$$

左、右末端分别计算距离，再取更靠近门面的那只手：

$$
d_t=\min\big(d(p_t^{L},\mathcal F_t),\ d(p_t^{R},\mathcal F_t)\big).
$$

环境在回合中第一次得到 $d_t$ 时，把它缓存为初始距离 $d_0$，并构造归一化接近度：

$$
s_t^{\text{approach}}
=
\max\left(
1-\frac{d_t^2}{d_0^2+\varepsilon_{\text{app}}},
0
\right).
$$

接近奖励只在门角尚未超过设定 stop 阈值时生效：

$$
r_t^{\text{approach}}
=
\mathbf 1[\theta_t<\theta_{\text{stop}}]
\cdot
w_{\text{approach}}
\cdot
s_t^{\text{approach}}.
$$

4. 底盘通用平滑因子

底盘相关任务奖励不再使用硬 `base_align_gate` 和硬 `base_corridor_gate` 作为主要学习信号。先定义推门方向：

$$
d^{\text{push}}=-n^{\text{doorway}}.
$$

记底盘前向在地面上的单位方向为 $f_t^{\text{base}}$，则底盘前向与推门方向的夹角为：

$$
\theta_t^{\text{align}}
=
\arccos\!\bigl(
\operatorname{clip}(f_t^{\text{base}}\cdot d^{\text{push}},-1,1)
\bigr).
$$

朝向平滑因子为：

$$
g_t^{\text{align}}
=
\sigma\left(
\frac{\phi_{\text{mid}}-\theta_t^{\text{align}}}{\tau_{\text{align}}}
\right),
\qquad
\phi_{\text{mid}}=35^\circ,\quad \tau_{\text{align}}=5^\circ.
$$

再记底盘矩形 footprint 的 4 个角点在门洞坐标系中的横向坐标为
$y_{t,1}^{\text{corner}},\dots,y_{t,4}^{\text{corner}}$，门洞走廊允许的横向半宽为 $w_{\text{corridor}}$，则横向越界量定义为：

$$
e_t^{\text{corridor}}
=
\max\left(
\max_{i\in\{1,2,3,4\}}
\left|y_{t,i}^{\text{corner}}\right|
- w_{\text{corridor}},
0
\right).
$$

范围平滑因子为：

$$
g_t^{\text{range}}
=
\exp\left[
-\frac{1}{2}
\left(
\frac{e_t^{\text{corridor}}}{\tau_{\text{range}}}
\right)^2
\right],
\qquad
\tau_{\text{range}}=0.05\text{ m}.
$$

当 footprint 没有越界时 $e_t^{\text{corridor}}=0$，因此 $g_t^{\text{range}}=1$；越界后该因子连续衰减。

5. `task/base_align`

新增弱朝向 shaping，鼓励底盘在门附近朝向推门方向。记底盘中心到门洞下沿线段的地面距离为 $d_t^{\text{line}}$：

$$
g_t^{\text{near}}
=
\exp\left[
-\frac{1}{2}
\left(
\frac{d_t^{\text{line}}}{\sigma_{\text{near}}}
\right)^2
\right],
\qquad
\sigma_{\text{near}}=0.8\text{ m}.
$$

则：

$$
r_t^{\text{base\_align}}
=
w_{\text{base\_align}}
\cdot
g_t^{\text{near}}
\cdot
g_t^{\text{range}}
\cdot
g_t^{\text{align}}.
$$

当前默认 $w_{\text{base\_align}}=0.005$，只作为姿态引导，不承担任务主奖励。

6. `task/base_forward`

定义 `base_link` 相对门洞平面的有符号距离为 $\delta_t^{\text{plane}}$，并约定机器人沿推门方向前进时该距离减小。底盘前进量为：

$$
\Delta_t^{\text{base\_forward}}
=
\max(\delta_{t-1}^{\text{plane}}-\delta_t^{\text{plane}},0).
$$

对应奖励为：

$$
r_t^{\text{base\_forward}}
=
w_{\text{base\_forward}}
\cdot
g_t^{\text{align}}
\cdot
g_t^{\text{range}}
\cdot
\Delta_t^{\text{base\_forward}}.
$$

7. `task/base_centerline`

记底盘中心在门洞坐标系中的横向偏移为 $y_t^{\text{base}}$，中线得分为：

$$
s_t^{\text{center}}
=
\exp\left[
-\frac{1}{2}
\left(
\frac{y_t^{\text{base}}}{\sigma_{\text{center}}}
\right)^2
\right],
\qquad
\sigma_{\text{center}}=0.25\text{ m}.
$$

对应奖励为：

$$
r_t^{\text{base\_centerline}}
=
w_{\text{base\_centerline}}
\cdot
g_t^{\text{align}}
\cdot
g_t^{\text{range}}
\cdot
s_t^{\text{center}}.
$$

8. `task/base_cross`

这项奖励底盘在门洞内侧半空间中的有效推进量。定义：

$$
p_t^{\text{inside}}=\max(-\delta_t^{\text{plane}}, 0).
$$

对应奖励为：

$$
r_t^{\text{base\_cross}}
=
g_t^{\text{align}}
\cdot
g_t^{\text{range}}
\cdot
\mathbf 1[\theta_t \ge \theta_{\text{base-cross}}]
\cdot
w_{\text{base\_cross}}
\cdot
\max\big(p_t^{\text{inside}} - p_{t-1}^{\text{inside}}, 0\big).
$$

对应的过门成功 latch 更新规则为：

$$
\text{crossed}_t
=
\text{crossed}_{t-1}
\;\text{or}\;
\Big(
\delta_{t-1}^{\text{plane}} > 0
\;\land\;
\delta_t^{\text{plane}} \le 0
\;\land\;
e_t^{\text{corridor}}=0
\Big).
$$

#### 3.4.3 稳定性模块

稳定性模块只在持杯侧激活，因此它不是纯粹的机械平滑项，而是“有杯时才需要认真压住晃动”的条件约束。

对任一侧 $k\in\{L,R\}$，设末端线加速度、角加速度和竖直稳定性 proxy 分别为

$$
a_t^k,\qquad \alpha_t^k,\qquad u_t^k.
$$

则 5 个稳定性子项分别定义为：

$$
r_t^{\text{zero\_acc},k}
=
w_{\text{zero-acc}}e^{-\lambda_{\text{acc}}\|a_t^k\|_2^2},
$$

$$
r_t^{\text{zero\_ang},k}
=
w_{\text{zero-ang}}e^{-\lambda_{\text{ang}}\|\alpha_t^k\|_2^2},
$$

$$
r_t^{\text{acc},k}
=
w_{\text{acc}}\|a_t^k\|_2^2,
$$

$$
r_t^{\text{ang},k}
=
w_{\text{ang}}\|\alpha_t^k\|_2^2,
$$

$$
r_t^{\text{tilt},k}
=
w_{\text{tilt}}\|u_t^k\|_2^2.
$$

因此整侧稳定性项可以写成：

$$
r_t^{\text{stab},k}
=
m_t^k\Big(
r_t^{\text{zero\_acc},k}
+
r_t^{\text{zero\_ang},k}
-
r_t^{\text{acc},k}
-
r_t^{\text{ang},k}
-
r_t^{\text{tilt},k}
\Big).
$$

如果某一侧没有持杯，则该侧 $m_t^k=0$，整项直接关闭。

#### 3.4.4 安全模块

安全模块的作用不是给策略“做任务的动力”，而是持续裁掉危险但又容易被策略利用的动作模式，包括关节越界、杯体擦门、底盘速度过猛、方向偏向门框以及底盘越出门洞走廊。

1. `safe/joint_vel`

设每个关节的软速度上限为 $\dot q_{\max,j}$，只在超过比例阈值后开始惩罚：

$$
e_{t,j}^{\text{vel}}
=
\max\big(|\dot q_{t,j}|-\mu\dot q_{\max,j},0\big).
$$

$$
r_t^{\text{vel}}
=
\beta_{\text{vel}}
\sum_{j=1}^{12}
\left(e_{t,j}^{\text{vel}}\right)^2.
$$

2. `safe/target_limit`

这项惩罚的是最终写入执行器的目标角 $\tilde q_t^\star$ 进入 joint limit 边界带，而不是原始高斯动作样本。定义目标到最近限位的距离：

$$
d_{t,j}^{\text{limit}}
=
\min\big(\tilde q_{t,j}^\star-q_{\min,j},\ q_{\max,j}-\tilde q_{t,j}^\star\big),
$$

边界带宽度为：

$$
m_j=\rho_{\text{margin}}(q_{\max,j}-q_{\min,j}),
$$

则边界带侵入量为：

$$
e_{t,j}^{\text{target}}
=
\max(m_j-d_{t,j}^{\text{limit}},0).
$$

对应惩罚为：

$$
r_t^{\text{target}}
=
\beta_{\text{target}}
\sum_{j=1}^{12}
\left(\frac{e_{t,j}^{\text{target}}}{m_j}\right)^2.
$$

3. `safe/joint_move`

相邻控制步之间的关节运动变化量被直接二次惩罚：

$$
r_t^{\text{joint\_move}}
=
\beta_{\text{joint\_move}}
\sum_{j=1}^{12}(q_{t,j}-q_{t-1,j})^2.
$$

4. `safe/cup_door_prox`

只对持杯侧生效。记杯体质心到门板矩形面的距离为 $d_t^k$，则当该距离低于阈值时施加二次惩罚：

$$
r_t^{\text{cup\_door\_prox}}
=
\sum_{k\in\{L,R\}} m_t^k\cdot
\beta_{\text{cup\_door\_prox}}\cdot
\bigl[\max(d_{\text{thresh}}-d_t^k,\;0)\bigr]^2.
$$

5. `safe/base_zero_speed`

设底盘在 `base_link` 系中的线速度和角速度分别为
$v_{x,t}, v_{y,t}, \omega_{z,t}$。这里不做额外归一化，而是直接使用原始速度平方和：

$$
s_t^{\text{base-speed}}
=
\left(v_{x,t}\right)^2
+
\left(v_{y,t}\right)^2
+
\left(\omega_{z,t}\right)^2.
$$

与稳定性模块中的 `zero_acc`、`zero_ang` 一样，先定义一个零速度奖励：

$$
r_t^{\text{base\_zero\_speed}}
=
w_{\text{base-zero-speed}}
\exp\!\bigl(-\lambda_{\text{base-speed}}\,s_t^{\text{base-speed}}\bigr).
$$

这项在底盘速度接近 0 时接近上界，随着速度变大而指数衰减。当前它只保留为弱正则，默认权重从旧方案的 `0.12` 下调为 `0.02`，避免低速生存奖励在长 episode 中压过真正的底盘推进与穿门奖励。

6. `safe/base_speed`

与上面的零速度奖励配对，再定义一个速度惩罚项：

$$
r_t^{\text{base\_speed}}
=
w_{\text{base-speed}}\,s_t^{\text{base-speed}}.
$$

这样后，若单独从 signed shaping 的角度看，底盘速度项可写成：

$$
r_t^{\text{base-speed,shaping}}
=
r_t^{\text{base\_zero\_speed}}
-
r_t^{\text{base\_speed}}.
$$

而在本文的总奖励记号中，它被拆开放在 `safe` 分支里，写成
$+r_t^{\text{base\_speed}}-r_t^{\text{base\_zero\_speed}}$，这样后续 TensorBoard 可以分开查看：

- 底盘是否因为速度偏大受到惩罚；
- 底盘是否因为低速静稳获得奖励。

这里需要强调：速度相关约束只保留这一组“零速度奖励 + 速度惩罚”双项结构，不再额外保留独立的 `safe/base_motion`，以避免重复惩罚同一类底盘速度行为。

7. `safe/base_cmd_delta`

记当前步底盘命令为
$u_t^{\text{base}}=(v_{x,t}^{\text{cmd}}, v_{y,t}^{\text{cmd}}, \omega_{z,t}^{\text{cmd}})$，则命令差分为：

$$
\Delta u_t^{\text{base}} = u_t^{\text{base}} - u_{t-1}^{\text{base}}.
$$

相应惩罚为：

$$
r_t^{\text{base\_cmd\_delta}}
=
\beta_{\text{base\_cmd}}
\left\|\Delta u_t^{\text{base}}\right\|_2^2.
$$

8. `safe/drop`

掉杯惩罚写成：

$$
r_t^{\text{drop}}=
w_{\text{drop}}\cdot \mathbf 1[\text{cup\_dropped}_t].
$$

9. `safe/base_heading`

这一项惩罚的不是底盘偏离当前门板法向，而是底盘朝门框横向方向偏过去。原因是门板法向会随门角旋转，不能作为稳定的底盘朝向参考。定义固定门洞坐标系：

- $n^{\text{doorway}}$：门洞平面的固定法向，指向穿门前进方向；
- $t^{\text{doorway}}$：门洞下沿线段方向，也就是门框横向方向。

记底盘前向在地面上的单位方向为 $f_t^{\text{base}}$，则：

$$
r_t^{\text{base\_heading}}
=
\beta_{\text{base\_heading}}
\bigl(f_t^{\text{base}}\cdot t^{\text{doorway}}\bigr)^2.
$$

等价地，也可写成：

$$
r_t^{\text{base\_heading}}
=
\beta_{\text{base\_heading}}
\left(1-\bigl(f_t^{\text{base}}\cdot n^{\text{doorway}}\bigr)^2\right).
$$

10. `safe/base_corridor`

定义底盘地面矩形 footprint 的半长和半宽分别为
$\ell_{\text{base}}$ 与 $w_{\text{base}}$。令底盘中心在门洞坐标系中的平面位置为
$(x_t^{\text{base}}, y_t^{\text{base}})$，底盘前向与门洞法向夹角为 $\psi_t$。若底盘 4 个角点在门洞坐标系中的横向坐标为
$y_{t,1}^{\text{corner}},\dots,y_{t,4}^{\text{corner}}$，门洞走廊允许的横向半宽为 $w_{\text{corridor}}$，则横向越界量定义为：

$$
e_t^{\text{corridor}}
=
\max\left(
\max_{i\in\{1,2,3,4\}}
\left|y_{t,i}^{\text{corner}}\right|
- w_{\text{corridor}},
0
\right).
$$

相应惩罚为：

$$
r_t^{\text{base\_corridor}}
=
\beta_{\text{base\_corridor}}
\left(e_t^{\text{corridor}}\right)^2.
$$

这项是纯软惩罚：越界越多，惩罚越大，但不直接终止 episode。

#### 3.4.5 Reward 子项的数学与参数分析

这一节不再只描述“公式形式”，而是明确给出一个数值标尺：我们希望每个 reward 子项在其对应的**典型激活工况**下，episode 级累计贡献尽量落在 `O(10)`，也就是大致 `5~20` 的范围。这样做的目的，是避免某一个子项轻易达到 `10^2~10^3`，从而把其他模块的梯度全部淹没。

为此，先给出 5 个对量级平衡最关键的背景量：

- 当前控制频率为 `60 Hz`，单个 episode 为 `15 s`，因此每回合有 `900` 个控制步。
- 若某个子项在整回合 `900` 步都激活，那么它的目标平均单步贡献应约为 `10/900≈0.011`。
- 若某个子项只在 `300 / 100 / 30 / 10` 步内激活，那么它的目标平均单步贡献应约为 `0.033 / 0.10 / 0.33 / 1.0`。
- 当前门洞内宽由 `y\in[-0.51,0.51]` 给出，因此横向总宽度为 `1.02 m`。
- 当前底盘 footprint 近似为长方形，半长 $\ell_{\text{base}}=0.285\ \text{m}$、半宽 $w_{\text{base}}=0.2104\ \text{m}$，因此外接尺寸约为 `0.57 m × 0.4208 m`。

在这个标尺下，调参时最有用的 4 个反推公式如下：

1. 对于线性 dense shaping
   $$
   r_t = w\,s_t,\qquad s_t\in[0,1],
   $$
   若它在 `N_{\text{act}}` 步内以平均值 $\bar s$ 激活，则要让 episode 级累计量级落到 `10` 左右，应取
   $$
   w^\star \approx \frac{10}{N_{\text{act}}\bar s}.
   $$

2. 对于二次惩罚
   $$
   r_t = \beta\,z_t,\qquad z_t\ge 0,
   $$
   若在 `N_{\text{act}}` 步内平均为 $\bar z$，则平衡到 `10` 左右的系数应取
   $$
   \beta^\star \approx \frac{10}{N_{\text{act}}\bar z}.
   $$

3. 对于积分型推进奖励，例如门角增量或穿门深度增量，
   $$
   R \approx w\cdot \Delta_{\text{ref}},
   $$
   因而要让参考工况下的累计量级落到 `10` 左右，应取
   $$
   w^\star \approx \frac{10}{\Delta_{\text{ref}}}.
   $$

4. 对于一次性 bonus / fail cost，若目标也是 `O(10)`，则最直接的标尺就是
   $$
   w^\star \approx 10.
   $$

下表只给出按上述标尺反推出的**建议量级**。这里分析的重点是负责数值尺度的主系数，也就是各项前面的 `w` 或 `\beta`；像 `open_gate`、`stop_angle`、`align` 阈值、`corridor` 阈值这类几何或阶段门控参数，主要控制“何时激活”，不直接参与这一张量级平衡表。

| 子项 | 参考工况 | 平衡到 `O(10)` 的数学分析 | 建议系数/建议量级 |
| --- | --- | --- | --- |
| `task/delta` | 参考一次有效开门增量累计 $\Delta\theta_{\text{ref}}\approx 1.2\ \text{rad}$。 | 由 $R\approx w_{\delta}\Delta\theta_{\text{ref}}$ 得 $w_{\delta}^\star\approx 10/1.2\approx 8.3$。 | 主尺度系数建议 `w_delta≈8~10`。`k_decay` 与 `alpha` 主要控制超过目标角后还保留多少残余梯度，它们应服务于“成功后仍有弱梯度，但不能继续刷太多 reward”这一目标，而不是用来放大主尺度。 |
| `task/open_bonus` | 门首次达到打开阈值时触发 1 次。 | 若希望 milestone 本身也是 `O(10)`，则直接取 $w_{\text{open}}^\star\approx 10$；若希望它略高于普通 dense shaping，也通常不应超过 `20`。 | 建议 `w_open≈10~20`。它应当是“阶段达成”的强化信号，但不应强到压过整段 dense shaping。 |
| `task/approach` | 参考在 `40` 步内激活，平均 $s_t^{\text{approach}}\approx 0.5$。 | 有 $R\approx w_{\text{approach}}\cdot 40\cdot 0.5=20w_{\text{approach}}$，故 $w_{\text{approach}}^\star\approx 0.5$。 | 建议 `w_approach≈0.4~0.6`。`approach_stop_angle` 负责控制它覆盖到开门的哪个阶段，但不应通过增大 `w_approach` 去承担任务主奖励的角色。 |
| `task/base_align` | 参考底盘在门附近对齐 `900` 步，平均平滑因子乘积约 `1.0`。 | 若希望它只是弱 shaping，则 $R\approx900w_{\text{base\_align}}$ 控制在 `5~10`，故 $w_{\text{base\_align}}\approx0.005~0.01$。 | 当前默认 `w_base_align=0.005`。它只帮助底盘转向推门方向，不应压过推进奖励。 |
| `task/base_forward` | 参考底盘沿推门方向有效推进 `0.4 m`。 | 有 $R\approx w_{\text{base\_forward}}\cdot0.4$，故 $w_{\text{base\_forward}}^\star\approx25$。 | 当前默认 `w_base_forward=25`。这一项替代旧 `base_approach`，作为持续推进 shaping。 |
| `task/base_centerline` | 参考底盘在中线附近保持 `900` 步，平均平滑因子乘积约 `1.0`。 | 有 $R\approx900w_{\text{base\_centerline}}$，若取 `0.03` 理论最大约 `27`，实际会被朝向和范围因子衰减。 | 当前默认 `w_base_centerline=0.03`。若出现站中线不动，应降到 `0.01`。 |
| `task/base_cross` | 参考底盘在门洞内有效向前推进 `0.20 m`。 | 有 $R\approx w_{\text{base\_cross}}\cdot 0.20$，故 $w_{\text{base\_cross}}^\star\approx 50$。 | 建议 `w_base_cross≈40~60`。这一项应该代表“真正的穿门进度”，但在 `O(10)` 标尺下不需要到 `10^3` 量级。 |
| `stab/zero_acc` | 持杯侧在 `300` 步内保持较稳，参考 $\|a_t\|\approx 0.25$，并希望该处仍保留约 `0.6` 倍零加速度奖励；取 $\lambda_{\text{acc}}=8$ 时，$e^{-8\cdot0.25^2}=e^{-0.5}\approx0.607$。 | 有 $R\approx 300\cdot w_{\text{zero-acc}}\cdot 0.607$，故 $w_{\text{zero-acc}}^\star\approx 10/(300\cdot0.607)\approx0.055$。 | 建议 `w_zero_acc≈0.055`、`lambda_acc≈8`。相比旧参考 $\|a\|\approx0.35$，现在更早衰减零加速度奖励。 |
| `stab/zero_ang` | 参考持杯侧在 `300` 步内保持较小角加速度，令 $\|\alpha_t\|\approx 0.45$，并希望该处保留约 `0.5~0.6` 倍零角加速度奖励；取 $\lambda_{\text{ang}}=3$ 时，$e^{-3\cdot0.45^2}\approx0.545$。 | 有 $R\approx 300\cdot w_{\text{zero-ang}}\cdot 0.545$，故 $w_{\text{zero-ang}}^\star\approx 10/(300\cdot0.545)\approx0.061$。 | 建议 `w_zero_ang≈0.06`、`lambda_ang≈3`。相比旧参考 $\|\alpha\|\approx0.7$，现在更严格地压制角加速度。 |
| `stab/acc` | 参考持杯侧在 `300` 步内有中等线加速度，降低期望后令 $\|a_t\|\approx0.9$，即 $\|a_t\|^2\approx0.81$。 | 有 $R\approx 300\cdot w_{\text{acc}}\cdot0.81$，故 $w_{\text{acc}}^\star\approx 10/(300\cdot0.81)\approx0.041$。 | 建议 `w_acc≈0.04`。它比旧值更强，是降低线加速度期望后的主惩罚项。 |
| `stab/ang` | 参考持杯侧在 `300` 步内有明显角加速度，降低期望后令 $\|\alpha_t\|\approx4.5$，即 $\|\alpha_t\|^2\approx20.25$。 | 有 $R\approx 300\cdot w_{\text{ang}}\cdot20.25$，故 $w_{\text{ang}}^\star\approx 10/(300\cdot20.25)\approx1.65\times10^{-3}$。 | 建议 `w_ang≈1.6e-3`。它仍小于线加速度惩罚，但比旧值更能压制角向抖动。 |
| `stab/tilt` | 参考持杯侧在 `300` 步内平均 tilt proxy 为 $\|u_t\|\approx 0.10$。 | 有 $R\approx 300\cdot w_{\text{tilt}}\cdot 0.01$，故 $w_{\text{tilt}}^\star\approx 3.33$。 | 建议 `w_tilt≈3.0~3.5`。这一项直接约束杯体姿态，通常应是稳定性模块里最强的单项之一。 |
| `safe/joint_vel` | 参考 `100` 步内有 `3` 个关节持续明显超限，降低速度期望后令平均超限量各为 `0.4`，则 $\sum_j(e_{t,j}^{\text{vel}})^2\approx3\times0.4^2=0.48$。 | 有 $R\approx100\cdot\beta_{\text{vel}}\cdot0.48$，故 $\beta_{\text{vel}}^\star\approx10/48\approx0.208$。 | 建议 `mu≈0.75`、`beta_vel≈0.20`。`mu` 让惩罚从更低速度比例开始，`beta_vel` 让同样的超限更痛。 |
| `safe/target_limit` | 参考 `100` 步内有 `4` 个关节进入边界带一半深度，即 $\left(e_{t,j}^{\text{target}}/m_j\right)^2\approx 0.25$，总和约 `1.0`。 | 有 $R\approx 100\cdot \beta_{\text{target}}\cdot 1.0$，故 $\beta_{\text{target}}^\star\approx 0.1$。 | 建议 `beta_target≈0.08~0.12`。`target_margin_ratio` 负责定义边界带宽度，`beta_target` 决定进入边界带后的实际成本。 |
| `safe/joint_move` | 参考 `300` 步内，`12` 个关节平均每步变化量从 `0.02 rad` 降到 `0.015 rad`，则 $\sum_j(\Delta q_j)^2\approx12\times0.015^2=0.0027$。 | 有 $R\approx300\cdot\beta_{\text{joint\_move}}\cdot0.0027$，故 $\beta_{\text{joint\_move}}^\star\approx10/0.81\approx12.35$。 | 建议 `beta_joint_move≈12`。这一项用于压制高频关节抽动，降低期望步进后需要显著提高。 |
| `safe/cup_door_prox` | 参考杯体对门板侵入 `1 cm`，并持续 `10` 步，则每步侵入平方为 `0.01^2=10^{-4}`。 | 有 $R\approx 10\cdot \beta_{\text{cup\_door\_prox}}\cdot 10^{-4}$，故 $\beta_{\text{cup\_door\_prox}}^\star\approx 10^4$。 | 建议 `beta_cup_door_prox≈10^4`。若希望 `2 cm` 侵入就明显更痛，可以维持阈值不变而把系数提升到 `1.5×10^4` 左右。 |
| `safe/base_zero_speed` | 参考整段 episode 低速静稳时可能持续激活；希望在 $s_t^{\text{base-speed}}\approx0.025$ 附近仍保留约 `0.6` 倍零速度奖励。 | 由 $e^{-\lambda_{\text{base-speed}}\cdot0.025}\approx0.64$ 得 $\lambda_{\text{base-speed}}\approx18$。`w_base_zero_speed` 仍保持弱正则 `0.02`，避免低速生存奖励压过任务进展。 | 建议 `w_base_zero_speed=0.02`、`lambda_base_speed≈18`。这会让零速度奖励比旧配置更快衰减。 |
| `safe/base_speed` | 参考底盘在 `100` 步内维持“明显偏快但未失控”的速度，降低速度期望后令 $s_t^{\text{base-speed}}\approx0.0625$。 | 有 $R\approx100\cdot w_{\text{base-speed}}\cdot0.0625$，故 $w_{\text{base-speed}}^\star\approx10/6.25=1.6$。 | 建议 `w_base_speed≈1.6`。这是底盘速度的主惩罚项，降低速度预估后需要提高系数。 |
| `safe/base_cmd_delta` | 参考底盘命令在 `100` 步内存在中等抖动，降低期望后令 $\|\Delta u_t^{\text{base}}\|_2^2\approx0.02$。 | 有 $R\approx100\cdot\beta_{\text{base\_cmd}}\cdot0.02$，故 $\beta_{\text{base\_cmd}}^\star\approx5.0$。 | 建议 `beta_base_cmd≈5`。这一项相当于压制底盘速度命令的快速变化。 |
| `safe/drop` | 掉杯时触发 1 次。 | 若坚持与其他子项同量级，则应取 $w_{\text{drop}}^\star\approx 10$；若希望它比一般 shaping 更像“硬安全事件”，也通常只需要 `20~30`。 | 建议 `w_drop≈20~30`。它可以比普通 dense shaping 更强，但不需要高到远离整体量级体系。 |
| `safe/base_heading` | 参考底盘在 `20` 步内持续以 `30^\circ` 偏向门框横向方向，则 $(f_t^{\text{base}}\cdot t^{\text{doorway}})^2=\sin^2 30^\circ=0.25$。 | 有 $R\approx 20\cdot \beta_{\text{base\_heading}}\cdot 0.25$，故 $\beta_{\text{base\_heading}}^\star\approx 2.0$。 | 建议 `beta_base_heading≈2.0`。它应让 `15^\circ` 偏航只是轻罚，`30^\circ` 以上偏航开始明显不划算。 |
| `safe/base_corridor` | 参考底盘矩形 footprint 持续 `10` 步越界 `3 cm`，则 $(e_t^{\text{corridor}})^2=0.03^2=9\times 10^{-4}$。 | 有 $R\approx 10\cdot \beta_{\text{base\_corridor}}\cdot 9\times 10^{-4}$，故 $\beta_{\text{base\_corridor}}^\star\approx 1111$。若把参考越界改成 `2 cm`，则会得到约 `2500`。 | 建议 `beta_base_corridor≈1000~2500`，中间值可取 `1500~2000`。这项是阻止“擦门框绕过去”的几何主约束，合理量级天然就在 `10^3`。 |

如果后续要在配置文件里按这一套 `O(10)` 原则落地，更合理的做法不是逐项独立拍脑袋，而是按下面的顺序组织：

1. 先固定真正的任务主项：
   `w_delta≈8~10`、`w_open≈10~20`、`w_base_cross≈40~60`。
   这三项共同定义“打开门并真正穿过去”。

2. 再固定前置引导项：
   `w_approach≈0.4~0.6`、`w_base_align≈0.005~0.01`、`w_base_forward≈25`、`w_base_centerline≈0.01~0.03`。
   它们只负责把手和底盘引到正确位置并维持正确通道，不应压过主任务项。

3. 然后固定持杯稳定项：
   `w_zero_acc≈0.055`、`lambda_acc≈8`、`w_zero_ang≈0.06`、`lambda_ang≈3`、`w_acc≈0.04`、`w_ang≈1.6e-3`、`w_tilt≈3.0~3.5`。
   目标是让“持续稳定持杯”的总成本也在 `O(10)`。

4. 最后固定安全约束：
   `mu≈0.75`、`beta_vel≈0.20`、`beta_target≈0.08~0.12`、`beta_joint_move≈12`、`beta_cup_door_prox≈10^4`、`lambda_base_speed≈18`、`w_base_speed≈1.6`、`beta_base_cmd≈5`、`w_drop≈20~30`、`beta_base_heading≈2.0`、`beta_base_corridor≈1000~2500`。
   这些项的目标不是让 reward 变大，而是让明显危险或 exploit 行为在 episode 级累计上“至少不比完成任务更划算”。

### 3.5 当前默认训练下奖励的实际激活情况

由于当前主训练按 `empty / left / right / both` 四种模式均匀采样，所以奖励项的激活情况是分模式的：

- 在 `empty` 回合里：
  $$
  r_t^{\text{stab},L}=r_t^{\text{stab},R}=r_t^{\text{cup\_door\_prox}}=r_t^{\text{drop}}=0.
  $$
- 在 `left` 回合里，只有左侧稳定性项，以及左侧持杯相关的 `cup_door_prox` / 掉杯约束可能生效。
- 在 `right` 回合里，只有右侧稳定性项，以及右侧持杯相关的 `cup_door_prox` / 掉杯约束可能生效。
- 在 `both` 回合里，左右两侧稳定性项以及两侧持杯相关约束都可能生效。

因此当前默认训练的总奖励不再退化成纯空手形式，而是一个 occupancy-conditioned 混合目标：

$$
r_t = r_t^{\text{task}} + r_t^{\text{stab},L}+r_t^{\text{stab},R}-r_t^{\text{safe}},
$$

其中不同 episode 会根据 occupancy 样本激活不同的分项。

### 3.6 一个必须说明的实现时序细节

从数学上，我们通常把 $r_t$ 视为由当前动作后的几何量直接计算：

$$
r_t = r(s_t,a_t,s_{t+1}).
$$

但当前实现里，若干用于奖励和终止的量，例如：

- 末端线加速度
- 末端角加速度
- 倾斜 proxy
- 接近距离
- 掉杯标记

是在 `_get_observations()` 中缓存的；而 Isaac Lab 的 `DirectRLEnv.step()` 调用顺序是：

1. `_pre_physics_step`
2. 物理积分
3. `_get_dones`
4. `_get_rewards`
5. auto-reset
6. `_get_observations`

因此当前实现实际上存在“一步缓存延迟”：

$$
\text{cached feature used at step } t
\approx
\text{feature refreshed after step } t-1.
$$

这不改变总体训练目标，但在精读 reward 时必须知道：当前代码里的部分稳定性项和接近项是“上一轮观测刷新、本轮奖励消费”的。

### 3.7 网络结构

当前策略网络使用 rl_games 的 `continuous_a2c_logstd` 模型（对应 `ModelA2CContinuousLogStd` 类），配置为 `separate: False`，即策略均值头 $\mu_\theta$ 和值函数头 $V_\phi$ 共享同一个 MLP 骨干网络。

#### 3.7.1 骨干网络

骨干网络的输入为 90 维策略观测 $o_t^\pi$（经运行均值/标准差归一化后），依次通过三个全连接层：

$$
h_t = \mathrm{ELU}\!\big(W_3\,\mathrm{ELU}(W_2\,\mathrm{ELU}(W_1\,\hat o_t^\pi + b_1) + b_2) + b_3\big)\in\mathbb R^{128},
$$

其中 $\hat o_t^\pi$ 是归一化后的观测。各层参数：

| 层 | 输入维度 | 输出维度 | 激活函数 | 初始化 |
| --- | --- | --- | --- | --- |
| $W_1$ | $90$ | $512$ | ELU | default |
| $W_2$ | $512$ | $256$ | ELU | default |
| $W_3$ | $256$ | $128$ | ELU | default |

其中 default 初始化为 PyTorch `nn.Linear` 的默认行为（Kaiming uniform）。

#### 3.7.2 策略均值头

$$
\mu_\theta(o_t^\pi) = W_\mu\,h_t + b_\mu\in\mathbb R^{15}.
$$

- 无激活函数（`mu_activation: None`）。
- 初始化：default。

#### 3.7.3 值函数头

$$
V_\phi(o_t^\pi) = W_v\,h_t + b_v\in\mathbb R.
$$

- 无激活函数。
- 与策略均值头共享同一个骨干输出 $h_t$。
- 输出经运行均值/标准差反归一化后作为最终值估计。

#### 3.7.4 标准差参数

$$
\sigma = \exp(\sigma_0)\in\mathbb R^{15}.
$$

- `fixed_sigma: True`：$\sigma_0$ 是一个与状态无关的可学习参数向量，不经过骨干网络。
- 初始化：`const_initializer`，$\sigma_0=-2\cdot\mathbf 1$，因此初始标准差 $\sigma=\exp(-2)\cdot\mathbf 1\approx0.135\cdot\mathbf 1$。
- 每个动作维度有独立的标准差参数。

#### 3.7.5 归一化

训练配置中启用了两种运行归一化：

- `normalize_input: True`：对策略观测 $o_t^\pi$ 维护运行均值 $\mu_{\text{obs}}$ 和标准差 $\sigma_{\text{obs}}$，实际骨干输入为 $\hat o_t^\pi = (o_t^\pi - \mu_{\text{obs}})/(\sigma_{\text{obs}}+\epsilon)$。
- `normalize_value: True`：对值函数输出维护运行均值 $\mu_V$ 和标准差 $\sigma_V$，最终值估计为 $V_\phi^{\text{raw}}\cdot\sigma_V+\mu_V$。

#### 3.7.6 完整前向计算路径

把以上组件拼接，一次前向传播的计算流程为：

$$
\hat o_t^\pi = \frac{o_t^\pi - \mu_{\text{obs}}}{\sigma_{\text{obs}}+\epsilon},
$$

$$
h_t = \mathrm{ELU}\!\big(W_3\,\mathrm{ELU}(W_2\,\mathrm{ELU}(W_1\hat o_t^\pi+b_1)+b_2)+b_3\big),
$$

$$
\mu_t = W_\mu h_t + b_\mu,
$$

$$
V_t^{\text{raw}} = W_v h_t + b_v,\qquad V_t = V_t^{\text{raw}}\cdot\sigma_V + \mu_V,
$$

$$
a_t \sim \mathcal N(\mu_t,\;\operatorname{diag}(\sigma^2)).
$$

#### 3.7.7 关于当前配置与不对称 Actor-Critic 的说明

当前 rl_games 配置仍然保持 actor 主干 `separate: False`，但已经额外启用了 `central_value_config`。这意味着：

- actor 仍以 90 维策略观测 $o_t^\pi$ 为输入，输出动作分布；
- critic 改为走独立的 central value 网络，以 103 维特权观测 $o_t^V$ 为输入。

环境通过 `state_space: 103` 提供 103 维特权观测，Isaac Lab 的 `RlGamesVecEnvWrapper` 会把环境中的 `critic` 观测组显式映射为 rl_games 的 `states`。在训练时，rl_games 的 `has_central_value=True`，因此这部分特权信息会被 critic 实际消费。

当前 central value 网络复用了 actor MLP 的骨干结构，即 `[512, 256, 128] + ELU`，但它与 actor 的策略头解耦，只负责估值。

<a id="sec-4"></a>
## 4. Policy 是如何训练的：完整 PPO 推导版

### 4.1 从强化学习目标开始

我们要学习一个参数化策略 $\pi_\theta(a_t\mid o_t^\pi)$，使得期望折扣回报最大：

$$
J(\theta)=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right].
$$

定义回报：

$$
G_t=\sum_{l=0}^{T-1-t}\gamma^l r_{t+l}.
$$

定义状态值函数和动作值函数：

$$
V^\pi(s_t)=\mathbb E_\pi[G_t\mid s_t],
$$
$$
Q^\pi(s_t,a_t)=\mathbb E_\pi[G_t\mid s_t,a_t].
$$

于是优势函数定义为：

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

优势函数的意义是：在当前状态下，动作 $a_t$ 比“按当前策略的平均水平”好多少。

### 4.2 Policy Gradient 定理

经典 policy gradient 给出：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t)\;
A^\pi(s_t,a_t)
\right].
$$

如果直接用这个式子做梯度上升，会遇到两个实际问题：

1. $A^\pi$ 不可直接得到，需要估计。
2. 采样策略和更新后策略一旦差异过大，训练会不稳定。

PPO 正是为了解决第 2 个问题而来。

### 4.3 从旧策略采样到重要性采样比

设采样 rollout 的行为策略是旧策略 $\pi_{\theta_{\text{old}}}$。我们想评估新参数 $\theta$ 下的目标，可以用重要性采样比：

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t^\pi)}
{\pi_{\theta_{\text{old}}}(a_t\mid o_t^\pi)}.
$$

于是最朴素的替代目标为：

$$
L^{\text{PG}}(\theta)=
\mathbb E_t\big[\rho_t(\theta)\hat A_t\big].
$$

如果直接最大化它，一旦 $\rho_t$ 偏离 1 太多，就会造成过大策略更新。

### 4.4 PPO 的 clipped surrogate

PPO 的核心思想是：如果新旧策略差得太远，就把改进幅度截住。

定义：

$$
L^{\text{CLIP}}(\theta)=
\mathbb E_t
\left[
\min\Big(
\rho_t(\theta)\hat A_t,\;
\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)\hat A_t
\Big)
\right].
$$

这里 $\varepsilon=0.2$。

这个式子分两种情况理解：

- 若 $\hat A_t>0$，说明动作比平均好，希望增大其概率，但不能增大太多。
- 若 $\hat A_t<0$，说明动作比平均差，希望减小其概率，但不能减小太猛。

也就是说，PPO 不是禁止策略变化，而是把单次更新限制在一个“相对信任域”近似里。

rl_games 实际实现的是最小化形式的 actor loss：

$$
L_{\text{actor}}(\theta)=
\mathbb E_t
\left[
\max\Big(
-\rho_t(\theta)\hat A_t,\;
-\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)\hat A_t
\Big)
\right].
$$

### 4.5 为什么需要 critic：优势函数的估计

在实际训练中，我们并不知道真实的 $A^\pi$。因此需要一个值函数近似器：

$$
V_\phi(o_t^V)\approx V^\pi(s_t).
$$

注意当前项目用的是不对称 actor-critic：

- actor 条件在 $o_t^\pi$
- critic 条件在 $o_t^V$

也就是说，critic 可以看到 actor 看不到的特权物理量，从而给出更低方差的优势估计。

### 4.6 TD 残差与 GAE 的推导

先定义一步 TD 残差：

$$
\delta_t=
r_t+\gamma(1-d_{t+1})V_\phi(o_{t+1}^V)-V_\phi(o_t^V),
$$

其中 $d_{t+1}\in\{0,1\}$ 表示下一步是否终止。

如果只用 $\delta_t$ 做优势估计，偏差小但方差大。Generalized Advantage Estimation 使用一个指数衰减加权和：

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\left(
\prod_{m=1}^{l}(1-d_{t+m})
\right)
\delta_{t+l}.
$$

把它展开可得递推式：

$$
\hat A_t=
\delta_t+\gamma\lambda(1-d_{t+1})\hat A_{t+1}.
$$

这正是当前 rl_games 实际使用的 backward recursion。

它的意义是：

- $\lambda\to 0$ 时，更像一步 TD。
- $\lambda\to 1$ 时，更像长回报 Monte Carlo。

当前训练取：

$$
\gamma=0.99,\qquad \lambda=0.95.
$$

### 4.7 回报目标与 advantage 标准化

用 GAE 得到优势后，return target 取为：

$$
\hat R_t=\hat A_t + V_{\phi,\text{old}}(o_t^V).
$$

当前训练还会对 advantage 做标准化：

$$
\tilde A_t=
\frac{\hat A_t-\mu_A}{\sigma_A+10^{-8}}.
$$

这样做的目的，是把不同 batch 上 advantage 的尺度拉回到可控范围，从而稳定 actor 梯度。

### 4.8 Critic 的 clipped value loss

critic 目标是让 $V_\phi(o_t^V)$ 逼近 $\hat R_t$。朴素 MSE 是：

$$
L_V^{\text{plain}}(\phi)
=
\mathbb E_t\Big[(V_\phi(o_t^V)-\hat R_t)^2\Big].
$$

PPO 常常也对 value 更新做 clipping。当前 rl_games 的定义是：

$$
V_\phi^{\text{clip}}(o_t^V)=
V_{\phi,\text{old}}(o_t^V)
+
\operatorname{clip}
\Big(
V_\phi(o_t^V)-V_{\phi,\text{old}}(o_t^V),
 -\varepsilon,\varepsilon
\Big).
$$

于是 critic loss 为：

$$
L_V(\phi)=
\mathbb E_t
\left[
\max\Big(
(V_\phi(o_t^V)-\hat R_t)^2,\;
(V_\phi^{\text{clip}}(o_t^V)-\hat R_t)^2
\Big)
\right].
$$

要特别说明一件实现细节：当前项目虽然在训练 YAML 中有 `value_clip_eps`，但主训练链路没有单独接这一路参数。当前 critic clipping 实际复用的就是 actor 的同一个 $\varepsilon=0.2$。

### 4.9 熵正则：维持探索

连续高斯策略的熵记为：

$$
\mathcal H\big(\pi_\theta(\cdot\mid o_t^\pi)\big).
$$

熵越大，动作分布越分散，探索越强；熵越小，策略越确定。PPO 中通常把熵作为负损失加入总目标：

$$
L_{\text{ent}}(\theta)= - c_{\text{ent}} \cdot
\mathbb E_t\big[\mathcal H(\pi_\theta(\cdot\mid o_t^\pi))\big].
$$

当前：

$$
c_{\text{ent}}=0.
$$

这意味着当前试训配置不再主动奖励更高熵。策略仍然是高斯策略，仍会根据 $\sigma_0$ 采样动作；但 PPO loss 不再额外推动 $\sigma_0$ 变大。这样做的目的，是先避免探索方差把动作推成随机饱和控制，再观察任务 reward 和安全/稳定项是否能形成有效学习信号。

### 4.10 Bounds loss：约束高斯均值不要漂太远

当前 rl_games 连续控制实现还额外加入一个均值边界损失。对每个动作维度 $j$，若均值 $\mu_{t,j}$ 超过软边界 $1.1$，就施加二次惩罚：

$$
L_{\text{bound}}(\theta)=
\mathbb E_t
\left[
\sum_{j=1}^{15}
\Big(
[\mu_{t,j}-1.1]_+^2 + [-1.1-\mu_{t,j}]_+^2
\Big)
\right].
$$

这里的求和维度是完整动作维度 $15=12+3$，约束对象是 policy 直接输出的未截断高斯均值 $\mu_\theta(o_t^\pi)$，不是经过 `clip_actions=1.0` 之后送入环境的动作。

当前权重是：

$$
c_{\text{bound}}=10^{-4}.
$$

这不是环境物理约束，而是策略输出层面的软约束，用于防止均值无界漂移。

### 4.11 当前项目实际优化的总损失

把 actor、critic、entropy 和 bounds loss 拼起来，当前 rl_games 实际最小化的是：

$$
L_{\text{total}}
=
L_{\text{actor}}
+
\frac{1}{2}c_V L_V
-c_{\text{ent}}\mathcal H
+c_{\text{bound}}L_{\text{bound}}.
$$

当前：

$$
c_V=0.5,\qquad c_{\text{ent}}=0,\qquad c_{\text{bound}}=10^{-4}.
$$

因此 critic 项的实际总前系数是：

$$
\frac{1}{2}c_V = 0.25.
$$

### 4.12 KL 与自适应学习率

当前训练不是固定学习率，而是 adaptive schedule。训练时会估计新旧高斯策略之间的经验 KL：

$$
\mathrm{KL}\big(\pi_{\theta_{\text{old}}}\|\pi_\theta\big).
$$

当前阈值是：

$$
\mathrm{KL}_{\text{th}}=0.008.
$$

rl_games 的自适应调度规则是：

- 若 $\mathrm{KL}>2\mathrm{KL}_{\text{th}}$，学习率除以 $1.5$
- 若 $\mathrm{KL}<0.5\mathrm{KL}_{\text{th}}$，学习率乘以 $1.5$
- 并约束在 $[10^{-6},10^{-3}]$ 内

即：

$$
\eta \leftarrow \max(\eta/1.5,10^{-6})
\quad\text{if}\quad \mathrm{KL}>0.016,
$$
$$
\eta \leftarrow \min(1.5\eta,10^{-3})
\quad\text{if}\quad \mathrm{KL}<0.004.
$$

需要注意，`rl_games` 原生 `AdaptiveScheduler` 的学习率上限是 $10^{-2}$。当前训练入口在启动 `Runner.load(...)` 前会按项目配置把 scheduler 上限 patch 为 `adaptive_lr_max=1e-3`，从而保留 KL 自适应机制，但避免学习率从初始值一路升到 $10^{-2}$。

当前初始学习率：

$$
\eta_0 = 2.0\times 10^{-4}.
$$

### 4.13 Rollout、batch、mini-batch 和优化步数

当前默认配置下：

- 并行环境数 $N=6144$
- horizon $H=64$

所以每轮 PPO 迭代收集的样本数是：

$$
B = NH = 6144\times 64 = 393216.
$$

mini-batch 数量为 24，因此每个 mini-batch 大小为：

$$
M = \frac{B}{24}=16384.
$$

每轮 PPO 迭代对同一批数据重复训练 2 个 epoch，因此每轮总优化步数为：

$$
2\times 24 = 48.
$$

总训练步数配置为：

$$
\text{total\_steps}=3{,}000{,}000{,}000.
$$

于是 PPO 训练总迭代轮数约为：

$$
\text{max\_epochs}
=
\left\lceil
\frac{3{,}000{,}000{,}000}{393216}
\right\rceil
=7630.
$$

### 4.14 当前训练链路下的几个关键实现约束

从训练数学上，还需要知道以下现实约束：

1. 当前 actor 和 critic 学习率必须相等。
   因为主链路只使用一个优化器，因此配置层面强制 `actor_lr == critic_lr`。
2. 当前是特权 critic，而不是完全对称 actor-critic。
3. 当前不是视觉 PPO。
   门信息通过 $6$ 维 door geometry 直接提供。

<a id="sec-5"></a>
## 5. 随机化：哪些量被随机化，如何进入训练

可以把当前环境的随机化统称为一个随机变量：

$$
\xi=
\big(\xi_{\text{geom}},\xi_{\text{dyn}},\xi_{\text{obs}},\xi_{\text{occ}}\big).
$$

### 5.1 几何随机化

几何随机化是基座位置与朝向的回合级采样：

$$
\xi_{\text{geom}}=(p_{\text{base}},\psi_{\text{base}}).
$$

它改变的是：

- 机器人与门的相对初始距离
- 与门面法向量的相对夹角
- 左右臂初始接触路径

因此，哪怕奖励函数不变，策略面对的最优动作序列也会随 $\xi_{\text{geom}}$ 改变。

### 5.2 动力学随机化

当前每个回合都独立采样：

$$
m_{\text{cup}}\sim\mathcal U(0.1,0.8),
$$
$$
m_{\text{door}}\sim\mathcal U(5.0,20.0),
$$
$$
c_{\text{door}}\sim\mathcal U(0.5,5.0).
$$

这些量在一个回合内保持常数，因此属于 episode-static domain randomization。

它们的作用路径是：

- $m_{\text{door}}$ 改变门板转动惯量和接触反力反馈。
- $c_{\text{door}}$ 改变铰链阻尼，直接影响开门难度和速度。
- $m_{\text{cup}}$ 改变持杯时的负载和惯性。

需要强调，当前每个环境只采样一个 `cup_mass` 标量，并把它同时写到左杯体和右杯体上，所以不是左右杯独立质量随机化，而是“同环境双杯同质量”的随机化。

### 5.3 观测噪声随机化

当前 actor 观测只对关节位置和关节速度加高斯噪声：

$$
\epsilon_t^q,\epsilon_t^{\dot q}\sim\mathcal N(0,0.01^2I).
$$

这属于 step-wise 随机化，因为每一步都会重新采样。

critic 不加这部分噪声，因此形成：

$$
o_t^\pi \neq o_t^V.
$$

这正是“actor 负责鲁棒决策，critic 负责更低方差估值”的典型不对称训练设计。

### 5.4 Occupancy 随机化接口与当前默认退化

环境本身支持在 reset 时由外部回调设置 occupancy：

$$
(m_t^L,m_t^R)\in\{0,1\}^2.
$$

理论上，这可以把任务分成四种模式：

- 空手
- 左手持杯
- 右手持杯
- 双手持杯

当前默认训练已经把这一定义接入到 Isaac Lab 的 reset event 中，因此策略面对的是一个固定混合分布：

$$
\xi_{\text{occ}}\sim p_{\text{train}}(m^L,m^R).
$$

其中当前默认配置给出：

$$
\begin{aligned}
p_{\text{train}}(0,0)&=0.25,\\
p_{\text{train}}(1,0)&=0.25,\\
p_{\text{train}}(0,1)&=0.25,\\
p_{\text{train}}(1,1)&=0.25.
\end{aligned}
$$

### 5.5 当前默认训练下“真正有效”的随机化

虽然代码里存在多种随机化接口，但当前默认训练里真正对学习产生实质影响的是：

1. 基座位置和朝向随机化
2. 门板质量随机化
3. 门铰链阻尼随机化
4. actor 的关节观测噪声
5. occupancy 模式随机化
6. 对持杯 episode 生效的杯质量随机化
7. 位置目标噪声

其中第 5 项和第 6 项会把训练任务从单一空手推门扩展成混合负载任务：

1. `empty` 样本只激活推门奖励和安全约束。
2. `left` 或 `right` 样本额外激活对应一侧的持杯稳定性奖励与掉杯终止。
3. `both` 样本同时激活两侧稳定性约束，因此策略需要在双侧负载下继续完成推门。

位置目标噪声当前也是启用的：

$$
\epsilon_t^{\text{target}}\sim\mathcal N(0,0.01^2I).
$$

它的影响路径是“归一化动作映射到 joint limits 之后，再叠加小幅目标角扰动并重新 clip”。相较于基座站位、门动力学和 occupancy，这一路随机化的量级更小，但它不是关闭状态。

这也是理解当前项目训练难度的关键：默认训练的 domain randomization 不再只是“门前站位 + 门动力学变化”，还包含“occupancy 模式切换 + 持杯负载变化 + 稳定性约束变化”。

<a id="sec-6"></a>
## 6. TensorBoard：当前记录了哪些参数，它们具体表示什么

### 6.1 日志来源和覆盖边界

当前 TensorBoard 标量由三条路径共同写入：

1. `rl_games.common.a2c_common.ContinuousA2CBase.write_stats(...)` 写入性能、PPO loss、学习率和 KL。
2. `rl_games.algos_torch.central_value.CentralValueTrain.train_net(...)` 写入 asymmetric central value 的 loss 和学习率。
3. `DoorPushTensorboardObserver` 基于环境 `extras` 写入 episode 级 reward、任务状态、occupancy 成功率和失败原因。

下面优先列当前默认训练链路实际会出现的标量 tag；最后再单独列出继承代码中存在、但当前默认环境通常不会产生的通用 tag。记第 $e$ 次写日志时，自定义 observer 收集到的完成 episode 集合为 $\mathcal K_e$，数量为 $K_e=|\mathcal K_e|$。对 episode $k$，长度为 $L_k$，单步总奖励为 $r_{k,t}^{\text{total}}$。任意单步分项 $x$ 的 episode 累计量记为：

$$
R_k[x]=\sum_{t=0}^{L_k-1} x_{k,t}.
$$

自定义 observer 的所有 `reward/*` 和 `reward_detail/*` 都按同一个规则聚合：

$$
\operatorname{TB}_e[x]=\frac{1}{K_e}\sum_{k\in\mathcal K_e} R_k[x].
$$

若某个日志周期内没有完成 episode，则这些 episode 级 tag 不写入。

### 6.2 性能类指标（rl_games 内置）

设本次 PPO epoch 采样到的环境步数为 $F_e$，环境 step 计时为 $T_{\text{step}}$，rollout 前向与采样计时为 $T_{\text{play}}$，PPO 更新计时为 $T_{\text{update}}$，`rl_games` 传入的整体计时为 $T_{\text{all}}$。

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `performance/step_inference_rl_update_fps` | 包含环境采样、策略前向和 PPO 更新的总体吞吐 | $\displaystyle F_e/T_{\text{all}}$ |
| `performance/step_inference_fps` | 不含 PPO 反向更新的 rollout 吞吐 | $\displaystyle F_e/T_{\text{play}}$ |
| `performance/step_fps` | 环境 step 阶段吞吐 | $\displaystyle F_e/T_{\text{step}}$ |
| `performance/rl_update_time` | PPO 更新耗时 | $\displaystyle T_{\text{update}}$ |
| `performance/step_inference_time` | rollout 前向与采样耗时 | $\displaystyle T_{\text{play}}$ |
| `performance/step_time` | 环境 step 耗时 | $\displaystyle T_{\text{step}}$ |

### 6.3 PPO 与 critic loss 指标（rl_games 内置）

设一个 PPO epoch 内实际执行的 mini-batch 更新集合为 $\mathcal U_e$。TensorBoard 中的 loss 是 `rl_games` 对这些 mini-batch loss 的均值。

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `losses/a_loss` | PPO actor clipped surrogate loss | $\displaystyle \frac{1}{\lvert\mathcal U_e\rvert}\sum_{u\in\mathcal U_e} L_{\text{actor}}^{(u)}$ |
| `losses/c_loss` | actor-critic 模型 value head 的 clipped value loss | $\displaystyle \frac{1}{\lvert\mathcal U_e\rvert}\sum_{u\in\mathcal U_e} L_V^{(u)}$ |
| `losses/entropy` | 高斯策略熵的均值；当前 `entropy_coef=0`，所以只监控、不进总 loss | $\displaystyle \frac{1}{\lvert\mathcal U_e\rvert}\sum_{u\in\mathcal U_e}\mathbb E_{t\in u}\left[\sum_{j=1}^{15}\frac{1}{2}\log(2\pi e\,\sigma_j^2)\right]$ |
| `losses/bounds_loss` | 策略高斯均值的软边界损失，约束的是未 clip 的 $\mu_\theta(o)$ | $\displaystyle \frac{1}{\lvert\mathcal U_e\rvert}\sum_{u\in\mathcal U_e}\mathbb E_{t\in u}\sum_{j=1}^{15}\left([\mu_{t,j}-1.1]_+^2+[-1.1-\mu_{t,j}]_+^2\right)$ |
| `losses/cval_loss` | asymmetric central value network 的 clipped value loss | $\displaystyle \frac{1}{\lvert\mathcal U_e^{V}\rvert}\sum_{u\in\mathcal U_e^{V}} L_{V,\text{central}}^{(u)}$ |

其中 actor loss 和 value loss 的展开式见第 4 节。`losses/cval_loss` 使用 critic 观测 $o_t^V$ 训练 central value；`losses/c_loss` 则来自 actor-critic 主模型里的 value 输出。当前配置启用了 `central_value_config`，所以 `losses/cval_loss` 会写入。

### 6.4 PPO 信息类指标（rl_games 内置）

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `info/last_lr` | 当前 actor-critic 优化器实际学习率 | $\displaystyle \eta_e\cdot m_e$，当前连续 PPO 中 $m_e=\text{lr\_mul}=1$ |
| `info/lr_mul` | 学习率乘子 | $\displaystyle m_e$，当前通常为 $1$ |
| `info/e_clip` | PPO ratio clip 半宽 | $\displaystyle \varepsilon_e=\varepsilon\cdot m_e$，当前通常为 $0.2$ |
| `info/kl` | 新旧高斯策略的经验 KL 均值 | $\displaystyle \frac{1}{\lvert\mathcal U_e\rvert}\sum_{u\in\mathcal U_e}\mathbb E_{t\in u}\left[\mathrm{KL}\big(\mathcal N(\mu_{\text{new}},\sigma_{\text{new}})\,\|\,\mathcal N(\mu_{\text{old}},\sigma_{\text{old}})\big)\right]$ |
| `info/epochs` | PPO epoch 编号 | $\displaystyle e$ |
| `info/cval_lr` | central value 优化器学习率 | $\displaystyle \eta_e^{V}$ |

当前项目保留 adaptive schedule，但在训练入口把 `rl_games` 原生上限 `1e-2` 改为 `adaptive_lr_max=1e-3`。因此正常情况下 `info/last_lr` 和 `info/cval_lr` 都不应超过 `0.001`。

### 6.5 rl_games 常规 episode 指标（rl_games 内置）

`rl_games` 还会维护一个完成 episode 的 rolling meter。记该 rolling meter 当前保留的 episode 集合为 $\mathcal W_e$，数量为 $W_e$。

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `rewards/step` | rolling window 内原始 episode return，横轴为 frame | $\displaystyle \frac{1}{W_e}\sum_{k\in\mathcal W_e}\sum_{t=0}^{L_k-1} r_{k,t}^{\text{total}}$ |
| `rewards/iter` | 同上，横轴为 PPO epoch | 同 `rewards/step` |
| `rewards/time` | 同上，横轴为墙钟时间 | 同 `rewards/step` |
| `shaped_rewards/step` | reward shaper 后的 episode return，横轴为 frame | $\displaystyle \frac{1}{W_e}\sum_{k\in\mathcal W_e}\sum_{t=0}^{L_k-1} \tilde r_{k,t}$ |
| `shaped_rewards/iter` | 同上，横轴为 PPO epoch | 同 `shaped_rewards/step` |
| `shaped_rewards/time` | 同上，横轴为墙钟时间 | 同 `shaped_rewards/step` |
| `episode_lengths/step` | rolling window 内平均 episode 长度，横轴为 frame | $\displaystyle \frac{1}{W_e}\sum_{k\in\mathcal W_e}L_k$ |
| `episode_lengths/iter` | 同上，横轴为 PPO epoch | 同 `episode_lengths/step` |
| `episode_lengths/time` | 同上，横轴为墙钟时间 | 同 `episode_lengths/step` |

`DefaultRewardsShaper` 的逐步公式是：

$$
\tilde r_t
=
\operatorname{clip}\big((r_t+\text{shift})\cdot\text{scale},\ \text{min},\ \text{max}\big),
$$

若 `log_val=True`，再取 $\log(\tilde r_t)$。当前默认只设置 `scale=1.0`，没有 shift、clip 或 log，因此默认近似为：

$$
\tilde r_t=r_t.
$$

### 6.6 大项 reward 指标（自定义 observer）

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `reward/task` | 任务模块 episode 累计均值 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t r_{k,t}^{\text{task}}$ |
| `reward/stab_left` | 左臂稳定性模块 episode 累计均值 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t r_{k,t}^{\text{stab},L}$ |
| `reward/stab_right` | 右臂稳定性模块 episode 累计均值 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t r_{k,t}^{\text{stab},R}$ |
| `reward/safe` | 安全模块 episode 累计均值；最终总奖励会减去它，其中 `base_zero_speed` 在该组合内以负号进入 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t r_{k,t}^{\text{safe}}$ |
| `reward/total` | 最终送给 PPO 的总奖励 episode 累计均值 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \left(r_{k,t}^{\text{task}}+r_{k,t}^{\text{stab},L}+r_{k,t}^{\text{stab},R}-r_{k,t}^{\text{safe}}\right)$ |

### 6.7 任务 reward 细项（自定义 observer）

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `reward_detail/task/delta` | 门角增量奖励 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w(\theta_{k,t})(\theta_{k,t}-\theta_{k,t-1})$ |
| `reward_detail/task/open_bonus` | 第一次达到开门阈值的一次性 bonus | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{open}}\mathbf 1[\theta_{k,t}\ge\theta_{\text{succ}}]\mathbf 1[\text{first-crossing}_{k,t}]$ |
| `reward_detail/task/approach` | 手部接近门板的加权 shaping | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \mathbf 1[\theta_{k,t}<\theta_{\text{stop}}]w_{\text{approach}}s_{k,t}^{\text{approach}}$ |
| `reward_detail/task/approach_raw` | 未乘权重和 stop gate 的归一化接近度 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t s_{k,t}^{\text{approach}}$ |
| `reward_detail/task/base_align` | 底盘朝推门方向对齐的弱 shaping | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{base\_align}}g_{k,t}^{\text{near}}g_{k,t}^{\text{range}}g_{k,t}^{\text{align}}$ |
| `reward_detail/task/base_forward` | 底盘沿推门方向的连续推进奖励 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{base\_forward}}g_{k,t}^{\text{align}}g_{k,t}^{\text{range}}\max(\delta_{k,t-1}^{\text{plane}}-\delta_{k,t}^{\text{plane}},0)$ |
| `reward_detail/task/base_centerline` | 底盘靠近门洞中线的弱 shaping | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{base\_centerline}}g_{k,t}^{\text{align}}g_{k,t}^{\text{range}}s_{k,t}^{\text{center}}$ |
| `reward_detail/task/base_cross` | 底盘穿门后在内侧半空间的推进奖励 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t g_{k,t}^{\text{align}}g_{k,t}^{\text{range}}\mathbf 1[\theta_{k,t}\ge\theta_{\text{base-cross}}]w_{\text{base\_cross}}\max(p_{k,t}^{\text{inside}}-p_{k,t-1}^{\text{inside}},0)$ |

这里的 $s_t^{\text{approach}}$、$g_t^{\text{align}}$、$g_t^{\text{range}}$、$g_t^{\text{near}}$、$s_t^{\text{center}}$ 和 $p_t^{\text{inside}}$ 的定义见第 3.4.2 节。

### 6.8 稳定性 reward 细项（自定义 observer）

对侧别 $c\in\{L,R\}$，`stab_left/*` 对应 $c=L$，`stab_right/*` 对应 $c=R$。注意代码写入 TensorBoard 的 `acc`、`ang`、`tilt` 是已经带负号的 signed reward 分项。

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `reward_detail/stab_left/zero_acc`、`reward_detail/stab_right/zero_acc` | 持杯侧末端线加速度接近 0 的奖励 | $\displaystyle \frac{1}{K_e}\sum_{i\in\mathcal K_e}\sum_t m_{i,t}^{c}w_{\text{zero-acc}}\exp(-\lambda_{\text{acc}}\|a_{i,t}^{c}\|_2^2)$ |
| `reward_detail/stab_left/zero_ang`、`reward_detail/stab_right/zero_ang` | 持杯侧末端角加速度接近 0 的奖励 | $\displaystyle \frac{1}{K_e}\sum_{i\in\mathcal K_e}\sum_t m_{i,t}^{c}w_{\text{zero-ang}}\exp(-\lambda_{\text{ang}}\|\alpha_{i,t}^{c}\|_2^2)$ |
| `reward_detail/stab_left/acc`、`reward_detail/stab_right/acc` | 持杯侧末端线加速度惩罚，日志中为负数 | $\displaystyle \frac{1}{K_e}\sum_{i\in\mathcal K_e}\sum_t -m_{i,t}^{c}w_{\text{acc}}\|a_{i,t}^{c}\|_2^2$ |
| `reward_detail/stab_left/ang`、`reward_detail/stab_right/ang` | 持杯侧末端角加速度惩罚，日志中为负数 | $\displaystyle \frac{1}{K_e}\sum_{i\in\mathcal K_e}\sum_t -m_{i,t}^{c}w_{\text{ang}}\|\alpha_{i,t}^{c}\|_2^2$ |
| `reward_detail/stab_left/tilt`、`reward_detail/stab_right/tilt` | 持杯侧杯体倾斜 proxy 惩罚，日志中为负数 | $\displaystyle \frac{1}{K_e}\sum_{i\in\mathcal K_e}\sum_t -m_{i,t}^{c}w_{\text{tilt}}\|u_{i,t}^{c}\|_2^2$ |

### 6.9 安全 reward 细项（自定义 observer）

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `reward_detail/safe/joint_vel` | 关节速度超过软阈值后的二次惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \beta_{\text{vel}}\sum_{j=1}^{12}\left[\max(\lvert\dot q_{k,t,j}\rvert-\mu\dot q_{\max,j},0)\right]^2$ |
| `reward_detail/safe/target_limit` | 执行目标进入 joint limit 边界带的惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \beta_{\text{target}}\sum_{j=1}^{12}\left(e_{k,t,j}^{\text{target}}/m_j\right)^2$ |
| `reward_detail/safe/cup_drop` | 杯子掉落惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{drop}}\mathbf 1[\text{cup\_dropped}_{k,t}]$ |
| `reward_detail/safe/joint_move` | 相邻控制步关节位置变化惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \beta_{\text{joint\_move}}\sum_{j=1}^{12}(q_{k,t,j}-q_{k,t-1,j})^2$ |
| `reward_detail/safe/cup_door_prox` | 持杯侧杯体低于门板距离阈值后的二次惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \sum_{c\in\{L,R\}}m_{k,t}^{c}\beta_{\text{cup\_door\_prox}}\left[\max(d_{\text{thresh}}-d_{k,t}^{c},0)\right]^2$ |
| `reward_detail/safe/base_zero_speed` | 底盘低速静稳奖励；在 `reward/safe` 内以负号进入 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{base-zero-speed}}\exp(-\lambda_{\text{base-speed}}s_{k,t}^{\text{base-speed}})$ |
| `reward_detail/safe/base_speed` | 底盘速度平方惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t w_{\text{base-speed}}s_{k,t}^{\text{base-speed}}$ |
| `reward_detail/safe/base_cmd_delta` | 底盘速度命令跳变惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \beta_{\text{base\_cmd}}\|\Delta u_{k,t}^{\text{base}}\|_2^2$ |
| `reward_detail/safe/base_heading` | 底盘前向偏向门框横向方向的惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \beta_{\text{base\_heading}}\left(f_{k,t}^{\text{base}}\cdot t^{\text{doorway}}\right)^2$ |
| `reward_detail/safe/base_corridor` | 底盘矩形 footprint 越出门洞走廊的惩罚 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\sum_t \beta_{\text{base\_corridor}}\left(e_{k,t}^{\text{corridor}}\right)^2$ |

其中：

$$
s_t^{\text{base-speed}}=v_{x,t}^2+v_{y,t}^2+\omega_{z,t}^2.
$$

`reward/safe` 中的组合方式是：

$$
r_t^{\text{safe}}
=
r_t^{\text{vel}}
+r_t^{\text{target}}
+r_t^{\text{drop}}
+r_t^{\text{joint\_move}}
+r_t^{\text{cup\_door\_prox}}
-r_t^{\text{base\_zero\_speed}}
+r_t^{\text{base\_speed}}
+r_t^{\text{base\_cmd\_delta}}
+r_t^{\text{base\_heading}}
+r_t^{\text{base\_corridor}}.
$$

### 6.10 任务状态、成功率和失败原因（自定义 observer）

| Tag | 含义 | 数学公式 |
| --- | --- | --- |
| `task_state/door_angle_final` | 完成 episode 的最终门角均值 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\theta_{k,L_k-1}$ |
| `task_state/base_crossed_rate` | 完成 episode 中 `base_link` 最终穿过门洞平面的比例 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\mathbf 1[\text{base\_crossed}_k]$ |
| `task_state/door_open_met_rate` | 完成 episode 中最终门角达到阈值的比例 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\mathbf 1[\theta_{k,L_k-1}\ge\theta_{\text{succ}}]$ |
| `success/all` | 完成 episode 总成功率 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}s_k$ |
| `success/empty` | 无杯场景条件成功率 | $\displaystyle \frac{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{empty}]s_k}{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{empty}]}$ |
| `success/left` | 仅左杯场景条件成功率 | $\displaystyle \frac{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{left}]s_k}{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{left}]}$ |
| `success/right` | 仅右杯场景条件成功率 | $\displaystyle \frac{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{right}]s_k}{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{right}]}$ |
| `success/both` | 双杯场景条件成功率 | $\displaystyle \frac{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{both}]s_k}{\sum_{k\in\mathcal K_e}\mathbf 1[\xi_k=\text{both}]}$ |
| `fail_reason/cup_drop` | 杯子掉落比例 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\mathbf 1[\text{cup\_dropped}_k]$ |
| `fail_reason/timeout` | 超时且非掉杯且非成功的比例 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\mathbf 1[\text{truncated}_k\land\lnot\text{cup\_dropped}_k\land\lnot s_k]$ |
| `fail_reason/not_crossed` | 超时、杯未掉、但底盘仍未穿过门洞的比例；它是 `timeout` 的子集 | $\displaystyle \frac{1}{K_e}\sum_{k\in\mathcal K_e}\mathbf 1[\text{truncated}_k\land\lnot\text{cup\_dropped}_k\land\lnot\text{base\_crossed}_k]$ |

成功指示量为：

$$
s_k=
\mathbf 1[
\theta_{k,L_k-1}\ge\theta_{\text{succ}}
\land
\text{base\_crossed}_k
\land
\lnot\text{cup\_dropped}_k
].
$$

occupancy 模式 $\xi_k$ 的定义为：

$$
\xi_k=
\begin{cases}
\text{empty}, & (m^L_k,m^R_k)=(0,0),\\
\text{left}, & (m^L_k,m^R_k)=(1,0),\\
\text{right}, & (m^L_k,m^R_k)=(0,1),\\
\text{both}, & (m^L_k,m^R_k)=(1,1).
\end{cases}
$$

### 6.11 PPO diagnostics（rl_games 内置）

当前默认 `use_diagnostics=True`，所以会启用 `PpoDiagnostics`。这些 tag 的横轴是 PPO epoch，而不是 frame。

| Tag | 出现条件 | 含义 | 数学公式 |
| --- | --- | --- | --- |
| `diagnostics/rms_value/mean` | 当前 `normalize_value=True` | value normalizer 的 running mean | $\displaystyle \mu_V$ |
| `diagnostics/rms_value/var` | 当前 `normalize_value=True` | value normalizer 的 running variance | $\displaystyle \sigma_V^2$ |
| `diagnostics/exp_var` | 当前 `use_diagnostics=True` | value 对 return 的 explained variance | $\displaystyle 1-\frac{\operatorname{Var}(\hat R_t-V_t)}{\operatorname{Var}(\hat R_t)}$ |
| `diagnostics/clip_frac/<mini_epoch>` | 当前 `use_diagnostics=True` | 第 `<mini_epoch>` 个 mini-epoch 中 PPO ratio 超出 clip 区间的比例 | $\displaystyle \frac{1}{B}\sum_t\mathbf 1\left[\log\rho_t<\log(1-\varepsilon)\lor\log\rho_t>\log(1+\varepsilon)\right]$ |
| `diagnostics/rms_advantage/mean` | 仅当 `normalize_rms_advantage=True` | advantage RMS normalizer 的 running mean | $\displaystyle \mu_A$ |
| `diagnostics/rms_advantage/var` | 仅当 `normalize_rms_advantage=True` | advantage RMS normalizer 的 running variance | $\displaystyle \sigma_A^2$ |

当前默认配置没有设置 `normalize_rms_advantage=True`，所以通常不会出现 `diagnostics/rms_advantage/mean` 和 `diagnostics/rms_advantage/var`。`diagnostics/clip_frac/<mini_epoch>` 中 `<mini_epoch>` 对当前 `mini_epochs=3` 通常是 `0`、`1`、`2`。

### 6.12 当前不会由默认环境产生的继承项

`IsaacAlgoObserver` 和 `rl_games` 代码中还有几类通用 tag，但当前 `DoorPushEnv` 默认没有提供对应输入，因此通常不会出现：

| Tag 形式 | 不出现原因 | 若出现时的公式 |
| --- | --- | --- |
| `Episode/<key>` | 当前环境没有在 `infos` 中提供 `episode` 字典 | $\displaystyle \frac{1}{K}\sum_k \text{episode}_k[\text{key}]$ |
| `<scalar_info>/frame`、`<scalar_info>/iter`、`<scalar_info>/time` | 当前 `extras` 中没有标量型 direct info；已有 extras 都由自定义 observer 专门消费 | 直接记录该 scalar info 的当前值 |
| `scores/mean`、`scores/iter`、`scores/time` | 当前 `IsaacAlgoObserver.mean_scores` 没有被本环境更新 | $\displaystyle \frac{1}{K}\sum_k \text{score}_k$ |
| `losses/<aux_loss_name>` | 当前 actor-critic 网络没有返回 auxiliary loss | $\displaystyle \frac{1}{|\mathcal U_e|}\sum_{u\in\mathcal U_e}L_{\text{aux}}^{(u)}$ |

需要特别区分：`use_diagnostics` 只负责 `diagnostics/*`，不会自动写出 `reward_detail/*`、`success/*` 或 `fail_reason/*`。这些项目指标来自 `DoorPushTensorboardObserver`。

<a id="sec-7"></a>
## 7. 一句话总结

当前项目默认训练的数学本质是：在随机基座站位、随机门动力学、随机 occupancy、观测噪声和位置目标噪声下，使用一个以 $90$ 维低维观测为输入、输出“12 维双臂归一化 joint command + 3 维底盘速度命令”的高斯策略，通过带特权 critic 的 PPO 最大化“开门增量 + 一次性开门成功 + 近门 shaping + 底盘朝向对齐 + 底盘沿推门方向推进 + 底盘靠近门洞中线 + 底盘穿门进展 + 持杯稳定性奖励 + 弱底盘零速度奖励 - 速度越界 - 目标边界带惩罚 - 关节移动惩罚 - 杯体贴门惩罚 - 底盘速度惩罚 - 底盘命令跳变惩罚 - 掉杯失败”的条件期望回报。
