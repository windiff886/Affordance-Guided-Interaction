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
| $N$ | 并行环境数 | $3072$ |
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
  \theta_t \ge \theta_{\text{target}},\qquad \theta_{\text{target}}=1.57\ \text{rad}.
  $$
- 失败事件：
  $$
  \text{cup\_dropped}_t = 1.
  $$
- 时间截断：
  $$
  t\ge 900.
  $$

当前代码里，episode 的 success 标记不是“达到 1.2 rad”，而是：

$$
\text{success}_t=
\mathbf 1[\theta_t\ge 1.57]\cdot \mathbf 1[\text{cup\_dropped}_t=0].
$$

因此有两点必须区分：

- $\theta_{\text{succ}}=1.2$ rad 只是奖励中的一次性 bonus 阈值，不是 success 阈值。
- 如果门只推到 $1.2$ rad，随后因为超时或掉杯结束，当前代码下仍然不算成功。
- 即使某一步同时满足“门角达到 $1.57$ rad”和“杯子脱落”，当前代码也不会把该回合记为成功。

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
c_{\text{ref}}=(3.72,\ 0.27).
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
v_t^{\text{ee},k,B},
\omega_t^{\text{ee},k,B},
a_t^{\text{ee},k,B},
\alpha_t^{\text{ee},k,B}
\big]\in\mathbb R^{19}.
$$

其中线加速度和角加速度不是简单在固定 `base_link` 系下跨帧差分。当前实现先在世界系中计算 EE 相对移动底盘的速度：

$$
v_{t,\text{rel}}^{\text{ee},k,W}
=
v_t^{\text{ee},k,W}
-\Big(v_t^{\text{base},W}+\omega_t^{\text{base},W}\times (p_t^{\text{ee},k,W}-p_t^{\text{base},W})\Big),
$$

$$
\omega_{t,\text{rel}}^{\text{ee},k,W}
=
\omega_t^{\text{ee},k,W}-\omega_t^{\text{base},W},
$$

然后再对世界系相对速度做差分，并旋回当前 base 坐标系：

$$
a_t^{\text{ee},k,B}=
\operatorname{Rot}(W\rightarrow B_t)\,
\frac{v_{t,\text{rel}}^{\text{ee},k,W}-v_{t-1,\text{rel}}^{\text{ee},k,W}}{\Delta t},
$$
$$
\alpha_t^{\text{ee},k,B}=
\operatorname{Rot}(W\rightarrow B_t)\,
\frac{\omega_{t,\text{rel}}^{\text{ee},k,W}-\omega_{t-1,\text{rel}}^{\text{ee},k,W}}{\Delta t}.
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

#### 3.1.6 底盘本体项

移动底盘版本还会给策略一个 6 维底盘块：

$$
o_t^{\text{base}}=
\big[
v_{x,t}^{B},\ v_{y,t}^{B},\ \omega_{z,t}^{B},\ u_t^{\text{base}}
\big]\in\mathbb R^6.
$$

其中前三维是当前底盘在 `base_link` 系下的真实速度，后三维是最近一次写入执行器的底盘速度命令。policy 控制的是这 3 维底盘速度命令；环境内部再把它们映射成四个轮子的速度目标。

### 3.2 Critic 观测 $o_t^V\in\mathbb R^{103}$

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
- $\sigma$ 不是状态相关函数，而是 state-independent 的固定参数向量。

因此当前策略学习的是：

1. 一个把 $90$ 维 actor 观测映射到 $15$ 维均值的函数 $\mu_\theta$。
2. 一个不随观测变化的对角协方差参数。

需要强调三点：

- 动作语义是“12 维双臂 joint command + 3 维底盘速度命令”，不是绝对目标角，也不是轮速，更不是 $\Delta q$。
- 动作先经过 rl_games 的 `clip_actions=1.0`，随后环境再把前 12 维映射到 joint limits，并在需要时叠加位置目标噪声；后 3 维则重标定为 $(v_x, v_y, \omega_z)$，再转换成轮速目标。
- gripper 不受策略控制，策略只控制双臂 12 个关节和底盘 3 个速度自由度。

### 3.4 任务奖励：开门 + 接近门板

总奖励先拆成任务项、稳定项和安全项：

$$
r_t = r_t^{\text{task}} + r_t^{\text{stab},L}+r_t^{\text{stab},R}-r_t^{\text{safe}}.
$$

任务项又分成三部分：

$$
r_t^{\text{task}}=
r_t^{\Delta \theta}
+
r_t^{\text{open}}
+
r_t^{\text{approach}}.
$$

#### 3.4.1 门角增量奖励

定义门角增量：

$$
\Delta\theta_t=\theta_t-\theta_{t-1}.
$$

当门尚未接近成功角时，增量奖励权重是常数 $w_\Delta$。一旦超过“成功 bonus 阈值” $\theta_{\text{succ}}=1.2$ rad，增量权重开始衰减：

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

默认训练入口的实际常数：

$$
w_\Delta=50,\qquad
k_{\text{decay}}=0.5,\qquad
\alpha=0.3.
$$

#### 3.4.2 一次性开门成功 bonus

当门角第一次超过 $\theta_{\text{succ}}=1.2$ rad 时，给予一次性 bonus：

这里的“success”只是奖励设计里的命名，它表示“达到 bonus 阈值”，不表示环境层面的 episode success。当前代码里 episode success 仍然要求：

$$
\theta_t \ge 1.57\ \text{rad}
\quad\text{且}\quad
\text{cup\_dropped}_t=0.
$$

$$
r_t^{\text{open}}=
w_{\text{open}}\cdot
\mathbf 1[\theta_t\ge \theta_{\text{succ}}]\cdot
\mathbf 1[\text{first-crossing at }t].
$$

默认训练入口的实际常数：

$$
w_{\text{open}}=250.
$$

#### 3.4.3 接近门板奖励

环境把门的可推面视为一个矩形面片。对任意末端点 $p$，先转到门面局部系：

$$
u = R_{\text{face}}^\top (p-c_{\text{face}})=
\begin{bmatrix}
u_x\\u_y\\u_z
\end{bmatrix}.
$$

矩形面片在局部坐标中满足：

$$
u_y\in[-0.45,0.45],\qquad u_z\in[-1.0,1.0].
$$

于是点到矩形面的距离被定义为：

$$
d(p,\mathcal F)=
\sqrt{
u_x^2 +
\big[\max(|u_y|-0.45,0)\big]^2 +
\big[\max(|u_z|-1.0,0)\big]^2
}.
$$

左、右末端分别算距离，再取更靠近门面的那只手：

$$
d_t=\min\big(d(p_t^{L},\mathcal F_t),\ d(p_t^{R},\mathcal F_t)\big).
$$

环境在回合中第一次得到 $d_t$ 时，把它缓存为初始距离 $d_0$。然后构造归一化接近度：

$$
s_t^{\text{approach}}
=
\max\left(
1-\frac{d_t^2}{d_0^2+\varepsilon_{\text{app}}},
0
\right),
\qquad
\varepsilon_{\text{app}}=10^{-6}.
$$

接近奖励只在门几乎还没打开时生效：

$$
r_t^{\text{approach}}=
\mathbf 1[\theta_t<\theta_{\text{stop}}]
\cdot w_{\text{approach}}
\cdot s_t^{\text{approach}},
\qquad \theta_{\text{stop}}=0.10.
$$

默认训练入口的实际常数：

$$
w_{\text{approach}}=2.
$$

这意味着训练初期，策略首先被鼓励“把至少一只手靠近门的可推面”，当门角超过 $0.10$ rad 以后，这个 shaping 就关闭，优化重点转向继续开门。

### 3.5 稳定性奖励：只在持杯侧激活

对每一侧 $k\in\{L,R\}$，定义 occupancy mask：

$$
m_t^k\in\{0,1\}.
$$

设该侧末端线加速度、角加速度和倾斜 proxy 分别为：

$$
a_t^k,\qquad \alpha_t^k,\qquad u_t^k.
$$

则该侧稳定性奖励为：

$$
r_t^{\text{stab},k}
=
m_t^k\Big(
w_{\text{zero-acc}}e^{-\lambda_{\text{acc}}\|a_t^k\|_2^2}

+w_{\text{zero-ang}}e^{-\lambda_{\text{ang}}\|\alpha_t^k\|_2^2}

-w_{\text{acc}}\|a_t^k\|_2^2
-w_{\text{ang}}\|\alpha_t^k\|_2^2
-w_{\text{tilt}}\|u_t^k\|_2^2
\Big).
$$

默认训练入口的实际常数：

$$
w_{\text{zero-acc}}=0.0,\quad \lambda_{\text{acc}}=2.0,
$$
$$
w_{\text{zero-ang}}=0.0,\quad \lambda_{\text{ang}}=1.0,
$$
$$
w_{\text{acc}}=5\times 10^{-4},\quad
w_{\text{ang}}=10^{-5},\quad
w_{\text{tilt}}=3\times 10^{-3}.
$$

如果 occupancy 为 0，则整项直接变成 0。

### 3.6 安全惩罚：速度越界、目标边界带、掉杯

安全惩罚定义为：

$$
r_t^{\text{safe}}=
r_t^{\text{vel}}
+r_t^{\text{target}}
+r_t^{\text{joint\_move}}
+r_t^{\text{drop}}.
$$

#### 3.6.1 关节速度超限惩罚

设每个关节的软速度上限为 $\dot q_{\max,j}$，环境只在超过比例阈值 $\mu\dot q_{\max,j}$ 后开始惩罚：

$$
e_{t,j}^{\text{vel}}
=
\max\big(|\dot q_{t,j}|-\mu\dot q_{\max,j},0\big).
$$

于是：

$$
r_t^{\text{vel}}
=
\beta_{\text{vel}}
\sum_{j=1}^{12}
\left(e_{t,j}^{\text{vel}}\right)^2.
$$

默认训练入口的实际常数：

$$
\mu=0.09,\qquad \beta_{\text{vel}}=0.5.
$$

#### 3.6.2 目标角边界带惩罚

注意这里惩罚的不是高斯策略的原始未裁剪样本，而是最终送入执行器的目标角 $\tilde q_t^\star$。先定义每个关节到最近限位的距离：

$$
d_{t,j}^{\text{limit}}
=
\min\big(\tilde q_{t,j}^\star-q_{\min,j},\ q_{\max,j}-\tilde q_{t,j}^\star\big),
$$

再定义边界带宽度比例 $\rho_{\text{margin}}$：

$$
m_j=\rho_{\text{margin}}(q_{\max,j}-q_{\min,j}),
$$

只有当目标进入边界带时才开始惩罚：

$$
e_{t,j}^{\text{target}}
=
\max(m_j-d_{t,j}^{\text{limit}},0).
$$

则：

$$
r_t^{\text{target}}
=
\beta_{\text{target}}
\sum_{j=1}^{12}
\left(\frac{e_{t,j}^{\text{target}}}{m_j}\right)^2.
$$

默认训练入口的实际常数：

$$
\beta_{\text{target}}=1.0,\qquad \rho_{\text{margin}}=0.1.
$$

这项的作用不是惩罚 raw overflow，而是告诉策略：“即使执行层始终安全，目标也不应长期贴着 joint limits 运行”。

#### 3.6.3 关节移动惩罚

惩罚相邻控制步之间的关节角度变化量，鼓励策略产生平滑的关节运动：

$$
r_t^{\text{joint\_move}}=
\beta_{\text{joint\_move}}\sum_{j=1}^{12}(q_{t,j}-q_{t-1,j})^2.
$$

默认训练入口的实际常数：

$$
\beta_{\text{joint\_move}}=0.1.
$$

#### 3.6.4 掉杯惩罚

$$
r_t^{\text{drop}}=
w_{\text{drop}}\cdot \mathbf 1[\text{cup\_dropped}_t].
$$

当前：

$$
w_{\text{drop}}=100.
$$

### 3.7 当前默认训练下奖励的实际激活情况

由于当前主训练按 `empty / left / right / both` 四种模式均匀采样，所以奖励项的激活情况是分模式的：

- 在 `empty` 回合里：
  $$
  r_t^{\text{stab},L}=r_t^{\text{stab},R}=r_t^{\text{drop}}=0.
  $$
- 在 `left` 回合里，只有左侧稳定性项和左侧掉杯约束可能生效。
- 在 `right` 回合里，只有右侧稳定性项和右侧掉杯约束可能生效。
- 在 `both` 回合里，左右两侧稳定性项和掉杯约束都可能生效。

因此当前默认训练的总奖励不再退化成纯空手形式，而是一个 occupancy-conditioned 混合目标：

$$
r_t = r_t^{\text{task}} + r_t^{\text{stab},L}+r_t^{\text{stab},R}-r_t^{\text{safe}},
$$

其中不同 episode 会根据 occupancy 样本激活不同的分项。

### 3.8 一个必须说明的实现时序细节

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

### 3.9 网络结构

当前策略网络使用 rl_games 的 `continuous_a2c_logstd` 模型（对应 `ModelA2CContinuousLogStd` 类），配置为 `separate: False`，即策略均值头 $\mu_\theta$ 和值函数头 $V_\phi$ 共享同一个 MLP 骨干网络。

#### 3.9.1 骨干网络

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

#### 3.9.2 策略均值头

$$
\mu_\theta(o_t^\pi) = W_\mu\,h_t + b_\mu\in\mathbb R^{15}.
$$

- 无激活函数（`mu_activation: None`）。
- 初始化：default。

#### 3.9.3 值函数头

$$
V_\phi(o_t^\pi) = W_v\,h_t + b_v\in\mathbb R.
$$

- 无激活函数。
- 与策略均值头共享同一个骨干输出 $h_t$。
- 输出经运行均值/标准差反归一化后作为最终值估计。

#### 3.9.4 标准差参数

$$
\sigma = \exp(\sigma_0)\in\mathbb R^{15}.
$$

- `fixed_sigma: True`：$\sigma_0$ 是一个与状态无关的可学习参数向量，不经过骨干网络。
- 初始化：`const_initializer`，$\sigma_0=\mathbf 0$，因此初始标准差 $\sigma=\exp(0)=\mathbf 1$。
- 每个动作维度有独立的标准差参数。

#### 3.9.5 归一化

训练配置中启用了两种运行归一化：

- `normalize_input: True`：对策略观测 $o_t^\pi$ 维护运行均值 $\mu_{\text{obs}}$ 和标准差 $\sigma_{\text{obs}}$，实际骨干输入为 $\hat o_t^\pi = (o_t^\pi - \mu_{\text{obs}})/(\sigma_{\text{obs}}+\epsilon)$。
- `normalize_value: True`：对值函数输出维护运行均值 $\mu_V$ 和标准差 $\sigma_V$，最终值估计为 $V_\phi^{\text{raw}}\cdot\sigma_V+\mu_V$。

#### 3.9.6 完整前向计算路径

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

#### 3.9.7 关于当前配置与不对称 Actor-Critic 的说明

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
c_{\text{ent}}=0.01.
$$

### 4.10 Bounds loss：约束高斯均值不要漂太远

当前 rl_games 连续控制实现还额外加入一个均值边界损失。对每个动作维度 $j$，若均值 $\mu_{t,j}$ 超过软边界 $1.1$，就施加二次惩罚：

$$
L_{\text{bound}}(\theta)=
\mathbb E_t
\left[
\sum_{j=1}^{12}
\Big(
[\mu_{t,j}-1.1]_+^2 + [-1.1-\mu_{t,j}]_+^2
\Big)
\right].
$$

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
c_V=0.5,\qquad c_{\text{ent}}=0.01,\qquad c_{\text{bound}}=10^{-4}.
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
- 并约束在 $[10^{-6},10^{-2}]$ 内

即：

$$
\eta \leftarrow \max(\eta/1.5,10^{-6})
\quad\text{if}\quad \mathrm{KL}>0.016,
$$
$$
\eta \leftarrow \min(1.5\eta,10^{-2})
\quad\text{if}\quad \mathrm{KL}<0.004.
$$

当前初始学习率：

$$
\eta_0 = 2.5\times 10^{-4}.
$$

### 4.13 Rollout、batch、mini-batch 和优化步数

当前默认配置下：

- 并行环境数 $N=3072$
- horizon $H=64$

所以每轮 PPO 迭代收集的样本数是：

$$
B = NH = 3072\times 64 = 196608.
$$

mini-batch 数量为 4，因此每个 mini-batch 大小为：

$$
M = \frac{B}{4}=49152.
$$

每轮 PPO 迭代对同一批数据重复训练 2 个 epoch，因此每轮总优化步数为：

$$
2\times 4 = 8.
$$

总训练步数配置为：

$$
\text{total\_steps}=300{,}000{,}000.
$$

于是 PPO 训练总迭代轮数约为：

$$
\text{max\_epochs}
=
\left\lceil
\frac{300{,}000{,}000}{196608}
\right\rceil
=1526.
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

### 6.1 一个先讲清楚的边界

当前训练日志由两部分共同组成：

- `rl_games` 默认训练日志
- 项目自定义 observer（`DoorPushTensorboardObserver`）基于 `extras` 追加的 episode 级指标

因此当前主训练不仅会记录常见 PPO/性能指标，也会把环境里的 `reward_info`、最终门角度、occupancy 分桶成功率和失败原因写进 TensorBoard。

项目自定义指标统一以累计环境步数（frame）作为横轴，rl_games 内置指标则使用其自身的横轴约定。

### 6.2 性能类指标（rl_games 内置）

#### `performance/step_inference_rl_update_fps`

定义近似为：

$$
\frac{\text{curr\_frames}}{\text{sampling + inference + update 总时间}}.
$$

它反映整轮训练闭环吞吐率，包括：

- 环境步进
- 策略前向
- PPO 反向更新

#### `performance/step_inference_fps`

定义近似为：

$$
\frac{\text{curr\_frames}}{\text{sampling + inference 时间}}.
$$

它反映不含反向更新时的 rollout 吞吐率。

#### `performance/step_fps`

定义为：

$$
\frac{\text{curr\_frames}}{\text{step\_time}}.
$$

它更接近“环境采样侧”的纯步进吞吐率。

#### `performance/rl_update_time`

单轮 PPO 更新耗时，单位秒。

#### `performance/step_inference_time`

单轮 rollout 采样和前向推理耗时，单位秒。

#### `performance/step_time`

环境步进阶段耗时，单位秒。

### 6.3 损失类指标（rl_games 内置）

#### `losses/a_loss`

当前 mini-batch 上 actor clipped loss 的均值，对应第 4 节中的：

$$
L_{\text{actor}}.
$$

数值越小不一定越好，因为它是带符号和 clipping 的 surrogate loss，不是直接的任务回报。

#### `losses/c_loss`

当前 mini-batch 上 critic clipped value loss 的均值，对应：

$$
L_V.
$$

它反映值函数对 $\hat R_t$ 的拟合误差。

#### `losses/entropy`

当前策略熵：

$$
\mathcal H\big(\pi_\theta(\cdot\mid o_t^\pi)\big).
$$

若该值快速塌到很低，通常说明探索不足。

#### `losses/bounds_loss`

当前配置里 `bounds_loss_coef=0.0001`，它对应：

$$
L_{\text{bound}}.
$$

如果这项持续增大，通常意味着策略均值在某些动作维上正试图远离合理区域。

### 6.4 PPO 信息类指标（rl_games 内置）

#### `info/last_lr`

当前实际学习率。由于当前连续 PPO 实现中 `lr_mul=1`，它基本就是 adaptive scheduler 更新后的真实 $\eta$。

#### `info/lr_mul`

学习率乘子。在当前连续控制实现里通常恒为 1，但 tag 仍会被写出。

#### `info/e_clip`

当前 clip 系数。写入逻辑是 `e_clip * lr_mul`，而当前 `lr_mul=1`，因此默认就是：

$$
0.2.
$$

#### `info/kl`

当前新旧策略之间的经验 KL：

$$
\mathrm{KL}\big(\pi_{\theta_{\text{old}}}\|\pi_\theta\big).
$$

这个指标直接驱动自适应学习率调度。

#### `info/epochs`

当前 PPO 训练迭代轮数，也就是第几次“采样一整个 rollout batch 并完成若干 mini-batch 更新”。

### 6.5 奖励指标（自定义 observer）

以下指标全部由 `DoorPushTensorboardObserver` 按已完成 episode 聚合后写入，横轴为累计环境步数（frame）。

环境在 episode 结束时会把 `episode_reward_info`、`door_angle`、`success`、`episode_left_occupied`、`episode_right_occupied`、`fail_cup_drop`、`fail_timeout` 放进 `infos`，observer 正是基于这些 extras 做窗口聚合。

#### `reward/*`

大项奖励，对应环境里的一级汇总项：

| Tag | 含义 |
| --- | --- |
| `reward/total` | 总回报 |
| `reward/task` | 任务奖励 |
| `reward/stab_left` | 左臂稳定性奖励 |
| `reward/stab_right` | 右臂稳定性奖励 |
| `reward/safe` | 安全惩罚 |

如果某个已完成 episode 的对应大项回报记为 $R_k^{\text{big}}$，那么 TensorBoard 中记录的是当前统计窗口（一个 PPO epoch）内完成 episode 的均值：

$$
\bar R^{\text{big}}=\frac{1}{K}\sum_{k=1}^{K}R_k^{\text{big}}.
$$

#### `reward_detail/*`

细项奖励，对应 `episode_reward_info` 中除一级汇总项以外的全部 key：

| Tag | 含义 |
| --- | --- |
| `reward_detail/task/delta` | 角度增量奖励 |
| `reward_detail/task/open_bonus` | 一次性开门奖励 |
| `reward_detail/task/approach` | 接近门板 shaping |
| `reward_detail/task/approach_raw` | 接近原始距离 |
| `reward_detail/stab_left/zero_acc` | 左加速度归零奖励 |
| `reward_detail/stab_left/zero_ang` | 左角速度归零奖励 |
| `reward_detail/stab_left/acc` | 左加速度惩罚 |
| `reward_detail/stab_left/ang` | 左角速度惩罚 |
| `reward_detail/stab_left/tilt` | 左倾斜惩罚 |
| `reward_detail/stab_right/*` | 右臂对应细项（同上 5 项） |
| `reward_detail/safe/joint_vel` | 关节速度越界惩罚 |
| `reward_detail/safe/target_limit` | 目标边界带惩罚 |
| `reward_detail/safe/joint_move` | 关节移动惩罚 |
| `reward_detail/safe/cup_drop` | 杯子掉落惩罚 |

它们同样记录当前统计窗口内已完成 episode 上对应分项的均值。

### 6.6 任务状态指标（自定义 observer）

#### `task_state/door_angle_final`

episode 结束时门角度的均值：

$$
\bar\theta_{\text{final}}=\frac{1}{K}\sum_{k=1}^{K}\theta_{k,\text{final}}.
$$

它不是 reward 分项，但属于任务进展最直接的状态量，因此单独记录。

### 6.7 成功率指标（自定义 observer）

当前成功率全部按 episode 完成口径定义，而不是逐步瞬时值。

#### `success/all`

记完成 episode 的成功指示量为 $s_k\in\{0,1\}$，则

$$
\text{success/all}=\frac{1}{K}\sum_{k=1}^{K}s_k.
$$

#### `success/empty`、`success/left`、`success/right`、`success/both`

把完成 episode 按 occupancy 模式分桶后，分别统计条件成功率：

$$
\text{success/mode}=
\frac{\sum_{k=1}^{K}\mathbf 1[\xi_k=\text{mode}]\,s_k}
{\sum_{k=1}^{K}\mathbf 1[\xi_k=\text{mode}]}.
$$

其中：

- `empty` 对应 $(m^L,m^R)=(0,0)$
- `left` 对应 $(1,0)$
- `right` 对应 $(0,1)$
- `both` 对应 $(1,1)$

### 6.8 失败原因指标（自定义 observer）

记录当前统计窗口内已完成 episode 中各类失败原因的发生率。两个失败原因互斥：

#### `fail_reason/cup_drop`

杯子掉落导致 episode 终止的比例：

$$
\text{fail\_reason/cup\_drop}=\frac{1}{K}\sum_{k=1}^{K}\mathbf 1[\text{cup\_dropped}_k].
$$

#### `fail_reason/timeout`

超时（达到最大步数）导致 episode 终止的比例，排除同时杯子掉落的情况：

$$
\text{fail\_reason/timeout}=\frac{1}{K}\sum_{k=1}^{K}\mathbf 1[\text{truncated}_k \land \lnot\text{cup\_dropped}_k].
$$

注意二者不重叠：如果杯子掉落和超时同时发生，只计入 `cup_drop`。未计入任何失败原因的 episode 即为成功 episode，因此：

$$
\text{success/all} + \text{fail\_reason/cup\_drop} + \text{fail\_reason/timeout} = 1.
$$

### 6.9 rl_games 常规回报指标（rl_games 内置）

只有当有完整 episode 结束时，下面这些 tag 才会出现。

#### `rewards/step`、`rewards/iter`、`rewards/time`

这是已完成 episode 的平均原始回报，只是横轴不同：

- `/step` 的横轴是累计环境步数 frame
- `/iter` 的横轴是 PPO epoch 编号
- `/time` 的横轴是墙钟时间

如果记某个完成 episode 的原始 return 为

$$
G_0=\sum_{t=0}^{T-1}r_t,
$$

那么这些 tag 记录的是最近一批完成 episode 的平均值。

#### `shaped_rewards/step`、`shaped_rewards/iter`、`shaped_rewards/time`

这是 reward shaper 处理后的 episode 回报。一般定义为：

$$
r_t^{\text{shaped}}=
\operatorname{clip}\big((r_t+\text{shift})\cdot \text{scale},\ \text{min},\text{max}\big),
$$

必要时还可附加对数变换。

但当前项目默认只设置了：

$$
\text{scale}=1.0,
$$

其余 shaping 选项没有打开，所以当前默认训练里：

$$
r_t^{\text{shaped}} \approx r_t.
$$

因此 `shaped_rewards/*` 与 `rewards/*` 在数值上通常几乎相同。

#### `episode_lengths/step`、`episode_lengths/iter`、`episode_lengths/time`

这是已完成 episode 的平均长度：

$$
\bar L = \frac{1}{K}\sum_{k=1}^{K} L_k.
$$

它能帮助判断训练是更多地提前成功终止，还是更多地拖到时间截断。

### 6.10 PPO Diagnostics 与 `use_diagnostics`

当前默认已经开启 `use_diagnostics=True`。它的意思不是“打开项目自定义 TensorBoard”，而是启用 `rl_games` 自己的 `PpoDiagnostics` 分支。

不开时，`rl_games` 使用的是 `DefaultDiagnostics`，不会额外写 PPO 诊断项。

开了以后，会额外出现：

- `diagnostics/rms_advantage/mean`
- `diagnostics/rms_advantage/var`
- `diagnostics/rms_value/mean`
- `diagnostics/rms_value/var`
- `diagnostics/exp_var`
- `diagnostics/clip_frac/<mini_epoch>`

其中：

- `diagnostics/exp_var` 是 value 预测与 return 之间的 explained variance
- `diagnostics/clip_frac/<mini_epoch>` 是该 mini-epoch 上 PPO clip fraction

需要特别区分：`use_diagnostics` 只负责这些 PPO 诊断量，它不会自动把 `reward_info`、`success/all` 或 `success/both` 之类的项目指标写出来。这些项目指标仍然来自自定义 observer。

<a id="sec-7"></a>
## 7. 一句话总结

当前项目默认训练的数学本质是：在随机基座站位、随机门动力学、随机 occupancy、观测噪声和位置目标噪声下，使用一个以 $90$ 维低维观测为输入、输出“12 维双臂归一化 joint command + 3 维底盘速度命令”的高斯策略，通过带特权 critic 的 PPO 最大化“开门增量 + 一次性开门成功 + 近门 shaping + 持杯稳定性奖励 - 速度越界 - 目标边界带惩罚 - 掉杯失败”的条件期望回报。
