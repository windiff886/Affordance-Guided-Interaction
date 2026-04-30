# 训练流水线详解（数学建模版）

本文只讨论当前项目默认训练入口对应的真实训练任务，即
`Affordance-DoorPush-Direct-v0`。文档目标不是复述代码调用关系，而是把“这个项目到底在优化什么、环境如何被数学化、动作和观测如何定义、奖励如何构造、PPO 如何训练、随机化如何进入模型、TensorBoard 究竟记录了什么”完整写清楚。

当前默认实现是一个 **handle-free push-door open-and-traverse** 任务：

- 一个 Isaac Lab `DirectRLEnv`。
- 一个 teacher PPO 策略。
- 一个 15 维 raw Gaussian action。
- 一个 79 维对称 actor-critic observation。
- 一个标量 reward。
- 无杯体、无 occupancy curriculum、无学生策略、无蒸馏、无门把手抓取。

除非特别注明，文中“当前默认值”均指 `scripts/train.py` 从
`configs/training/default.yaml`、`configs/env/default.yaml`、
`configs/task/default.yaml`、`configs/reward/default.yaml` 和 task registry 中的
`DoorPushEnvCfg` / `rl_games_ppo_cfg.yaml` 合并后的实际训练配置。

## 目录

- [0. 符号、时间尺度与当前默认常数](#sec-0)
- [1. 任务介绍：项目在训练什么，解决什么问题](#sec-1)
- [2. 环境构建：资产、坐标系与仿真初始化](#sec-2)
- [3. Action、Observation、Reward](#sec-3)
- [4. Policy 是如何训练的：PPO 与 rl_games 配置](#sec-4)
  - [4.4 策略分布](#sec-44)
  - [4.5 GAE](#sec-45)
  - [4.6 Value Bootstrap](#sec-46)
  - [4.7 Return 与 Advantage 的构造](#sec-47)
  - [4.8 PPO Actor Loss](#sec-48)
  - [4.9 Critic Loss](#sec-49)
  - [4.10 Entropy Bonus](#sec-410)
  - [4.11 总 Loss](#sec-411)
  - [4.12 优化与梯度处理](#sec-412)
  - [4.13 自适应学习率调度](#sec-413)
  - [4.14 完整训练循环小结](#sec-414)
- [5. 随机化：哪些量被随机化，如何进入训练](#sec-5)
- [6. TensorBoard：当前记录了哪些参数，它们具体表示什么](#sec-6)
- [7. 代码路径索引与当前实现边界](#sec-7)
- [8. 一句话总结](#sec-8)

<a id="sec-0"></a>
## 0. 符号、时间尺度与当前默认常数

| 记号 | 含义 | 当前默认值 |
| --- | --- | --- |
| $N$ | 并行环境数 | 训练 profile `env_6144` 下为 $6144$；`DoorPushEnvCfg.scene.num_envs` 的回退默认值为 $64$ |
| $\Delta t_{\text{phys}}$ | 物理积分步长 | `0.008333`，约为 $1/120\,\text{s}$ |
| $d$ | decimation，两个控制步之间包含的物理步数 | $2$ |
| $\Delta t$ | 策略控制步长 | $d\Delta t_{\text{phys}}\approx 1/60\,\text{s}$ |
| $T_{\text{ep}}$ | 单回合最大时长 | $10\,\text{s}$ |
| $H_{\text{ep}}$ | 单回合最大控制步数 | $600$ |
| $H_{\text{rollout}}$ | rl_games rollout horizon | $24$ |
| $d_a$ | 动作维度 | $15$ |
| $d_o$ | actor/critic 观测维度 | $79$ |
| $\gamma$ | 折扣因子 | $0.99$ |
| $\lambda$ | GAE 参数 | $0.95$ |
| $\varepsilon$ | PPO clip 系数 | $0.2$ |
| $c_{\text{critic}}$ | critic loss 权重 | `1.0` |
| $c_{\text{entropy}}$ | entropy bonus 权重 | `0.01` |
| $\sigma$ | 策略标准差（fixed） | 初始 $\exp(-2.0)\approx 0.135$ |

本文中：

- $t$ 表示策略控制步，而不是 PhysX 物理积分子步。
- $i$ 表示第 $i$ 个并行环境。
- $W$ 表示世界坐标系，$B$ 表示机器人 `base_link` 坐标系。
- $\theta_t$ 表示门铰链角度。
- $\dot\theta_t$ 表示门铰链角速度。
- $q_t,\dot q_t\in\mathbb R^{12}$ 表示双臂 12 个受控关节的位置和速度。
- $a_t\in\mathbb R^{15}$ 表示策略输出的 raw action，其中前 12 维控制双臂，后 3 维控制底盘。
- $o_t\in\mathbb R^{79}$ 表示策略和 critic 共用的低维状态观测。

需要特别注意：当前动作空间在 Gym 层定义为

$$
\mathcal A=[-10^6,10^6]^{15},
$$

rl_games 的 `clip_actions` 也设置为 `1000000.0`。因此 policy 输出不是常见的
`[-1,1]` 归一化动作；动作只在映射到关节目标和底盘速度命令时被安全限幅。

<a id="sec-1"></a>
## 1. 任务介绍：项目在训练什么，解决什么问题

### 1.1 任务本质

当前项目训练的是一个双臂移动底盘机器人策略，使其在门外随机初始化后，通过双臂关节位置控制和底盘速度控制，推开一个无把手门板并让 `base_link` 穿过门洞。

用马尔可夫决策过程表示：

$$
\mathcal M=(\mathcal S,\mathcal A,P,r,\rho_0,\gamma).
$$

其中：

- 状态 $s_t\in\mathcal S$ 是 Isaac Lab + PhysX 中的完整物理状态。
- 动作 $a_t\in\mathbb R^{15}$ 是 raw Gaussian action。
- 转移 $P(s_{t+1}\mid s_t,a_t)$ 由仿真器、机器人 articulation、门 articulation、接触和控制器共同决定。
- 奖励 $r_t$ 鼓励开门、穿门推进、减少手臂运动、避免手臂过度伸展、避免 raw action 过大、避免硬碰撞。
- 初始分布 $\rho_0$ 来自 reset-time 基座位姿随机化和物理参数随机化。

策略优化目标仍然是最大化期望折扣回报：

$$
J(\phi)=
\mathbb E_{\tau\sim\pi_\phi}
\left[
\sum_{t=0}^{H_{\text{ep}}-1}\gamma^t r_t
\right].
$$

当前实现没有图像输入，没有门把手操作，没有抓取动作，也没有占用模式条件分支。任务可以概括为：

$$
\text{非视觉、双臂关节目标控制、底盘速度控制、无把手门推开并穿门。}
$$

### 1.2 完整状态的物理组成

从建模上看，完整仿真状态至少包括：

$$
s_t=
\Big(
x_t^{\text{base}},\dot x_t^{\text{base}},
q_t,\dot q_t,
x_t^{\text{door}},\dot x_t^{\text{door}},
\theta_t,\dot\theta_t,
\xi
\Big),
$$

其中 $\xi$ 是每个 episode 的随机化参数：

$$
\xi=
\left(
m_{\text{door}},
\tau_{\text{res}},
c_{\text{air}},
c_{\text{closer}},
p_{\text{base},0},
\psi_{\text{base},0}
\right).
$$

这些参数不会作为独立“模式变量”进入 reward 分支，而是直接改变动力学和观测。
手臂 PD gain 使用固定配置值，不属于 episode 随机化参数。

### 1.3 成功、失败和时间截断

当前实现里的成功判定要求“门已经推开”且“底盘已经越过门洞一定距离”：

$$
\text{success}_t=
\mathbf 1[\theta_t\ge \theta_{\text{open}}]\cdot
\mathbf 1[x_t^D\ge 0.5],
\qquad
\theta_{\text{open}}=\pi/6.
$$

这里 $x_t^D$ 是底盘相对门根部沿穿门方向的 progress：

$$
x_t^D =
\left(p_{\text{base},t}^W-p_{\text{door-root},t}^W\right)^\top e_{\text{cross}}^W,
\qquad
e_{\text{cross}}^W=(-1,0,0).
$$

`door_cross_dir_xy=(-1,0)` 表示从门外指向门内。reset 时机器人在门外侧，因此初始
$x^D$ 通常为负；当 $x^D\ge 0.5$ 时，代码认为底盘已经穿过门洞。

失败事件只有时间耗尽且尚未成功：

$$
\text{fail\_timeout}_t = [t\ge 600]\cdot [\lnot\text{success}_t].
$$

硬碰撞 `hard_collision=True` 和反向开门 $\theta_t < -0.05$ 仍作为 reward / TensorBoard 诊断信号保留，但不再提前结束 episode。

环境的 done 逻辑是：

$$
\text{terminated}_t =
\text{success}_t,
$$

$$
\text{truncated}_t = [t\ge 600].
$$

<a id="sec-2"></a>
## 2. 环境构建：资产、坐标系与仿真初始化

### 2.1 资产构成

每个并行环境包含如下资产：

| 资产 | 当前路径或配置 | 角色 |
| --- | --- | --- |
| 双臂移动机器人 | `assets/robot/usd/uni_dingo_lite.usd`，不存在时回退到 `assets/robot/usd/uni_dingo_dual_arm.usd` | 受控主体 |
| 门 | `assets/minimal_push_door/solid_push_door.usda` | 单铰链无把手门板 |
| 门侧墙 | `assets/minimal_push_door/door_side_walls.usda` | 几何约束与可视化场景 |
| 地面 | `CuboidCfg(size=(100,100,0.1), pos=(0,0,-0.05))` | 接触地面 |
| 环境光 | `DomeLightCfg(intensity=3000)` | 渲染辅助 |
| 硬碰撞传感器 | `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(chassis_link\|left_link.*\|right_link.*)")` | 读取底盘与左右臂 link 接触力 |

由于硬碰撞传感器依赖 PhysX rigid/contact view，当前场景保持 `replicate_physics=True`，但显式设置 `clone_in_fabric=False`。若启用 Fabric clone，ContactSensor 在 `sim.reset()` 初始化时会出现 contact reporter body 数量不匹配，训练会在进入 rl_games 前中断。

门的默认 root pose 为：

$$
p_{\text{door-root},0}^W=(2.93,0,0),\qquad
q_{\text{door-root},0}^W=(1,0,0,0),
$$

门关节 reset 到：

$$
\theta_0=0,\qquad \dot\theta_0=0.
$$

### 2.2 关节和 body 约定

策略直接控制的 12 个手臂关节为：

$$
\begin{aligned}
q=&[
\texttt{left\_joint1},\ldots,\texttt{left\_joint6},\\
&\texttt{right\_joint1},\ldots,\texttt{right\_joint6}
].
\end{aligned}
$$

双侧 gripper 不是策略动作维度。当前实现把
`left_jointGripper` 和 `right_jointGripper` 固定在闭合角：

$$
q_{\text{gripper}}=\operatorname{rad}(-32^\circ).
$$

底盘训练默认使用 planar joint velocity backend，对应：

$$
[
\texttt{base\_x\_joint},
\texttt{base\_y\_joint},
\texttt{base\_yaw\_joint}
].
$$

硬碰撞监测 body 是 ContactSensor body 列表中满足下列条件的 body：

$$
\text{name}=\texttt{chassis\_link}
\quad\text{or}\quad
\text{name starts with }\texttt{left\_link}
\quad\text{or}\quad
\text{name starts with }\texttt{right\_link}.
$$

轮子、gripper 和其他未匹配 body 不在当前硬碰撞 mask 中。

### 2.3 门坐标系与几何量

当前代码中固定使用：

$$
p_{\text{door-center}}^{xy}=(2.95,0),\qquad
e_{\text{cross}}^{xy}=(-1,0),\qquad
e_{\text{lat}}^{xy}=(0,1).
$$

DoorLeaf 局部几何常数为：

$$
d_{\text{center}}^{\text{leaf}}=(0.02,0.45,1.0),
\qquad
n_{\text{door}}^{\text{leaf}}=(1,0,0).
$$

门洞中心在代码里用 door root 加固定 offset 表示：

$$
p_{\text{doorway-center}}^W
=p_{\text{door-root}}^W+(0.02,0.45,1.0).
$$

世界系向量到 `base_link` 系的变换为：

$$
v^B=(R_{WB})^\top v^W.
$$

世界系位置 $p^W$ 相对 base 的表达为：

$$
p^B=(R_{WB})^\top(p^W-p_{\text{base}}^W).
$$

这套变换用于 observation 中的门洞中心、穿门方向、横向方向、铰链点、门板法向和门板中心。

<a id="sec-3"></a>
## 3. Action、Observation、Reward

### 3.1 15 维 raw action

策略输出：

$$
a_t=[a_t^{\text{arm}},a_t^{\text{base}}],
\qquad
a_t^{\text{arm}}\in\mathbb R^{12},\quad
a_t^{\text{base}}\in\mathbb R^3.
$$

#### 双臂动作映射

每个手臂关节的默认姿态为：

$$
q_{\text{default}}=
(0,0,0,0,0,\pi/2,0,0,0,0,0,\pi/2).
$$

先得到 raw target：

$$
q_{\text{raw}}=
q_{\text{default}}+0.25\,a_t^{\text{arm}}.
$$

然后用 torque-proxy 模型把目标夹在当前关节位置附近：

$$
\Delta q_i^{\max}
=
\sigma \frac{\tau_i^{\max}}{K_{p,i}},
\qquad
\sigma=0.7.
$$

最终写入 articulation 的目标为：

$$
q_{\text{target},i}
=
\operatorname{clip}
\left(
q_{\text{raw},i},
q_{t,i}-\Delta q_i^{\max},
q_{t,i}+\Delta q_i^{\max}
\right).
$$

当前 effort limits 为：

$$
\tau^{\max}=
(30,60,30,30,30,30,30,60,30,30,30,30).
$$

$K_p$ 和 $K_d$ 不做 episode 随机化。当前固定配置为
$K_p=50.0$、$K_d=4.5$，其中 $K_p$ 同时进入上面的 torque-proxy action clamp。

#### 底盘动作映射

底盘 raw action 先按最大速度缩放：

$$
u_{\text{raw}} =
a_t^{\text{base}}\odot(0.5,0.5,1.0).
$$

然后只对物理命令做限幅：

$$
u =
\operatorname{clip}
\left(
u_{\text{raw}},
-(0.5,0.5,1.0),
(0.5,0.5,1.0)
\right).
$$

最后应用 deadband：

$$
u_j \leftarrow
\begin{cases}
0, & |u_j|<0.1,\\
u_j, & \text{otherwise}.
\end{cases}
$$

训练时 `scripts/train.py` 调用 `_apply_training_env_simplifications`，把底盘 backend 固定为：

$$
\texttt{base\_control\_backend}=\texttt{planar\_joint\_velocity}.
$$

因此 body-frame 命令 $u=(v_x^B,v_y^B,\omega_z)$ 会被旋转成 world-frame planar joint 速度：

$$
\dot x^W=\cos\psi\,v_x^B-\sin\psi\,v_y^B,
$$

$$
\dot y^W=\sin\psi\,v_x^B+\cos\psi\,v_y^B,
$$

$$
\dot\psi=\omega_z.
$$

代码中还保留 `root_force_torque`、`isaac_holonomic_controller` 和
`analytic_mecanum_fallback` 后端，但默认训练路径不使用它们。

### 3.2 79 维对称 observation

环境返回：

```python
{"policy": obs, "critic": obs}
```

也就是说 actor 和 critic 使用完全相同的 79 维观测，没有特权 critic 额外状态。

当前 observation 拼接顺序严格如下：

| 段 | 维度 | 含义 |
| --- | ---: | --- |
| `base_twist` | 3 | base frame 下的 $(v_x,v_y,\omega_z)$ |
| `arm_q` | 12 | 12 个手臂关节位置 |
| `arm_qd` | 12 | 12 个手臂关节速度 |
| `left_ee_pose_base` | 7 | 左臂末端相对 base 的位置和四元数姿态 $(p^B,q^B)$ |
| `right_ee_pose_base` | 7 | 右臂末端相对 base 的位置和四元数姿态 $(p^B,q^B)$ |
| `prev_action` | 15 | 上一步 raw action |
| `doorway_center_base` | 3 | 门洞中心相对 base 的坐标 |
| `cross_dir_base` | 3 | 穿门方向 $e_{\text{cross}}$ 在 base frame 下的表达 |
| `lat_dir_base` | 3 | 门横向方向 $e_{\text{lat}}$ 在 base frame 下的表达 |
| `hinge_point_base` | 3 | door root 相对 base 的坐标 |
| `door_normal_base` | 3 | 门板法向在 base frame 下的表达 |
| `door_center_base` | 3 | 门板中心相对 base 的坐标 |
| `theta` | 1 | 门铰链角 |
| `theta_dot` | 1 | 门铰链角速度 |
| `door_mass` | 1 | 当前 episode 门板质量 |
| `door_hinge_dyn_torque` | 1 | 当前门铰链动态阻力幅值 |
| `stage` | 1 | 是否进入 passing stage |
| **总计** | **79** |  |

其中：

$$
\text{stage}_t=\mathbf 1[\theta_t>\theta_{\text{pass}}],
\qquad
\theta_{\text{pass}}=70^\circ.
$$

注意这里是严格大于号，不是大于等于。

`door_hinge_dyn_torque` 是一个非负幅值：

$$
\tau_{\text{dyn}}
=
\tau_{\text{res}}
+c_{\text{air}}|\dot\theta|^2
+c_{\text{closer}}|\dot\theta|.
$$

与此同时，环境每步写入门铰链 effort target：

$$
\tau_{\text{effort}}
=
-
\left(
\tau_{\text{res}}\operatorname{sign}(\dot\theta)
+c_{\text{air}}\dot\theta^2\operatorname{sign}(\dot\theta)
\right).
$$

线性 closer damping 没有进入这个 effort target；当前实现把
$c_{\text{air}}+c_{\text{closer}}$ 写入门关节 damping 参数。

### 3.3 Reward 总体结构

当前 reward 分成三组：

$$
r_t=
\begin{cases}
r_o+r_s, & \theta_t\le \theta_{\text{pass}},\\
3.0+r_p+r_s, & \theta_t>\theta_{\text{pass}}.
\end{cases}
$$

这里 $3.0$ 是 opening reward 的最大尺度 `rew_opening_scale`。

### 3.4 Opening reward

开门目标角为：

$$
\hat\theta=75^\circ.
$$

无量纲开门奖励为：

$$
r_{\text{od}}
=
1-\frac{|\theta_t-\hat\theta|}{\hat\theta}.
$$

实际 opening reward 为：

$$
r_o=3.0\,r_{\text{od}}.
$$

这个项在 passing stage 之前直接进入总奖励；进入 passing stage 后，总奖励使用常数 $3.0$ 代替即时 $r_o$。

### 3.5 Passing reward

代码先计算底盘穿门进度：

$$
x_t^D =
\left(p_{\text{base},t}^W-p_{\text{door-root},t}^W\right)^\top e_{\text{cross}}^W.
$$

若 $x_t^D<0$，说明底盘仍在门外侧，progress 方向指向门洞中心：

$$
d_t^W=
\frac{
p_{\text{doorway-center},t}^W-p_{\text{base},t}^W
}{
\|p_{\text{doorway-center},t}^W-p_{\text{base},t}^W\|_2
}.
$$

若 $x_t^D\ge 0$，progress 方向直接取穿门方向：

$$
d_t^W=e_{\text{cross}}^W.
$$

把 $d_t^W$ 转到 base frame 得到 $d_t^B$，再与 base frame 线速度点乘：

$$
r_p=
\operatorname{clip}
\left(
\frac{(d_t^B)^\top v_{\text{base},t}^B}{0.5},
0,
1
\right).
$$

这里的 passing reward 使用速度投影，不使用相邻两步 signed distance 差分。

### 3.6 Shaping reward

shaping reward 为：

$$
r_s=
0.3\,r_{\text{ma}}
+1.0\,r_{\text{psa}}
+1.0\,r_{\text{eep}}
+0.1\,r_{\text{pcl}}
+2.0\,r_{\text{pc}}.
$$

#### 最小手臂运动项

$$
r_{\text{ma}}
=
\sum_{i=1}^{12}
\left[
\exp(-0.01\dot q_i^2)
+
\exp(-10^{-6}\ddot q_i^2)
\right],
$$

其中：

$$
\ddot q_t=\frac{\dot q_t-\dot q_{t-1}}{\Delta t}.
$$

#### 手臂过伸惩罚

当前实现把左右臂的 shoulder anchor 明确绑定为左右 `joint1` 对应的 link，
计算双末端在 base frame 下到各自 anchor 的距离：

$$
r_{\text{psa}}
=
-
\sum_{j\in\{L,R\}}
\operatorname{clip}
\left(
\frac{\|p_{\text{ee},j}^B-p_{\text{shoulder},j}^B\|_2-0.5}{0.1},
0,
1
\right).
$$

其中：

$$
p_{\text{shoulder},L}^B=\texttt{left\_link00 在 base frame 下的位置},
\qquad
p_{\text{shoulder},R}^B=\texttt{right\_link00 在 base frame 下的位置}.
$$

#### 末端到门板靠近奖励

仿照论文中的 end-effector-to-handle/door 接近项，当前无把手任务使用双臂末端到门板中心的更近距离：

$$
r_{\text{eep}}
=
\exp
\left(
-\min
\left(
\|p_{\text{ee},L}^B-p_{\text{door-center}}^B\|_2,\;
\|p_{\text{ee},R}^B-p_{\text{door-center}}^B\|_2
\right)
\right).
$$

该项鼓励至少一只手靠近门板；距离在 base frame 中计算。

#### raw command limit 惩罚

该项作用在上一步 raw action 上：

$$
r_{\text{pcl}}
=
-
\sum_{k=1}^{15}
\operatorname{clip}
\left(
\frac{|a_{t-1,k}|-1.0}{1.0},
0,
1
\right).
$$

因为当前 action 没有被 rl_games 裁到 `[-1,1]`，这个惩罚是限制 raw policy 输出幅值的主要 reward 约束。

#### 硬碰撞惩罚

ContactSensor 给出机器人各 body 的 net contact force：

$$
F_{b,t}=\|\text{net\_force}_{b,t}^W\|_2.
$$

若被监测 body 中存在：

$$
F_{b,t}>1.0,
$$

则：

$$
\text{hard\_collision}_t=1,
\qquad
r_{\text{pc}}=-1.
$$

否则：

$$
r_{\text{pc}}=0.
$$

若 contact sensor 缺失、尚未初始化，或 sensor body 数量与缓存 mask 不一致，当前实现返回全 false，不触发硬碰撞。

### 3.7 Episode reward extras

每个环境在 episode 内累计下列 reward key：

| extras key | 含义 |
| --- | --- |
| `opening` | $r_o$ 的 episode 累计 |
| `opening/open_door_target` | $r_{\text{od}}$ 的 episode 累计 |
| `passing` | $r_p$ 的 episode 累计 |
| `shaping` | $r_s$ 的 episode 累计 |
| `shaping/min_arm_motion` | $r_{\text{ma}}$ 的 episode 累计 |
| `shaping/stretched_arm` | $r_{\text{psa}}$ 的 episode 累计 |
| `shaping/end_effector_to_panel` | $r_{\text{eep}}$ 的 episode 累计 |
| `shaping/command_limit` | $r_{\text{pcl}}$ 的 episode 累计 |
| `shaping/collision` | $r_{\text{pc}}$ 的 episode 累计 |
| `total` | $r_t$ 的 episode 累计 |
| `_step_count` | episode 完成时的控制步数 |

这些值只在 done env 上写入 `extras["episode_reward_info"]`，随后由自定义 rl_games observer 汇总。

<a id="sec-4"></a>
## 4. Policy 是如何训练的：PPO 与 rl_games 配置

### 4.1 训练入口

当前训练入口是：

```bash
python scripts/train.py
```

默认 task name 为：

```text
Affordance-DoorPush-Direct-v0
```

task registry 位于：

```text
src/affordance_guided_interaction/tasks/door_push_direct/__init__.py
```

它注册：

- env entry point：`affordance_guided_interaction.envs.door_push_env:DoorPushEnv`
- env cfg：`affordance_guided_interaction.envs.door_push_env_cfg:DoorPushEnvCfg`
- rl_games cfg：`src/affordance_guided_interaction/tasks/door_push_direct/agents/rl_games_ppo_cfg.yaml`

训练脚本的主链路为：

```text
YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher
-> gym.make -> RlGamesVecEnvWrapper -> rl_games.Runner
```

训练时会把实际合并后的配置写到：

```text
runs/rl_games/door_push_direct/<run_name>/params/
```

包括：

- `env.yaml`
- `agent.yaml`
- `project.yaml`

### 4.2 默认训练 profile

`configs/training/default.yaml` 的默认合并 profile 是 `env_6144`：

| 参数 | 当前默认值 |
| --- | --- |
| `num_envs` | `6144` |
| `total_steps` | `3000_000_000` |
| `n_steps_per_rollout` | `24` |
| `checkpoint_interval` | `20` |
| `num_mini_batches` | `4` |
| `num_epochs` | `5` |
| `headless` | `true` |
| `device` | `cuda:1` |
| `seed` | `42` |
| `log_dir` | `runs` |

有效 batch size 为：

$$
B=N\cdot H_{\text{rollout}}=6144\times 24=147456.
$$

mini-batch size 由训练脚本计算：

$$
B_{\text{mini}}=\frac{B}{4}=36864.
$$

最大 epoch 数由训练脚本根据 total steps 重新计算：

$$
\text{max\_epochs}
=
\left\lceil
\frac{3{,}000{,}000{,}000}{147456}
\right\rceil
=20346.
$$

因此，虽然默认 agent YAML 里写有 `max_epochs: 4578`，实际训练路径会被 profile 的
`total_steps` 覆盖为上式结果。

### 4.3 rl_games agent 配置

当前 agent 类型为：

| 字段 | 当前值 |
| --- | --- |
| `algo.name` | `a2c_continuous` |
| `model.name` | `continuous_a2c_logstd` |
| `network.name` | `actor_critic` |
| `network.separate` | `False` |
| MLP | `[512, 256, 128]` |
| activation | `elu` |
| `fixed_sigma` | `True` |
| `sigma_init.val` | `-2.0` |
| `normalize_input` | `True` |
| `normalize_value` | `True` |
| `mixed_precision` | `True` |

PPO 主要超参数为：

| 参数 | 当前值 |
| --- | --- |
| `gamma` | `0.99` |
| `tau` / GAE $\lambda$ | `0.95` |
| `learning_rate` | `1.0e-3` |
| `lr_schedule` | `adaptive` |
| `kl_threshold` | `0.01` |
| `adaptive_lr_min` | `1.0e-5` |
| `adaptive_lr_max` | `1.0e-2` |
| `e_clip` | `0.2` |
| `entropy_coef` | `0.01` |
| `critic_coef` | `1.0` |
| `bounds_loss_coef` | `0.0` |
| `grad_norm` | `1.0` |
| `clip_value` | `True` |
| `horizon_length` | `24` |
| `mini_epochs` | `5` |
| `seq_length` | `4` |

训练脚本要求 `actor_lr == critic_lr`，因为当前 rl_games-only 路径使用单优化器；默认二者均为 `1.0e-3`。`bounds_loss_coef=0.0` 显式禁用 rl_games 的 action mean bounds loss；该键不能省略，否则当前 rl_games 版本会让 `bound_loss()` 返回 Python `int`，在 PPO 更新阶段触发 `b_loss.unsqueeze(...)` 类型错误。

### 4.4 策略分布

当前模型为 `continuous_a2c_logstd`，即独立对角高斯策略。网络输出均值 $\mu_\phi(o_t)\in\mathbb R^{15}$，标准差由参数化 $\log\sigma$ 给出（`fixed_sigma=True`，$\sigma$ 不依赖观测）。

初始化时 $\log\sigma = -2.0$，对应 $\sigma\approx 0.135$。

策略分布为：

$$
\pi_\phi(a\mid o)=\prod_{k=1}^{15}\mathcal N(a_k;\,\mu_{\phi,k}(o),\,\sigma_k^2).
$$

给定动作 $a$ 和分布参数 $(\mu,\sigma)$，负对数概率的精确实现为（`ModelA2CContinuousLogStd.Network.neglogp`）：

$$
-\log\pi_\phi(a\mid o)=
\frac{1}{2}\sum_{k=1}^{15}\left(\frac{a_k-\mu_k}{\sigma_k}\right)^2
+\frac{1}{2}\log(2\pi)\cdot 15
+\sum_{k=1}^{15}\log\sigma_k.
$$

对应的微分熵为：

$$
\mathcal H[\pi_\phi(\cdot\mid o)]=
\frac{1}{2}\sum_{k=1}^{15}\left(1+\log(2\pi\sigma_k^2)\right).
$$

训练时 rollout 阶段使用 `distr.sample()` 采样动作，并记录 $-\log\pi_{\phi_{\text{old}}}(a_t\mid o_t)$；更新阶段用相同动作重新计算 $-\log\pi_\phi(a_t\mid o_t)$。

### 4.5 GAE（Generalized Advantage Estimation）

优势估计使用逆时序递推（`a2c_common.A2CBase.discount_values`）。设 rollout 长度为 $H=H_{\text{rollout}}=24$。

对 $t=H-1,\ldots,0$ 逆序计算：

$$
\text{next\_terminal}_t=
\begin{cases}
1-d_{\text{done}}, & t=H-1,\\
1-d_{t+1}, & \text{otherwise},
\end{cases}
$$

$$
\text{next\_value}_t=
\begin{cases}
V_\phi(o_{\text{last}}), & t=H-1,\\
V_\phi(o_{t+1}), & \text{otherwise}.
\end{cases}
$$

TD 残差：

$$
\delta_t=r_t+\gamma\cdot\text{next\_value}_t\cdot\text{next\_terminal}_t-V_\phi(o_t).
$$

GAE 递推（$\hat\Lambda_{-1}=0$）：

$$
\hat\Lambda_t=\delta_t+\gamma\lambda\cdot\text{next\_terminal}_t\cdot\hat\Lambda_{t-1}.
$$

这里的 $d_t$ 是 step $t$ 后环境的 done 标志。`next_terminal` 确保 terminated 环境不bootstrap后续值。

### 4.6 Value Bootstrap（时间截断修正）

当 episode 因时间截断而 done（非真正的 terminated），rl_games 通过 `value_bootstrap` 修正奖励（`play_steps` 中）：

$$
\tilde r_t = r_t + \gamma\cdot V_\phi(o_t)\cdot\mathbb 1[\text{time\_out}_t].
$$

其中 `time_outs` 来自环境 `infos`，标识哪些环境因超时而被截断。这个修正确保截断回合的最后一个 reward 包含了 value continuation，避免人为低估值。

当前配置下 `value_bootstrap=True`（rl_games 默认）。

### 4.7 Return 与 Advantage 的构造

**Returns**（`play_steps`）：

$$
\hat R_t = \hat\Lambda_t + V_\phi(o_t).
$$

即 GAE 优势加上当前 value 估计。

**Raw Advantage**（`prepare_dataset`）：

$$
\hat A_t^{\text{raw}} = \hat R_t - V_\phi(o_t).
$$

这在数值上等于 $\hat\Lambda_t$，但代码从 returns 和 values 显式相减得到。

**Value 归一化**（`normalize_value=True` 时）：对 returns 和 old_values 都做 running mean/std 归一化：

$$
V_{\text{norm}} = \frac{V - \mu_V}{\sigma_V+\epsilon},
\qquad
\hat R_{\text{norm}} = \frac{\hat R - \mu_R}{\sigma_R+\epsilon}.
$$

Value 和 Return 共享同一个 `RunningMeanStd` 实例。

**Advantage 降维**（`value_size=1` 时）：

$$
\hat A_t = \sum_{j}\hat A_{t,j}^{\text{raw}}.
$$

**Advantage 归一化**（`normalize_advantage=True` 时）：

$$
\hat A_t \leftarrow \frac{\hat A_t - \bar{\hat A}}{\operatorname{std}(\hat A)+10^{-8}}.
$$

归一化在整个 flattened batch（$N\cdot H=147456$ 个样本）上进行。

### 4.8 PPO Actor Loss（Clipped Surrogate Objective）

概率比（`common_losses.actor_loss`）：

$$
\rho_t(\phi)=
\frac{\pi_\phi(a_t\mid o_t)}{\pi_{\phi_{\text{old}}}(a_t\mid o_t)}
=\exp\big(-\log\pi_{\phi_{\text{old}}}(a_t\mid o_t)+\log\pi_\phi(a_t\mid o_t)\big).
$$

代码实现为 `ratio = exp(old_neglogp - new_neglogp)`。

Clipped surrogate（注意代码取 max of negative，等效于 min of positive）：

$$
L_t^{\text{actor}}(\phi)=
\max\Big(
-\rho_t(\phi)\hat A_t,\;
-\operatorname{clip}\big(\rho_t(\phi),\;1-\varepsilon,\;1+\varepsilon\big)\hat A_t
\Big).
$$

等效于标准形式的：

$$
L_t^{\text{actor}}(\phi)=
-\min\Big(
\rho_t(\phi)\hat A_t,\;
\operatorname{clip}\big(\rho_t(\phi),\;1-\varepsilon,\;1+\varepsilon\big)\hat A_t
\Big).
$$

其中 $\varepsilon=\texttt{e\_clip}=0.2$。

当前配置 `use_smooth_clamp=False`，使用硬 clamp（`torch.clamp`）；若设为 `True` 则使用 sigmoid 平滑 clamp。

### 4.9 Critic Loss（Clipped Value Loss）

当前配置 `clip_value=True`，使用 clipped value loss（`common_losses.default_critic_loss`）。

定义 value 偏移量：

$$
\Delta V_t = V_\phi(o_t) - V_{\phi_{\text{old}}}(o_t).
$$

Clipped value 预测：

$$
V_t^{\text{clip}}=V_{\phi_{\text{old}}}(o_t)+\operatorname{clip}\big(\Delta V_t,\;-\varepsilon,\;\varepsilon\big).
$$

Critic loss 取 unclipped 和 clipped 的最大值：

$$
L_t^{\text{critic}}(\phi)=
\max\Big(
\big(V_\phi(o_t)-\hat R_t\big)^2,\;
\big(V_t^{\text{clip}}-\hat R_t\big)^2
\Big).
$$

此处 $\hat R_t$ 和 $V_{\phi_{\text{old}}}(o_t)$ 都已过 value 归一化（若 `normalize_value=True`）。$\varepsilon$ 与 actor 共享 `e_clip=0.2`。

### 4.10 Entropy Bonus

Entropy 从模型的 forward pass 直接获取（`distr.entropy().sum(dim=-1)`），即 15 个独立高斯分量熵之和：

$$
\mathcal H_t=\sum_{k=1}^{15}\frac{1}{2}\big(1+\log(2\pi\sigma_{\phi,k}^2)\big).
$$

在总 loss 中以 **负号** 加入（即鼓励高熵）：

$$
L_t^{\text{entropy}} = -\mathcal H_t.
$$

系数 `entropy_coef=0.01`。

### 4.11 总 Loss

所有子项在 `calc_losses` 中先经过 `apply_masks`（非 RNN 时 mask 为全 1），做 sum/mean 后组合：

$$
L_t^{\text{total}}(\phi)=
L_t^{\text{actor}}
+0.5\cdot c_{\text{critic}}\cdot L_t^{\text{critic}}
-c_{\text{entropy}}\cdot\mathcal H_t.
$$

其中各系数当前值为：

| 系数 | 符号 | 当前值 |
| --- | --- | --- |
| critic 系数 | $c_{\text{critic}}$ | `1.0` |
| entropy 系数 | $c_{\text{entropy}}$ | `0.01` |

注意代码中 critic loss 前有一个额外的 $0.5$ 因子（`0.5 * c_loss * self.critic_coef`），这是 rl_games 的惯例。因此 critic loss 的实际权重为 $0.5\times 1.0=0.5$。

此外，若模型有 auxiliary loss（如 `get_aux_loss()` 非零），则额外加入：

$$
L^{\text{total}} \leftarrow L^{\text{total}} + L^{\text{aux}}.
$$

当前 `actor_critic` 网络的 `get_aux_loss()` 返回 `None`，所以 aux loss 不生效。

### 4.12 优化与梯度处理

**优化器**：Adam，$\text{lr}=1\times 10^{-3}$，$\epsilon=10^{-8}$，无 weight decay。

**Mixed Precision**：`mixed_precision=True`，使用 `torch.amp.autocast('cuda', dtype=torch.bfloat16)`。前向和 loss 计算在 bfloat16 下执行，梯度通过 `GradScaler` 缩放。

**梯度裁剪**：`truncate_grads=True`，在 scaler unscale 之后执行：

$$
\|\nabla_\phi L\|_2 \leftarrow \operatorname{clip}\big(\|\nabla_\phi L\|_2,\;\texttt{grad\_norm}=1.0\big).
$$

即如果全局梯度范数超过 1.0，则等比缩放所有梯度使范数恰为 1.0。

**Mini-batch 训练**：每个 epoch 的 rollout（$B=147456$）被分成 `num_mini_batches` 个 mini-batch。当前 `minibatch_size=4096`（配置文件）但训练脚本会根据 `num_mini_batches=4` 覆盖计算 `minibatch_size = B / 4 = 36864`。每个 mini-batch 做一次前向 + 反向 + 梯度更新。

**Mini-epochs**：每次 rollout 后重复 `mini_epochs=5` 轮更新。

### 4.13 自适应学习率调度

当前使用 `adaptive` 学习率调度（`AdaptiveScheduler`），基于 mini-epoch 内新旧策略的 KL 散度。

**KL 散度计算**（`torch_ext.policy_kl`）：

$$
D_{\text{KL}}\big(\pi_{\phi_{\text{old}}}\|\pi_\phi\big)=
\frac{1}{B_{\text{mini}}}\sum_{i=1}^{B_{\text{mini}}}
\sum_{k=1}^{15}
\left[
\log\frac{\sigma_{\text{new},k}}{\sigma_{\text{old},k}}
+\frac{\sigma_{\text{old},k}^2+(\mu_{\text{old},k}-\mu_{\text{new},k})^2}{2\sigma_{\text{new},k}^2}
-\frac{1}{2}
\right].
$$

当前配置 `schedule_type='legacy'`，即 KL 在每个 mini-batch 后即时计算并更新学习率。

**自适应规则**（`kl_threshold=0.01`，`adaptive_lr_min=1e-5`，`adaptive_lr_max=1e-2`）：

$$
\text{lr} \leftarrow
\begin{cases}
\max\big(\text{lr}/1.5,\;10^{-5}\big), & D_{\text{KL}} > 2\times 0.01 = 0.02,\\
\min\big(\text{lr}\times 1.5,\;10^{-2}\big), & D_{\text{KL}} < 0.5\times 0.01 = 0.005,\\
\text{lr}, & \text{otherwise}.
\end{cases}
$$

### 4.14 完整训练循环小结

一个训练 epoch 的完整流程：

1. **Rollout**：用当前策略 $\pi_\phi$ 在 $N=6144$ 个并行环境中收集 $H=24$ 步数据，记录 $(o_t, a_t, r_t, d_t, -\log\pi_\phi(a_t|o_t), V_\phi(o_t), \mu_t, \sigma_t)$。
2. **GAE**：逆序计算 $\hat\Lambda_t$，得到 returns $\hat R_t=\hat\Lambda_t+V_\phi(o_t)$。
3. **Dataset 准备**：计算 advantage $\hat A_t=\hat R_t-V_\phi(o_t)$，归一化 value/return，归一化 advantage。
4. **Mini-epoch 循环**（5 轮）：
   - 遍历 mini-batch：
     - 前向计算新的 $-\log\pi_\phi$、$V_\phi$、$\mathcal H$、$\mu$、$\sigma$。
     - 计算 $L^{\text{total}}=L^{\text{actor}}+0.5\cdot L^{\text{critic}}-0.01\cdot\mathcal H$。
     - Backward，梯度裁剪 $\|\nabla\|\le 1.0$，Adam 更新。
     - 计算 KL 散度，自适应调整学习率。

<a id="sec-5"></a>
## 5. 随机化：哪些量被随机化，如何进入训练

### 5.1 reset 顺序

每个 episode reset 时，`DoorPushEnv._reset_idx` 按如下顺序执行：

1. 采样 domain randomization。
2. 采样 base pose。
3. 写 robot root state。
4. 写 robot joint state，包括双臂默认姿态、planar base pose 和初始速度。
5. 固定 gripper closed target。
6. 把 door joint state 重置为 0。
7. 写 arm target 和 planar base velocity target；只有在 `training_planar_base_only=False` 的非默认路径下才额外写 wheel velocity target。
8. `scene.write_data_to_sim()`。
9. 把物理参数（门质量、门 joint damping）以及固定手臂 PD stiffness/damping 写入 PhysX / articulation。
10. 清空 per-env step、prev action、prev arm velocity、prev door angle 和 episode reward 累计，重置 wrench composer。

### 5.2 基座位姿随机化

当前基座采样函数为 `sample_base_poses_in_door_frame`。设：

$$
d\sim\mathcal U(1.0,2.0),
\qquad
y\sim\mathcal U(-2.0,2.0),
$$

则：

$$
p_{\text{base},0}^{xy}
=
p_{\text{door-center}}^{xy}
-d\,e_{\text{cross}}^{xy}
+y\,e_{\text{lat}}^{xy}.
$$

因为 $e_{\text{cross}}=(-1,0)$，所以 reset 时底盘位于门外侧，即 $x$ 坐标通常大于门中心。

base 高度固定为：

$$
z_{\text{base},0}=0.014855.
$$

yaw 采样为：

$$
\psi_0=
\operatorname{atan2}(e_{\text{cross},y},e_{\text{cross},x})
+\eta,
\qquad
\eta\sim\mathcal U(-\pi,\pi).
$$

由于 $e_{\text{cross}}=(-1,0)$，基础朝向角为 $\pi$，再叠加完整 $[-\pi,\pi]$ 随机扰动。

初始 planar base velocity 为：

$$
\dot x_0,\dot y_0\sim\mathcal U(-0.5,0.5),
\qquad
\dot\psi_0=0.
$$

双臂和轮关节初始速度均写为 0。

### 5.3 门动力学随机化

门板质量：

$$
m_{\text{door}}\sim\mathcal U(15,75).
$$

铰链静态阻力：

$$
\tau_{\text{res}}\sim\mathcal U(0,30),
$$

并以概率 $0.2$ 被置零。

空气阻尼：

$$
c_{\text{air}}\sim\mathcal U(0,4).
$$

closer damping 先按阻力成比例采样：

$$
\alpha\sim\mathcal U(1.5,3.0),
\qquad
c_{\text{closer}}=\alpha\,\tau_{\text{res}},
$$

再以概率 $0.4$ 被置零。

写入仿真时：

- 门质量通过 `door.root_physx_view.get_masses()` / `set_masses()` 写到 `DoorLeaf` body。
- 门 joint damping 写为：

$$
c_{\text{joint}}=c_{\text{air}}+c_{\text{closer}}.
$$

### 5.4 手臂 PD 固定值

手臂 PD gain 不参与 domain randomization。每个 episode、每个环境、每个受控手臂关节都使用固定值：

$$
K_{p,i}=50.0,\qquad
K_{d,i}=4.5.
$$

若当前 Isaac Lab articulation 支持 per-env stiffness/damping 写入，则 reset 后会通过
`write_joint_stiffness_to_sim` 和 `write_joint_damping_to_sim` 把这些固定值写入对应手臂关节。
其中 $K_p$ 同时影响 torque-proxy action clamp：

$$
\Delta q_i^{\max}=0.7\frac{\tau_i^{\max}}{K_{p,i}}.
$$

<a id="sec-6"></a>
## 6. TensorBoard：当前记录了哪些参数，它们具体表示什么

### 6.1 聚合规则

自定义 observer 为 `DoorPushTensorboardObserver`。它在 `process_infos` 中只处理 done env，并在 `after_print_stats` 中把缓存统计写入 TensorBoard。

对 reward 类 tag，聚合规则为：

$$
\bar R_k
=
\frac{1}{|\mathcal K|}
\sum_{e\in\mathcal K}
R_{k,e},
$$

其中 $\mathcal K$ 是当前打印周期内完成 episode 的集合，$R_{k,e}$ 是某个 episode 内对应 reward key 的累计值。

### 6.2 reward tags

| TensorBoard tag | 来源 extras key | 含义 |
| --- | --- | --- |
| `reward/total` | `total` | episode 总 reward 累计均值 |
| `reward/opening` | `opening` | opening reward 累计均值 |
| `reward/open_door_target` | `opening/open_door_target` | 未乘 scale 的开门目标项累计均值 |
| `reward/passing` | `passing` | passing reward 累计均值 |
| `reward/shaping` | `shaping` | shaping reward 累计均值 |
| `reward/min_arm_motion` | `shaping/min_arm_motion` | 最小手臂运动项累计均值 |
| `reward/penalize_stretched_arm` | `shaping/stretched_arm` | 手臂过伸惩罚累计均值 |
| `reward/end_effector_to_panel` | `shaping/end_effector_to_panel` | 末端到门板靠近奖励累计均值 |
| `reward/penalize_command_limit` | `shaping/command_limit` | raw action 过大惩罚累计均值 |
| `reward/penalize_collision` | `shaping/collision` | 硬碰撞惩罚累计均值 |

### 6.3 success 和 episode tags

| TensorBoard tag | 含义 |
| --- | --- |
| `success/rate` | done episodes 中 `success=True` 的比例 |
| `success/opened_enough_rate` | done episodes 中 $\theta\ge\pi/6$ 的比例 |
| `success/passed_through_rate` | done episodes 中 $x^D\ge0.5$ 的比例 |
| `success/no_collision_rate` | done episodes 中没有硬碰撞失败的比例 |
| `episode/length` | done episodes 的 `_step_count` 均值 |

这里的 `success/rate` 同时要求 `opened_enough=True` 和 `passed_through=True`。

### 6.4 task state tags

| TensorBoard tag | 含义 |
| --- | --- |
| `task/door_angle_mean` | done env 的门角均值 |
| `task/door_angle_final` | 当前实现与 `task/door_angle_mean` 相同，都是 done env 的门角均值 |
| `task/base_cross_progress` | done env 的 $x^D$ 均值 |
| `task/door_angular_velocity` | done env 的 $\dot\theta$ 均值 |
| `task/progress_reward` | episode passing reward 累计均值 |
| `task/open_stage_rate` | done env 中 `stage<=0.5` 的比例 |
| `task/passing_stage_rate` | done env 中 `stage>0.5` 的比例 |
| `task/base_lateral_error` | $(p_{\text{base}}^W-p_{\text{door-root}}^W)^\top e_{\text{lat}}^W$ 的 done-env 均值 |
| `task/base_heading_error` | base yaw 与 `atan2(e_cross_y,e_cross_x)` 的 wrapped absolute error 均值 |
| `task/hard_collision_rate` | done env 最后一帧中硬碰撞诊断比例 |
| `task/reverse_open_rate` | done env 最后一帧中反向开门诊断比例 |
| `task/fail_timeout_rate` | done env 中 timeout 且非 success 的失败比例 |

### 6.5 randomization tags

| TensorBoard tag | 含义 |
| --- | --- |
| `random/door_mass` | done env 的门板质量均值 |
| `random/hinge_resistance` | done env 的铰链阻力均值 |
| `random/reset_x` | reset 时采样的 base world x 均值 |
| `random/reset_y` | reset 时采样的 base world y 均值 |
| `random/reset_yaw` | reset 时采样的 base yaw 均值 |

这些 tag 不是逐步均值，而是 episode 完成时的 done-env 统计。

<a id="sec-7"></a>
## 7. 代码路径索引与当前实现边界

### 7.1 关键代码路径

| 内容 | 当前代码路径 |
| --- | --- |
| Gym task 注册 | `src/affordance_guided_interaction/tasks/door_push_direct/__init__.py` |
| 环境配置 | `src/affordance_guided_interaction/envs/door_push_env_cfg.py` |
| DirectRLEnv 实现 | `src/affordance_guided_interaction/envs/door_push_env.py` |
| reward helper | `src/affordance_guided_interaction/envs/door_reward_math.py` |
| base control helper | `src/affordance_guided_interaction/envs/base_control_math.py` |
| joint target helper | `src/affordance_guided_interaction/envs/joint_target_math.py` |
| batch 坐标和 reset helper | `src/affordance_guided_interaction/envs/batch_math.py` |
| 训练入口 | `scripts/train.py` |
| 训练 profile | `configs/training/default.yaml` |
| 环境 YAML | `configs/env/default.yaml` |
| 任务阈值 YAML | `configs/task/default.yaml` |
| reward YAML | `configs/reward/default.yaml` |
| rl_games agent YAML | `src/affordance_guided_interaction/tasks/door_push_direct/agents/rl_games_ppo_cfg.yaml` |
| TensorBoard observer | `src/affordance_guided_interaction/utils/rl_games_observer.py` |
| inference 默认配置 | `configs/inference/default.yaml` |

### 7.2 当前实现边界

当前默认训练管线已经不包含下列旧版概念：

- 杯体稳定性任务。
- reset-time occupancy 模式。
- occupancy curriculum。
- 学生策略和 distillation。
- asymmetric 102D/115D actor-critic observation。
- 门把手抓取、handle target、PushPlate 几何。
- reward stage 参数覆盖。

当前仍然保留但默认训练不使用的能力包括：

- `root_force_torque` 底盘控制后端。
- Isaac holonomic controller / analytic mecanum wheel fallback。
- video recording 训练开关。
- inference rollout 配置。

因此，若要判断代码是否符合本文，应以默认训练路径为准：

```text
scripts/train.py
-> build_env_cfg(..., for_training=True)
-> DoorPushEnvCfg
-> DoorPushEnv
-> RlGamesVecEnvWrapper
-> Runner(DoorPushTensorboardObserver)
```

<a id="sec-8"></a>
## 8. 一句话总结

当前项目默认训练的数学本质是：在随机门动力学、固定手臂 PD、随机门外基座位姿和 60 Hz 控制频率下，训练一个输入 79 维低维状态、输出 15 维 raw action 的高斯 PPO 策略；该策略通过 torque-proxy 双臂关节目标和 planar 底盘速度命令推开无把手门，并以“门已推开且底盘穿过门洞”为成功标准，以 timeout 作为唯一失败类别，最大化由开门目标、穿门速度投影、手臂运动约束、过伸惩罚、末端靠近门板奖励、raw command 惩罚和硬碰撞惩罚组成的 episode 回报。
