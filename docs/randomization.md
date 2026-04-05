# 项目中的随机化量与参数

这份文档只保留一件事：说明当前项目里哪些量会被随机化，以及对应的随机化参数是多少。

本文以当前默认实现为准，主要对应：

- `src/affordance_guided_interaction/training/domain_randomizer.py`
- `src/affordance_guided_interaction/training/curriculum_manager.py`
- `configs/curriculum/default.yaml`

---

## 1. 回合级随机化

这些量在每次采样时生成一组参数，并在该组参数生效期间保持不变。

### 1.1 杯体质量

记杯体质量为 `m_cup`，当前采用均匀分布：

`m_cup ~ Uniform(0.1, 0.8)`

- 单位：kg
- 范围：`[0.1, 0.8]`

### 1.2 门板质量

记门板质量为 `m_door`，当前采用均匀分布：

`m_door ~ Uniform(5.0, 20.0)`

- 单位：kg
- 范围：`[5.0, 20.0]`

### 1.3 门铰链阻尼

记门铰链阻尼为 `d_hinge`，当前采用均匀分布：

`d_hinge ~ Uniform(0.5, 5.0)`

- 单位：N·m·s/rad
- 范围：`[0.5, 5.0]`

### 1.4 机器人基座位置

记机器人基座世界坐标为 `p_base ∈ R^3`。当前模型中只对 `XY` 平面做均匀扰动，`Z` 保持标称值不变：

```text
p_base = [p0_x + delta_x, p0_y + delta_y, p0_z]
delta_x, delta_y ~ Uniform(-0.03, 0.03)
```

当前默认标称位置：

`p0 = (0.0, 0.0, 0.0)`

因此默认参数为：

- `base_pos_nominal = (0.0, 0.0, 0.0)`
- `base_pos_delta = 0.03`
- 单位：m

---

## 2. 步级噪声参数

这些量的定义是逐步采样的高斯噪声参数。

### 2.1 动作噪声

记动作噪声为 `epsilon_a`，当前定义为：

`epsilon_a ~ Normal(0, sigma_a^2 I)`

其中：

`sigma_a = 0.02`

即：

- `action_noise_std = 0.02`

### 2.2 观测噪声

记观测噪声为 `epsilon_o`，当前定义为：

`epsilon_o ~ Normal(0, sigma_o^2 I)`

其中：

`sigma_o = 0.01`

即：

- `observation_noise_std = 0.01`

---

## 3. 课程中的上下文随机化

除了物理参数随机化，训练课程还会随机采样持杯上下文。

这里需要明确：

- 持杯上下文采样属于**课程分布**
- 它和本文件前两章描述的物理参数随机化、步级噪声是两套不同机制
- 当前项目不再使用“左右臂独立 Bernoulli”去拼接上下文

### 3.1 episode 上下文变量

当前课程系统使用显式的离散上下文变量：

```text
c_episode ∈ {none, left_only, right_only, both}
```

其语义为：

- `none`：双臂都不持杯
- `left_only`：仅左臂持杯
- `right_only`：仅右臂持杯
- `both`：双臂都持杯

在每次 episode reset 时，系统先从当前课程阶段的上下文分布中采样 `c_episode`，再映射为：

```text
none       -> (left_occupied=0, right_occupied=0)
left_only  -> (left_occupied=1, right_occupied=0)
right_only -> (left_occupied=0, right_occupied=1)
both       -> (left_occupied=1, right_occupied=1)
```

### 3.2 当前默认三阶段分布

当前默认课程分布为：

| 阶段 | 上下文分布 |
|---|---|
| Stage 1 | `none: 1.0` |
| Stage 2 | `left_only: 0.5, right_only: 0.5` |
| Stage 3 | `none: 0.25, left_only: 0.25, right_only: 0.25, both: 0.25` |

因此当前项目的课程语义是：

- Stage 1 只训练无持杯
- Stage 2 只训练单臂持杯，且左右均衡
- Stage 3 训练最终混合分布，并首次正式引入 `both`

### 3.3 为什么不再使用独立 Bernoulli

不再使用 `o_L ~ Bernoulli(p)`、`o_R ~ Bernoulli(p)` 的主要原因是：

- 它无法直接表达“Stage 2 只允许单臂持杯、不允许双臂持杯”
- 它会把课程目标和采样实现细节耦合在一起
- 用显式离散分布更容易和 reward、日志、evaluation 以及 curriculum 配置保持一致

---

## 4. 门类型采样

训练里门类型变量可写为 `g`。形式上它从当前阶段允许的集合中采样：

`g ~ Categorical(G_stage)`

但当前默认三阶段配置里：

`G_stage = {push}`

所以在默认配置下，门类型实际上是常量：

`g = push`

也就是说：

- 接口上支持门类型采样
- 默认参数下没有实际的门类型随机化

---

## 5. 机械臂初始化随机化要求

除了上面的物理参数和上下文随机化，还需要在训练开始时对机械臂初始状态做随机化。

这里的随机化对象记为：

- `q_arm_init`：机械臂初始关节位姿

这部分目前应理解为“文档中新增的训练初始化要求”。当前文档先记录约束，不额外虚构具体数值范围。

### 5.1 真实初始位姿与强化学习初始位置

机械臂首先有一个“真正的初始位姿”：

- 如果没有抓取水杯，这个真正初始位姿指机械臂零位。
- 如果已经抓取水杯，这个真正初始位姿指预先设定好的持杯姿态。

在此基础上，才进入强化学习开始前的初始化随机化阶段：

- 没有抓取水杯时，可以直接从无杯可行集合中随机初始化。
- 已经抓取水杯时，需要在牢固抓取并保持水杯竖直的前提下，再把水杯运动到随机化的位置。

这里的“随机化位置”应理解为强化学习 episode 真正开始时的初始位置，也就是策略在第一个时刻看到的起始状态。

### 5.2 无持杯时的初始化

如果机械臂没有持拿水杯，那么初始关节位姿可以随机初始化，但必须限制在可行集合内：

```text
q_arm_init ~ sample from Q_free
```

其中 `Q_free` 需要同时满足：

- 避开关节限位
- 不发生自碰撞
- 不与周围环境发生碰撞

也就是：

```text
Q_free = {
  q |
  q avoids joint limits,
  q has no self-collision,
  q has no collision with surrounding environment
}
```

### 5.3 持杯时的初始化

如果机械臂持有水杯，那么初始关节位姿仍然要随机化，但必须额外满足“水杯被竖直持拿”的约束：

```text
q_arm_init ~ sample from Q_cup
```

其中 `Q_cup` 需要同时满足：

- 避开关节限位
- 不发生自碰撞
- 不与周围环境发生碰撞
- 水杯在初始化时保持竖直持拿

也就是：

```text
Q_cup = {
  q |
  q avoids joint limits,
  q has no self-collision,
  q has no collision with surrounding environment,
  cup is held upright at initialization
}
```

同时，在持杯情况下，还应满足下面这条额外语义：

- 机械臂不是简单地停留在“预设持杯姿态”上不动，而是要在牢固抓取水杯的情况下，把水杯运动到随机化的位置；
- 这个随机化后的位置，才是强化学习开始时的初始位置。

### 5.4 单臂持杯时的解释

如果是一侧机械臂持杯，另一侧不持杯，那么两侧应分别处理：

- 持杯侧：从 `Q_cup` 中采样
- 非持杯侧：从 `Q_free` 中采样

如果场景中存在水杯但当前策略不需要持杯约束，也可以理解为：

- 机械臂从零位或无杯可行姿态出发；
- 然后把末端或水杯引导到随机化后的强化学习初始位置；
- 整个过程中仍然必须避免关节限位、自碰撞和环境碰撞。

### 5.5 目前还没有写死到文档里的数值参数

这部分当前只有约束，还没有给出固定数值。后续若实现，需要进一步补充至少下面这些参数：

- 关节限位安全余量
- 水杯“竖直持拿”的允许倾斜角阈值
- 与环境的最小安全距离
- 初始关节采样分布的具体形式和范围

---

## 6. 当前随机化参数总表

| 类别 | 量 | 记号 | 分布/取值 | 默认参数 |
|---|---|---|---|---|
| 回合级 | 杯体质量 | `m_cup` | `Uniform(0.1, 0.8)` | `cup_mass_range=(0.1, 0.8)` |
| 回合级 | 门板质量 | `m_door` | `Uniform(5.0, 20.0)` | `door_mass_range=(5.0, 20.0)` |
| 回合级 | 门铰链阻尼 | `d_hinge` | `Uniform(0.5, 5.0)` | `door_damping_range=(0.5, 5.0)` |
| 回合级 | 基座位置扰动 | `p_base` | `p0 + [delta_x, delta_y, 0]` | `base_pos_nominal=(0,0,0)`, `base_pos_delta=0.03` |
| 步级 | 动作噪声 | `epsilon_a` | `Normal(0, sigma_a^2 I)` | `sigma_a=0.02` |
| 步级 | 观测噪声 | `epsilon_o` | `Normal(0, sigma_o^2 I)` | `sigma_o=0.01` |
| 上下文 | episode 持杯上下文 | `c_episode` | `Categorical({none,left_only,right_only,both})` | Stage 1:`none`; Stage 2:`left_only/right_only`; Stage 3:`none/left_only/right_only/both` |
| 任务类型 | 门类型 | `g` | `Categorical(G_stage)` | 默认 `push` |
| 初始化要求 | 无杯机械臂初始位姿 | `q_arm_init` | `sample from Q_free` | 数值范围待补充 |
| 初始化要求 | 持杯机械臂初始位姿 | `q_arm_init` | `sample from Q_cup` | 数值范围待补充 |
