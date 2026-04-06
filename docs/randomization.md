# 项目中的随机化量与参数

这份文档只保留一件事：说明当前项目里哪些量会被随机化，以及对应的随机化参数是多少。

本文以当前默认实现为准，主要对应：

- `src/affordance_guided_interaction/training/domain_randomizer.py`
- `src/affordance_guided_interaction/training/curriculum_manager.py`
- `configs/curriculum/default.yaml`

---

## 1. 回合级随机化

这些量在每次 episode reset 时重新采样，并在该 episode 内保持不变。

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

记机器人基座世界坐标为 `p_base ∈ R^3`。当前使用门外侧**扇形环采样**，`Z` 保持标称值不变：

```text
# 以推板中心为圆心，在门外侧扇形环内均匀采样
r ~ Uniform(r_min, r_max)          # 到推板中心的距离
theta ~ Uniform(-half_angle, +half_angle)  # 相对标称方向的角度偏移
yaw = atan2(push_center - base_pos) + delta_yaw  # 朝向推板中心，带小扰动
```

当前默认参数（定义在 `DoorPushEnvCfg`）：

- `push_plate_center_xy = (2.98, 0.27)` — 推板中心世界坐标
- `base_reference_xy = (3.72, 0.27)` — 推板正前方参考点（使扇形对称，nominal_angle = 0°）
- `base_height = 0.12` — 基座固定高度 (m)
- `base_radius_range = (0.45, 0.60)` — 到推板中心的半径范围 (m)
- `base_sector_half_angle_deg = 20.0` — 扇形半角 (°)
- `base_yaw_delta_deg = 10.0` — yaw 扰动范围 (°)

#### 半径范围选取依据（工作带前移）

Z1 机械臂关键尺寸（from URDF）：

| 参数 | 值 |
|------|-----|
| 臂链总长（joint1 → gripper） | 0.849 m |
| 肩关节到 base_link 的前方偏移 | 0.151 m |
| 理论最大前向可达（关节全伸直） | 1.00 m |
| Z1 官方水平可达（考虑关节限位） | 0.80 m |
| **有效可达（肩偏移 + 官方可达）** | **~0.95 m** |

可达性判断：

| 采样距离 r | 余量 (0.95 - r) | 判断 |
|-----------|----------------|------|
| 0.45 m | +0.50 m | 明显前移，仍在工作带内 |
| 0.52 m | +0.43 m | 近端工作区 |
| 0.60 m | +0.35 m | 保持推门跟随余量 |
| 0.95 m | ±0.00 m | 极限（不采用） |
| ≥1.00 m | < 0 | 物理不可达 |

当前区间不再以“尽量铺满臂展”为目标，而是整体前移到更适合稳定建立接触和持续推门的工作带。
下限 0.45m 明显近于旧配置，但仍避免把基座直接压到门前极近位置；上限 0.60m 则避免把大量样本放到“初始能碰到、但门转动后难以持续跟随”的远端区域。

#### 采样后机器人位置范围

```
采样结果（nominal_angle = 0°，扇形对称）：
  X ∈ [~3.40, ~3.54]    推板前方 0.45-0.60m
  Y ∈ [~0.12, ~0.42]    围绕 y=0.27 对称展开
  Z = 0.12               固定
```

#### 碰撞安全验证

- 机器人最小 X ≈ 3.40（r=0.45, θ=±20°）
- 前墙面 X = 2.95
- 间距 = 0.45m > 底盘半径 0.30m → 仍保留净空

---

## 2. 步级噪声参数

这些量在每步 step() 时独立采样并注入。

- **动作噪声**：在力矩截断（clip）之后注入，注入后重新截断以保证物理安全。
- **观测噪声**：仅注入到 Actor 观测的 proprio 分支（关节位置 + 关节速度，共 24 维）。Critic 观测使用无噪声的真实状态（不对称设计）。

注入位置：`DoorPushEnv._pre_physics_step()` 内部，由 `DomainRandomizer` 实例驱动。

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

## 5. 当前随机化参数总表

| 类别 | 量 | 记号 | 分布/取值 | 默认参数 |
|---|---|---|---|---|
| 回合级 | 杯体质量 | `m_cup` | `Uniform(0.1, 0.8)` | `cup_mass_range=(0.1, 0.8)` |
| 回合级 | 门板质量 | `m_door` | `Uniform(5.0, 20.0)` | `door_mass_range=(5.0, 20.0)` |
| 回合级 | 门铰链阻尼 | `d_hinge` | `Uniform(0.5, 5.0)` | `door_damping_range=(0.5, 5.0)` |
| 回合级 | 基座径向距离 | `r` | `Uniform(0.45, 0.60)` | `base_radius_range=(0.45, 0.60)` |
| 回合级 | 基座扇形角度 | `θ` | `Uniform(-20°, +20°)` | `base_sector_half_angle_deg=20.0` |
| 回合级 | 基座 yaw 扰动 | `δ_yaw` | `Uniform(-10°, +10°)` | `base_yaw_delta_deg=10.0` |
| 步级 | 动作噪声 | `epsilon_a` | `Normal(0, sigma_a^2 I)` | `sigma_a=0.02` |
| 步级 | 观测噪声 | `epsilon_o` | `Normal(0, sigma_o^2 I)` | `sigma_o=0.01` |
| 上下文 | episode 持杯上下文 | `c_episode` | `Categorical({none,left_only,right_only,both})` | Stage 1:`none`; Stage 2:`left_only/right_only`; Stage 3:`none/left_only/right_only/both` |
| 任务类型 | 门类型 | `g` | `Categorical(G_stage)` | 默认 `push` |
