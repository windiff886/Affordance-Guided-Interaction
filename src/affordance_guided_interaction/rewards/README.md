# rewards — 奖励与安全约束层

## 1. 本层在系统中的位置

```
observations/
  │  actor_obs, critic_obs
  │
  ▼
policy/  ──→  action (τ_t)
  │
  ▼
envs/  ──→  next_state, contact_events, done
  │
  ├── 任务进展状态（门角度、按钮状态 ...）
  ├── 接触事件（冲击力、self-collision ...）
  └── 精确物理状态（杯体位姿 ...）
          │
          ▼
    ┌─────────────────────┐
    │  rewards/            │  ◄── 本层
    │                      │
    │  task_reward.py       │  主任务进展奖励
    │  stability_reward.py  │  持杯稳定性奖励
    │  safety_penalty.py    │  碰撞、限位等安全惩罚
    │  reward_manager.py    │  聚合分项 + 日志输出
    └────────┬────────────┘
             │  reward_info dict
             ▼
       training/ (PPO update)
```

rewards 层独立于环境和策略实现，通过纯函数从当前步的状态信息中计算奖励标量。核心设计原则：

- **所有奖励分项独立计算**，由 `RewardManager` 统一聚合
- **权重全部由配置驱动**，不硬编码在函数内
- **每步输出分项日志字典**，供 TensorBoard / WandB 监控各奖励曲线
- **occupancy mask 条件化激活** 稳定性奖励：`occupied=0` 时不施加末端约束

---

## 2. 总体奖励结构

```
total_reward =
    r_task_progress                                  # 主任务
  + m_occ · λ_stab(stability_level) · r_carry_stability  # 持杯稳定
  + r_effective_contact                              # 有效接触 bonus
  - r_invalid_collision                              # 无效碰撞
  - r_self_collision                                 # 自碰撞
  - r_joint_limit                                    # 关节限位
  - r_torque_penalty                                 # 力矩正则
```

其中 `m_occ = 1 if occupied else 0`，`λ_stab` 由 `stability_level` 映射而来。

---

## 3. 文件清单与职责

| 文件 | 状态 | 说明 |
|------|------|------|
| `task_reward.py` | 🔲 待实现 | 按 affordance 类型计算主任务进展奖励 |
| `stability_reward.py` | 🔲 待实现 | 持杯稳定性奖励（7 个子项），参考 SoFTA |
| `safety_penalty.py` | 🔲 待实现 | 碰撞、self-collision、关节限位、力矩过大等安全惩罚 |
| `reward_manager.py` | 🔲 待重写 | 聚合各分项、应用 occupancy mask、输出分项日志 |
| `__init__.py` | 🔲 待更新 | 导出公共接口 |

---

## 4. `task_reward.py` — 主任务进展奖励

### 4.1 设计思路

主任务奖励不绑定单一操作类型，而是根据当前 affordance 类型分派计算。每种 affordance 定义自己的进展度量。

### 4.2 需要从 envs 层获取的信息

| 量 | 来源 | 说明 |
|----|------|------|
| `task_type` | `TaskManager` | 当前 affordance 类型标识 |
| `door_joint_pos` | `ArticulationView` | 门铰链当前角度 (rad) |
| `door_joint_pos_prev` | 上一步缓存 | 计算角度增量 |
| `door_target_angle` | `TaskManager` | 任务完成目标角度 |
| `button_pressed` | `ContactMonitor` / 传感器 | 按钮是否被按下 |
| `handle_triggered` | `ContactMonitor` / 传感器 | 把手是否已触发 |
| `contact_on_affordance` | `ContactMonitor` | 是否在有效 affordance 区域产生接触 |

### 4.3 各 affordance 类型奖励定义

#### push-affordance

```python
r_push = w_door_progress * (door_angle_t - door_angle_{t-1})   # 门角度增量
       + w_door_open * I(door_angle >= target)                 # 成功 bonus
```

#### press-affordance

```python
r_press = w_button_contact * I(contact_on_button)              # 接触按钮区域
        + w_button_trigger * I(button_pressed)                 # 触发成功 bonus
        + w_door_progress * (door_angle_t - door_angle_{t-1})  # 后续开门进展
```

#### handle-affordance

```python
r_handle = w_handle_contact * I(contact_on_handle)             # 接触把手
         + w_handle_trigger * I(handle_triggered)              # 把手到位 bonus
         + w_door_progress * (door_angle_t - door_angle_{t-1}) # 开门进展
```

#### sequential-affordance

分两阶段计算：stage 1 (前置动作) + stage 2 (开门)，使用阶段完成标志切换奖励目标。

### 4.4 接口设计

```python
def compute_task_reward(
    *,
    task_type: str,                     # "push" | "press" | "handle" | "sequential"
    door_joint_pos: float,              # 当前门铰链角度
    door_joint_pos_prev: float,         # 上一步门角度
    door_target_angle: float,           # 目标角度
    contact_on_affordance: bool,        # 是否在有效区域产生接触
    button_pressed: bool = False,       # press 类型专用
    handle_triggered: bool = False,     # handle 类型专用
    stage_completed: bool = False,      # sequential 类型专用
    weights: TaskRewardWeights,         # 配置权重 dataclass
) -> TaskRewardInfo:
```

`TaskRewardInfo` 返回 `total` 和各子项的命名分解。

---

## 5. `stability_reward.py` — 持杯稳定性奖励

### 5.1 设计思路

参考 SoFTA / Hold My Beer 的末端稳定性奖励设计。所有指标围绕 gripper frame 定义。该奖励项只在 `occupied = 1` 时激活。

### 5.2 需要从 observations 层获取的信息

直接消费 `actor_obs["stability_proxy"]` 中的量：

| 量 | 键名 | 来源 |
|----|------|------|
| EE 线加速度 | `stability_proxy["linear_acceleration"]` | `(3,)` 差分估计 |
| EE 角加速度 | `stability_proxy["angular_acceleration"]` | `(3,)` 差分估计 |
| 杯体倾斜 | `stability_proxy["tilt"]` | `float` |
| Jerk proxy | `stability_proxy["jerk_proxy"]` | `float` |

另外需要从 envs 层获取：

| 量 | 来源 | 说明 |
|----|------|------|
| `tau_t` | 当前步力矩输出 | `(6,)` |
| `tau_{t-1}` | 上一步力矩输出 | `(6,)` |

### 5.3 七个子项定义

所有 α 和 λ 系数均为可配置权重。

#### (1) 线加速度惩罚

```
r_acc = -α₁ · ‖ä_EE‖²
```

#### (2) 角加速度惩罚

```
r_ang_acc = -α₂ · ‖α̇_EE‖²
```

#### (3) 零线加速度奖励

```
r_zero_acc = α₃ · exp(-λ_acc · ‖ä_EE‖²)
```

#### (4) 零角加速度奖励

```
r_zero_ang_acc = α₄ · exp(-λ_ang · ‖α̇_EE‖²)
```

#### (5) 重力倾斜惩罚

```
r_grav_xy = -α₅ · tilt²
```

其中 `tilt = |P_xy(R_EE^T @ g)|`，由 `stability_proxy["tilt"]` 直接提供。

#### (6) 力矩变化平滑项

```
r_torque_smooth = -α₆ · ‖τ_t - τ_{t-1}‖²
```

#### (7) 力矩幅值正则项

```
r_torque_reg = -α₇ · ‖τ_t‖²
```

### 5.4 总稳定性奖励

```
r_carry_stability = r_acc + r_ang_acc + r_zero_acc + r_zero_ang_acc
                  + r_grav_xy + r_torque_smooth + r_torque_reg
```

经 occupancy mask 和 stability_level 调制后：

```
final_stability = m_occ · λ_stab(stability_level) · r_carry_stability
```

### 5.5 接口设计

```python
@dataclass
class StabilityRewardWeights:
    alpha_acc: float = 0.1
    alpha_ang_acc: float = 0.1
    alpha_zero_acc: float = 0.05
    alpha_zero_ang_acc: float = 0.05
    lambda_acc: float = 10.0
    lambda_ang: float = 10.0
    alpha_grav_xy: float = 0.2
    alpha_torque_smooth: float = 0.01
    alpha_torque_reg: float = 0.001

def compute_carry_stability_reward(
    *,
    stability_proxy: dict,              # actor_obs["stability_proxy"]
    torque: np.ndarray,                 # 当前步力矩 (6,)
    torque_prev: np.ndarray,            # 上一步力矩 (6,)
    occupied: float,                    # 0.0 或 1.0
    stability_level: float,             # 稳定性等级标量
    weights: StabilityRewardWeights,    # 配置权重
) -> StabilityRewardInfo:
```

`StabilityRewardInfo` 返回 `total` 和全部 7 个子项的分解。

---

## 6. `safety_penalty.py` — 安全惩罚

### 6.1 设计思路

惩罚危险行为，包括无效碰撞、自碰撞、关节限位违规和过大力矩。这些惩罚项在 occupied 与 unoccupied 状态下**始终激活**。

### 6.2 需要从 envs 层获取的信息

| 量 | 来源 | 说明 |
|----|------|------|
| `contact_events` | `ContactMonitor` | 包含各 link 的碰撞力、碰撞对象等 |
| `self_collision` | `ContactMonitor` | 是否发生自碰撞 |
| `joint_positions` | `ArticulationView` | 关节角度 `(6,)`，用于限位检查 |
| `joint_limits` | 配置 | 各关节角度上下限 `(6, 2)` |
| `joint_velocities` | `ArticulationView` | 关节角速度 `(6,)` |
| `torque` | 当前步输出 | `(6,)` |

### 6.3 惩罚子项

#### (1) 无效碰撞

对不在有效 affordance 区域内的碰撞给予惩罚：

```
r_invalid_collision = -β₁ · Σ(contact_forces_on_non_affordance_links)
```

#### (2) 自碰撞

```
r_self_collision = -β₂ · I(self_collision_detected)
```

#### (3) 关节限位逼近

当关节角度逼近限位时施加递增惩罚：

```
r_joint_limit = -β₃ · Σ max(0, |q_i - q_center_i| - q_range_i * margin)²
```

#### (4) 关节速度过大

```
r_velocity = -β₄ · max(0, ‖dq‖ - v_threshold)²
```

#### (5) Gripper 高冲击接触（occupied 时增强）

```
r_gripper_impact = -β₅ · m_occ · gripper_impact_force
```

### 6.4 接口设计

```python
@dataclass
class SafetyPenaltyWeights:
    beta_invalid_collision: float = 0.5
    beta_self_collision: float = 1.0
    beta_joint_limit: float = 0.1
    beta_velocity: float = 0.01
    beta_gripper_impact: float = 0.5
    joint_limit_margin: float = 0.9       # 范围的百分比，超过开始惩罚
    velocity_threshold: float = 5.0       # rad/s

def compute_safety_penalty(
    *,
    contact_events: dict,               # ContactMonitor 输出
    self_collision: bool,
    joint_positions: np.ndarray,         # (6,)
    joint_limits: np.ndarray,            # (6, 2) 上下限
    joint_velocities: np.ndarray,        # (6,)
    torque: np.ndarray,                  # (6,)
    occupied: float,                     # 用于 gripper 冲击增强
    gripper_impact_force: float = 0.0,   # gripper link 受到的冲击力
    weights: SafetyPenaltyWeights,
) -> SafetyPenaltyInfo:
```

---

## 7. `reward_manager.py` — 奖励聚合器

### 7.1 设计思路

`RewardManager` 是奖励层的唯一对外接口。它调用各子模块的计算函数，应用 occupancy mask，聚合总奖励，并输出包含所有分项的日志字典。

### 7.2 接口设计

```python
@dataclass
class RewardConfig:
    task: TaskRewardWeights
    stability: StabilityRewardWeights
    safety: SafetyPenaltyWeights
    stability_level_map: dict[float, float]   # stability_level → λ_stab

class RewardManager:
    def __init__(self, config: RewardConfig) -> None: ...

    def compute(
        self,
        *,
        # 任务进展相关
        task_type: str,
        door_joint_pos: float,
        door_joint_pos_prev: float,
        door_target_angle: float,
        contact_on_affordance: bool,
        button_pressed: bool = False,
        handle_triggered: bool = False,
        stage_completed: bool = False,
        # 稳定性相关
        stability_proxy: dict,
        torque: np.ndarray,
        torque_prev: np.ndarray,
        # 安全相关
        contact_events: dict,
        self_collision: bool,
        joint_positions: np.ndarray,
        joint_limits: np.ndarray,
        joint_velocities: np.ndarray,
        gripper_impact_force: float = 0.0,
        # 上下文
        occupied: float,
        stability_level: float,
    ) -> dict[str, float]:
        """返回包含 total_reward 和全部分项的字典。"""
```

### 7.3 输出字典结构

```python
{
    # 主任务
    "task_progress":          float,  # r_task_progress
    "task/door_angle_delta":  float,  # 分项细节
    "task/success_bonus":     float,

    # 稳定性（占用时）
    "stability_total":        float,  # m_occ * λ_stab * r_carry_stability
    "stability/acc":          float,
    "stability/ang_acc":      float,
    "stability/zero_acc":     float,
    "stability/zero_ang_acc": float,
    "stability/grav_xy":      float,
    "stability/torque_smooth":float,
    "stability/torque_reg":   float,

    # 安全惩罚
    "safety_total":           float,
    "safety/invalid_collision": float,
    "safety/self_collision":  float,
    "safety/joint_limit":     float,
    "safety/velocity":        float,
    "safety/gripper_impact":  float,

    # 聚合
    "total_reward":           float,
}
```

---

## 8. 配置结构 (`configs/reward/`)

```yaml
# configs/reward/default.yaml

task:
  w_door_progress: 5.0
  w_door_open: 10.0
  w_button_contact: 1.0
  w_button_trigger: 5.0
  w_handle_contact: 1.0
  w_handle_trigger: 5.0

stability:
  alpha_acc: 0.1
  alpha_ang_acc: 0.1
  alpha_zero_acc: 0.05
  alpha_zero_ang_acc: 0.05
  lambda_acc: 10.0
  lambda_ang: 10.0
  alpha_grav_xy: 0.2
  alpha_torque_smooth: 0.01
  alpha_torque_reg: 0.001

safety:
  beta_invalid_collision: 0.5
  beta_self_collision: 1.0
  beta_joint_limit: 0.1
  beta_velocity: 0.01
  beta_gripper_impact: 0.5
  joint_limit_margin: 0.9
  velocity_threshold: 5.0

# stability_level → λ_stab 映射
stability_level_map:
  0.0: 0.0   # 无稳定性要求
  1.0: 0.5   # 中等
  2.0: 1.0   # 严格
```

---

## 9. 与 observations 层的数据接口

rewards 层消费的数据来自三个来源：

| 数据来源 | 提供的关键量 | 消费者 |
|----------|-------------|--------|
| `actor_obs["stability_proxy"]` | tilt, linear_acceleration, angular_acceleration, jerk_proxy | `stability_reward.py` |
| `actor_obs["context"]` | occupied, stability_level | `reward_manager.py` (mask 控制) |
| `actor_obs["proprio"]` | joint_positions, joint_velocities | `safety_penalty.py` |
| `envs/` 环境层直接 | door_joint_pos, contact_events, self_collision 等 | `task_reward.py`, `safety_penalty.py` |
| `policy/` 策略输出 | torque (τ_t, τ_{t-1}) | `stability_reward.py` |

**关键约束**：rewards 层不依赖 `critic_obs["privileged"]`。所有奖励计算使用仿真地面真值（通过 envs 层获取），而不是 actor 观测。这是因为奖励函数在训练时由仿真环境调用，可以访问精确状态。

---

## 10. occupancy mask 机制

```python
m_occ = 1.0 if occupied == 1.0 else 0.0

# stability_level → 权重系数的映射
λ_stab = stability_level_map.get(stability_level, 0.0)

# 最终稳定性奖励
stability_contribution = m_occ * λ_stab * r_carry_stability
```

当 `occupied = 0` 时：
- 稳定性 7 个子项全部不计入总奖励
- 策略可以自由使用高速、高加速度的交互方式

当 `occupied = 1` 时：
- 稳定性项激活，强度由 `stability_level` 控制
- 策略受到末端平稳约束，倾向于低冲击交互

---

## 11. 模块间依赖关系

```
task_reward.py          ← 纯函数，无内部依赖
stability_reward.py     ← 纯函数，无内部依赖
safety_penalty.py       ← 纯函数，无内部依赖
        ↑
reward_manager.py       ← 聚合以上三个模块
        ↑
__init__.py             ← 导出 RewardManager 及配置类
```

所有子模块为**纯函数**（给定输入确定输出），不维护内部状态。`RewardManager` 持有配置权重，但不维护跨步状态（上一步力矩由调用方传入）。

---

## 12. 实现优先级

根据 `project_architecture.md` 推荐的实现顺序：

1. **阶段 A — 最小闭环**
   - 实现 `task_reward.py` 中的 push-affordance 分支
   - 实现 `safety_penalty.py` 中的关节限位和力矩正则
   - 实现 `reward_manager.py` 的基本聚合逻辑
   - 暂不实现稳定性奖励（occupied=0 场景优先）

2. **阶段 B — 稳定性约束**
   - 实现 `stability_reward.py` 全部 7 个子项
   - 在 `reward_manager.py` 中加入 occupancy mask 逻辑
   - 验证 occupied/unoccupied 行为分化

3. **阶段 C — 扩展 affordance**
   - 在 `task_reward.py` 中补充 press、handle、sequential 分支
   - 在 `safety_penalty.py` 中补充碰撞检测子项

---

## 13. 测试计划

```bash
python3 -m pytest tests/test_rewards.py -v
```

应覆盖：

- **task_reward** — 各 affordance 类型的正确奖励值、边界条件
- **stability_reward** — 7 个子项的数值正确性、occupancy mask 行为
- **safety_penalty** — 关节限位检测、自碰撞标志、力矩惩罚
- **reward_manager** — 聚合总值正确、分项日志完整、配置权重生效
- **零速度/零加速度场景** — 静止时奖励应为最优值
- **极端值** — 大力矩、大加速度时惩罚应显著
