# 训练流程详解

> 生成日期：2026-04-02
>
> 本文档详细梳理 Affordance-Guided-Interaction 项目从环境初始化到策略收敛的完整训练流程，
> 涵盖数据流、模块调用顺序、张量维度、公式细节和课程进阶逻辑。

---

## 目录

1. [训练全景图](#1-训练全景图)
2. [单步数据流：从环境到梯度](#2-单步数据流从环境到梯度)
3. [训练主循环伪代码](#3-训练主循环伪代码)
4. [感知管线：从 RGB-D 到 z_aff](#4-感知管线从-rgb-d-到-z_aff)
5. [观测构建](#5-观测构建)
6. [策略网络前向传播](#6-策略网络前向传播)
7. [环境步进与物理状态读取](#7-环境步进与物理状态读取)
8. [奖励计算](#8-奖励计算)
9. [轨迹采集（Rollout Collection）](#9-轨迹采集rollout-collection)
10. [GAE 优势函数估计](#10-gae-优势函数估计)
11. [PPO 参数更新](#11-ppo-参数更新)
12. [课程管理（Curriculum）](#12-课程管理curriculum)
13. [域随机化（Domain Randomization）](#13-域随机化domain-randomization)
14. [训练指标与日志](#14-训练指标与日志)
15. [Checkpoint 与恢复](#15-checkpoint-与恢复)
16. [完整训练时序图](#16-完整训练时序图)

---

## 1. 训练全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        训练主循环（尚需实现）                              │
│                                                                         │
│  for iteration in 1..N:                                                 │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│    │  Rollout      │───▶│  GAE 计算    │───▶│  PPO 更新    │             │
│    │  Collector    │    │  (Buffer)    │    │  (Trainer)   │             │
│    └──────┬───────┘    └──────────────┘    └──────┬───────┘             │
│           │                                        │                     │
│           ▼                                        ▼                     │
│    ┌──────────────┐                        ┌──────────────┐             │
│    │  VecDoorEnv  │                        │  Metrics     │             │
│    │  (N 个并行)   │                        │  Logging     │             │
│    └──────┬───────┘                        └──────┬───────┘             │
│           │                                        │                     │
│           ▼                                        ▼                     │
│    ┌──────────────┐                        ┌──────────────┐             │
│    │  Curriculum   │◀──────────────────────│  Success Rate │             │
│    │  Manager     │                        │  (滑动窗口)    │             │
│    └──────┬───────┘                        └──────────────┘             │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                      │
│    │  Domain      │                                                      │
│    │  Randomizer  │                                                      │
│    └──────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 单步数据流：从环境到梯度

以下是**一个仿真步（single env step）** 的完整数据流向。
字段与维度严格对齐 `DoorInteractionEnv._read_physics_state()` 的实际返回内容：

```
action τ ∈ R^12 (双臂力矩)
    │
    ├── 1. clip(τ, -effort_limit, effort_limit)
    │
    ├── 2. _sim_step(clipped_action) × decimation 次物理步进
    │
    ├── 3. _read_physics_state() → state dict，完整字段如下：
    │      │
    │      ├── 关节状态（左右臂各 6 维）
    │      │     left_joint_positions        (6,)   左臂关节角度 q
    │      │     left_joint_velocities       (6,)   左臂关节速度 dq
    │      │     left_joint_torques          (6,)   左臂关节力矩 τ
    │      │     right_joint_positions       (6,)   右臂关节角度 q
    │      │     right_joint_velocities      (6,)   右臂关节速度 dq
    │      │     right_joint_torques         (6,)   右臂关节力矩 τ
    │      │
    │      ├── 左臂末端
    │      │     left_ee_position            (3,)   夹爪世界坐标位置
    │      │     left_ee_orientation         (4,)   夹爪姿态四元数 (w,x,y,z)
    │      │     left_ee_linear_velocity     (3,)   线速度
    │      │     left_ee_angular_velocity    (3,)   角速度
    │      │
    │      ├── 右臂末端
    │      │     right_ee_position           (3,)
    │      │     right_ee_orientation        (4,)
    │      │     right_ee_linear_velocity    (3,)
    │      │     right_ee_angular_velocity   (3,)
    │      │
    │      ├── 门状态
    │      │     door_joint_pos              float   门铰链角度 (rad)
    │      │     door_joint_vel              float   门铰链角速度
    │      │     door_pose                   (7,)    门板位姿 (pos3+quat4)
    │      │
    │      ├── 杯体状态（无杯体时全零/None）
    │      │     cup_position                (3,) | None  杯体世界坐标（用于脱落检测）
    │      │     cup_pose                    (7,)   杯体完整位姿 (pos3+quat4)
    │      │     cup_linear_vel              (3,)   杯体线速度
    │      │     cup_angular_vel             (3,)   杯体角速度
    │      │
    │      └── 视觉特征
    │            door_embedding              ndarray | None  门点云 embedding（当前占位 None）
    │
    ├── 4. ContactMonitor.update() → ContactSummary
    │      ├── link_forces      dict[str, float]   各 link 的接触力
    │      ├── self_collision    bool               是否自碰撞
    │      └── cup_dropped       bool               杯是否脱落
    │
    ├── 5. TaskManager.update() → TaskStatus
    │      ├── done             bool               回合是否结束
    │      ├── success          bool               任务是否成功
    │      ├── door_angle       float              当前门角度
    │      ├── door_angle_prev  float              上一步门角度
    │      └── step_count       int                当前步数
    │
    ├── 6. ActorObsBuilder.build() → actor_obs (dict)
    │      （door_embedding 从 state dict 直接传入；为 None 时填零向量 768 维）
    │
    ├── 7. CriticObsBuilder.build() → critic_obs (dict)
    │      （追加 privileged 信息：door_pose, cup_*, domain_params 等）
    │
    └── 8. RewardManager.step() → (reward, terminate, info)
```

> **关于 AffordancePipeline 与当前环境的关系**
>
> `AffordancePipeline.encode()` 确实设计为返回 `(z_aff, z_prog)` 两个值——
> 其中 `z_aff` 是 `np.ndarray (embed_dim,)` 的 Point-MAE embedding，
> `z_prog` 是一个 dict（包含 `"vector"(4,)`, `"door_angle"`, `"button_pressed"`,
> `"handle_triggered"`, `"progress"` 五个键）。
>
> 但在**当前代码中**，`DoorInteractionEnv._read_physics_state()` 将
> `door_embedding` 作为 state dict 的一个字段（当前值为 `None`），
> 直接传递给 `ActorObsBuilder.build(door_embedding=state.get("door_embedding"))`。
> **环境的 `step()` / `reset()` 循环并未调用 `AffordancePipeline.encode()`。**
>
> AffordancePipeline 的实际集成需要在以下两种方案中选择：
> - **方案 A**：在 `DoorInteractionEnv._read_physics_state()` 内部调用管线，
>   将返回的 `z_aff` 填入 `state["door_embedding"]`
> - **方案 B**：在外部 wrapper 或 `RolloutCollector` 层面调用管线，
>   在观测送入 Actor 前注入 `door_embedding`

---

## 3. 训练主循环伪代码

以下是完整训练循环的编排逻辑（`scripts/train.py` 需要实现的内容）：

```python
# ═══════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════

# 1. 创建并行环境
envs = VecDoorEnv(n_envs=N, env_config=env_cfg)

# 2. 创建网络
actor = Actor(actor_cfg)             # 含 RecurrentBackbone + ActionHead
critic = Critic(critic_cfg)          # 非对称 MLP

# 3. 创建训练组件
buffer = RolloutBuffer(
    n_envs=N, n_steps=T,
    actor_branch_dims={"proprio": 60, "ee": 26, "context": 2,
                       "stability": 24, "visual": 768},
    privileged_dim=30, action_dim=12,
    rnn_hidden_dim=512, rnn_num_layers=1,
)
collector = RolloutCollector(actor, critic, buffer,
                             batch_actor_flatten_fn, priv_flatten_fn)
ppo = PPOTrainer(actor, critic, ppo_cfg)
curriculum = CurriculumManager(window_size=50, threshold=0.8)
randomizer = DomainRandomizer(rand_cfg)
metrics = TrainingMetrics()

# 4. 初始环境 reset
actor_obs, critic_obs = envs.reset()

# ═══════════════════════════════════════════════════════
# 训练主循环
# ═══════════════════════════════════════════════════════

for iteration in range(1, max_iterations + 1):

    # ── Phase 1: 采集 T 步轨迹 ──────────────────────────
    actor_obs, critic_obs, collect_stats = collector.collect(
        envs, n_steps=T,
        current_actor_obs=actor_obs,
        current_critic_obs=critic_obs,
    )

    # ── Phase 2: 计算 GAE 优势函数 ─────────────────────
    buffer.compute_gae(
        gamma=ppo_cfg.gamma,       # 0.99
        lam=ppo_cfg.lam,           # 0.95
        last_values=collector.last_values,
        last_dones=collector.last_dones,
    )

    # ── Phase 3: PPO 多轮更新 ──────────────────────────
    update_stats = ppo.update(buffer)

    # ── Phase 4: 指标记录 ──────────────────────────────
    metrics.update_ppo(**update_stats)
    summary = metrics.summarize()
    # → 写入 TensorBoard / WandB

    # ── Phase 5: 课程跃迁 ──────────────────────────────
    advanced = curriculum.report_epoch(metrics.success_rate)
    if advanced:
        print(f"进入阶段 {curriculum.current_stage}")
        # 下一个 reset 时会使用新阶段的 door_types 和 cup_probability

    # ── Phase 6: 清空缓冲，准备下一轮 ──────────────────
    buffer.clear()
    metrics.reset()

    # ── Phase 7: 定期保存 Checkpoint ───────────────────
    if iteration % save_interval == 0:
        save_checkpoint(actor, critic, ppo, curriculum, iteration)
```

---

## 4. 感知管线：从 RGB-D 到 z_aff

感知管线是整个系统的"眼睛"，将原始视觉输入转化为策略可用的 affordance 特征。

### 4.1 管线流程

```
RGB (H,W,3) + Depth (H,W) + task_goal("push"/"press"/...)
    │
    ├── Step 1: 开集分割 (LangSAM / Grounded-SAM 2)
    │   文本提示: ["door", "door handle", "button"]
    │   输出: binary masks (per region)
    │
    ├── Step 2: 深度反投影
    │   对每个 mask 区域:
    │     point_cloud = backproject_depth(depth, mask, camera_intrinsics)
    │   合并所有局部点云
    │
    ├── Step 3: 体素降采样
    │   voxel_downsample(merged_pc, voxel_size=0.005)
    │   sample_or_pad(downsampled, max_points=1024)
    │
    ├── Step 4: Point-MAE 编码 (权重完全冻结)
    │   冻结 Transformer:
    │     FPS 分组 → KNN 局部邻域 → PointEncoder → TransformerEncoder
    │   输出: mean_pool ⊕ max_pool → z_aff ∈ R^768
    │
    └── Step 5: z_prog 构建 (来自仿真状态)
        z_prog = [door_angle, button_pressed, handle_triggered, progress]
                                                                    │
        其中 progress 根据 task_goal 计算:                            │
          push/handle: min(door_angle / 1.57, 1.0)                  │
          press:       button_pressed (0 或 1)                       │
          sequential:  0.5 * button_pressed + 0.5 * min(θ/1.57, 1) │
```

### 4.2 Point-MAE 编码器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trans_dim` | 384 | Transformer 隐层维度 |
| `encoder_dims` | 384 | Point Encoder 通道数 |
| `depth` | 12 | Transformer 层数 |
| `num_heads` | 6 | 注意力头数 |
| `group_size` | 32 | 每个局部组的点数 |
| `num_group` | 64 | FPS 采样的组数 |
| `embed_dim` | 768 | 输出维度 (trans_dim × 2) |

> **关键设计**：Point-MAE 权重**完全冻结**（`requires_grad=False` + `.eval()`），
> 不参与 RL 梯度回传。这保证了感知层在训练初期不会因为策略探索产生的梯度而崩溃。

---

## 5. 观测构建

### 5.1 Actor 观测（actor_obs）

Actor 只能看到**部署时可获得的信息**。

```python
actor_obs = {
    "proprio": {                           # 本体感知
        "left_joint_positions":    (6,),   # 左臂关节角度 q
        "left_joint_velocities":   (6,),   # 左臂关节速度 dq
        "right_joint_positions":   (6,),   # 右臂关节角度 q
        "right_joint_velocities":  (6,),   # 右臂关节速度 dq
        "previous_actions":        (k,12), # 历史动作 (k=3)
        # 可选:
        "left_joint_torques":      (6,),   # 力矩反馈
        "right_joint_torques":     (6,),   # 力矩反馈
    },
    "left_gripper_state": {                # 左臂末端
        "position":         (3,),          # 夹爪位置
        "orientation":      (4,),          # 夹爪姿态 (w,x,y,z)
        "linear_velocity":  (3,),          # 线速度
        "angular_velocity": (3,),          # 角速度
    },
    "right_gripper_state": { ... },        # 右臂末端（同结构）

    "context": {                           # 任务上下文
        "left_occupied":  (1,),            # 左手是否持杯 (0/1)
        "right_occupied": (1,),            # 右手是否持杯 (0/1)
    },
    "left_stability_proxy": {              # 左臂稳定性代理
        "tilt":                   float,   # 重力倾斜度
        "linear_velocity_norm":   float,   # 线速度范数
        "linear_acceleration":    (3,),    # 线加速度（有限差分）
        "angular_velocity_norm":  float,   # 角速度范数
        "angular_acceleration":   (3,),    # 角加速度（有限差分）
        "jerk_proxy":             float,   # Jerk 近似
        "recent_acc_history":     (10,),   # 近期加速度历史
    },
    "right_stability_proxy": { ... },      # 右臂稳定性（同结构）

    "door_embedding": (768,),              # 冻结 Point-MAE 输出的视觉特征
}
```

### 5.2 Critic 观测（critic_obs）

Critic 额外拥有**训练时特权信息**（部署时不可获得）。

```python
critic_obs = {
    "actor_obs": { ... },                  # 完整的 actor 观测
    "privileged": {                        # 特权信息
        "door_pose":       (7,),           # 门板位姿 (pos+quat)
        "door_joint_pos":  (1,),           # 门铰链角度
        "door_joint_vel":  (1,),           # 门铰链角速度
        "cup_pose":        (7,),           # 杯体位姿
        "cup_linear_vel":  (3,),           # 杯体线速度
        "cup_angular_vel": (3,),           # 杯体角速度
        "cup_mass":        (1,),           # 杯体质量（域随机化参数）
        "door_mass":       (1,),           # 门质量
        "door_damping":    (1,),           # 门阻尼
        "base_pos":        (3,),           # 基座世界坐标
    }
}
```

### 5.3 稳定性代理（Stability Proxy）计算公式

稳定性代理是整个框架的核心创新之一，用部署时可获得的末端传感信息近似杯体稳定性。

| 指标 | 公式 | 含义 |
|------|------|------|
| **tilt** | `‖g_local[:2]‖` 其中 `g_local = R_ee^T · g_world` | 重力在末端局部 XY 平面的投影长度 → 越大越倾斜 |
| **lin_vel_norm** | `‖v_ee‖` | 末端线速度范数 |
| **lin_acc** | `(v_t - v_{t-1}) / dt` | 线加速度（一阶有限差分） |
| **ang_vel_norm** | `‖ω_ee‖` | 末端角速度范数 |
| **ang_acc** | `(ω_t - ω_{t-1}) / dt` | 角加速度（一阶有限差分） |
| **jerk_proxy** | `‖a_t - a_{t-1}‖ / dt` | Jerk 近似（加速度变化率） |

### 5.4 Actor 观测分支展平

为了送入神经网络，原始 dict 被展平为 5 个命名分支张量：

| 分支名 | 来源 | 展平后维度（参考值） |
|--------|------|---------------------|
| `proprio` | 关节状态 + 历史动作 | ~60 |
| `ee` | 左右末端位姿与速度 | ~26 |
| `context` | occupied 标志 | 2 |
| `stability` | 左右稳定性代理 | ~24 |
| `visual` | door_embedding | 768 |

---

## 6. 策略网络前向传播

### 6.1 Actor 网络结构

```
各分支张量 (proprio, ee, context, stability, visual)
    │
    ├── proprio    → MLP encoder → f_proprio
    ├── ee         → MLP encoder → f_ee
    ├── context    → 直接拼接
    ├── stability  → MLP encoder → f_stab
    └── visual     → MLP encoder → f_vis
                          │
    ┌─────────────────────┘
    │  concat([f_proprio, f_ee, context, f_stab, f_vis])
    ▼
RecurrentBackbone (GRU / LSTM)
    │  隐状态 h_t ∈ R^{layers × hidden_dim}
    ▼
ActionHead (对角高斯分布)
    │  μ ∈ R^12,  log_σ ∈ R^12
    ▼
τ_t ~ N(μ, diag(σ²))  ──clip──▶  τ_t ∈ R^12 (双臂关节力矩)
```

### 6.2 Actor 接口

```python
# 采样模式（Rollout 阶段）
action, log_prob, entropy, hidden_new = actor.forward(
    flat_obs_branches,  # dict[str, Tensor(n_envs, dim)]
    hidden,             # Tensor(layers, n_envs, hidden_dim)
)

# 评估模式（PPO 更新阶段）
log_prob, entropy, hidden_new = actor.evaluate_actions(
    obs_t,     # dict[str, Tensor(B, dim)]  — 单时间步
    act_t,     # Tensor(B, action_dim)
    hidden,    # Tensor(layers, B, hidden_dim)
)
```

### 6.3 Critic 网络结构

```
actor 各分支张量 (展平后)  +  privileged 向量
    │                            │
    └────── concat ──────────────┘
                 │
          MLP (多层全连接)
                 │
                 ▼
           V(s) ∈ R^1
```

> **关键设计**：Critic 不含循环结构。它直接通过 MLP 拟合价值函数，
> 并利用 privileged information 获得比 Actor 更准确的状态估计。

---

## 7. 环境步进与物理状态读取

### 7.1 DoorInteractionEnv.step() 流程

```python
def step(self, action: np.ndarray):
    # 1. 动作预处理
    action = np.clip(action, -effort_limit, effort_limit)
    action += domain_noise  # 如果提供

    # 2. 物理步进 (× decimation)
    self._sim_step(action)          # [ISAAC_API] 占位

    # 3. 读取物理状态
    state = self._read_physics_state()  # [ISAAC_API] 占位
    # state 包含: joint_positions(12), joint_velocities(12),
    #            joint_torques(12), ee_states(左右各13维),
    #            door_angle(1), door_pose(7), cup_pose(7), cup_vel(6)

    # 4. 更新接触监控
    contact_summary = self._contact_monitor.update()

    # 5. 更新任务状态
    task_status = self._task_manager.update(
        door_angle=state.door_angle,
        contact_summary=contact_summary,
    )

    # 6. 感知编码 (如果启用)
    z_aff, z_prog = self._perception.encode(...) if enabled

    # 7. 构建观测
    actor_obs = self._actor_obs_builder.build(...)
    critic_obs = CriticObsBuilder.build(actor_obs=actor_obs, ...)

    # 8. 计算奖励
    reward, terminate, reward_info = self._reward_manager.step(...)

    # 9. 判定终止
    done = task_status.done or terminate

    return actor_obs, critic_obs, reward, done, info
```

### 7.2 VecDoorEnv 并行环境

```python
class VecDoorEnv:
    """包装 N 个 DoorInteractionEnv 的向量化环境。"""

    def step(self, actions):    # actions: (N, 12)
        results = [env.step(actions[i]) for i, env in enumerate(self._envs)]
        # 自动 reset: 如果 done=True 则 reset 该环境
        for i, (_, _, _, done, _) in enumerate(results):
            if done:
                results[i] = self._envs[i].reset(
                    domain_params=randomizer.sample_episode_params(),
                    door_type=curriculum.get_stage_config().door_types[...],
                    left_occupied=...,
                )
        return stacked_results
```

---

## 8. 奖励计算

### 8.1 总奖励公式

```
r_total = r_task + r_stab - r_safe
```

其中：

```
r_stab = m_L · (bonus_L + s_t · penalty_L)
       + m_R · (bonus_R + s_t · penalty_R)
```

- `m_L` = 1 if `left_occupied` else 0（右臂同理）
- `s_t` = 动态缩放因子（从 `s_min` 线性退火到 1.0）

### 8.2 任务奖励 r_task

```python
# 权重函数（密集进展引导）
if θ_t <= θ_target:
    w(θ) = w_delta
else:
    w(θ) = w_delta · max(α, 1 - k_decay · (θ_t - θ_target))

# 进展奖励（密集）
r_progress = w(θ) · (θ_t - θ_{t-1})

# 一次性成功奖励（稀疏）
r_success = w_open   (仅在首次 θ_t ≥ θ_target 时给予)

# 合计
r_task = r_progress + r_success
```

| 参数 | 含义 | 参考值 |
|------|------|--------|
| `w_delta` | 进展奖励基础权重 | 10.0 |
| `α` | 衰减下限 | 0.1 |
| `k_decay` | 超标后衰减速率 | 0.5 |
| `w_open` | 成功 bonus | 50.0 |
| `θ_target` | 成功门槛角度 (rad) | 1.05 (~60°) |

### 8.3 稳定性奖励 r_stab

对每个 occupied 手臂独立计算 `(bonus, penalty)`：

**Bonus（高斯核，鼓励零运动）：**

$$r_{\text{zero\_acc}} = w_{\text{zero\_acc}} \cdot \exp(-\lambda_{\text{acc}} \cdot \|a_{\text{lin}}\|^2)$$

$$r_{\text{zero\_ang}} = w_{\text{zero\_ang}} \cdot \exp(-\lambda_{\text{ang}} \cdot \|a_{\text{ang}}\|^2)$$

$$\text{bonus} = r_{\text{zero\_acc}} + r_{\text{zero\_ang}}$$

**Penalty（二次惩罚）：**

$$r_{\text{acc}} = -w_{\text{acc}} \cdot \|a_{\text{lin}}\|^2$$

$$r_{\text{ang}} = -w_{\text{ang}} \cdot \|a_{\text{ang}}\|^2$$

$$r_{\text{tilt}} = -w_{\text{tilt}} \cdot \|\text{tilt}_{xy}\|^2$$

$$r_{\text{smooth}} = -w_{\text{smooth}} \cdot \|\tau_t - \tau_{t-1}\|^2$$

$$r_{\text{reg}} = -w_{\text{reg}} \cdot \|\tau_t\|^2$$

$$\text{penalty} = r_{\text{acc}} + r_{\text{ang}} + r_{\text{tilt}} + r_{\text{smooth}} + r_{\text{reg}}$$

### 8.4 安全惩罚 r_safe

| 组成 | 公式 | 说明 |
|------|------|------|
| 无效碰撞 | `β_collision · Σ(non-affordance contact forces)` | 非目标 link 的接触力之和 |
| 自碰撞 | `β_self` (常数) | 检测到自碰撞时 |
| 关节限位 | `β_limit · Σ(max(0, |q - center| - μ·half_range)²)` | 接近关节极限时 |
| 速度限位 | `β_vel · Σ(max(0, |dq| - μ·vel_limit)²)` | 关节速度过大时 |
| 杯体掉落 | `w_drop` (常数) + **立即终止 episode** | 杯子脱落时 |

### 8.5 动态缩放 s_t（Stability Penalty Annealing）

```python
progress = min(1.0, global_step / n_anneal)
s_t = s_min + (1.0 - s_min) * progress
```

**设计意图**：训练初期 `s_t ≈ s_min`（较小），稳定性惩罚较轻，让策略先学会基本接触；
随着训练推进 `s_t → 1.0`，惩罚逐渐加重，迫使策略学会柔顺持杯。

```
s_t
 1.0 ─────────────────────────────────────────── ●
                                              ╱
                                           ╱
                                        ╱
 s_min ●──────────────────────────────╱
       0              n_anneal           global_step
```

---

## 9. 轨迹采集（Rollout Collection）

### 9.1 RolloutCollector.collect() 流程

```python
def collect(envs, n_steps=T, current_actor_obs, current_critic_obs):
    """在 N 个并行环境中同时采集 T 步轨迹数据。"""

    hidden = actor.init_hidden(n_envs)  # (layers, N, H)

    for t in range(T):
        # 1. 展平观测为分支张量
        actor_branches = batch_flatten(current_actor_obs)
        # → {"proprio": (N, 60), "ee": (N, 26), ... "visual": (N, 768)}

        priv_flat = batch_flatten_priv(current_critic_obs)
        # → (N, 30)

        # 2. 缓存当前隐状态（用于 TBPTT）
        cached_hidden = hidden.detach().clone()  # (layers, N, H)

        # 3. Actor 前向（采样动作）
        action, log_prob, _, hidden_new = actor.forward(
            actor_branches, hidden
        )
        # action: (N, 12), log_prob: (N,)

        # 4. Critic 前向（估计价值）
        value = critic.forward(actor_branches, priv_flat).squeeze(-1)
        # value: (N,)

        # 5. 环境步进
        next_obs, next_cobs, rewards, dones, infos = envs.step(action.numpy())

        # 6. 写入 Buffer
        buffer.add(t,
            actor_obs_branches=actor_branches,
            privileged_flat=priv_flat,
            actions=action,
            log_probs=log_prob,
            values=value,
            rewards=rewards,
            dones=dones,
            hidden_states=cached_hidden,
        )

        # 7. 更新隐状态（done 的环境清零）
        hidden = hidden_new
        hidden[:, dones.bool(), :] = 0.0

        # 8. 推进到下一步观测
        current_actor_obs = next_obs

    # 9. 计算 bootstrap 值（用于 GAE）
    last_values = critic.forward(
        batch_flatten(current_actor_obs),
        batch_flatten_priv(current_critic_obs),
    ).squeeze(-1)
```

### 9.2 Buffer 存储布局

所有张量按 **time-major** 格式存储：`(n_steps, n_envs, ...)`

```
                    时间维度 (T)
                    ─────────────▶
              ┌──────────────────────┐
    环境 0    │ t=0  t=1  t=2 ... T-1│
    环境 1    │ t=0  t=1  t=2 ... T-1│
    ...       │ ...                   │
    环境 N-1  │ t=0  t=1  t=2 ... T-1│
              └──────────────────────┘

存储张量:
  actor_obs_branches[name]: (T, N, branch_dim)
  privileged_obs:           (T, N, priv_dim)
  actions:                  (T, N, 12)
  log_probs:                (T, N)
  values:                   (T, N)
  rewards:                  (T, N)
  dones:                    (T, N)
  hidden_states:            (T, layers, N, H)
  advantages:               (T, N)    ← compute_gae 后填充
  returns:                  (T, N)    ← compute_gae 后填充
```

---

## 10. GAE 优势函数估计

### 10.1 GAE 公式

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}$$

$$R_t = \hat{A}_t + V(s_t)$$

### 10.2 反向递推实现

```python
def compute_gae(gamma, lam, last_values, last_dones):
    last_gae = 0  # shape: (N,)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values       # bootstrap
            next_non_terminal = 1.0 - last_dones
        else:
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
```

### 10.3 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子，控制对未来奖励的重视程度 |
| `lam` | 0.95 | GAE 偏差-方差权衡系数（越大方差越高但偏差越低） |

---

## 11. PPO 参数更新

### 11.1 TBPTT 序列切分

轨迹数据 `(T, N, ...)` 被切分为长度 `L` 的连续序列片段：

```
原始轨迹 (T=128 步, N=64 环境):

环境 0: [---seg0---][---seg1---][---seg2---][---seg3---]
环境 1: [---seg0---][---seg1---][---seg2---][---seg3---]
  ...
环境 63:[---seg0---][---seg1---][---seg2---][---seg3---]

总片段数 = (T / L) × N = (128/16) × 64 = 512
每片段: (L=16, branch_dim)

随机打乱后分为 num_mini_batches 个 mini-batch:
mini_batch_size = 512 / 4 = 128 个序列

每个 mini-batch 的张量形状:
  actor 分支:  (128, 16, branch_dim)
  privileged:  (128, 16, priv_dim)
  actions:     (128, 16, 12)
  log_probs:   (128, 16)
  advantages:  (128, 16)
  hidden_init: (layers, 128, H)  ← 每个序列起始时的 RNN 隐状态
```

### 11.2 单个 Mini-Batch 的更新步骤

```python
def _update_step(batch):
    B, L = batch["actions"].shape[:2]  # batch_size, seq_length

    # ═══ 1. 优势标准化 ═══
    if normalize_advantages:
        adv = (advantages - mean) / (std + 1e-8)

    # ═══ 2. TBPTT: 逐时间步重新计算 Actor 输出 ═══
    hidden = batch["hidden_init"].detach()    # (layers, B, H)
    all_log_probs, all_entropies = [], []

    for t in range(L):
        obs_t = {name: batch[name][:, t, :] for name in branches}
        act_t = batch["actions"][:, t, :]

        log_prob_t, entropy_t, hidden = actor.evaluate_actions(
            obs_t, act_t, hidden
        )
        all_log_probs.append(log_prob_t)
        all_entropies.append(entropy_t)

    new_log_probs = stack(all_log_probs, dim=1)  # (B, L)
    entropies = stack(all_entropies, dim=1)       # (B, L)

    # ═══ 3. PPO-Clip Actor Loss ═══
    ratio = exp(new_log_probs - old_log_probs)
    surr1 = ratio × advantages
    surr2 = clamp(ratio, 1-ε, 1+ε) × advantages
    L_actor = -min(surr1, surr2).mean()

    # ═══ 4. Entropy Bonus ═══
    L_entropy = entropies.mean()

    # ═══ 5. Critic Value Loss (可选 clipped) ═══
    # 展平时间维: (B, L, ...) → (B×L, ...)
    new_values = critic.forward(flat_branches, flat_priv)  # (B×L,)

    if use_clipped_value_loss:
        v_clipped = old_values + clamp(new_values - old_values, -ε_v, ε_v)
        L_critic = 0.5 × max((new_values - returns)²,
                              (v_clipped - returns)²).mean()
    else:
        L_critic = 0.5 × (new_values - returns)².mean()

    # ═══ 6. 总损失与梯度更新 ═══
    L_total = L_actor + c_value × L_critic - c_entropy × L_entropy

    L_total.backward()
    clip_grad_norm_(actor.parameters(), max_grad_norm)
    clip_grad_norm_(critic.parameters(), max_grad_norm)
    actor_optimizer.step()
    critic_optimizer.step()
```

### 11.3 PPO 超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `clip_eps` (ε) | 0.2 | PPO 策略比率裁剪范围 |
| `value_clip_eps` (ε_v) | 0.2 | Value function 裁剪范围 |
| `entropy_coef` (c_entropy) | 0.01 | 熵正则化系数 |
| `value_coef` (c_value) | 0.5 | Critic loss 权重 |
| `max_grad_norm` | 1.0 | 梯度裁剪最大范数 |
| `actor_lr` | 3e-4 | Actor 学习率 (Adam) |
| `critic_lr` | 3e-4 | Critic 学习率 (Adam) |
| `num_mini_batches` | 4 | Mini-batch 划分数 |
| `num_epochs` (K) | 5 | 每次更新的 epoch 数 |
| `seq_length` (L) | 16 | TBPTT 截断长度 |

### 11.4 监控指标

每次 `update()` 调用返回以下指标：

| 指标 | 公式 | 含义 |
|------|------|------|
| `actor_loss` | PPO-Clip surrogate 均值 | 策略损失 |
| `critic_loss` | Value loss 均值 | 价值估计误差 |
| `entropy` | 策略熵均值 | 探索程度 |
| `clip_fraction` | `mean(|ratio - 1| > ε)` | 被裁剪的样本比例 |
| `approx_kl` | `mean((ratio - 1) - log(ratio))` | 近似 KL 散度 |
| `explained_variance` | `1 - Var(returns - values) / Var(returns)` | 价值函数解释能力 |

---

## 12. 课程管理（Curriculum）

### 12.1 五阶段课程设计

```
Stage 1 ──▶ Stage 2 ──▶ Stage 3 ──▶ Stage 4 ──▶ Stage 5
 基础接触     持杯力控    多 affordance  时序子任务    全域泛化
```

| 阶段 | 持杯概率 | 门类型 | 训练目标 |
|------|----------|--------|----------|
| **Stage 1** | 0% | push | 跑通网络闭环，学会基础视觉引导接触 |
| **Stage 2** | 100% | push | 在 `r_stab` 和 `s_t` 约束下学会力控 |
| **Stage 3** | 50% | push, pull | 视觉区分 affordance 类型，调整接触策略 |
| **Stage 4** | 50% | handle_push, handle_pull | 学习时序子任务组合，RNN 跨越 reward delay |
| **Stage 5** | 50% | push, pull, handle_push, handle_pull | 高强度域随机化下的全域泛化 |

### 12.2 跃迁条件

使用**滑动窗口平均成功率**判断是否进入下一阶段：

$$\frac{1}{M} \sum_{e=E-M+1}^{E} \eta_e \geq \eta_{\text{thresh}}$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `window_size` (M) | 50 | 滑动窗口包含的 epoch 数 |
| `threshold` (η_thresh) | 0.8 | 成功率跃迁阈值 |

```python
# 每个 epoch 结束时：
advanced = curriculum.report_epoch(success_rate)
if advanced:
    # 清空滑动窗口
    # 切换到下一阶段的 StageConfig
    # 下次 env.reset() 时使用新阶段的参数
```

### 12.3 课程如何影响训练

每个阶段的 `StageConfig` 决定了环境 reset 时的采样策略：

```python
stage_config = curriculum.get_stage_config()

# 决定是否持杯
left_occupied = random() < stage_config.cup_probability

# 决定门类型
door_type = random.choice(stage_config.door_types)

# 传递给环境 reset
env.reset(
    door_type=door_type,
    left_occupied=left_occupied,
    domain_params=randomizer.sample_episode_params(),
)
```

---

## 13. 域随机化（Domain Randomization）

### 13.1 Episode 级参数（reset 时采样，整个 episode 固定）

| 参数 | 随机化范围 | 单位 |
|------|-----------|------|
| `cup_mass` | [0.1, 0.8] | kg |
| `door_mass` | [5.0, 20.0] | kg |
| `door_damping` | [0.5, 5.0] | N·m·s/rad |
| `base_pos` (XY) | nominal ± 0.03 | m |

### 13.2 Step 级噪声（每步独立采样）

| 噪声 | 分布 | 标准差 |
|-------|------|--------|
| 动作噪声 `ε_a` | `N(0, σ_a²I)` | σ_a = 0.02 |
| 观测噪声 `ε_o` | `N(0, σ_o²I)` | σ_o = 0.01 |

### 13.3 随机化与课程的关系

```
Stage 1-2: 基础随机化范围
Stage 3-4: 中等随机化 + 更多门类型
Stage 5:   高强度域随机化（范围可能进一步扩大）
```

---

## 14. 训练指标与日志

### 14.1 TrainingMetrics 聚合器

```python
# ── 每个 episode 结束时 ──
metrics.update_episode(
    success=bool,
    cup_dropped=bool,
    episode_length=int,
    reward_info=dict,    # 来自 RewardManager.step() 的分项信息
)

# ── 每次 PPO 更新后 ──
metrics.update_ppo(
    actor_loss=float,
    critic_loss=float,
    entropy=float,
    clip_fraction=float,
    approx_kl=float,
    explained_variance=float,
)

# ── 记录日志时 ──
summary = metrics.summarize()
# 返回扁平化 dict，可直接写入 TensorBoard / WandB:
# {
#   "episode/success_rate":     0.75,
#   "episode/cup_drop_rate":    0.02,
#   "episode/count":            128.0,
#   "episode/mean_length":      450.0,
#   "reward/task_progress":     12.5,
#   "reward/stability_bonus":   0.8,
#   "reward/safety_penalty":    -1.2,
#   "ppo/actor_loss":           0.03,
#   "ppo/critic_loss":          0.12,
#   "ppo/entropy":              2.1,
#   "ppo/clip_fraction":        0.08,
#   "ppo/approx_kl":            0.015,
#   "ppo/explained_variance":   0.85,
# }
```

### 14.2 关键监控信号

| 信号 | 健康范围 | 异常应对 |
|------|----------|----------|
| `success_rate` | 逐步上升 → 0.8 触发跃迁 | 长期卡在低值：检查奖励设计或降低课程难度 |
| `cup_drop_rate` | 持杯阶段后应 < 0.05 | 过高：加大 `w_drop` 或 `s_t` 退火速度 |
| `clip_fraction` | 0.05~0.15 | >0.3：学习率过大或 `clip_eps` 过小 |
| `approx_kl` | < 0.02 | >0.05：策略变化过快，降低学习率 |
| `explained_variance` | 趋近 1.0 | < 0.5：Critic 学习不足或 privileged info 不够 |
| `entropy` | 缓慢下降 | 过快下降：增大 `entropy_coef` |

---

## 15. Checkpoint 与恢复

### 15.1 需要保存的状态

```python
checkpoint = {
    # 模型权重
    "actor_state_dict": actor.state_dict(),
    "critic_state_dict": critic.state_dict(),

    # 优化器状态
    "ppo_state_dict": ppo.state_dict(),
    # → {"actor_optimizer": ..., "critic_optimizer": ...}

    # 课程进度
    "curriculum_state_dict": curriculum.state_dict(),
    # → {"current_idx": 2, "success_window": [...], "total_epochs": 150}

    # 训练进度
    "iteration": iteration,
    "global_step": reward_manager.global_step,
}
```

### 15.2 恢复训练

```python
checkpoint = torch.load(path)
actor.load_state_dict(checkpoint["actor_state_dict"])
critic.load_state_dict(checkpoint["critic_state_dict"])
ppo.load_state_dict(checkpoint["ppo_state_dict"])
curriculum.load_state_dict(checkpoint["curriculum_state_dict"])
start_iteration = checkpoint["iteration"]
```

---

## 16. 完整训练时序图

以下时序图展示了**一个训练 iteration** 中各模块的调用顺序：

```
时间 ──────────────────────────────────────────────────────────────────▶

                    Phase 1: Rollout                Phase 2    Phase 3: PPO Update
              ┌──────────────────────────┐        ┌──────┐  ┌──────────────────┐
              │                          │        │      │  │                  │
    for t in range(T):                            │      │  │ for epoch in K:  │
              │                          │        │      │  │   for mb in M:   │
    ┌─────────┼──────────────────────────┼────────┼──────┼──┼──────────────────┼──────┐
    │         │                          │        │      │  │                  │      │
    │  Perception                        │        │      │  │                  │      │
    │  ├ LangSAM.segment()               │        │      │  │                  │      │
    │  ├ backproject_depth()              │        │      │  │                  │      │
    │  └ PointMAE.encode() → z_aff       │        │      │  │                  │      │
    │         │                          │        │      │  │                  │      │
    │  ObsBuilder                        │        │      │  │                  │      │
    │  ├ ActorObsBuilder.build()         │        │      │  │                  │      │
    │  │  ├ StabilityProxy.estimate()    │        │      │  │                  │      │
    │  │  └ HistoryBuffer.append()       │        │      │  │                  │      │
    │  └ CriticObsBuilder.build()        │        │      │  │                  │      │
    │         │                          │        │      │  │                  │      │
    │  batch_flatten → actor_branches    │        │      │  │  TBPTT Actor     │      │
    │         │                          │        │      │  │  ├ for t in L:   │      │
    │  Actor.forward()                   │        │      │  │  │  evaluate()   │      │
    │  ├ branch encoders                 │        │      │  │  └ → new logπ   │      │
    │  ├ RecurrentBackbone               │        │      │  │                  │      │
    │  └ ActionHead.sample()             │        │      │  │  PPO-Clip Loss  │      │
    │         │ action (N,12)            │        │      │  │  Critic Loss     │      │
    │         │                          │        │      │  │  Entropy Bonus   │      │
    │  Critic.forward()                  │        │      │  │         │        │      │
    │  └ → V(s) (N,)                     │        │      │  │  Backprop +      │      │
    │         │                          │        │      │  │  Gradient Clip   │      │
    │  Env.step(action)                  │        │      │  │  Optimizer.step()│      │
    │  ├ _sim_step() × decimation        │        │      │  │                  │      │
    │  ├ _read_physics_state()           │        │      │  └──────────────────┘      │
    │  ├ ContactMonitor.update()         │        │      │                             │
    │  ├ TaskManager.update()            │        │      │  Phase 4: Metrics           │
    │  └ RewardManager.step()            │        │      │  ├ metrics.summarize()      │
    │     → (reward, done, info)         │        │      │  ├ curriculum.report_epoch()│
    │         │                          │        │      │  └ save_checkpoint()        │
    │  Buffer.add(t, ...)                │  GAE   │      │                             │
    │         │                          │ compute│      │                             │
    └─────────┼──────────────────────────┼────────┼──────┼─────────────────────────────┘
              │                          │        │      │
              └──────────────────────────┘        └──────┘

    ──────── 一个 iteration 完成，buffer.clear()，进入下一轮 ────────
```

---

## 附录 A: 关键模块文件索引

| 模块 | 文件路径 | 核心类/函数 |
|------|----------|-------------|
| 环境层 | `envs/door_env.py` | `DoorInteractionEnv` |
| 场景工厂 | `envs/scene_factory.py` | `SceneFactory` |
| 接触监控 | `envs/contact_monitor.py` | `ContactMonitor` |
| 任务管理 | `envs/task_manager.py` | `TaskManager` |
| 向量化环境 | `envs/vec_env.py` | `VecDoorEnv` |
| Actor 观测 | `observations/actor_obs_builder.py` | `ActorObsBuilder` |
| Critic 观测 | `observations/critic_obs_builder.py` | `CriticObsBuilder` |
| 稳定性代理 | `observations/stability_proxy.py` | `estimate_stability_proxy` |
| 感知管线 | `door_perception/affordance_pipeline.py` | `AffordancePipeline` |
| 冻结编码器 | `door_perception/frozen_encoder.py` | `PointMAEEncoder` |
| Actor 网络 | `policy/actor.py` | `Actor` |
| Critic 网络 | `policy/critic.py` | `Critic` |
| 循环主干 | `policy/recurrent_backbone.py` | `RecurrentBackbone` |
| 动作头 | `policy/action_head.py` | `ActionHead` |
| PPO 训练器 | `training/ppo_trainer.py` | `PPOTrainer` |
| 轨迹缓冲 | `training/rollout_buffer.py` | `RolloutBuffer` |
| 轨迹采集 | `training/rollout_collector.py` | `RolloutCollector` |
| 课程管理 | `training/curriculum_manager.py` | `CurriculumManager` |
| 域随机化 | `training/domain_randomizer.py` | `DomainRandomizer` |
| 奖励管理 | `rewards/reward_manager.py` | `RewardManager` |
| 任务奖励 | `rewards/task_reward.py` | `compute_task_reward` |
| 稳定性奖励 | `rewards/stability_reward.py` | `compute_stability_reward` |
| 安全惩罚 | `rewards/safety_penalty.py` | `compute_safety_penalty` |
| 训练指标 | `training/metrics.py` | `TrainingMetrics` |

## 附录 B: 张量维度速查表

| 张量 | 形状 | 存在于 |
|------|------|--------|
| `action` | `(N, 12)` | 环境交互、Buffer |
| `actor_obs_branches["proprio"]` | `(N, ~60)` | Buffer `(T, N, 60)` |
| `actor_obs_branches["ee"]` | `(N, ~26)` | Buffer `(T, N, 26)` |
| `actor_obs_branches["context"]` | `(N, 2)` | Buffer `(T, N, 2)` |
| `actor_obs_branches["stability"]` | `(N, ~24)` | Buffer `(T, N, 24)` |
| `actor_obs_branches["visual"]` | `(N, 768)` | Buffer `(T, N, 768)` |
| `privileged_flat` | `(N, ~30)` | Buffer `(T, N, 30)` |
| `hidden_states` | `(layers, N, H)` | Buffer `(T, layers, N, H)` |
| `values` | `(N,)` | Buffer `(T, N)` |
| `rewards` | `(N,)` | Buffer `(T, N)` |
| `advantages` | — | Buffer `(T, N)` |
| `returns` | — | Buffer `(T, N)` |
| **TBPTT mini-batch** | | |
| `batch[name]` | `(B, L, dim)` | PPO update |
| `batch["hidden_init"]` | `(layers, B, H)` | PPO update |
| `batch["actions"]` | `(B, L, 12)` | PPO update |
| `batch["log_probs"]` | `(B, L)` | PPO update |
| `batch["advantages"]` | `(B, L)` | PPO update |
| `batch["returns"]` | `(B, L)` | PPO update |

> 其中：`N` = 并行环境数，`T` = rollout 步数，`B` = mini-batch 中的序列数，
> `L` = TBPTT 截断长度，`H` = RNN 隐层维度，`layers` = RNN 层数。
