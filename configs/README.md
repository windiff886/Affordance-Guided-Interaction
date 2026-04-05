# configs/ — 配置参数总览与审计报告

> **审计日期**：2026-04-04
>
> 本文档整合了 6 个 YAML 配置文件的全部参数，标注每个参数是否在代码中实际消费，
> 并索引到各模块 README 中的数学定义与设计文档。

---

## 1. 配置文件清单

| 文件 | 参数数 | 代码消费率 | 对应模块 README |
|------|--------|-----------|----------------|
| `training/default.yaml` | 21 | **100%** | `training/README.md` §7 |
| `reward/default.yaml` | 22 | **100%** | `rewards/README.md` §9 |
| `env/default.yaml` | 13 | 54% (7/13) | `envs/README.md` §6 |
| `policy/default.yaml` | 19 | 84% (16/19) | `policy/README.md` §3 |
| `curriculum/default.yaml` | 4 | 75% (3/4) | `training/README.md` §4.3 |
| `task/default.yaml` | 10 | 20% (2/10) | `envs/README.md` §6.2 |

**整体**：89 个参数，69 个被代码消费（77.5%），20 个未被代码引用。

---

## 2. 配置加载机制

`scripts/train.py` 中的 `load_config()` 函数加载 6 个 YAML 文件，返回字典：

```python
cfg = {
    "training":   yaml.safe_load("configs/training/default.yaml"),
    "env":        yaml.safe_load("configs/env/default.yaml"),
    "policy":     yaml.safe_load("configs/policy/default.yaml"),
    "task":       yaml.safe_load("configs/task/default.yaml"),
    "curriculum":  yaml.safe_load("configs/curriculum/default.yaml"),
    "reward":     yaml.safe_load("configs/reward/default.yaml"),
}
```

**访问模式**：
- `train.py`：字典 `.get()` → `cfg["training"].get("num_envs", 4)`
- 模块内部：转为 dataclass 属性 → `self.cfg.gamma`
- 奖励模块：保持嵌套字典 → `self._cfg_scale["s_min"]`

---

## 3. 各文件参数详细审计

### 3.1 `training/default.yaml` — 训练管线超参数 (21 参数, 100% 消费)

> **README 索引**：`src/affordance_guided_interaction/training/README.md`

| YAML 键路径 | 默认值 | 代码消费 | README 章节 | 数学符号 | 含义 |
|---|---|---|---|---|---|
| `algorithm` | `ppo` | ✅ train.py | §2 | — | 训练算法选择 |
| `num_envs` | `64` | ✅ train.py | §7.2 | $N_{\text{env}}$ | 并行环境数量 |
| `total_steps` | `10_000_000` | ✅ train.py | — | — | 总训练步数 |
| `n_steps_per_rollout` | `128` | ✅ train.py → rollout_collector | §7.2 | $T$ | 每轮 rollout 采集步数 |
| `ppo.gamma` | `0.99` | ✅ ppo_trainer, rollout_buffer | §7.1 | $\gamma$ | 折扣因子 |
| `ppo.lam` | `0.95` | ✅ rollout_buffer (GAE) | §7.1 | $\lambda$ | GAE 偏差-方差权衡 |
| `ppo.clip_eps` | `0.2` | ✅ ppo_trainer | §7.1 | $\epsilon$ | PPO-Clip 策略比率裁剪 |
| `ppo.value_clip_eps` | `0.2` | ✅ ppo_trainer | §7.1 | $\epsilon_v$ | Value function 裁剪 |
| `ppo.use_clipped_value_loss` | `true` | ✅ ppo_trainer | §2.4 | — | 是否使用 clipped value loss |
| `ppo.entropy_coef` | `0.01` | ✅ ppo_trainer | §7.1 | $c_e$ | 熵正则化系数 |
| `ppo.value_coef` | `0.5` | ✅ ppo_trainer | §7.1 | $c_v$ | Critic 损失权重 |
| `ppo.max_grad_norm` | `1.0` | ✅ ppo_trainer | §7.1 | $g_{\max}$ | 全局梯度范数上限 |
| `ppo.actor_lr` | `3.0e-4` | ✅ train.py → optimizer | §7.1 | $\alpha_\theta$ | Actor 学习率 |
| `ppo.critic_lr` | `3.0e-4` | ✅ train.py → optimizer | §7.1 | $\alpha_\phi$ | Critic 学习率 |
| `ppo.num_mini_batches` | `4` | ✅ rollout_buffer | §7.1 | $N_{\text{mb}}$ | Mini-batch 数量 |
| `ppo.num_epochs` | `5` | ✅ train.py 主循环 | §7.1 | $K$ | 优化轮数 |
| `ppo.seq_length` | `16` | ✅ rollout_buffer | §7.2 | $L$ | TBPTT 截断长度 |
| `ppo.normalize_advantages` | `true` | ✅ ppo_trainer | — | — | 是否标准化优势值 |
| `log_interval` | `1` | ✅ train.py | — | — | 日志打印间隔 |
| `checkpoint_interval` | `50` | ✅ train.py | — | — | Checkpoint 保存间隔 |
| `eval_interval` | `10` | ✅ train.py | — | — | 评估执行间隔（预留） |

---

### 3.2 `reward/default.yaml` — 奖励函数超参数 (22 参数, 100% 消费)

> **README 索引**：`src/affordance_guided_interaction/rewards/README.md`

#### 主任务权重 (§9.1)

| YAML 键路径 | 默认值 | 代码消费 | 数学符号 | 含义 |
|---|---|---|---|---|
| `task.w_delta` | `8.0` | ✅ task_reward.py | $w_\delta$ | 角度增量基准奖励 |
| `task.alpha` | `0.25` | ✅ task_reward.py | $\alpha$ | 超出目标后权重衰减下限比例 |
| `task.k_decay` | `2.0` | ✅ task_reward.py | $k_{\text{decay}}$ | 超出目标角度后的衰减速率 |
| `task.w_open` | `10.0` | ✅ task_reward.py | $w_{\text{open}}$ | 任务成功一次性 bonus |
| `task.theta_target` | `1.2` | ✅ task_reward.py | $\theta_{\text{reward\_success}}$ | Success bonus 触发角度 (rad) |

**重要区分**：此处 `theta_target = 1.2 rad` 是**奖励层 bonus 触发角度**，与 `task/default.yaml` 中的 `door_angle_target = 1.57 rad`（episode 成功终止角度）是两个独立阈值。详见 `rewards/README.md` §4.2。

#### 稳定性权重 (§9.2, SoFTA 对齐)

| YAML 键路径 | 默认值 | 代码消费 | 数学符号 | 含义 |
|---|---|---|---|---|
| `stability.w_zero_acc` | `1.0` | ✅ stability_reward.py | $w_{\text{zero-acc}}$ | 零线加速度高斯核正奖励权重 |
| `stability.lambda_acc` | `0.25` | ✅ stability_reward.py | $\lambda_{acc}$ | 线加速度高斯核衰减率 |
| `stability.w_zero_ang` | `1.5` | ✅ stability_reward.py | $w_{\text{zero-ang}}$ | 零角加速度高斯核正奖励权重 |
| `stability.lambda_ang` | `0.0044` | ✅ stability_reward.py | $\lambda_{ang}$ | 角加速度高斯核衰减率 |
| `stability.w_acc` | `0.10` | ✅ stability_reward.py | $w_{\text{acc}}$ | 线加速度二次惩罚系数 |
| `stability.w_ang` | `0.01` | ✅ stability_reward.py | $w_{\text{ang}}$ | 角加速度二次惩罚系数 |
| `stability.w_tilt` | `5.0` | ✅ stability_reward.py | $w_{\text{tilt}}$ | 重力倾斜惩罚系数 |
| `stability.w_smooth` | `0.01` | ✅ stability_reward.py | $w_{\text{smooth}}$ | 力矩变化平滑惩罚系数 |
| `stability.w_reg` | `0.001` | ✅ stability_reward.py | $w_{\text{reg}}$ | 力矩幅值正则惩罚系数 |

**奖励结构**：子项 (1)(2) 为正向激励（$r_{\text{stab\_bonus}}$），子项 (3)-(7) 为负向惩罚（$r_{\text{stab\_penalty}}$），后者受动态缩放因子 $s_t$ 调制。详见 `rewards/README.md` §5。

#### 安全惩罚权重 (§9.3)

| YAML 键路径 | 默认值 | 代码消费 | 数学符号 | 含义 |
|---|---|---|---|---|
| `safety.beta_self` | `1.0` | ✅ safety_penalty.py | $\beta_1$ | 自碰撞固定惩罚 |
| `safety.beta_limit` | `0.1` | ✅ safety_penalty.py | $\beta_2$ | 关节限位系数 |
| `safety.mu` | `0.9` | ✅ safety_penalty.py | $\mu$ | 关节限位 / 限速触发比例 |
| `safety.beta_vel` | `0.01` | ✅ safety_penalty.py | $\beta_3$ | 关节速度系数 |
| `safety.beta_torque` | `0.01` | ✅ safety_penalty.py | $\beta_4$ | 原始控制力矩超限系数 |
| `safety.w_drop` | `100.0` | ✅ safety_penalty.py | $w_{\text{drop}}$ | 杯体脱落惩罚权重 |

#### 动态缩放参数 (§7.2, 全局退火)

| YAML 键路径 | 默认值 | 代码消费 | 数学符号 | 含义 |
|---|---|---|---|---|
| `scaling.s_min` | `0.1` | ✅ reward_manager.py | $s_{\min}$ | 退火初始惩罚基数 |
| `scaling.n_anneal` | `10_000_000` | ✅ reward_manager.py | $N_{\text{anneal}}$ | 退火窗口期（全局步数） |

**退火公式**：$s_t = s_{\min} + (1.0 - s_{\min}) \cdot \min(1.0, N_{\text{step}} / N_{\text{anneal}})$

---

### 3.3 `env/default.yaml` — 环境配置 (13 参数, 7 消费 / 6 未消费)

> **README 索引**：`src/affordance_guided_interaction/envs/README.md`

| YAML 键路径 | 默认值 | 代码消费 | README 章节 | 含义 |
|---|---|---|---|---|
| `name` | `door_interaction_env` | ❌ 未消费 | — | 环境名称标识（仅文档用途） |
| `sim_backend` | `isaac_lab` | ❌ 未消费 | — | 仿真后端标识（仅文档用途） |
| `physics_dt` | `0.008333` | ✅ base_env, scene_factory, door_env | §6.1 | 物理步长 $\Delta t = 1/120$ s |
| `decimation` | `2` | ✅ base_env, door_env | §6.1 | 控制频率 = $\Delta t \times$ decimation = 60Hz |
| `max_episode_steps` | `500` | ✅ task_manager | §6.1 | 最大 episode 步数（$T_{\max}$） |
| `robot` | `unitree_z1_dual` | ❌ 未消费 | — | 机器人型号标识（仅文档用途） |
| `joints_per_arm` | `6` | ✅ base_env | — | 每条 Z1 臂关节数 |
| `total_joints` | `12` | ✅ base_env, door_env | — | 双臂总关节数 |
| `contact_force_threshold` | `0.1` | ✅ door_env → contact_monitor | §6.2 | 接触力判定阈值 $f_{\text{thresh}}$ (N) |
| `action_history_length` | `3` | ✅ door_env, actor_obs_builder | — | 动作历史缓存步数 |
| `acc_history_length` | `10` | ✅ door_env, stability_proxy | — | 加速度历史窗口长度 |
| `assets.robot_usd` | 路径字符串 | ❌ 未消费 | — | 机器人 USD 路径（代码中硬编码） |
| `assets.push_door_usd` | 路径字符串 | ❌ 未消费 | — | 推门 USD 路径（代码中硬编码） |
| `assets.cup_usd` | 路径字符串 | ❌ 未消费 | — | 杯体 USD 路径（代码中硬编码） |

**未消费参数说明**：
- `name`、`sim_backend`、`robot`：元数据标识，方便人类阅读，代码不依赖
- `assets.*`：asset 路径当前在 `scene_factory.py` 和 `load_scene.py` 中硬编码，未从 YAML 读取。建议后续改为从配置文件读取以提高可维护性

---

### 3.4 `policy/default.yaml` — 策略网络配置 (19 参数, 16 消费 / 3 未消费)

> **README 索引**：`src/affordance_guided_interaction/policy/README.md`

#### Actor 网络

| YAML 键路径 | 默认值 | 代码消费 | 含义 |
|---|---|---|---|
| `actor.type` | `recurrent` | ❌ 未消费 | 网络类型标识（仅文档用途） |
| `actor.proprio_hidden` | `128` | ✅ actor.py | 本体感受 encoder 隐层维度 |
| `actor.proprio_out` | `64` | ✅ actor.py | 本体感受 encoder 输出维度 |
| `actor.ee_hidden` | `64` | ✅ actor.py | 末端执行器 encoder 隐层 |
| `actor.ee_out` | `32` | ✅ actor.py | 末端执行器 encoder 输出 |
| `actor.stab_hidden` | `64` | ✅ actor.py | 稳定性 proxy encoder 隐层 |
| `actor.stab_out` | `32` | ✅ actor.py | 稳定性 proxy encoder 输出 |
| `actor.vis_hidden` | `256` | ✅ actor.py | 视觉 encoder 隐层 |
| `actor.vis_out` | `128` | ✅ actor.py | 视觉 encoder 输出 |
| `actor.rnn_hidden` | `512` | ✅ actor.py → recurrent_backbone | GRU 隐状态维度 |
| `actor.rnn_layers` | `1` | ✅ actor.py → recurrent_backbone | GRU 层数 |
| `actor.rnn_type` | `gru` | ✅ actor.py → recurrent_backbone | 循环类型 (gru / lstm) |
| `actor.action_dim` | `12` | ✅ actor.py → action_head | 双臂关节力矩维度 |
| `actor.log_std_init` | `-0.5` | ✅ action_head.py | 高斯策略 $\log\sigma$ 初始值 |
| `actor.action_history_length` | `3` | ✅ actor_obs_builder | 动作历史步数 |
| `actor.acc_history_length` | `10` | ✅ stability_proxy | 加速度历史窗口 |
| `actor.include_torques` | `true` | ✅ actor.py, critic.py | 是否在 proprio 包含关节力矩 |

#### Critic 网络

| YAML 键路径 | 默认值 | 代码消费 | 含义 |
|---|---|---|---|
| `critic.type` | `asymmetric` | ❌ 未消费 | 网络类型标识（仅文档用途） |
| `critic.hidden_dims` | `[512, 256, 128]` | ✅ critic.py | MLP 各层维度 |
| `critic.share_actor_encoder` | `false` | ❌ 未消费 | 预留：是否共享 actor encoder 权重 |

**注**：`action_history_length` 和 `acc_history_length` 同时出现在 `env/default.yaml` 和 `policy/default.yaml` 中，代码实际通过 `train.py` 从 `policy/` 配置传入 actor 构造。两处定义应保持一致。

---

### 3.5 `curriculum/default.yaml` — 课程学习配置 (4 参数, 3 消费 / 1 未消费)

> **README 索引**：`src/affordance_guided_interaction/training/README.md` §4

| YAML 键路径 | 默认值 | 代码消费 | 数学符号 | 含义 |
|---|---|---|---|---|
| `initial_stage` | `stage_1` | ✅ curriculum_manager.py | — | 训练起始阶段 |
| `window_size` | `50` | ✅ curriculum_manager.py | $M$ | 滑动窗口长度（epoch 数） |
| `threshold` | `0.8` | ✅ curriculum_manager.py | $\eta_{\text{thresh}}$ | 跃迁成功率阈值 |
| `manual_override` | `false` | ❌ 未消费 | — | 预留：手动控制阶段跃迁 |

**三阶段定义**（硬编码于 `curriculum_manager.py`，YAML 中仅作注释记录）：

| 阶段 | 上下文分布 | 门类型 | 核心学习目标 | 跃迁条件 |
|------|------------|--------|------------|----------|
| Stage 1 | `none: 1.0` | push | 基础视觉引导接触，先跑通视觉-控制闭环 | 滑动窗口成功率 $\geq 0.8$ |
| Stage 2 | `left_only: 0.5, right_only: 0.5` | push | 在单臂持杯约束下学会稳定推门 | 滑动窗口成功率 $\geq 0.8$ |
| Stage 3 | `none / left_only / right_only / both` 各 `0.25` | push | 在最终混合分布下统一覆盖无杯、单臂持杯和双臂持杯 | 最终阶段 |

---

### 3.6 `task/default.yaml` — 任务定义配置 (10 参数, 2 消费 / 8 未消费)

> **README 索引**：`src/affordance_guided_interaction/envs/README.md` §6.2

| YAML 键路径 | 默认值 | 代码消费 | 含义 |
|---|---|---|---|
| `task_family` | `door_related_interaction` | ❌ 未消费 | 任务族标识（仅文档用途） |
| `default_task` | `push` | ❌ 未消费 | 默认任务类型（仅文档用途） |
| `door_angle_target` | `1.57` | ✅ task_manager.py | Episode 成功终止角度 $\theta_{\text{episode\_success}}$ (rad) |
| `door_angle_tolerance` | `0.05` | ❌ 未消费 | 预留：角度容差 |
| `cup_drop_threshold` | `0.15` | ✅ contact_monitor.py | 杯体脱落距离阈值 $\epsilon_{\text{drop}}$ (m) |
| `self_collision_penalty` | `true` | ❌ 未消费 | 预留：自碰撞检测开关（当前始终启用） |
| `affordance_types.push.*` | — | ❌ 未消费 | 预留：push affordance 描述 |
| `affordance_types.pull.*` | — | ❌ 未消费 | 预留：pull affordance 描述 |
| `affordance_types.handle_push.*` | — | ❌ 未消费 | 预留：handle_push 描述 |
| `affordance_types.handle_pull.*` | — | ❌ 未消费 | 预留：handle_pull 描述 |

**说明**：`task/default.yaml` 大量参数为**未来扩展预留**。当前只实现了 push 门任务，`affordance_types` 映射表为将来支持 pull/handle_push/handle_pull 等任务预埋。

---

## 4. 未消费参数汇总

### 4.1 文档标识类（无需修复，保留即可）

| 参数 | 所在文件 | 用途 |
|------|----------|------|
| `env.name` | env/default.yaml | 环境名称标识 |
| `env.sim_backend` | env/default.yaml | 仿真后端标识 |
| `env.robot` | env/default.yaml | 机器人型号标识 |
| `policy.actor.type` | policy/default.yaml | Actor 网络类型标识 |
| `policy.critic.type` | policy/default.yaml | Critic 网络类型标识 |
| `task.task_family` | task/default.yaml | 任务族标识 |
| `task.default_task` | task/default.yaml | 默认任务标识 |

### 4.2 未来扩展预留（无需修复，待实现时消费）

| 参数 | 所在文件 | 预期用途 |
|------|----------|----------|
| `curriculum.manual_override` | curriculum/default.yaml | 手动控制课程阶段跃迁 |
| `task.door_angle_tolerance` | task/default.yaml | 角度容差判定 |
| `task.self_collision_penalty` | task/default.yaml | 自碰撞检测开关 |
| `task.affordance_types.*` | task/default.yaml | 多任务类型描述映射 |
| `policy.critic.share_actor_encoder` | policy/default.yaml | Actor-Critic 权重共享 |

### 4.3 建议改进（资产路径应从配置读取）

| 参数 | 所在文件 | 当前状况 | 建议 |
|------|----------|----------|------|
| `env.assets.robot_usd` | env/default.yaml | 代码中硬编码路径 | `scene_factory.py` 应从配置读取 |
| `env.assets.push_door_usd` | env/default.yaml | 代码中硬编码路径 | 同上 |
| `env.assets.cup_usd` | env/default.yaml | 代码中硬编码路径 | 同上 |

---

## 5. 参数与 README 数学定义交叉索引

### 5.1 总奖励公式（`rewards/README.md` §2）

$$r_t = r_{\text{task}} + m_L \cdot (r_{\text{stab\_bonus}}^L + s_t \cdot r_{\text{stab\_penalty}}^L) + m_R \cdot (r_{\text{stab\_bonus}}^R + s_t \cdot r_{\text{stab\_penalty}}^R) - r_{\text{safe}}$$

涉及的配置参数：
- $s_t$ 由 `scaling.s_min` + `scaling.n_anneal` 控制
- $r_{\text{task}}$ 由 `task.*` 5 个参数控制
- $r_{\text{stab}}$ 由 `stability.*` 9 个参数控制
- $r_{\text{safe}}$ 由 `safety.*` 6 个参数控制

### 5.2 PPO 损失函数（`training/README.md` §2）

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{actor}}(\theta) + c_v \cdot \mathcal{L}_{\text{critic}}(\theta) - c_e \cdot \mathcal{H}[\pi_\theta]$$

涉及的配置参数：
- $c_v$ = `ppo.value_coef`
- $c_e$ = `ppo.entropy_coef`
- $\epsilon$ = `ppo.clip_eps`（PPO-Clip）
- $\epsilon_v$ = `ppo.value_clip_eps`（Value Clip）
- $g_{\max}$ = `ppo.max_grad_norm`（梯度裁剪）

### 5.3 GAE 优势估计（`training/README.md` §2.2）

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

涉及的配置参数：
- $\gamma$ = `ppo.gamma`
- $\lambda$ = `ppo.lam`

### 5.4 课程跃迁判据（`training/README.md` §4.3）

$$\frac{1}{M} \sum_{e=E-M+1}^{E} \eta_e \geq \eta_{\text{thresh}}$$

涉及的配置参数：
- $M$ = `curriculum.window_size`
- $\eta_{\text{thresh}}$ = `curriculum.threshold`

### 5.5 动态缩放退火（`rewards/README.md` §7.2）

$$s_t = s_{\min} + (1.0 - s_{\min}) \cdot \min\!\left(1.0, \frac{N_{\text{step}}}{N_{\text{anneal}}}\right)$$

涉及的配置参数：
- $s_{\min}$ = `scaling.s_min`
- $N_{\text{anneal}}$ = `scaling.n_anneal`

### 5.6 关键阈值区分

| 阈值 | 配置路径 | 值 | 语义 | 消费模块 |
|------|----------|-----|------|----------|
| $\theta_{\text{reward\_success}}$ | `reward.task.theta_target` | 1.2 rad | 奖励 bonus 触发角度 | `task_reward.py` |
| $\theta_{\text{episode\_success}}$ | `task.door_angle_target` | 1.57 rad | Episode 成功终止角度 | `task_manager.py` |

这两个阈值服务于不同目的：奖励层在 1.2 rad 时开始给予成功 bonus 并衰减权重，环境层在 1.57 rad 时才判定 episode 成功终止。

---

## 6. 配置文件间的依赖关系

```
training/default.yaml
    ├── 控制 train.py 主循环
    ├── PPO 参数 → ppo_trainer.py
    ├── Rollout 参数 → rollout_collector.py, rollout_buffer.py
    └── 与 reward/scaling.n_anneal 共享退火窗口期

reward/default.yaml
    ├── task.* → task_reward.py
    ├── stability.* → stability_reward.py
    ├── safety.* → safety_penalty.py
    └── scaling.* → reward_manager.py（$s_t$ 退火）

env/default.yaml
    ├── 物理参数 → scene_factory.py, door_env.py
    ├── 任务控制 → task_manager.py（max_episode_steps）
    └── 观测参数 → actor_obs_builder, stability_proxy

policy/default.yaml
    ├── actor.* → actor.py, recurrent_backbone.py, action_head.py
    └── critic.* → critic.py

curriculum/default.yaml
    └── → curriculum_manager.py（阶段跃迁）

task/default.yaml
    ├── door_angle_target → task_manager.py（episode 终止）
    └── cup_drop_threshold → contact_monitor.py（杯体脱落）
```

**关键协同**：
- `reward.scaling.n_anneal` 与 `training.total_steps` 应保持一致（当前均为 10M）
- `env.action_history_length` 与 `policy.actor.action_history_length` 应保持一致（当前均为 3）
- `env.acc_history_length` 与 `policy.actor.acc_history_length` 应保持一致（当前均为 10）

---

## 7. 域随机化参数

> **README 索引**：`src/affordance_guided_interaction/training/README.md` §5.2, §7.4

域随机化参数范围当前**硬编码**在 `training/domain_randomizer.py` 中，**未外置到 YAML 配置**。
记录于此供将来配置化参考：

| 参数 | 符号 | 范围 | 说明 |
|------|------|------|------|
| 杯体质量 | $m_{\text{cup}}$ | $[0.1, 0.8]$ kg | 模拟不同液体装载量 |
| 门板质量 | $m_{\text{door}}$ | $[5.0, 20.0]$ kg | 影响推门力矩需求 |
| 门铰链阻尼 | $d_{\text{hinge}}$ | $[0.5, 5.0]$ N·m·s/rad | 控制门运动阻力 |
| 基座位移 | $\Delta p$ | $[0.02, 0.05]$ m | XY 平面微扰 |
| 动作噪声 | $\sigma_a$ | $0.02$ | 步级高斯噪声 |
| 观测噪声 | $\sigma_o$ | $0.01$ | 步级高斯噪声 |

这些参数同时作为 Critic 的 privileged information（`observations/README.md` §5）。
