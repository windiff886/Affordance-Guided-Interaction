# 训练流程详解

本文档描述 `scripts/train.py` 从启动到结束的完整训练流程。

---

## 1. 启动

运行 `python scripts/train.py` 后，系统依次完成以下准备工作。

### 1.1 读取配置

系统从项目根目录的 `configs/` 下读取五份配置文件并合并为一份统一配置：

- `training/default.yaml` — 训练超参数、运行时设置
- `env/default.yaml` — 环境物理参数
- `policy/default.yaml` — 网络结构
- `task/default.yaml` — 任务目标
- `curriculum/default.yaml` — 课程阶段定义

其中 `training/default.yaml` 同时包含运行时设置：设备选择、随机种子、是否无头模式、日志目录、checkpoint 目录、是否从 checkpoint 恢复等。这些参数不通过命令行传入，直接修改 yaml 文件即可切换本地验证和服务器训练。

### 1.2 初始化设备与随机种子

系统根据配置选择计算设备（CUDA 或 CPU），然后设置 PyTorch 和 NumPy 的随机种子，保证训练可复现。

### 1.3 启动仿真运行时

在构建环境之前，系统先通过 `launch_simulation_app()` 启动 Isaac Sim 仿真后端，然后基于 `env/default.yaml` 中的 `physics_dt` 与 `decimation` 显式创建 `SimulationContext`。若当前 Python 环境中没有安装 Isaac Sim / Isaac Lab，系统会打印提示并终止。

---

## 2. 组件构建

所有训练组件在进入主循环前按以下顺序创建。

### 2.1 GPU 批量并行环境

系统创建一个 `DoorPushEnv` 实例（`DirectRLEnv` 子类），内含 $N$ 个 GPU 并行仿真的门推交互环境。$N$ 的值由 `training/default.yaml` 中的 `num_envs` 决定，写入 `DoorPushEnvCfg.scene.num_envs`。

`DoorPushEnv` 是**自包含**的环境类，一个类承担了旧架构中 `DoorInteractionEnv`、`SceneFactory`、`ContactMonitor`、`TaskManager`、`ActorObsBuilder`、`RewardManager` 六个组件的全部职责。所有 per-env 状态使用 `(N, ...)` 形状的 torch tensor 表示，观测/奖励/终止判定均为纯 tensor 操作，无 Python 循环。

#### 2.1.1 DoorPushEnv 内部结构

`DoorPushEnv` 继承自 Isaac Lab 的 `DirectRLEnv`，在 `__init__()` 中完成以下初始化：

| 初始化步骤 | 内容 |
|------|------|
| 关节/body 索引解析 | 一次性从 `Articulation` 中查找 12 个臂关节 (`ARM_JOINT_NAMES`)、2 个 gripper 关节、`base_link`/`left_gripper_link`/`right_gripper_link` body 索引、门铰链关节索引 |
| Per-env 状态 tensor | 分配 `(N, ...)` 形状的零值 tensor：`_prev_action`、`_left_occupied`/`_right_occupied`、`_step_count`、`_prev_door_angle`、`_already_succeeded`、域随机化参数缓存、EE 速度缓存、视觉 embedding 缓存 |
| 自碰撞分组 | 解析左臂/右臂/底座三组 body 索引，建立交叉碰撞检测分组 |

环境不持有独立的子组件实例 — 所有逻辑（观测构建、奖励计算、终止判定、接触检测、杯体脱落判定）都作为 `DoorPushEnv` 的方法直接实现。

#### 2.1.2 声明式场景配置与 Cloner 自动复制

场景通过 `@configclass` 装饰的 `DoorPushSceneCfg`（`InteractiveSceneCfg` 子类）声明式定义。配置中每个资产的 `prim_path` 使用 `{ENV_REGEX_NS}` 占位符：

```python
@configclass
class DoorPushSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot", ...
    )
    door: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door", ...
    )
    cup_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CupLeft", ...
    )
    cup_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CupRight", ...
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", ...
    )
```

`DoorPushEnv._setup_scene()` 将资产注册到 `InteractiveScene`，Isaac Lab 的 **Cloner** 随后自动将 `{ENV_REGEX_NS}` 替换为 `/World/envs/env_\d+`，为 $N$ 个并行环境各复制一棵完整的场景子树。

与旧架构的 `SceneFactory.build()` 相比，新架构完全不需要手动 8 步装配：

| 旧架构 (`SceneFactory.build()`) | 新架构 (`DoorPushSceneCfg` + Cloner) |
|------|------|
| 命令式 Python 脚本逐步加载资产 | 声明式 `@configclass` 配置 |
| 每个子环境独立执行 USD 加载 | Cloner 自动批量复制 |
| 手动创建 `SimulationContext` 视图 | `DirectRLEnv` 基类自动管理 |
| `reset()` 时可能需要清理旧场景 | 资产预生成，reset 时仅 teleport |

**杯体预生成策略**：左右杯体在场景创建时即被预生成（默认 `init_state.pos` 在远处 `(100, 0, 0)` / `(100, 1, 0)`），不会在 episode 边界反复创建/删除。reset 时根据 occupancy 将需要的杯体 teleport 到夹爪位置，不需要的保持在远处。

#### 2.1.3 DirectRLEnvAdapter

`DirectRLEnvAdapter` 将 `DoorPushEnv` 的 GPU tensor 接口包装为 `VecEnvProtocol`（`list[dict]` 格式），使现有训练管线（`RolloutCollector` + PPO）能直接消费 GPU 并行环境而无需大规模改写训练侧代码。

适配器的核心职责：
- `reset()` — 将 `occupancy` 和 `domain_params` 传递给底层 `DoorPushEnv`，调用 `env.reset()`，将返回的 `(N, obs_dim)` tensor 解包为 `list[dict]`
- `step(np.ndarray)` — 将 numpy 动作转为 torch tensor，调用 `env.step()`，将结果转回 numpy 和 `list[dict]`
- `set_curriculum()` — 更新 occupancy 和域随机化参数（兼容旧 `VecDoorEnv.set_curriculum()` 接口）

### 2.2 Actor 与 Critic

Actor 是一个带 GRU 隐状态的循环网络，接收五个观测分支（本体感觉、末端状态、上下文标记、稳定性代理、视觉编码）并输出关节力矩的高斯分布。Critic 是一个前馈网络，接收与 Actor 相同的观测分支，外加一个包含域随机化参数和门状态的 privileged 向量，输出标量状态价值。这种不对称设计使得 Actor 只能看到部署时可用的信息，而 Critic 可以利用仿真内部状态来降低价值估计的方差。

### 2.3 PPO 训练器

PPO 训练器持有 Actor 和 Critic 各自独立的 Adam 优化器，负责在每轮迭代中执行 PPO-Clip 参数更新。

### 2.4 轨迹采集器与缓冲区

轨迹采集器（`RolloutCollector`）负责驱动并行环境推进并记录 on-policy 数据。它内部维护 Actor 的 GRU 隐状态，在每步前向推理后将观测、动作、对数概率、价值估计、奖励和终止标志写入缓冲区。

缓冲区（`RolloutBuffer`）是固定容量的张量存储，rollout 结束后负责计算 GAE 优势估计，并按序列长度切分出 TBPTT 所需的 mini-batch。

### 2.5 视觉感知运行时

`PerceptionRuntime` 负责在训练侧定期刷新视觉 embedding。它不内嵌在 `DoorPushEnv` 中，而是作为外部模块由 `RolloutCollector` 在采集循环中调用。刷新后的 768 维 embedding 会直接注入 `actor_obs_list / critic_obs_list` 的 `visual.door_embedding` 字段；若当前帧视觉模块异常，则立即抛错终止，而不是把 `visual_valid` 继续传给策略。

### 2.6 课程管理器

课程管理器维护一个三阶段训练课程，通过滑动窗口平均成功率判定阶段跃迁。三个阶段分别是：

- Stage 1 — 无持杯推门：双臂空闲，只需学会视觉引导下的基本接触
- Stage 2 — 单臂持杯推门：左臂或右臂持杯，学会在约束下稳定推门
- Stage 3 — 混合分布推门：无杯、单臂持杯、双臂持杯四种上下文均匀采样

当滑动窗口内的平均成功率达到阈值（默认 0.8）时，课程自动跃迁到下一阶段。

### 2.7 域随机化器

域随机化器在两个时间尺度上工作：

**回合级**：每个新 episode 开始时（通过 `_reset_idx()` 内部采样）重新采样一组物理参数（杯体质量、门板质量、门铰链阻尼、基座位置 `base_pos`、基座朝向 `base_yaw`）。其中 base pose 的采样几何不是方形抖动，而是"以推板中心为圆心、门外侧小扇形环中的位置采样 + 朝向推板中心的小角度 yaw 扰动"（由 `sample_base_poses()` 实现）。这组参数在整个 episode 内保持不变。

**步级**：动作噪声和观测噪声直接在 `DoorPushEnv` 内部以 tensor 操作实现。动作噪声在 `_pre_physics_step()` 中力矩截断之后注入，注入后重新截断以保证安全（由 `DoorPushEnvCfg.action_noise_std` 控制）；观测噪声仅注入 Actor 观测中的关节位置和关节速度（共 24 维），Critic 始终看到无噪声的真实状态（由 `DoorPushEnvCfg.obs_noise_std` 控制）。

### 2.8 回调注入

域随机化器和课程管理器通过两个兼容性接口与 `DirectRLEnvAdapter` 连接：

- **episode 重采样回调**：通过 `envs.set_episode_reset_fn()` 注册。当某个子环境的 episode 结束并触发 auto-reset 时，回调会查询课程管理器的当前阶段，然后为该环境独立采样新的域参数、门类型和持杯上下文。
- **步级噪声配置**：噪声标准差直接在 `DoorPushEnvCfg` 中声明，`DoorPushEnv` 内部自动注入。`envs.set_randomizer()` 保留为兼容接口，实际为空操作。

### 2.9 初始环境重置

进入主循环前，系统根据当前课程阶段（通常是 Stage 1）为所有并行环境批量采样一组初始参数，然后通过 `DirectRLEnvAdapter.reset()` 执行一次全量重置。这会触发 `DoorPushEnv.reset()` → `_reset_idx(all_env_ids)`，为每个环境采样域随机化参数、写入 base pose、重置门关节、处理杯体 teleport、应用物理参数、清零 per-env 状态 tensor。同时清空所有 Actor 的 GRU 隐状态。

---

## 3. 训练主循环

主循环以"迭代"为单位反复执行，每轮迭代依次完成：采集轨迹、计算优势、更新参数、推进课程、记录日志。训练在累计步数达到上限或用户手动中断时结束。

### 3.1 轨迹采集

采集器在 $N$ 个并行环境中推进 $T$ 步（$T$ 由 `n_steps_per_rollout` 决定），每轮共产出 $N \times T$ 条 transition。

单步采集流程如下：

1. 将各环境的原始观测展平为网络输入格式。Actor 输入包含本体感觉（关节角度、角速度、力矩、上一步动作）、末端执行器状态、上下文标记、稳定性代理和视觉编码五个分支。Critic 在此基础上额外接收 privileged 向量。

2. 缓存当前 GRU 隐状态，供后续 TBPTT 恢复使用。

3. Actor 前向推理：输入观测分支和当前隐状态，输出动作采样、对数概率、策略熵和更新后的隐状态。

4. Critic 前向推理：输入观测分支和 privileged 向量，输出状态价值标量。

5. 将动作发送给 `DirectRLEnvAdapter.step()`，适配器将 numpy 动作转为 torch tensor 后调用 `DoorPushEnv.step()`。环境内部的单步处理是：
   - `_pre_physics_step()` — 力矩裁剪 → 注入动作噪声 → 重新裁剪 → 写入关节力矩目标
   - 物理引擎执行 `decimation` 次步进
   - `_get_observations()` — 批量读取物理状态、变换到 base_link 系、计算加速度和 tilt、构建 Actor/Critic 观测 tensor
   - `_get_rewards()` — 计算 12 项奖励
   - `_get_dones()` — 判定 terminated/truncated

6. 将本步数据（观测、动作、对数概率、价值、奖励、终止标志、隐状态）写入缓冲区。

7. 处理终止的环境：对 done 的子环境清零其 GRU 隐状态。`DirectRLEnv` 框架对 done 的子环境自动触发 `_reset_idx(env_ids)` 执行选择性重置，仅影响终止的环境而不干扰其他环境。重置后返回新 episode 的初始观测。轨迹因此自然接续到下一个 episode 的起点，不会出现观测中断。

采集结束后，采集器会对当前时刻的观测额外做一次 Critic 前向，得到用于 bootstrap 的末端价值估计。

### 3.2 GAE 优势估计

缓冲区从 rollout 末端向前逐步计算广义优势估计（GAE）。在每个时间步上，先算出单步 TD 误差：

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) \cdot (1 - \text{done}_t) - V(s_t)$$

再按指数加权递推得到优势：

$$\hat{A}_t = \delta_t + \gamma \lambda \cdot (1 - \text{done}_t) \cdot \hat{A}_{t+1}$$

回报为优势加上基线：

$$R_t = \hat{A}_t + V(s_t)$$

末端 bootstrap 使用采集阶段缓存的 `last_values` 和 `last_dones`：如果 rollout 末尾某个环境未终止，则用 Critic 估计的价值进行 bootstrap；如果已终止，则屏蔽 bootstrap。

### 3.3 PPO 参数更新

PPO 训练器从缓冲区中按序列长度切分 mini-batch，每个 mini-batch 是一段连续时间片段，附带该片段起始时刻的 GRU 隐状态。

**Actor 更新**：在每个 mini-batch 内，从缓存的隐状态出发逐时间步展开 Actor，得到新策略下的对数概率和熵。计算重要性比率 $r(\theta) = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})$，然后按 PPO-Clip 目标裁剪。总 Actor 损失 = clipped surrogate loss $-$ 熵系数 $\times$ 熵。

**Critic 更新**：Critic 不使用 RNN，直接将整个序列展平后批量前向，得到新的价值估计。计算 clipped value loss 或普通 MSE value loss。

两个网络分别做梯度裁剪后由各自的 Adam 优化器更新参数。

每次更新返回六项指标：Actor 损失、Critic 损失、策略熵、clip 比例、近似 KL 散度、explained variance。

### 3.4 课程跃迁判定

每轮迭代结束后，系统将本轮的 episode 成功率报告给课程管理器。课程管理器内部执行以下判定流程：

```
CurriculumManager.report_epoch(success_rate)
  │
  ├── 1. 将 success_rate 追加到滑动窗口 deque(maxlen=M)
  │
  ├── 2. 若已在最终阶段 (stage_3) → 返回 False，不做跃迁
  │
  ├── 3. 若窗口未填满 (len < M) → 返回 False，等待更多数据
  │
  └── 4. 计算窗口均值并与阈值比较
        若 (1/M) * Σ η_e ≥ η_thresh → 触发跃迁
```

跃迁条件的数学判据为：

$$\frac{1}{M} \sum_{e=E-M+1}^{E} \eta_e \geq \eta_{\text{thresh}}$$

其中 $M$ 为滑动窗口长度（默认 50），$\eta_e$ 为第 $e$ 轮的 episode 成功率，$\eta_{\text{thresh}}$ 为跃迁阈值（默认 0.8）。

#### 跃迁执行

当条件满足时，`_advance()` 方法执行以下操作：

1. **推进阶段索引**：`current_idx += 1`，指向下一个 `StageConfig`
2. **清空滑动窗口**：`success_window.clear()`，避免旧阶段的成功率残留影响新阶段的跃迁判定
3. **train.py 响应跃迁**：检测到 `report_epoch()` 返回 `True` 后，立即为所有 $N$ 个并行环境重新批量采样一组新课程阶段下的域参数和上下文分布，并通过 `envs.set_curriculum()` 推送给 `DirectRLEnvAdapter`，后者调用 `DoorPushEnv.set_occupancy()` 更新 occupancy tensor
4. **后续 auto-reset 自动生效**：此后每个子环境触发 auto-reset 时，episode 重采样回调会自动使用新阶段的 `context_probabilities` 进行采样

三个阶段的上下文概率分布如下：

| 阶段 | `none` | `left_only` | `right_only` | `both` | 学习目标 |
|------|--------|-------------|-------------|--------|----------|
| Stage 1 | 1.0 | — | — | — | 基础视觉引导接触 |
| Stage 2 | — | 0.5 | 0.5 | — | 单臂持杯约束下推门 |
| Stage 3 | 0.25 | 0.25 | 0.25 | 0.25 | 全覆盖混合分布 |

#### Checkpoint 中的课程状态

课程管理器的状态（当前阶段索引、滑动窗口内容、总 epoch 数）会被序列化到每个 checkpoint 中。从 checkpoint 恢复训练时，课程进度可以无缝接续。

### 3.5 日志与 checkpoint

**控制台日志**：按配置间隔打印当前迭代号、累计步数、FPS、Actor/Critic 损失、策略熵、clip 比例、平均奖励、成功率、当前课程阶段和预计剩余时间。

**TensorBoard**：如果可用，每轮写入训练损失指标、PPO 优化指标、rollout 统计、各持杯上下文的分类成功率、课程阶段和滑动窗口均值。

**Checkpoint**：按配置间隔保存中间 checkpoint，每个 checkpoint 包含当前迭代号、累计步数、Actor 和 Critic 的网络权重、PPO 优化器状态、课程管理器状态和历史最佳成功率。训练可从任意 checkpoint 恢复。

### 3.6 清空缓冲区

每轮迭代末尾清零缓冲区中的张量数据，但不销毁缓冲区对象本身，下一轮采集继续复用同一批预分配内存。

---

## 4. Episode 生命周期

本节详述单个 episode 从创建到终止的完整生命周期。在 GPU 批量架构下，所有操作都是对 `env_ids` 子集的 tensor 索引操作，而非独立的 Python 对象。

### 4.1 Episode 创建（`_reset_idx`）

当 `DirectRLEnv` 框架检测到某些环境的 `done = True` 时，自动调用 `DoorPushEnv._reset_idx(env_ids)` 执行**选择性重置**，仅重置终止的环境而不干扰其余环境。重置按以下 7 步顺序执行：

```
DoorPushEnv._reset_idx(env_ids)
  │
  ├── 1. 采样域随机化参数
  │     对 env_ids 中的每个环境独立采样：
  │       cup_mass   ∈ [0.1, 0.8] kg
  │       door_mass  ∈ [5.0, 20.0] kg
  │       door_damping ∈ [0.5, 5.0] Nm·s/rad
  │       base_pos, base_yaw ← sample_base_poses()
  │         （门外侧扇形环采样 + 朝向推板中心的 yaw 扰动）
  │
  ├── 2. 写入机器人 base pose
  │     将 base_pos + env_origins[env_ids] 写入 root state
  │     将 base_yaw 转换为 quaternion 写入 root orientation
  │     重置关节到默认位置，速度清零
  │
  ├── 3. 重置门关节角度
  │     将门铰链关节 pos / vel 全部清零
  │
  ├── 4. Occupancy 保持不变
  │     不重置 _left_occupied / _right_occupied
  │     保留外部课程管理器通过 set_occupancy() 注入的值
  │
  ├── 5. 杯体处理
  │     将不需要持杯的环境的杯体 teleport 到远处 (x=100)
  │     对需要持杯的环境执行 _batch_cup_grasp_init():
  │       直接将臂关节写到预设抓取姿态（最终状态）
  │       直接关闭 gripper 到 -34°
  │       将杯体 teleport 到基座相对坐标的世界位置
  │       （纯 teleport，无 sim.step()，不影响其他环境）
  │
  ├── 6. 应用域随机化物理参数
  │     _apply_domain_params(env_ids):
  │       写入门板 body 质量 → PhysX view set_masses()
  │       写入门铰链阻尼 → write_joint_damping_to_sim()
  │       写入左/右杯体质量 → PhysX view set_masses()
  │
  └── 7. 清零 per-env 状态 tensor
        _step_count[env_ids] = 0
        _prev_door_angle[env_ids] = 0
        _prev_action[env_ids] = 0
        _already_succeeded[env_ids] = False
        _prev_{left,right}_ee_{lin,ang}_vel[env_ids] = 0
        _door_embedding[env_ids] = 0
```

> **与旧架构的关键差异**：旧架构中 `SceneFactory.build()` 每次 reset 都可能删除和重新加载 USD 资产。新架构中所有资产在场景创建时由 Cloner 一次性复制，reset 仅做状态写入和 teleport，性能显著更高。

### 4.2 Episode 推进

每一步由 `DoorPushEnv` 的三个框架回调方法驱动：

#### 4.2.1 `_pre_physics_step(actions)` — 动作执行

```
actions (N, 12) 策略输出的力矩
  │
  ├── 缓存原始动作（clip 前），用于力矩超限惩罚 §6.4
  ├── 力矩裁剪到 [-effort_limit, +effort_limit]
  ├── 注入步级动作噪声 ε_a (σ = action_noise_std)
  ├── 重新裁剪以保证安全
  ├── 构建全关节力矩向量（仅 12 个臂关节非零）
  ├── robot.set_joint_effort_target(efforts)
  └── 保存 clipped action 到 _prev_action
```

物理引擎随后执行 `decimation` 次步进（$D=2$，对应策略频率 60 Hz）。

#### 4.2.2 `_get_observations()` — 观测构建

观测构建为纯 tensor 操作，一次性为所有 $N$ 个环境批量计算：

**Actor 观测** (858 维)：

| 分支 | 维度 | 内容 |
|------|------|------|
| proprio | 48 | 关节角度(12) + 关节速度(12) + 力矩(12) + 上一步动作(12)，含观测噪声 |
| left_ee | 19 | base 系下的 pos(3) + quat(4) + 线速度(3) + 角速度(3) + 线加速度(3) + 角加速度(3) |
| right_ee | 19 | 同上 |
| context | 2 | left_occupied(1) + right_occupied(1) |
| stability | 2 | left_tilt(1) + right_tilt(1)，重力在 EE 局部系中的 xy 投影范数 |
| visual | 768 | 门 embedding(768) |

**Critic 观测** (874 维)：

| 分支 | 维度 | 内容 |
|------|------|------|
| actor_obs 同构部分 | 858 | 与 Actor 相同结构，但 **无观测噪声** |
| privileged | 16 | door_pose_base(7) + door_joint_pos(1) + door_joint_vel(1) + cup_mass(1) + door_mass(1) + door_damping(1) + base_pos(3) + cup_dropped(1) |

关键计算步骤：
- **坐标变换**：所有 EE 状态和门状态都从世界系变换到 `base_link` 相对系（`_ee_world_to_base()`），使观测对机器人全局位置不敏感
- **数值微分加速度**：线加速度 / 角加速度通过当前帧与上一帧的速度差除以 `control_dt` 计算
- **Tilt proxy**：将重力向量旋转到 EE 局部系后取 xy 分量，作为杯体倾斜的代理信号

#### 4.2.3 `_get_rewards()` — 奖励计算

奖励由 12 个子项组成，按任务/稳定性/安全三类组织：

$$r = r_\text{task} + r_\text{stab} - r_\text{safe}$$

**§4 任务奖励** (2 子项)：

| 子项 | 公式 |
|------|------|
| 角度增量 | $w(\theta) \cdot \Delta\theta$，其中 $w = w_\delta$ 当 $\theta \leq \theta_s$；$w = w_\delta \cdot \max(\alpha,\; 1 - k(\theta - \theta_s))$ 当 $\theta > \theta_s$ |
| 成功 bonus | $\theta$ 首次达到 $\theta_s = 1.2\ \text{rad}$ 时一次性奖励 $w_\text{open} = 50$ |

**§5 稳定性奖励** (7 子项，双臂分别计算并以 occupancy mask $m$ 加权)：

| 编号 | 子项 | 公式 |
|------|------|------|
| 5.1 | 零线加速度 bonus | $w_\text{zacc} \cdot \exp(-\lambda_\text{acc} \cdot \|\mathbf{a}_\text{lin}\|^2)$ |
| 5.2 | 零角加速度 bonus | $w_\text{zang} \cdot \exp(-\lambda_\text{ang} \cdot \|\mathbf{a}_\text{ang}\|^2)$ |
| 5.3 | 线加速度惩罚 | $-w_\text{acc} \cdot \|\mathbf{a}_\text{lin}\|^2$ |
| 5.4 | 角加速度惩罚 | $-w_\text{ang} \cdot \|\mathbf{a}_\text{ang}\|^2$ |
| 5.5 | 杯体倾斜惩罚 | $-w_\text{tilt} \cdot \|\text{tilt}_{xy}\|^2$ |
| 5.6 | 力矩平滑惩罚 | $-w_\text{smooth} \cdot \|\Delta\tau\|^2$ |
| 5.7 | 力矩正则惩罚 | $-w_\text{reg} \cdot \|\tau\|^2$ |

**§6 安全惩罚** (5 子项)：

| 编号 | 子项 | 公式 |
|------|------|------|
| 6.1 | 自碰撞 | $\beta_\text{self} \cdot \mathbb{1}[\text{collision}]$ |
| 6.2 | 关节限位 | $\beta_\text{limit} \cdot \sum_j [\max(0, |q_j - c_j| - \mu \cdot h_j)]^2$ |
| 6.3 | 关节速度 | $\beta_\text{vel} \cdot \sum_j [\max(0, |\dot{q}_j| - \mu \cdot v_j^\text{lim})]^2$ |
| 6.4 | 力矩超限 | $\beta_\text{torque} \cdot \sum_j [\max(0, |a_j^\text{raw}| - \tau^\text{lim})]^2$ |
| 6.5 | 杯体掉落 | $w_\text{drop} \cdot \mathbb{1}[\text{cup\_dropped}]$ |

#### 4.2.4 `_get_dones()` — 终止判定

终止由两个独立来源共同决定：

```python
terminated = cup_dropped | angle_reached   # 杯掉落 or 门角度达标(≥1.57 rad)
truncated  = step_count >= max_episode_length   # 超时(5400 步 = 90 秒)
```

> **重要区分：成功标记 ≠ 终止条件**
>
> 奖励计算中维护两个不同的角度阈值：
> - **成功阈值** `success_angle_threshold = 1.2 rad`（约 69°）：门角度首次达到此值时，触发一次性成功 bonus（$w_\text{open} = 50$），并设置 `_already_succeeded = True`，但 **不会终止** episode
> - **终止阈值** `door_angle_target = 1.57 rad`（约 90°）：门角度达到此值时 **终止** episode（`terminated = True`）
>
> 这意味着策略需要将门从 69° 继续推到 90° 才能正常结束 episode。如果策略在 69°~90° 之间失去控制（例如杯体掉落或超时），该 episode 仍然已经获得过成功 bonus，但 episode 以失败原因终止。

### 4.3 终止后的处理

在 `DirectRLEnv` 框架中，当某个环境的 `done = True` 时：

1. 该环境的 episode 统计信息（通过 `_build_info_list()` 构建）被收集并上报
2. 框架自动调用 `_reset_idx(env_ids)` 执行选择性重置（详见 §4.1），仅重置终止的环境
3. 采集器清零该环境对应的 GRU 隐状态
4. 下一步采集从新 episode 的初始观测无缝继续

---

## 5. 场景装配的时机

训练过程中，场景的物理拓扑在 Cloner 复制后**永不改变**。与旧架构不同，新架构没有"装配"和"拆解"的概念，只有状态重写。

**一次性场景创建**：`DoorPushEnv.__init__()` → `_setup_scene()` 将 robot/door/cup_left/cup_right/contact_sensor 注册到 `InteractiveScene`，Cloner 为 $N$ 个环境复制完整场景子树。此后场景拓扑固定。

**episode 内**：只做物理步进。每步向关节施加力矩、推进仿真、读取新状态。

**选择性 reset**：当某些环境 done 时，`_reset_idx(env_ids)` 仅对这些环境的状态 tensor 执行写入操作（root state、joint state、cup teleport、物理参数），其他环境不受影响。由于杯体是预生成的，不需要的杯体始终停留在远处 `(100, y, 0)`，不参与物理碰撞。

因此，一个 episode 的初始状态由以下输入决定：域随机化参数（由 `_reset_idx` 内部采样）、occupancy 标记（由外部课程管理器通过 `set_occupancy()` 注入）。这些输入每个新 episode 都会独立重新设定。

---

## 6. 训练结束

训练在以下两种情况下结束：

- 累计步数达到配置的 `total_steps` 上限
- 用户按 Ctrl+C 手动中断

无论哪种方式，系统都会在退出前执行以下收尾工作：

1. 保存最终 checkpoint（`ckpt_final.pt`）
2. 打印训练总结：总耗时、总步数、平均 FPS、最终课程阶段、当前课程窗口成功率
3. 释放所有环境资源（`envs.close()`）
4. 关闭 TensorBoard 写入器
5. 关闭仿真运行时（`simulation_app.close()`）
