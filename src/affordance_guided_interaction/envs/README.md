# envs — GPU 批量并行仿真环境

## 1. 本层做什么

envs 层是整个系统中**唯一直接与 Isaac Lab 物理引擎交互的层**。它以 Isaac Lab `DirectRLEnv` 为基类，在单个 GPU 上并行运行数千个环境实例，完成场景装配、动作执行、观测构建、奖励计算和终止判定的完整闭环。

```
              training/ (PPO 训练循环)
                   │
                   │  RolloutCollector
                   │  消费 VecEnvProtocol 接口
                   │
                   ▼
         ┌─────────────────────────────────────────────────┐
         │  DirectRLEnvAdapter                              │
         │  tensor → list[dict] 格式适配                    │
         └─────────────────────┬───────────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────────────────┐
         │  DoorPushEnv  (DirectRLEnv)          ◄── 核心   │
         │                                                  │
         │  _setup_scene()         场景资产注册              │
         │  _reset_idx()           选择性重置 + 持杯初始化   │
         │  _pre_physics_step()    力矩注入                  │
         │  _get_observations()    Actor(859D)/Critic(875D) │
         │  _get_rewards()         12-term 奖励              │
         │  _get_dones()           终止判定                  │
         └─────────────────────────────────────────────────┘
         │                                                  │
         │  DoorPushEnvCfg + DoorPushSceneCfg               │
         │  @configclass 声明式配置                          │
         │                                                  │
         │  batch_math.py                                   │
         │  GPU tensor 四元数 / 坐标变换工具                 │
         │                                                  │
         │  Reward.md                                       │
         │  12-term 奖励函数的数学参考文档                   │
         └──────────────────────────────────────────────────┘
```

---

## 2. 文件清单

| 文件 | 职责 |
|---|---|
| `door_push_env.py` | **DoorPushEnv** — `DirectRLEnv` 子类，自包含全部环境逻辑 |
| `door_push_env_cfg.py` | **DoorPushEnvCfg** + **DoorPushSceneCfg** — 声明式 `@configclass` 配置 |
| `direct_rl_env_adapter.py` | **DirectRLEnvAdapter** — 将 tensor 接口包装为 `VecEnvProtocol`，桥接 RolloutCollector |
| `batch_math.py` | GPU 批量四元数运算、坐标系变换、基座位姿采样 |
| `Reward.md` | 12-term 奖励函数的数学定义与超参参考 |
| `__init__.py` | 对外导出 `DoorPushEnv`、`DoorPushEnvCfg`、`DoorPushSceneCfg`、`DirectRLEnvAdapter` |

---

## 3. 架构总览

### 3.1 场景声明：DoorPushSceneCfg

使用 Isaac Lab 的 `@configclass` + `InteractiveSceneCfg` 以声明式方式定义完整场景：

| 资产 | 类型 | prim_path | 说明 |
|---|---|---|---|
| 双臂移动机器人 | `ArticulationCfg` | `{ENV_REGEX_NS}/Robot` | Unitree Z1 双臂 + Dingo 底座，力矩直驱 |
| 推门 | `ArticulationCfg` | `{ENV_REGEX_NS}/Door` | 单铰链门，无主动力矩，阻尼可随机化 |
| 左杯体 | `RigidObjectCfg` | `{ENV_REGEX_NS}/CupLeft` | 预生成在远处，reset 时按 occupancy teleport |
| 右杯体 | `RigidObjectCfg` | `{ENV_REGEX_NS}/CupRight` | 同上 |
| 接触传感器 | `ContactSensorCfg` | `{ENV_REGEX_NS}/Robot/.*` | 覆盖机器人全身，用于自碰撞检测 |
| 地面 + 照明 | `AssetBaseCfg` | 全局 | `GroundPlaneCfg` + `DomeLightCfg` |

`{ENV_REGEX_NS}` 是 Isaac Lab Cloner 的占位符，Cloner 会将整棵场景子树自动复制到 `/World/envs/env_0`、`env_1`、……，实现 GPU 批量并行。

### 3.2 环境主体：DoorPushEnv

`DoorPushEnv` 继承 `DirectRLEnv`，是**完全自包含**的环境实现——观测构建、奖励计算、终止判定全部在此完成，不依赖外部 observations/ 或 rewards/ 模块。

核心生命周期：

```
_setup_scene()        ──▶  向 InteractiveScene 注册 5 种资产
                            Cloner 自动复制

_reset_idx(env_ids)   ──▶  采样域随机化参数
                            写入 base pose + 关节默认值
                            重置门角度
                            teleport 杯体（持杯 env 执行批量抓取初始化）
                            应用物理参数（质量 / 阻尼）
                            清零 per-env 状态

_pre_physics_step()   ──▶  缓存 raw action（用于 §6.4 力矩超限惩罚）
                            clamp + 注入步级噪声
                            写入 arm 关节 effort target

_get_observations()   ──▶  读取关节状态 + body 状态
                            世界系 → base_link 系变换
                            数值微分算加速度
                            计算 tilt proxy
                            拼接 Actor obs (859D) / Critic obs (875D)
                            缓存加速度/tilt_xy 供奖励使用

_get_rewards()        ──▶  §4 任务奖励（角度增量 + 一次性成功 bonus）
                            §5 稳定性奖励（7 子项 × 双臂，mask 条件化）
                            §6 安全惩罚（5 子项，始终激活）

_get_dones()          ──▶  杯体脱落 → terminated
                            门角度达标 → terminated
                            步数超限 → truncated
```

### 3.3 训练管线适配：DirectRLEnvAdapter

`DirectRLEnvAdapter` 将 DoorPushEnv 的 GPU batch tensor 接口包装为 RolloutCollector 期望的 `VecEnvProtocol` 格式：

| DoorPushEnv 接口 | DirectRLEnvAdapter 转换 | RolloutCollector 期望 |
|---|---|---|
| `reset()` → `{"policy": (N, 859), "critic": (N, 875)}` | 解包为 per-env dict list | `(list[dict], list[dict])` |
| `step(Tensor)` → `(obs_dict, reward, term, trunc, info)` | tensor→numpy + info list | `(list[dict], list[dict], ndarray, ndarray, list[dict])` |

适配器同时提供 `set_curriculum()`、`set_episode_reset_fn()` 等兼容旧训练接口的方法。长期目标是训练侧原生支持 tensor dict，届时可移除。

---

## 4. 观测空间

### 4.1 Actor 观测 (859D)

Actor 观测包含噪声（`obs_noise_std`），不含 privileged 信息：

| 分段 | 内容 | 维度 |
|---|---|---|
| proprio | 关节位置 q(12) + 速度 dq(12) + 力矩 tau(12) + 上一步动作(12) | 48 |
| left_ee | 位置(3) + 朝向 quat(4) + 线速度(3) + 角速度(3) + 线加速度(3) + 角加速度(3) | 19 |
| right_ee | 同上 | 19 |
| context | left_occupied(1) + right_occupied(1) | 2 |
| stability | left_tilt(1) + right_tilt(1) | 2 |
| visual | Point-MAE door embedding(768) + valid flag(1) | 769 |
| **总计** | | **859** |

所有 EE 状态均在 **base_link 相对坐标系**下表达。加速度通过数值微分（帧间速度差 / $\Delta t_{\text{control}}$）计算。tilt 为重力向量在 EE 局部系中的 $xy$ 投影范数。

### 4.2 Critic 观测 (875D)

Critic 观测 = Actor 观测（无噪声版） + privileged 信息：

| 分段 | 内容 | 维度 |
|---|---|---|
| actor 部分 | 与 Actor 相同的 proprio + ee + context + stability + visual，但**无噪声** | 859 |
| privileged | 门位姿(7) + 门铰链角度(1) + 门铰链速度(1) + 杯质量(1) + 门质量(1) + 门阻尼(1) + 基座位置(3) + 杯掉落标志(1) | 16 |
| **总计** | | **875** |

门位姿在 base_link 坐标系下。域随机化参数（质量、阻尼）作为 privileged info 仅供 critic 消费，actor 对此完全不可知。

---

## 5. 动作空间

策略输出 $\mathbf{a}_t \in \mathbb{R}^{12}$ — 双臂各 6 关节的**绝对力矩**（$\text{N}\cdot\text{m}$）。

执行流程：

1. 缓存 raw action（用于 §6.4 力矩超限惩罚）
2. 硬裁剪至 $[-\tau_{\text{limit}}, \tau_{\text{limit}}]$（默认 33.5 N·m，来自 URDF effort_limit）
3. 注入步级高斯噪声 $\sigma_a$（训练时）
4. 写入 arm 关节 effort target（gripper 和 wheel 关节不受策略控制）

系统不做重力补偿——抵抗刚体重力和摩擦扰动的工作完全由策略网络隐式学习。

---

## 6. 奖励函数概要

DoorPushEnv 内置完整的 12-term 奖励计算（详见 `Reward.md`）。总体公式：

$$
r_t = r_{\text{task}} + m_L \cdot r_{\text{stab}}^L + m_R \cdot r_{\text{stab}}^R - r_{\text{safe}}
$$

### §4 任务奖励（2 项）

| 子项 | 公式概要 |
|---|---|
| 角度增量奖励 | $w(\theta_t) \cdot (\theta_t - \theta_{t-1})$，达标前满额激励，超标后线性衰减至下限 $\alpha$ |
| 一次性成功 bonus | $\theta_t \geq \theta_{\text{success}}$ 时触发 $w_{\text{open}}$，仅触发一次 |

### §5 稳定性奖励（7 子项 × 双臂）

按持杯 mask $m_L, m_R$ 条件化激活，不持杯侧归零：

| 子项 | 类型 | 作用 |
|---|---|---|
| 零线加速度奖励 | Gaussian bonus | 鼓励 EE 线加速度趋近 0 |
| 零角加速度奖励 | Gaussian bonus | 鼓励 EE 角加速度趋近 0 |
| 线加速度惩罚 | 二次 penalty | 抑制高加速冲击 |
| 角加速度惩罚 | 二次 penalty | 抑制角加速度 |
| 重力倾斜惩罚 | 二次 penalty | 保持杯口朝上 |
| 力矩平滑项 | 二次 penalty | 抑制力矩抖动 |
| 力矩正则项 | 二次 penalty | 约束力矩幅值 |

### §6 安全惩罚（5 项，始终激活）

| 子项 | 触发条件 |
|---|---|
| 自碰撞 | 交叉分组 body 对同时有力 → 固定惩罚 $\beta_{\text{self}}$ |
| 关节限位逼近 | $\|q_i - q_i^c\| > \mu \cdot \delta_i$ → 二次惩罚 |
| 关节速度过大 | $\|\dot{q}_i\| > \mu \cdot \dot{q}_i^{\max}$ → 二次惩罚 |
| 力矩超限 | raw action 超出 effort_limit → 二次惩罚 |
| 杯体脱落 | cup-EE 距离 > $\epsilon_{\text{drop}}$ → 固定惩罚 $w_{\text{drop}}$ + episode 终止 |

---

## 7. 终止条件

| 条件 | 信号类型 | 触发逻辑 |
|---|---|---|
| 门角度达标 | `terminated` | $\theta_d \geq \theta_{\text{target}}$（默认 1.57 rad） |
| 杯体脱落 | `terminated` | 持杯侧 cup-EE 距离 > `cup_drop_threshold`（默认 0.15 m） |
| 步数超限 | `truncated` | `step_count >= max_episode_length` |

注意 episode 成功终止阈值（1.57 rad）与奖励 success bonus 阈值（1.2 rad）是**不同的**：前者终止 episode，后者仅触发一次性 bonus。

---

## 8. 选择性重置与持杯初始化

`_reset_idx(env_ids)` 仅重置指定的环境子集（Isaac Lab auto-reset 框架），流程：

1. **采样域随机化参数** — 杯质量、门质量、门阻尼、基座位姿（扇形环采样）
2. **写入机器人 root state** — base_pos + base_yaw → 世界坐标
3. **重置门关节** — 铰链角度归零
4. **杯体 teleport** — 不持杯的杯体放到 $(100, 0, 0)$ 远处
5. **批量持杯初始化** — 对需要持杯的 env：将臂关节直接设到预设抓取姿态，gripper 闭合，杯体 teleport 到夹爪位置（纯状态写入，不调 `sim.step()`，避免破坏其他 env）
6. **应用物理参数** — 门板质量、铰链阻尼、杯体质量写入 PhysX
7. **清零 per-env 状态** — step_count、prev_action、速度缓存、视觉 embedding 等

---

## 9. 视觉感知集成

DoorPushEnv **不包含任何相机代码**。视觉信息通过外部接口注入：

```python
env.update_visual_embedding(
    embedding: Tensor,  # (N, 768)  Point-MAE 门体几何 embedding
    valid: Tensor,      # (N,)      当前帧是否有有效视觉输入
)
```

外部的 `PerceptionRuntime` 负责获取点云、运行 Point-MAE 推理，然后调用上述接口将 768 维 embedding 写入环境缓存。DoorPushEnv 将 embedding + valid flag 直接拼入 Actor/Critic 观测。

---

## 10. 域随机化

每次 `_reset_idx()` 时自动采样的回合级物理参数：

| 参数 | 范围 | 落地方式 |
|---|---|---|
| `cup_mass` | [0.1, 0.8] kg | 修改杯体 RigidObject 质量 |
| `door_mass` | [5.0, 20.0] kg | 修改 DoorLeaf body 质量 |
| `door_damping` | [0.5, 5.0] | 修改门铰链阻尼系数 |
| `base_pos` | 门外扇形环 | 机器人 root pose 位置 |
| `base_yaw` | 标称朝向 ± 10° | 机器人 root pose 朝向 |

这些参数在 Critic 观测中作为 privileged information 暴露，Actor 不可见。外部课程管理器也可通过 `set_domain_params_batch()` 覆写。

---

## 11. 坐标变换工具：batch_math.py

提供 `(N, ...)` 批量 GPU tensor 操作，四元数约定为 `(w, x, y, z)`（与 Isaac Lab 一致）：

| 函数 | 作用 |
|---|---|
| `batch_quat_conjugate` / `multiply` / `normalize` | 四元数基础运算 |
| `batch_quat_to_rotation_matrix` | 四元数 → 3×3 旋转矩阵 |
| `batch_quat_from_yaw` / `batch_yaw_from_quat` | yaw 角 ↔ 四元数互转 |
| `batch_vector_world_to_base` | 世界系向量 → base_link 系 |
| `batch_orientation_world_to_base` | 世界系四元数 → base_link 相对四元数 |
| `batch_pose_world_to_base` | 世界系 pose → base_link 系 (pos3 + quat4 = 7D) |
| `sample_base_poses` | 门外扇形环中批量采样基座位姿 |
| `batch_rotate_relative_by_yaw` | base_link 局部偏移按 yaw 旋转到世界系 |

---

## 12. 环境参数一览

### 12.1 仿真参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `physics_dt` | 物理仿真步长 | 1/120 s |
| `decimation` | 策略控制间隔（物理步数） | 2（策略频率 60 Hz） |
| `episode_length_s` | 单 episode 时长上限 | 90.0 s（5400 控制步） |
| `num_envs` | 并行环境数 | 64（可调至数千） |
| `env_spacing` | 环境间距 | 5.0 m |

### 12.2 任务判定参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `door_angle_target` | episode 成功终止角度 | 1.57 rad（≈ 90°） |
| `success_angle_threshold` | 奖励 success bonus 触发角度 | 1.2 rad |
| `cup_drop_threshold` | 杯体脱落距离阈值 | 0.15 m |

### 12.3 动作与噪声参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `effort_limit` | 关节力矩硬截断 | 33.5 N·m |
| `action_noise_std` | 步级动作噪声 $\sigma_a$ | 0.02 |
| `obs_noise_std` | 观测噪声 $\sigma_o$（仅注入 Actor q/dq） | 0.01 |

### 12.4 奖励超参

所有奖励权重定义在 `DoorPushEnvCfg` 中，分为三组：

- **任务奖励 (§4)**：`rew_w_delta`, `rew_alpha`, `rew_k_decay`, `rew_w_open`
- **稳定性奖励 (§5)**：`rew_w_zero_acc`, `rew_lambda_acc`, `rew_w_zero_ang`, `rew_lambda_ang`, `rew_w_acc`, `rew_w_ang`, `rew_w_tilt`, `rew_w_smooth`, `rew_w_reg`
- **安全惩罚 (§6)**：`rew_beta_self`, `rew_beta_limit`, `rew_mu`, `rew_beta_vel`, `rew_beta_torque`, `rew_w_drop`

具体数值和数学定义见 `Reward.md`。

---

## 13. 外部注入接口

DoorPushEnv 提供三个供外部模块调用的注入接口：

| 方法 | 调用者 | 作用 |
|---|---|---|
| `set_occupancy(left, right)` | 课程管理器 | 设置所有 env 的左/右臂持杯 occupancy |
| `update_visual_embedding(emb, valid)` | PerceptionRuntime | 更新 768D Point-MAE 视觉 embedding |
| `set_domain_params_batch(params_list)` | 课程管理器 / 训练循环 | 覆写域随机化参数 |

---

## 14. 关键设计决策

### 为什么奖励计算内置在 DoorPushEnv 中？

Isaac Lab `DirectRLEnv` 要求 `_get_rewards()` 作为子类方法实现。将全部 12 个奖励子项内置在环境中可以：

- 直接访问仿真 ground truth tensor，无需序列化/反序列化
- 奖励计算与观测构建共享缓存（如加速度、tilt_xy），避免重复计算
- 所有 per-env 状态（如 `_already_succeeded`、`_prev_door_angle`）天然可用

### 为什么 DoorPushEnv 自包含而不是拆分为多个模块？

旧架构将环境、观测、奖励拆分到不同模块，导致数据传递复杂且容易出现语义分裂（如"加速度"在不同模块的定义不一致）。GPU 路径将所有逻辑收敛到 `DirectRLEnv` 子类中：

- 加速度只有一种计算方式（数值微分 + 缓存）
- tilt 的几何定义全局唯一
- 稳定性 proxy 在观测和奖励间通过 `_cached_*` 字段天然共享

### 为什么持杯初始化用 teleport 而不是物理步进？

`_batch_cup_grasp_init()` 直接将关节设到最终抓取姿态，并将杯体 teleport 到夹爪位置。这避免了调用 `sim.step()` 推进**所有**环境——在选择性重置场景中，只有部分 env 需要重置，物理步进会破坏非目标 env 的状态。

### 为什么需要 DirectRLEnvAdapter？

现有训练管线（RolloutCollector + PPO）期望 `list[dict]` 格式的观测，而 DoorPushEnv 输出的是 batch tensor。适配器是一个过渡层，将 tensor 解包为 per-env dict，使训练侧无需大规模重构即可消费 GPU 并行环境。

### 为什么视觉感知不内置在 DoorPushEnv 中？

DoorPushEnv 中没有相机代码。视觉信息通过 `update_visual_embedding()` 从外部注入，原因是：

- Point-MAE 推理有独立的计算图和刷新频率
- 解耦后可以在无相机的 headless 模式下训练（embedding 全零）
- 未来更换视觉骨干网络（如 PointNet++、3D-LLM）时环境代码无需修改
