# 项目完成度全面分析报告

> 生成日期：2026-04-02
>
> 本文档对 Affordance-Guided-Interaction 项目的所有模块进行逐一审查，
> 明确已完成的部分、仍需实现的部分，以及距离在 IsaacLab 中启动训练所需的优先级排序。

---

## 一、项目总览

本项目目标是让 **Uni-Dingo 双臂移动机器人**（Dingo 全向底盘 + 2×Unitree Z1 六自由度臂）
在**持杯约束**下完成各种门交互任务（推门、按按钮、拉把手等）。

- 训练框架：**PPO + Recurrent Actor + Asymmetric Critic**
- 感知管线：**冻结 Point-MAE 编码器**（零训练，开箱即用）
- 机器人：20 自由度（双臂12 + 云台2 + 底盘4 + 双夹爪2）
- 动作空间：12 维双臂关节力矩
- 关键上下文：`left_occupied` / `right_occupied` 持杯标志，控制行为分化

---

## 二、已完成的模块

### 2.1 策略网络层（`policy/`）— ✅ 全部完成

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `policy/actor.py` | ~400 | ✅ 完整实现 | 多分支编码器 → RecurrentBackbone → ActionHead，输出 12 维双臂关节力矩 |
| `policy/critic.py` | ~229 | ✅ 完整实现 | 非对称 MLP Critic，接收 actor\_obs + privileged info |
| `policy/recurrent_backbone.py` | — | ✅ 完整实现 | GRU / LSTM 可切换 |
| `policy/action_head.py` | — | ✅ 完整实现 | 对角高斯分布参数化，力矩裁剪 |

### 2.2 训练组件层（`training/`）— ✅ 全部完成

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `training/ppo_trainer.py` | ~319 | ✅ 完整实现 | 完整 PPO loss、梯度裁剪、value clipping、entropy bonus |
| `training/rollout_buffer.py` | ~325 | ✅ 完整实现 | 预分配张量、GAE 计算、TBPTT mini-batch 生成器 |
| `training/rollout_collector.py` | ~320 | ✅ 完整实现 | 多环境并行采样、RNN 隐状态管理、bootstrap |
| `training/curriculum_manager.py` | — | ✅ 完整实现 | 滑动窗口成功率驱动的阶段切换 |
| `training/domain_randomizer.py` | — | ✅ 完整实现 | 杯体 / 门 / 摩擦 / 噪声参数随机化 |
| `training/metrics.py` | — | ✅ 完整实现 | 成功率、稳定性、碰撞率统计 |

### 2.3 奖励层（`rewards/`）— ✅ 全部完成

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `rewards/reward_manager.py` | ~197 | ✅ 完整实现 | 聚合各项奖励、动态缩放 s\_t 退火 |
| `rewards/task_reward.py` | — | ✅ 完整实现 | 进展奖励 + 一次性成功 bonus |
| `rewards/stability_reward.py` | — | ✅ 完整实现 | 持杯稳定 bonus / penalty 分解 |
| `rewards/safety_penalty.py` | — | ✅ 完整实现 | 碰撞 / 自碰撞 / 关节限位 / 速度 / 掉杯检测 |

### 2.4 观测构建层（`observations/`）— ✅ 全部完成

| 文件 | 状态 | 说明 |
|------|------|------|
| `observations/actor_obs_builder.py` | ✅ 完整实现 | 组装 proprio + gripper + context + stability + z\_aff |
| `observations/critic_obs_builder.py` | ✅ 完整实现 | actor\_obs + privileged information |
| `observations/stability_proxy.py` | ✅ 完整实现 | 倾斜角、加速度、角速度、jerk 有限差分估计 |
| `observations/history_buffer.py` | ✅ 完整实现 | FIFO 历史动作 / 观测缓存 |

### 2.5 感知管线（`door_perception/`）— ✅ 全部完成

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `door_perception/affordance_pipeline.py` | ~189 | ✅ 完整实现 | 分割 → 深度反投影 → 体素降采样 → Point-MAE 编码 |
| `door_perception/frozen_encoder.py` | ~509 | ✅ 完整实现 | 完整 Point-MAE PyTorch 实现，权重冻结，FPS/KNN 内建 |
| `door_perception/segmentation.py` | — | ✅ 完整实现 | LangSAM / Grounded-SAM 2 包装器 |
| `door_perception/depth_projection.py` | — | ✅ 完整实现 | mask 区域深度 → 局部点云 |
| `door_perception/config.py` | — | ✅ 完整实现 | 相机内参、编码器类型等配置 dataclass |

### 2.6 环境层（`envs/`）— 部分完成

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `envs/base_env.py` | ~187 | ✅ 完整实现 | 抽象基类 + EnvConfig 配置 |
| `envs/task_manager.py` | — | ✅ 完整实现 | 成功 / 失败 / 超时判定逻辑 |
| `envs/vec_env.py` | — | ✅ 完整实现 | VecDoorEnv 包装 N 个环境，auto-reset |
| `envs/door_env.py` | ~375 | ⚠️ 部分实现 | 高层逻辑完整，但 `_sim_step()` 为 `pass`、`_read_physics_state()` 返回全零 |
| `envs/scene_factory.py` | ~272 | ❌ 占位 | 几乎全部 Isaac API 调用为 `_PlaceholderView()` 或 `pass` |
| `envs/contact_monitor.py` | — | ⚠️ 部分实现 | 接触处理逻辑完整，但 `_read_raw_contacts()` 返回空列表 |

### 2.7 资产文件 — ✅ 全部完成

| 资产 | 位置 | 状态 |
|------|------|------|
| 机器人 URDF / Xacro / USD / Mesh | `assets/robot/` | ✅ 全套 |
| 门场景 USD | `assets/minimal_push_door/` | ✅ 含主场景 + 门板 |
| 持物资产（杯 / 托盘）| `assets/grasp_objects/` | ✅ USD + catalog.json |
| URDF→USD 转换脚本 | `assets/robot/scripts/convert_urdf_to_usd.py` | ✅ 可运行 |
| Isaac Sim 场景加载脚本 | `scripts/load_scene.py` | ✅ 可运行 |

### 2.8 遥操作抓杯（`src/teleop_cup_grasp/`）— ✅ 视为已完成

> 此模块用于 RL 训练前的水杯抓取初始化，由用户本人负责完成。

| 文件 | 状态 | 说明 |
|------|------|------|
| `teleop_cup_grasp/cup_grasp_teleop.py` | ✅ | UI 滑块 + Pinocchio IK + 轨迹录制 |
| `teleop_cup_grasp/README.md` | ✅ | 任务说明文档 |

### 2.9 文档 — ✅ 完成

| 文档 | 说明 |
|------|------|
| `README.md` | 项目总览与使用方法 |
| `project_architecture.md` | 完整的六层架构设计、数据流、接口定义、实现顺序建议 |
| `affordance_layer_rewritten_single_visual_v5.md` | 感知层技术方案（问题定义 + Affordance 设计） |
| 各子模块 `README.md` | `envs/`、`policy/`、`rewards/`、`training/`、`observations/`、`door_perception/` 均有详细设计文档 |

---

## 三、尚未完成的模块

### 3.1 🔴 关键缺失：Isaac Sim / IsaacLab 环境层接入

这是距离训练**最大的缺口**。当前整个环境层的物理交互部分都是占位实现。

#### 需要填充的 `[ISAAC_API]` 占位点

| 文件 | 方法 | 当前状态 | 需要实现的内容 |
|------|------|----------|----------------|
| `envs/scene_factory.py` | `_spawn_robot()` | 返回 `_PlaceholderView()` | 加载机器人 USD，返回 `ArticulationView` |
| `envs/scene_factory.py` | `_spawn_door()` | 返回 `_PlaceholderView()` | 加载门 USD，创建铰链关节 |
| `envs/scene_factory.py` | `_spawn_cup()` | 返回 `_PlaceholderView()` | 加载杯体 USD，绑定到夹爪 |
| `envs/scene_factory.py` | `_clear_scene()` | `pass` | 清理场景 prim |
| `envs/scene_factory.py` | `_set_rigid_body_mass()` | `pass` | 域随机化：设置刚体质量 |
| `envs/scene_factory.py` | `_set_joint_damping()` | `pass` | 域随机化：设置关节阻尼 |
| `envs/scene_factory.py` | `_teleport_base()` | `pass` | 域随机化：重置机器人基座位置 |
| `envs/door_env.py` | `_sim_step()` | `pass` | 发送关节力矩 + 步进物理世界 |
| `envs/door_env.py` | `_read_physics_state()` | 返回全零 | 读取关节位置/速度/力矩、末端位姿、门角度 |
| `envs/contact_monitor.py` | `_read_raw_contacts()` | 返回空列表 | 获取 Isaac 接触力数据 |

### 3.2 🔴 关键缺失：训练主循环脚本

| 文件 | 当前状态 | 需要实现的内容 |
|------|----------|----------------|
| `scripts/train.py` | 仅打印 config path 后退出 | 创建环境 → 实例化 Actor/Critic/PPO → RolloutCollector 采样 → GAE → PPO update → CurriculumManager → 日志/checkpoint 循环 |
| `scripts/evaluate.py` | 占位 | 加载 checkpoint → rollout → 统计指标 |
| `scripts/rollout_demo.py` | 占位 | 可视化 rollout + 视频导出 |
| `scripts/export_policy.py` | 占位 | 模型导出（ONNX / TorchScript） |

### 3.3 🟡 重要缺失：IsaacLab 适配层

当前代码框架是**自建的 Gym-like 环境接口**（`BaseEnv` + `DoorInteractionEnv`），
而**不是**原生的 IsaacLab（`omni.isaac.lab`）Task/Environment 格式。

如果目标是在 IsaacLab 中训练，还需要以下适配工作：

| 需求 | 说明 |
|------|------|
| IsaacLab Task 注册 | 将环境封装为 `ManagerBasedRLEnv` 或 `DirectRLEnv`，或通过 `gymnasium.Env` 包装器适配 |
| IsaacLab Scene 配置 | 用 `InteractiveSceneCfg` 重新定义场景，替代自建 `SceneFactory` |
| IsaacLab 观测 Manager | 将 `actor_obs_builder` / `critic_obs_builder` 适配为 IsaacLab 的 `ObservationManager` |
| IsaacLab 奖励 Manager | 将 `reward_manager` 适配为 IsaacLab 的 `RewardManager` |
| IsaacLab 终止条件 | 将 `task_manager` 适配为 IsaacLab 的 `TerminationManager` |
| IsaacLab Curriculum | 适配 IsaacLab 的 `CurriculumManager` |
| IsaacLab 训练启动 | 使用 `rsl_rl` 或 `skrl` 等 IsaacLab 兼容的 RL 库，或保留自建 PPO 通过包装器对接 |

### 3.4 🟡 其他缺失项

| 项目 | 状态 | 说明 |
|------|------|------|
| 配置文件内容 | ⚠️ 占位 | 5 个 YAML 文件均为 `notes: "待补充"`，需要填入真实超参数 |
| `pyproject.toml` | ❌ 缺失 | 架构文档建议创建但尚未实现，无法 `pip install -e .` |
| 单元测试 | ❌ 缺失 | `tests/` 目录在文档中提及但实际不存在任何测试文件 |
| `evaluation/` 模块 | ❌ 缺失 | 架构文档中定义了该子模块但未创建 |
| 日志与 TensorBoard | ❌ 缺失 | 训练日志记录、模型 checkpoint 保存逻辑未实现 |

---

## 四、完成度总览

```
整体进度  ████████████████████░░░░░░░░░░  ~65%

按层拆分:
  策略网络 (policy/)         ██████████████████████████████  100%
  训练组件 (training/)       ██████████████████████████████  100%
  奖励系统 (rewards/)        ██████████████████████████████  100%
  观测构建 (observations/)   ██████████████████████████████  100%
  感知管线 (door_perception/)██████████████████████████████  100%
  资产文件 (assets/)         ██████████████████████████████  100%
  文档 (docs/)               ██████████████████████████████  100%
  环境层 (envs/)             ████████████░░░░░░░░░░░░░░░░░░  40%
  训练脚本 (scripts/)        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
  IsaacLab 适配              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
  配置参数                   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  10%
  测试                       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
```

**核心结论**：整个"软件算法层"（RL 训练器、神经网络、奖励函数、观测构建、感知管线）
已有完整的 Python/PyTorch 实现。缺失的本质是**"仿真桥接层"**——
即代码与 Isaac Sim/IsaacLab 物理仿真器之间的连接，以及训练主循环的编排。

---

## 五、推荐实现优先级

### 第一优先级（🔴 必须，否则无法训练）

1. **决定技术路线：自建环境 vs IsaacLab 原生**
   - **路线 A**：在自建的 `DoorInteractionEnv` 框架内填充 Isaac Sim API 调用（填充 `[ISAAC_API]` 占位）
   - **路线 B**：重写环境层为 IsaacLab 原生格式（`ManagerBasedRLEnv` / `DirectRLEnv`），复用现有奖励/观测逻辑
   - 建议选**路线 B**，因为 IsaacLab 提供了 GPU 并行化管线、域随机化和课程的原生支持

2. **实现仿真环境注册与场景加载**
   - 机器人 + 门 + 杯体在 Isaac 中的实例化
   - 物理步进与状态读取

3. **实现训练主循环** (`scripts/train.py`)
   - 环境创建 → Actor/Critic 实例化 → 采样 → GAE → PPO update → Curriculum → 日志 → Checkpoint

### 第二优先级（🟡 重要）

4. **填充配置文件真实参数**（PPO 超参数、奖励权重、课程阶段定义等）
5. **添加日志 / Checkpoint / TensorBoard 集成**
6. **实现评估脚本** (`scripts/evaluate.py`)

### 第三优先级（🟢 可后续）

7. 单元测试
8. 模型导出（ONNX / TorchScript）
9. Rollout 可视化与视频导出
10. `pyproject.toml` 与包管理

---

## 六、附录：文件清单与实现状态速查

### 全量 Python 源码文件状态

```
src/affordance_guided_interaction/
├── __init__.py                                  ✅ 实现
├── utils/
│   ├── __init__.py                              ✅ 实现
│   ├── paths.py                                 ✅ 实现
│   ├── runtime_env.py                           ✅ 实现
│   ├── runtime_timing.py                        ✅ 实现
│   ├── usd_assets.py                            ✅ 实现
│   └── usd_math.py                              ✅ 实现
├── door_perception/
│   ├── __init__.py                              ✅ 实现
│   ├── config.py                                ✅ 实现
│   ├── segmentation.py                          ✅ 实现
│   ├── depth_projection.py                      ✅ 实现
│   ├── frozen_encoder.py                        ✅ 实现 (~509行)
│   └── affordance_pipeline.py                   ✅ 实现 (~189行)
├── observations/
│   ├── __init__.py                              ✅ 实现
│   ├── actor_obs_builder.py                     ✅ 实现
│   ├── critic_obs_builder.py                    ✅ 实现
│   ├── stability_proxy.py                       ✅ 实现
│   └── history_buffer.py                        ✅ 实现
├── policy/
│   ├── __init__.py                              ✅ 实现
│   ├── actor.py                                 ✅ 实现 (~400行)
│   ├── critic.py                                ✅ 实现 (~229行)
│   ├── recurrent_backbone.py                    ✅ 实现
│   └── action_head.py                           ✅ 实现
├── rewards/
│   ├── __init__.py                              ✅ 实现
│   ├── reward_manager.py                        ✅ 实现 (~197行)
│   ├── task_reward.py                           ✅ 实现
│   ├── stability_reward.py                      ✅ 实现
│   └── safety_penalty.py                        ✅ 实现
├── training/
│   ├── __init__.py                              ✅ 实现
│   ├── ppo_trainer.py                           ✅ 实现 (~319行)
│   ├── rollout_buffer.py                        ✅ 实现 (~325行)
│   ├── rollout_collector.py                     ✅ 实现 (~320行)
│   ├── curriculum_manager.py                    ✅ 实现
│   ├── domain_randomizer.py                     ✅ 实现
│   └── metrics.py                               ✅ 实现
└── envs/
    ├── __init__.py                              ✅ 实现
    ├── base_env.py                              ✅ 实现 (~187行)
    ├── task_manager.py                          ✅ 实现
    ├── vec_env.py                               ✅ 实现
    ├── door_env.py                              ⚠️ 部分 (~375行，ISAAC_API 占位)
    ├── scene_factory.py                         ❌ 占位 (~272行，几乎全部 ISAAC_API)
    └── contact_monitor.py                       ⚠️ 部分 (ISAAC_API 占位)
```

### 脚本文件状态

```
scripts/
├── train.py                                     ❌ 占位（仅打印后退出）
├── evaluate.py                                  ❌ 占位
├── rollout_demo.py                              ❌ 占位
├── export_policy.py                             ❌ 占位
└── load_scene.py                                ✅ 实现（Isaac Sim 场景加载 + UI）
```

### 配置文件状态

```
configs/
├── env/default.yaml                             ⚠️ 占位（notes: "待补充"）
├── task/default.yaml                            ⚠️ 占位
├── policy/default.yaml                          ⚠️ 占位
├── training/default.yaml                        ⚠️ 占位
└── curriculum/default.yaml                      ⚠️ 占位
```
