# Affordance-Guided-Interaction 项目架构文档

## 1. 文档目的

本文档基于
`affordance_guided_door_interaction_conditional_stability_v11.md`
中的研究方案，为后续代码实现提供一份面向工程落地的项目架构设计。

需要先明确一点：当前仓库仍以研究方案为主，尚未形成实际代码骨架。因此本文档描述的不是“现有代码结构”，而是一个与 v11 方法论一致、可逐步实现的目标架构。

本文档重点回答四个问题：

1. 项目代码应拆成哪些核心模块
2. 各模块之间如何传递信息
3. 训练与部署时分别依赖哪些输入
4. 仓库目录应如何组织，便于后续扩展

---

## 2. 架构目标

根据 v11 的问题定义，项目应围绕以下目标构建：

1. 支持多种 door-related interaction，而不只是一类固定 push-door 任务
2. 将环境 affordance 表示与低层约束感知控制明确分层
3. 支持 `occupied / unoccupied` 两种上下文下的统一策略学习
4. 在训练中支持 asymmetric actor-critic、curriculum 和参数随机化
5. 允许 affordance 模块在不同阶段替换实现，而不破坏整体训练框架

对应地，工程架构上应坚持以下原则：

- 上层表示与下层控制解耦
- actor 与 critic 的观测边界清晰
- 环境、奖励、策略、训练器、配置系统分层独立
- 核心实验逻辑由配置驱动，而不是硬编码在脚本中
- affordance 模块保留可插拔接口，避免过早锁死实现形式

---

## 3. 总体架构

整体系统可拆为六层：

1. **仿真与任务层**
   负责 Isaac Sim 场景、机器人、门对象、按钮、把手、杯体、接触与状态更新。
2. **观测构建层**
   负责从仿真状态中整理 actor/critic 所需观测，并构造稳定性 proxy、上下文变量和历史缓存。
3. **Affordance / Progress 表示层**
   负责从视觉输入和任务目标生成 `z_aff` 与 `z_prog`。
4. **策略执行层**
   负责根据机器人状态、上下文约束和上层表示输出动作。
5. **奖励与约束层**
   负责主任务进展奖励、持杯稳定项、安全项和 occupancy mask。
6. **训练与评估层**
   负责 PPO、rollout、curriculum、域随机化、评估与日志记录。

可以将其理解为以下信息流：

```text
Isaac Sim 场景
   -> 原始状态/视觉观测
   -> 表示层生成 z_aff, z_prog
   -> 观测构建层拼装 actor_obs / critic_obs
   -> 策略网络输出 torque action
   -> 仿真执行与接触反馈
   -> 奖励层计算 reward / done / metrics
   -> PPO 训练器更新策略
```

这里最关键的边界是：

- **Affordance 模块负责“哪里可以做什么”**
- **策略模块负责“在当前约束下如何做”**

这与 v11 中“上层提供结构化交互表示，下层负责约束感知执行”的思想完全一致。

---

## 4. 核心模块设计

## 4.1 `envs`：仿真环境层

该层直接对接 Isaac Sim，用于封装机器人、门对象和任务逻辑。

建议职责如下：

- 创建 Franka 单臂、gripper、杯体和 door-related objects
- 管理对象实例化，如 push door、button door、handle door、sequential door
- 提供 reset、step、state query、contact query 等标准环境接口
- 维护任务完成信号，例如按钮是否触发、门是否部分打开、把手是否到位
- 对外暴露统一的底层状态，供 observation builder 和 reward 模块消费

建议拆分为：

- `base_env.py`
  定义统一环境生命周期与 Isaac Sim 交互接口
- `door_env.py`
  单个门交互环境主体
- `task_manager.py`
  管理当前 episode 的任务类型、goal 和阶段状态
- `scene_factory.py`
  负责装配门、按钮、把手、杯体等对象组合
- `contact_monitor.py`
  汇总 link 级接触、冲击强度和安全事件

这一层不应直接包含 PPO 或策略更新逻辑，只负责环境物理与任务状态。

## 4.2 `perception`：Affordance / Progress 表示层

该层是整个系统的上层表示模块，对应 v11 中尚未完全定型的 affordance 模块。

其统一接口建议为：

```text
输入:
- 原始视觉观测
- task goal
- 可选历史帧

输出:
- z_aff: 对象 affordance 表示
- z_prog: 任务进展表示
- optional aux predictions: 分区域分类、进展头、可视化结果
```

为了适应不同实验阶段，应将其设计为可替换后端，而不是单一实现：

1. **OracleAffordanceEncoder**
   直接从仿真标注或对象元数据构造 affordance/progress，适合早期验证下层控制。
2. **LearnedAffordanceEncoder**
   从 RGB、深度或点云中预测 `z_aff` 和 `z_prog`。
3. **HybridAffordanceEncoder**
   一部分由规则或标注提供，一部分由网络预测，适合过渡阶段。

建议拆分为：

- `interfaces.py`
  定义 `encode_affordance()`、`encode_progress()` 的抽象接口
- `oracle_encoder.py`
  早期实验使用的 oracle 版本
- `vision_encoder.py`
  视觉 backbone
- `affordance_head.py`
  affordance 表示头
- `progress_head.py`
  任务进展表示头

这样可以在不修改 `policy` 和 `training` 代码的前提下，切换表示层实现。

## 4.3 `observations`：观测构建层

该层负责把环境底层状态加工成 actor 和 critic 真正接收的输入。

其职责包括：

- 拼装 proprioception：`q`、`dq`、`tau(optional)`、历史动作
- 提取 gripper pose / velocity
- 构造稳定性 proxy：tilt、线加速度、角速度、角加速度、jerk proxy
- 注入上下文：`occupied`
- 合并 `z_aff`、`z_prog`
- 构建可选接触摘要
- 为 critic 额外加入 privileged information

建议拆分为：

- `actor_obs_builder.py`
- `critic_obs_builder.py`
- `history_buffer.py`
- `stability_proxy.py`

其中 `stability_proxy.py` 非常关键，因为它直接承载 v11 中“尽量使用部署可得信息表达末端稳定性”的设计原则。

## 4.4 `policy`：约束感知执行层

该层对应 v11 中的低层执行策略，输入结构化表示与机器人状态，输出关节力矩动作。

推荐结构：

- `recurrent_backbone.py`
  使用 GRU 或 LSTM 编码历史信息
- `actor.py`
  输出 torque action
- `critic.py`
  基于 asymmetric observation 输出 value
- `action_head.py`
  力矩分布参数化与动作裁剪

建议策略输入划分为四组：

1. **机器人本体状态**
   `q`、`dq`、`tau(optional)`、past actions
2. **末端相关状态**
   gripper pose / velocity、stability proxies
3. **任务上下文**
   `occupied`
4. **上层表示**
   `z_aff`、`z_prog`

这层不应直接读取原始视觉图像，也不应依赖 `cup_mass`、`door_mass` 等部署时难获得参数。

## 4.5 `rewards`：奖励与安全约束层

奖励模块应独立于环境和策略实现，以保证实验可比较、权重可配置。

建议拆分为：

- `task_reward.py`
  按 affordance 类型计算主任务进展奖励
- `stability_reward.py`
  计算持杯稳定项
- `safety_penalty.py`
  计算碰撞、关节限位、力矩过大等惩罚
- `reward_manager.py`
  聚合各项奖励并输出分项日志

奖励结构建议遵循：

```text
total_reward =
  r_task_progress
+ m_occ * lambda_stab * r_carry_stability
+ r_effective_contact
- r_self_collision
- r_joint_limit
- r_torque_penalty
```

其中 `m_occ` 由 `occupied` 决定，只在持杯场景中激活稳定项。

## 4.6 `training`：训练与实验管理层

这一层负责 PPO 训练主循环与实验编排。

建议拆分为：

- `ppo_trainer.py`
  参数更新、loss 计算、梯度裁剪
- `rollout_collector.py`
  多环境并行采样
- `rollout_buffer.py`
  PPO 所需轨迹缓存
- `curriculum_manager.py`
  控制 Stage 1 到 Stage 5 的课程切换
- `domain_randomizer.py`
  随机化杯体、门、按钮、摩擦、噪声等参数
- `metrics.py`
  统计 success rate、stability score、collision rate

该层应以配置驱动，不应把课程阶段、奖励权重和对象随机化范围写死在脚本中。

## 4.7 `evaluation`：评估与可视化层

该层用于统一管理评估实验与结果导出。

建议包含：

- 任务成功率统计
- occupied / unoccupied 分桶评估
- 不同 affordance 类型的分项成功率
- 持杯稳定性指标统计
- 接触分布可视化
- 策略 rollout 视频导出

这样才能验证项目是否真的学到了“约束感知的 affordance-conditioned interaction”，而不是只在单一门实例上过拟合。

---

## 5. 训练态与部署态数据流

## 5.1 训练态

训练时建议采用以下流程：

1. 环境 reset，随机采样任务类型、对象参数和上下文
2. 仿真输出原始状态与视觉观测
3. 表示层生成 `z_aff` 与 `z_prog`
4. observation builder 生成 `actor_obs` 与 `critic_obs`
5. actor 输出 torque action
6. 环境执行动作并返回下一状态
7. reward manager 计算奖励分项
8. critic 使用 privileged information 学习 value
9. PPO trainer 完成参数更新

训练态中的关键设计：

- actor 只看现实可得信息
- critic 可额外看精确接触、对象状态和隐藏参数
- curriculum manager 控制任务难度递进
- domain randomizer 防止策略退化为单实例记忆

## 5.2 部署态

部署态应比训练态更简单：

1. 获取现实可得观测
2. 通过 affordance/progress 模块生成结构化表示
3. 拼装 actor observation
4. 输出动作

部署态必须移除 critic 与 privileged information，确保训练部署边界清晰。

---

## 6. 建议目录结构

建议以 Python 包形式组织项目，目录如下：

```text
Affordance-Guided-Interaction/
├── README.md
├── affordance_guided_door_interaction_conditional_stability_v11.md
├── project_architecture.md
├── pyproject.toml
├── configs/
│   ├── env/
│   ├── task/
│   ├── policy/
│   ├── reward/
│   ├── training/
│   └── curriculum/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── rollout_demo.py
│   └── export_policy.py
├── assets/
│   ├── doors/
│   ├── buttons/
│   ├── handles/
│   └── cups/
├── src/
│   └── affordance_guided_interaction/
│       ├── envs/
│       ├── perception/
│       ├── observations/
│       ├── policy/
│       ├── rewards/
│       ├── training/
│       ├── evaluation/
│       ├── utils/
│       └── common/
├── tests/
│   ├── test_observations.py
│   ├── test_rewards.py
│   ├── test_policy_io.py
│   └── test_curriculum.py
└── outputs/
    ├── logs/
    ├── checkpoints/
    └── videos/
```

其中各目录职责如下：

- `configs/`
  存放实验配置，是所有训练与评估脚本的入口
- `scripts/`
  只放轻量启动脚本，不承载核心逻辑
- `assets/`
  存放场景资产和对象元数据
- `src/affordance_guided_interaction/`
  项目核心源码
- `tests/`
  对观测构建、奖励计算、输入输出维度等进行单元测试
- `outputs/`
  训练输出目录，不应提交核心产物到源码目录

---

## 7. 配置系统建议

为了避免实验代码迅速失控，建议尽早建立配置分层。

至少应支持以下配置类别：

1. **环境配置**
   机器人、场景、物体集合、物理步长
2. **任务配置**
   affordance 类型、goal、episode 长度、成功阈值
3. **上下文配置**
   `occupied` 采样比例
4. **奖励配置**
   稳定项、接触项、安全项权重
5. **训练配置**
   PPO 超参数、并行环境数、序列长度
6. **课程配置**
   各阶段启用任务与切换条件
7. **随机化配置**
   杯体、门、按钮、噪声、摩擦等随机化范围

建议所有实验都通过配置文件组合启动，而不是在 `train.py` 中写大量 `if/else`。

---

## 8. 关键接口定义建议

为避免模块耦合，建议尽早统一几个核心数据接口。

## 8.1 Actor Observation

```text
actor_obs = {
  "proprio": ...,
  "gripper_state": ...,
  "context": {
    "occupied": ...
  },
  "stability_proxy": ...,
  "z_aff": ...,
  "z_prog": ...,
  "contact_summary": optional
}
```

## 8.2 Critic Observation

```text
critic_obs = {
  "actor_obs": ...,
  "privileged": {
    "object_state": ...,
    "contact_state": ...,
    "hidden_params": ...,
    "affordance_distances": ...
  }
}
```

## 8.3 Reward Breakdown

```text
reward_info = {
  "task_progress": ...,
  "carry_stability": ...,
  "effective_contact": ...,
  "safety_penalty": ...,
  "total_reward": ...
}
```

这类接口一旦提前统一，后续替换表示层、奖励项或训练器时会容易很多。

---

## 9. 推荐实现顺序

由于 v11 中的 affordance 模块尚未完全定型，项目不宜一开始就把所有模块同时做满。推荐按以下顺序推进：

### 阶段 A：先打通最小训练闭环

- 建立基础 Isaac Sim 环境
- 先只支持单一 push-affordance
- 使用 oracle `z_aff` 与 `z_prog`
- 跑通 PPO + recurrent actor + asymmetric critic

目标是先验证“表示层与执行层分离”的框架是否可训练。

### 阶段 B：补稳定性约束与上下文

- 引入 `occupied`
- 加入持杯稳定 proxy
- 实现 occupancy mask 下的 stability reward
- 验证 occupied / unoccupied 两种行为模式是否分化

目标是先验证“同一策略在不同上下文下切换行为”的可行性。

### 阶段 C：扩展 affordance 类型

- 增加 press affordance
- 增加 handle affordance
- 增加 sequential interaction

目标是让任务从单一交互扩展为统一 door-related interaction。

### 阶段 D：替换上层表示实现

- 从 oracle encoder 迁移到 learned encoder
- 比较不同 `z_aff` / `z_prog` 表示方式的性能差异

目标是把系统从“研究框架验证”推进到“真实 affordance 学习系统”。

---

## 10. 风险与预留接口

当前方案中最不确定的部分有三类，因此工程上必须留接口：

1. **Affordance 模块实现未定**
   需要通过接口解耦，不应让策略依赖某一种固定视觉网络。
2. **任务进展表示形式可能变化**
   `z_prog` 既可能是离散阶段，也可能是连续 embedding，应避免写死维度假设。
3. **稳定性 proxy 的最佳组合仍需实验验证**
   gripper acceleration、angular acceleration、tilt、jerk proxy 的具体组合应由配置控制。

因此，项目初版最重要的不是“写出最复杂的网络”，而是先建立：

- 可插拔表示层
- 可配置奖励层
- 可比较的训练评估闭环

---

## 11. 总结

基于 v11，项目最合理的代码架构不是围绕“某个推门控制器”组织，而是围绕一个统一的
**affordance-guided interaction framework** 来组织。

从工程实现角度看，整个仓库应至少分成五个稳定边界：

1. `envs`：场景与任务
2. `perception`：`z_aff` / `z_prog`
3. `observations`：actor/critic 输入构建
4. `policy + training`：约束感知策略学习
5. `rewards + evaluation`：优化目标与实验验证

这样的结构有两个直接好处：

- 能与 v11 的理论分层保持一致
- 能支持项目从 oracle 表示逐步演化到真实 affordance 学习

如果后续开始实际搭代码，建议优先落地
“`envs + observations + rewards + PPO` 最小闭环”，
再逐步替换上层 affordance 模块。
