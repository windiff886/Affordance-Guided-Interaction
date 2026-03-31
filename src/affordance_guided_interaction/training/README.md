# Training 模块开发计划 (Training Module Design)

## 1. 模块职责 (Module Responsibilities)

`training` 层是整个系统的**总调度与实验编排中心**。
根据 `project_architecture.md` 与 affordance 理论 V5 版本的设计原则，本模块不处理具体的物理接触和视觉编码，而是聚焦于：**如何高效、稳定地通过强化学习（RL）找出满足约束条件的门交互策略**。

它的核心职责包括：
1. **策略优化**：使用 PPO (Proximal Policy Optimization) 更新网络参数。
2. **轨迹收集**：管理多环境的高度并行 Rollout 采样。
3. **课程推进**：管理并自动切换从简单交互到复杂持杯混合任务的训练阶段（Curriculum）。
4. **域随机化**：在每个 Reset 周期中注入随机性，防止 Actor 过拟合单一环境实例。

---

## 2. 核心算法特征 (Core Algorithm Features)

根据 V5 理论设定，本模块将支持以下高级训练特性：

### 2.1 Asymmetric Actor-Critic (非对称 Actor-Critic)
* **Actor (部署态执行者)**：只接收现实世界中绝对可靠的观测信息（本体状态、执行臂与持杯臂的 Gripper Pose、统一视觉表征 $z_{\text{aff}}$、Context）。
* **Critic (训练态价值评估者)**：具备全局视野，额外接收精确对象状态、隐藏门阻尼、杯体质量、精确距离等 Privileged Information（特权信息），从而极大加速 Value 网络的收敛。

### 2.2 Recurrent Policy (循环神经网络策略)
* 由于环境部分可观测（POMDP），且 Actor 看不到 `door_mass`、`cup_fill_ratio` 等隐藏参数，必须在 Actor 和 Critic 的 Backbone 中引入 **GRU** 或 **LSTM**。
* 依靠隐状态 (Hidden State) 的时序记忆能力，让策略在交互过程中去“探测”和隐式辨识环境动力学模型。

---

## 3. 文件架构与组件拆分 (Component Breakdown)

该模块将遵循“配置驱动 (Configuration-Driven)”原则，代码结构计划拆分为以下几个核心文件：

### 3.1 `ppo_trainer.py`
包含 PPO 的主更新循环。
* **职责**：计算 Actor 的 Surrogate Loss、Critic 的 Value Loss，处理 GAE (Generalized Advantage Estimation)，执行梯度裁剪并反向传播。
* **设计原则**：保持其数学纯粹性，尽量不参杂特定任务的逻辑。对包含 Recurrent 层的 Batch Tensor 进行截断时间反向传播 (BPTT)。

### 3.2 `rollout_collector.py`
管理并行环境与交互步进。
* **职责**：驱动多个 Isaac Sim 环境（如 1024 个并行 Env）执行 `step()`。
* **设计原则**：协调 `observations` 构建模块和 `policy` 推理层，分离收集 Asymmetric Critic 需要的 Privileged Obs 与 Actor 需要的 Standard Obs。

### 3.3 `rollout_buffer.py`
PPO 算法所需的轨迹缓存。
* **职责**：存储 state, action, reward, hidden_states, value, log_prob，并支持按序列 (Sequence) 维度进行 mini-batch 采样，以满足 RNN 需要。

### 3.4 `curriculum_manager.py`
控制训练阶段演进（详见下文第 4 节）。
* **职责**：在 `evaluate` 指标（如成功率连续 M 个 Epoch 超过阈值）达标时，触发下一阶段。修改当前启用的门类型、持杯概率（Occupied Mask）等配置。

### 3.5 `domain_randomizer.py`
环境隐性资产变异管理。
* **职责**：控制每次 Episode Reset 时采样的参数。涵盖：杯体质量、液体填充率、门阻尼、按钮刚度、动作执行与观测噪声等。

### 3.6 `metrics.py`
关键指标记录与 TensorBoard 输出。
* **职责**：对 Reward Manager 中的日志进行时间平滑运算，统计 `success_rate`（按类别）、`drop_rate`、平均 `stability_penalty` 和安全违规次数，反馈给 Curriculum Manager。

---

## 4. 课程学习设计 (Curriculum Design)

根据 V5 理论中避免策略陷入局部最优的要求，`curriculum_manager.py` 将实现 **5 阶段自动跃迁**：

* **Stage 1: 基础起步 (单一 Affordance)**
  * 上下文：`occupied = 0`（完全空手，关闭所有稳定性约束）。
  * 任务：仅针对一种简单门（如直接 Push 门板），不要求多步骤。
  * 目标：让 Agent 学会基本的视觉引导接触，以及跑通表示层到控制层的网络闭环。

* **Stage 2: 引入持杯约束 (持杯稳定)**
  * 上下文：`occupied = 1`（单臂持杯，激活 $r_{\text{carry-stability}}$ 与 $s_t$）。
  * 任务：依然是基础推门任务。
  * 目标：让 Agent 体验到因为动作太猛烈而被惩罚，学会在保证杯体末端极低线/角加速度的前提下缓慢接触门。

* **Stage 3: 视觉认知增强 (两类 Affordance 混合)**
  * 设定：引入 Push 与 Press（按压按钮）两种操作。
  * 目标：促使 $z_{\text{aff}}$ (视觉表征) 发挥作用，让 Agent 知道该如何根据视觉反馈区分这两种目标并调整接触部位。

* **Stage 4: 拓扑结构进阶 (复杂门交互)**
  * 设定：包含 Button + Door 或 Handle + Door 等具有时序顺序 (Sequential Interaction) 的操作。
  * 目标：让 Agent 学习“先按再推”的子策略组合，依靠 RNN 的长程记忆跨越由于等待门开带来的 Reward Delay。

* **Stage 5: 全域泛化 (统一混合训练)**
  * 设定：`occupied` 以特定概率随机采样为 0 或 1。包含所有的 Affordance 种类和高强度的 Domain Randomization。
  * 目标：打磨出能够在复杂约束和任意上下文中自由穿梭，自主决定身体哪个具体 Link 发挥作用的健壮策略。

---

## 5. 数据流向示例 (Data Flow Illustration)

在 `ppo_trainer.py` 开始一次更新（Epoch）的过程如下：

1. **收集阶段**：`rollout_collector.py` 循环执行 N 步。
   * 对于每一个 `step()`，获取 $\text{obs}_{actor}$ 送给 Actor，前向传播获取 Action。
   * 环境推进一步，由 Reward 层吐出 `total_reward`。
   * 同步收集 $\text{obs}_{critic}$（含 Privileged Info）并送给 Critic 估算 `Value`。
   * 将数据放入 `rollout_buffer.py`。
2. **域随机化拦截**：如果某环境触发 `done = True`，`domain_randomizer.py` 首先对该环境的物理材质进行重置，生成新的 `door_mass` 和 `cup_fill_ratio`。
3. **指标审计**：`metrics.py` 收集此次 Rollout 的成功率。
4. **课程切换判定**：`curriculum_manager.py` 判断是否满足跃迁条件，若是，更新 `TaskContext` 发往相关并行环境。
5. **策略更新**：`ppo_trainer.py` 取出 Buffer 中的多步序列数据，分别在 Actor 和 Critic 上反向传播执行 PPO-Clip 优化。

---

## 6. 开发与测试建议 (Development Notes)

1. **分离 PPO 引擎**：早期测试时，可以先用 RSL-RL 等经过验证的三方 PPO 工具替换底层的梯度计算代码，只编写针对 Asymmetric Actor-Critic 和 RNN 的外壳接口，提高工程容错率。
2. **监控先行**：在写完 PPO 前，先把 `metrics.py` 和 TensorBoard 挂载好。尤其是我们要重点评估 `occupied=0` 和 `occupied=1` 两次 Rollout 之下，策略动作剧烈程度的数值差异。
3. **不要将参数写死**：诸如 PPO 的 `clip_param`, `entropy_coef`, 以及 Stage 切换的超参数必须全部放到 `configs/training/default.yaml` 中，保持 Training 模块代码体的干净。
