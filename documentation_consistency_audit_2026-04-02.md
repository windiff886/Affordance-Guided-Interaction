# 项目文档一致性审查报告

审查日期：2026-04-02

审查对象：`/home/windiff/Code/Affordance-Guided-Interaction`

审查重点：

- 项目内所有主要文档之间是否存在描述出入
- 文档与当前源码/配置是否一致
- 尤其关注数学公式、数学建模、数学过程、参数定义、维度定义、课程学习判据、奖励函数、任务成功判定、感知接口

---

## 1. 审查结论摘要

本项目当前的文档体系并不是“同一套系统的不同说明层次”，而是混杂了至少三类内容：

1. **设计草案/研究设想文档**
   - 例如 `affordance_layer_rewritten_single_visual_v5.md`
   - 更偏研究建模与方案设想

2. **过渡期技术说明文档**
   - 例如 `project_architecture.md`
   - 一部分描述的是目标架构，一部分描述的是实现方向，不完全等同于当前代码

3. **当前实现附近的模块文档**
   - 例如 `src/affordance_guided_interaction/*/README.md`
   - 其中一部分与代码接近，但也有明显漂移

其中最严重的问题不在“措辞不同”，而在于以下几类核心建模对象出现了**多套定义并存**：

- 同一个数学量在不同文档里有不同数值
- 同一个公式在文档里与代码接线的变量语义不同
- 同一个训练流程在文档里和实际实现里使用了不同的统计量
- 同一个任务空间在课程学习文档、配置文件、环境能力里定义不同
- 同一个观测/奖励/感知接口在不同文档里维度、坐标系、输出结构都不完全一致

从风险角度看，当前最危险的不是历史文档还存在，而是这些文档没有被明确标注为“设计草案”或“历史分析”，导致读者很容易把它们误当作“当前实现规范”。

---

## 2. 审查范围

### 2.1 审查的文档文件

本次审查覆盖以下 15 份主要文档：

1. `README.md`
2. `project_architecture.md`
3. `training_pipeline_detailed.md`
4. `project_completion_analysis.md`
5. `affordance_layer_rewritten_single_visual_v5.md`
6. `src/teleop_cup_grasp/README.md`
7. `src/affordance_guided_interaction/door_perception/README.md`
8. `src/affordance_guided_interaction/envs/README.md`
9. `assets/minimal_push_door/README.md`
10. `assets/grasp_objects/README.md`
11. `src/affordance_guided_interaction/rewards/README.md`
12. `src/affordance_guided_interaction/policy/README.md`
13. `assets/robot/README.md`
14. `src/affordance_guided_interaction/observations/README.md`
15. `src/affordance_guided_interaction/training/README.md`

### 2.2 对照的源码/配置

本次审查还交叉核对了以下代码与配置：

- `configs/env/default.yaml`
- `configs/task/default.yaml`
- `configs/reward/default.yaml`
- `configs/training/default.yaml`
- `configs/curriculum/default.yaml`
- `configs/policy/default.yaml`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/export_policy.py`
- `scripts/rollout_demo.py`
- `src/affordance_guided_interaction/envs/*.py`
- `src/affordance_guided_interaction/observations/*.py`
- `src/affordance_guided_interaction/policy/*.py`
- `src/affordance_guided_interaction/rewards/*.py`
- `src/affordance_guided_interaction/training/*.py`
- `src/affordance_guided_interaction/door_perception/*.py`

---

## 3. 审查方法

本次审查使用的是**文档-源码静态对照法**，主要检查以下一致性：

1. **术语一致性**
   - 同一概念是否在不同文档中使用不同名字

2. **数学符号一致性**
   - 同一符号是否表示不同物理量
   - 同一物理量是否在不同文档中使用不同符号

3. **参数一致性**
   - 阈值、权重、维度、默认值是否一致

4. **接口一致性**
   - 文档中的输入输出结构、shape、字段含义是否和代码一致

5. **流程一致性**
   - 文档写的训练流程、环境流程、课程流程是否真正按代码执行

6. **状态定位**
   - 文档是在描述“目标架构”还是“当前实现”
   - 是否显式标注为历史文档/方案文档

本次审查是**静态检查**，没有运行训练或 Isaac Sim 仿真。因此结论集中在“文档如何描述系统”和“代码接口实际是什么”之间的偏差，而不是运行时数值结果本身。

---

## 4. 总体判断：问题集中在哪些地方

### 4.1 最严重的区域

严重程度最高的区域集中在：

1. `src/affordance_guided_interaction/rewards/README.md`
2. `training_pipeline_detailed.md`
3. `src/affordance_guided_interaction/training/README.md`
4. `src/affordance_guided_interaction/door_perception/README.md`
5. `project_completion_analysis.md`

原因：

- 这些文档都在定义核心建模对象
- 且它们容易被读者当作“当前系统规范”
- 但它们与源码/配置之间存在多处非局部冲突

### 4.2 中等问题区域

中等问题区域主要在：

1. `src/affordance_guided_interaction/observations/README.md`
2. `src/affordance_guided_interaction/policy/README.md`
3. `src/affordance_guided_interaction/envs/README.md`
4. `README.md`

这些文档整体方向与当前实现较接近，但存在若干关键漂移。

### 4.3 轻微问题区域

轻微问题主要在资产类文档：

1. `assets/minimal_push_door/README.md`
2. `assets/robot/README.md`
3. `assets/grasp_objects/README.md`
4. `src/teleop_cup_grasp/README.md`

这些问题多是路径拼写、状态说明不够精确，或与当前主训练框架存在上下文脱节，但不属于核心数学建模错误。

---

## 5. 关键总问题一览

这里先列出跨文档、跨模块的核心问题。后文再展开。

1. **奖励函数中的数学量定义与实际实现接线不一致**
2. **任务成功阈值 `theta_target` 同时存在 1.05 / 1.2 / 1.57 三套版本**
3. **课程学习的任务类型定义不统一**
4. **课程跃迁判据文档写成功率，代码却用 `mean_reward` 近似**
5. **感知接口 `z_aff / z_prog / door_embedding` 的结构和接入状态没有统一说法**
6. **观测与 privileged 信息的坐标系、pose 排列顺序描述不一致**
7. **维度分析文档存在明显过期值**
8. **若干文档实际上描述的是“目标方案”而不是“当前实现”，但没有明确标注**
9. **`project_completion_analysis.md` 明显过时，且会误导读者判断项目状态**
10. **若干小文档存在路径错误或局部术语漂移**

---

## 6. 详细问题：奖励数学与建模不一致

这是本次审查中最严重的一组问题。

### 6.1 文档中的稳定性奖励数学定义

`src/affordance_guided_interaction/rewards/README.md` 将稳定性相关量定义为：

- 线加速度
  - `src/affordance_guided_interaction/rewards/README.md:121`
- 角加速度
  - `src/affordance_guided_interaction/rewards/README.md:122`
- jerk proxy
  - `src/affordance_guided_interaction/rewards/README.md:123`
- tilt
  - `src/affordance_guided_interaction/rewards/README.md:124`

其具体数学过程是：

- \( \mathbf{a}_t = (\mathbf{v}_t - \mathbf{v}_{t-1})/\Delta t \)
- \( \boldsymbol{\alpha}_t = (\boldsymbol{\omega}_t - \boldsymbol{\omega}_{t-1})/\Delta t \)
- `tilt` 通过 \( R_{EE}^\top g \) 的 xy 投影得到

相关位置：

- `src/affordance_guided_interaction/rewards/README.md:121`
- `src/affordance_guided_interaction/rewards/README.md:122`
- `src/affordance_guided_interaction/rewards/README.md:131`
- `src/affordance_guided_interaction/rewards/README.md:137`

### 6.2 奖励实现函数本身的假设

`compute_stability_reward()` 也确实按这些量来算：

- `lin_acc_sq = dot(lin_acc, lin_acc)`
- `ang_acc_sq = dot(ang_acc, ang_acc)`
- `tilt_sq = dot(tilt_xy, tilt_xy)`

参考：

- `src/affordance_guided_interaction/rewards/stability_reward.py:51`
- `src/affordance_guided_interaction/rewards/stability_reward.py:52`
- `src/affordance_guided_interaction/rewards/stability_reward.py:53`

所以**从奖励 README 到奖励函数实现本身，是相对一致的**。

### 6.3 真正的断裂点：环境传给奖励函数的不是这些量

`DoorInteractionEnv._compute_reward()` 实际上传给 `RewardManager.step()` 的稳定性 proxy 是：

- `lin_acc = state["left_ee_linear_velocity"]`
- `ang_acc = state["left_ee_angular_velocity"]`
- `tilt_xy = zeros(2)`

参考：

- `src/affordance_guided_interaction/envs/door_env.py:505`
- `src/affordance_guided_interaction/envs/door_env.py:507`
- `src/affordance_guided_interaction/envs/door_env.py:508`
- `src/affordance_guided_interaction/envs/door_env.py:509`
- `src/affordance_guided_interaction/envs/door_env.py:512`
- `src/affordance_guided_interaction/envs/door_env.py:514`
- `src/affordance_guided_interaction/envs/door_env.py:515`

这意味着：

1. 文档中的 \( \mathbf a_t \) 被实现里替换成了速度 `v`
2. 文档中的 \( \boldsymbol\alpha_t \) 被实现里替换成了角速度 `\omega`
3. 文档中的 `tilt` 在当前接线里永远是 0

这是一个**建模级断裂**，不是轻微误差。

### 6.4 这会带来的后果

从数学上讲，当前实现中的稳定性奖励景观已经不是文档描述的那个对象：

- “惩罚高加速度”变成了“惩罚高速度”
- “鼓励低加速度接近 0 的高斯核”变成了“鼓励低速度”
- “倾斜惩罚”在当前路径上完全失效

因此：

- 奖励 README 不能被视为当前实现规范
- 训练流程文档里对稳定性 reward 的解释不能直接映射到代码
- 如果研究人员根据 README 解释实验结果，很容易得出错误结论

### 6.5 observations 模块其实已经有正确的 proxy

`observations/stability_proxy.py` 确实实现了文档想表达的那套差分量：

- 线加速度差分
  - `src/affordance_guided_interaction/observations/stability_proxy.py:149`
- 角加速度差分
  - `src/affordance_guided_interaction/observations/stability_proxy.py:155`
- jerk proxy
  - `src/affordance_guided_interaction/observations/stability_proxy.py:162`
- tilt
  - `src/affordance_guided_interaction/observations/stability_proxy.py:142`

这进一步说明问题不是“系统没有这些量”，而是**奖励接线没有用这些量**。

---

## 7. 详细问题：任务成功阈值和进度归一化存在三套版本

### 7.1 版本一：1.2 rad

用于任务成功判定的多个地方写的是 `1.2 rad`：

- `configs/task/default.yaml:12`
- `src/affordance_guided_interaction/envs/task_manager.py:61`
- `src/affordance_guided_interaction/envs/task_manager.py:68`
- `src/affordance_guided_interaction/envs/README.md:210`
- `src/affordance_guided_interaction/envs/base_env.py:67`

这套定义控制的是：

- episode 何时 `success`
- 何时 `done`
- 环境文档中的任务成功条件

### 7.2 版本二：1.57 rad

另一套文档/代码写的是 `1.57 rad`：

- `configs/reward/default.yaml:12`
- `src/affordance_guided_interaction/door_perception/affordance_pipeline.py:169`
- `src/affordance_guided_interaction/door_perception/affordance_pipeline.py:173`

这里控制的是：

- reward 中 `theta_target`
- `z_prog.progress = min(door_angle / 1.57, 1.0)`

这代表：

- 成功 bonus 的触发阈值
- progress 归一化终点

### 7.3 版本三：1.05 rad

`training_pipeline_detailed.md` 中又给了一套参考值：

- `training_pipeline_detailed.md:549`

文档写的是：

- `θ_target = 1.05 (~60°)`
- `w_delta = 10.0`
- `alpha = 0.1`
- `k_decay = 0.5`
- `w_open = 50.0`

### 7.4 这三套版本为什么是严重问题

这不是参数表不同步那么简单，因为同一个训练系统里以下几个对象理论上应该共享一套目标角度：

1. 任务成功判定
2. episode 终止条件
3. 主任务奖励中成功 bonus
4. 进展奖励的衰减起点
5. `progress` 归一化的满刻度

当前系统中它们被分裂成了：

- `done/success` 用 1.2
- reward 默认配置用 1.57
- 训练流程说明用 1.05

这意味着：

- 同一篇文档中的“成功”未必和代码中的“成功”是同一个事件
- reward 曲线与 episode 终止逻辑可能描述不同
- `progress=1.0` 未必等于“任务判定成功”

从数学建模角度看，这会直接破坏“任务目标函数”的单一性。

---

## 8. 详细问题：课程学习的任务空间定义不统一

### 8.1 `training/README.md` 的定义

`src/affordance_guided_interaction/training/README.md` 中：

- Stage 3 写成 `Push + Press`
  - `src/affordance_guided_interaction/training/README.md:184`
- Stage 4 写成 `Button+Door, Handle+Door`
  - `src/affordance_guided_interaction/training/README.md:185`

这是一个带有“按钮按压”和“时序组合”的任务空间定义。

### 8.2 根 README 和详细训练流程文档的定义

`README.md` 与 `training_pipeline_detailed.md` 则写成：

- Stage 3：`push + pull`
  - `README.md:172`
  - `training_pipeline_detailed.md:867`
- Stage 4：`handle_push, handle_pull`
  - `training_pipeline_detailed.md:868`

这是一套“push/pull/handle”门交互类型定义。

### 8.3 `CurriculumManager` 的定义

代码中的 `CurriculumManager` 实际定义为：

- Stage 1: `["push"]`
- Stage 2: `["push"]`
- Stage 3: `["push", "pull"]`
- Stage 4: `["handle_push", "handle_pull"]`
- Stage 5: `["push", "pull", "handle_push", "handle_pull"]`

参考：

- `src/affordance_guided_interaction/training/curriculum_manager.py:31`
- `src/affordance_guided_interaction/training/curriculum_manager.py:50`
- `src/affordance_guided_interaction/training/curriculum_manager.py:57`
- `src/affordance_guided_interaction/training/curriculum_manager.py:64`

### 8.4 环境能力实际只支持 push

更严重的是，环境装配层当前只支持 push 门：

- 文件头明确写“当前仅支持 push”
  - `src/affordance_guided_interaction/envs/scene_factory.py:8`
- `_DOOR_ASSET_MAP` 只有 `"push"`
  - `src/affordance_guided_interaction/envs/scene_factory.py:66`
- 未知门型会回退到 push
  - `src/affordance_guided_interaction/envs/scene_factory.py:369`

### 8.5 结论

当前关于课程学习任务空间，至少存在三层不一致：

1. 一套文档认为 Stage 3/4 是 `press/button` 型时序任务
2. 另一套文档和课程配置认为 Stage 3/4 是 `push/pull/handle` 型任务
3. 环境实现实际上目前只有 `push`

这意味着：

- 课程学习文档不能直接作为实验设置依据
- 训练详细流程文档描述的“课程任务复杂度递进”没有在环境能力层闭合
- 项目当前并没有统一的“任务族定义”

---

## 9. 详细问题：课程跃迁的数学判据与训练实现不一致

### 9.1 文档中的数学判据

`training/README.md` 和 `training_pipeline_detailed.md` 都把课程跃迁定义为：

\[
\frac{1}{M}\sum_{e=E-M+1}^E \eta_e \ge \eta_{\text{thresh}}
\]

其中：

- \( \eta_e \) 是第 `e` 个 epoch 的成功率
- `window_size = 50`
- `threshold = 0.8`

参考：

- `src/affordance_guided_interaction/training/README.md:190`
- `src/affordance_guided_interaction/training/README.md:200`
- `training_pipeline_detailed.md:875`
- `training_pipeline_detailed.md:879`

`CurriculumManager` 代码本身也按这个定义实现：

- `src/affordance_guided_interaction/training/curriculum_manager.py:73`
- `src/affordance_guided_interaction/training/curriculum_manager.py:91`

### 9.2 训练脚本中的实际做法

`scripts/train.py` 的课程推进却写成：

- 读取 `collect/mean_reward`
- 然后 `approx_success_rate = min(mean_reward, 1.0)`
- 再交给 `curriculum.report_epoch()`

参考：

- `scripts/train.py:447`
- `scripts/train.py:451`
- `scripts/train.py:452`

### 9.3 为什么这属于数学建模偏差

“成功率”与“平均奖励”不是同一个统计量：

- 成功率是事件比例
- 平均奖励是标量回报的均值

即便把 reward 限制到 `[0,1]`，它们也不等价；更不用说当前 reward 明显不是伯努利变量。

因此：

- 文档中的课程判据是“基于任务完成率”
- 代码中的课程判据近似变成了“基于平均 reward”

这会导致：

- 课程推进条件和文档解释不一致
- 奖励 shaping 的变化会影响 stage 跃迁
- 课程学习不再只由任务掌握程度控制

### 9.4 更不合理之处

项目内部已经有真正的成功率聚合器：

- `src/affordance_guided_interaction/training/metrics.py:50`
- `src/affordance_guided_interaction/training/metrics.py:137`
- `src/affordance_guided_interaction/training/metrics.py:163`

也就是说，文档想要的统计量在代码基础设施里其实存在，但训练脚本没有真正使用它。

---

## 10. 详细问题：感知接口 `z_aff / z_prog / door_embedding` 说法不统一

### 10.1 v5 总体设计文档的说法

`affordance_layer_rewritten_single_visual_v5.md` 明确说：

- 当前版本 Affordance 层只输出统一视觉 latent
- 不再分别输出 task-progress representation 和 object-affordance representation

参考：

- `affordance_layer_rewritten_single_visual_v5.md:224`
- `affordance_layer_rewritten_single_visual_v5.md:226`

### 10.2 observations 文档的说法

`src/affordance_guided_interaction/observations/README.md` 也延续了这个方向：

- 只强调 `door_embedding (768,)`
- 明确写“不单独区分任务进展（z_prog）与 affordance 表示（z_aff）”

参考：

- `src/affordance_guided_interaction/observations/README.md:75`
- `src/affordance_guided_interaction/observations/README.md:83`
- `src/affordance_guided_interaction/observations/README.md:341`

### 10.3 door_perception 文档的说法

但 `src/affordance_guided_interaction/door_perception/README.md` 又写成：

- Embeddings 拼接任务进展 `z_prog` 再进入 Policy
  - `src/affordance_guided_interaction/door_perception/README.md:24`
- `encode()` 返回 `(z_aff_embedding, z_prog_dict)`
  - `src/affordance_guided_interaction/door_perception/README.md:74`
- 还举了 `(512,)` 维的例子
  - `src/affordance_guided_interaction/door_perception/README.md:77`

### 10.4 代码里的实际情况

代码层面：

1. `PointMAEEncoderConfig.embed_dim = trans_dim * 2 = 768`
   - `src/affordance_guided_interaction/door_perception/config.py:65`
2. `pipeline.encode()` 确实返回 `(z_aff, z_prog)`
   - `src/affordance_guided_interaction/door_perception/affordance_pipeline.py:77`
3. 但环境当前并未集成 `AffordancePipeline`
   - `training_pipeline_detailed.md:141`
   - `src/affordance_guided_interaction/envs/door_env.py:399`

### 10.5 结论

当前项目中关于感知接口，至少存在三种不同状态描述：

1. **统一单输出 latent 版本**
2. **双输出 `(z_aff, z_prog)` 版本**
3. **环境尚未接入感知管线，仅保留 `door_embedding=None` 占位版本**

这三种状态都在文档里出现了，但没有明确说明“哪一个是当前实现，哪一个是设计方向”。

### 10.6 额外问题：维度示例错误

`door_perception/README.md` 中 `z_aff_feature` 举的是 `(512,)` 维，但代码与其他文档都以 768 为准：

- 错误示例
  - `src/affordance_guided_interaction/door_perception/README.md:77`
- 768 维证据
  - `src/affordance_guided_interaction/policy/README.md:49`
  - `src/affordance_guided_interaction/observations/README.md:75`
  - `src/affordance_guided_interaction/policy/actor.py:42`

---

## 11. 详细问题：观测、坐标系、pose 排列顺序不一致

### 11.1 文档中的坐标系描述

`observations/README.md` 和 `actor_obs_builder.py` 文档字符串都说：

- gripper position/orientation/velocity 在 `base_link` 坐标系下

参考：

- `src/affordance_guided_interaction/observations/README.md:51`
- `src/affordance_guided_interaction/observations/README.md:53`
- `src/affordance_guided_interaction/observations/actor_obs_builder.py:159`
- `src/affordance_guided_interaction/observations/actor_obs_builder.py:163`

### 11.2 环境实际读取的是 world 系状态

`door_env.py` 中读取的是：

- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

参考：

- `src/affordance_guided_interaction/envs/door_env.py:306`
- `src/affordance_guided_interaction/envs/door_env.py:307`
- `src/affordance_guided_interaction/envs/door_env.py:308`
- `src/affordance_guided_interaction/envs/door_env.py:309`

这说明当前实现真正传给 observation builder 的是**世界系量**，而不是文档反复描述的 `base_link` 系量。

### 11.3 pose 排列顺序不一致

文档说 `door_pose/cup_pose` 是：

- `pos(3) + quat(4)`

参考：

- `src/affordance_guided_interaction/observations/README.md:100`
- `src/affordance_guided_interaction/observations/README.md:102`
- `src/affordance_guided_interaction/observations/critic_obs_builder.py:11`

但 `door_env.py` 实际拼接的是：

- `door_pose = concat([door_root_quat, door_root_pos])`
  - `src/affordance_guided_interaction/envs/door_env.py:348`
- `cup_pose = concat([cup_quat_t, cup_pos_t])`
  - `src/affordance_guided_interaction/envs/door_env.py:364`

也就是实际格式是：

- `quat(4) + pos(3)`

### 11.4 这为什么重要

如果文档把 `(7,)` pose 的语义写错，那么：

- 任何依赖文档写数据转换代码的人都会把位置和四元数拆错
- critic 的 privileged 张量解释会出错
- 这类错误通常很难从 shape 上看出来

这属于典型的“形状一致但语义错误”问题。

---

## 12. 详细问题：安全惩罚的输入语义不一致

### 12.1 文档与实现函数的预期

`envs/README.md` 和 `safety_penalty.py` 都把 `contact_forces` 描述成“力大小/幅值”：

- `src/affordance_guided_interaction/envs/README.md:77`
- `src/affordance_guided_interaction/rewards/safety_penalty.py:30`

### 12.2 ContactMonitor 实际返回的是向量

`ContactMonitor` 的 `link_forces` 实际存的是 `(3,)` 力向量：

- `src/affordance_guided_interaction/envs/contact_monitor.py:32`
- `src/affordance_guided_interaction/envs/contact_monitor.py:33`
- `src/affordance_guided_interaction/envs/contact_monitor.py:123`

### 12.3 环境直接透传给奖励层

- `src/affordance_guided_interaction/envs/door_env.py:531`

这导致 `safety_penalty.py` 内部：

- `invalid_force_sum += force`

语义上已经和“标量接触力幅值”不匹配。

### 12.4 文档中的其他安全项也未完整接线

`RewardManager.step()` 支持：

- `joint_pos`
- `joint_vel`
- `joint_limits`
- `joint_vel_limits`
- `affordance_links`

但环境 `_compute_reward()` 当前并未提供这些量：

- `src/affordance_guided_interaction/envs/door_env.py:522`
- `src/affordance_guided_interaction/envs/door_env.py:533`

因此文档中关于：

- 关节限位惩罚
- 速度限位惩罚
- 有效 affordance 区域过滤

虽然在奖励 README 中写得完整，但当前环境接线并没有按那个“完整版本”闭环。

---

## 13. 详细问题：维度分析和默认超参数存在过期值

### 13.1 `training_pipeline_detailed.md` 中的维度不一致

该文档写了：

- `stability = 24`
  - `training_pipeline_detailed.md:174`
- `privileged_dim = 30`
  - `training_pipeline_detailed.md:175`
- 后文又写 `stability ~24`
  - `training_pipeline_detailed.md:376`

但当前代码真实维度是：

- `stability = 40`
  - `src/affordance_guided_interaction/policy/actor.py:56`
  - `src/affordance_guided_interaction/training/rollout_buffer.py:35`
- `privileged = 28`
  - `src/affordance_guided_interaction/policy/critic.py:31`
  - `src/affordance_guided_interaction/training/rollout_buffer.py:37`

### 13.2 `training/README.md` 中的训练规模默认值过期

该文档写的是：

- `N_env = 1024`
  - `src/affordance_guided_interaction/training/README.md:319`
- `T = 24`
  - `src/affordance_guided_interaction/training/README.md:320`

而当前配置是：

- `num_envs = 64`
  - `configs/training/default.yaml:8`
- `n_steps_per_rollout = 128`
  - `configs/training/default.yaml:10`

### 13.3 这类问题的影响

这类问题表面上不如奖励/课程问题严重，但它会让读者：

- 错估显存占用
- 错估 buffer 大小
- 错误理解 rollout 组织方式
- 错误估计一个迭代包含多少样本

对于训练系统文档来说，这是实质性偏差。

---

## 14. 详细问题：`rewards/README.md` 自身内部存在版本混杂

### 14.1 总公式与文字解释不完全一致

总公式写的是：

\[
r_t = r_{task} + ... - r_{safe}
\]

参考：

- `src/affordance_guided_interaction/rewards/README.md:41`

但其后文字又写：

- “所有惩罚信号（包括安全惩罚）都会被 `s_t` 缩放”
  - `src/affordance_guided_interaction/rewards/README.md:48`

而实际代码是：

- 只对稳定性 penalty 乘 `s_t`
- 安全惩罚不乘 `s_t`

参考：

- `src/affordance_guided_interaction/rewards/reward_manager.py:150`
- `src/affordance_guided_interaction/rewards/reward_manager.py:157`
- `src/affordance_guided_interaction/rewards/reward_manager.py:172`

### 14.2 参数表中还残留未使用项

参数表中还列出了：

- `w_term`
  - `src/affordance_guided_interaction/rewards/README.md:375`
- `v_thresh`
  - `src/affordance_guided_interaction/rewards/README.md:386`
- `beta_5`
  - `src/affordance_guided_interaction/rewards/README.md:387`

当前配置和代码没有这些对等项。

### 14.3 `s_min` 数值也不一致

README 表里写：

- `s_min = 0.05`
  - `src/affordance_guided_interaction/rewards/README.md:394`

当前配置写：

- `s_min = 0.1`
  - `configs/reward/default.yaml:44`

### 14.4 结论

`rewards/README.md` 不是单纯“落后于代码”，而是内部也夹杂了不同版本的设计信息。

---

## 15. 详细问题：`project_completion_analysis.md` 已明显过时

### 15.1 该文档对训练脚本状态的判断已经错误

文档声称：

- `scripts/train.py` 仅打印 config path 后退出
  - `project_completion_analysis.md:139`
- `scripts/evaluate.py` 占位
  - `project_completion_analysis.md:140`
- `scripts/rollout_demo.py` 占位
  - `project_completion_analysis.md:141`
- `scripts/export_policy.py` 占位
  - `project_completion_analysis.md:142`

实际情况是：

- `scripts/train.py` 已经实现了完整训练主循环
  - `scripts/train.py:265`
  - `scripts/train.py:401`
- 其余三个脚本确实仍是占位
  - `scripts/evaluate.py:8`
  - `scripts/rollout_demo.py:7`
  - `scripts/export_policy.py:7`

### 15.2 该文档对配置状态的判断也已过时

文档声称：

- 5 个 YAML 文件均为占位
  - `project_completion_analysis.md:165`

但当前 `configs/*.yaml` 已经填有真实参数，例如：

- `configs/training/default.yaml:7`
- `configs/reward/default.yaml:7`
- `configs/env/default.yaml:6`

### 15.3 这份文档的定位问题

这份文档如果保留，应该被明确标成：

- 某一历史日期下的完成度快照
- 或“已过时分析”

否则它会直接误导读者对项目现状的判断。

---

## 16. 详细问题：`project_architecture.md` 更像目标架构，不是当前实现规范

### 16.1 奖励结构与当前实现不同

文档中建议的奖励结构是：

```text
total_reward =
  r_task_progress
+ m_occ * lambda_stab * r_carry_stability
+ r_effective_contact
- r_invalid_collision
- r_self_collision
- r_joint_limit
- r_torque_penalty
```

参考：

- `project_architecture.md:218`
- `project_architecture.md:221`

这与当前实现的结构：

- `r_task + masked(bonus + s_t * penalty) - r_safe`

并不相同：

- `src/affordance_guided_interaction/rewards/reward_manager.py:4`
- `src/affordance_guided_interaction/reward_manager.py:172`

### 16.2 文档中引用了当前仓库并不存在的文件/目录

文档提到了：

- `affordance_guided_door_interaction_conditional_stability_v11.md`
  - `project_architecture.md:6`
  - `project_architecture.md:312`
- `pyproject.toml`
  - `project_architecture.md:314`
- `evaluation/`
  - `project_architecture.md:340`
- `tests/`
  - `project_architecture.md:343`

当前仓库里这些均不存在：

- `affordance_guided_door_interaction_conditional_stability_v11.md` 不存在
- `pyproject.toml` 不存在
- `evaluation/` 不存在
- `tests/` 不存在

### 16.3 结论

`project_architecture.md` 更适合被理解为“架构设计蓝图/目标结构”，而不是“当前仓库结构说明”。

---

## 17. 详细问题：`policy/README.md` 有局部实现描述错误

### 17.1 动作裁剪位置写错

`policy/README.md` 写道：

- `action_head.py` 执行力矩截断操作
  - `src/affordance_guided_interaction/policy/README.md:98`
  - `src/affordance_guided_interaction/policy/README.md:100`

但 `action_head.py` 自己明确写：

- 不进行力矩 clip，力矩限制由仿真环境层负责
  - `src/affordance_guided_interaction/policy/action_head.py:5`

实际环境裁剪发生在：

- `src/affordance_guided_interaction/envs/door_env.py:156`

### 17.2 其余主要内容基本接近当前实现

除上述问题外，`policy/README.md` 关于：

- `door_embedding = 768`
- 双臂 12 维动作
- recurrent actor
- asymmetric critic

整体与当前实现大体一致。

---

## 18. 详细问题：`observations/README.md` 整体较好，但与环境接线存在若干关键漂移

### 18.1 优点

这份 README 在模块级文档中属于较清晰的一份：

- 对 actor/critic 分层说明清楚
- 对稳定性 proxy 的数学过程描述较完整
- 对 768 维 `door_embedding` 的说明与主代码基本一致

### 18.2 主要问题

仍然存在以下关键偏差：

1. **坐标系写为 base_link，但环境读的是 world**
2. **pose 排列写成 pos+quat，但环境实际是 quat+pos**
3. **文档声称不再单独维护 z_prog，但 `door_perception` 代码仍会构建 z_prog**

因此它不能单独作为“完全正确的接口规范”。

---

## 19. 详细问题：`training_pipeline_detailed.md` 是信息最多的文档，但也混入了最多过渡状态

### 19.1 优点

这份文档的优点是：

- 数据流梳理非常完整
- GAE/PPO/TBPTT 数学过程写得详细
- 训练阶段拆解最细

### 19.2 主要问题类型

这份文档的问题主要不是“完全错误”，而是混合了三类状态：

1. **当前实现**
2. **预期未来接入路径**
3. **较早一版参数/维度**

例如：

- 它正确指出 `AffordancePipeline` 尚未接入环境
  - `training_pipeline_detailed.md:141`
- 但又在其他地方使用了过期的维度/超参数
  - `training_pipeline_detailed.md:174`
  - `training_pipeline_detailed.md:549`

### 19.3 结论

这份文档适合作为“综合技术说明”，但需要大规模校订，否则它很容易把“当前实现”和“计划接入方案”混写在一起。

---

## 20. 详细问题：根 README 的信息层级合理，但存在课程/实现状态漂移

### 20.1 优点

根 README 的整体结构清晰：

- 安装
- 训练
- 配置
- 课程学习
- 项目结构

作为项目入口文档，它是可读的。

### 20.2 问题

主要问题是：

1. 课程学习表写了 Stage 3-5 的完整任务族
   - `README.md:168`
2. 但又补充“当前版本仅训练 push 门（Stage 1-2）”
   - `README.md:176`

这个表述虽然试图缓和冲突，但并没有明确说明：

- Stage 3-5 是设计目标
- 还是代码里已经配置但环境尚未支持

因此仍会让读者误以为完整五阶段任务空间已落地。

---

## 21. 详细问题：资产文档和辅助文档的轻微错误

### 21.1 `assets/minimal_push_door/README.md`

存在路径拼写错误：

- 写成了 `asserts/minimal_push_door/...`
  - `assets/minimal_push_door/README.md:5`
  - `assets/minimal_push_door/README.md:9`

应为 `assets/...`

### 21.2 `assets/robot/README.md`

目录树也写成了：

- `asserts/robot/`
  - `assets/robot/README.md:17`

同样是路径拼写错误。

### 21.3 `src/teleop_cup_grasp/README.md`

该文档更像早期任务规划说明，而不是当前训练主线文档。它与主训练系统的数学建模关系不大，但不属于错误最严重的文档。

### 21.4 `assets/grasp_objects/README.md`

这份文档整体与 `scripts/load_scene.py` 的用途较贴近，问题相对少。主要是它描述的是场景初始化/抓取脚本逻辑，不应被误解为 RL 训练核心接口。

---

## 22. 按“当前可信度”给文档分层

下面给出一个更实用的判断：如果有人要理解“当前系统到底是什么”，哪些文档更可信。

### 22.1 高可信度，但仍需修订

这些文档整体最接近当前实现，但仍有明显漂移：

1. `src/affordance_guided_interaction/observations/README.md`
2. `src/affordance_guided_interaction/policy/README.md`
3. `src/affordance_guided_interaction/envs/README.md`
4. `README.md`

### 22.2 中可信度，信息多但混入过渡状态

1. `training_pipeline_detailed.md`
2. `src/affordance_guided_interaction/training/README.md`
3. `src/affordance_guided_interaction/rewards/README.md`
4. `src/affordance_guided_interaction/door_perception/README.md`

### 22.3 低可信度，不适合当作“当前实现规范”

1. `project_completion_analysis.md`
2. `project_architecture.md`
3. `affordance_layer_rewritten_single_visual_v5.md`

其中：

- `project_completion_analysis.md` 是明显过时的项目状态分析
- `project_architecture.md` 更像目标架构图
- `affordance_layer_rewritten_single_visual_v5.md` 更像研究方案/理论设计稿

---

## 23. 建议修订优先级

如果要把项目文档体系整理干净，建议按下面顺序处理。

### 优先级 P0：必须先修

1. `src/affordance_guided_interaction/rewards/README.md`
   - 因为这里直接涉及奖励数学定义
   - 当前最容易误导实验解释

2. `training_pipeline_detailed.md`
   - 信息量最大
   - 也是最容易被当成权威技术文档的一份

3. `src/affordance_guided_interaction/training/README.md`
   - 课程任务空间与当前代码定义冲突明显

4. `src/affordance_guided_interaction/door_perception/README.md`
   - `z_prog` 和 embedding 维度说法冲突

### 优先级 P1：紧接着修

1. `README.md`
2. `src/affordance_guided_interaction/observations/README.md`
3. `src/affordance_guided_interaction/policy/README.md`
4. `src/affordance_guided_interaction/envs/README.md`

### 优先级 P2：明确标注定位即可

1. `project_architecture.md`
   - 标注为“目标架构设计稿”

2. `affordance_layer_rewritten_single_visual_v5.md`
   - 标注为“研究方案文档/设计草案”

3. `project_completion_analysis.md`
   - 标注为“历史状态分析（已过时）”

### 优先级 P3：顺手修复

1. `assets/minimal_push_door/README.md`
2. `assets/robot/README.md`
3. 其余资产类 README 的小路径和措辞问题

---

## 24. 推荐的文档治理策略

为了避免以后再次出现这种“多版本混写”的情况，建议采用以下治理策略。

### 24.1 区分三类文档

建议把所有文档显式分成三类：

1. **Current Implementation**
   - 当前实现规范
   - 必须与代码/配置一致

2. **Design / Proposal**
   - 目标架构
   - 可先于代码
   - 允许和当前实现不同，但必须明确说明“尚未实现”

3. **Historical Analysis**
   - 历史快照
   - 不作为现状依据

### 24.2 对涉及数学建模的文档增加“单一真值源”

对于以下对象，必须指定单一来源：

1. `theta_target`
2. reward 权重
3. 课程阶段定义
4. actor/critic 输入维度
5. privileged 信息维度
6. `door_embedding` 维度
7. `progress` 的定义

最合理的做法是：

- 默认值来自 `configs/*.yaml`
- 代码实现从配置读取
- README 只引用配置，不再手工维护另一套数值

### 24.3 对“当前未接入”的模块强制加状态说明

例如：

- `AffordancePipeline` 是否已进入环境主环
- Stage 3-5 环境是否真实可运行
- `evaluate/export/rollout_demo` 是否已实现

建议在文档显著位置统一写：

- `Status: implemented`
- `Status: partially implemented`
- `Status: design only`

---

## 25. 最终结论

这次审查的最终结论如下：

1. 项目文档不是简单的“存在一些文字差异”，而是存在**系统级定义漂移**
2. 漂移最严重的地方是：
   - 奖励数学
   - 课程学习任务建模
   - 任务成功阈值
   - 感知接口定义
3. 当前很多文档没有明确说明自己是在描述：
   - 当前实现
   - 未来设计
   - 历史状态
4. 因此读者很容易把不同阶段的建模版本混成一套系统理解
5. 从工程角度看，应该尽快把“当前实现规范”和“设计草案”分开

最核心的一句话可以概括为：

> 当前项目的主要问题不是“缺少文档”，而是“文档中同时存在多套系统定义，且没有明确标注边界”。

---

## 26. 附录：本次审查中重点引用的源码/文档位置

### 奖励相关

- `src/affordance_guided_interaction/rewards/README.md:41`
- `src/affordance_guided_interaction/rewards/README.md:121`
- `src/affordance_guided_interaction/rewards/README.md:137`
- `src/affordance_guided_interaction/rewards/stability_reward.py:51`
- `src/affordance_guided_interaction/rewards/reward_manager.py:150`
- `src/affordance_guided_interaction/rewards/reward_manager.py:172`
- `src/affordance_guided_interaction/rewards/task_reward.py:42`

### 环境与任务判定

- `configs/task/default.yaml:12`
- `src/affordance_guided_interaction/envs/task_manager.py:68`
- `src/affordance_guided_interaction/envs/README.md:210`
- `src/affordance_guided_interaction/envs/door_env.py:505`
- `src/affordance_guided_interaction/envs/contact_monitor.py:123`

### 感知相关

- `src/affordance_guided_interaction/door_perception/README.md:24`
- `src/affordance_guided_interaction/door_perception/README.md:77`
- `src/affordance_guided_interaction/door_perception/config.py:65`
- `src/affordance_guided_interaction/door_perception/affordance_pipeline.py:77`
- `src/affordance_guided_interaction/door_perception/affordance_pipeline.py:169`

### 观测与策略

- `src/affordance_guided_interaction/observations/README.md:75`
- `src/affordance_guided_interaction/observations/README.md:83`
- `src/affordance_guided_interaction/observations/README.md:100`
- `src/affordance_guided_interaction/policy/README.md:49`
- `src/affordance_guided_interaction/policy/README.md:98`
- `src/affordance_guided_interaction/policy/action_head.py:5`
- `src/affordance_guided_interaction/policy/critic.py:31`

### 训练与课程学习

- `README.md:172`
- `README.md:176`
- `src/affordance_guided_interaction/training/README.md:184`
- `src/affordance_guided_interaction/training/README.md:190`
- `src/affordance_guided_interaction/training/curriculum_manager.py:31`
- `src/affordance_guided_interaction/training/curriculum_manager.py:50`
- `training_pipeline_detailed.md:174`
- `training_pipeline_detailed.md:549`
- `training_pipeline_detailed.md:867`
- `scripts/train.py:451`

### 项目状态/架构文档

- `project_architecture.md:218`
- `project_architecture.md:312`
- `project_completion_analysis.md:139`
- `project_completion_analysis.md:165`
- `affordance_layer_rewritten_single_visual_v5.md:226`

