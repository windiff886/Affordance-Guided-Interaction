# 项目文档一致性审查报告

审查日期：2026-04-02
复核整理：2026-04-03

审查对象：`/home/windiff/Code/Affordance-Guided-Interaction`

说明：

- 本版本已删除**确认修复**的问题条目。
- 文档现在只保留**截至当前仓库状态仍然成立**的审计问题。
- 本次结论基于仓库内当前文件的静态核对，不包含 Isaac Sim 运行时动态验证。

---

## 1. 审查结论摘要

与最初审查相比，奖励建模、成功阈值、课程跃迁判据、感知接口、坐标系/pose 顺序、安全惩罚语义、维度过期值、`policy/README.md` 动作裁剪描述等一批核心问题已经被修复，因此已从本报告中移除。

当前仍然存在的主要问题，已经从“核心数学/接口断裂”收缩为以下几类：

1. **课程学习相关文档仍未统一到当前真实环境能力**
2. **`training_pipeline_detailed.md` 仍混入过渡状态和旧接口名**
3. **`project_completion_analysis.md` 明显过时**
4. **`project_architecture.md` 更像目标架构蓝图，不是当前实现规范**
5. **若干资产 README 仍有轻微路径错误**

当前最需要继续修的不是奖励或观测接口，而是**训练体系说明文档**和**项目定位文档**。

---

## 2. 审查范围

本轮保留审查重点为以下文档：

1. `README.md`
2. `project_architecture.md`
3. `training_pipeline_detailed.md`
4. `project_completion_analysis.md`
5. `src/affordance_guided_interaction/training/README.md`
6. `src/affordance_guided_interaction/envs/README.md`
7. `assets/minimal_push_door/README.md`
8. `assets/robot/README.md`

交叉对照的当前实现与配置包括：

- `configs/curriculum/default.yaml`
- `configs/task/default.yaml`
- `configs/training/default.yaml`
- `scripts/train.py`
- `src/affordance_guided_interaction/training/curriculum_manager.py`
- `src/affordance_guided_interaction/envs/scene_factory.py`
- `src/affordance_guided_interaction/door_perception/affordance_pipeline.py`
- `src/affordance_guided_interaction/envs/door_env.py`

---

## 3. 审查方法

本次审查继续采用**文档-源码静态对照法**，重点检查：

1. 术语与接口是否仍指向当前代码中的真实对象
2. 课程学习的阶段定义是否与 `CurriculumManager` 和环境能力一致
3. 历史分析文档是否还在描述已不存在的仓库状态
4. 目标架构文档是否被误当成“当前实现说明”
5. 路径、目录、模块名等基础信息是否仍存在拼写级错误

---

## 4. 当前仍有效的问题总览

截至当前仓库状态，仍应保留的审计问题有 5 组：

1. **课程学习任务空间定义不统一，且超出当前环境能力**
2. **`training_pipeline_detailed.md` 仍混入旧字段名与过时接线说明**
3. **`project_completion_analysis.md` 对项目状态的判断已明显错误**
4. **`project_architecture.md` 仍是目标结构蓝图，不适合作为现状规范**
5. **资产 README 仍有路径拼写错误**

这些问题不会像早期奖励/接口问题那样直接破坏训练逻辑，但会显著误导读者对“当前系统是什么、当前能跑什么”的理解。

---

## 5. 详细问题：课程学习任务空间定义仍未统一

### 5.1 `training/README.md` 的阶段定义

`src/affordance_guided_interaction/training/README.md` 当前仍写：

- Stage 3 = `Push + Press`
  - `src/affordance_guided_interaction/training/README.md:184`
- Stage 4 = `Button+Door, Handle+Door`
  - `src/affordance_guided_interaction/training/README.md:185`

这是一套带有 `press` / `button` 时序任务的课程定义。

### 5.2 根 README 的阶段定义

根 `README.md` 当前写的是：

- Stage 3 = `push + pull`
  - `README.md:172`
- Stage 4 = `handle`
  - `README.md:173`
- Stage 5 = `全部`
  - `README.md:174`
- 但同时又写“当前版本仅训练 push 门（Stage 1-2）”
  - `README.md:176`

这说明根 README 一边展示五阶段全任务族，一边又承认当前只跑前两阶段。

### 5.3 `CurriculumManager` 的真实定义

代码中的 `CurriculumManager` 当前定义为：

- Stage 1: `["push"]`
  - `src/affordance_guided_interaction/training/curriculum_manager.py:36`
- Stage 2: `["push"]`
  - `src/affordance_guided_interaction/training/curriculum_manager.py:43`
- Stage 3: `["push", "pull"]`
  - `src/affordance_guided_interaction/training/curriculum_manager.py:50`
- Stage 4: `["handle_push", "handle_pull"]`
  - `src/affordance_guided_interaction/training/curriculum_manager.py:57`
- Stage 5: `["push", "pull", "handle_push", "handle_pull"]`
  - `src/affordance_guided_interaction/training/curriculum_manager.py:64`

这与 `training/README.md` 的 `Push + Press / Button+Door` 方案并不一致。

### 5.4 环境能力仍只支持 push

环境装配层当前仍明确只支持 push：

- 文件头写“当前仅支持 push”
  - `src/affordance_guided_interaction/envs/scene_factory.py:8`
- 资产映射 `_DOOR_ASSET_MAP` 只有 `"push"`
  - `src/affordance_guided_interaction/envs/scene_factory.py:65`
  - `src/affordance_guided_interaction/envs/scene_factory.py:66`
- 未知门型回退到 push
  - `src/affordance_guided_interaction/envs/scene_factory.py:374`

同时 `envs/README.md` 还写成：

- “根据当前课程阶段决定生成哪种类型的门（push / button / handle）”
  - `src/affordance_guided_interaction/envs/README.md:165`

这也没有对齐当前代码能力。

### 5.5 结论

当前课程学习相关文档至少仍有三层定义并存：

1. `training/README.md`：`press/button` 型任务族
2. `README.md` / `CurriculumManager`：`push/pull/handle` 型任务族
3. `scene_factory.py`：环境实际只支持 `push`

因此课程学习部分的文档仍不能直接被当作“当前实验设置说明”。

---

## 6. 详细问题：`training_pipeline_detailed.md` 仍混入过渡状态

### 6.1 已修部分

这份文档里此前关于维度的明显过期值已经修正，例如：

- `stability = 40`
  - `training_pipeline_detailed.md:174`
- `privileged_dim = 28`
  - `training_pipeline_detailed.md:175`

但它仍然混入多处旧接口名和旧接线说明。

### 6.2 仍存在的旧字段名

文档当前仍使用：

- `door_embedding`
  - `training_pipeline_detailed.md:111`
  - `training_pipeline_detailed.md:126`
  - `training_pipeline_detailed.md:142`
  - `training_pipeline_detailed.md:143`
  - `training_pipeline_detailed.md:148`
  - `training_pipeline_detailed.md:150`
  - `training_pipeline_detailed.md:328`

但当前代码主线接口已经统一为 `z_aff`，不再使用 `door_embedding`。

### 6.3 接触事件类型仍写成旧语义

文档仍写：

- `link_forces dict[str, float]`
  - `training_pipeline_detailed.md:114`

但当前 `ContactMonitor` 实际维护的是 `(3,)` 接触力向量，且奖励层也不再直接消费接触力标量。

### 6.4 感知接入状态描述仍过时

文档当前仍写：

- 环境把 `door_embedding` 放进 state dict，当前值为 `None`
  - `training_pipeline_detailed.md:142`
- 直接传给 `ActorObsBuilder.build(door_embedding=...)`
  - `training_pipeline_detailed.md:143`
- 环境主循环未调用 `AffordancePipeline.encode()`
  - `training_pipeline_detailed.md:144`

这些表述已经不再对应当前接口命名，且会把读者带回旧版接线状态。

### 6.5 仍残留历史超参数

文档里还保留一套旧主任务奖励参考值：

- `w_delta = 10.0`
- `alpha = 0.1`
- `k_decay = 0.5`
- `w_open = 50.0`
- `θ_target = 1.05`
  - `training_pipeline_detailed.md:543`
  - `training_pipeline_detailed.md:549`

这套值与当前 `configs/reward/default.yaml` 已不一致。

### 6.6 结论

`training_pipeline_detailed.md` 当前仍是“信息量最大，但版本混杂最多”的文档。
它不能再被视为当前实现的权威说明，除非继续做一次系统性清理。

---

## 7. 详细问题：`project_completion_analysis.md` 已明显过时

这份文档当前仍在描述一个远早于当前仓库状态的实现阶段。

### 7.1 对环境层状态的判断已错误

文档仍写：

- `_sim_step()` 为 `pass`
  - `project_completion_analysis.md:131`
- `_read_physics_state()` 返回全零
  - `project_completion_analysis.md:132`
- `_read_raw_contacts()` 返回空列表
  - `project_completion_analysis.md:133`

这些判断与当前代码状态已经不符。

### 7.2 对训练脚本状态的判断已错误

文档仍写：

- `scripts/train.py` 仅打印 config path 后退出
  - `project_completion_analysis.md:139`

但当前 `scripts/train.py` 已经实现完整训练主循环。

### 7.3 对配置与测试状态的判断已错误

文档仍写：

- 5 个 YAML 文件均为占位
  - `project_completion_analysis.md:165`
- `tests/` 目录缺失
  - `project_completion_analysis.md:165` 附近语义块

但当前配置文件已有真实参数，仓库里也存在测试目录和多组回归测试。

### 7.4 仍有效的部分

这份文档关于以下内容仍部分成立：

- `scripts/evaluate.py` 仍是占位
- `scripts/rollout_demo.py` 仍是占位
- `scripts/export_policy.py` 仍是占位

但由于整体判断基线已经严重过时，这份文档已不适合作为当前项目状态分析。

### 7.5 结论

`project_completion_analysis.md` 如果继续保留，应该明确标注为：

- 历史快照
- 已过时分析

否则会直接误导读者判断当前完成度。

---

## 8. 详细问题：`project_architecture.md` 是目标架构蓝图，不是现状说明

### 8.1 奖励结构仍是另一套目标方案

文档中仍写有：

- `r_effective_contact`
  - `project_architecture.md:222`
- `r_torque_penalty`
  - `project_architecture.md:225`

这不是当前仓库实现的奖励结构，而是另一套更偏目标方案的设计。

### 8.2 目录结构仍引用当前仓库不存在的对象

`project_architecture.md` 的建议目录树仍包含：

- `affordance_guided_door_interaction_conditional_stability_v11.md`
  - `project_architecture.md:6`
  - `project_architecture.md:311`
- `pyproject.toml`
  - `project_architecture.md:313`
- `evaluation/`
  - `project_architecture.md:339`

这些对象在当前仓库中并不存在。

### 8.3 结论

这份文档更适合被理解为：

- 目标架构设计稿
- 未来理想结构提案

而不是“当前仓库结构和当前实现规范”。

---

## 9. 详细问题：资产文档和辅助文档仍有轻微错误

### 9.1 `assets/minimal_push_door/README.md`

仍写成：

- `asserts/minimal_push_door/...`
  - `assets/minimal_push_door/README.md:5`
  - `assets/minimal_push_door/README.md:9`

应为 `assets/...`。

### 9.2 `assets/robot/README.md`

目录树仍写成：

- `asserts/robot/`
  - `assets/robot/README.md:17`

同样是路径拼写错误。

### 9.3 结论

这些问题不影响主训练逻辑，但会损害文档可信度，应顺手修复。

---

## 10. 按当前可信度给文档分层

### 10.1 高可信度

这些文档当前与代码最接近，可作为现状入口：

1. `src/affordance_guided_interaction/observations/README.md`
2. `src/affordance_guided_interaction/policy/README.md`
3. `src/affordance_guided_interaction/envs/README.md`
4. `src/affordance_guided_interaction/rewards/README.md`

### 10.2 中可信度

这些文档信息量大，但仍混入历史状态或设计目标：

1. `README.md`
2. `src/affordance_guided_interaction/training/README.md`
3. `training_pipeline_detailed.md`
4. `src/affordance_guided_interaction/door_perception/README.md`

### 10.3 低可信度

这些文档不应被当成“当前实现规范”：

1. `project_completion_analysis.md`
2. `project_architecture.md`
3. `affordance_layer_rewritten_single_visual_v5.md`

---

## 11. 建议修订优先级

### 优先级 P0：当前最该修

1. `src/affordance_guided_interaction/training/README.md`
   - 课程学习任务空间定义仍与 `CurriculumManager`/环境能力冲突
2. `README.md`
   - 五阶段课程表与“当前仅训练 Stage 1-2”并存，容易误导
3. `training_pipeline_detailed.md`
   - 仍混入 `door_embedding`、旧超参数、旧接线说明

### 优先级 P1：应明确定位

1. `project_completion_analysis.md`
   - 标注为“历史状态分析（已过时）”
2. `project_architecture.md`
   - 标注为“目标架构设计稿”

### 优先级 P2：顺手修

1. `assets/minimal_push_door/README.md`
2. `assets/robot/README.md`

---

## 12. 最终结论

当前仓库中，最危险的一批文档一致性问题已经被修掉了；剩余问题主要集中在**训练体系说明仍未收口**和**历史/设计型文档没有明确定位**。

最核心的一句话可以概括为：

> 当前项目的文档问题，已经从“核心数学和接口定义冲突”收缩为“训练说明仍混入多版本状态、历史文档仍未隔离”。

这意味着文档治理的下一阶段重点不再是修奖励公式，而是：

1. 把课程学习定义统一到当前真实能力
2. 把 `training_pipeline_detailed.md` 清理成单一版本
3. 把历史分析文档和目标架构文档明确标成非现状规范

---

## 13. 附录：当前仍有效问题的重点引用位置

### 课程学习

- `README.md:172`
- `README.md:176`
- `src/affordance_guided_interaction/training/README.md:184`
- `src/affordance_guided_interaction/training/README.md:185`
- `src/affordance_guided_interaction/training/curriculum_manager.py:50`
- `src/affordance_guided_interaction/training/curriculum_manager.py:57`
- `src/affordance_guided_interaction/envs/scene_factory.py:8`
- `src/affordance_guided_interaction/envs/scene_factory.py:65`

### 训练详细流程文档

- `training_pipeline_detailed.md:111`
- `training_pipeline_detailed.md:114`
- `training_pipeline_detailed.md:142`
- `training_pipeline_detailed.md:143`
- `training_pipeline_detailed.md:328`
- `training_pipeline_detailed.md:549`

### 历史状态与目标架构文档

- `project_completion_analysis.md:131`
- `project_completion_analysis.md:139`
- `project_completion_analysis.md:165`
- `project_architecture.md:222`
- `project_architecture.md:225`
- `project_architecture.md:311`
- `project_architecture.md:313`
- `project_architecture.md:339`

### 资产文档

- `assets/minimal_push_door/README.md:5`
- `assets/minimal_push_door/README.md:9`
- `assets/robot/README.md:17`
