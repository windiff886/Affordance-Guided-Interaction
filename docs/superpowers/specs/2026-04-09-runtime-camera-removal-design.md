# Runtime Camera Removal Design

**Goal:** 在保留机器人资产层 D455 / URDF 描述的前提下，移除项目默认运行路径中的相机与视觉依赖，使训练、评估、rollout、导出、配置与文档全部以 `door_geometry` 为唯一门相关输入来源。

## 1. 背景

旧版本项目通过 RGB-D 相机、分割、点云和 Point-MAE 编码得到门相关视觉 embedding，再注入到 policy。当前代码库已经完成了核心策略输入的切换：Actor / Critic 的主观测不再使用 768 维视觉 embedding，而是直接使用从仿真 ground truth 计算得到的 `door_geometry(6D)`。

但仓库中仍存在多类视觉残留：

- 训练主路径虽然已关闭相机，但环境配置和部分运行时辅助代码仍保留相机入口。
- 评估与 rollout 脚本仍显式开启 `enable_cameras=True`，并实例化旧的 `PerceptionRuntime`。
- 导出脚本仍按旧的 `visual(768)` 接口组织 flat input。
- 多份文档与配置仍描述视觉感知链路为当前方案。

这会造成三个问题：

1. 新成员会误判当前系统仍依赖相机。
2. 部分脚本与当前观测接口不一致，存在运行失败风险。
3. 后续开发在“默认路径”和“历史路径”之间边界不清，容易再次引入不必要的视觉耦合。

## 2. 目标与边界

### 2.1 本次修订目标

本次修订采用“运行路径去相机化”方案：

- 清除训练、评估、rollout、导出、默认配置、默认文档中的相机与视觉依赖。
- 统一对外描述：当前默认方案不需要任何相机信息。
- 保证所有默认运行入口都不依赖 `RGB`、`Depth`、`visual embedding`、`PerceptionRuntime` 或相机传感器。
- 保留资产层的 D455 / URDF / RViz / ROS 描述，不在本次修改中删除。

### 2.2 非目标

以下内容不在本次修订范围内：

- 删除 `assets/robot/` 中的 D455 相关模型、URDF、RViz、ROS topic 配置。
- 重写机器人资产结构。
- 彻底删除所有历史视觉源码。
- 新增新的感知替代模块。

换句话说，本次要做的是：**把视觉链路从默认运行系统中退场，而不是彻底抹掉所有历史痕迹。**

## 3. 当前审计结论

### 3.1 已经完成的正确方向

当前主训练路径已经基本符合无相机目标：

- `scripts/train.py` 显式设置 `enable_cameras=False`。
- `build_env_cfg()` 在关闭相机时会将 `env_cfg.scene.tiled_camera = None`。
- `RolloutCollector` 在训练主路径中传入 `perception_runtime=None`。
- `DoorPushEnv._get_observations()` 当前构建的是 `proprio + ee + context + stability + door_geometry`。
- `policy/actor.py` 和 `policy/critic.py` 当前编码的是 `door_geometry(6D)` 而不是 `visual(768D)`。

### 3.2 明确存在的残留

#### A. 仍会主动开启相机的运行入口

- `scripts/evaluate.py`
- `scripts/rollout_demo.py`

这两个脚本仍会：

- 调用 `launch_simulation_app(..., enable_cameras=True)`
- 创建 `PerceptionRuntime(refresh_interval=4, embedding_dim=768)`
- 请求 `envs.get_visual_observations()`

它们与当前 door geometry 路径已经不一致。

#### B. 环境与运行时中的相机兼容残留

- `src/affordance_guided_interaction/envs/door_push_env_cfg.py`
  - 仍声明 `tiled_camera: TiledCameraCfg`
- `src/affordance_guided_interaction/envs/door_push_env.py`
  - 仍保留 `self._camera = self.scene.sensors.get("tiled_camera")`
- `src/affordance_guided_interaction/utils/sim_runtime.py`
  - 仍维护相机启动分支与相机 carb setting 辅助逻辑
- `src/affordance_guided_interaction/training/rollout_collector.py`
  - 仍保留视觉 batch 准备钩子

#### C. 已与当前 policy 接口不一致的旧工具

- `scripts/export_policy.py`
  - 仍按 `[proprio | ee | context | stability | visual]` 组织导出输入
- `src/affordance_guided_interaction/training/perception_runtime.py`
  - 仍将 `door_embedding(768)` 写入旧位置
- `src/affordance_guided_interaction/door_perception/`
  - 仍是完整视觉感知链路

#### D. 文档和配置层残留

- `README.md`
- `docs/training_pipeline_detailed.md`
- `docs/tensorboard_guide.md`
- `configs/README.md`
- `configs/training/default.yaml`

这些文件中仍存在“当前默认系统使用视觉 embedding”的描述或视觉调试项。

## 4. 修订后的目标状态

修订完成后，默认系统应满足以下原则：

### 4.1 单一门相关输入源

默认策略只接受：

- `proprio`
- `ee`
- `context`
- `stability`
- `door_geometry`

其中 `door_geometry` 直接从仿真中的门叶位姿计算，作为唯一门相关输入。

### 4.2 默认运行入口不触碰相机

以下默认入口不得再请求相机或视觉观测：

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/rollout_demo.py`
- `scripts/export_policy.py`

### 4.3 配置层不再暴露视觉开关

默认训练配置和默认文档中，不再出现：

- `visualize_detections`
- `strict_mode` 的视觉依赖解释
- `visual embedding`
- `camera_fetch_s`
- `enable_cameras=True` 作为推荐路径

### 4.4 历史视觉模块退为“非默认历史代码”

`door_perception/`、`PerceptionRuntime` 等模块可以暂时保留，但必须满足：

- 不被默认运行入口引用
- 不被默认文档描述为当前方案
- 在文档中明确标记为历史实验路径或后续单独归档对象

## 5. 设计原则

### 5.1 先切运行链路，再清文档语义

代码层必须先实现“默认入口完全不需要相机”，再同步文档。否则文档与实际行为会再次分裂。

### 5.2 优先收敛接口，不急于删历史模块

本次修订的核心不是“删得越多越好”，而是“默认接口只表达当前真实方案”。因此优先修正入口、配置、观测接口、导出接口，再决定是否进一步清理历史视觉模块。

### 5.3 避免引入双轨制说明

不要继续在 README 或训练主文档里同时并列介绍“当前默认 door_geometry 路径”和“旧视觉路径”作为同等级选项。默认文档只描述当前路径，历史路径另行标注。

## 6. 分模块修订计划

### 6.1 Phase 1: 入口链路收敛

#### 目标

让所有默认运行入口都基于 `door_geometry` 工作，不再请求任何视觉观测。

#### 涉及文件

- `scripts/evaluate.py`
- `scripts/rollout_demo.py`
- `scripts/export_policy.py`

#### 具体改动

1. `evaluate.py`
   - 去掉 `PerceptionRuntime` 引用与初始化。
   - 去掉 `envs.get_visual_observations()` 调用。
   - 将 `launch_simulation_app(..., enable_cameras=True)` 改为 `False`。
   - 直接消费环境返回的 actor obs，并通过 `batch_flatten_actor_obs()` 进入 actor。

2. `rollout_demo.py`
   - 去掉 `PerceptionRuntime` 引用与刷新逻辑。
   - 去掉 `envs.get_visual_observations()` 调用。
   - 将 `enable_cameras=True` 改为 `False`。
   - 重新定义“帧捕获”语义：
     - 如果 rollout 仍需要视频，应明确使用仿真视口渲染或其他非 policy 相机路径。
     - 如果当前没有稳定的无相机帧捕获路径，则先降级为“只做 rollout，不做视频抓帧”，避免继续伪装为视觉依赖。

3. `export_policy.py`
   - 将 flat input 顺序由 `[proprio | ee | context | stability | visual]` 改为 `[proprio | ee | context | stability | door_geometry]`。
   - 维度由 `768` 改为 `6`。
   - 与 `policy/actor.py` 中的 `build_actor_branches_from_tensor()` 保持一致。

#### 交付结果

- 评估、rollout、导出全部与当前 actor 真实输入接口对齐。
- 默认运行入口不再依赖相机启动。

### 6.2 Phase 2: 环境与运行时收口

#### 目标

让环境和 runtime helper 不再对外暴露“默认仍有相机”的信号。

#### 涉及文件

- `src/affordance_guided_interaction/envs/door_push_env_cfg.py`
- `src/affordance_guided_interaction/envs/door_push_env.py`
- `src/affordance_guided_interaction/utils/sim_runtime.py`
- `src/affordance_guided_interaction/training/rollout_collector.py`

#### 具体改动

1. `door_push_env_cfg.py`
   - 将 `tiled_camera` 从默认场景配置中移除。
   - 若担心未来实验恢复，可改为注释清晰的可选扩展配置，而不是默认字段。

2. `door_push_env.py`
   - 删除 `_camera` 句柄缓存。
   - 确保环境本身完全不持有相机状态。

3. `sim_runtime.py`
   - 将 `enable_cameras` 相关逻辑降级为兼容层，或限制为非默认用途。
   - 默认路径的启动说明只强调 headless / rendering，不再强调 camera settings。

4. `rollout_collector.py`
   - 删除或隔离 `_prepare_visual_batch()` 中的视觉抓取逻辑。
   - 如果保留兼容代码，必须通过显式实验开关触发，而非默认 collector 路径。

#### 交付结果

- 环境定义本身不再暗示“默认带相机传感器”。
- rollout collector 不再保留误导性的默认视觉分支。

### 6.3 Phase 3: 历史视觉模块降级

#### 目标

保留历史模块源码，但使其彻底退出默认运行链路和默认叙事。

#### 涉及文件

- `src/affordance_guided_interaction/training/perception_runtime.py`
- `src/affordance_guided_interaction/training/__init__.py`
- `src/affordance_guided_interaction/door_perception/`
- `src/affordance_guided_interaction/envs/camera_batch_utils.py`

#### 具体改动

1. 给 `PerceptionRuntime` 增加明确定位：
   - 历史模块
   - 默认训练不使用
   - 未来若恢复感知研究，应在独立分支或实验配置中接入

2. 审查 `training/__init__.py` 是否还要默认导出 `PerceptionRuntime`。

3. 为 `door_perception/README.md` 增加醒目标记：
   - 当前默认项目路径不依赖本目录
   - 本目录仅保留为历史实验与后续研究参考

4. 对 `camera_batch_utils.py` 做归类处理：
   - 若无默认调用方，标记为历史工具
   - 若完全无用，可移除

#### 交付结果

- 历史视觉模块不再与当前默认训练架构混淆。

### 6.4 Phase 4: 文档与配置全面对齐

#### 目标

让所有面向开发者的说明都明确表达：当前默认系统不需要相机信息。

#### 涉及文件

- `README.md`
- `docs/training_pipeline_detailed.md`
- `docs/tensorboard_guide.md`
- `configs/README.md`
- `configs/training/default.yaml`
- `src/affordance_guided_interaction/envs/README.md`
- `src/affordance_guided_interaction/policy/README.md`
- `src/affordance_guided_interaction/observations/README.md`

#### 具体改动

1. 顶层 README
   - 首页简介改为 `door_geometry` 驱动，不再描述 affordance 视觉编码为当前方案。

2. 训练详细文档
   - 删除或改写 `PerceptionRuntime` 作为当前训练环节的描述。
   - 将 Actor 五分支中的“视觉编码”改成 `door_geometry`。
   - 清理“视觉引导”“视觉 embedding 注入”等历史表述。

3. TensorBoard 文档
   - 删除默认视觉 timing 的说明，或改为“历史路径指标，默认未启用”。

4. 配置 README 与 training YAML
   - 删除 `visualize_detections`
   - 删除 `strict_mode` 中与视觉依赖绑定的说明
   - 保留真正仍需要的运行时检查配置，但要重新命名和重新解释

5. env / policy / observations 文档
   - 统一使用 `door_geometry` 叙事
   - 强调默认环境不含相机观测接口

#### 交付结果

- 新成员只靠文档即可正确理解当前系统。
- 默认配置不再暴露历史视觉选项。

## 7. 执行顺序与里程碑

### Milestone 1: 默认入口可运行且无相机

完成标志：

- `train.py`、`evaluate.py`、`rollout_demo.py`、`export_policy.py` 全部不请求相机
- 所有默认脚本都与 `door_geometry` 接口一致

这是最优先里程碑，因为它直接影响代码真实可运行性。

### Milestone 2: 环境与 runtime 边界收紧

完成标志：

- 默认环境配置不再定义 `tiled_camera`
- `DoorPushEnv` 不再持有 `_camera`
- collector 默认路径不再包含视觉 batch 逻辑

### Milestone 3: 文档与配置完成对齐

完成标志：

- 顶层 README、训练文档、配置文档全部改写完成
- 默认配置中不再出现视觉调试项

### Milestone 4: 历史模块归档完成

完成标志：

- 历史视觉模块完成“非默认、仅参考”定位
- 仓库中不再出现“当前默认方案仍使用视觉”的误导性说法

## 8. 风险与应对

### 风险 1: rollout 可视化功能与“policy 不用相机”被混为一谈

风险说明：
rollout 视频抓帧可能仍需要某种渲染输出，但这不应被表述为“策略使用相机观测”。

应对：

- 明确区分“policy input camera”和“viewer/render output”
- 如果当前视频抓取实现强绑定 `get_visual_observation()`，就先切断，再单独设计可视化渲染方案

### 风险 2: 导出接口与训练接口继续分叉

风险说明：
`export_policy.py` 仍是旧 768 维接口，若不及时修正，会导致部署端继续构建错误输入。

应对：

- 将导出接口改造放入 Phase 1
- 使用 `build_actor_branches_from_tensor()` 作为唯一切分真相来源

### 风险 3: 历史模块仍被误用

风险说明：
即使默认路径已清理，开发者仍可能从 `training/__init__.py` 或旧 README 中误用 `PerceptionRuntime`。

应对：

- 在历史模块头部和 README 中增加明确弃用说明
- 从默认入口和默认文档中去掉所有引用

### 风险 4: 文档更新滞后于代码

风险说明：
如果只改代码不改文档，团队会继续依据旧描述开发。

应对：

- 将文档更新作为独立 Milestone，而不是“有空再补”
- 验收必须包含文档检查

## 9. 验收标准

修订完成后，应满足以下验收条件：

### 9.1 代码行为验收

- `scripts/train.py` 启动时不要求相机
- `scripts/evaluate.py` 启动时不要求相机
- `scripts/rollout_demo.py` 启动时不要求相机
- `scripts/export_policy.py` 导出接口使用 `door_geometry(6D)` 而不是 `visual(768D)`

### 9.2 接口一致性验收

- Actor flat obs 切分与导出脚本完全一致
- 环境观测维度、adapter 解析、policy branch 名称完全一致
- 默认代码路径中不存在 `get_visual_observations()` 之类的调用链

### 9.3 文档一致性验收

- 顶层 README 不再描述视觉 embedding 为当前默认方案
- 训练/配置文档不再把 `PerceptionRuntime` 作为当前训练必经环节
- 默认配置不再暴露视觉 debug 选项

### 9.4 范围边界验收

- `assets/robot/` 中的 D455 资产仍保留
- 但这些资产不会影响默认训练、评估、rollout 和导出路径

## 10. 建议的提交拆分

为了降低回归风险，建议按以下顺序拆分提交：

1. `refactor: remove camera dependency from evaluate and rollout entrypoints`
2. `refactor: align export interface with door geometry observations`
3. `refactor: remove default camera sensor remnants from env runtime`
4. `docs: align project docs with door geometry default pipeline`
5. `docs: mark legacy perception modules as non-default`

这样的提交粒度便于回滚，也便于逐步验证每一层接口。

## 11. 最终建议

本次修订不建议一开始就大规模删除 `door_perception/` 或机器人 D455 资产。更稳妥的路线是：

- 先把默认运行路径和默认文档全部收敛到 `door_geometry`
- 再把历史视觉模块降级为非默认参考
- 最后视后续研究计划决定是否彻底归档或移除

这样既能快速实现“当前系统不再需要任何相机信息”的目标，也能避免一次性删除过多历史代码带来的追溯成本。
