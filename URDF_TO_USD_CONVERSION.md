# URDF → USD 严格转换脚本流程详解

> 文档对象：`assets/robot/scripts/convert_urdf_to_usd.py`
>
> 本文档逐阶段剖析该脚本的完整工作流，涵盖从 URDF 解析到 USD 导出的每一个技术细节。

---

## 目录

1. [概述](#1-概述)
2. [脚本入口与参数解析](#2-脚本入口与参数解析)
3. [Phase 1 — URDF 预解析：建立期望基线](#3-phase-1--urdf-预解析建立期望基线)
4. [Phase 2 — 启动 Isaac Sim 并创建空 Stage](#4-phase-2--启动-isaac-sim-并创建空-stage)
5. [Phase 3 — URDF 导入配置与执行](#5-phase-3--urdf-导入配置与执行)
6. [Phase 4 — Articulation 自碰撞配置](#6-phase-4--articulation-自碰撞配置)
7. [Phase 5 — 关节驱动配置](#7-phase-5--关节驱动配置)
8. [Phase 6 — 空碰撞子树修复](#8-phase-6--空碰撞子树修复)
9. [Phase 7 — 悬空内部引用清理](#9-phase-7--悬空内部引用清理)
10. [Phase 8 — 资产打包与共享几何重映射](#10-phase-8--资产打包与共享几何重映射)
11. [Phase 9 — 碰撞引用内联](#11-phase-9--碰撞引用内联)
12. [Phase 10 — 去实例化（Deinstance）](#12-phase-10--去实例化deinstance)
13. [Phase 11 — 碰撞体 Purpose 归一化](#13-phase-11--碰撞体-purpose-归一化)
14. [Phase 12 — 转换校验](#14-phase-12--转换校验)
15. [Phase 13 — 导出 USD 文件](#15-phase-13--导出-usd-文件)
16. [辅助工具函数索引](#16-辅助工具函数索引)
17. [整体流程图](#17-整体流程图)

---

## 1. 概述

本脚本负责将 Uni-Dingo 双臂移动机器人的 URDF 描述文件转换为 Isaac Sim / Isaac Lab 可直接使用的 USD 资产文件。转换过程**并非简单的格式转换**，而是一条包含多个后处理阶段的严格流水线：

| 阶段 | 目的 |
|------|------|
| URDF 预解析 | 提取每个 link 的 visual/collision 期望作为校验基线 |
| URDF 导入 | 通过 Isaac Sim URDF Importer 将 URDF 转为 USD Stage |
| Articulation 配置 | 启用 PhysX Articulation 自碰撞检测 |
| 关节驱动配置 | 为轮子/机械臂关节分别设置 velocity/force 驱动 |
| 碰撞修复 | 修复 importer 留下的空碰撞子树 |
| 引用清理 | 清除悬空的内部引用 |
| 资产打包 | 将分散的共享几何根整合进单一导出根 |
| 碰撞内联 | 将碰撞体引用内联到 link 子树中，消除共享 /colliders |
| 去实例化 | 关闭 instanceable 标记，解决 PhysX 碰撞识别问题 |
| Purpose 归一化 | 将碰撞体 purpose 从 `guide` 改为 `default` |
| 严格校验 | 逐 link 比对 URDF 期望与 USD 实际状态 |
| 导出 | 校验通过后才写入 USD 文件 |

**核心原则**：校验未通过时脚本会直接失败退出（返回 1），绝不导出一个语义缺失的 USD 文件。

---

## 2. 脚本入口与参数解析

```
入口：main() → build_arg_parser() → run_conversion(args)
```

### 2.1 环境准备（模块级代码）

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 项目根目录
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.utils.runtime_env import (
    configure_omniverse_client_environment,
)
configure_omniverse_client_environment(os.environ)
```

- 计算项目根路径（从 `assets/robot/scripts/` 向上 3 级）
- 将 `src/` 加入 `sys.path` 以便导入项目内部模块
- 调用 `configure_omniverse_client_environment()` 配置 Omniverse Client Library 所需的环境变量（如 `OMNI_USER`, `OMNI_PASS` 等）

### 2.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--headless` | `True` | 无窗口模式运行 Isaac Sim |
| `--urdf` | `assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf` | 输入 URDF 路径 |
| `--output` | `assets/robot/usd/uni_dingo_dual_arm.usd` | 输出 USD 路径 |

### 2.3 关节名称常量

脚本在模块级定义了两组关节名称列表，用于后续驱动配置：

- **`ARM_JOINTS`**（16 个）：包括左右臂各 6 个关节 + 夹爪关节 + 云台 yaw/pitch 关节
- **`WHEEL_JOINTS`**（4 个）：四个麦克纳姆轮关节

---

## 3. Phase 1 — URDF 预解析：建立期望基线

**函数**：`collect_urdf_link_expectations(urdf_path)`

```
URDF XML
  └─ 遍历所有 <link> 元素
       └─ 对每个 link 记录：
            ├─ name: link 名称
            ├─ has_visual: 是否包含 <visual> 子元素
            └─ has_collision: 是否包含 <collision> 子元素
```

此阶段独立于 Isaac Sim，使用标准库 `xml.etree.ElementTree` 直接解析 URDF XML 文件，产出数据结构：

```python
@dataclass(frozen=True)
class URDFLinkExpectation:
    name: str
    has_visual: bool
    has_collision: bool
```

**关键作用**：这份期望字典是后续所有校验的**黄金基线**（ground truth）。在脚本结束时，USD stage 中每个 link 的 visual/collision 状态必须与此基线完全吻合，否则转换失败。

---

## 4. Phase 2 — 启动 Isaac Sim 并创建空 Stage

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)    # Z 轴向上
UsdGeom.SetStageMetersPerUnit(stage, 1.0)           # 1 USD 单位 = 1 米
```

**关键配置**：
- `SimulationApp` 是 Isaac Sim Python API 的入口，必须在导入 `pxr` 等 USD 模块之前创建
- Stage 设置 Z-up 坐标系和米制单位，与 Isaac Lab 训练环境保持一致
- `simulation_app.update()` 在关键操作后调用，确保 USD Stage 状态完全刷新

---

## 5. Phase 3 — URDF 导入配置与执行

### 5.1 创建导入配置

```python
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
```

### 5.2 配置参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `merge_fixed_joints` | `False` | 保留所有 fixed joint，不合并为单一 link |
| `convex_decomp` | `True` | 对碰撞 mesh 启用凸分解（V-HACD），生成更精确的碰撞体 |
| `import_inertia_tensor` | `True` | 使用 URDF 中定义的惯性张量，而非自动计算 |
| `fix_base` | `False` | 不固定机器人基座（移动机器人需要自由移动） |
| `collision_from_visuals` | `False` | 不从 visual mesh 自动生成碰撞体，使用 URDF 原生碰撞定义 |
| `self_collision` | `False` | **不**在资产级别写入碰撞组（避免破坏 Isaac Lab 多环境 PhysX 复制） |
| `distance_scale` | `1.0` | 距离缩放因子为 1（不缩放） |
| `create_physics_scene` | `False` | 不创建 PhysicsScene prim（由训练环境负责创建） |

> **注意**：`self_collision = False` 仅阻止 importer 创建 PhysicsCollisionGroup prim。运行时自碰撞检测在 Phase 4 中通过 `PhysxArticulationAPI.enabledSelfCollisions` 单独启用。

### 5.3 执行导入

```python
status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=str(urdf_path),
    import_config=import_config,
)
```

导入完成后，USD Stage 中产生以下典型层级：

```
/<robot_name>        ← 机器人 Articulation 根
  ├─ <link_A>/
  │   ├─ visuals/    ← visual 几何体（可能是引用）
  │   └─ collisions/ ← collision 几何体（通常引用到 /colliders/<link_A>）
  ├─ <link_B>/
  │   ├─ visuals/
  │   └─ collisions/
  ...
/visuals/            ← 共享 visual 几何体根
/colliders/          ← 共享 collision 几何体根
/meshes/             ← 共享 mesh 数据根
```

---

## 6. Phase 4 — Articulation 自碰撞配置

```python
if not robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
    PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)

attr = robot_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
attr.Set(True)
```

**作用**：在 Articulation 根 prim 上应用 `PhysxArticulationAPI` 并设置 `enabledSelfCollisions = True`。这确保机器人在仿真时，不同 link 之间可以检测并产生碰撞响应（例如机械臂与底盘之间的碰撞）。

**与 Phase 3 的关系**：Phase 3 中 `import_config.self_collision = False` 禁止了 **资产级** 碰撞组（会破坏 Isaac Lab 多环境克隆），而此处启用的是 **Articulation 级** 自碰撞，两者互不冲突。

---

## 7. Phase 5 — 关节驱动配置

**函数**：`configure_joint_drives(stage, robot_prim_path, Usd, UsdPhysics)`

遍历 stage 中所有属于该机器人的 `RevoluteJoint` 或 `Joint` prim，按关节类型应用不同驱动：

| 关节类型 | 驱动模式 | 刚度 (stiffness) | 阻尼 (damping) | 说明 |
|---------|---------|------------------|----------------|------|
| 轮子 (`WHEEL_JOINTS`) | `velocity` | 0.0 | 1500.0 | 纯速度控制，无位置保持 |
| 机械臂 (`ARM_JOINTS`) | `force` | 1000.0 | 100.0 | 力矩-位置控制，高刚度保证位置精度 |

**实现细节**：
```python
drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
drive.CreateTypeAttr("velocity")   # 或 "force"
drive.CreateDampingAttr(1500.0)
drive.CreateStiffnessAttr(0.0)
```

---

## 8. Phase 6 — 空碰撞子树修复

**函数**：`repair_empty_collisions(stage, robot_prim_path, urdf_expectations, ...)`

这是脚本中最复杂的阶段，用于解决 URDF Importer 的一个常见问题：**某些 link 的 `collisions/` 子树虽然存在但内容为空**，导致这些 link 在仿真中没有碰撞体。

### 8.1 修复判定条件

```python
def should_repair_link(*, expected_has_collision, usd_has_collision) -> bool:
    return expected_has_collision and not usd_has_collision
```

仅当 URDF 期望有碰撞体、但 USD 中实际没有时才触发修复。

### 8.2 修复流程

```
对每个需要修复的 link:
  1. 查找 visuals 容器和 collisions 容器
     └─ resolve_geometry_container() 按优先级查找：
          ├─ 本地路径: /<robot>/<link>/visuals
          └─ 共享路径: /visuals/<link>
  
  2. 跳过已有 authored 引用的 collisions 容器
     └─ authored_geometry_container() 检查是否有 Instance 或 AuthoredReferences
  
  3. 为 collisions 子树中已有的子 prim 添加碰撞 API
     └─ add_collision_api() 应用:
          ├─ UsdPhysics.CollisionAPI
          ├─ PhysxSchema.PhysxCollisionAPI
          ├─ contactOffset = 0.005
          └─ restOffset = 0.001
  
  4. 从 visual 几何体复制碰撞体
     └─ iter_supported_visual_prims() 遍历 visual 子 prim:
          ├─ Mesh → copy_mesh_geometry() + PhysxConvexDecompositionCollisionAPI
          ├─ Cube/Sphere/Cylinder/Capsule/Cone → copy_gprim_geometry()
          └─ 其他类型 → 跳过
     └─ 对复制后的 prim:
          ├─ add_collision_api() 添加碰撞属性
          └─ MakeInvisible() 设置为不可见（碰撞体默认不渲染）
```

### 8.3 几何体复制策略

脚本通过 `classify_visual_copy_strategy()` 分类：

| 类型 | 策略 | 处理方式 |
|------|------|---------|
| `Mesh` | `"mesh"` | 复制顶点/面/索引 + 应用凸分解碰撞 API |
| `Cube/Sphere/Cylinder/Capsule/Cone` | `"gprim"` | 复制全部 authored 属性 |
| `Xform/Scope/""` | 穿透 | 作为 wrapper 节点继续向下搜索 |
| 其他 | `None` | 跳过 |

### 8.4 关键辅助函数

- **`resolve_geometry_container()`**：双路径查找，优先本地 `<link>/collisions`，回退共享 `/collisions/<link>`
- **`shallow_children()`**：遍历直接子节点 + 穿透一层 wrapper（Xform/Scope）
- **`copy_xform_ops()`**：复制变换操作（translate/orient/scale/rotateXYZ）
- **`copy_authored_attributes()`**：复制所有 authored 属性（跳过 `visibility`）

---

## 9. Phase 7 — 悬空内部引用清理

**函数**：`clear_invalid_internal_references(stage)`

遍历 stage 中所有 prim，检查其内部引用（无 `assetPath` 的 `Sdf.Reference`）是否指向有效 prim：
- 如果引用目标不存在或无效 → 从引用列表中移除
- 如果所有引用都无效 → 完全清除引用

此阶段在碰撞修复后执行，清理修复过程中可能产生的无效引用。

---

## 10. Phase 8 — 资产打包与共享几何重映射

**函数**：`package_imported_robot_asset(stage, robot_prim_path, ...)`

### 10.1 问题背景

URDF Importer 导入后，USD Stage 的顶层结构是分散的：

```
/<robot_name>    ← 机器人主体
/visuals         ← 共享 visual 几何体
/colliders       ← 共享 collision 几何体
/meshes          ← 共享 mesh 数据
```

这种结构在 Isaac Lab 中作为可引用资产使用时会有问题——需要一个统一的 `defaultPrim` 作为资产入口。

### 10.2 打包流程

```
1. 创建包装根 /RobotAsset (Xform)

2. 将所有源根复制到包装根下：
   /<robot_name> → /RobotAsset/<robot_name>
   /visuals      → /RobotAsset/visuals
   /colliders    → /RobotAsset/colliders
   /meshes       → /RobotAsset/meshes

3. 重写内部引用路径（rewrite_internal_references_in_subtree）
   ├─ 遍历 /RobotAsset 子树中所有 prim
   ├─ 对每个内部引用的 primPath 应用 path_map 重映射
   └─ 例如: /colliders/base_link → /RobotAsset/colliders/base_link

4. 重写关系目标路径（rewrite_relationship_targets_in_subtree）
   ├─ 遍历 /RobotAsset 子树中所有 prim 的 relationships
   └─ 应用相同的 path_map 重映射

5. 设置 defaultPrim = /RobotAsset
```

### 10.3 路径重映射算法

`remap_internal_reference_path()` 按路径长度降序匹配，确保最长前缀优先匹配：

```python
# path_map = {"/colliders": "/RobotAsset/colliders", ...}
# 输入: "/colliders/base_link/mesh_0"
# 匹配: "/colliders" → "/RobotAsset/colliders"
# 输出: "/RobotAsset/colliders/base_link/mesh_0"
```

---

## 11. Phase 9 — 碰撞引用内联

**函数**：`inline_wrapper_collision_references(stage, ...)`

### 11.1 问题背景

打包后，每个 link 的 `collisions` prim 仍然通过内部引用指向 `/RobotAsset/colliders/<link>`。这个共享 `/colliders` 子树中可能包含 `PhysicsCollisionGroup` prim，在 Isaac Lab 的多环境 PhysX 复制（multi-env cloning）中会导致问题。

### 11.2 内联流程

```
1. 遍历 /RobotAsset/<robot> 子树，找到所有名为 "collisions" 的 prim
2. 对每个 collisions prim：
   ├─ 检查其内部引用是否指向 /RobotAsset/colliders/ 下
   ├─ 清除引用: prim.GetReferences().ClearReferences()
   └─ 用 Sdf.CopySpec() 将引用目标的内容直接复制到 collisions prim
3. 最后删除整个 /RobotAsset/colliders 子树: stage.RemovePrim()
```

**效果**：每个 link 的碰撞数据成为自包含的，不再依赖共享层级。

---

## 12. Phase 10 — 去实例化（Deinstance）

**函数**：`deinstance_all_prims(stage, Usd)`

```python
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if prim.IsInstance():
        prim.SetInstanceable(False)
```

**问题根源**：URDF Importer 会自动将 `visuals/collisions` 节点标记为 `instanceable=True`，使子节点变成 `InstanceProxy`。PhysX 引擎无法将 InstanceProxy 下的碰撞体关联到其所属的 rigid body，导致碰撞失效。

**解决方案**：遍历全部 prim，关闭 `instanceable` 标记。

---

## 13. Phase 11 — 碰撞体 Purpose 归一化

**函数**：`normalize_collision_purpose(stage, Usd, UsdGeom)`

```python
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    imageable = UsdGeom.Imageable(prim)
    purpose_attr = imageable.GetPurposeAttr()
    if purpose_attr and purpose_attr.Get() == "guide":
        purpose_attr.Set("default")
```

**问题根源**：URDF Importer 将碰撞体容器的 `purpose` 设为 `guide`（USD 中 `guide` purpose 的 prim 在 viewport 默认不渲染）。

**解决方案**：将 `purpose` 改为 `default` 后，可以通过 `MakeVisible()`/`MakeInvisible()` 正常控制碰撞体的显隐。碰撞体初始状态为 `invisible`。

---

## 14. Phase 12 — 转换校验

### 14.1 收集 USD 状态

**函数**：`collect_usd_link_states(stage, robot_prim_path, ...)`

对照 URDF 期望字典中的每个 link 名称，在 USD Stage 的打包路径下查找对应 prim，记录其 visual/collision 状态：

```python
@dataclass(frozen=True)
class USDLinkState:
    name: str
    has_visual: bool      # has_visual_geometry() 检测
    has_collision: bool    # has_collision_geometry() 检测
```

### 14.2 校验逻辑

**函数**：`validate_robot_conversion(...)`

校验项目清单：

| 校验项 | 失败条件 |
|--------|---------|
| Robot prim 存在性 | `robot_prim_found == False` |
| defaultPrim 正确性 | `default_prim_matches == False` |
| Articulation 存在性 | `articulation_found == False` |
| Link 完整性 | URDF 中的 link 在 USD 中缺失 |
| Visual 语义保持 | URDF 有 visual 但 USD 没有 |
| Collision 语义保持 | URDF 有 collision 但 USD 没有 |

### 14.3 Articulation 检测

```python
def articulation_exists(root, Usd, UsdPhysics, PhysxSchema) -> bool:
    return root.HasAPI(UsdPhysics.ArticulationRootAPI) or \
           root.HasAPI(PhysxSchema.PhysxArticulationAPI)
```

### 14.4 校验报告

```python
@dataclass(frozen=True)
class ConversionValidationReport:
    success: bool
    issues: list[str]
    total_links: int
    expected_visual_links: int
    expected_collision_links: int
    usd_links_found: int
    repaired_collision_links: int = 0
```

**如果校验失败**：脚本输出所有 issues 并返回 exit code 1，**不执行导出**。

---

## 15. Phase 13 — 导出 USD 文件

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
stage.GetRootLayer().Export(str(output_path))
```

仅在校验通过后执行。输出文件路径默认为 `assets/robot/usd/uni_dingo_dual_arm.usd`。

导出后打印文件大小以供确认。

---

## 16. 辅助工具函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `log()` | L84 | 统一日志输出（`flush=True`） |
| `build_arg_parser()` | L88 | 构建命令行参数解析器 |
| `collect_urdf_link_expectations()` | L111 | 解析 URDF XML 建立期望基线 |
| `validate_robot_conversion()` | L124 | 执行完整转换校验 |
| `format_validation_report()` | L172 | 格式化校验报告为可读文本 |
| `add_collision_api()` | L190 | 为 prim 添加 CollisionAPI + PhysxCollisionAPI |
| `classify_visual_copy_strategy()` | L199 | 分类 visual prim 的复制策略 |
| `copy_xform_ops()` | L207 | 复制 USD 变换操作 |
| `copy_mesh_geometry()` | L223 | 复制 Mesh 几何数据（vertices/faces/indices） |
| `copy_authored_attributes()` | L243 | 复制所有 authored 属性 |
| `copy_gprim_geometry()` | L256 | 复制基础几何体（Cube/Sphere 等） |
| `wrapper_type()` | L262 | 判断是否为穿透类型（Xform/Scope/""） |
| `immediate_valid_children()` | L266 | 获取有效直接子节点 |
| `shallow_children()` | L272 | 获取子节点 + 穿透一层 wrapper |
| `resolve_geometry_container()` | L280 | 双路径查找几何体容器 |
| `collision_target_path()` | L301 | 计算碰撞体目标路径 |
| `get_applied_references()` | L305 | 获取 prim 的已应用引用列表 |
| `remap_internal_reference_path()` | L312 | 按 path_map 重映射引用路径 |
| `clone_reference()` | L322 | 克隆 Sdf.Reference 并替换 primPath |
| `clear_invalid_internal_references()` | L331 | 清理悬空内部引用 |
| `rewrite_internal_references_in_subtree()` | L364 | 重写子树中所有内部引用路径 |
| `rewrite_relationship_targets_in_subtree()` | L405 | 重写子树中所有 relationship targets |
| `package_imported_robot_asset()` | L437 | 打包资产到统一导出根 |
| `inline_wrapper_collision_references()` | L476 | 内联碰撞引用并移除共享 /colliders |
| `authored_geometry_container()` | L533 | 检查是否有 authored 引用/实例 |
| `editable_authoring_prim()` | L541 | 获取可编辑的 prim（穿透 InstanceProxy） |
| `iter_supported_visual_prims()` | L549 | 迭代支持复制的 visual prim |
| `should_repair_link()` | L563 | 判定 link 是否需要碰撞修复 |
| `repair_empty_collisions()` | L567 | 执行碰撞子树修复 |
| `deinstance_all_prims()` | L651 | 关闭所有 instanceable 标记 |
| `normalize_collision_purpose()` | L665 | 将 guide purpose 改为 default |
| `configure_joint_drives()` | L684 | 配置关节驱动参数 |
| `has_visual_geometry()` | L714 | 检测 link 是否有 visual 几何 |
| `has_collision_geometry()` | L727 | 检测 link 是否有 collision 几何 |
| `collect_usd_link_states()` | L742 | 收集 USD 中所有 link 的状态 |
| `articulation_exists()` | L763 | 检测 Articulation API 是否存在 |
| `run_conversion()` | L769 | 主转换流程 |
| `main()` | L928 | 脚本入口 |

---

## 17. 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        main() 入口                               │
│  build_arg_parser() → parse_args() → run_conversion(args)       │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 1: URDF 预解析                │
│  collect_urdf_link_expectations()   │
│  → urdf_expectations (黄金基线)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 2: 启动 Isaac Sim            │
│  SimulationApp() → new_stage()      │
│  SetStageUpAxis(Z) + Meters(1.0)    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 3: URDF 导入                  │
│  URDFCreateImportConfig             │
│  URDFParseAndImportFile             │
│  → robot_prim_path                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 4: Articulation 自碰撞       │
│  PhysxArticulationAPI.Apply()       │
│  enabledSelfCollisions = True       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 5: 关节驱动配置               │
│  configure_joint_drives()           │
│  轮子=velocity, 机械臂=force         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 6: 碰撞子树修复              │
│  repair_empty_collisions()          │
│  visual → collision 几何复制         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 7: 悬空引用清理 (第一轮)      │
│  clear_invalid_internal_references()│
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 8: 资产打包                   │
│  package_imported_robot_asset()     │
│  → /RobotAsset (defaultPrim)        │
│  重写内部引用 + relationship targets │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 9: 碰撞引用内联              │
│  inline_wrapper_collision_references│
│  移除共享 /colliders 子树            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 7b: 悬空引用清理 (第二轮)     │
│  clear_invalid_internal_references()│
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 10: 去实例化                  │
│  deinstance_all_prims()             │
│  修复 PhysX InstanceProxy 碰撞问题  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 11: Purpose 归一化            │
│  normalize_collision_purpose()      │
│  guide → default                    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Phase 12: 严格校验                               │
│  collect_usd_link_states()                       │
│  validate_robot_conversion()                     │
│  逐 link 比对 visual/collision 语义               │
│                                                   │
│  ┌─────────┐        ┌──────────┐                  │
│  │ PASS ✅  │        │ FAIL ❌   │                  │
│  └────┬────┘        └────┬─────┘                  │
│       │                  │                         │
│       ▼                  ▼                         │
│   Phase 13          打印 issues                    │
│   导出 USD          exit(1)                        │
└─────────────────────────────────────────────────┘
```

---

## 使用示例

```bash
# 使用默认参数（headless 模式）
python assets/robot/scripts/convert_urdf_to_usd.py

# 指定输入输出路径
python assets/robot/scripts/convert_urdf_to_usd.py \
    --urdf assets/robot/urdf/custom_robot.urdf \
    --output assets/robot/usd/custom_robot.usd
```

---

## 设计思想总结

1. **防御性转换**：不信任 URDF Importer 的输出，通过独立的 URDF 预解析建立基线，在最终阶段严格比对
2. **多层修复**：针对 Importer 的已知缺陷（空碰撞、instanceable、guide purpose）逐一修复
3. **Isaac Lab 兼容**：打包资产、内联碰撞引用、关闭碰撞组，确保在多环境克隆场景下正常工作
4. **Fail-fast**：任何校验不通过都立即终止，绝不输出有缺陷的资产文件
