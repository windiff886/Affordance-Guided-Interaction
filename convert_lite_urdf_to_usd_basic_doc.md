# convert_lite_urdf_to_usd_basic.py 机制详解

## 概述

`assets/robot/scripts/convert_lite_urdf_to_usd_basic.py` 是一个 **极简版** URDF-to-USD 转换脚本，将 `uni_dingo_lite.urdf`（固定底座双臂机器人模型）直接导入 Isaac Sim 并导出为 USD 格式。

该脚本的核心设计哲学是 **最朴素导入**：仅调用 Isaac Sim 的 `URDFCreateImportConfig` + `URDFParseAndImportFile` 两条命令完成转换，不做任何额外的资产后处理。

### 与完整版脚本的区别

项目中存在两个版本的转换脚本：

| 特性 | `convert_lite_urdf_to_usd_basic.py`（极简版） | `convert_lite_urdf_to_usd.py`（完整版） |
|------|-----|------|
| convex decomposition | 不做 | 开启 |
| 空碰撞修复 | 不做 | 有 |
| RobotAsset 打包 | 不做 | 有 |
| colliders 内联 | 不做 | 有 |
| deinstance | 不做 | 有 |
| 关节驱动重写 | 不做 | 有 |
| 自碰撞检测 | 不做 | 有 |
| 导入后校验 | 不做 | 有（URDF vs USD link 完整性比对） |
| 代码行数 | ~190 行 | ~810 行 |

极简版适用于快速迭代和调试场景，完整版适用于生产环境部署。

---

## 文件路径

```
assets/robot/scripts/convert_lite_urdf_to_usd_basic.py
```

输入文件：
```
assets/robot/urdf/uni_dingo_lite.urdf
```

输出文件：
```
assets/robot/usd/uni_dingo_lite.usd
```

---

## 执行流程

整个脚本的执行流程可分为以下 6 个阶段：

```
┌─────────────────────────────────────────┐
│  1. 环境初始化                            │
│     - 配置 Omniverse 运行时环境            │
│     - 设置 ROS_PACKAGE_PATH               │
├─────────────────────────────────────────┤
│  2. 命令行参数解析                         │
│     - --headless, --urdf, --output        │
├─────────────────────────────────────────┤
│  3. ROS 包 Overlay 构建                   │
│     - 创建 z1_description 最小包           │
│     - 建立 mesh 目录符号链接               │
├─────────────────────────────────────────┤
│  4. Isaac Sim 启动                        │
│     - headless 模式启动 SimulationApp      │
├─────────────────────────────────────────┤
│  5. URDF 导入与导出                        │
│     - 创建空 Stage                        │
│     - 配置导入参数                         │
│     - 执行 URDFParseAndImportFile         │
│     - 导出 USD 文件                       │
├─────────────────────────────────────────┤
│  6. 关闭清理                              │
│     - 关闭 SimulationApp                  │
└─────────────────────────────────────────┘
```

---

## 各阶段详细机制

### 1. 环境初始化

```python
from affordance_guided_interaction.utils.runtime_env import (
    configure_omniverse_client_environment,
)
configure_omniverse_client_environment(os.environ)
```

**作用**：在 Isaac Sim 启动之前，为 Omniverse 客户端配置一组隔离的运行时环境变量。

**具体行为**（由 `runtime_env.py` 提供）：

- 在 `$TMPDIR/isaacsim-runtime`（或 `/tmp/isaacsim-runtime`）下创建运行时目录结构
- 设置以下环境变量（仅在未设置时）：

| 环境变量 | 用途 |
|---------|------|
| `OMNICLIENT_HUB_MODE` | 设为 `disabled`，禁用 Hub 模式 |
| `MPLCONFIGDIR` | Matplotlib 配置目录 |
| `OMNI_CACHE_DIR` | Omniverse 缓存目录 |
| `OMNI_LOG_DIR` | Omniverse 日志目录 |
| `OMNI_DATA_DIR` | Omniverse 数据目录 |
| `OMNI_USER_DIR` | Omniverse 用户文件目录 |
| `XDG_CACHE_HOME` | XDG 缓存标准目录 |
| `XDG_DATA_HOME` | XDG 数据标准目录 |
| `XDG_CONFIG_HOME` | XDG 配置标准目录 |

**为什么需要**：Isaac Sim 在启动时会读取这些环境变量来决定缓存、日志和配置的存放位置。通过将其隔离到临时目录，可以避免并发运行时的冲突，也避免污染用户主目录。

### 2. 命令行参数解析

```python
parser = argparse.ArgumentParser(description="Lite URDF -> USD 极简直导")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--headless` | flag | `True`（默认开启） | 无窗口模式运行 |
| `--urdf` | Path | `assets/robot/urdf/uni_dingo_lite.urdf` | 输入 URDF 文件路径 |
| `--output` | Path | `assets/robot/usd/uni_dingo_lite.usd` | 输出 USD 文件路径 |

**默认路径解析**：路径基于项目根目录计算（`Path(__file__).resolve().parents[3]`），即从 `assets/robot/scripts/` 向上回溯 3 级到达项目根目录。

### 3. ROS 包 Overlay 构建

这是脚本中**唯一的预处理逻辑**，也是最关键的一步。

#### 问题背景

URDF 文件中的 Z1 机械臂 mesh 使用了 ROS `package://` URI 引用：

```xml
<mesh filename="package://z1_description/meshes/visual/z1_Link00.dae" scale="1 1 1"/>
```

Isaac Sim 的 URDF 导入器需要通过 `ROS_PACKAGE_PATH` 环境变量来解析这些 URI。它会在 `ROS_PACKAGE_PATH` 指定的目录下寻找名为 `z1_description` 的 ROS 包，然后在包内查找 `meshes/` 子目录。

#### 解决方案：`configure_lite_mesh_package()`

脚本通过构建一个**最小化的 ROS 包 overlay** 来满足这一需求：

```
assets/robot/.ros_pkg_overlay/          ← ROS_PACKAGE_PATH 指向这里
  └── z1_description/                   ← 包名匹配 URDF 中的 package://z1_description
      ├── package.xml                   ← 最小化的 ROS 包描述文件
      └── meshes -> assets/robot/meshes/z1   ← 符号链接到真实 mesh 目录
```

**执行步骤**：

1. **检查 mesh 目录存在**：确认 `assets/robot/meshes/z1` 存在，否则抛出 `FileNotFoundError`
2. **创建 overlay 目录**：`assets/robot/.ros_pkg_overlay/z1_description/`
3. **创建符号链接**：`meshes` -> `assets/robot/meshes/z1`
   - 如果已有符号链接指向不同位置，先删除再重建
   - 如果已存在非符号链接的文件/目录，抛出 `RuntimeError`（避免误删真实数据）
4. **写入 `package.xml`**：一个最小化的 ROS 包描述文件，只包含包名、版本和基本元信息
5. **更新 `ROS_PACKAGE_PATH`**：将 overlay 根目录插入到环境变量的最前面（优先级最高），同时去重

**`package.xml` 的内容**：

```xml
<?xml version="1.0"?>
<package format="3">
  <name>z1_description</name>
  <version>0.0.0</version>
  <description>Temporary overlay package for Isaac Sim URDF import.</description>
  <maintainer email="noreply@example.com">Codex</maintainer>
  <license>Apache-2.0</license>
</package>
```

**路径解析链路**：

```
URDF: package://z1_description/meshes/visual/z1_Link00.dae
  ↓ Isaac Sim URDF Importer 查找 ROS_PACKAGE_PATH
  ↓ 找到 assets/robot/.ros_pkg_overlay/z1_description/
  ↓ 通过符号链接 meshes -> assets/robot/meshes/z1
  ↓ 最终定位到 assets/robot/meshes/z1/visual/z1_Link00.dae
```

### 4. Isaac Sim 启动

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp(
    launch_config={"headless": args.headless, "width": 1280, "height": 720}
)
```

- 以 headless 模式启动 Isaac Sim 应用实例
- 窗口分辨率设为 1280x720（headless 模式下不会实际显示窗口，但 Stage 仍需此配置）

### 5. URDF 导入与导出

这是转换的核心阶段，分为以下子步骤：

#### 5.1 创建空 Stage

```python
omni.usd.get_context().new_stage()
simulation_app.update()
stage = omni.usd.get_context().get_stage()
```

创建一个全新的空 USD Stage，并刷新一次确保初始化完成。

#### 5.2 设置 Stage 坐标系

```python
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
```

- **Up Axis = Z**：URDF 使用 Z 轴朝上，USD 保持一致
- **Meters Per Unit = 1.0**：单位为米，与 URDF 标准一致

#### 5.3 创建 URDF 导入配置

```python
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
```

调用 Isaac Sim 内置命令创建导入配置对象，然后逐一设置参数：

```python
import_config.merge_fixed_joints = False      # 不合并固定关节（保留完整运动链结构）
import_config.fix_base = False                 # 不固定基座（允许自由漂浮）
import_config.make_default_prim = True         # 将导入的机器人设为 Stage 的 defaultPrim
import_config.import_inertia_tensor = True     # 导入惯性张量（物理仿真必需）
import_config.create_physics_scene = False     # 不创建 PhysicsScene（由下游环境配置）
import_config.collision_from_visuals = False   # 不从视觉网格生成碰撞体
import_config.self_collision = False           # 不启用自碰撞（极简版不处理）
import_config.distance_scale = 1.0             # 距离缩放为 1（无缩放）
```

**各参数的设计考量**：

| 参数 | 值 | 理由 |
|------|-----|------|
| `merge_fixed_joints` | `False` | 保留 `base_link_joint`、`base_leftarm_joint` 等固定关节，使 USD 中的 link 结构与 URDF 完全对应，方便下游按名称定位 |
| `fix_base` | `False` | 训练场景中机器人可能被放置在不同位置，不锁定基座更灵活 |
| `make_default_prim` | `True` | 导出后的 USD 文件可以直接作为资产引用，defaultPrim 是 USD 资产的标准要求 |
| `import_inertia_tensor` | `True` | RL 训练需要准确的物理参数 |
| `create_physics_scene` | `False` | 物理场景由训练环境统一配置，避免重复创建 |
| `collision_from_visuals` | `False` | URDF 已显式定义了碰撞几何体，不需要从视觉网格自动推导 |
| `self_collision` | `False` | 极简版不做此项配置，由完整版或下游环境处理 |
| `distance_scale` | `1.0` | URDF 和 USD 均使用米制单位，无需缩放 |

#### 5.4 执行导入

```python
status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=str(urdf_path),
    import_config=import_config,
)
simulation_app.update()
```

调用 Isaac Sim 的 URDF 解析器，将 URDF 文件解析并导入到当前 Stage 中。导入完成后调用 `update()` 确保 Stage 完全更新。

#### 5.5 导出 USD

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
stage.GetRootLayer().Export(str(output_path))
```

将 Stage 的根 Layer 直接导出为 USD 文件。

**注意**：极简版使用的是最朴素的 `Export` 方式，不涉及任何资产重组或引用重写。

### 6. 关闭清理

```python
finally:
    simulation_app.close()
```

使用 `try/finally` 确保即使转换过程中出现异常，Isaac Sim 实例也会被正确关闭，释放 GPU 和内存资源。

---

## 输入 URDF 结构

`uni_dingo_lite.urdf` 描述的是一个**固定底座双臂机器人**（Uni-Dingo Lite），专门为 RL 训练裁剪：

```
base_link                        ← 根节点（微小占位方块）
├── chassis_link                 ← 底盘（0.57×0.42×0.24m box 代理）
├── leftarm_link                 ← 左臂安装支架（cylinder 代理）
│   └── left_link00              ← 左 Z1 臂基座
│       └── left_link01 (joint1) ← 左臂关节1 (revolute)
│           └── left_link02 (joint2)
│               └── left_link03 (joint3)
│                   └── left_link04 (joint4)
│                       └── left_link05 (joint5)
│                           └── left_link06 (joint6)
│                               └── left_gripperStator (fixed)
│                                   └── left_gripperMover (jointGripper, revolute)
└── rightarm_link                ← 右臂安装支架（与左臂镜像）
    └── right_link00
        └── ...（结构与左臂对称）
```

**裁剪内容**（相对完整版 `uni_dingo_dual_arm.urdf`）：

- 移除了：车轮、IMU、云台（pan-tilt）、相机、装饰性 mesh
- 保留了：底盘、双臂安装支架、完整 Z1 双臂 + 夹爪
- 替换了：25MB 的 `omni_chassis` mesh 用 box 代理；4.8MB 的安装支架 mesh 用 cylinder 代理
- 保留了：Z1 臂 link 的原始碰撞几何体（保证接触仿真的保真度）

**关节统计**：
- 固定关节（fixed）：6 个（base_link_joint, base_leftarm_joint, base_rightarm_joint, left/right_base_joint, left/right_gripperStator_joint）
- 旋转关节（revolute）：14 个（左右各 7 个：joint1-6 + jointGripper）

---

## 依赖关系

### Python 包依赖

| 包 | 用途 |
|----|------|
| `isaacsim` | Isaac Sim 核心（SimulationApp, omni.*） |
| `omni.kit.commands` | Isaac Sim 命令系统（URDFCreateImportConfig, URDFParseAndImportFile） |
| `omni.usd` | USD Stage 管理 |
| `pxr.UsdGeom` | Pixar USD 几何库（坐标设置） |
| `affordance_guided_interaction.utils.runtime_env` | 项目内部工具：Omniverse 环境配置 |

### 文件系统依赖

```
项目根目录/
├── assets/robot/
│   ├── urdf/uni_dingo_lite.urdf          ← 输入 URDF
│   ├── meshes/z1/
│   │   ├── visual/*.dae                  ← Z1 臂视觉 mesh
│   │   └── collision/*.STL               ← Z1 臂碰撞 mesh
│   ├── .ros_pkg_overlay/z1_description/  ← 自动生成的 ROS 包 overlay
│   │   ├── package.xml
│   │   └── meshes -> ../../../meshes/z1
│   └── usd/uni_dingo_lite.usd            ← 输出 USD
└── src/affordance_guided_interaction/utils/runtime_env.py
```

---

## 使用方法

```bash
# 默认参数运行（使用默认输入输出路径，headless 模式）
python assets/robot/scripts/convert_lite_urdf_to_usd_basic.py

# 指定输入输出
python assets/robot/scripts/convert_lite_urdf_to_usd_basic.py \
    --urdf path/to/custom.urdf \
    --output path/to/output.usd

# 禁用 headless（需要显示器支持）
python assets/robot/scripts/convert_lite_urdf_to_usd_basic.py --headless=False
```

**前提条件**：
1. Isaac Sim 已正确安装并可用
2. `EXPAND_ISAACSIM_PATH` 或 Isaac Sim Python 解释器可用
3. 项目 `src/` 目录在 Python 路径中可访问

---

## 返回值

| 返回值 | 含义 |
|--------|------|
| `0` | 转换成功 |
| `1` | 转换失败（URDF 文件不存在、导入配置创建失败、导入失败） |

---

## 设计意图

极简版脚本的设计意图是提供一个**确定性的、可复现的基线转换**：

1. **最小化变量**：不做任何后处理，如果转换结果有问题，可以确定是 Isaac Sim URDF importer 本身的行为
2. **快速迭代**：在开发 URDF 时，可以用极简版快速验证 URDF 结构是否被正确解析
3. **对比基线**：与完整版的输出对比，可以定位后处理步骤引入的变化
4. **CI 友好**：代码简洁，headless 默认开启，适合自动化流水线
