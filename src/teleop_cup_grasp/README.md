# Teleoperated Cup Grasp

Isaac Sim 中双臂机器人 (Unitree Z1 on Dingo) 的水杯抓取演示，包含交互式调参和自动回放两个阶段。

## 工作流

```
grasp_demo.py  ──(按 q 记录坐标)──>  grasp_replay.py
   手动调参阶段                          自动回放阶段
```

### 1. `grasp_demo.py` — 手动调参

交互式场景，用于找到合适的杯子位置和抓取姿态。

- 机器人所有关节初始化为零位
- 浮空板上放着水杯，通过 UI 滑块调整 XYZ 位置
- 16 个关节滑块实时控制双臂（各 6 关节 + 夹爪）和云台（yaw/pitch）
- Visual / Collision / Both 显示模式切换
- **终端按 `q`** 记录杯子相对 `base_link` 的坐标（不退出程序，可多次记录）

```bash
python src/teleop_cup_grasp/grasp_demo.py
```

输出示例：
```
============================================================
  [RECORD] Cup relative to base_link
    base_link world: [0.0127, 0.0000, 0.0149]
    cup world:       [0.3027, 0.1111, 0.6962]
    relative (cup-base): [0.2900, 0.1111, 0.6814]
============================================================
```

### 2. `grasp_replay.py` — 自动回放

将记录的坐标写入文件顶部，启动后自动完成抓取。

- 左/右臂可独立启用，通过顶部变量控制：
  ```python
  ENABLE_LEFT_GRASP = True
  ENABLE_RIGHT_GRASP = True
  ```
- 右臂数据由左臂自动镜像（Y 取反，J1/J4/J6 取反）
- 每个启用的臂：预设姿态 → 放置杯子 → 夹爪 smoothstep 闭合
- 闭合后所有关节可通过滑块自由操作

```bash
python src/teleop_cup_grasp/grasp_replay.py
```

## 回放流程

```
关节归零 (240帧)
    │
    ▼
设置臂姿态: J6=±90°, Gripper=-90° (240帧)
    │
    ▼
放置杯子: base_link + recorded offset
    │
    ▼
夹爪闭合: -90° → -34° (smoothstep, 120帧)
    │
    ▼
自由控制: 全关节滑块可操作
```

## 左右臂对称关系

机器人双臂关于 Y=0 平面镜像安装：

| 参数 | 左臂 | 右臂 |
|------|------|------|
| 安装 Y 偏移 | +0.11708 | -0.11708 |
| 杯子 Y | +0.1111 | -0.1111 |
| J1 (yaw) | +deg | -deg |
| J4 | +deg | -deg |
| J6 (wrist roll) | +90° | -90° |
| J2, J3, J5 | 不变 | 不变 |
| Gripper | 不变 | 不变 |

## 依赖文件

```
assets/
├── robot/
│   ├── usd/uni_dingo_dual_arm.usd    # 机器人 USD (含关节驱动配置)
│   └── urdf/uni_dingo_dual_arm_absolute.urdf  # 用于读取关节范围
└── grasp_objects/
    └── cup/carry_cup.usda             # 水杯模型
```

## 物理参数说明

- **Contact offset**: 默认 2cm 改为 1mm，避免"隔空抓取"
- **Rest offset**: 0.1mm，减少接触面颤抖
- **关节驱动**: 臂 stiffness=1000 / damping=100，轮 velocity mode / damping=1500（在 USD 中预配置）
- **板子摩擦**: staticFriction=2.0, dynamicFriction=1.5

## 环境

```bash
conda activate isaaclab  # Isaac Sim 5.1
```
