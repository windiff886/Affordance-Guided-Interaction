# 强化学习训练场景搭建说明

> 本文档说明训练场景的物理世界布局、代码组装流程和运行时动态，帮助理解从 USD 资产到可交互仿真环境的完整链路。

---

## 目录

1. [物理世界布局（USD 静态场景）](#1-物理世界布局)
2. [代码层场景组装流程](#2-代码层场景组装流程)
3. [运行时场景动态](#3-运行时场景动态)

---

## 1. 物理世界布局

整个训练场景定义在 `assets/minimal_push_door/minimal_push_door.usda` 中，是一个带门的小房间。

### 1.1 场景俯视图

```
              Y = +2.45
        ┌─────────────────────────┐  RightWall
        │                         │
        │      Room (6m × 5m)     │
        │                         │
        │                         │
        │   ☆ Robot (4.6, 0)      │
        │   面朝 -X 方向(180°yaw) │
        │                         │
        │         ┌──门──┐        │   ← X = 2.93
        │ FrontWL │ Door │ FrontWR│   ← 门框开口 Y∈[-0.5, 0.5]
        ├─────────┤      ├────────┤
        │         └──────┘        │
   X=-2.95                    X=+2.95
        │      BackWall           │
        └─────────────────────────┘
              Y = -2.45
```

坐标系: Z 轴向上，X 轴朝右，Y 轴朝前。重力方向 (0, 0, -1)，g = 9.81 m/s^2。

### 1.2 房间结构

| 元素 | 位置 | 尺寸 | 说明 |
|------|------|------|------|
| Floor | (0, 0, -0.05) | 6m x 5m x 0.1m | 有 PhysicsCollisionAPI |
| BackWall | (-2.95, 0, 1.1) | 0.1 x 5 x 2.2m | |
| LeftWall | (0, -2.45, 1.1) | 6 x 0.1 x 2.2m | |
| RightWall | (0, +2.45, 1.1) | 6 x 0.1 x 2.2m | |
| FrontWallLeft | (2.95, -1.50, 1.1) | 0.1 x 2.0 x 2.2m | 门左侧墙体 |
| FrontWallRight | (2.95, +1.50, 1.1) | 0.1 x 2.0 x 2.2m | 门右侧墙体 |
| DoorFrameLeft/Right | (2.93, +/-0.54, 1) | 0.12 x 0.06 x 2.0m | 门框立柱 |
| DoorFrameHeader | (2.93, 0, 2.08) | 0.12 x 1.08 x 0.06m | 门框横梁 |

所有墙体和地面均带有 `PhysicsCollisionAPI`，机器人和门与之发生物理碰撞。

### 1.3 门（核心交互对象）

门由三个 USD prim 组成:

```
DoorHingeAnchor  (运动学固定体, 铰链锚点)
    位置: (2.93, -0.45, 0)
    physics:kinematicEnabled = true     ← 固定在世界中不动

DoorLeaf  (动力学刚体, 门板)
    位置: (2.93, -0.45, 0)             ← 与锚点重合
    ├── Panel  (门板):   0.04 x 0.82 x 1.90m, 棕色
    └── PushPlate (推板): 0.01 x 0.18 x 0.32m, 金属色, 位于门板靠外侧

DoorHingeJoint  (旋转关节, 连接两者)
    axis = "Z"                          ← 绕 Z 轴旋转（水平推门）
    lowerLimit = -105 deg
    upperLimit = +105 deg
    body0 = DoorHingeAnchor             ← 固定端
    body1 = DoorLeaf                    ← 活动端
```

门的铰链在 Y = -0.45 处（门框左侧），门板从铰链往 +Y 方向延伸 0.82m。机器人在 (4.6, 0) 面朝 -X，正对门的推板。

### 1.4 杯体（条件生成）

定义在 `assets/grasp_objects/cup/carry_cup.usda`，是一个小型无把手杯体:

| 属性 | 值 | 说明 |
|------|------|------|
| 外形 | 圆柱体, 半径 1.8cm, 高 12cm | |
| 默认质量 | 0.28 kg | 训练时被域随机化覆盖为 0.1-0.8 kg |
| linearDamping | 5.0 | 高阻尼，防止弹飞 |
| angularDamping | 5.0 | |
| maxLinearVelocity | 0.5 m/s | 限速，即使被弹也飞不远 |
| maxAngularVelocity | 2.0 rad/s | |
| 摩擦力 | static=0.8, dynamic=0.6 | 中等摩擦，接触时不会产生巨大切向力 |
| 抓取参考帧 | GraspFrames/Pinch at (0, 0, 0.06) | 杯体中心 |

杯体仅在持杯上下文（left_only / right_only / both）中生成，Stage 1 训练时不存在。

### 1.5 机器人

机器人 USD 资产位于 `assets/robot/uni_dingo_dual_arm.usd`（55MB），由 Dingo 移动底座 + 双 Unitree Z1 机械臂组成:

- **底座**: Dingo 四轮底座，训练时通过高阻尼锁死不动
- **双臂**: 左右各一条 Z1 6-DoF 机械臂，共 12 个关节
- **末端**: left_gripper_link / right_gripper_link
- **相机**: 头部挂载 Intel D455 深度相机

---

## 2. 代码层场景组装流程

### 2.1 入口链路

```
train.py main()
  |
  +-> build_env_config(cfg) -> EnvConfig dataclass
  |    physics_dt=1/120, decimation=2, max_episode_steps=500
  |    door_angle_target=1.57 rad, cup_drop_threshold=0.15m
  |    total_joints=12, effort_limits=[33.5]*12
  |
  +-> VecDoorEnv(n_envs=64, cfg=env_config)
       +-> 创建 64 个 DoorInteractionEnv 实例
           每个 DoorInteractionEnv.__init__() 内部创建:
             +-> SceneFactory(physics_dt=1/120)
             +-> ContactMonitor(force_threshold=0.1N, cup_drop_threshold=0.15m)
             +-> TaskManager(success_angle=1.2rad, end_angle=1.57rad, max_steps=500)
             +-> ActorObsBuilder(action_history_length=3)
             +-> RewardManager(reward_cfg)
```

涉及的源文件:

| 文件 | 职责 |
|------|------|
| `envs/base_env.py` | 定义 EnvConfig dataclass 和 BaseEnv 抽象基类 |
| `envs/scene_factory.py` | 唯一直接调用 Isaac Lab API 的位置 |
| `envs/door_env.py` | 单环境完整实现 (reset / step / close) |
| `envs/vec_env.py` | N 个单环境的向量化并行封装 |
| `envs/contact_monitor.py` | 接触事件监控 (自碰撞 / 杯脱落) |
| `envs/task_manager.py` | 任务进度状态机 (成功判定 / 终止判定) |

### 2.2 SceneFactory.build() — 场景装配 8 步

每次 episode reset 时由 `DoorInteractionEnv.reset()` 调用，执行以下步骤:

```
SceneFactory.build(door_type, left_occupied, right_occupied, domain_params)

  Step 1: _clear_scene()
          删除上一局的动态实体 (主要是 /World/Cup)
          机器人和门是持久的，不需要每局重建

  Step 2: _spawn_robot() -> Articulation 句柄
          加载双臂机器人 USD，配置力矩控制模式

  Step 3: _spawn_door(door_type) -> (Articulation, RigidObject) 句柄
          加载门 USD，配置铰链关节

  Step 4: _get_ee_view("left"/"right") -> 末端执行器引用
          _spawn_camera() -> D455 相机句柄 (可选)

  Step 5: _spawn_cup() -> RigidObject 句柄 (条件执行)
          仅当 left_occupied 或 right_occupied 为 True 时生成

  Step 6: 组装 SceneHandles
          将所有句柄集中到一个 dataclass 中管理

  Step 7: _resolve_indices()
          解析关节和 body 的数值索引:
            robot.find_joints(["left_joint1"..."right_joint6"]) -> arm_joint_indices
            robot.find_bodies(["base_link"])            -> base_body_idx
            robot.find_bodies(["left_gripper_link"])     -> left_ee_body_idx
            robot.find_bodies(["right_gripper_link"])    -> right_ee_body_idx
            door.find_joints(".*")                       -> door_hinge_joint_idx

  Step 8: apply_domain_params()
          将域随机化参数写入物理引擎:
            cup_mass   -> RigidObject.root_physx_view.set_masses()
            door_mass  -> Articulation.root_physx_view.set_masses()
            door_damping -> Articulation.write_joint_damping_to_sim()
            base_pos   -> Articulation.write_root_pose_to_sim()
```

### 2.3 机器人 Articulation 配置

```python
ArticulationCfg(
    prim_path = "/World/Robot",
    spawn = UsdFileCfg(
        usd_path = "assets/robot/uni_dingo_dual_arm.usd",
        rigid_props = RigidBodyPropertiesCfg(
            disable_gravity = False,
            retain_accelerations = True,      # 保留加速度数据，用于稳定性观测
            max_depenetration_velocity = 1.0,
        ),
        articulation_props = ArticulationRootPropertiesCfg(
            enabled_self_collisions = True,   # 开启自碰撞检测
            solver_position_iteration_count = 8,
            solver_velocity_iteration_count = 4,
        ),
        activate_contact_sensors = True,      # 开启接触传感器
    ),
    init_state = InitialStateCfg(
        pos = (4.6, 0.0, 0.12),               # 离地 12cm (底座高度)
        rot = (0.0, 0.0, 1.0, 0.0),           # 180 deg yaw，面朝 -X (朝门)
        joint_pos = {"left_joint.*": 0.0, "right_joint.*": 0.0},
    ),
    actuators = {
        "arms": ImplicitActuatorCfg(
            joint_names_expr = ["left_joint.*", "right_joint.*"],
            effort_limit = 33.5,     # Z1 臂最大力矩 33.5 N*m
            velocity_limit = 2.175,
            stiffness = 0.0,         # 纯力矩控制 (无位置环)
            damping = 0.0,           # 无阻尼 (Agent 需要自己学会控制)
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr = [".*wheel"],
            effort_limit = 20.0,
            stiffness = 0.0,
            damping = 10.0,          # 高阻尼锁死底座，训练时不动
        ),
    },
)
```

关键设计选择:

| 配置 | 值 | 原因 |
|------|------|------|
| arms stiffness=0, damping=0 | 纯力矩控制 | 策略直接输出 12 维力矩，无位置/速度环 |
| retain_accelerations=True | 保留加速度 | 稳定性 proxy 需要末端加速度数据 |
| enabled_self_collisions=True | 自碰撞检测 | ContactMonitor 据此计算安全惩罚 |
| wheels damping=10.0 | 锁死底座 | 训练时只训练双臂，底座不移动 |
| activate_contact_sensors=True | 接触传感器 | 读取 body_contact_net_forces_w |

### 2.4 门 Articulation 配置

```python
ArticulationCfg(
    prim_path = "/World/Door",
    spawn = UsdFileCfg(usd_path = "minimal_push_door.usda"),
    init_state = InitialStateCfg(joint_pos = {".*": 0.0}),   # 门初始关闭
    actuators = {
        "hinge": ImplicitActuatorCfg(
            effort_limit = 0.0,      # 门没有主动力矩，自由转动
            velocity_limit = 100.0,
            stiffness = 0.0,         # 无回弹弹簧
            damping = 2.0,           # 默认阻尼 (训练时被域随机化覆盖为 0.5-5.0)
        ),
    },
)
```

### 2.5 相机配置

```python
# 挂载在机器人头部的 Intel D455 深度相机
Camera(
    prim_path = "/World/Robot/base_link/head_d455_link/Camera",
    frequency = 30,            # 30 fps
    resolution = (640, 480),   # VGA
)
# 内参: focalLength=11.2mm, horizontalAperture=20.955mm,
#       verticalAperture=15.716mm, clippingRange=[0.01, 100.0]m
```

相机输出 RGB-D，经 AffordancePipeline (LangSAM 分割 -> 深度反投影 -> Point-MAE 编码) 产生 768 维 door_embedding。

### 2.6 SceneHandles 数据结构

所有句柄集中管理在一个 dataclass 中，避免在环境代码中散落:

```python
@dataclass
class SceneHandles:
    robot_view: Articulation          # 机器人
    left_ee_view: str                 # "left_gripper_link"
    right_ee_view: str                # "right_gripper_link"
    door_view: Articulation           # 门 (含铰链关节)
    door_panel_view: RigidObject      # 门板 (读位姿用)
    cup_view: RigidObject | None      # 杯体 (None = 本局无杯)
    camera_view: Camera | None        # D455 深度相机

    door_type: str                    # "push"
    left_occupied: bool               # 左手是否持杯
    right_occupied: bool              # 右手是否持杯

    arm_joint_names: list[str]        # 12 个关节名
    arm_joint_indices: Tensor         # 12 个关节在 Articulation 中的索引
    base_body_idx: int                # base_link 在 body 数组中的索引
    left_ee_body_idx: int             # left_gripper_link 索引
    right_ee_body_idx: int            # right_gripper_link 索引
    door_hinge_joint_idx: Tensor      # 门铰链关节索引

    sim_context: SimulationContext    # Isaac Lab 仿真上下文
```

### 2.7 Isaac Lab 不可用时的 Fallback

`scene_factory.py` 是 envs 层中唯一直接调用 Isaac Lab API 的位置。当 Isaac Lab 不可用时（纯 CPU 开发/测试），所有 `_spawn_*` 方法返回 `_PlaceholderView()` 占位对象，`_read_physics_state()` 返回全零状态字典。这使得代码可以在无 Isaac Lab 环境下运行单元测试。

---

## 3. 运行时场景动态

### 3.1 每个 Episode 的场景变化

每次 episode reset 时，场景根据两个来源发生变化:

**课程管理器决定上下文配置**:

| 阶段 | 上下文分布 | 杯体生成 |
|------|-----------|---------|
| Stage 1 | none: 1.0 | 不生成杯体 |
| Stage 2 | left_only: 0.5, right_only: 0.5 | 单侧生成杯体 |
| Stage 3 | none/left/right/both: 各 0.25 | 按需生成 |

**域随机化器决定物理参数**:

| 参数 | 分布 | 落地 API |
|------|------|---------|
| cup_mass | U[0.1, 0.8] kg | `set_masses()` |
| door_mass | U[5.0, 20.0] kg | `set_masses()` |
| door_damping | U[0.5, 5.0] | `write_joint_damping_to_sim()` |
| base_pos | 标称 +/- 0.03m (XY) | `write_root_pose_to_sim()` |

### 3.2 单步物理交互

`DoorInteractionEnv._sim_step()` 的执行流程:

```
策略输出 action(12,)
    |
    +-> clip 到 +/-33.5 N*m                      (力矩安全截断)
    +-> 构建完整关节力矩向量 (双臂12维 + wheel=0)
    +-> robot.write_joint_effort_to_sim()          (写入力矩到仿真)
    +-> sim.step(render=False) x decimation=2      (两次 PhysX 步进)
    +-> robot.update() / door.update() / cup.update()   (刷新数据缓存)
```

控制频率 = 1/120s x 2 = 60 Hz，即策略每秒做 60 次决策。

### 3.3 物理状态读取

`DoorInteractionEnv._read_physics_state()` 每步从 Isaac Lab 读取的完整状态:

```
机器人本体 (通过 arm_joint_indices 索引):
  +-- joint_pos (12)            关节角度
  +-- joint_vel (12)            关节角速度
  +-- applied_torque (12)       实际施加力矩

末端执行器 (左右各，通过 body 索引):
  +-- position (3)              base_link 相对位置
  +-- orientation (4)           base_link 相对四元数
  +-- linear_velocity (3)
  +-- angular_velocity (3)
  +-- linear_acceleration (3)   用于稳定性 proxy
  +-- angular_acceleration (3)  用于稳定性 proxy

门 (通过 door_view.data):
  +-- door_joint_pos            铰链角度 (rad)
  +-- door_joint_vel            铰链角速度
  +-- door_pose (7)             pos(3) + quat(4), base_link 相对

杯体 (如有，通过 cup_view.data):
  +-- cup_position (3)
  +-- cup_pose (7)
  +-- cup_linear_vel (3)
  +-- cup_angular_vel (3)
```

**坐标系转换**: 所有世界系量通过 `_vector_world_to_base()` 和 `_orientation_world_to_base()` 转换到 base_link 坐标系。这样当域随机化微调 base_pos 时，观测仍然自洽。

### 3.4 接触监控

`ContactMonitor.update()` 每步执行:

```
1. 读取 robot.data.body_contact_net_forces_w    所有 body 的净接触力
2. 力阈值过滤 (< 0.1N 的微碰撞忽略)
3. 分类:
   - link 名在 _SELF_COLLISION_PAIRS 中 -> self_collision = True
   - 其余 -> 累加 total_external_force
4. 杯体脱落检测:
   - 计算 cup_pos 与持杯臂末端 ee_pos 的距离
   - 距离 > 0.15m -> cup_dropped = True
```

### 3.5 任务状态判定

`TaskManager.update()` 每步执行，判定优先级: 杯脱落 > 角度终止 > 超时

| 条件 | 结果 | TerminationReason |
|------|------|-------------------|
| cup_dropped = True | done=True | CUP_DROPPED |
| door_angle >= 1.57 rad (90 deg) | done=True | ANGLE_LIMIT_REACHED |
| step_count >= 500 | done=True | TIMEOUT |
| door_angle >= 1.2 rad (首次) | success=True | (不终止，仅标记成功) |

注意: 成功角度阈值 (1.2 rad) 不等于 episode 终止角度 (1.57 rad)，策略在达到成功后仍需继续控制直到终止。

### 3.6 VecDoorEnv 并行化与自动重置

```
VecDoorEnv(n_envs=64)
    |
    +-> 64 个独立的 DoorInteractionEnv 实例

step(actions: (64, 12)):
    for i in range(64):
        obs, reward, done, info = envs[i].step(actions[i])
        if done:
            obs = envs[i].reset(           # auto-reset
                domain_params = ...,       # 当前域随机化参数
                door_type = ...,           # 当前课程门类型
                left_occupied = ...,       # 当前课程持杯配置
                right_occupied = ...,
            )
            info["terminal_observation"] = True
```

自动重置时:
- 使用当前课程阶段的上下文分布
- 使用当前域随机化器采样的新物理参数
- 返回新 episode 的初始观测（而非终止时的观测）
- 在 info 中标记 `terminal_observation=True`

### 3.7 课程阶段跃迁时的环境更新

当 CurriculumManager 判定跃迁时，通过 `VecDoorEnv.set_curriculum()` 更新所有环境的配置:

```python
envs.set_curriculum(
    door_types = ["push"] * n_envs,
    left_occupied_list = [...],          # 新阶段的持杯配置
    right_occupied_list = [...],
    domain_params_list = [...],          # 新的域随机化参数
)
```

新配置在下一次 reset（包括 auto-reset）时生效。跃迁不重置模型权重或优化器状态。
