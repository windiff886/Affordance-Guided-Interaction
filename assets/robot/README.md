# Uni-Dingo 双臂机器人描述文件

本目录包含 Uni-Dingo 双臂移动机器人的完整描述文件，整理自 `Z1-Dingo-Robot` 工作空间。

## 机器人组成

| 组件 | 说明 | 来源包 |
|------|------|--------|
| **Dingo 底盘** | Clearpath Dingo-O 全向移动底盘（4 轮麦克纳姆） | `dingo_description` |
| **Z1 机械臂 × 2** | Unitree Z1 六自由度机械臂（左臂 + 右臂），含夹爪 | `z1_description` / `unidingo_description` |
| **Pan-Tilt 云台** | IQR 两自由度云台（偏航 + 俯仰） | `pan_tilt_description` |
| **D455 相机** | Intel RealSense D455 深度相机，挂载于云台顶部 | `unidingo_description` / `uni_dingo_api` |

## 目录结构

```
assets/robot/
├── README.md                          # 本文件
├── urdf/                              # URDF / Xacro 描述文件
│   ├── uni_dingo_dual_arm.urdf.xacro  # ★ 主入口：整机描述（底盘+双臂+云台+相机）
│   ├── uni_dingo.urdf                 # SolidWorks 导出的静态 URDF（仅底盘+安装座）
│   ├── z1_arm.urdf.xacro             # Z1 机械臂 Xacro 宏（支持前缀，可复用为左/右臂）
│   ├── d455_only.urdf.xacro          # D455 相机独立模型
│   ├── z1/                            # Z1 机械臂原始定义
│   │   ├── const.xacro               #   物理常量（惯性、质量、关节限位等）
│   │   ├── robot.xacro               #   单臂完整描述（含 world link）
│   │   ├── gazebo.xacro              #   Gazebo 仿真插件
│   │   ├── transmission.xacro        #   传动定义
│   │   └── z1.urdf                   #   展开后的纯 URDF
│   ├── dingo/                         # Dingo 底盘定义
│   │   ├── dingo.urdf.xacro          #   顶层入口（根据环境变量选择 O/D 型号）
│   │   ├── dingo-o.urdf.xacro        #   全向轮型号（Omni）
│   │   ├── dingo-d.urdf.xacro        #   差速型号（Diff）
│   │   ├── dingo.gazebo              #   通用 Gazebo 配置
│   │   ├── dingo-o.gazebo            #   全向轮 Gazebo 配置
│   │   ├── pacs.urdf.xacro           #   PACS 扩展平台
│   │   ├── accessories.urdf.xacro    #   可选配件（激光雷达、相机等）
│   │   ├── empty.urdf                #   空白 URDF 占位符
│   │   ├── accessories/              #   配件详细描述
│   │   │   ├── intel_realsense.urdf.xacro
│   │   │   ├── hokuyo_ust10.urdf.xacro
│   │   │   ├── sick_lms1xx_mount.urdf.xacro
│   │   │   ├── vlp16_mount.urdf.xacro
│   │   │   ├── hdl32_mount.urdf.xacro
│   │   │   └── flir_blackfly_mount.urdf.xacro
│   │   └── configs/                   #   URDF 配置模板
│   ├── pan_tilt/                      # Pan-Tilt 云台定义
│   │   ├── pan_tilt.urdf.xacro       #   云台 Xacro 宏
│   │   ├── pan_tilt_only.urdf.xacro  #   云台独立模型
│   │   ├── pan_tilt_st.urdf.xacro    #   云台 + Kinect DK 组合
│   │   └── kinect_dk.xacro           #   Kinect DK 相机描述
│   └── sensors/                       # 传感器仿真插件
│       └── sensors.gazebo.xacro      #   D455 相机 Gazebo 插件
├── meshes/                            # 3D 模型文件
│   ├── unidingo/                      # Uni-Dingo 机体结构（SolidWorks 导出）
│   │   ├── base_link.STL             #   底座安装板 (~25 MB)
│   │   ├── leftarm_link.STL          #   左臂安装座 (~5 MB)
│   │   └── rightarm_link.STL         #   右臂安装座 (~5 MB)
│   ├── z1/                            # Z1 机械臂
│   │   ├── visual/                   #   可视化模型 (.dae, 含材质)
│   │   │   ├── z1_Link00~06.dae
│   │   │   ├── z1_GripperStator.dae
│   │   │   └── z1_GripperMover.dae
│   │   └── collision/                #   碰撞模型 (.STL, 简化)
│   │       ├── z1_Link00~06.STL
│   │       ├── z1_GripperStator.STL
│   │       └── z1_GripperMover.STL
│   ├── dingo/                         # Dingo 底盘
│   │   ├── omni_chassis.dae          #   全向底盘可视化模型
│   │   ├── diff_chassis.dae          #   差速底盘可视化模型
│   │   ├── *_collision.stl           #   碰撞模型
│   │   └── wheel.stl 等              #   轮子、支架等配件
│   └── pan_tilt/                      # 云台
│       ├── base.stl                  #   云台底座
│       ├── yaw.stl                   #   偏航部件
│       └── pitch.stl                 #   俯仰部件
├── config/                            # 控制器与仿真配置
│   ├── left_controllers.yaml         # 左臂 ros_control 控制器
│   ├── right_controllers.yaml        # 右臂 ros_control 控制器
│   ├── z1_gazebo_pid.yaml            # Z1 关节 PID 参数
│   ├── robot_control.yaml            # Z1 单臂控制器配置
│   ├── pan_tilt_controllers.yaml     # 云台控制器配置
│   └── robot_config.yaml             # 机器人总体配置
└── rviz/                              # RViz 可视化配置
    ├── dual_arm.rviz                 # 双臂整机可视化
    └── d455_only.rviz                # D455 相机独立可视化
```

## TF 树结构（简化）

```
base_link (Dingo 底盘)
├── chassis_link
│   ├── front_left_wheel_link
│   ├── front_right_wheel_link
│   ├── rear_left_wheel_link
│   └── rear_right_wheel_link
├── leftarm_link (左臂安装座 + 支撑柱)
│   └── left_link00 → left_link01 → ... → left_link06
│       └── left_gripperStator → left_gripperMover
├── rightarm_link (右臂安装座 + 支撑柱)
│   └── right_link00 → right_link01 → ... → right_link06
│       └── right_gripperStator → right_gripperMover
└── pan_tilt_mount_link (云台安装座 + 支撑柱)
    └── pan_tilt_base_link
        └── pan_tilt_yaw_link
            └── pan_tilt_pitch_link
                └── pan_tilt_surface
                    └── camera_support_link
                        └── head_d455_link (D455 相机)
```

## 关节列表

| 关节名称 | 类型 | 自由度 | 说明 |
|---------|------|--------|------|
| `left_joint1` ~ `left_joint6` | revolute | 6 | 左臂关节 |
| `left_jointGripper` | revolute | 1 | 左臂夹爪 |
| `right_joint1` ~ `right_joint6` | revolute | 6 | 右臂关节 |
| `right_jointGripper` | revolute | 1 | 右臂夹爪 |
| `pan_tilt_yaw_joint` | revolute | 1 | 云台偏航 (±60°) |
| `pan_tilt_pitch_joint` | revolute | 1 | 云台俯仰 (±60°) |
| `front_left_wheel` 等 × 4 | continuous | 4 | 底盘轮子 |
| **总计** | | **20** | |

## 文件来源映射

| 目标路径 | 原始来源 |
|---------|---------|
| `urdf/uni_dingo_dual_arm.urdf.xacro` | `unidingo_description/urdf/` |
| `urdf/z1_arm.urdf.xacro` | `unidingo_description/urdf/` |
| `urdf/z1/*` | `z1_ros/unitree_ros/robots/z1_description/xacro/` |
| `urdf/dingo/*` | `dingo/dingo_description/urdf/` |
| `urdf/pan_tilt/*` | `pan_tilt_ros/pan_tilt_description/urdf/` |
| `urdf/sensors/*` | `uni_dingo_api/urdf/` |
| `meshes/unidingo/*` | `unidingo_description/meshes/` |
| `meshes/z1/*` | `z1_ros/unitree_ros/robots/z1_description/meshes/` |
| `meshes/dingo/*` | `dingo/dingo_description/meshes/` |
| `meshes/pan_tilt/*` | `pan_tilt_ros/pan_tilt_description/urdf/mesh/` |
| `config/*` | `unidingo_description/config/` + `z1_description/config/` + `uni_dingo_api/config/` |
| `rviz/*` | `unidingo_description/config/` + `unidingo_description/rviz/` |

> **注意**: URDF/Xacro 文件中的 `package://` 路径引用（如 `package://z1_description/meshes/...`）
> 仍然指向原始 ROS 包名称。如需独立使用这些文件，需要相应调整 mesh 路径。
