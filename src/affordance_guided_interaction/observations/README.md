# observations — 当前观测接口

默认路径中，观测张量直接由 [DoorPushEnv._get_observations()](../envs/door_push_env.py) 构建；本目录不再持有独立的 actor/critic builder 实现。

本文档只描述当前实际接口。

## 1. 观测来源

观测由以下信号构成：

- 双臂关节位置与速度
- 上一步送入仿真的关节位置目标
- 左右末端在 `base_link` 坐标系下的位姿、速度、加速度
- 当前 occupancy 上下文
- 左右倾斜 proxy
- 门板 6 维几何信号
- critic 专用 privileged 信息

当前 actor 观测不再包含 joint torque。

## 2. Actor 观测布局

Actor flat tensor 维度为 `84`，拼接顺序与 [direct_rl_env_adapter.py](../envs/direct_rl_env_adapter.py) 一致：

| 区间 | 含义 | 维度 |
|---|---|---|
| `[0:12)` | `joint_positions` | 12 |
| `[12:24)` | `joint_velocities` | 12 |
| `[24:36)` | `prev_joint_target` | 12 |
| `[36:55)` | left ee: `position + orientation + linear_velocity + angular_velocity + linear_acceleration + angular_acceleration` | 19 |
| `[55:74)` | right ee: 同上 | 19 |
| `[74:76)` | `left_occupied + right_occupied` | 2 |
| `[76:78)` | `left_tilt + right_tilt` | 2 |
| `[78:84)` | `door_center_in_base + door_normal_in_base` | 6 |

即：

`actor_obs = proprio(36) + ee(38) + context(2) + stability(2) + door_geometry(6)`

## 3. Critic 观测布局

Critic flat tensor 维度为 `97`：

`critic_obs = actor_obs_clean(84) + privileged(13)`

`privileged` 的顺序为：

| 区间 | 含义 | 维度 |
|---|---|---|
| `[84:91)` | `door_pose` | 7 |
| `[91:92)` | `door_joint_pos` | 1 |
| `[92:93)` | `door_joint_vel` | 1 |
| `[93:94)` | `cup_mass` | 1 |
| `[94:95)` | `door_mass` | 1 |
| `[95:96)` | `door_damping` | 1 |
| `[96:97)` | `cup_dropped` | 1 |

## 4. 适配器输出的嵌套结构

训练侧通过 `DirectRLEnvAdapter` 读到的 actor 观测结构为：

```python
actor_obs = {
    "proprio": {
        "joint_positions": (12,),
        "joint_velocities": (12,),
        "prev_joint_target": (12,),
    },
    "ee": {
        "left": {
            "position": (3,),
            "orientation": (4,),
            "linear_velocity": (3,),
            "angular_velocity": (3,),
            "linear_acceleration": (3,),
            "angular_acceleration": (3,),
        },
        "right": {
            "position": (3,),
            "orientation": (4,),
            "linear_velocity": (3,),
            "angular_velocity": (3,),
            "linear_acceleration": (3,),
            "angular_acceleration": (3,),
        },
    },
    "context": {
        "left_occupied": (1,),
        "right_occupied": (1,),
    },
    "stability": {
        "left_tilt": (1,),
        "right_tilt": (1,),
    },
    "door_geometry": {
        "door_center_in_base": (3,),
        "door_normal_in_base": (3,),
    },
}
```

critic 结构为：

```python
critic_obs = {
    "actor_obs": actor_obs,
    "privileged": {
        "door_pose": (7,),
        "door_joint_pos": (1,),
        "door_joint_vel": (1,),
        "cup_mass": (1,),
        "door_mass": (1,),
        "door_damping": (1,),
        "cup_dropped": (1,),
    },
}
```

## 5. 噪声与边界

当前默认路径下：

- actor 观测噪声只注入 `joint_positions` 和 `joint_velocities`
- critic 永远读取无噪声真值
- `prev_joint_target` 是 clip 后真正送入仿真的目标角，不是 raw policy action
- `door_geometry` 来自门叶刚体位姿和固定局部偏移的几何计算，不依赖视觉模块

## 6. 已删除的旧字段

以下字段已经从默认观测接口中删除：

- `joint_torques`
- `prev_action`

如果其他文档或外部脚本仍引用这些名字，应按当前接口改为：

- `joint_torques` -> 删除
- `prev_action` -> `prev_joint_target`
