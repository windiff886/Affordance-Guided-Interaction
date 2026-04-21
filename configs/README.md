# configs/

当前默认训练链路会消费以下 4 份配置：

- `training/default.yaml`
- `env/default.yaml`
- `task/default.yaml`
- `reward/default.yaml`

另外，仓库还提供 1 份推理/可视化配置：

- `inference/default.yaml`

训练入口只走这一条链路：

`YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher -> gym.make -> RlGamesVecEnvWrapper -> rl_games.Runner`

说明：

- `training/default.yaml` 提供运行时参数和 rl_games/PPO 超参数。
- `env/default.yaml` 覆盖仿真步长、decimation、双臂控制参数以及移动底盘速度/轮系参数。
- `task/default.yaml` 覆盖任务成功判据相关参数。
- `reward/default.yaml` 通过 `scripts/train.py` 的 `_inject_reward_params()` 写回 `DoorPushEnvCfg`，其中也包含移动底盘的安全项系数。
- `inference/default.yaml` 供 `scripts/render_policy_rollouts.py` 使用，负责 checkpoint、推理模式、视频输出目录、录制步数、设备和录制相机 `eye/lookat`。

仓库中已删除旧训练链路对应的 `policy`、`curriculum`、`visualization` 配置，以及自定义 PPO / rollout / adapter 相关入口。
