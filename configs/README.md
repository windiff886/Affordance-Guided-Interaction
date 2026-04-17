# configs/

当前仅保留新训练链路会消费的 4 份配置：

- `training/default.yaml`
- `env/default.yaml`
- `task/default.yaml`
- `reward/default.yaml`

训练入口只走这一条链路：

`YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher -> gym.make -> RlGamesVecEnvWrapper -> rl_games.Runner`

说明：

- `training/default.yaml` 提供运行时参数和 rl_games/PPO 超参数。
- `env/default.yaml` 覆盖仿真步长、decimation、控制参数。
- `task/default.yaml` 覆盖任务成功判据相关参数。
- `reward/default.yaml` 通过 `scripts/train.py` 的 `_inject_reward_params()` 写回 `DoorPushEnvCfg`。

仓库中已删除旧链路对应的 `policy`、`curriculum`、`visualization` 配置，以及自定义 PPO / rollout / adapter 相关入口。
