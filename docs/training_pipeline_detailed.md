# Training Pipeline

当前训练入口为 `scripts/train.py`。

完整链路：

1. 从 `configs/` 读取 `training`、`env`、`task`、`reward` 四份 YAML。
2. 从任务注册表加载默认 `env_cfg` 与 `rl_games` agent config。
3. 将 YAML 覆盖写回 `env_cfg` 和 agent config。
4. 通过 `AppLauncher` 启动 Isaac Sim。
5. 使用 `gym.make(task_name, cfg=env_cfg, ...)` 创建环境。
6. 使用 `RlGamesVecEnvWrapper` 包装环境。
7. 通过 `rl_games.Runner.load(agent_cfg)` 和 `runner.run(...)` 执行训练。

仓库中不存在第二套训练主链路。
