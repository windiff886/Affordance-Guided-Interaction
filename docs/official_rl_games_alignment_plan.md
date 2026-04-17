# RL-Games Alignment Plan

目标已经收敛为单一路径：仅保留官方 IsaacLab `rl_games` 训练链路。

保留内容：

- DoorPush task registry
- YAML 到 `env_cfg` / `agent_cfg` 的映射
- `scripts/train.py` 中的官方下游链路
- `DoorPushEnv` / `DoorPushEnvCfg`

训练流程：

`YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher -> gym.make -> RlGamesVecEnvWrapper -> rl_games.Runner`

清理原则：

- 删除自定义 PPO 训练栈
- 删除 adapter-based 训练与推理脚本
- 删除只服务于旧链路的配置、测试和文档
- 所有保留文档只描述当前有效链路
