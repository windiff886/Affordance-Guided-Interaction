# RL-Games Alignment Review

当前仓库只保留以下训练主链路：

`YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher -> gym.make -> RlGamesVecEnvWrapper -> rl_games.Runner`

审查结论：

- 默认训练入口已对齐官方 IsaacLab `rl_games` 流程。
- 旧的自定义 PPO、rollout、adapter、导出与评估脚本已从主路径移除。
- 配置面仅保留新链路会消费的 YAML 文件。
- 测试面仅保留 task registration、YAML 覆盖、reward 注入和基础数学逻辑的覆盖。

后续约束：

- 新增训练、评估、导出功能时，必须直接面向 `rl_games` 产物实现。
- 不再恢复自定义 PPO 主循环、旧 checkpoint 格式或 adapter 兼容层。
