# Logging

当前训练日志由 `rl_games` 训练链路产生，日志目录根由 `training/default.yaml` 中的 `log_dir` 控制。

仓库中已删除旧的自定义 PPO 指标收集与 TensorBoard 命名规则。若需要新增日志，请直接基于当前 `rl_games` 训练链路扩展。
