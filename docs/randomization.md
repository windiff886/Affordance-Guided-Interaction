# Randomization

当前随机化逻辑只存在于 `DoorPushEnvCfg` / `DoorPushEnv` 内部，并通过 `scripts/train.py` 构建的 `env_cfg` 进入训练环境。

新的随机化修改应直接落在：

- `src/affordance_guided_interaction/envs/door_push_env_cfg.py`
- `src/affordance_guided_interaction/envs/door_push_env.py`
