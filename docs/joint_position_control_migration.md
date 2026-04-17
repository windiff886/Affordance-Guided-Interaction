# Joint Position Control

当前 DoorPush 训练路径默认使用关节位置控制，并通过 `configs/env/default.yaml` 中的 `control` 字段覆盖到 `DoorPushEnvCfg`。

控制参数由 `scripts/train.py` 写回注册表默认 `env_cfg` 后，再通过 `gym.make(...)` 创建环境。仓库中不存在旧控制链路的兼容分支。
