"""DoorPush direct RL task registration."""

from __future__ import annotations

import gymnasium as gym

from . import agents


TASK_NAME = "Affordance-DoorPush-Direct-v0"


gym.register(
    id=TASK_NAME,
    entry_point="affordance_guided_interaction.envs.door_push_env:DoorPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "affordance_guided_interaction.envs.door_push_env_cfg:DoorPushEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
