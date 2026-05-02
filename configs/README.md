# configs/

Active training pipeline consumes 4 config files:

- `training/default.yaml` — runtime params and rl_games/PPO hyperparameters (RoboDuet aligned)
- `env/default.yaml` — simulation timestep, decimation, arm/base control parameters
- `task/default.yaml` — task success/failure thresholds (theta_open, theta_pass, theta_hat)
- `reward/default.yaml` — reward coefficients injected back into DoorPushEnvCfg

Inference/visualization config:

- `inference/default.yaml` — checkpoint, inference mode, video output for render_policy_rollouts.py

Training entry flow:

`YAML -> task registry -> env_cfg / agent_cfg -> AppLauncher -> gym.make -> RlGamesVecEnvWrapper -> rl_games.Runner`

Notes:

- No curriculum config. The occupancy curriculum has been removed.
- No cup/occupancy parameters in any active config.
- PPO uses RoboDuet unified values: horizon=24, epochs=5, batches=4, LR=1e-3, entropy=0.01, KL=0.01.
- Raw actions enter the environment without rl_games action preprocessing (`config.clip_actions=false`).
- The IsaacLab wrapper keeps `env.clip_actions=1e6` only as an emergency bound; it must not be used by rl_games to rescale actions.
