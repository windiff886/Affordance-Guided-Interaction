# Reward Scaling Design

## Goal

Adjust reward weights so that:

- `r_task` becomes `500x` its current scale.
- `r_stab` becomes `1/100` of its current scale.
- TensorBoard continues to show the true reward contributions used by training.

## Current Reward Structure

The training pipeline loads reward hyperparameters from `configs/reward/default.yaml`, injects them into `DoorPushEnvCfg`, and computes reward directly in `door_push_env.py`.

- Task reward:
  - `r_task = r_task_delta + r_task_open_bonus`
  - `r_task_delta` is controlled by `task.w_delta`
  - `r_task_open_bonus` is controlled by `task.w_open`
- Stability reward:
  - `r_stab` is the sum of seven per-side terms
  - each term is scaled by a `stability.w_*` coefficient

TensorBoard already logs the signed reward contributions after they are computed, so changing the source weights is sufficient to make logged values reflect the new real magnitudes.

## Chosen Approach

Apply a pure linear rescaling to reward weights only.

### Task reward

Multiply these fields by `500`:

- `task.w_delta`
- `task.w_open`

Keep these fields unchanged because they affect reward shape rather than linear magnitude:

- `task.alpha`
- `task.k_decay`
- `task.success_angle_threshold`

### Stability reward

Multiply these fields by `0.01`:

- `stability.w_zero_acc`
- `stability.w_zero_ang`
- `stability.w_acc`
- `stability.w_ang`
- `stability.w_tilt`
- `stability.w_smooth`
- `stability.w_reg`

Keep these fields unchanged because they define the Gaussian kernel shape rather than linear magnitude:

- `stability.lambda_acc`
- `stability.lambda_ang`

## Resulting Config Values

### Task

- `w_delta: 10.0 -> 5000.0`
- `w_open: 50.0 -> 25000.0`

### Stability

- `w_zero_acc: 1.0 -> 0.01`
- `w_zero_ang: 0.5 -> 0.005`
- `w_acc: 0.5 -> 0.005`
- `w_ang: 0.3 -> 0.003`
- `w_tilt: 0.3 -> 0.003`
- `w_smooth: 0.1 -> 0.001`
- `w_reg: 0.01 -> 0.0001`

## Scope

Only change `configs/reward/default.yaml`.

Do not change:

- reward formulas in `src/affordance_guided_interaction/envs/door_push_env.py`
- reward injection logic in `scripts/train.py`
- TensorBoard logging logic

## Verification

Verification for this change is configuration-focused:

- confirm the config file contains the intended scaled values
- confirm the reward injection path remains unchanged
- confirm TensorBoard logging continues to read from computed reward contributions, not from any separately scaled display value

## Risks

- `r_task` will dominate optimization much more aggressively than before.
- Stage 2 and Stage 3 may place much less optimization pressure on cup stability because `r_stab` becomes very small.
- The change is intentional, but expected training dynamics will shift substantially.
