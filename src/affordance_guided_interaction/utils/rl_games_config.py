from __future__ import annotations

from copy import deepcopy
from typing import Any

_RL_GAMES_OBS_GROUPS = {"obs": ["policy"], "states": ["critic"]}


def build_rl_games_wrapper_kwargs() -> dict[str, dict[str, list[str]]]:
    """Return the explicit Isaac Lab observation-group mapping for rl_games."""
    return {"obs_groups": deepcopy(_RL_GAMES_OBS_GROUPS)}


def ensure_central_value_config(agent_cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure rl_games uses a central value network fed by Isaac Lab critic states."""
    params = agent_cfg.setdefault("params", {})
    network_cfg = params.get("network")
    if not isinstance(network_cfg, dict):
        raise ValueError("agent_cfg['params']['network'] must be a dict.")

    config_cfg = params.setdefault("config", {})

    cv_cfg = deepcopy(config_cfg.get("central_value_config", {}))
    cv_network_cfg = deepcopy(cv_cfg.get("network", {}))

    cv_cfg["minibatch_size"] = int(config_cfg["minibatch_size"])
    cv_cfg["mini_epochs"] = int(config_cfg["mini_epochs"])
    cv_cfg["learning_rate"] = float(config_cfg["learning_rate"])
    cv_cfg["lr_schedule"] = config_cfg.get("lr_schedule")
    if "kl_threshold" in config_cfg:
        cv_cfg["kl_threshold"] = float(config_cfg["kl_threshold"])
    cv_cfg["clip_value"] = bool(config_cfg["clip_value"])
    cv_cfg["normalize_input"] = bool(config_cfg["normalize_input"])
    cv_cfg["truncate_grads"] = bool(config_cfg["truncate_grads"])
    if "grad_norm" in config_cfg:
        cv_cfg["grad_norm"] = float(config_cfg["grad_norm"])

    cv_network_cfg["name"] = network_cfg.get("name", "actor_critic")
    cv_network_cfg["central_value"] = True
    cv_network_cfg["mlp"] = deepcopy(network_cfg.get("mlp", {}))
    cv_cfg["network"] = cv_network_cfg

    config_cfg["central_value_config"] = cv_cfg
    return agent_cfg
