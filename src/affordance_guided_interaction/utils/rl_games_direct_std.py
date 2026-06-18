from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


DIRECT_STD_MODEL_NAME = "continuous_a2c_direct_std"
DIRECT_STD_NETWORK_NAME = "door_push_actor_critic_direct_std"

DEFAULT_ARM_ACTION_DIM = 12
DEFAULT_BASE_ACTION_DIM = 3
DEFAULT_ARM_STD = 0.1
DEFAULT_BASE_STD = 1.0
DEFAULT_RAW_ACTION_SATURATION_THRESHOLD = 1.0

_REGISTERED = False


def normalize_direct_std_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized RoboDuet-style direct-std config."""
    config = deepcopy(config or {})
    normalized = {
        "enabled": bool(config.get("enabled", True)),
        "arm_action_dim": int(config.get("arm_action_dim", DEFAULT_ARM_ACTION_DIM)),
        "base_action_dim": int(config.get("base_action_dim", DEFAULT_BASE_ACTION_DIM)),
        "arm_init": float(config.get("arm_init", DEFAULT_ARM_STD)),
        "base_init": float(config.get("base_init", DEFAULT_BASE_STD)),
        "validate_positive": bool(config.get("validate_positive", True)),
        "raw_action_saturation_threshold": float(
            config.get("raw_action_saturation_threshold", DEFAULT_RAW_ACTION_SATURATION_THRESHOLD)
        ),
    }
    if normalized["arm_action_dim"] < 0:
        raise ValueError(f"direct_std.arm_action_dim must be non-negative, got {normalized['arm_action_dim']}.")
    if normalized["base_action_dim"] < 0:
        raise ValueError(f"direct_std.base_action_dim must be non-negative, got {normalized['base_action_dim']}.")
    if normalized["arm_init"] <= 0.0:
        raise ValueError(f"direct_std.arm_init must be positive, got {normalized['arm_init']}.")
    if normalized["base_init"] <= 0.0:
        raise ValueError(f"direct_std.base_init must be positive, got {normalized['base_init']}.")
    if normalized["raw_action_saturation_threshold"] <= 0.0:
        raise ValueError(
            "direct_std.raw_action_saturation_threshold must be positive, "
            f"got {normalized['raw_action_saturation_threshold']}."
        )
    return normalized


def register_direct_std_rl_games_components() -> None:
    """Register DoorPush direct-std rl_games model/network components."""
    global _REGISTERED
    if _REGISTERED:
        return

    from rl_games.algos_torch import model_builder, network_builder
    from rl_games.algos_torch.models import BaseModel, BaseModelNetwork
    from rl_games.common import divergence

    class DoorPushDirectStdNetwork(network_builder.A2CBuilder.Network):
        _policy_diag_mu_abs_sum: torch.Tensor | None
        _policy_diag_mu_abs_max: torch.Tensor | None
        _policy_diag_mu_count: int
        _policy_diag_raw_abs_sum: torch.Tensor | None
        _policy_diag_raw_abs_max: torch.Tensor | None
        _policy_diag_raw_saturated_sum: torch.Tensor | None
        _policy_diag_raw_count: int

        def __init__(self, params: dict[str, Any], **kwargs):
            super().__init__(params, **kwargs)
            self.direct_std_config = normalize_direct_std_config(
                params.get("space", {}).get("continuous", {}).get("direct_std")
            )
            if self.direct_std_config["enabled"]:
                self._initialize_direct_std()
            self._reset_policy_diagnostics()

        def _initialize_direct_std(self) -> None:
            if not getattr(self, "is_continuous", False):
                raise ValueError("DoorPush direct std requires a continuous action space.")
            if not getattr(self, "fixed_sigma", False):
                raise ValueError("DoorPush direct std requires fixed_sigma=True.")
            if self.space_config.get("sigma_activation") != "None":
                raise ValueError("DoorPush direct std requires sigma_activation: None.")
            if not isinstance(self.sigma, torch.nn.Parameter):
                raise ValueError("DoorPush direct std requires sigma to be a trainable parameter.")

            arm_dim = int(self.direct_std_config["arm_action_dim"])
            base_dim = int(self.direct_std_config["base_action_dim"])
            expected_dim = arm_dim + base_dim
            action_dim = int(self.sigma.numel())
            if expected_dim != action_dim:
                raise ValueError(
                    "direct_std arm/base dimensions must match action dimension. "
                    f"Got arm={arm_dim}, base={base_dim}, action_dim={action_dim}."
                )

            values = torch.empty_like(self.sigma)
            values[:arm_dim] = float(self.direct_std_config["arm_init"])
            values[arm_dim:] = float(self.direct_std_config["base_init"])
            with torch.no_grad():
                self.sigma.copy_(values)

        def forward(self, obs_dict):
            result = super().forward(obs_dict)
            if self.direct_std_config.get("enabled") and self.direct_std_config.get("validate_positive", True):
                sigma = result[1]
                _validate_sigma(sigma)
            return result

        def record_policy_diagnostics(self, mu: torch.Tensor, raw_actions: torch.Tensor | None) -> None:
            if raw_actions is None:
                return

            with torch.no_grad():
                mu_abs = mu.detach().abs()
                raw_abs = raw_actions.detach().abs()
                threshold = float(self.direct_std_config["raw_action_saturation_threshold"])
                saturated = raw_abs > threshold

                self._policy_diag_mu_abs_sum = _sum_accumulate(self._policy_diag_mu_abs_sum, mu_abs.sum())
                self._policy_diag_mu_count += int(mu_abs.numel())
                self._policy_diag_mu_abs_max = _max_accumulate(self._policy_diag_mu_abs_max, mu_abs.max())

                self._policy_diag_raw_abs_sum = _sum_accumulate(self._policy_diag_raw_abs_sum, raw_abs.sum())
                self._policy_diag_raw_count += int(raw_abs.numel())
                self._policy_diag_raw_abs_max = _max_accumulate(self._policy_diag_raw_abs_max, raw_abs.max())
                self._policy_diag_raw_saturated_sum = _sum_accumulate(
                    self._policy_diag_raw_saturated_sum,
                    saturated.to(dtype=raw_abs.dtype).sum(),
                )

        def consume_policy_diagnostics(self) -> dict[str, float]:
            stats: dict[str, float] = {}
            if self._policy_diag_mu_count > 0:
                assert self._policy_diag_mu_abs_sum is not None
                assert self._policy_diag_mu_abs_max is not None
                stats["mu_abs_mean"] = float(
                    (self._policy_diag_mu_abs_sum / self._policy_diag_mu_count).detach().item()
                )
                stats["mu_abs_max"] = float(self._policy_diag_mu_abs_max.detach().item())
            if self._policy_diag_raw_count > 0:
                assert self._policy_diag_raw_abs_sum is not None
                assert self._policy_diag_raw_abs_max is not None
                assert self._policy_diag_raw_saturated_sum is not None
                stats["raw_action_abs_mean"] = float(
                    (self._policy_diag_raw_abs_sum / self._policy_diag_raw_count).detach().item()
                )
                stats["raw_action_abs_max"] = float(self._policy_diag_raw_abs_max.detach().item())
                stats["raw_action_saturation_rate"] = float(
                    (self._policy_diag_raw_saturated_sum / self._policy_diag_raw_count).detach().item()
                )
            self._reset_policy_diagnostics()
            return stats

        def _reset_policy_diagnostics(self) -> None:
            self._policy_diag_mu_abs_sum = None
            self._policy_diag_mu_abs_max = None
            self._policy_diag_mu_count = 0
            self._policy_diag_raw_abs_sum = None
            self._policy_diag_raw_abs_max = None
            self._policy_diag_raw_saturated_sum = None
            self._policy_diag_raw_count = 0

    class DoorPushDirectStdA2CBuilder(network_builder.A2CBuilder):
        def build(self, name, **kwargs):
            return DoorPushDirectStdNetwork(self.params, **kwargs)

    class ModelA2CContinuousDirectStd(BaseModel):
        def __init__(self, network):
            BaseModel.__init__(self, "a2c")
            self.network_builder = network

        class Network(BaseModelNetwork):
            def __init__(self, a2c_network, **kwargs):
                BaseModelNetwork.__init__(self, **kwargs)
                self.a2c_network = a2c_network

            def get_aux_loss(self):
                return self.a2c_network.get_aux_loss()

            def is_rnn(self):
                return self.a2c_network.is_rnn()

            def get_default_rnn_state(self):
                return self.a2c_network.get_default_rnn_state()

            def get_value_layer(self):
                return self.a2c_network.get_value_layer()

            def kl(self, p_dict, q_dict):
                p = p_dict["mu"], p_dict["sigma"]
                q = q_dict["mu"], q_dict["sigma"]
                return divergence.d_kl_normal(p, q)

            def forward(self, input_dict):
                is_train = input_dict.get("is_train", True)
                prev_actions = input_dict.get("prev_actions", None)
                input_dict["obs"] = self.norm_obs(input_dict["obs"])
                mu, sigma, value, states = self.a2c_network(input_dict)
                _validate_sigma(sigma)
                distr = torch.distributions.Normal(mu, sigma, validate_args=False)

                if is_train:
                    self.a2c_network.record_policy_diagnostics(mu, prev_actions)
                    entropy = distr.entropy().sum(dim=-1)
                    prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                    return {
                        "prev_neglogp": torch.squeeze(prev_neglogp),
                        "values": value,
                        "entropy": entropy,
                        "rnn_states": states,
                        "mus": mu,
                        "sigmas": sigma,
                    }

                selected_action = distr.sample()
                self.a2c_network.record_policy_diagnostics(mu, selected_action)
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                entropy = distr.entropy().sum(dim=-1)
                return {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }

    model_builder.register_network(DIRECT_STD_NETWORK_NAME, DoorPushDirectStdA2CBuilder)
    model_builder.register_model(DIRECT_STD_MODEL_NAME, ModelA2CContinuousDirectStd)
    _REGISTERED = True


def _validate_sigma(sigma: torch.Tensor) -> None:
    if torch.isfinite(sigma).all() and torch.all(sigma > 0.0):
        return

    finite = sigma[torch.isfinite(sigma)]
    if finite.numel() > 0:
        min_value = float(finite.min().detach().cpu())
        max_value = float(finite.max().detach().cpu())
    else:
        min_value = float("nan")
        max_value = float("nan")
    raise FloatingPointError(
        "DoorPush direct std became non-positive or non-finite. "
        f"finite_min={min_value}, finite_max={max_value}."
    )


def _sum_accumulate(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if current is None:
        return value
    return current + value


def _max_accumulate(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if current is None:
        return value
    return torch.maximum(current, value)
