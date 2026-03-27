from __future__ import annotations

from typing import Any


def build_critic_observation(
    *,
    actor_obs: dict[str, Any],
    privileged: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "actor_obs": actor_obs,
        "privileged": privileged or {},
    }

