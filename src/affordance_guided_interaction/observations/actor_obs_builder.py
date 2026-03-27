from __future__ import annotations

from typing import Any


def build_actor_observation(
    *,
    proprio: dict[str, Any],
    gripper_state: dict[str, Any],
    context: dict[str, Any],
    stability_proxy: dict[str, Any],
    z_aff: dict[str, Any],
    z_prog: dict[str, Any],
    contact_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "proprio": proprio,
        "gripper_state": gripper_state,
        "context": context,
        "stability_proxy": stability_proxy,
        "z_aff": z_aff,
        "z_prog": z_prog,
        "contact_summary": contact_summary,
    }

