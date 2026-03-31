from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ObservationDict = dict[str, Any]
RewardDict = dict[str, float]


@dataclass(slots=True)
class TaskContext:
    occupied: int = 0


@dataclass(slots=True)
class ActorObservation:
    proprio: ObservationDict
    gripper_state: ObservationDict
    context: ObservationDict
    stability_proxy: ObservationDict
    z_aff: ObservationDict
    z_prog: ObservationDict
    contact_summary: ObservationDict | None = None


@dataclass(slots=True)
class CriticObservation:
    actor_obs: ObservationDict
    privileged: ObservationDict = field(default_factory=dict)

