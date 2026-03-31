from __future__ import annotations


def configure_omniverse_client_environment(env: dict[str, str]) -> None:
    env.setdefault("OMNICLIENT_HUB_MODE", "disabled")
