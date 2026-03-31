from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_configure_omniverse_client_environment_disables_hub_by_default() -> None:
    from affordance_guided_interaction.utils.runtime_env import (
        configure_omniverse_client_environment,
    )

    env = {}
    configure_omniverse_client_environment(env)

    assert env["OMNICLIENT_HUB_MODE"] == "disabled"


def test_configure_omniverse_client_environment_preserves_existing_hub_mode() -> None:
    from affordance_guided_interaction.utils.runtime_env import (
        configure_omniverse_client_environment,
    )

    env = {"OMNICLIENT_HUB_MODE": "shared"}
    configure_omniverse_client_environment(env)

    assert env["OMNICLIENT_HUB_MODE"] == "shared"
