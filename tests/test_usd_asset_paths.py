from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_to_usd_asset_path_converts_path_objects_to_strings() -> None:
    from affordance_guided_interaction.utils.usd_assets import to_usd_asset_path

    asset_path = Path("/tmp/example.usd")

    assert to_usd_asset_path(asset_path) == "/tmp/example.usd"


def test_to_usd_asset_path_preserves_string_inputs() -> None:
    from affordance_guided_interaction.utils.usd_assets import to_usd_asset_path

    assert to_usd_asset_path("/tmp/example.usd") == "/tmp/example.usd"
