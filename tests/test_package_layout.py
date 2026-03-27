from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_package_exposes_version() -> None:
    import affordance_guided_interaction as package

    assert package.__version__ == "0.1.0"


def test_package_exports_framework_root() -> None:
    import affordance_guided_interaction as package

    assert package.PACKAGE_ROOT.name == "affordance_guided_interaction"
