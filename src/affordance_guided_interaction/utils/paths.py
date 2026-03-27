from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"

