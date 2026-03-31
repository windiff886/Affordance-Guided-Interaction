from __future__ import annotations

from pathlib import Path
from os import PathLike


def to_usd_asset_path(path: str | PathLike[str]) -> str:
    if isinstance(path, Path):
        return str(path)
    return str(path)
