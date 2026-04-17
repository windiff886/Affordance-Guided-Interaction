from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainRuntimeConfig:
    """Resolved runtime settings for the rl_games-only training entrypoint."""

    headless: bool
    device: str | None
    seed: int
    resume: Path | None
    log_dir: Path
    num_envs: int | None


def resolve_train_runtime_config(
    cfg: dict[str, Any],
    *,
    project_root: Path,
) -> TrainRuntimeConfig:
    """Resolve runtime settings from the merged training YAML."""
    t_cfg = cfg.get("training", {})

    return TrainRuntimeConfig(
        headless=bool(t_cfg.get("headless", False)),
        device=_normalize_optional_str(t_cfg.get("device")),
        seed=int(t_cfg.get("seed", 42)),
        resume=_resolve_optional_path(project_root, t_cfg.get("resume")),
        log_dir=_resolve_required_path(project_root, t_cfg.get("log_dir", "runs")),
        num_envs=_normalize_optional_int(t_cfg.get("num_envs")),
    )


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_optional_path(project_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return _resolve_required_path(project_root, text)


def _resolve_required_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root / path
