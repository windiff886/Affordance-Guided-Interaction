from __future__ import annotations

from os import path as osp
from pathlib import Path


def configure_omniverse_client_environment(env: dict[str, str]) -> None:
    env.setdefault("OMNICLIENT_HUB_MODE", "disabled")
    runtime_root = Path(env.get("TMPDIR", "/tmp")) / "isaacsim-runtime"
    runtime_dirs = {
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "OMNI_CACHE_DIR": runtime_root / "cache",
        "OMNI_LOG_DIR": runtime_root / "logs",
        "OMNI_DATA_DIR": runtime_root / "data",
        "OMNI_USER_DIR": runtime_root / "user",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "XDG_DATA_HOME": runtime_root / "xdg-data",
        "XDG_CONFIG_HOME": runtime_root / "xdg-config",
    }
    for key, path in runtime_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        env.setdefault(key, str(path))


def resolve_headless_mode(requested_headless: bool, env: dict[str, str]) -> bool:
    """在无图形显示环境下自动回退到 headless 模式。"""
    if requested_headless:
        return True
    return not _has_valid_display(env)


def _has_valid_display(env: dict[str, str]) -> bool:
    wayland_display = str(env.get("WAYLAND_DISPLAY", "")).strip()
    if wayland_display:
        runtime_dir = str(env.get("XDG_RUNTIME_DIR", "")).strip()
        if runtime_dir:
            return osp.exists(osp.join(runtime_dir, wayland_display))

    display = str(env.get("DISPLAY", "")).strip()
    if not display:
        return False

    if display.startswith(":") and display[1:].isdigit():
        return osp.exists(f"/tmp/.X11-unix/X{display[1:]}")

    if display.startswith("unix/:") and display[6:].isdigit():
        return osp.exists(f"/tmp/.X11-unix/X{display[6:]}")

    # 远程 X11/Wayland 转发场景无法在本地验证 socket，保守地认为可用。
    return True
