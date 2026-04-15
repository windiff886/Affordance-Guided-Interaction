from __future__ import annotations

import os
import inspect
from typing import Any, Callable

from .runtime_env import (
    configure_omniverse_client_environment,
    resolve_headless_mode,
)


SimulationAppFactory = Callable[..., Any]
AppLauncherFactory = Callable[..., Any]
SimulationCfgFactory = Callable[..., Any]
SimulationContextFactory = Callable[[Any], Any]


def build_simulation_app_config(
    *,
    headless: bool,
    enable_cameras: bool = False,
    width: int = 1280,
    height: int = 720,
) -> dict[str, object]:
    """Build the launch config passed to Isaac Sim's ``SimulationApp``."""
    return {
        "headless": headless,
        "enable_cameras": enable_cameras,
        "width": width,
        "height": height,
    }


def launch_simulation_app(
    *,
    headless: bool,
    enable_cameras: bool = False,
    width: int = 1280,
    height: int = 720,
    app_factory: SimulationAppFactory | None = None,
    app_launcher_factory: AppLauncherFactory | None = None,
    env: dict[str, str] | None = None,
    import_error_message: str = (
        "未检测到 isaacsim 运行时，跳过 SimulationApp 启动。"
    ),
) -> Any | None:
    """Launch ``SimulationApp`` when Isaac Sim is available.

    The helper keeps train/eval scripts on a single bootstrap path while still
    allowing tests and pure-Python fallback environments to run without an
    Isaac Sim installation.
    """
    runtime_env = os.environ if env is None else env
    configure_omniverse_client_environment(runtime_env)
    resolved_headless = resolve_headless_mode(headless, runtime_env)

    factory = app_factory
    launcher_factory = app_launcher_factory

    if enable_cameras:
        if launcher_factory is None:
            try:
                from isaaclab.app import AppLauncher as launcher_factory
            except ImportError:
                try:
                    from omni.isaac.lab.app import AppLauncher as launcher_factory  # type: ignore
                except ImportError:
                    launcher_factory = None
        if launcher_factory is not None:
            try:
                launcher = launcher_factory(
                    headless=resolved_headless,
                    enable_cameras=True,
                    width=width,
                    height=height,
                )
                return getattr(launcher, "app", launcher)
            except Exception as exc:
                print(f"⚠️ AppLauncher 启动失败，回退到 SimulationApp: {exc}")

    if factory is None:
        try:
            from isaacsim import SimulationApp as factory  # type: ignore
        except ImportError:
            if import_error_message:
                print(f"⚠️ {import_error_message}")
            return None

    launch_config = build_simulation_app_config(
        headless=resolved_headless,
        enable_cameras=enable_cameras,
        width=width,
        height=height,
    )
    app = factory(launch_config=launch_config)
    if enable_cameras:
        _configure_isaaclab_camera_settings(headless=resolved_headless)
    return app


def close_simulation_app(
    app: Any | None,
    *,
    wait_for_replicator: bool = False,
    skip_cleanup: bool = False,
) -> None:
    """Close a SimulationApp-compatible object without assuming every runtime supports the same kwargs."""
    if app is None:
        return

    close_fn = getattr(app, "close", None)
    if close_fn is None:
        return

    kwargs: dict[str, object] = {}
    try:
        params = inspect.signature(close_fn).parameters
    except (TypeError, ValueError):
        params = {}

    if "wait_for_replicator" in params:
        kwargs["wait_for_replicator"] = wait_for_replicator
    if "skip_cleanup" in params:
        kwargs["skip_cleanup"] = skip_cleanup

    close_fn(**kwargs)


def _configure_isaaclab_camera_settings(*, headless: bool) -> None:
    """Set the Isaac Lab carb flags needed by camera sensors.

    .. deprecated::
        当前默认训练路径不使用相机传感器。此函数仅为历史兼容保留，
        不应在新代码中依赖。若未来需要恢复相机实验，应在独立实验配置中启用。
    """
    try:
        import carb
    except ImportError:
        return

    settings = carb.settings.get_settings()
    settings.set_bool("/isaaclab/cameras_enabled", True)
    settings.set_bool("/isaaclab/render/offscreen", headless)
    settings.set_bool("/isaaclab/render/active_viewport", not headless)
    settings.set_string("/isaaclab/rendering/rendering_mode", "")


def create_simulation_context(
    *,
    physics_dt: float,
    render_interval: int,
    device: str,
    sim_cfg_factory: SimulationCfgFactory | None = None,
    context_factory: SimulationContextFactory | None = None,
    import_error_message: str = (
        "未检测到 Isaac Lab 运行时，跳过 SimulationContext 创建。"
    ),
) -> Any | None:
    """Create an Isaac Lab ``SimulationContext`` when the runtime is available."""
    cfg_factory = sim_cfg_factory
    sim_factory = context_factory

    if cfg_factory is None or sim_factory is None:
        try:
            from isaaclab.sim import SimulationCfg, SimulationContext
        except ImportError:
            try:
                from omni.isaac.lab.sim import SimulationCfg, SimulationContext  # type: ignore
            except ImportError:
                if import_error_message:
                    print(f"⚠️ {import_error_message}")
                return None
        cfg_factory = cfg_factory or SimulationCfg
        sim_factory = sim_factory or SimulationContext

    sim_cfg = cfg_factory(
        dt=float(physics_dt),
        render_interval=int(render_interval),
        device=str(device),
    )
    return sim_factory(sim_cfg)
