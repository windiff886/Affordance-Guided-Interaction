"""UI window helpers for DirectRLEnv-based tasks."""

from __future__ import annotations

from isaaclab.envs.ui import BaseEnvWindow


class DirectRLEnvWindow(BaseEnvWindow):
    """BaseEnvWindow variant that skips manager panels absent in DirectRLEnv."""

    def _visualize_manager(self, title: str, class_name: str) -> None:
        if not hasattr(self.env, class_name):
            return
        if not hasattr(self.env, "manager_visualizers"):
            return
        if class_name not in self.env.manager_visualizers:
            return
        super()._visualize_manager(title, class_name)
