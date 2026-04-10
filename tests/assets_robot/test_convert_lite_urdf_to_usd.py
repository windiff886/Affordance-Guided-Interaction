from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "assets/robot/scripts/convert_lite_urdf_to_usd.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_lite_urdf_to_usd", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_ros_package_overlay_creates_named_package_dir(tmp_path):
    module = _load_module()

    mesh_dir = tmp_path / "meshes" / "z1"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "collision").mkdir()
    (mesh_dir / "visual").mkdir()

    overlay_root = tmp_path / "overlay"
    package_root = module.prepare_ros_package_overlay(mesh_dir=mesh_dir, overlay_root=overlay_root)

    assert package_root.name == "z1_description"
    assert package_root.parent == overlay_root
    assert package_root.is_dir()
    assert (package_root / "meshes").is_symlink()
    assert (package_root / "package.xml").is_file()


def test_prepend_ros_package_path_uses_overlay_parent_directory(tmp_path):
    module = _load_module()

    env = {"ROS_PACKAGE_PATH": "/existing/path"}
    overlay_root = tmp_path / "overlay_root"

    module.prepend_ros_package_path(env, overlay_root)

    assert env["ROS_PACKAGE_PATH"] == f"{overlay_root}:/existing/path"
