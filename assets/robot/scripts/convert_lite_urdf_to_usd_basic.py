"""
URDF -> USD 轻量版极简转换脚本。

目标：尽量直接照抄 assets/Z1_ISAACSIM 的导入链路，只做最朴素的
URDFCreateImportConfig + URDFParseAndImportFile 导入，不做任何额外
资产后处理。

这个脚本会把 uni_dingo_lite.urdf 直接导出为：
    assets/robot/usd/uni_dingo_lite.usd

明确不做：
    - convex decomposition
    - 空碰撞修复
    - RobotAsset 打包
    - colliders 内联
    - deinstance
    - 关节驱动重写

用法:
    python assets/robot/scripts/convert_lite_urdf_to_usd_basic.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.utils.runtime_env import (
    configure_omniverse_client_environment,
)

configure_omniverse_client_environment(os.environ)

_Z1_MESH_DIR = PROJECT_ROOT / "assets/robot/meshes/z1"
_ROS_OVERLAY_ROOT = PROJECT_ROOT / "assets/robot/.ros_pkg_overlay"
_Z1_PACKAGE_NAME = "z1_description"


def log(message: str) -> None:
    print(message, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite URDF -> USD 极简直导")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="无窗口模式 (默认开启)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "assets/robot/urdf/uni_dingo_lite.urdf",
        help="输入轻量版 URDF 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "assets/robot/usd/uni_dingo_lite.usd",
        help="输出轻量版 USD 文件路径",
    )
    return parser


def _write_minimal_package_xml(package_root: Path, *, package_name: str) -> None:
    package_xml = package_root / "package.xml"
    package_xml.write_text(
        (
            "<?xml version=\"1.0\"?>\n"
            "<package format=\"3\">\n"
            f"  <name>{package_name}</name>\n"
            "  <version>0.0.0</version>\n"
            "  <description>Temporary overlay package for Isaac Sim URDF import.</description>\n"
            "  <maintainer email=\"noreply@example.com\">Codex</maintainer>\n"
            "  <license>Apache-2.0</license>\n"
            "</package>\n"
        ),
        encoding="utf-8",
    )


def configure_lite_mesh_package(env: dict[str, str]) -> None:
    if not _Z1_MESH_DIR.exists():
        raise FileNotFoundError(f"Z1 mesh directory not found: {_Z1_MESH_DIR}")

    package_root = _ROS_OVERLAY_ROOT / _Z1_PACKAGE_NAME
    package_root.mkdir(parents=True, exist_ok=True)

    meshes_link = package_root / "meshes"
    if meshes_link.is_symlink() and meshes_link.resolve() != _Z1_MESH_DIR.resolve():
        meshes_link.unlink()
    elif meshes_link.exists() and not meshes_link.is_symlink():
        raise RuntimeError(
            f"Expected symlink at {meshes_link}, found an existing non-symlink path."
        )

    if not meshes_link.exists():
        meshes_link.symlink_to(_Z1_MESH_DIR)

    _write_minimal_package_xml(package_root, package_name=_Z1_PACKAGE_NAME)

    overlay_entry = str(_ROS_OVERLAY_ROOT)
    existing = [entry for entry in env.get("ROS_PACKAGE_PATH", "").split(":") if entry]
    env["ROS_PACKAGE_PATH"] = ":".join([overlay_entry] + [entry for entry in existing if entry != overlay_entry])


def run_conversion(args: argparse.Namespace) -> int:
    urdf_path = args.urdf.resolve()
    output_path = args.output.resolve()

    if not urdf_path.exists():
        log(f"❌ URDF 不存在: {urdf_path}")
        return 1

    configure_lite_mesh_package(os.environ)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={"headless": args.headless, "width": 1280, "height": 720}
    )
    log("✅ Isaac Sim 已启动")

    try:
        import omni
        import omni.kit.commands
        from pxr import UsdGeom

        omni.usd.get_context().new_stage()
        simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            log("❌ 创建 URDF 导入配置失败")
            return 1

        import_config.merge_fixed_joints = False
        import_config.fix_base = False
        import_config.make_default_prim = True
        import_config.import_inertia_tensor = True
        import_config.create_physics_scene = False
        import_config.collision_from_visuals = False
        import_config.self_collision = False
        import_config.distance_scale = 1.0

        log(f"📂 直接导入 Lite URDF: {urdf_path}")
        status, robot_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
        simulation_app.update()

        if not status or not robot_prim_path:
            log("❌ Lite URDF 导入失败")
            return 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(output_path))
        file_kb = output_path.stat().st_size / 1024
        log(f"✅ Lite URDF 已导入: {robot_prim_path}")
        log(f"✅ Lite USD 已保存: {output_path}")
        log(f"   文件大小: {file_kb:.1f} KB")
        return 0
    finally:
        simulation_app.close()
        log("👋 极简转换流程结束")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_conversion(args)


if __name__ == "__main__":
    raise SystemExit(main())
