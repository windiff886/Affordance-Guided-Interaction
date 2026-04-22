"""
URDF -> USD 轻量版极简转换脚本。

目标：尽量直接照抄 assets/Z1_ISAACSIM 的导入链路，只做最朴素的
URDFCreateImportConfig + URDFParseAndImportFile 导入，再补最小后处理，
直接导出训练/调试共用的轻量版 USD。

这个脚本会把 uni_dingo_lite.urdf 直接导出为：
    assets/robot/usd/uni_dingo_lite.usd

明确不做：
    - 空碰撞修复
    - RobotAsset 打包
    - colliders 内联
    - deinstance

已加入：
    - convex decomposition (凸分解)
    - 关节驱动配置 (arms=force, planar base=velocity)
    - wheel collision deactivation（保留 wheel visual，禁用 wheel-ground contact）

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
from affordance_guided_interaction.envs.base_control_math import default_dingo_mecanum_angles_deg

configure_omniverse_client_environment(os.environ)

_Z1_MESH_DIR = PROJECT_ROOT / "assets/robot/meshes/z1"
_DINGO_MESH_DIR = PROJECT_ROOT / "assets/robot/meshes/dingo"
_ROS_OVERLAY_ROOT = PROJECT_ROOT / "assets/robot/.ros_pkg_overlay"
_Z1_PACKAGE_NAME = "z1_description"
_DINGO_PACKAGE_NAME = "dingo_description"


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


def _setup_package_overlay(mesh_dir: Path, package_name: str, env: dict[str, str]) -> None:
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    package_root = _ROS_OVERLAY_ROOT / package_name
    package_root.mkdir(parents=True, exist_ok=True)

    meshes_link = package_root / "meshes"
    if meshes_link.is_symlink() and meshes_link.resolve() != mesh_dir.resolve():
        meshes_link.unlink()
    elif meshes_link.exists() and not meshes_link.is_symlink():
        raise RuntimeError(
            f"Expected symlink at {meshes_link}, found an existing non-symlink path."
        )

    if not meshes_link.exists():
        meshes_link.symlink_to(mesh_dir)

    _write_minimal_package_xml(package_root, package_name=package_name)

    overlay_entry = str(_ROS_OVERLAY_ROOT)
    existing = [entry for entry in env.get("ROS_PACKAGE_PATH", "").split(":") if entry]
    env["ROS_PACKAGE_PATH"] = ":".join([overlay_entry] + [entry for entry in existing if entry != overlay_entry])


def configure_lite_mesh_package(env: dict[str, str]) -> None:
    _setup_package_overlay(_Z1_MESH_DIR, _Z1_PACKAGE_NAME, env)
    _setup_package_overlay(_DINGO_MESH_DIR, _DINGO_PACKAGE_NAME, env)


# lite 版本所有 actuated 关节
_LITE_ARM_JOINTS = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6", "left_jointGripper",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6", "right_jointGripper",
]

# Dingo-O 轮子关节（planar base 模式下仅保留为被动视觉/接触件）
_WHEEL_JOINTS = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]
_PLANAR_TRANSLATION_JOINTS = [
    "base_x_joint",
    "base_y_joint",
]
_PLANAR_ROTATION_JOINTS = [
    "base_yaw_joint",
]
_DEFAULT_WHEEL_RADIUS_METERS = 0.05
_TRAINING_DISABLED_WHEEL_PRIMS = (
    "/uni_dingo_lite/front_left_wheel_link/collisions",
    "/uni_dingo_lite/front_right_wheel_link/collisions",
    "/uni_dingo_lite/rear_left_wheel_link/collisions",
    "/uni_dingo_lite/rear_right_wheel_link/collisions",
    "/colliders/front_left_wheel_link",
    "/colliders/front_right_wheel_link",
    "/colliders/rear_left_wheel_link",
    "/colliders/rear_right_wheel_link",
)


def configure_joint_drives(stage, robot_prim_path: str) -> int:
    """配置关节驱动：臂用力矩驱动，平面底盘用速度驱动，轮子保持被动。"""
    from pxr import UsdPhysics

    drive_count = 0
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
            continue
        name = prim.GetName()
        prim_path = str(prim.GetPath())
        if robot_prim_path not in prim_path:
            continue

        is_wheel = any(wheel_name in name for wheel_name in _WHEEL_JOINTS)
        is_arm = any(joint_name in name for joint_name in _LITE_ARM_JOINTS)
        is_planar_translation = name in _PLANAR_TRANSLATION_JOINTS
        is_planar_rotation = name in _PLANAR_ROTATION_JOINTS

        if is_planar_translation:
            drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.CreateTypeAttr("velocity")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(100000.0)
            drive_count += 1
        elif is_planar_rotation:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("velocity")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(100000.0)
            drive_count += 1
        elif is_wheel:
            continue
        elif is_arm:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(1000.0)
            drive.CreateDampingAttr(100.0)
            drive_count += 1
    return drive_count


def configure_wheel_holonomic_metadata(
    stage,
    robot_prim_path: str,
    *,
    wheel_radius: float,
) -> int:
    """Annotate Dingo wheel joints with Isaac Sim holonomic metadata."""
    from pxr import Sdf, UsdPhysics

    mecanum_angles = dict(zip(_WHEEL_JOINTS, default_dingo_mecanum_angles_deg(_WHEEL_JOINTS), strict=True))
    metadata_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if robot_prim_path not in str(prim.GetPath()):
            continue
        joint_name = prim.GetName()
        if joint_name not in mecanum_angles:
            continue

        radius_attr = prim.GetAttribute("isaacmecanumwheel:radius")
        if not radius_attr.IsValid():
            radius_attr = prim.CreateAttribute("isaacmecanumwheel:radius", Sdf.ValueTypeNames.Float, True)
        radius_attr.Set(float(wheel_radius))

        angle_attr = prim.GetAttribute("isaacmecanumwheel:angle")
        if not angle_attr.IsValid():
            angle_attr = prim.CreateAttribute("isaacmecanumwheel:angle", Sdf.ValueTypeNames.Float, True)
        angle_attr.Set(float(mecanum_angles[joint_name]))
        metadata_count += 1
    return metadata_count


def get_applied_references(prim) -> list:
    refs = prim.GetMetadata("references")
    if not refs:
        return []
    return list(refs.GetAppliedItems())


def clear_invalid_internal_references(stage) -> int:
    """Drop unresolved internal references authored by the URDF importer."""
    cleared = 0
    for prim in stage.Traverse():
        applied_refs = get_applied_references(prim)
        if not applied_refs:
            continue
        valid_refs = []
        changed = False
        for ref in applied_refs:
            if ref.assetPath or not ref.primPath:
                valid_refs.append(ref)
                continue
            target_prim = stage.GetPrimAtPath(ref.primPath)
            if target_prim and target_prim.IsValid():
                valid_refs.append(ref)
            else:
                changed = True
        if not changed:
            continue
        references = prim.GetReferences()
        if valid_refs:
            references.SetReferences(valid_refs)
        else:
            references.ClearReferences()
        cleared += 1
    return cleared


def deactivate_wheel_collision_prims(stage) -> int:
    """Disable wheel collision prims while keeping wheel visuals and joints intact."""
    deactivated = 0
    for prim_path in _TRAINING_DISABLED_WHEEL_PRIMS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsActive():
            continue
        prim.SetActive(False)
        deactivated += 1
    return deactivated


def enable_self_collision(stage, robot_prim_path: str) -> None:
    """在 articulation root 及所有子 prim 上启用自碰撞检测。"""
    from pxr import PhysxSchema, Sdf, Usd

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim or not robot_prim.IsValid():
        return
    if not robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    # 设置根节点
    attr = robot_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
    if not attr.IsValid():
        attr = robot_prim.CreateAttribute(
            "physxArticulation:enabledSelfCollisions",
            Sdf.ValueTypeNames.Bool,
            True,
        )
    attr.Set(True)
    # 覆盖所有子 prim 的 False 值
    for prim in Usd.PrimRange(robot_prim):
        child_attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if child_attr.IsValid() and child_attr.Get() == False:
            child_attr.Set(True)


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
        import_config.convex_decomp = True
        import_config.fix_base = True
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

        log(f"✅ Lite URDF 已导入: {robot_prim_path}")

        drive_count = configure_joint_drives(stage, robot_prim_path)
        log(f"✅ 已配置 {drive_count} 个关节驱动 (arms=force, planar_base=velocity)")

        deactivated_wheel_prims = deactivate_wheel_collision_prims(stage)
        log(f"✅ 已停用 {deactivated_wheel_prims} 个 wheel collision prim")

        cleared_refs = clear_invalid_internal_references(stage)
        log(f"✅ 已清理 {cleared_refs} 个悬空内部引用")

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
