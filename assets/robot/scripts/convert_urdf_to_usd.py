"""
URDF → USD 严格转换脚本

本脚本将 Uni-Dingo 机器人 URDF 导入 Isaac Sim，并在导出前完成：
    1. 启用 Articulation 自碰撞检测
    2. 配置轮子/机械臂关节驱动
    3. 自动修复 importer 留下的空 collisions 子树
    4. 逐 link 校验 USD 是否完整保留了 URDF 的 visual/collision 语义

校验未通过时脚本会直接失败退出，不导出一个语义缺失的 USD 文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from affordance_guided_interaction.utils.runtime_env import (
    configure_omniverse_client_environment,
)

configure_omniverse_client_environment(os.environ)

ARM_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_jointGripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_jointGripper",
    "pan_tilt_yaw_joint",
    "pan_tilt_pitch_joint",
]
WHEEL_JOINTS = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]


@dataclass(frozen=True)
class URDFLinkExpectation:
    name: str
    has_visual: bool
    has_collision: bool


@dataclass(frozen=True)
class USDLinkState:
    name: str
    has_visual: bool
    has_collision: bool


@dataclass(frozen=True)
class ConversionValidationReport:
    success: bool
    issues: list[str]
    total_links: int
    expected_visual_links: int
    expected_collision_links: int
    usd_links_found: int
    repaired_collision_links: int = 0


def log(message: str = "") -> None:
    print(message, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="URDF → USD 严格转换")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="无窗口模式 (默认开启)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf",
        help="输入 URDF 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "assets/robot/usd/uni_dingo_dual_arm.usd",
        help="输出 USD 文件路径",
    )
    return parser


def collect_urdf_link_expectations(urdf_path: Path) -> dict[str, URDFLinkExpectation]:
    root = ET.parse(str(urdf_path)).getroot()
    expectations: dict[str, URDFLinkExpectation] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        expectations[name] = URDFLinkExpectation(
            name=name,
            has_visual=link.find("visual") is not None,
            has_collision=link.find("collision") is not None,
        )
    return expectations


def validate_robot_conversion(
    urdf_expectations: dict[str, URDFLinkExpectation],
    usd_links: dict[str, USDLinkState],
    *,
    robot_prim_found: bool,
    default_prim_matches: bool,
    articulation_found: bool,
    repaired_collision_links: int = 0,
) -> ConversionValidationReport:
    issues: list[str] = []

    if not robot_prim_found:
        issues.append("Robot prim was not created in USD stage.")
    if not default_prim_matches:
        issues.append("Stage defaultPrim does not point to the robot prim.")
    if not articulation_found:
        issues.append("Articulation root/API was not found on the imported robot.")

    for name, expectation in urdf_expectations.items():
        usd_state = usd_links.get(name)
        if usd_state is None:
            issues.append(f"Link '{name}' is missing from the USD stage.")
            continue

        if expectation.has_visual and not usd_state.has_visual:
            issues.append(
                f"Link '{name}' has visual geometry in URDF but no visual geometry in USD."
            )
        if expectation.has_collision and not usd_state.has_collision:
            issues.append(
                f"Link '{name}' has collision geometry in URDF but no collision geometry in USD."
            )

    return ConversionValidationReport(
        success=not issues,
        issues=issues,
        total_links=len(urdf_expectations),
        expected_visual_links=sum(
            1 for expectation in urdf_expectations.values() if expectation.has_visual
        ),
        expected_collision_links=sum(
            1 for expectation in urdf_expectations.values() if expectation.has_collision
        ),
        usd_links_found=len(usd_links),
        repaired_collision_links=repaired_collision_links,
    )


def format_validation_report(report: ConversionValidationReport) -> str:
    lines = [
        "USD conversion validation summary:",
        f"  Total URDF links: {report.total_links}",
        f"  Expected visual links: {report.expected_visual_links}",
        f"  Expected collision links: {report.expected_collision_links}",
        f"  USD links found: {report.usd_links_found}",
        f"  Collision links repaired: {report.repaired_collision_links}",
    ]
    if report.success:
        lines.append("  Result: PASS")
    else:
        lines.append("  Result: FAIL")
        lines.append("  Issues:")
        lines.extend(f"    - {issue}" for issue in report.issues)
    return "\n".join(lines)


def add_collision_api(prim, UsdPhysics, PhysxSchema) -> None:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
    prim.GetAttribute("physxCollision:contactOffset").Set(0.005)
    prim.GetAttribute("physxCollision:restOffset").Set(0.001)


def classify_visual_copy_strategy(type_name: str) -> str | None:
    if type_name == "Mesh":
        return "mesh"
    if type_name in {"Cube", "Sphere", "Cylinder", "Capsule", "Cone"}:
        return "gprim"
    return None


def copy_xform_ops(src_prim, dst_prim, UsdGeom) -> None:
    src_xf = UsdGeom.Xformable(src_prim)
    dst_xf = UsdGeom.Xformable(dst_prim)
    for op in src_xf.GetOrderedXformOps():
        op_type = op.GetOpType()
        suffix = op.GetOpName().split(":")[-1] if ":" in op.GetOpName() else ""
        if op_type == UsdGeom.XformOp.TypeTranslate:
            dst_xf.AddTranslateOp(opSuffix=suffix).Set(op.Get())
        elif op_type == UsdGeom.XformOp.TypeOrient:
            dst_xf.AddOrientOp(opSuffix=suffix).Set(op.Get())
        elif op_type == UsdGeom.XformOp.TypeScale:
            dst_xf.AddScaleOp(opSuffix=suffix).Set(op.Get())
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            dst_xf.AddRotateXYZOp(opSuffix=suffix).Set(op.Get())


def copy_mesh_geometry(stage, source_mesh, target_path, UsdGeom):
    new_prim = stage.DefinePrim(target_path, "Mesh")
    src_mesh = UsdGeom.Mesh(source_mesh)
    dst_mesh = UsdGeom.Mesh(new_prim)

    points = src_mesh.GetPointsAttr().Get()
    face_vert_counts = src_mesh.GetFaceVertexCountsAttr().Get()
    face_vert_indices = src_mesh.GetFaceVertexIndicesAttr().Get()

    if points:
        dst_mesh.CreatePointsAttr(points)
    if face_vert_counts:
        dst_mesh.CreateFaceVertexCountsAttr(face_vert_counts)
    if face_vert_indices:
        dst_mesh.CreateFaceVertexIndicesAttr(face_vert_indices)

    copy_xform_ops(source_mesh, new_prim, UsdGeom)
    return new_prim


def copy_authored_attributes(source_prim, target_prim) -> None:
    for attr in source_prim.GetAuthoredAttributes():
        if attr.GetName() == "visibility":
            continue
        target_attr = target_prim.CreateAttribute(
            attr.GetName(),
            attr.GetTypeName(),
            attr.IsCustom(),
        )
        if attr.HasValue():
            target_attr.Set(attr.Get())


def copy_gprim_geometry(stage, source_prim, target_path):
    new_prim = stage.DefinePrim(target_path, source_prim.GetTypeName())
    copy_authored_attributes(source_prim, new_prim)
    return new_prim


def wrapper_type(type_name: str) -> bool:
    return type_name in {"Xform", "Scope", ""}


def immediate_valid_children(parent_prim) -> list:
    if not parent_prim or not parent_prim.IsValid():
        return []
    return [child for child in parent_prim.GetChildren() if child and child.IsValid()]


def shallow_children(parent_prim):
    for child in immediate_valid_children(parent_prim):
        yield child
        if wrapper_type(child.GetTypeName()):
            for grandchild in immediate_valid_children(child):
                yield grandchild


def resolve_geometry_container(stage, link_path: str, link_name: str, kind: str):
    local_prim = stage.GetPrimAtPath(f"{link_path}/{kind}")
    shared_prim = stage.GetPrimAtPath(f"/{kind}/{link_name}")

    if (
        local_prim
        and local_prim.IsValid()
        and immediate_valid_children(local_prim)
    ):
        return local_prim
    if (
        shared_prim
        and shared_prim.IsValid()
        and immediate_valid_children(shared_prim)
    ):
        return shared_prim
    if local_prim and local_prim.IsValid():
        return local_prim
    return shared_prim


def collision_target_path(collisions_prim, child_name: str) -> str:
    return f"{collisions_prim.GetPath()}/{child_name}"


def get_applied_references(prim) -> list:
    refs = prim.GetMetadata("references")
    if not refs:
        return []
    return list(refs.GetAppliedItems())


def remap_internal_reference_path(prim_path: str, path_map: dict[str, str]) -> str:
    for old_prefix in sorted(path_map, key=len, reverse=True):
        if prim_path == old_prefix:
            return path_map[old_prefix]
        if prim_path.startswith(f"{old_prefix}/"):
            suffix = prim_path[len(old_prefix) :]
            return f"{path_map[old_prefix]}{suffix}"
    return prim_path


def clone_reference(ref, Sdf, *, prim_path: str):
    return Sdf.Reference(
        assetPath=ref.assetPath,
        primPath=Sdf.Path(prim_path),
        layerOffset=ref.layerOffset,
        customData=ref.customData,
    )


def clear_invalid_internal_references(stage) -> int:
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


def rewrite_internal_references_in_subtree(
    stage,
    *,
    path_map: dict[str, str],
    Sdf,
    root_prefix: str,
) -> int:
    rewritten = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not (prim_path == root_prefix or prim_path.startswith(f"{root_prefix}/")):
            continue

        applied_refs = get_applied_references(prim)
        if not applied_refs:
            continue

        changed = False
        rewritten_refs = []
        for ref in applied_refs:
            if ref.assetPath or not ref.primPath:
                rewritten_refs.append(ref)
                continue

            old_target = str(ref.primPath)
            new_target = remap_internal_reference_path(old_target, path_map)
            if new_target != old_target:
                changed = True
                rewritten_refs.append(clone_reference(ref, Sdf, prim_path=new_target))
            else:
                rewritten_refs.append(ref)

        if not changed:
            continue

        prim.GetReferences().SetReferences(rewritten_refs)
        rewritten += 1

    return rewritten


def rewrite_relationship_targets_in_subtree(stage, *, path_map: dict[str, str], Sdf, root_prefix: str) -> int:
    rewritten = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not (prim_path == root_prefix or prim_path.startswith(f"{root_prefix}/")):
            continue

        changed = False
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            if not targets:
                continue

            rewritten_targets = []
            rel_changed = False
            for target in targets:
                new_target = remap_internal_reference_path(str(target), path_map)
                rewritten_target = Sdf.Path(new_target)
                rewritten_targets.append(rewritten_target)
                if rewritten_target != target:
                    rel_changed = True

            if rel_changed:
                relationship.SetTargets(rewritten_targets)
                changed = True

        if changed:
            rewritten += 1

    return rewritten


def package_imported_robot_asset(
    stage,
    robot_prim_path: str,
    *,
    UsdGeom,
    Sdf,
    wrapper_name: str = "RobotAsset",
) -> tuple[str, str]:
    wrapper_path = f"/{wrapper_name}"
    wrapper_prim = UsdGeom.Xform.Define(stage, wrapper_path).GetPrim()
    layer = stage.GetRootLayer()

    path_map: dict[str, str] = {}
    source_roots = [robot_prim_path, "/visuals", "/colliders", "/meshes"]
    for old_path in source_roots:
        source_prim = stage.GetPrimAtPath(old_path)
        if not source_prim or not source_prim.IsValid():
            continue

        new_path = f"{wrapper_path}{old_path}"
        Sdf.CopySpec(layer, Sdf.Path(old_path), layer, Sdf.Path(new_path))
        path_map[old_path] = new_path

    rewrite_internal_references_in_subtree(
        stage,
        path_map=path_map,
        Sdf=Sdf,
        root_prefix=wrapper_path,
    )
    rewrite_relationship_targets_in_subtree(
        stage,
        path_map=path_map,
        Sdf=Sdf,
        root_prefix=wrapper_path,
    )
    stage.SetDefaultPrim(wrapper_prim)
    return wrapper_path, f"{wrapper_path}{robot_prim_path}"


def inline_wrapper_collision_references(
    stage,
    *,
    wrapper_path: str,
    robot_prim_path: str,
    Sdf,
) -> tuple[int, bool]:
    """Inline packaged collision refs into the robot subtree and drop /colliders.

    The URDF importer authors each ``<link>/collisions`` prim as an internal
    reference to the shared ``/colliders/<link>`` scope. Keeping that shared
    scope in the final packaged asset also keeps its PhysicsCollisionGroup
    prims, which break Isaac Lab physics replication for multi-env cloning.

    This pass materializes the referenced collision subtree directly under each
    robot link's ``collisions`` prim so the packaged asset no longer depends on
    the shared ``/colliders`` scope.
    """

    wrapper_robot_path = f"{wrapper_path}{robot_prim_path}"
    wrapper_colliders_path = f"{wrapper_path}/colliders"
    layer = stage.GetRootLayer()

    inline_jobs: list[tuple[str, str]] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(f"{wrapper_robot_path}/"):
            continue
        if prim.GetName() != "collisions":
            continue

        for ref in get_applied_references(prim):
            if ref.assetPath or not ref.primPath:
                continue
            source_path = str(ref.primPath)
            if not source_path.startswith(f"{wrapper_colliders_path}/"):
                continue
            inline_jobs.append((prim_path, source_path))
            break

    for dest_path, source_path in inline_jobs:
        dest_prim = stage.GetPrimAtPath(dest_path)
        source_prim = stage.GetPrimAtPath(source_path)
        if not (dest_prim and dest_prim.IsValid() and source_prim and source_prim.IsValid()):
            continue

        dest_prim.GetReferences().ClearReferences()
        Sdf.CopySpec(layer, Sdf.Path(source_path), layer, Sdf.Path(dest_path))

    removed_shared_colliders = False
    shared_colliders_prim = stage.GetPrimAtPath(wrapper_colliders_path)
    if shared_colliders_prim and shared_colliders_prim.IsValid():
        removed_shared_colliders = stage.RemovePrim(wrapper_colliders_path)

    return len(inline_jobs), removed_shared_colliders


def authored_geometry_container(prim) -> bool:
    return bool(
        prim
        and prim.IsValid()
        and (prim.IsInstance() or prim.HasAuthoredReferences())
    )


def editable_authoring_prim(prim):
    if prim and prim.IsValid() and prim.IsInstanceProxy():
        source_prim = prim.GetPrimInPrototype()
        if source_prim and source_prim.IsValid():
            return source_prim
    return prim


def iter_supported_visual_prims(parent_prim):
    for child in immediate_valid_children(parent_prim):
        strategy = classify_visual_copy_strategy(child.GetTypeName())
        if strategy is not None:
            yield child, strategy
            continue

        if wrapper_type(child.GetTypeName()):
            for sub in immediate_valid_children(child):
                sub_strategy = classify_visual_copy_strategy(sub.GetTypeName())
                if sub_strategy is not None:
                    yield sub, sub_strategy


def should_repair_link(*, expected_has_collision: bool, usd_has_collision: bool) -> bool:
    return expected_has_collision and not usd_has_collision


def repair_empty_collisions(
    stage,
    robot_prim_path: str,
    urdf_expectations,
    UsdPhysics,
    PhysxSchema,
    UsdGeom,
) -> tuple[int, int]:
    """Repair links whose visuals exist but collisions subtree is empty."""
    repaired_links = 0
    added_collision_prims = 0

    for link_name, expectation in urdf_expectations.items():
        link_path = f"{robot_prim_path}/{link_name}"
        prim = stage.GetPrimAtPath(link_path)
        if not prim or not prim.IsValid():
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        visuals_prim = resolve_geometry_container(stage, link_path, link_name, "visuals")
        collisions_prim = resolve_geometry_container(
            stage,
            link_path,
            link_name,
            "collisions",
        )
        if not collisions_prim or not collisions_prim.IsValid():
            continue

        link_has_collision = has_collision_geometry(stage, link_path, UsdPhysics)
        if not should_repair_link(
            expected_has_collision=expectation.has_collision,
            usd_has_collision=link_has_collision,
        ):
            continue
        if authored_geometry_container(collisions_prim):
            continue

        editable_collisions_prim = editable_authoring_prim(collisions_prim)
        for child in shallow_children(editable_collisions_prim):
            add_collision_api(child, UsdPhysics, PhysxSchema)

        if not visuals_prim or not visuals_prim.IsValid():
            continue

        if not immediate_valid_children(visuals_prim):
            continue

        added_for_link = 0
        for visual_child, strategy in iter_supported_visual_prims(visuals_prim):
            try:
                child_name = visual_child.GetName()
                target_path = collision_target_path(editable_collisions_prim, child_name)
                if strategy == "mesh":
                    new_prim = copy_mesh_geometry(
                        stage,
                        visual_child,
                        target_path,
                        UsdGeom,
                    )
                    if not new_prim.HasAPI(
                        PhysxSchema.PhysxConvexDecompositionCollisionAPI
                    ):
                        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(new_prim)
                elif strategy == "gprim":
                    new_prim = copy_gprim_geometry(stage, visual_child, target_path)
                else:
                    continue

                add_collision_api(new_prim, UsdPhysics, PhysxSchema)
                UsdGeom.Imageable(new_prim).MakeInvisible()
                added_for_link += 1
            except Exception as exc:
                log(f"⚠️  跳过无法复制的 visual prim {visual_child.GetPath()}: {exc}")

        if added_for_link:
            repaired_links += 1
            added_collision_prims += added_for_link
            log(f"🔧 修复空碰撞子树: {link_name} (+{added_for_link} prims)")

    return repaired_links, added_collision_prims


def deinstance_all_prims(stage, Usd) -> int:
    """关闭所有 prim 的 instanceable 标记。

    URDF 导入器会自动把 visuals/collisions 节点标记为 instanceable=True，
    这导致子节点变成 InstanceProxy，PhysX 无法将碰撞体关联到 rigid body。
    """
    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            count += 1
    return count


def normalize_collision_purpose(stage, Usd, UsdGeom) -> int:
    """将碰撞几何的 purpose 从 guide 改为 default。

    URDF 导入器会把碰撞体容器的 purpose 设为 guide，导致 viewport 默认不渲染。
    改为 default 后，就可以通过 MakeVisible/MakeInvisible 正常控制碰撞体的显示。
    碰撞体初始状态设为 invisible，只在需要时显示。
    """
    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        purpose_attr = imageable.GetPurposeAttr()
        if purpose_attr and purpose_attr.Get() == "guide":
            purpose_attr.Set("default")
            count += 1
    return count


def configure_joint_drives(stage, robot_prim_path, Usd, UsdPhysics) -> int:
    drive_count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
            continue

        name = prim.GetName()
        prim_path = str(prim.GetPath())
        if robot_prim_path not in prim_path:
            continue

        is_wheel = any(wheel_name in name for wheel_name in WHEEL_JOINTS)
        is_arm = any(joint_name in name for joint_name in ARM_JOINTS)

        if is_wheel:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("velocity")
            drive.CreateDampingAttr(1500.0)
            drive.CreateStiffnessAttr(0.0)
            drive_count += 1
        elif is_arm:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(1000.0)
            drive.CreateDampingAttr(100.0)
            drive_count += 1

    return drive_count


def has_visual_geometry(stage, link_path, UsdGeom) -> bool:
    link_name = link_path.rsplit("/", 1)[-1]
    visuals_prim = resolve_geometry_container(stage, link_path, link_name, "visuals")
    if not visuals_prim or not visuals_prim.IsValid():
        return False
    for prim in immediate_valid_children(visuals_prim):
        if wrapper_type(prim.GetTypeName()):
            return True
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            return True
    return False


def has_collision_geometry(stage, link_path, UsdPhysics) -> bool:
    link_name = link_path.rsplit("/", 1)[-1]
    collisions_prim = resolve_geometry_container(stage, link_path, link_name, "collisions")
    if not collisions_prim or not collisions_prim.IsValid():
        return False
    if authored_geometry_container(collisions_prim):
        return True
    for prim in shallow_children(collisions_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if classify_visual_copy_strategy(prim.GetTypeName()) is not None:
            return True
    return False


def collect_usd_link_states(
    stage,
    robot_prim_path: str,
    expectations,
    UsdPhysics,
    UsdGeom,
) -> dict[str, USDLinkState]:
    usd_links: dict[str, USDLinkState] = {}
    for name in expectations:
        link_path = f"{robot_prim_path}/{name}"
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim or not link_prim.IsValid():
            continue
        usd_links[name] = USDLinkState(
            name=name,
            has_visual=has_visual_geometry(stage, link_path, UsdGeom),
            has_collision=has_collision_geometry(stage, link_path, UsdPhysics),
        )
    return usd_links


def articulation_exists(root, Usd, UsdPhysics, PhysxSchema) -> bool:
    return root.HasAPI(UsdPhysics.ArticulationRootAPI) or root.HasAPI(
        PhysxSchema.PhysxArticulationAPI
    )


def run_conversion(args: argparse.Namespace) -> int:
    urdf_path = args.urdf.resolve()
    output_path = args.output.resolve()

    if not urdf_path.exists():
        log(f"❌ URDF 不存在: {urdf_path}")
        return 1

    urdf_expectations = collect_urdf_link_expectations(urdf_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={"headless": args.headless, "width": 1280, "height": 720}
    )
    log("✅ Isaac Sim 已启动")

    try:
        import omni
        import omni.kit.commands
        from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        omni.usd.get_context().new_stage()
        simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        log(f"📂 导入 URDF: {urdf_path}")

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = True
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.collision_from_visuals = False
        # Avoid authoring asset-level collision groups that break PhysX
        # replication for Isaac Lab multi-env cloning. Runtime articulation
        # self-collision is still enabled below.
        import_config.self_collision = False
        import_config.distance_scale = 1.0
        import_config.create_physics_scene = False

        status, robot_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
        simulation_app.update()

        if not status or not robot_prim_path:
            log("❌ URDF 导入失败")
            return 1

        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim or not robot_prim.IsValid():
            log(f"❌ 未找到导入后的机器人 prim: {robot_prim_path}")
            return 1

        log(f"✅ URDF 已导入: {robot_prim_path}")

        if not robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
        attr = robot_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if not attr.IsValid():
            attr = robot_prim.CreateAttribute(
                "physxArticulation:enabledSelfCollisions",
                Sdf.ValueTypeNames.Bool,
                True,
            )
        # Keep runtime self-collision enabled on the articulation root.
        attr.Set(True)
        log("✅ 已启用 Articulation 自碰撞检测")

        drive_count = configure_joint_drives(stage, robot_prim_path, Usd, UsdPhysics)
        log(f"✅ 已配置 {drive_count} 个关节驱动")

        repaired_links, repaired_prims = repair_empty_collisions(
            stage,
            robot_prim_path,
            urdf_expectations,
            UsdPhysics,
            PhysxSchema,
            UsdGeom,
        )
        log(
            f"✅ 碰撞后处理完成: 修复 {repaired_links} 个 links, 新增 {repaired_prims} 个碰撞 prims"
        )

        cleared_refs = clear_invalid_internal_references(stage)
        log(f"✅ 已清理 {cleared_refs} 个悬空内部引用")

        asset_root_path, exported_robot_path = package_imported_robot_asset(
            stage,
            robot_prim_path,
            UsdGeom=UsdGeom,
            Sdf=Sdf,
        )
        asset_root = stage.GetPrimAtPath(asset_root_path)
        exported_robot_prim = stage.GetPrimAtPath(exported_robot_path)
        log(f"✅ 已打包共享几何根到默认导出根: {asset_root_path}")

        inlined_collision_refs, removed_shared_colliders = inline_wrapper_collision_references(
            stage,
            wrapper_path=asset_root_path,
            robot_prim_path=robot_prim_path,
            Sdf=Sdf,
        )
        log(
            "✅ 已内联机器人碰撞子树引用: "
            f"{inlined_collision_refs} 个 collisions prim"
        )
        if removed_shared_colliders:
            log("✅ 已移除包装资产中的共享 /colliders 子树")

        cleared_packaged_refs = clear_invalid_internal_references(stage)
        log(f"✅ 已清理包装资产中的 {cleared_packaged_refs} 个悬空内部引用")

        deinstanced = deinstance_all_prims(stage, Usd)
        log(f"✅ 已关闭 {deinstanced} 个 prim 的 instanceable 标记（修复 PhysX 碰撞识别）")

        normalized = normalize_collision_purpose(stage, Usd, UsdGeom)
        log(f"✅ 已将 {normalized} 个 purpose=guide 改为 default（修复碰撞体可视化）")

        simulation_app.update()

        usd_links = collect_usd_link_states(
            stage,
            exported_robot_path,
            urdf_expectations,
            UsdPhysics,
            UsdGeom,
        )
        report = validate_robot_conversion(
            urdf_expectations=urdf_expectations,
            usd_links=usd_links,
            robot_prim_found=exported_robot_prim.IsValid(),
            default_prim_matches=stage.GetDefaultPrim() == asset_root,
            articulation_found=articulation_exists(
                exported_robot_prim, Usd, UsdPhysics, PhysxSchema
            ),
            repaired_collision_links=repaired_links,
        )
        log("\n" + format_validation_report(report))
        if not report.success:
            log("❌ URDF → USD 严格校验失败，已停止导出。")
            return 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(output_path))
        log(f"\n✅ USD 已保存: {output_path}")
        log(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
        return 0
    finally:
        simulation_app.close()
        log("👋 转换流程结束")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_conversion(args)


if __name__ == "__main__":
    raise SystemExit(main())
