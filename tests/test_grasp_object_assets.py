from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cup_asset_pinch_frame_is_offset_from_root_for_side_pickup() -> None:
    cup_asset_path = PROJECT_ROOT / "assets" / "grasp_objects" / "cup" / "carry_cup.usda"
    asset_text = cup_asset_path.read_text(encoding="utf-8")

    match = re.search(
        r'def Xform "Pinch"\s*\{\s*double3 xformOp:translate = \(([^)]+)\)',
        asset_text,
        re.DOTALL,
    )

    assert match is not None
    components = tuple(float(component.strip()) for component in match.group(1).split(","))

    assert components != (0.0, 0.0, 0.0)
    assert components[0] == 0.0
    assert components[1] == 0.0
    assert components[2] > 0.0


def test_cup_asset_binds_high_friction_physics_material_to_body_collider() -> None:
    cup_asset_path = PROJECT_ROOT / "assets" / "grasp_objects" / "cup" / "carry_cup.usda"
    asset_text = cup_asset_path.read_text(encoding="utf-8")

    assert 'def Material "GripPhysicsMaterial"' in asset_text
    assert 'prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]' in asset_text
    assert "float physics:staticFriction" in asset_text
    assert "float physics:dynamicFriction" in asset_text
    assert 'uniform token physxMaterial:frictionCombineMode = "max"' in asset_text
    assert 'rel material:binding:physics = </CarryCup/GripPhysicsMaterial>' in asset_text


def test_cup_asset_uses_training_dimensions_for_gripper_fit() -> None:
    cup_asset_path = PROJECT_ROOT / "assets" / "grasp_objects" / "cup" / "carry_cup.usda"
    asset_text = cup_asset_path.read_text(encoding="utf-8")

    match = re.search(
        r'def Cylinder "OuterBody".*?float3 xformOp:scale = \(([^)]+)\)',
        asset_text,
        re.DOTALL,
    )

    assert match is not None
    scale = tuple(float(component.strip()) for component in match.group(1).split(","))
    assert scale == (0.04, 0.04, 0.06)
