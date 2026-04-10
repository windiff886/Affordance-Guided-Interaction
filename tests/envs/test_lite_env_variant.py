from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/envs/door_push_env_cfg.py"
TRAIN_PATH = PROJECT_ROOT / "scripts/train.py"
RUNTIME_CFG_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/utils/train_runtime_config.py"
TRAINING_YAML_PATH = PROJECT_ROOT / "configs/training/default.yaml"


def test_lite_env_config_classes_exist_and_encode_the_first_phase_boundary():
    text = CFG_PATH.read_text(encoding="utf-8")

    assert "class DoorPushLiteSceneCfg" in text
    assert "class DoorPushLiteEnvCfg" in text
    assert "self.room = None" in text
    assert "fix_root_link = True" in text
    assert 'self.robot.init_state.joint_pos.pop(".*wheel", None)' in text
    assert 'self.robot.init_state.joint_pos.pop("pan_tilt_.*", None)' in text
    assert "env_spacing=4.0" in text
    assert "base_radius_range = (0.74, 0.74)" in text
    assert "base_sector_half_angle_deg = 0.0" in text
    assert "base_yaw_delta_deg = 0.0" in text


def test_train_runtime_and_entrypoint_can_select_lite_variant():
    runtime_text = RUNTIME_CFG_PATH.read_text(encoding="utf-8")
    train_text = TRAIN_PATH.read_text(encoding="utf-8")
    training_yaml_text = TRAINING_YAML_PATH.read_text(encoding="utf-8")

    assert "env_variant" in runtime_text
    assert "variant: str = \"full\"" in train_text
    assert "DoorPushLiteEnvCfg" in train_text
    assert 'env_variant = _normalize_optional_str(t_cfg.get("env_variant")) or "full"' in runtime_text
    assert 'env_variant:' in training_yaml_text
