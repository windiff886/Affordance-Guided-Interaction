from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/envs/door_push_env.py"
CFG_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/envs/door_push_env_cfg.py"
TRAIN_PATH = PROJECT_ROOT / "scripts/train.py"
REWARD_YAML_PATH = PROJECT_ROOT / "configs/reward/default.yaml"
REWARD_DOC_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/envs/Reward.md"
ENV_DOC_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/envs/README.md"
TENSORBOARD_DOC_PATH = PROJECT_ROOT / "docs/tensorboard_guide.md"
PIPELINE_DOC_PATH = PROJECT_ROOT / "docs/training_pipeline_detailed.md"


def test_runtime_no_longer_contains_self_collision_reward_path():
    text = ENV_PATH.read_text(encoding="utf-8")

    assert "_self_collision_groups" not in text
    assert "_compute_batch_self_collision" not in text
    assert 'reward_info["safe/self_collision"]' not in text
    assert "r_safe_self_collision" not in text


def test_config_and_injection_remove_self_collision_reward_dependencies():
    cfg_text = CFG_PATH.read_text(encoding="utf-8")
    train_text = TRAIN_PATH.read_text(encoding="utf-8")
    reward_yaml_text = REWARD_YAML_PATH.read_text(encoding="utf-8")

    assert "enabled_self_collisions=True" in cfg_text
    assert "ContactSensorCfg" not in cfg_text
    assert "activate_contact_sensors=True" not in cfg_text
    assert "contact_sensor:" not in cfg_text
    assert "rew_beta_self" not in cfg_text

    assert '"beta_self": "rew_beta_self"' not in train_text
    assert "beta_self:" not in reward_yaml_text


def test_docs_no_longer_describe_self_collision_as_active_reward_term():
    reward_doc = REWARD_DOC_PATH.read_text(encoding="utf-8")
    env_doc = ENV_DOC_PATH.read_text(encoding="utf-8")
    tensorboard_doc = TENSORBOARD_DOC_PATH.read_text(encoding="utf-8")
    pipeline_doc = PIPELINE_DOC_PATH.read_text(encoding="utf-8")

    assert "### 6.1 自碰撞惩罚" not in reward_doc
    assert "_compute_batch_self_collision()" not in reward_doc
    assert "safety.beta_self" not in reward_doc

    assert "ContactSensorCfg" not in env_doc
    assert "rew_beta_self" not in env_doc

    assert "self_collision" not in tensorboard_doc

    assert "contact_sensor: ContactSensorCfg" not in pipeline_doc
    assert "自碰撞分组" not in pipeline_doc
    assert "β_text{self}" not in pipeline_doc
