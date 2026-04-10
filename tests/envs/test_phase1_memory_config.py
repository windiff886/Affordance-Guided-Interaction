from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = PROJECT_ROOT / "src/affordance_guided_interaction/envs/door_push_env_cfg.py"


def test_door_push_env_cfg_enables_phase1_memory_optimizations():
    text = CFG_PATH.read_text(encoding="utf-8")

    assert "create_stage_in_memory=True" in text
    assert "replicate_physics=True" in text
    assert "clone_in_fabric=True" in text
