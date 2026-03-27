from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_reward_manager_combines_named_terms() -> None:
    from affordance_guided_interaction.rewards.reward_manager import RewardManager

    reward_info = RewardManager().combine(
        task_progress=1.0,
        carry_stability=0.2,
        effective_contact=0.1,
        invalid_collision=0.3,
        safety_penalty=0.4,
    )

    assert reward_info["total_reward"] == 0.6
    assert reward_info["carry_stability"] == 0.2
