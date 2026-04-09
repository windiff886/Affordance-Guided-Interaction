from scripts.train import _iter_reward_scalar_tags


def test_iter_reward_scalar_tags_keeps_only_summary_metrics_under_reward() -> None:
    collect_stats = {
        "reward/total": 1.0,
        "reward/task": 0.8,
        "reward/stab_left": 0.1,
        "reward/stab_right": 0.2,
        "reward/safe": 0.3,
        "reward/task/delta": 0.5,
        "reward/task/open_bonus": 0.3,
        "reward/stab_left/zero_acc": 0.1,
        "reward/safe/joint_vel": 0.05,
        "collect/mean_reward": 1.0,
    }

    routed = dict(_iter_reward_scalar_tags(collect_stats))

    assert routed == {
        "reward/total": 1.0,
        "reward/task": 0.8,
        "reward/stab_left": 0.1,
        "reward/stab_right": 0.2,
        "reward/safe": 0.3,
        "reward_terms/task/delta": 0.5,
        "reward_terms/task/open_bonus": 0.3,
        "reward_terms/stab_left/zero_acc": 0.1,
        "reward_terms/safe/joint_vel": 0.05,
    }


def test_iter_reward_scalar_tags_ignores_non_reward_metrics() -> None:
    routed = list(_iter_reward_scalar_tags({
        "collect/mean_reward": -0.2,
        "train/fps": 1200.0,
    }))

    assert routed == []
