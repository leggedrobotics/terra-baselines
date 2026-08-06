from scripts import v8_10m_stage_b_selection as selection


def _identity_record(arm: str, update: int, *, exact: int, macro: float) -> dict:
    return {
        "checkpoint": (
            "/cluster/scratch/campaign/screen/capability/s1/"
            f"{arm}/checkpoints/model_update_{update:06d}.pkl"
        ),
        "checkpoint_sha256": f"{update:064x}",
        "checkpoint_update": update,
        "summary": {
            "overall": {"successes": exact, "episodes": 720},
            "by_primary_cell": {
                "v7-fnd-slab-adjacent": {
                    "successes": min(exact, 16),
                    "episodes": 16,
                },
                "fnd-slab-apron-d16": {
                    "successes": min(exact, 16),
                    "episodes": 16,
                },
            },
            "graded": {
                "macro_completion": macro,
                "by_primary_cell": {
                    "v7-fnd-slab-adjacent": {"mean": macro},
                    "fnd-slab-apron-d16": {"mean": macro / 2},
                },
            },
        },
    }


def _capability_record(source: dict, count: int) -> dict:
    return {
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_update": source["checkpoint_update"],
        "summary": {
            "by_primary_cell": {
                condition_id: {"successes": count, "episodes": 16}
                for condition_id in selection.stage_gate.CAPABILITY_IDS
            }
        },
    }


def test_select_parent_uses_promotion_and_requires_capability_retention():
    arm = selection.ARMS[0]
    promotion = [
        _identity_record(arm, update, exact=exact, macro=macro)
        for update, exact, macro in (
            (1000, 100, 0.4),
            (2000, 120, 0.5),
            (3000, 200, 0.8),
            (4000, 110, 0.6),
        )
    ]
    development = [
        _identity_record(arm, update, exact=300 - index, macro=0.9 - index / 10)
        for index, update in enumerate(selection.UPDATES)
    ]
    capability_promotion = [_capability_record(record, 16) for record in promotion]
    capability_development = [
        _capability_record(record, 11 if index == 2 else 12)
        for index, record in enumerate(promotion)
    ]

    result = selection.select_parent(
        arm,
        promotion,
        development,
        capability_promotion,
        capability_development,
        ("v7-fnd-slab-adjacent",),
    )

    assert result["update"] == 2000
    assert result["promotion"]["exact_successes"] == 120
    assert result["development"]["exact_successes"] == 299
    assert result["weakest_development_conditions"][0]["curriculum_stage"] == "full"
