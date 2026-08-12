from scripts.analysis.diagnose_f0_trajectories import (
    run_length_encode,
    summarize_trace,
)


def test_run_length_encode_uses_one_based_inclusive_bounds():
    assert run_length_encode(["a", "a", "b", "a"]) == [
        {
            "start_step": 1,
            "end_step": 2,
            "count": 2,
            "value": "a",
        },
        {
            "start_step": 3,
            "end_step": 3,
            "count": 1,
            "value": "b",
        },
        {
            "start_step": 4,
            "end_step": 4,
            "count": 1,
            "value": "a",
        },
    ]


def _step(index, action, *, effect, do_available, loaded, observation_digest):
    effect_possible = [True] * 8
    effect_possible[6] = do_available
    return {
        "step": index,
        "action": action,
        "action_name": (
            "DO" if action == 6 else "DO_NOTHING" if action == 7 else "FORWARD"
        ),
        "action_had_effect": effect,
        "counterfactual_effect_mask": effect_possible,
        "do_logit_rank": 2,
        "reward": 0.25,
        "reward_components": {
            "absolute": 0.5,
            "dig": 0.5,
            "dump_purity": 1.0,
            "dump_volume": 0.5,
        },
        "pre": {
            "dug_required_volume": 0.0,
            "accepted_dump_volume": 0.0,
            "illegal_dump_volume": 0.0,
            "loaded_volume": float(loaded),
        },
        "post": {
            "dug_required_volume": float(effect),
            "accepted_dump_volume": 0.0,
            "illegal_dump_volume": 0.0,
            "loaded_volume": float(loaded + effect),
        },
        "observation_digest": observation_digest,
        "state_digest": f"state-{index}",
        "counterfactual_do": (
            {
                "reward_advantage": 1.0,
                "action_had_effect": True,
            }
            if do_available and action != 6
            else None
        ),
    }


def test_summarize_trace_exposes_stall_and_unchosen_do_opportunity():
    steps = [
        _step(
            1,
            7,
            effect=False,
            do_available=True,
            loaded=0,
            observation_digest="same",
        ),
        _step(
            2,
            7,
            effect=False,
            do_available=True,
            loaded=0,
            observation_digest="same",
        ),
        _step(
            3,
            6,
            effect=True,
            do_available=True,
            loaded=0,
            observation_digest="new",
        ),
    ]
    summary = summarize_trace(steps)
    assert summary["action_counts"] == {"DO": 1, "DO_NOTHING": 2}
    assert summary["no_effect_steps"] == 2
    assert summary["unchosen_do_opportunities"] == 2
    assert summary["counterfactual_do"]["positive_immediate_advantage"] == 2
    assert summary["maximum_policy_observation_repetitions"] == 2
    assert len(summary["terrain_or_load_events"]) == 1
