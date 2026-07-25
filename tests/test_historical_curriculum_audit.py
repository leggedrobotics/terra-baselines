from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.analysis.audit_historical_curriculum import (
    _collapse_with_boundary_flow,
    _historical_terminal_reward,
    completion_observer,
    parse_labelled_values,
    summarize_rows,
    terminal_reward_reconstruction_is_valid,
    verify_unique_training_manifest,
)


def test_parse_labelled_values_rejects_ambiguous_inputs():
    assert parse_labelled_values(
        ["flat_u1000=/tmp/checkpoint.pkl"], option="--checkpoint"
    ) == {"flat_u1000": "/tmp/checkpoint.pkl"}
    with pytest.raises(ValueError, match="expects LABEL=VALUE"):
        parse_labelled_values(["checkpoint.pkl"], option="--checkpoint")
    with pytest.raises(ValueError, match="repeats label"):
        parse_labelled_values(["flat=a", "flat=b"], option="--checkpoint")


def test_unique_training_manifest_rejects_repeated_identity():
    unique = [
        {"source_id": "a", "map_id": "map-a"},
        {"source_id": "b", "map_id": "map-b"},
    ]
    assert verify_unique_training_manifest(unique) == {
        "passed": True,
        "slots": 2,
        "unique_source_ids": 2,
        "unique_map_ids": 2,
    }
    with pytest.raises(RuntimeError, match="not identity-unique"):
        verify_unique_training_manifest(
            unique + [{"source_id": "a", "map_id": "map-c"}]
        )


def test_completion_observer_separates_visible_and_buffer_soil():
    target = jnp.zeros((1, 5, 5), dtype=jnp.int8)
    target = target.at[0, 2, 2].set(1)
    action = jnp.zeros_like(target).at[0, 2, 3].set(1)
    state = SimpleNamespace(
        world=SimpleNamespace(
            target_map=SimpleNamespace(map=target),
            action_map=SimpleNamespace(map=action),
            padding_mask=SimpleNamespace(map=jnp.zeros_like(target)),
        ),
        agent=SimpleNamespace(
            agent_states=(SimpleNamespace(loaded=jnp.zeros((1, 1), dtype=jnp.int8)),)
        ),
    )
    observed = completion_observer(SimpleNamespace(state=state))
    np.testing.assert_allclose(observed["exact_completion"], [0.0])
    np.testing.assert_allclose(observed["accepted_buffer_completion"], [1.0])
    np.testing.assert_allclose(observed["buffer_only_positive_volume"], [1.0])


def test_historical_terminal_reward_uses_code_default_when_field_is_absent():
    state = SimpleNamespace(
        env_cfg=SimpleNamespace(
            rewards=SimpleNamespace(
                terminal=jnp.array([100.0], dtype=jnp.float32),
                normalizer=jnp.array([10.0], dtype=jnp.float32),
            )
        ),
        agent=SimpleNamespace(agent_active=jnp.array([[1]], dtype=jnp.int32)),
    )
    reward, full_terminal = _historical_terminal_reward(
        state,
        jnp.array([0.8], dtype=jnp.float32),
        jnp.array([True]),
        jnp.array([True]),
    )
    np.testing.assert_allclose(reward, [5.0])
    np.testing.assert_allclose(full_terminal, [20.0])


def test_terminal_reward_reconstruction_tolerance_is_float32_scale_only():
    assert terminal_reward_reconstruction_is_valid(1.430511474609375e-6)
    assert not terminal_reward_reconstruction_is_valid(1.1e-5)


class _AllValidSoilState:
    @staticmethod
    def _expand_mask_for_soil_mechanics(mask):
        return mask


def test_boundary_flow_observer_conserves_mass_and_counts_outward_flow():
    action = jnp.zeros((5, 5), dtype=jnp.int8).at[2, 2].set(4)
    affected = jnp.ones((5, 5), dtype=jnp.bool_)
    exact = jnp.zeros((5, 5), dtype=jnp.bool_).at[2, 2].set(True)
    collapsed, outward, inward = _collapse_with_boundary_flow(
        _AllValidSoilState(), action, affected, exact
    )
    assert int(collapsed.astype(jnp.int32).sum()) == int(action.astype(jnp.int32).sum())
    assert int(outward) > 0
    assert int(inward) == 0


def _audit_row(
    *,
    success,
    reason,
    exact,
    delta,
    reward_delta,
    reward_scale=1.0,
):
    return {
        "success": success,
        "termination_reason": reason,
        "task_done_with_inexact_visible_completion": success and exact < 1.0,
        "exact_completion": exact,
        "accepted_buffer_completion": exact + delta,
        "completion_delta": delta,
        "current_terminal_reward": 0.0,
        "counterfactual_terminal_reward": reward_delta,
        "terminal_reward_delta": reward_delta,
        "full_terminal_reward_scale": reward_scale,
        "buffer_only_positive_volume": 1.0 if delta else 0.0,
        "illegal_positive_volume": 0.0,
        "policy_entropy": 0.5,
        "action_logit_margin": 1.0,
        "sampled_argmax_disagreement_rate": 0.0,
        "potential_veto_attempts": 0,
        "dump_attempts": 1,
        "executed_dump_attempts": 1,
        "outward_boundary_relaxation_volume": 0,
        "inward_boundary_relaxation_volume": 0,
        "integrity_failure": False,
        "maximum_mass_residual": 0,
    }


def test_semantic_materiality_uses_successes_and_top_quartile_timeouts():
    rows = [
        _audit_row(
            success=True,
            reason="task_done",
            exact=0.9,
            delta=0.1,
            reward_delta=0.0,
        ),
        *[
            _audit_row(
                success=False,
                reason="timeout",
                exact=exact,
                delta=0.0,
                reward_delta=0.0,
            )
            for exact in (0.8, 0.7, 0.6, 0.5)
        ],
    ]
    summary = summarize_rows(rows)
    assert summary["task_done_with_inexact_visible_completion"] == 1
    assert summary["semantic_materiality"]["success_fraction"] == 1.0
    assert summary["semantic_materiality"]["top_quartile_timeouts_denominator"] == 1
    assert summary["semantic_materiality"]["material_contributor"]
