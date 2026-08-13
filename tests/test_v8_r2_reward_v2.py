from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from terra.actions import TrackedAction
from terra.config import (
    REWARD_V2_POTENTIAL_GAMMA,
    BatchConfig,
    EnvConfig,
    MapsDimsConfig,
    RewardStage,
)
from terra.env import TerraEnv, TerraEnvBatch
from terra.state import State

import train_mixed
from scripts import materialize_v8_r2_distance_bank as materialize


def reward_v2_state() -> State:
    shape = (64, 64)
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=shape[0])
    )
    base = EnvConfig()
    updated = batch_env.update_env_cfgs(
        base._replace(
            agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
        )
    )
    env_config = base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=shape[0]),
        max_steps_in_episode=450,
        agent_types=(0,),
        action_types=(0,),
        reward_stage=RewardStage.REWARD_V2,
    )
    target = np.zeros(shape, dtype=np.int8)
    target[20, 20:24] = -1
    target[40, 40:48] = 1
    distance = np.ones(shape, dtype=np.float32)
    distance[target > 0] = 0.0
    return State.new(
        jax.random.PRNGKey(20260810),
        env_config,
        target,
        np.zeros(shape, dtype=np.int8),
        -97.0 * np.ones((4, 3), dtype=np.float32),
        np.int32(-1),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.int8),
        distance_map_override=distance,
    )


def test_reward_protocols_fail_closed():
    common = {
        "gamma": 0.9984,
        "distance_sidecar_sha256": "a" * 64,
    }
    treatment = train_mixed._r2_protocol_receipt(
        SimpleNamespace(
            **common,
            reward_stage="reward_v2",
            distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        )
    )
    assert treatment["reward_protocol_id"] == "material_potential_v2"
    assert treatment["distance_artifact_kind"] == (
        "canonical_distance_sidecar_dataset_json"
    )
    assert treatment["constants"]["potential_gamma"] == 0.9984

    scratch = train_mixed._r2_protocol_receipt(
        SimpleNamespace(
            gamma=0.9984,
            distance_sidecar_sha256="b" * 64,
            reward_stage="reward_v2",
            distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        )
    )
    assert scratch["reward_protocol_id"] == "material_potential_v2"
    assert scratch["distance_sidecar_sha256"] == "b" * 64
    assert (
        train_mixed._r2_protocol_receipt(
            SimpleNamespace(
                reward_stage="dense_skill",
            )
        )
        is None
    )


def test_restored_environment_uses_selected_arm_reward_stage_only():
    restored = train_mixed._strip_checkpoint_env_axis(
        EnvConfig()._replace(relocation_progress_mult=7.25),
        num_envs_per_device=512,
    )
    control = train_mixed._overlay_env_reward_stage(restored, "dense_skill")
    treatment = train_mixed._overlay_env_reward_stage(restored, "reward_v2", 1)

    assert control.reward_stage == RewardStage.DENSE_SKILL
    assert treatment.reward_stage == RewardStage.REWARD_V2
    # Both reward selectors are command-line treatments, so both are overlaid.
    assert int(control.reward_v2_timing_variant) == 0
    assert int(treatment.reward_v2_timing_variant) == 1
    for field in restored._fields:
        if field not in ("reward_stage", "reward_v2_timing_variant"):
            assert getattr(control, field) is getattr(restored, field)
            assert getattr(treatment, field) is getattr(restored, field)


def test_r2_resume_requires_identical_protocol_and_optimizer_clock():
    receipt = {"schema": "terra_v8_r2_reward_protocol_v1", "value": 1}
    config = SimpleNamespace(update_epochs=2, num_minibatches=32)
    checkpoint = {
        "r2_protocol_receipt": receipt,
        "optimizer_state": {},
        "train_state_step": np.asarray(64_000),
        "next_update": 1_000,
    }
    train_mixed._validate_r2_resume_checkpoint(checkpoint, receipt, config)

    with pytest.raises(ValueError, match="protocol receipt"):
        train_mixed._validate_r2_resume_checkpoint(
            {**checkpoint, "r2_protocol_receipt": {"value": 2}}, receipt, config
        )
    with pytest.raises(ValueError, match="optimizer clock"):
        train_mixed._validate_r2_resume_checkpoint(
            {**checkpoint, "train_state_step": np.asarray(63_999)}, receipt, config
        )


def test_a_pre_v21_receipt_resumes_only_into_baseline_timing():
    """Checkpoints predating the v2.1 selector are baseline timing, and only that.

    The v6.1 u14000 continuation resumes such a checkpoint: its receipt has no
    timing fields at all, so the guard fills them at their baseline values
    rather than reading the absence as a protocol change.
    """
    gamma = float(REWARD_V2_POTENTIAL_GAMMA)
    legacy = {
        "schema": "terra_v8_r2_reward_protocol_v1",
        "reward_stage": "reward_v2",
        "constants": {"potential_gamma": gamma, "step_cost_total": 1.0},
    }
    baseline = {
        **legacy,
        "reward_v2_timing": "baseline",
        "reward_v2_timing_variant": 0,
        "constants": {**legacy["constants"], "shaping_gamma": gamma},
    }
    v21 = {
        **legacy,
        "reward_v2_timing": "gamma1_stepcost_3.6",
        "reward_v2_timing_variant": 1,
        "constants": {
            "potential_gamma": gamma,
            "step_cost_total": 3.6,
            "shaping_gamma": 1.0,
        },
    }
    config = SimpleNamespace(update_epochs=2, num_minibatches=32)
    checkpoint = {
        "r2_protocol_receipt": legacy,
        "optimizer_state": {},
        "train_state_step": np.asarray(896_000),
        "next_update": 14_000,
    }
    train_mixed._validate_r2_resume_checkpoint(checkpoint, baseline, config)
    with pytest.raises(ValueError, match="protocol receipt"):
        train_mixed._validate_r2_resume_checkpoint(checkpoint, v21, config)
    # A receipt that already declares its timing is compared as written.
    with pytest.raises(ValueError, match="protocol receipt"):
        train_mixed._validate_r2_resume_checkpoint(
            {**checkpoint, "r2_protocol_receipt": v21}, baseline, config
        )


def test_terra_reset_reward_components_match_step_inside_jax_scan():
    assert "dummy_components" not in Path(train_mixed.__file__).read_text()
    state = reward_v2_state()
    reset_components = TerraEnv._zero_reward_components(state)
    _, step_components = state._get_reward(
        state._replace(env_steps=1), TrackedAction.do_nothing()
    )
    r2_keys = {
        "reward_v2_q",
        "reward_v2_q_next",
        "reward_v2_p",
        "reward_v2_p_next",
        "reward_v2_phi",
        "reward_v2_phi_next",
        "reward_v2_material_work",
        "reward_v2_h_reset",
        "reward_v2_carry_work",
        "reward_v2_shaping",
        "reward_v2_success",
        "reward_v2_horizon_failure",
        "reward_v2_step",
        "reward_v2_valid",
    }
    assert r2_keys <= reset_components.keys()
    assert jax.tree_util.tree_structure(reset_components) == (
        jax.tree_util.tree_structure(step_components)
    )

    def scan_components(initial):
        def body(_, index):
            components = jax.lax.cond(
                index == 0,
                lambda: reset_components,
                lambda: step_components,
            )
            return components, components["reward_v2_phi"]

        return jax.lax.scan(body, initial, jnp.arange(2))[0]

    scanned = jax.jit(scan_components)(reset_components)
    assert jax.tree_util.tree_structure(scanned) == (
        jax.tree_util.tree_structure(step_components)
    )


def test_materializer_pins_authoritative_static_v2_only():
    assert materialize.EXPECTED_SIDECAR_DATASET_SHA256 == (
        "f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980"
    )
    assert materialize.EXPECTED_SIDECAR_ROWS_SHA256 == (
        "b6bcae37a1750f7d78c1645af408320c50b0fa28d38098ff91e9cecbfec251a8"
    )
