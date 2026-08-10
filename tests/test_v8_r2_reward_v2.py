from pathlib import Path
import stat
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze
from terra.actions import TrackedAction
from terra.config import BatchConfig, EnvConfig, MapsDimsConfig, RewardStage
from terra.env import TerraEnv, TerraEnvBatch
from terra.state import State

import train_mixed
from scripts import materialize_v8_r2_distance_bank as materialize
from scripts.prepare_v8_r2_fork import CARRY_KERNEL, expand_carry_input


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


def test_carry_expansion_adds_one_zero_input_and_preserves_parent_weights():
    source_kernel = jnp.arange(32, dtype=jnp.float32).reshape(2, 16)
    target_kernel = jnp.full((3, 16), -7.0, dtype=jnp.float32)

    def tree(kernel):
        return freeze(
            {
                "params": {
                    "agent_state_net": {
                        "mlp_continuous": {"layers_0": {"kernel": kernel}}
                    }
                }
            }
        )

    expanded, receipt = expand_carry_input(tree(source_kernel), tree(target_kernel))
    kernel = expanded
    for key in CARRY_KERNEL:
        kernel = kernel[key]
    np.testing.assert_array_equal(np.asarray(kernel[:2]), np.asarray(source_kernel))
    np.testing.assert_array_equal(np.asarray(kernel[2]), np.zeros(16, np.float32))
    assert receipt["path"] == "/".join(CARRY_KERNEL)


def test_historical_prepared_fork_and_reward_protocols_fail_closed():
    checkpoint = {
        "next_update": 20_000,
        "train_config": {
            "carry_work_observation": True,
            "config_name": "G-V8-CONTINUOUS-V2",
            "pooled_sampler": {"rule": "continuous_banded_v2"},
            "accepted_bank": {"sampler_profile": "continuous_banded_v2"},
        },
        "pooled_sampler_state": {"settings": {"rule": "continuous_banded_v2"}},
        "r2_prepared_fork": {
            "schema": train_mixed.R2_PREPARED_FORK_SCHEMA,
            "source_checkpoint_sha256": train_mixed.R2_PARENT_SHA256,
        },
    }
    train_mixed._validate_r2_prepared_fork(checkpoint)

    common = {
        "prepared_fork_from": "prepared.pkl",
        "gamma": 0.9984,
        "distance_sidecar_sha256": "a" * 64,
    }
    control = train_mixed._r2_protocol_receipt(
        SimpleNamespace(
            **common,
            reward_stage="dense_skill",
            distance_protocol_id="legacy_dataset_distance",
        )
    )
    treatment = train_mixed._r2_protocol_receipt(
        SimpleNamespace(
            **common,
            reward_stage="reward_v2",
            distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        )
    )
    assert control["reward_protocol_id"] == "dense_skill_legacy_relocation_v1"
    assert control["distance_artifact_kind"] == "accepted_bank_dataset_json"
    assert treatment["reward_protocol_id"] == "material_potential_v2"
    assert treatment["distance_artifact_kind"] == (
        "canonical_distance_sidecar_dataset_json"
    )
    assert treatment["constants"]["potential_gamma"] == 0.9984

    scratch = train_mixed._r2_protocol_receipt(
        SimpleNamespace(
            prepared_fork_from=None,
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
                prepared_fork_from=None,
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
    treatment = train_mixed._overlay_env_reward_stage(restored, "reward_v2")

    assert control.reward_stage == RewardStage.DENSE_SKILL
    assert treatment.reward_stage == RewardStage.REWARD_V2
    for field in restored._fields:
        if field != "reward_stage":
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
    submit = (
        Path(__file__).parents[1] / "scripts/euler_v8_r2_reward_v2/submit.sh"
    ).read_text()
    assert "static_v2/receipt_manifest.json" in submit
    assert "/static/receipt_manifest.json" not in submit


def test_launcher_is_one_from_scratch_reward_v2_system_run():
    root = Path(__file__).parents[1]
    runner = (root / "scripts/run_v8_r2_reward_v2.sh").read_text()
    batch = (root / "scripts/euler_v8_r2_reward_v2/run.sbatch").read_text()
    submit = (root / "scripts/euler_v8_r2_reward_v2/submit.sh").read_text()
    normalized_batch = " ".join(batch.split())
    normalized_submit = " ".join(submit.split())
    goal = (root / "docs/research/V8_R2_IMPLEMENTATION_GOAL.md").read_text()
    assert "--config G-V8-CONTINUOUS-V2" in runner
    assert "--carry_work_observation" in runner
    assert "--reward_stage reward_v2" in runner
    assert "--ent_schedule_start 0.15" in runner
    assert "--ent_schedule_end 0.02" in runner
    assert 'ENTROPY_SCHEDULE_STEPS="${ENTROPY_SCHEDULE_STEPS:-20000}"' in runner
    assert "UPDATES=14000" in batch
    assert '"phase1_selection_updates=1000_to_14000_promotion_only"' in batch
    assert (
        '"promotion_selection_rule=combined_exact_then_47_condition_macro_then_worst_then_earliest"'
        in batch
    )
    assert '"initial_sampler_depth_mass_d0=0.11346390374331551"' in batch
    assert '"initial_sampler_depth_mass_d1=0.3836076203208556"' in batch
    assert '"initial_sampler_depth_mass_d2=0.5029284759358292"' in batch
    assert "agent_states_8_normalized_carry_work" in batch
    assert "/cluster/scratch/lterenzi/codex_terra_edge_runs/" in batch
    assert "--accepted-panel" in batch and "--capability-panel" in batch
    assert "sealed" not in batch
    assert "3051054bc4c713d95905d3f954e6eabf55d6a85a" in batch
    assert "3051054bc4c713d95905d3f954e6eabf55d6a85a" in submit
    assert "6905300337310456a28ec6177a8c7d74f73892ebe052d11d29e9e0fa5bec7362" in submit
    assert "cc969a69810b5ed0d14b85d58a0932ae26659a34686c4eadb760ae24b7cc87a4" in submit
    assert ") == (16.0, 2.5, 0.9984, 6.0, 1.0, 1.0, 1.5, 1.0, 1.0)" in normalized_batch
    for contract in (
        '"horizon=$R2_HORIZON"',
        '"statistical_continuation_supported=true"',
        '"bit_exact_continuation=false"',
        '"initialization=random_no_teacher"',
        '"comparison_role=historical_descriptive_reference"',
    ):
        assert contract in batch
    assert "one from-scratch compact reward-v2 system" in goal
    assert "descriptive" in goal and "reference only" in goal
    assert "40,000" in goal
    assert "14,000" in goal
    assert "R2_HORIZON=450" in submit
    assert 'test "$R2_HORIZON" -eq "$EXPECTED_HORIZON"' in batch
    assert ") == (16.0, 2.5, 0.9984, 6.0, 1.0, 1.0, 1.5, 1.0, 1.0)" in normalized_submit
    assert "sbatch --parsable" in submit
    assert "trap 'cleanup_new_job $?' ERR" in submit
    assert "\"rmdir -- '$RUN_DIR'\"" in submit
    assert "rm -rf" not in submit
    for path in (
        root / "scripts/run_v8_r2_reward_v2.sh",
        root / "scripts/euler_v8_r2_reward_v2/run.sbatch",
        root / "scripts/euler_v8_r2_reward_v2/submit.sh",
    ):
        assert path.stat().st_mode & stat.S_IXUSR
    for active in (runner, batch, submit):
        assert "PREPARED_FORK" not in active
        assert "prepared_fork.pkl" not in active
    assert "ARMS=" not in submit
    assert "terra-r2-control" not in submit
