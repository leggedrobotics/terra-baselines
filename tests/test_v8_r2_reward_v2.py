from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from flax.core import freeze

import train_mixed
from scripts import materialize_v8_r2_distance_bank as materialize
from scripts.prepare_v8_r2_fork import CARRY_KERNEL, expand_carry_input


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


def test_prepared_fork_and_arm_protocols_fail_closed():
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


def test_launcher_is_one_matched_prepared_fork_screen():
    root = Path(__file__).parents[1]
    runner = (root / "scripts/run_v8_r2_reward_v2.sh").read_text()
    batch = (root / "scripts/euler_v8_r2_reward_v2/run.sbatch").read_text()
    assert "--config G-V8-CONTINUOUS-V2" in runner
    assert "--prepared_fork_from" in runner
    assert "--carry_work_observation" in runner
    assert "--kickstart_lr_warmup_updates 100" in runner
    assert "--ent_schedule_start 0.02" in runner
    assert "ABSOLUTE_UPDATES=26000" in batch
    assert "ADDITIONAL_UPDATES=6000" in batch
    assert "agent_states_8_normalized_carry_work" in batch
    assert "/cluster/scratch/lterenzi/codex_terra_edge_runs/" in batch
    assert "--accepted-panel" in batch and "--capability-panel" in batch
    assert "sealed" not in batch
