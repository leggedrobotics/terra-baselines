import json
import subprocess
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.euler_accepted_bank_v1.select_promotion import (
    compare_reset_hashes,
    select_promotion,
    slice_agent_batch,
    verify_smoke,
)
from utils.helpers import save_pkl_object


def _record(arm, update, *, passed=True, macro=0.4, exact=2, worst=0.1):
    rule = "uniform" if arm == "G-UNIFORM" else "adaptive"
    return {
        "schema": "terra_fixed_bank_eval_v4",
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint": f"/checkpoints/{arm}-{update}.pkl",
        "checkpoint_sha256": f"{update:064x}",
        "checkpoint_update": update,
        "accepted_bank": {
            "terra_revision": "terra",
            "environment_protocol_sha256": "a" * 64,
            "source_registry_sha256": "b" * 64,
        },
        "split": "promotion",
        "stratum": "all",
        "manifest_sha256": "c" * 64,
        "horizon": 450,
        "deterministic": True,
        "policy_mode": "deterministic",
        "exact_manifest_enumeration": True,
        "treatment_fingerprint": {
            "contract": {
                "run": {
                    "name": arm,
                    "seed": 7,
                    "config_name": arm,
                    "accepted_bank_arm": arm,
                },
                "bank": {"identity": "same"},
                "ppo": {"lr": 3e-4},
                "reward_action": {"relocation_progress_mult": 1.5},
                "architecture": {"map_encoder": "resnet_spatial_8x8_se"},
                "sampler": {
                    "rule": rule,
                    "update_interval": 150,
                },
            },
            "sha256": f"{arm}-{update}",
        },
        "summary": {
            "overall": {"successes": exact, "episodes": 2},
            "graded": {
                "macro_completion": macro,
                "worst_condition_completion": worst,
            },
            "integrity": {"passed": True},
        },
        "comparison_to_previous": (
            None
            if update == 500
            else {
                "schema": "terra_fixed_bank_comparison_gate_v1",
                "passed": passed,
                "reference_checkpoint": f"/checkpoints/{arm}-previous.pkl",
                "integrity_passed": True,
            }
        ),
        "per_map": [
            {"episode_id": "episode-1", "map_id": "map-1"},
            {"episode_id": "episode-2", "map_id": "map-2"},
        ],
    }


def _write_screen(path, arm, *, passed=True, **final_metrics):
    records = [
        _record(arm, 500),
        _record(arm, 1000),
        _record(arm, 2000, passed=passed, **final_metrics),
    ]
    records[-1]["comparison_to_previous"]["reference_checkpoint"] = (
        records[-2]["checkpoint"]
    )
    path.write_text(json.dumps(records) + "\n")


def test_promotion_uses_only_generalists_and_uniform_wins_exact_tie(tmp_path):
    uniform = tmp_path / "uniform.json"
    adaptive = tmp_path / "adaptive.json"
    _write_screen(uniform, "G-UNIFORM", passed=True)
    _write_screen(adaptive, "G-ADAPTIVE", passed=True)
    decision = select_promotion(uniform, adaptive)
    assert decision["promotion_passed"]
    assert decision["selected_arm"] == "G-UNIFORM"


def test_promotion_ranks_passing_generalists_and_can_decline(tmp_path):
    uniform = tmp_path / "uniform.json"
    adaptive = tmp_path / "adaptive.json"
    _write_screen(uniform, "G-UNIFORM", passed=True, macro=0.4)
    _write_screen(adaptive, "G-ADAPTIVE", passed=True, macro=0.5)
    assert select_promotion(uniform, adaptive)["selected_arm"] == "G-ADAPTIVE"

    _write_screen(uniform, "G-UNIFORM", passed=False)
    _write_screen(adaptive, "G-ADAPTIVE", passed=False)
    decision = select_promotion(uniform, adaptive)
    assert not decision["promotion_passed"]
    assert decision["selected_arm"] is None


def test_promotion_rejects_wrong_arm_or_panel_identity(tmp_path):
    uniform = tmp_path / "uniform.json"
    adaptive = tmp_path / "adaptive.json"
    _write_screen(uniform, "G-UNIFORM")
    _write_screen(adaptive, "G-ADAPTIVE")
    payload = json.loads(adaptive.read_text())
    payload[-1]["per_map"][0]["episode_id"] = "different"
    adaptive.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="identical promotion panel"):
        select_promotion(uniform, adaptive)

    _write_screen(adaptive, "T-ANCHOR")
    with pytest.raises(ValueError, match="expected treatment arm"):
        select_promotion(uniform, adaptive)


def test_promotion_rejects_wrong_semantics_and_failed_integrity(tmp_path):
    uniform = tmp_path / "uniform.json"
    adaptive = tmp_path / "adaptive.json"
    _write_screen(uniform, "G-UNIFORM")
    _write_screen(adaptive, "G-ADAPTIVE")
    payload = json.loads(adaptive.read_text())
    payload[-1]["completion_contract"] = "legacy_buffered_dump"
    adaptive.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="completion contract"):
        select_promotion(uniform, adaptive)

    _write_screen(adaptive, "G-ADAPTIVE")
    payload = json.loads(adaptive.read_text())
    payload[-1]["comparison_to_previous"]["integrity_passed"] = False
    adaptive.write_text(json.dumps(payload) + "\n")
    decision = select_promotion(uniform, adaptive)
    assert decision["selected_arm"] == "G-UNIFORM"
    assert not decision["candidates"]["G-ADAPTIVE"]["passed"]


def test_reset_hash_comparison_requires_ordered_cpu_gpu_identity(tmp_path):
    base = {
        "schema": "terra_accepted_bank_reset_hashes_v1",
        "panel": "promotion",
        "slots": 2,
        "terra_revision": "terra",
        "environment_protocol_sha256": "a" * 64,
        "source_registry_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "episode_ids": ["e1", "e2"],
        "ordered_agent_state_sha256": ["h1", "h2"],
        "reset_verification": {"passed": True},
        "devices": [],
    }
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(json.dumps({**base, "backend": "cpu"}) + "\n")
    gpu.write_text(json.dumps({**base, "backend": "gpu"}) + "\n")
    assert compare_reset_hashes(cpu, gpu)["passed"]

    changed = {**base, "backend": "gpu"}
    changed["ordered_agent_state_sha256"] = ["h2", "h1"]
    gpu.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="ordered_agent_state_sha256"):
        compare_reset_hashes(cpu, gpu)


def test_agent_batch_slice_preserves_scalar_leaves():
    FakeAgent = namedtuple("FakeAgent", "scalar batched")
    agent = FakeAgent(
        scalar=jnp.asarray(3, dtype=jnp.int32),
        batched=jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int16),
    )
    selected = slice_agent_batch(agent, 1, 2)
    assert int(selected.scalar) == 3
    np.testing.assert_array_equal(selected.batched, np.array([3, 4]))


def test_smoke_validation_checks_update_finiteness_and_integrity(tmp_path):
    bank = SimpleNamespace(arm="G-UNIFORM")
    config = SimpleNamespace(accepted_bank=bank)
    checkpoint = {
        "next_update": 1,
        "train_config": config,
        "model": {"w": jnp.asarray([1.0])},
        "optimizer_state": {"m": jnp.asarray([0.0])},
        "loss_info": {"loss": jnp.asarray(0.5)},
        "transition_integrity": {
            "maximum_mass_residual": 0,
            "target_mutation_count": 0,
            "obstacle_mutation_count": 0,
        },
    }
    periodic = tmp_path / "periodic.pkl"
    final = tmp_path / "final.pkl"
    save_pkl_object(checkpoint, str(periodic))
    save_pkl_object(checkpoint, str(final))
    assert verify_smoke("G-UNIFORM", periodic, final)["passed"]

    checkpoint["model"] = {"w": jnp.asarray([np.nan])}
    save_pkl_object(checkpoint, str(final))
    with pytest.raises(ValueError, match="non-finite model"):
        verify_smoke("G-UNIFORM", periodic, final)


def test_launch_scripts_keep_dry_run_before_any_remote_mutation():
    root = Path(__file__).resolve().parents[1]
    prepare = (
        root / "scripts/euler_accepted_bank_v1/prepare_submit.sh"
    ).read_text()
    sbatch = (root / "scripts/euler_accepted_bank_v1/run.sbatch").read_text()
    assert prepare.index('if [ "$SUBMIT" = 0 ]') < prepare.index(
        'ssh "$REMOTE_HOST"'
    )
    assert "NON_ADMISSION.md" in prepare
    assert "REVIEW_ONLY.md" in prepare
    assert "gpuhe.24h" in prepare
    assert "gpuhe.120h" in prepare
    assert "separate 256-train-maps/condition bank is not implemented" in (
        prepare
    )
    assert "separate 256-train-maps/condition bank is not implemented" in (
        sbatch
    )
    assert "api.wandb.ai credentials in ~/.netrc" in sbatch
    assert "NUM_DEVICES=4" in sbatch
    assert "NUM_ENVS_PER_DEVICE=1024" in sbatch
    assert "NUM_STEPS=32" in sbatch
    assert "EXPECTED_TRAIN_MAPS_PER_CONDITION=64" in prepare
    assert (
        '"train_maps_per_condition": int(train_maps_per_condition)'
        in prepare
    )
    assert (
        'test "$MANIFEST_TRAIN_MAPS_PER_CONDITION" = 64'
        in sbatch
    )
    assert sbatch.count(
        '"train_maps_per_condition=$MANIFEST_TRAIN_MAPS_PER_CONDITION"'
    ) == 2
    assert sbatch.index(
        'test "$MANIFEST_TRAIN_MAPS_PER_CONDITION" = 64'
    ) < sbatch.index("module load stack/2024-06")
    assert '/$PHASE/s$SEED"' in prepare
    assert 'RUN_DIR="$RUN_PARENT/$ARM"' in prepare
    assert "/$PHASE/s$SEED/$ARM" in sbatch
    assert sbatch.index("compare-reset-hashes") < sbatch.index(
        '"${TRAIN_COMMAND[@]}"'
    )


def test_promote_phase_fails_before_reading_paths():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/euler_accepted_bank_v1/prepare_submit.sh"
    )
    result = subprocess.run(
        [str(script), "promote", "/missing/terra", "/missing/bank"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 9
    assert "256-train-maps/condition bank is not implemented" in result.stderr
