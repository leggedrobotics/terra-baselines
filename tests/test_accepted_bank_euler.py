import hashlib
import json
import os
import subprocess
import sys
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
from utils.accepted_bank import ARMS
from utils.helpers import save_pkl_object


RESET_ARRAY_FOLDERS = (
    "images",
    "occupancy",
    "dumpability",
    "actions",
    "distance",
)


def _write_prepare_bank(
    root,
    *,
    declared_count=64,
    manifest_count=None,
    array_count=None,
):
    manifest_count = declared_count if manifest_count is None else manifest_count
    array_count = declared_count if array_count is None else array_count
    level = root / "train" / "condition"
    level.mkdir(parents=True)
    review_admission = root / "review_admission.json"
    review_admission.write_text(
        json.dumps(
            {
                "schema": "terra-accepted-condition-set-v1",
                "release": "map-curriculum-diverse64-visual-review-20260730",
                "manifest_sha256": (
                    "39f7cd2e8ce565bd384de214da5f2eee5e76764cb554e149c0ba675d815d6d51"
                ),
                "review_data_sha256": (
                    "8404fcaa9a6b66949ade2b0225d3e7800968951953d2b6363aabffe38100cc0b"
                ),
                "accepted_conditions": ["condition"],
            }
        )
        + "\n"
    )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "review_admission": "review_admission.json",
                "review_admission_sha256": hashlib.sha256(
                    review_admission.read_bytes()
                ).hexdigest(),
                "train": [
                    {
                        "condition_id": "condition",
                        "family": "foundation",
                        "branch_depth": "Anchor",
                        "maps_path": "train/condition",
                        "map_count": declared_count,
                    }
                ]
            }
        )
        + "\n"
    )
    (level / "dataset.json").write_text(
        json.dumps(
            {
                "slot_count": declared_count,
                "num_maps": declared_count,
            }
        )
        + "\n"
    )
    rows = [
        {
            "slot_index": slot,
            "map_id": f"map-{slot}",
            "family": "foundation",
            "primary_cell": "condition",
        }
        for slot in range(1, manifest_count + 1)
    ]
    (level / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    for folder in RESET_ARRAY_FOLDERS:
        directory = level / folder
        directory.mkdir()
        for slot in range(1, array_count + 1):
            (directory / f"img_{slot}.npy").write_bytes(b"test")
    return root


def _run_prepare_validator(bank_root):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    return subprocess.run(
        [
            sys.executable,
            str(
                root
                / "scripts/euler_accepted_bank_v1/validate_training_bank.py"
            ),
            str(bank_root),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _record(arm, update, *, passed=True, macro=0.4, exact=2, worst=0.1):
    rule = "adaptive" if arm == "G-ADAPTIVE" else "uniform"
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

    _write_screen(adaptive, "F-SPECIALIST")
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


@pytest.mark.parametrize("arm", ARMS)
def test_smoke_validation_checks_update_finiteness_and_integrity(tmp_path, arm):
    bank = SimpleNamespace(arm=arm)
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
    assert verify_smoke(arm, periodic, final)["passed"]

    checkpoint["model"] = {"w": jnp.asarray([np.nan])}
    save_pkl_object(checkpoint, str(final))
    with pytest.raises(ValueError, match="non-finite model"):
        verify_smoke(arm, periodic, final)


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
    for arm in ARMS:
        assert arm in prepare
        assert arm in sbatch
    assert '"schema": "terra_accepted_bank_euler_campaign_v2"' in prepare
    assert '"arms": arms_csv.split(",")' in prepare
    assert "terra_accepted_bank_euler_receipt_v2" in prepare
    assert "terra_accepted_bank_euler_receipt_v2" in sbatch
    assert "campaign arm matrix mismatch" in sbatch
    assert '"$BANK_ROOT/review_admission.json"' in prepare
    assert '"bank_review_admission_sha256": review_admission_sha' in prepare
    assert "validate_training_bank.py" in prepare
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
    assert "bank_review_admission_sha256" in sbatch
    assert "review admission SHA mismatch" in sbatch
    assert '/$PHASE/s$SEED"' in prepare
    assert 'RUN_DIR="$RUN_PARENT/$ARM"' in prepare
    assert "/$PHASE/s$SEED/$ARM" in sbatch
    assert sbatch.index("compare-reset-hashes") < sbatch.index(
        '"${TRAIN_COMMAND[@]}"'
    )


def test_prepare_validator_accepts_complete_64_map_bank(tmp_path):
    result = _run_prepare_validator(_write_prepare_bank(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "64"


def test_prepare_validator_requires_review_admission(tmp_path):
    bank = _write_prepare_bank(tmp_path)
    (bank / "review_admission.json").unlink()
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert "review_admission.json" in result.stderr


def test_prepare_validator_rejects_review_hash_or_condition_mismatch(tmp_path):
    bank = _write_prepare_bank(tmp_path / "hash")
    receipt_path = bank / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["note"] = "changed after the descriptor was written"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert "review admission hash mismatch" in result.stderr

    bank = _write_prepare_bank(tmp_path / "conditions")
    receipt_path = bank / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["accepted_conditions"] = ["different"]
    receipt_path.write_text(json.dumps(receipt) + "\n")
    index_path = bank / "dataset.json"
    index = json.loads(index_path.read_text())
    index["review_admission_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index) + "\n")
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert "do not match train condition IDs" in result.stderr


@pytest.mark.parametrize(
    "field",
    ["release", "manifest_sha256", "review_data_sha256"],
)
def test_prepare_validator_rejects_stale_review_identity(tmp_path, field):
    bank = _write_prepare_bank(tmp_path)
    receipt_path = bank / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = "stale"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    index_path = bank / "dataset.json"
    index = json.loads(index_path.read_text())
    index["review_admission_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index) + "\n")
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert f"{field} must match the pinned diverse-64 review release" in (
        result.stderr
    )


def test_prepare_validator_rejects_wrong_declaration(tmp_path):
    result = _run_prepare_validator(
        _write_prepare_bank(tmp_path, declared_count=63)
    )
    assert result.returncode != 0
    assert "must declare exactly 64 train maps" in result.stderr


def test_prepare_validator_rejects_64_declared_63_manifest(tmp_path):
    result = _run_prepare_validator(
        _write_prepare_bank(
            tmp_path,
            declared_count=64,
            manifest_count=63,
            array_count=64,
        )
    )
    assert result.returncode != 0
    assert "declares 64 maps" in result.stderr
    assert "contains 63" in result.stderr


def test_prepare_validator_requires_local_slot_count(tmp_path):
    bank = _write_prepare_bank(tmp_path)
    metadata_path = bank / "train/condition/dataset.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("slot_count")
    metadata_path.write_text(json.dumps(metadata) + "\n")
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert "slot_count must be 64" in result.stderr


def test_prepare_validator_rejects_missing_or_extra_reset_array(tmp_path):
    bank = _write_prepare_bank(tmp_path)
    last_array = bank / "train/condition/images/img_64.npy"
    last_array.unlink()
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert "must contain exactly img_1.npy..img_64.npy" in result.stderr

    last_array.write_bytes(b"test")
    (bank / "train/condition/images/img_65.npy").write_bytes(b"test")
    result = _run_prepare_validator(bank)
    assert result.returncode != 0
    assert "must contain exactly img_1.npy..img_64.npy" in result.stderr


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
