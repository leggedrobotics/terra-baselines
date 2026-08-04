import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import v8_10m_student

ROOT = Path(__file__).resolve().parents[1]


def _hashed_contract(contract: dict) -> dict:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {"contract": contract, "sha256": hashlib.sha256(encoded).hexdigest()}


def _complete_treatment():
    conditions = ("fnd-a", "trn-a")
    bank = {
        "environment_protocol_sha256": "a" * 64,
        "source_registry_sha256": "b" * 64,
        "dataset": {
            "train": [
                {"condition_id": condition, "maps_path": f"train/{condition}"}
                for condition in conditions
            ]
        },
    }
    sampling = {"conditions": list(conditions)}
    levels = [
        {
            "maps_path": f"train/{condition}",
            "max_steps_in_episode": 450,
            "rewards_type": 0,
            "apply_trench_rewards": False,
        }
        for condition in conditions
    ]
    contract = {
        "schema": "terra_fixed_bank_treatment_v1",
        "run": {
            "name": "v8_screen_full_g_deep_xattn_v8_dense_warm_s7",
            "seed": 7,
            "config_name": "G-V8-FIXED",
            "accepted_bank_arm": "G-UNIFORM",
        },
        "bank": {
            "terra_revision": v8_10m_student.stage_gate.TERRA_REVISION,
            "environment_protocol_sha256": bank["environment_protocol_sha256"],
            "source_registry_sha256": bank["source_registry_sha256"],
        },
        "ppo": {
            "num_devices": 4,
            "num_envs_per_device": 1024,
            "num_steps": 32,
            "update_epochs": 2,
            "num_minibatches": 32,
            "lr": 0.0003,
            "gamma": 0.9984,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "vf_coef": 2.0,
            "max_grad_norm": 0.5,
            "ent_schedule_start": 0.02,
            "ent_schedule_end": 0.005,
            "ent_schedule_steps": 10000,
            "use_value_clip": False,
            "flat_minibatch_shuffle": True,
        },
        "reward_action": {
            "agent_types": [0],
            "action_types": [0],
            "relocation_progress_mult": 1.5,
            "curriculum_levels": levels,
        },
        "sampler": {
            "enabled": True,
            "rule": "fixed",
            "update_interval": 150,
            "uniform_floor": 0.20,
            "mastery_threshold": 0.75,
            "temperature": 0.25,
            "min_episodes": 20,
            "competence_ema": 0.30,
            "max_mass": 0.15,
            "seed": 7,
        },
        "architecture": v8_10m_student.TEACHER_ARCHITECTURE,
    }
    return bank, sampling, contract


def _receipt(parent):
    updates = [10_000, 12_000, 14_000]
    gate = {
        "contract": {
            "consecutive_scheduled_evaluations": 3,
            "overall_exact_min": [576, 720],
            "foundation_exact_min": [308, 384],
            "trench_exact_min": [269, 336],
            "every_main_condition_exact_min": [10, 16],
            "integrity_required": True,
        },
        "latest_window": {"updates": updates, "passed": True},
        "checkpoint_results": [
            {"update": update, "passed": True} for update in updates
        ],
        "selected_dense_parent": parent,
    }
    return {
        "schema": v8_10m_student.TEACHER_RECEIPT_SCHEMA,
        "passed": True,
        "qualified_for_reward_curriculum": True,
        "reward_launched": False,
        "arm": v8_10m_student.TEACHER_ARM,
        "release_id": v8_10m_student.stage_gate.RELEASE_ID,
        "terra_revision": v8_10m_student.stage_gate.TERRA_REVISION,
        "bank_archive_sha256": v8_10m_student.stage_gate.BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": v8_10m_student.stage_gate.BANK_DATASET_SHA256,
        "training_mixture_sha256": (v8_10m_student.stage_gate.TRAINING_MIXTURE_SHA256),
        "selected_checkpoint_validation": {"all_passed": True, "count": 4},
        "treatment_fingerprint": {},
        "selected_dense_parent": parent,
        "gate": gate,
        "inputs": {},
    }


def _manifest_record():
    manifest = []
    per_map = []
    successes = {"fnd-a": 12, "trn-a": 11}
    slot = 0
    for condition, family in (("fnd-a", "foundation"), ("trn-a", "trench")):
        for condition_index in range(16):
            slot += 1
            row = {
                "slot_index": slot,
                "episode_id": f"episode-{slot}",
                "primary_cell": condition,
                "family": family,
            }
            success = condition_index < successes[condition]
            manifest.append(row)
            per_map.append(
                {
                    **row,
                    "success": success,
                    "terminal_absolute": 1.0 if success else 0.75,
                    "integrity_failure": False,
                }
            )
    record = {
        "per_map": per_map,
        "summary": {
            "overall": {"successes": 23, "episodes": 32},
            "by_family": {
                "foundation": {"successes": 12, "episodes": 16},
                "trench": {"successes": 11, "episodes": 16},
            },
            "by_primary_cell": {
                "fnd-a": {"successes": 12, "episodes": 16},
                "trn-a": {"successes": 11, "episodes": 16},
            },
            "integrity": {"passed": True},
        },
    }
    return manifest, record


def test_frozen_parameter_count_is_approximately_10m():
    result = v8_10m_student.probe_parameter_count()
    assert result["passed"] is True
    assert result["teacher_parameter_count"] == 2_856_685
    assert result["parameter_count"] == 10_257_209
    assert result["finite_grown_parameters"] is True
    assert result["growth_categories"] == {
        "copied": 56,
        "dense-embed": 1,
        "sliced": 117,
    }


def test_teacher_inspection_has_no_external_evidence_bypass(tmp_path, monkeypatch):
    parent = {"path": "/teacher.pkl", "sha256": "c" * 64, "update": 14_000}
    value = _receipt(parent)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(value))
    calls = []

    def validate_external(receipt, observed_parent, bank_root):
        calls.append((receipt, observed_parent, bank_root))
        return {"four_panel_identity_validated": True}

    monkeypatch.setattr(v8_10m_student, "validate_external_evidence", validate_external)
    result = v8_10m_student.inspect_teacher(receipt_path, tmp_path)
    assert result["same_distribution"] is True
    assert result["teacher_checkpoint"] == "/teacher.pkl"
    assert len(calls) == 1

    value["bank_dataset_sha256"] = "d" * 64
    receipt_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="bank_dataset_sha256"):
        v8_10m_student.inspect_teacher(receipt_path, tmp_path)


def test_complete_teacher_treatment_rejects_ppo_or_reward_drift():
    bank, sampling, contract = _complete_treatment()
    v8_10m_student.validate_fingerprint(
        {"treatment_fingerprint": _hashed_contract(contract)}, bank, sampling, 7
    )

    changed = deepcopy(contract)
    changed["ppo"]["vf_coef"] = 1.0
    with pytest.raises(ValueError, match="PPO treatment"):
        v8_10m_student.validate_fingerprint(
            {"treatment_fingerprint": _hashed_contract(changed)}, bank, sampling, 7
        )

    changed = deepcopy(contract)
    changed["reward_action"]["curriculum_levels"][0]["rewards_type"] = 1
    with pytest.raises(ValueError, match="reward/action/map treatment"):
        v8_10m_student.validate_fingerprint(
            {"treatment_fingerprint": _hashed_contract(changed)}, bank, sampling, 7
        )


def test_per_map_accounting_is_recomputed_from_manifest():
    manifest, record = _manifest_record()
    result = v8_10m_student.validate_per_map_accounting(
        record,
        manifest,
        ("fnd-a", "trn-a"),
        {"fnd-a": "foundation", "trn-a": "trench"},
    )
    assert result["exact"]["successes"] == 23
    assert result["by_condition"]["fnd-a"]["successes"] == 12

    changed = deepcopy(record)
    changed["per_map"][0]["episode_id"] = "swapped"
    with pytest.raises(ValueError, match="identity differs"):
        v8_10m_student.validate_per_map_accounting(
            changed,
            manifest,
            ("fnd-a", "trn-a"),
            {"fnd-a": "foundation", "trn-a": "trench"},
        )

    changed = deepcopy(record)
    changed["summary"]["overall"]["successes"] = 24
    with pytest.raises(ValueError, match="overall score"):
        v8_10m_student.validate_per_map_accounting(
            changed,
            manifest,
            ("fnd-a", "trn-a"),
            {"fnd-a": "foundation", "trn-a": "trench"},
        )


def test_selected_development_gate_covers_families_cells_and_capability():
    main_conditions = {
        **{f"fnd-{index}": "foundation" for index in range(24)},
        **{f"trn-{index}": "trench" for index in range(21)},
    }
    main = {
        "exact": {
            "successes": 585,
            "episodes": 720,
            "by_family": {
                "foundation": {"successes": 312, "episodes": 384},
                "trench": {"successes": 273, "episodes": 336},
            },
        },
        "by_condition": {
            condition: {"family": family, "successes": 13, "episodes": 16}
            for condition, family in main_conditions.items()
        },
    }
    capability = {
        "by_condition": {
            condition: {"successes": 12, "episodes": 16}
            for condition in v8_10m_student.stage_gate.CAPABILITY_IDS
        }
    }
    v8_10m_student.validate_selected_development_gate(main, capability)

    first = v8_10m_student.stage_gate.CAPABILITY_IDS[0]
    capability["by_condition"][first]["successes"] = 11
    with pytest.raises(ValueError, match="capability-development exact"):
        v8_10m_student.validate_selected_development_gate(main, capability)


def test_euler_launcher_cannot_bypass_provisional_teacher_or_matched_smoke():
    submit = (ROOT / "scripts/euler_v8_10m_v1/submit.sh").read_text()
    sbatch = (ROOT / "scripts/euler_v8_10m_v1/run.sbatch").read_text()
    train = (ROOT / "scripts/run_v8_10m_screen.sh").read_text()

    assert "SUBMIT=0: no SSH, scratch, W&B, or Slurm mutation" in submit
    assert "smoke_validation.json" in submit
    assert "initialization_diagnostic.json" in submit
    assert "SMOKE_JOB_IDS" in submit
    assert "--dependency=afterok" in submit
    assert "SMOKE_STATE" in sbatch
    assert 'exact_frozen_map_slots\\"] == 720' in submit
    assert "deterministic_exact_slot_keys_v1" in submit
    assert "TEACHER_SHA" in submit
    assert "TEACHER_RUN_CONTRACT_SHA" in submit
    assert "v8_10m_provisional_teacher.py" in sbatch
    assert (
        "performance_mastery_gate_waived_by_user"
        in (ROOT / "scripts/v8_10m_provisional_teacher.py").read_text()
    )
    assert '--bank-root "$BANK"' in sbatch
    assert "--resnet_stage_channels 64,128,192,256" in sbatch
    assert "EXPECTED_PARAMETERS=10257209" in sbatch
    assert "UPDATES=2000" in sbatch
    assert "v8_10m_initialization.py" in sbatch
    assert "--capability-panel" in sbatch
    assert "v8_10m_curriculum_gate.py" in sbatch
    assert '--accepted-bank-stage "$STAGE"' in train
    assert "--teacher_checkpoint" in train
    assert "--kickstart_kl_anneal_updates 1500" in train
    assert "terminal_objective" not in train
