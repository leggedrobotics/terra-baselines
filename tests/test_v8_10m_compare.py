import hashlib
import json
from pathlib import Path

import pytest

from scripts import v8_10m_compare


def _fingerprint(arm: str):
    contract = {
        "schema": "terra_fixed_bank_treatment_v1",
        "run": {
            "name": f"screen_full_{arm.lower().replace('-', '_')}",
            "seed": 7,
            "config_name": "G-V8-FIXED",
            "accepted_bank_arm": "G-UNIFORM",
        },
        "bank": {
            "terra_revision": v8_10m_compare.stage_gate.TERRA_REVISION,
            "environment_protocol_sha256": "c" * 64,
            "source_registry_sha256": "b" * 64,
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
            "curriculum_levels": [
                {
                    "maps_path": f"train/condition-{index}",
                    "max_steps_in_episode": 450,
                    "rewards_type": 0,
                    "apply_trench_rewards": False,
                }
                for index in range(47)
            ],
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
        "architecture": v8_10m_compare.EXPECTED_ARCHITECTURE[arm],
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {"contract": contract, "sha256": hashlib.sha256(encoded).hexdigest()}


def _record(update: int, checkpoint: Path, offset: float = 0.0):
    rows = []
    slot = 0
    for index, (condition, family) in enumerate(
        (("fnd-a", "foundation"), ("trn-a", "trench")), start=1
    ):
        completion = min(1.0, 0.2 * index + offset)
        for _ in range(16):
            slot += 1
            rows.append(
                {
                    "episode_id": f"episode-{slot}",
                    "slot_index": slot,
                    "primary_cell": condition,
                    "family": family,
                    "terminal_absolute": completion,
                    "success": completion == 1.0,
                }
            )
    return {
        "completion_contract": "exact_visible_dump_v1",
        "deterministic": True,
        "policy_mode": "deterministic",
        "exact_manifest_enumeration": True,
        "horizon": 450,
        "checkpoint_update": update,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "a" * 64,
        "split": "promotion",
        "accepted_bank": {
            "schema": "terra_curriculum_loader_bank_v1",
            "terra_revision": v8_10m_compare.stage_gate.TERRA_REVISION,
            "environment_protocol_sha256": "c" * 64,
            "source_registry_sha256": "b" * 64,
            "diagnostic_control": False,
            "diagnostic_contract_sha256": None,
        },
        "reset_verification": {"passed": True},
        "summary": {
            "integrity": {"passed": True},
            "overall": {
                "successes": sum(row["success"] for row in rows),
                "episodes": len(rows),
            },
        },
        "per_map": rows,
    }


def test_common_schedule_keeps_kickstart_diagnostics_and_latest():
    assert v8_10m_compare.selected_common_updates(
        list(range(500, 5_501, 500)), list(range(500, 4_501, 500))
    ) == [500, 1_000, 1_500, 2_000, 4_000, 4_500]


def test_panel_snapshot_recomputes_family_and_condition_macro():
    record = _record(500, Path("/checkpoint.pkl"))
    result = v8_10m_compare.panel_snapshot(record)
    assert result["macro_completion"] == pytest.approx(0.3)
    assert result["by_family"]["foundation"]["macro_completion"] == pytest.approx(0.2)
    assert result["by_family"]["trench"]["macro_completion"] == pytest.approx(0.4)
    assert result["worst_condition"] == "fnd-a"


def test_treatment_fingerprint_rejects_shared_ppo_drift():
    arm = v8_10m_compare.CONTROL
    record = _record(500, Path("/checkpoint.pkl"))
    fingerprint = _fingerprint(arm)
    record["treatment_fingerprint"] = fingerprint
    v8_10m_compare.validate_treatment_fingerprint(record, arm, "7")

    fingerprint["contract"]["ppo"]["vf_coef"] = 1.0
    encoded = json.dumps(
        fingerprint["contract"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    fingerprint["sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="PPO treatment"):
        v8_10m_compare.validate_treatment_fingerprint(record, arm, "7")


def test_paired_episode_identity_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        v8_10m_compare,
        "EXPECTED_PANEL_SHAPE",
        {panel: (2, 32, 16, 16) for panel in v8_10m_compare.PANELS},
    )
    checkpoints = {}
    paths = {}
    for arm in v8_10m_compare.ARMS:
        checkpoint = tmp_path / f"{arm}.pkl"
        checkpoint.write_bytes(b"checkpoint")
        identity = {
            "update": 500,
            "path": str(checkpoint.resolve()),
            "sha256": "a" * 64,
        }
        checkpoints[arm] = identity
        for panel in v8_10m_compare.PANELS:
            record = _record(500, checkpoint.resolve())
            record["split"] = panel.removeprefix("capability_")
            record["treatment_fingerprint"] = _fingerprint(arm)
            path = tmp_path / f"{arm}-{panel}.json"
            path.write_text(json.dumps([record]))
            paths[(arm, panel)] = path
    treatment = json.loads(paths[(v8_10m_compare.TREATMENT, "promotion")].read_text())
    treatment[0]["per_map"][0]["episode_id"] = "changed"
    paths[(v8_10m_compare.TREATMENT, "promotion")].write_text(json.dumps(treatment))
    inventory = {
        "selected_common_updates": [500],
        "jobs": {
            arm: {
                "selected_checkpoints": [checkpoints[arm]],
                "run_contract": {"seed": "7"},
            }
            for arm in v8_10m_compare.ARMS
        },
    }
    with pytest.raises(ValueError, match="episode sequence"):
        v8_10m_compare.validate_evaluations(inventory=inventory, paths=paths)
