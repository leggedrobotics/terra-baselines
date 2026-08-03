import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.euler_v8_deep_xattn_v1 import continuation_tail_eval

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_deep_xattn_v1"
ARM = "G-DEEP-V8-DENSE-WARM"
REVISION = "a" * 40


def write_contract(path: Path, receipt: Path, source: Path, job_id: str) -> None:
    values = {
        "arm": ARM,
        "pairing": "matched_architecture_pair",
        "curriculum_stage": "full",
        "reward_stage": "dense_skill",
        "reward_type": "DENSE",
        "condition_sampler": "fixed_v8_stage_weights",
        "condition_count": "47",
        "seed": "7",
        "phase": "continuation",
        "resume_update": "8000",
        "absolute_target_update": "80000",
        "terra_revision": continuation_tail_eval.stage_gate.TERRA_REVISION,
        "terra_baselines_revision": REVISION,
        "training_bank_release_id": continuation_tail_eval.stage_gate.RELEASE_ID,
        "training_bank_archive_sha256": continuation_tail_eval.stage_gate.BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": continuation_tail_eval.stage_gate.BANK_DATASET_SHA256,
        "resume_checkpoint_path": str(source),
        "resume_checkpoint_sha256": continuation_tail_eval.sha256_file(source),
        "qualified_receipt": str(receipt),
        "qualified_receipt_sha256": continuation_tail_eval.sha256_file(receipt),
        "initialization": "true_resume_optimizer_schedule_sampler",
        "statistical_continuation": "true",
        "bit_exact_continuation": "false",
        "source_treatment_name": "v8_screen_full_g_deep_v8_dense_warm_s7",
        "trench_shaping": "false",
        "horizon": "450",
        "full_resets": "true",
        "checkpoint_interval": "500",
        "slurm_job_id": job_id,
    }
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")


def test_inventory_hashes_all_checkpoints_and_selects_cadence_plus_latest(
    tmp_path, monkeypatch
):
    remote_root = tmp_path / "runs"
    monkeypatch.setattr(
        continuation_tail_eval.stage_gate, "REMOTE_RUN_ROOT", remote_root
    )
    receipt = tmp_path / "qualified.json"
    receipt.write_text("{}\n")
    source = tmp_path / "source_update_008000.pkl"
    source.write_bytes(b"source")
    run_dir = remote_root / REVISION / "continuation" / "full" / "s7" / f"{ARM}-matched"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for update in range(8500, 13_001, 500):
        (checkpoints / f"v8_full_update_{update:06d}.pkl").write_bytes(
            f"checkpoint-{update}".encode()
        )
    contract = run_dir / "run_contract.env"
    write_contract(contract, receipt.resolve(), source.resolve(), "123")
    info = {
        "arm": ARM,
        "seed": 7,
        "terra_baselines_revision": REVISION,
        "candidate_path": str(source.resolve()),
        "candidate_sha256": continuation_tail_eval.sha256_file(source),
        "candidate_update": 8000,
    }
    monkeypatch.setattr(
        continuation_tail_eval.continuation_contract,
        "inspect_receipt",
        lambda path: ({}, info),
    )

    inventory = continuation_tail_eval.build_inventory(
        qualified_receipt=receipt,
        run_dir=run_dir,
        run_contract=contract,
        job_id="123",
        job_state="TIMEOUT",
        job_exit_code="0:0",
        job_partition="gpuhe.120h",
        evaluator_job_id="124",
    )

    assert len(inventory["continuation_checkpoints"]) == 10
    assert [item["update"] for item in inventory["selected_checkpoints"]] == [
        8000,
        10000,
        12000,
        13000,
    ]
    assert [
        item["reward_gate_scheduled"] for item in inventory["selected_checkpoints"]
    ] == [True, True, True, False]
    assert all(
        len(item["sha256"]) == 64 for item in inventory["continuation_checkpoints"]
    )


def test_inventory_rejects_non_terminal_state_and_checkpoint_gaps(tmp_path):
    with pytest.raises(ValueError, match="only COMPLETED or TIMEOUT"):
        continuation_tail_eval.normalize_job_state("OUT_OF_MEMORY")

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "v8_update_008500.pkl").write_bytes(b"8500")
    (checkpoints / "v8_update_009500.pkl").write_bytes(b"9500")
    with pytest.raises(ValueError, match=r"gaps: \[9000\]"):
        continuation_tail_eval.discover_continuation_checkpoints(checkpoints, 8000)


def test_selected_checkpoint_validation_reloads_state_and_sampler(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pkl"
    resumed = tmp_path / "run" / "checkpoints" / "resume_update_010000.pkl"
    resumed.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    resumed.write_bytes(b"resumed")
    fingerprint = {"sha256": "f" * 64, "contract": {}}
    architecture = {
        "model_size": "medium",
        "model_core": "mlp",
        "map_encoder": "resnet_spatial_8x8_se",
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": "encoder",
        "token_mixer_residual_init_scale": 0.0,
        "critic_hidden_dims": (512, 256),
        "resnet_stage_channels": (24, 48, 64, 96),
        "resnet_blocks_per_stage": (2, 2, 3, 3),
        "loaded_max": 100,
    }

    def checkpoint(path, update, optimizer_step, resumed_checkpoint):
        config = {
            **architecture,
            "accepted_bank": {
                "release_id": continuation_tail_eval.stage_gate.RELEASE_ID,
                "terra_revision": continuation_tail_eval.stage_gate.TERRA_REVISION,
                "curriculum_stage": "full",
            },
            "resume_from": str(source.resolve()) if resumed_checkpoint else None,
            "warm_start_from": None if resumed_checkpoint else "/prior.pkl",
            "teacher_checkpoint": None,
            "load_env_from_checkpoint": True,
        }
        return {
            "next_update": update,
            "model": {"weight": np.asarray([1.0])},
            "optimizer_state": {"moment": np.asarray([1.0])},
            "train_state_step": np.asarray(optimizer_step),
            "train_config": config,
            "pooled_sampler_state": {"state": path.name},
            "fingerprint": fingerprint,
        }

    checkpoints = {
        source.resolve(): checkpoint(source, 8000, 512000, False),
        resumed.resolve(): checkpoint(resumed, 10000, 640000, True),
    }
    monkeypatch.setattr(
        continuation_tail_eval.stage_gate,
        "_load_checkpoint",
        lambda path: checkpoints[Path(path).resolve()],
    )
    sampler_calls = []
    monkeypatch.setattr(
        continuation_tail_eval.stage_gate,
        "_validate_sampler_state",
        lambda state, sampling, seed, path: sampler_calls.append(
            (state, sampling, seed, path)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "eval_fixed_bank",
        SimpleNamespace(
            checkpoint_treatment_fingerprint=lambda value: value["fingerprint"]
        ),
    )
    inventory = {
        "arm": ARM,
        "seed": 7,
        "run_dir": str((tmp_path / "run").resolve()),
        "source_checkpoint": {"path": str(source.resolve())},
        "selected_checkpoints": [
            {
                "update": 8000,
                "path": str(source.resolve()),
                "sha256": continuation_tail_eval.sha256_file(source),
                "selection": "source_candidate",
            },
            {
                "update": 10000,
                "path": str(resumed.resolve()),
                "sha256": continuation_tail_eval.sha256_file(resumed),
                "selection": "scheduled_2000",
            },
        ],
    }
    records = [{"treatment_fingerprint": fingerprint}] * 2
    sampling = {"stage": "full"}

    validated = continuation_tail_eval.validate_selected_checkpoint_states(
        inventory, records, sampling
    )

    assert [item["optimizer_step"] for item in validated] == [512000, 640000]
    assert len(sampler_calls) == 2
    checkpoints[resumed.resolve()]["optimizer_state"]["moment"][0] = np.nan
    with pytest.raises(ValueError, match="optimizer state contains non-finite"):
        continuation_tail_eval.validate_selected_checkpoint_states(
            inventory, records, sampling
        )


def panel(condition_ids, families, successes):
    by_condition = {
        condition_id: {
            "family": families[condition_id],
            "successes": successes[condition_id],
            "episodes": 16,
            "mean_completion": 0.9,
        }
        for condition_id in condition_ids
    }
    by_family = {
        family: {
            "successes": sum(
                result["successes"]
                for result in by_condition.values()
                if result["family"] == family
            ),
            "episodes": sum(
                result["episodes"]
                for result in by_condition.values()
                if result["family"] == family
            ),
        }
        for family in ("foundation", "trench")
    }
    return {
        "exact": {
            "successes": sum(successes.values()),
            "episodes": 16 * len(condition_ids),
            "by_family": by_family,
        },
        "integrity": {"passed": True},
        "by_condition": by_condition,
    }


def test_reward_gate_requires_latest_three_scheduled_evaluations():
    foundations = tuple(f"fnd-{index}" for index in range(24))
    trenches = tuple(f"trn-{index}" for index in range(21))
    main_ids = (*foundations, *trenches)
    core_ids = (*foundations[:6], *trenches[:7])
    capability_ids = ("fnd-cap", "trn-cap")
    families = {
        **dict.fromkeys(foundations, "foundation"),
        **dict.fromkeys(trenches, "trench"),
        "fnd-cap": "foundation",
        "trn-cap": "trench",
    }
    main_successes = dict.fromkeys(main_ids, 13)
    capability_successes = dict.fromkeys(capability_ids, 12)
    thresholds = {
        **dict.fromkeys(core_ids, 11),
        **dict.fromkeys(capability_ids, 12),
    }
    history = []
    for update in (8000, 10000, 12000):
        history.append(
            {
                "update": update,
                "checkpoint": {"path": f"/{update}.pkl", "sha256": "a" * 64},
                "reward_gate_scheduled": True,
                "panels": {
                    "promotion": panel(main_ids, families, main_successes),
                    "capability_promotion": panel(
                        capability_ids, families, capability_successes
                    ),
                },
            }
        )
    gate = continuation_tail_eval.build_reward_gate(
        history, thresholds, core_ids, capability_ids
    )
    assert gate["qualified_for_reward_curriculum"] is True
    assert gate["selected_dense_parent"]["path"] == "/12000.pkl"
    assert gate["reward_launched"] is False

    history[-1]["panels"]["promotion"]["by_condition"]["fnd-0"]["successes"] = 9
    gate = continuation_tail_eval.build_reward_gate(
        history, thresholds, core_ids, capability_ids
    )
    assert gate["qualified_for_reward_curriculum"] is False
    assert gate["latest_window"]["passed"] is False


def test_launcher_is_afterany_four_panel_and_never_launches_reward():
    submit = (LAUNCHER / "submit_continuation_tail.sh").read_text()
    sbatch = (LAUNCHER / "continuation_tail.sbatch").read_text()
    instructions = (LAUNCHER / "CONTINUATION_TAIL.md").read_text()

    assert "dependency='afterany:$CONTINUATION_JOB_ID'" in submit
    assert "#SBATCH --partition=gpuhe.24h" in sbatch
    assert 'case "$PARENT_STATE" in' in sbatch
    assert "COMPLETED|TIMEOUT" in sbatch
    assert '--accepted-panel "$PANEL"' in sbatch
    assert '--capability-panel "$PANEL"' in sbatch
    assert "continuation_tail_eval.py" in sbatch
    assert "reward_launched=false" in sbatch
    assert "terminal_objective" not in sbatch
    assert "dense_reward_gate_receipt.json" in instructions
