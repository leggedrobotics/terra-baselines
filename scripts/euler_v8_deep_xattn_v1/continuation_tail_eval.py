#!/usr/bin/env python3
"""Freeze and summarize the tail of one V8 120-hour continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from . import continuation_contract, stage_gate
else:
    import continuation_contract
    import stage_gate


INVENTORY_SCHEMA = "terra_v8_continuation_tail_inventory_v1"
LEADERBOARD_SCHEMA = "terra_v8_continuation_leaderboard_v1"
GATE_SCHEMA = "terra_v8_dense_reward_gate_v1"
TARGET_UPDATE = 80_000
CHECKPOINT_INTERVAL = 500
EVALUATION_INTERVAL = 2_000
CHECKPOINT_RE = re.compile(r".+_update_([0-9]{6})\.pkl")
ALLOWED_JOB_STATES = {"COMPLETED", "TIMEOUT"}

GATE_OVERALL = 576
GATE_FOUNDATION = 308
GATE_TRENCH = 269
GATE_CELL = 10
GATE_CONSECUTIVE = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, expected_type):
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{path}: expected {expected_type.__name__}")
    return value


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_job_state(state: str) -> str:
    normalized = state.split("+", 1)[0]
    if normalized not in ALLOWED_JOB_STATES:
        raise ValueError(
            f"continuation job ended {normalized!r}; only COMPLETED or TIMEOUT "
            "is tail-evaluable"
        )
    return normalized


def validate_continuation_contract(
    path: Path,
    run_dir: Path,
    qualified_receipt: Path,
    qualified_info: dict,
    job_id: str,
) -> dict[str, str]:
    path = path.resolve()
    run_dir = run_dir.resolve()
    qualified_receipt = qualified_receipt.resolve()
    if path != run_dir / "run_contract.env":
        raise ValueError("continuation contract must be RUN_DIR/run_contract.env")
    contract = stage_gate.parse_run_contract(path)
    arm = qualified_info["arm"]
    seed = qualified_info["seed"]
    revision = qualified_info["terra_baselines_revision"]
    qualified_sha = sha256_file(qualified_receipt)
    expected = {
        "arm": arm,
        "curriculum_stage": "full",
        "reward_stage": "dense_skill",
        "reward_type": "DENSE",
        "condition_sampler": "fixed_v8_stage_weights",
        "condition_count": "47",
        "seed": str(seed),
        "phase": "continuation",
        "resume_update": str(qualified_info["candidate_update"]),
        "absolute_target_update": str(TARGET_UPDATE),
        "terra_revision": stage_gate.TERRA_REVISION,
        "terra_baselines_revision": revision,
        "training_bank_release_id": stage_gate.RELEASE_ID,
        "training_bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "resume_checkpoint_path": qualified_info["candidate_path"],
        "resume_checkpoint_sha256": qualified_info["candidate_sha256"],
        "qualified_receipt": str(qualified_receipt),
        "qualified_receipt_sha256": qualified_sha,
        "initialization": "true_resume_optimizer_schedule_sampler",
        "statistical_continuation": "true",
        "bit_exact_continuation": "false",
        "trench_shaping": "false",
        "horizon": "450",
        "full_resets": "true",
        "checkpoint_interval": str(CHECKPOINT_INTERVAL),
        "slurm_job_id": job_id,
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"continuation contract {field} must be {value!r}, "
                f"got {contract.get(field)!r}"
            )
    if contract.get("pairing") not in {
        "unpaired_single_qualifying_arm",
        "matched_architecture_pair",
    }:
        raise ValueError("continuation contract has an unsupported pairing")
    treatment_name = contract.get("source_treatment_name")
    if (
        not isinstance(treatment_name, str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", treatment_name) is None
    ):
        raise ValueError("continuation contract lacks a valid source treatment name")

    expected_parent = (
        stage_gate.REMOTE_RUN_ROOT / revision / "continuation" / "full" / f"s{seed}"
    )
    if run_dir.parent != expected_parent:
        raise ValueError("continuation run directory changed campaign identity")
    expected_names = {f"{arm}-unpaired", f"{arm}-matched"}
    if run_dir.name not in expected_names:
        raise ValueError("continuation run directory changed arm/pairing identity")
    return contract


def discover_continuation_checkpoints(
    checkpoints_dir: Path, resume_update: int
) -> list[dict]:
    checkpoints_dir = checkpoints_dir.resolve()
    by_update: dict[int, Path] = {}
    for path in sorted(checkpoints_dir.glob("*_update_*.pkl")):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unsupported periodic checkpoint name: {path.name}")
        update = int(match.group(1))
        if not resume_update < update <= TARGET_UPDATE:
            raise ValueError(
                f"continuation checkpoint {update} is outside "
                f"({resume_update}, {TARGET_UPDATE}]"
            )
        if update % CHECKPOINT_INTERVAL:
            raise ValueError(
                f"continuation checkpoint {update} is off the 500 schedule"
            )
        if update in by_update:
            raise ValueError(f"duplicate continuation checkpoint for update {update}")
        by_update[update] = path.resolve()
    if not by_update:
        raise ValueError("continuation produced no complete periodic checkpoint")
    observed = sorted(by_update)
    expected = list(
        range(
            resume_update + CHECKPOINT_INTERVAL, observed[-1] + 1, CHECKPOINT_INTERVAL
        )
    )
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        raise ValueError(f"continuation checkpoint schedule has gaps: {missing}")
    return [
        {
            "update": update,
            "path": str(by_update[update]),
            "sha256": sha256_file(by_update[update]),
        }
        for update in observed
    ]


def select_evaluation_checkpoints(source: dict, discovered: list[dict]) -> list[dict]:
    resume_update = int(source["update"])
    selected = [
        {
            **source,
            "selection": "source_candidate",
            "reward_gate_scheduled": True,
        }
    ]
    selected_updates = {resume_update}
    for checkpoint in discovered:
        update = int(checkpoint["update"])
        if (update - resume_update) % EVALUATION_INTERVAL == 0:
            selected.append(
                {
                    **checkpoint,
                    "selection": "scheduled_2000",
                    "reward_gate_scheduled": True,
                }
            )
            selected_updates.add(update)
    latest = discovered[-1]
    if latest["update"] not in selected_updates:
        selected.append(
            {
                **latest,
                "selection": "latest_complete_diagnostic",
                "reward_gate_scheduled": False,
            }
        )
    return sorted(selected, key=lambda checkpoint: checkpoint["update"])


def build_inventory(
    *,
    qualified_receipt: Path,
    run_dir: Path,
    run_contract: Path,
    job_id: str,
    job_state: str,
    job_exit_code: str,
    job_partition: str,
    evaluator_job_id: str,
) -> dict:
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("continuation job id must contain only digits")
    if re.fullmatch(r"[0-9]+", evaluator_job_id) is None:
        raise ValueError("evaluator job id must contain only digits")
    if re.fullmatch(r"[0-9]+:[0-9]+", job_exit_code) is None:
        raise ValueError("continuation exit code must use Slurm status:signal syntax")
    state = normalize_job_state(job_state)
    if state == "COMPLETED" and job_exit_code != "0:0":
        raise ValueError("a COMPLETED continuation must have exit code 0:0")
    if job_partition != "gpuhe.120h":
        raise ValueError("continuation must have run on gpuhe.120h")

    qualified_receipt = qualified_receipt.resolve()
    _, qualified_info = continuation_contract.inspect_receipt(qualified_receipt)
    contract = validate_continuation_contract(
        run_contract,
        run_dir,
        qualified_receipt,
        qualified_info,
        job_id,
    )
    source_path = Path(qualified_info["candidate_path"])
    if sha256_file(source_path) != qualified_info["candidate_sha256"]:
        raise ValueError("qualified source checkpoint hash changed")
    source = {
        "update": int(qualified_info["candidate_update"]),
        "path": str(source_path),
        "sha256": qualified_info["candidate_sha256"],
    }
    discovered = discover_continuation_checkpoints(
        run_dir.resolve() / "checkpoints", source["update"]
    )
    selected = select_evaluation_checkpoints(source, discovered)
    return {
        "schema": INVENTORY_SCHEMA,
        "generated_at_utc": utc_now(),
        "arm": qualified_info["arm"],
        "seed": qualified_info["seed"],
        "terra_baselines_revision": qualified_info["terra_baselines_revision"],
        "qualified_full_receipt": {
            "path": str(qualified_receipt),
            "sha256": sha256_file(qualified_receipt),
        },
        "continuation_job": {
            "job_id": job_id,
            "state": state,
            "exit_code": job_exit_code,
            "partition": job_partition,
            "evaluator_job_id": evaluator_job_id,
        },
        "run_dir": str(run_dir.resolve()),
        "run_contract": {
            "path": str(run_contract.resolve()),
            "sha256": sha256_file(run_contract.resolve()),
            "source_treatment_name": contract["source_treatment_name"],
            "pairing": contract["pairing"],
        },
        "resume_update": source["update"],
        "target_update": TARGET_UPDATE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "evaluation_interval": EVALUATION_INTERVAL,
        "source_checkpoint": source,
        "continuation_checkpoints": discovered,
        "selected_checkpoints": selected,
    }


def validate_inventory(inventory_path: Path, qualified_receipt: Path) -> dict:
    inventory = read_json(inventory_path, dict)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unsupported continuation inventory schema")
    normalize_job_state(inventory.get("continuation_job", {}).get("state", ""))
    qualified_receipt = qualified_receipt.resolve()
    frozen = inventory.get("qualified_full_receipt", {})
    if frozen.get("path") != str(qualified_receipt) or frozen.get(
        "sha256"
    ) != sha256_file(qualified_receipt):
        raise ValueError("continuation inventory names a different qualified receipt")
    for checkpoint in [inventory["source_checkpoint"]] + inventory[
        "continuation_checkpoints"
    ]:
        if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError(
                f"checkpoint changed after inventory: {checkpoint['path']}"
            )
    return inventory


def validate_evaluation_records(
    *,
    path: Path,
    panel_group: str,
    split: str,
    inventory: dict,
    bank: dict,
    sampling: dict,
) -> list[dict]:
    records = read_json(path, list)
    selected = inventory["selected_checkpoints"]
    if len(records) != len(selected):
        raise ValueError(f"{path}: evaluation did not cover every selected checkpoint")
    expected_panel = stage_gate.panel_contract(bank, panel_group, split)
    condition_ids = (
        stage_gate.CAPABILITY_IDS if panel_group == "capability" else bank["main_ids"]
    )
    reference_fingerprint = None
    for record, checkpoint in zip(records, selected):
        expected = {
            "schema": stage_gate.EVAL_SCHEMA,
            "completion_contract": stage_gate.COMPLETION_CONTRACT,
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_update": checkpoint["update"],
            "bank_root": expected_panel["bank_root"],
            "manifest": expected_panel["manifest"],
            "manifest_sha256": expected_panel["manifest_sha256"],
            "horizon": 450,
            "deterministic": True,
            "policy_mode": "deterministic",
            "exact_manifest_enumeration": True,
            "split": split,
            "stratum": expected_panel["stratum"],
            "seed": inventory["seed"],
            "reset_verification": expected_panel["reset_verification"],
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValueError(f"{path}: {field} must be {value!r}")
        expected_bank = {
            "schema": "terra_curriculum_loader_bank_v1",
            "terra_revision": stage_gate.TERRA_REVISION,
            "environment_protocol_sha256": bank["environment_protocol_sha256"],
            "source_registry_sha256": bank["source_registry_sha256"],
            "diagnostic_control": False,
            "diagnostic_contract_sha256": None,
        }
        if record.get("accepted_bank") != expected_bank:
            raise ValueError(f"{path}: accepted-bank identity changed")
        stage_gate._validate_treatment(
            record,
            bank,
            "full",
            inventory["arm"],
            inventory["seed"],
            sampling,
        )
        fingerprint = record.get("treatment_fingerprint")
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        elif fingerprint != reference_fingerprint:
            raise ValueError(f"{path}: treatment changed between checkpoints")
    stage_gate.validate_panel_conditions(records, condition_ids)
    return records


def validate_selected_checkpoint_states(
    inventory: dict, promotion_records: list[dict], sampling: dict
) -> list[dict]:
    """Reload every scored checkpoint and verify its true training state."""
    from eval_fixed_bank import checkpoint_treatment_fingerprint

    selected = inventory["selected_checkpoints"]
    if len(selected) != len(promotion_records):
        raise ValueError("checkpoint validation does not cover the promotion history")
    source_path = Path(inventory["source_checkpoint"]["path"]).resolve()
    continuation_dir = Path(inventory["run_dir"]).resolve() / "checkpoints"
    previous_step = None
    validations = []
    for identity, record in zip(selected, promotion_records):
        path = Path(identity["path"]).resolve()
        if identity["selection"] == "source_candidate":
            if path != source_path:
                raise ValueError("source evaluation names a different checkpoint")
        else:
            if path.parent != continuation_dir:
                raise ValueError("resumed checkpoint is outside the continuation run")
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"checkpoint changed before state validation: {path}")

        checkpoint = stage_gate._load_checkpoint(path)
        if int(checkpoint.get("next_update", -1)) != int(identity["update"]):
            raise ValueError(f"{path}: checkpoint update changed")
        for field in ("model", "optimizer_state", "train_state_step"):
            if field not in checkpoint:
                raise ValueError(f"{path}: checkpoint lacks {field}")
        continuation_contract._require_finite_tree(
            checkpoint["model"], f"{path}: model parameters"
        )
        continuation_contract._require_finite_tree(
            checkpoint["optimizer_state"], f"{path}: optimizer state"
        )
        step_array = np.asarray(checkpoint["train_state_step"])
        if step_array.size != 1 or not np.isfinite(step_array).all():
            raise ValueError(f"{path}: train_state_step is not one finite scalar")
        optimizer_step = int(step_array.reshape(()))
        if optimizer_step <= 0 or (
            previous_step is not None and optimizer_step <= previous_step
        ):
            raise ValueError(f"{path}: optimizer step did not advance")
        previous_step = optimizer_step

        config = checkpoint.get("train_config")
        if config is None:
            raise ValueError(f"{path}: checkpoint lacks train_config")
        bank = stage_gate._field(config, "accepted_bank")
        expected_bank = {
            "release_id": stage_gate.RELEASE_ID,
            "terra_revision": stage_gate.TERRA_REVISION,
            "curriculum_stage": "full",
        }
        for field, value in expected_bank.items():
            if stage_gate._field(bank, field) != value:
                raise ValueError(f"{path}: accepted-bank {field} changed")
        architecture = {
            field: stage_gate._field(config, field)
            for field in (
                "model_size",
                "model_core",
                "map_encoder",
                "encoder_compute_dtype",
                "attention_compute_dtype",
                "token_mixer_residual_init_scale",
                "critic_hidden_dims",
                "resnet_stage_channels",
                "resnet_blocks_per_stage",
                "loaded_max",
            )
        }
        stage_gate._validate_architecture(architecture, inventory["arm"])
        observed_fingerprint = checkpoint_treatment_fingerprint(checkpoint)
        if observed_fingerprint != record.get("treatment_fingerprint"):
            raise ValueError(f"{path}: checkpoint treatment differs from evaluation")
        stage_gate._validate_sampler_state(
            checkpoint.get("pooled_sampler_state"),
            sampling,
            int(inventory["seed"]),
            path,
        )

        if identity["selection"] != "source_candidate":
            if stage_gate._field(config, "resume_from") != str(source_path):
                raise ValueError(f"{path}: continuation source changed")
            if stage_gate._field(config, "warm_start_from") is not None:
                raise ValueError(f"{path}: continuation became a warm start")
            if stage_gate._field(config, "teacher_checkpoint") is not None:
                raise ValueError(f"{path}: continuation unexpectedly loaded a teacher")
            if stage_gate._field(config, "load_env_from_checkpoint") is not True:
                raise ValueError(f"{path}: continuation stopped restoring env config")
        validations.append(
            {
                "update": int(identity["update"]),
                "path": str(path),
                "sha256": identity["sha256"],
                "optimizer_step": optimizer_step,
                "finite_model_optimizer": True,
                "sampler_state_validated": True,
                "treatment_fingerprint_sha256": observed_fingerprint["sha256"],
            }
        )
    return validations


def panel_snapshot(
    record: dict, condition_ids: tuple[str, ...], families: dict
) -> dict:
    cells = record["summary"]["by_primary_cell"]
    graded = record["summary"]["graded"]
    if graded.get("available") is not True:
        raise ValueError("continuation evaluation lacks graded completion")
    by_condition = {}
    exact_by_family = {
        "foundation": {"successes": 0, "episodes": 0},
        "trench": {"successes": 0, "episodes": 0},
    }
    for condition_id in condition_ids:
        cell = cells[condition_id]
        episodes = cell.get("episodes")
        successes = cell.get("successes")
        completion = graded["by_primary_cell"][condition_id].get("mean")
        if (
            episodes != 16
            or not isinstance(successes, int)
            or not 0 <= successes <= 16
            or not isinstance(completion, (int, float))
            or not math.isfinite(completion)
        ):
            raise ValueError(f"invalid per-condition result for {condition_id}")
        family = families[condition_id]
        exact_by_family[family]["successes"] += successes
        exact_by_family[family]["episodes"] += episodes
        by_condition[condition_id] = {
            "family": family,
            "successes": successes,
            "episodes": episodes,
            "mean_completion": float(completion),
        }
    overall_successes = sum(item["successes"] for item in by_condition.values())
    overall_episodes = sum(item["episodes"] for item in by_condition.values())
    summary_overall = record["summary"]["overall"]
    if (
        summary_overall.get("successes") != overall_successes
        or summary_overall.get("episodes") != overall_episodes
    ):
        raise ValueError("aggregate exact count differs from per-condition accounting")
    values = {
        "macro_completion": float(graded["macro_completion"]),
        "micro_p10": float(graded["micro"]["p10"]),
        "worst_condition_completion": float(graded["worst_condition_completion"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("evaluation contains non-finite graded aggregates")
    return {
        "exact": {
            "successes": overall_successes,
            "episodes": overall_episodes,
            "by_family": exact_by_family,
        },
        "graded": {
            **values,
            "by_family": {
                family: float(graded["by_family"][family]["macro_completion"])
                for family in exact_by_family
                if exact_by_family[family]["episodes"]
            },
        },
        "integrity": record["summary"]["integrity"],
        "by_condition": by_condition,
    }


def checkpoint_reward_gate(
    main: dict,
    capability: dict,
    thresholds: dict[str, int],
    core_ids: tuple[str, ...],
    capability_ids: tuple[str, ...],
) -> dict:
    exact = main["exact"]
    foundation = exact["by_family"]["foundation"]
    trench = exact["by_family"]["trench"]
    cell_passed = all(
        result["successes"] >= GATE_CELL for result in main["by_condition"].values()
    )
    retention_counts = {
        **{
            condition_id: main["by_condition"][condition_id]["successes"]
            for condition_id in core_ids
        },
        **{
            condition_id: capability["by_condition"][condition_id]["successes"]
            for condition_id in capability_ids
        },
    }
    if set(thresholds) != set(retention_counts):
        raise ValueError("qualified receipt retention set is not capability plus core")
    retention_passed = all(
        retention_counts[condition_id] >= thresholds[condition_id]
        for condition_id in thresholds
    )
    checks = {
        "overall_exact": exact["successes"] >= GATE_OVERALL,
        "foundation_exact": foundation["successes"] >= GATE_FOUNDATION,
        "trench_exact": trench["successes"] >= GATE_TRENCH,
        "every_main_condition": cell_passed,
        "capability_core_retention": retention_passed,
        "integrity": (
            main["integrity"].get("passed") is True
            and capability["integrity"].get("passed") is True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "overall_exact": exact,
            "foundation_exact": foundation,
            "trench_exact": trench,
            "minimum_main_cell_exact": min(
                result["successes"] for result in main["by_condition"].values()
            ),
            "retention_counts": retention_counts,
        },
    }


def build_reward_gate(
    history: list[dict],
    thresholds: dict[str, int],
    core_ids: tuple[str, ...],
    capability_ids: tuple[str, ...],
) -> dict:
    scheduled = [entry for entry in history if entry["reward_gate_scheduled"]]
    checkpoint_results = []
    for entry in scheduled:
        result = checkpoint_reward_gate(
            entry["panels"]["promotion"],
            entry["panels"]["capability_promotion"],
            thresholds,
            core_ids,
            capability_ids,
        )
        checkpoint_results.append(
            {
                "update": entry["update"],
                "checkpoint": entry["checkpoint"],
                **result,
            }
        )
    windows = []
    for end in range(GATE_CONSECUTIVE - 1, len(checkpoint_results)):
        window = checkpoint_results[end - GATE_CONSECUTIVE + 1 : end + 1]
        windows.append(
            {
                "updates": [result["update"] for result in window],
                "passed": all(result["passed"] for result in window),
            }
        )
    latest_window = windows[-1] if windows else None
    qualified = bool(latest_window and latest_window["passed"])
    selected_parent = (
        {
            "update": checkpoint_results[-1]["update"],
            **checkpoint_results[-1]["checkpoint"],
        }
        if qualified
        else None
    )
    return {
        "contract": {
            "panel": "promotion",
            "consecutive_scheduled_evaluations": GATE_CONSECUTIVE,
            "scheduled_interval_updates": EVALUATION_INTERVAL,
            "overall_exact_min": [GATE_OVERALL, 720],
            "foundation_exact_min": [GATE_FOUNDATION, 384],
            "trench_exact_min": [GATE_TRENCH, 336],
            "every_main_condition_exact_min": [GATE_CELL, 16],
            "capability_core_thresholds": thresholds,
            "integrity_required": True,
        },
        "scheduled_updates": [result["update"] for result in checkpoint_results],
        "checkpoint_results": checkpoint_results,
        "windows": windows,
        "latest_window": latest_window,
        "qualified_for_reward_curriculum": qualified,
        "selected_dense_parent": selected_parent,
        "reward_launched": False,
    }


def write_csv_artifacts(output_dir: Path, history: list[dict]) -> None:
    with (output_dir / "history.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "update",
                "checkpoint_sha256",
                "selection",
                "reward_gate_scheduled",
                "panel",
                "exact_successes",
                "episodes",
                "foundation_exact",
                "trench_exact",
                "macro_completion",
                "micro_p10",
                "worst_condition_completion",
                "integrity_passed",
            ]
        )
        for entry in history:
            for panel_name, panel in entry["panels"].items():
                exact = panel["exact"]
                writer.writerow(
                    [
                        entry["update"],
                        entry["checkpoint"]["sha256"],
                        entry["selection"],
                        str(entry["reward_gate_scheduled"]).lower(),
                        panel_name,
                        exact["successes"],
                        exact["episodes"],
                        exact["by_family"]["foundation"]["successes"],
                        exact["by_family"]["trench"]["successes"],
                        panel["graded"]["macro_completion"],
                        panel["graded"]["micro_p10"],
                        panel["graded"]["worst_condition_completion"],
                        str(panel["integrity"].get("passed") is True).lower(),
                    ]
                )
    with (output_dir / "per_condition.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "update",
                "checkpoint_sha256",
                "panel",
                "condition_id",
                "family",
                "exact_successes",
                "episodes",
                "mean_completion",
            ]
        )
        for entry in history:
            for panel_name, panel in entry["panels"].items():
                for condition_id, result in sorted(panel["by_condition"].items()):
                    writer.writerow(
                        [
                            entry["update"],
                            entry["checkpoint"]["sha256"],
                            panel_name,
                            condition_id,
                            result["family"],
                            result["successes"],
                            result["episodes"],
                            result["mean_completion"],
                        ]
                    )


def write_markdown(output_dir: Path, history: list[dict], gate: dict) -> None:
    lines = [
        "# V8 continuation leaderboard",
        "",
        "Deterministic fixed-bank evaluation. Main and capability panels are separate.",
        "",
        "| Update | Selection | Promotion exact | Promotion macro | Development exact | Development macro | Integrity |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for entry in history:
        promotion = entry["panels"]["promotion"]
        development = entry["panels"]["development"]
        integrity = all(
            panel["integrity"].get("passed") is True
            for panel in entry["panels"].values()
        )
        lines.append(
            f"| {entry['update']} | {entry['selection']} | "
            f"{promotion['exact']['successes']}/{promotion['exact']['episodes']} | "
            f"{promotion['graded']['macro_completion']:.3f} | "
            f"{development['exact']['successes']}/{development['exact']['episodes']} | "
            f"{development['graded']['macro_completion']:.3f} | "
            f"{'yes' if integrity else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Dense-to-reward gate",
            "",
            f"Qualified: **{'yes' if gate['qualified_for_reward_curriculum'] else 'no'}**.",
            "No reward experiment was launched by this evaluator.",
            "",
            "See `per_condition.csv` for every condition at every evaluated checkpoint.",
        ]
    )
    (output_dir / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")


def summarize(
    *,
    inventory_path: Path,
    qualified_receipt: Path,
    bank_root: Path,
    promotion_path: Path,
    development_path: Path,
    capability_promotion_path: Path,
    capability_development_path: Path,
    output_dir: Path,
) -> dict:
    inventory = validate_inventory(inventory_path.resolve(), qualified_receipt)
    qualified, qualified_info = continuation_contract.inspect_receipt(
        qualified_receipt.resolve()
    )
    bank = stage_gate.load_bank_contract(bank_root.resolve())
    sampling = stage_gate.stage_sampling_contract(bank, "full")
    paths = {
        "promotion": promotion_path.resolve(),
        "development": development_path.resolve(),
        "capability_promotion": capability_promotion_path.resolve(),
        "capability_development": capability_development_path.resolve(),
    }
    records = {
        "promotion": validate_evaluation_records(
            path=paths["promotion"],
            panel_group="main",
            split="promotion",
            inventory=inventory,
            bank=bank,
            sampling=sampling,
        ),
        "development": validate_evaluation_records(
            path=paths["development"],
            panel_group="main",
            split="development",
            inventory=inventory,
            bank=bank,
            sampling=sampling,
        ),
        "capability_promotion": validate_evaluation_records(
            path=paths["capability_promotion"],
            panel_group="capability",
            split="promotion",
            inventory=inventory,
            bank=bank,
            sampling=sampling,
        ),
        "capability_development": validate_evaluation_records(
            path=paths["capability_development"],
            panel_group="capability",
            split="development",
            inventory=inventory,
            bank=bank,
            sampling=sampling,
        ),
    }
    fingerprints = {
        json.dumps(
            record["treatment_fingerprint"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for panel in records.values()
        for record in panel
    }
    if len(fingerprints) != 1:
        raise ValueError("four-panel evaluation changed treatment identity")
    fingerprint = records["promotion"][0]["treatment_fingerprint"]
    if (
        fingerprint["contract"]["run"]["name"]
        != inventory["run_contract"]["source_treatment_name"]
    ):
        raise ValueError("evaluation treatment name differs from source candidate")
    checkpoint_validation = validate_selected_checkpoint_states(
        inventory,
        records["promotion"],
        sampling,
    )

    families = bank["family_by_condition"]
    selected = inventory["selected_checkpoints"]
    history = []
    for index, checkpoint in enumerate(selected):
        panels = {
            "promotion": panel_snapshot(
                records["promotion"][index], bank["main_ids"], families
            ),
            "development": panel_snapshot(
                records["development"][index], bank["main_ids"], families
            ),
            "capability_promotion": panel_snapshot(
                records["capability_promotion"][index],
                stage_gate.CAPABILITY_IDS,
                families,
            ),
            "capability_development": panel_snapshot(
                records["capability_development"][index],
                stage_gate.CAPABILITY_IDS,
                families,
            ),
        }
        history.append(
            {
                "update": checkpoint["update"],
                "checkpoint": {
                    "path": checkpoint["path"],
                    "sha256": checkpoint["sha256"],
                },
                "selection": checkpoint["selection"],
                "reward_gate_scheduled": checkpoint["reward_gate_scheduled"],
                "panels": panels,
            }
        )
    thresholds = qualified["retention"]["frozen_thresholds"]
    gate = build_reward_gate(
        history,
        thresholds,
        bank["core_ids"],
        stage_gate.CAPABILITY_IDS,
    )
    leaderboard = {
        "schema": LEADERBOARD_SCHEMA,
        "generated_at_utc": utc_now(),
        "arm": inventory["arm"],
        "seed": inventory["seed"],
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "terra_baselines_revision": qualified_info["terra_baselines_revision"],
        "completion_contract": stage_gate.COMPLETION_CONTRACT,
        "policy_mode": "deterministic",
        "treatment_fingerprint": fingerprint,
        "selected_checkpoint_validation": checkpoint_validation,
        "history": history,
        "dense_to_reward_gate": gate,
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "leaderboard.json", leaderboard)
    write_csv_artifacts(output_dir, history)
    write_markdown(output_dir, history, gate)
    artifact_names = (
        "leaderboard.json",
        "history.csv",
        "per_condition.csv",
        "LEADERBOARD.md",
    )
    receipt = {
        "schema": GATE_SCHEMA,
        "generated_at_utc": utc_now(),
        "passed": True,
        "qualified_for_reward_curriculum": gate["qualified_for_reward_curriculum"],
        "reward_launched": False,
        "selected_dense_parent": gate["selected_dense_parent"],
        "gate": gate,
        "arm": inventory["arm"],
        "seed": inventory["seed"],
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "terra_baselines_revision": qualified_info["terra_baselines_revision"],
        "bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "training_mixture_sha256": stage_gate.TRAINING_MIXTURE_SHA256,
        "treatment_fingerprint": fingerprint,
        "selected_checkpoint_validation": {
            "count": len(checkpoint_validation),
            "all_passed": True,
        },
        "inputs": {
            "qualified_full_receipt": {
                "path": str(qualified_receipt.resolve()),
                "sha256": sha256_file(qualified_receipt.resolve()),
            },
            "checkpoint_inventory": {
                "path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path.resolve()),
            },
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
        "artifacts": {name: sha256_file(output_dir / name) for name in artifact_names},
    }
    write_json(output_dir / "dense_reward_gate_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--qualified-receipt", type=Path, required=True)
    inventory.add_argument("--run-dir", type=Path, required=True)
    inventory.add_argument("--run-contract", type=Path, required=True)
    inventory.add_argument("--job-id", required=True)
    inventory.add_argument("--job-state", required=True)
    inventory.add_argument("--job-exit-code", required=True)
    inventory.add_argument("--job-partition", required=True)
    inventory.add_argument("--evaluator-job-id", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--inventory", type=Path, required=True)
    summary.add_argument("--qualified-receipt", type=Path, required=True)
    summary.add_argument("--bank-root", type=Path, required=True)
    summary.add_argument("--promotion", type=Path, required=True)
    summary.add_argument("--development", type=Path, required=True)
    summary.add_argument("--capability-promotion", type=Path, required=True)
    summary.add_argument("--capability-development", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        inventory = build_inventory(
            qualified_receipt=args.qualified_receipt,
            run_dir=args.run_dir,
            run_contract=args.run_contract,
            job_id=args.job_id,
            job_state=args.job_state,
            job_exit_code=args.job_exit_code,
            job_partition=args.job_partition,
            evaluator_job_id=args.evaluator_job_id,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, inventory)
        for checkpoint in inventory["selected_checkpoints"]:
            print(checkpoint["path"])
    else:
        receipt = summarize(
            inventory_path=args.inventory,
            qualified_receipt=args.qualified_receipt,
            bank_root=args.bank_root,
            promotion_path=args.promotion,
            development_path=args.development,
            capability_promotion_path=args.capability_promotion,
            capability_development_path=args.capability_development,
            output_dir=args.output_dir,
        )
        print(
            "V8_CONTINUATION_TAIL "
            f"qualified_for_reward={str(receipt['qualified_for_reward_curriculum']).lower()} "
            f"receipt={args.output_dir / 'dense_reward_gate_receipt.json'}"
        )


if __name__ == "__main__":
    main()
