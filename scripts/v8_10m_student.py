#!/usr/bin/env python3
"""Frozen architecture and teacher gate for the deferred V8 10M experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.euler_v8_deep_xattn_v1 import continuation_contract
from scripts.euler_v8_deep_xattn_v1 import continuation_tail_eval
from scripts.euler_v8_deep_xattn_v1 import stage_gate

TEACHER_ARM = "G-DEEP-XATTN-V8-DENSE-WARM"
TEACHER_RECEIPT_SCHEMA = "terra_v8_dense_reward_gate_v1"
TEACHER_ARCHITECTURE = {
    "model_size": "medium",
    "model_core": "mlp",
    "map_encoder": "resnet_spatial_8x8_se_xattn",
    "encoder_compute_dtype": "bfloat16",
    "attention_compute_dtype": "float32",
    "token_mixer_residual_init_scale": 0.0,
    "critic_hidden_dims": [512, 256],
    "resnet_stage_channels": [24, 48, 64, 96],
    "resnet_blocks_per_stage": [2, 2, 3, 3],
    "loaded_max": 100,
}
TARGET_ARCHITECTURE = {
    **TEACHER_ARCHITECTURE,
    "resnet_stage_channels": [64, 128, 192, 256],
}
TARGET_PARAMETER_COUNT = 10_257_209
GATE_OVERALL = (576, 720)
GATE_FOUNDATION = (308, 384)
GATE_TRENCH = (269, 336)
GATE_CELL = (10, 16)
GATE_CAPABILITY_CELL = (12, 16)
GATE_CONSECUTIVE = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, expected_type: type) -> dict | list:
    value = json.loads(path.read_text())
    if not isinstance(value, expected_type):
        raise ValueError(f"{path}: expected {expected_type.__name__}")
    return value


def canonical_architecture(value: dict) -> dict:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.items()
    }


def validate_architecture(observed: dict, expected: dict, label: str) -> None:
    observed = canonical_architecture(observed)
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(
                f"{label} architecture {key} must be {value!r}, "
                f"got {observed.get(key)!r}"
            )


def validate_fingerprint(receipt: dict, bank: dict, sampling: dict, seed: int) -> None:
    """Validate the complete frozen treatment, not just its architecture."""
    stage_gate._validate_treatment(
        {"treatment_fingerprint": receipt.get("treatment_fingerprint")},
        bank,
        "full",
        TEACHER_ARM,
        seed,
        sampling,
    )


def validate_promotion_gate(receipt: dict) -> dict:
    gate = receipt.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("teacher receipt lacks its gate")
    expected_contract = {
        "consecutive_scheduled_evaluations": GATE_CONSECUTIVE,
        "overall_exact_min": list(GATE_OVERALL),
        "foundation_exact_min": list(GATE_FOUNDATION),
        "trench_exact_min": list(GATE_TRENCH),
        "every_main_condition_exact_min": list(GATE_CELL),
        "integrity_required": True,
    }
    contract = gate.get("contract", {})
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"teacher promotion gate {key} changed")
    latest = gate.get("latest_window")
    if not isinstance(latest, dict) or latest.get("passed") is not True:
        raise ValueError("teacher latest promotion window did not pass")
    updates = latest.get("updates")
    if not isinstance(updates, list) or len(updates) != GATE_CONSECUTIVE:
        raise ValueError("teacher gate does not contain three scheduled passes")
    results = gate.get("checkpoint_results")
    if not isinstance(results, list) or len(results) < GATE_CONSECUTIVE:
        raise ValueError("teacher gate lacks checkpoint results")
    latest_results = results[-GATE_CONSECUTIVE:]
    if [entry.get("update") for entry in latest_results] != updates:
        raise ValueError("teacher latest gate window is not the latest result window")
    if not all(entry.get("passed") is True for entry in latest_results):
        raise ValueError("teacher latest scheduled checkpoints did not all pass")
    parent = gate.get("selected_dense_parent")
    if not isinstance(parent, dict) or parent != receipt.get("selected_dense_parent"):
        raise ValueError("teacher selected parent changed between gate and receipt")
    if parent.get("update") != updates[-1]:
        raise ValueError("teacher selected parent is not the latest passing checkpoint")
    return parent


def receipt_input_path(receipt: dict, name: str) -> Path:
    identity = receipt.get("inputs", {}).get(name)
    if not isinstance(identity, dict):
        raise ValueError(f"teacher receipt lacks input {name}")
    path = Path(str(identity.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"teacher input {name} is unavailable")
    if sha256_file(path) != identity.get("sha256"):
        raise ValueError(f"teacher input {name} hash changed")
    return path.resolve()


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: manifest must contain JSON objects")
    rows.sort(key=lambda row: int(row["slot_index"]))
    if [row["slot_index"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: manifest slots are not contiguous")
    return rows


def validate_per_map_accounting(
    record: dict,
    manifest_rows: list[dict],
    condition_ids: tuple[str, ...],
    family_by_condition: dict[str, str],
) -> dict:
    per_map = record.get("per_map")
    if not isinstance(per_map, list) or len(per_map) != len(manifest_rows):
        raise ValueError("teacher evaluation does not exactly enumerate its manifest")
    by_condition = {
        condition_id: {
            "family": family_by_condition[condition_id],
            "successes": 0,
            "episodes": 0,
        }
        for condition_id in condition_ids
    }
    for expected, observed in zip(manifest_rows, per_map):
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "teacher evaluation per-map identity differs from manifest"
            )
        success = observed.get("success")
        completion = observed.get("terminal_absolute")
        if (
            not isinstance(success, bool)
            or isinstance(completion, bool)
            or not isinstance(completion, (int, float))
            or not np.isfinite(completion)
            or not -1e-6 <= completion <= 1.0 + 1e-6
            or success != bool(np.isclose(completion, 1.0, rtol=0.0, atol=1e-6))
            or observed.get("integrity_failure") is not False
        ):
            raise ValueError("teacher evaluation contains an invalid per-map result")
        condition_id = expected["primary_cell"]
        if condition_id not in by_condition:
            raise ValueError("teacher evaluation contains an unexpected condition")
        by_condition[condition_id]["episodes"] += 1
        by_condition[condition_id]["successes"] += int(success)
    if any(result["episodes"] != 16 for result in by_condition.values()):
        raise ValueError("teacher evaluation must contain 16 episodes per condition")

    summary = record.get("summary", {})
    if summary.get("integrity", {}).get("passed") is not True:
        raise ValueError("teacher evaluation integrity failed")
    declared_cells = summary.get("by_primary_cell", {})
    if set(declared_cells) != set(by_condition):
        raise ValueError("teacher declared condition set differs from the manifest")
    for condition_id, recomputed in by_condition.items():
        declared = declared_cells[condition_id]
        if (
            declared.get("successes") != recomputed["successes"]
            or declared.get("episodes") != recomputed["episodes"]
        ):
            raise ValueError("teacher declared cell score differs from per-map results")

    exact_by_family = {
        "foundation": {"successes": 0, "episodes": 0},
        "trench": {"successes": 0, "episodes": 0},
    }
    for result in by_condition.values():
        family = result["family"]
        exact_by_family[family]["successes"] += result["successes"]
        exact_by_family[family]["episodes"] += result["episodes"]
    overall = {
        "successes": sum(result["successes"] for result in by_condition.values()),
        "episodes": sum(result["episodes"] for result in by_condition.values()),
    }
    declared_overall = summary.get("overall", {})
    if any(declared_overall.get(key) != value for key, value in overall.items()):
        raise ValueError("teacher declared overall score differs from per-map results")
    declared_families = summary.get("by_family", {})
    for family, recomputed in exact_by_family.items():
        declared = declared_families.get(family, {})
        if any(declared.get(key) != value for key, value in recomputed.items()):
            raise ValueError(
                "teacher declared family score differs from per-map results"
            )
    return {
        "exact": {**overall, "by_family": exact_by_family},
        "by_condition": by_condition,
        "integrity": summary["integrity"],
    }


def validate_portable_evaluation_records(
    *,
    path: Path,
    panel_group: str,
    split: str,
    inventory: dict,
    bank: dict,
    sampling: dict,
) -> tuple[list[dict], list[dict]]:
    records = read_json(path, list)
    selected = inventory["selected_checkpoints"]
    if len(records) != len(selected):
        raise ValueError(f"{path}: evaluation does not cover every checkpoint")
    expected_panel = stage_gate.panel_contract(bank, panel_group, split)
    manifest_rows = read_manifest(Path(expected_panel["manifest"]))
    condition_ids = (
        stage_gate.CAPABILITY_IDS if panel_group == "capability" else bank["main_ids"]
    )
    expected_bank = {
        "schema": "terra_curriculum_loader_bank_v1",
        "terra_revision": stage_gate.TERRA_REVISION,
        "environment_protocol_sha256": bank["environment_protocol_sha256"],
        "source_registry_sha256": bank["source_registry_sha256"],
        "diagnostic_control": False,
        "diagnostic_contract_sha256": None,
    }
    reference_fingerprint = None
    snapshots = []
    for record, checkpoint in zip(records, selected):
        expected = {
            "schema": stage_gate.EVAL_SCHEMA,
            "completion_contract": stage_gate.COMPLETION_CONTRACT,
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_update": checkpoint["update"],
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
                raise ValueError(f"{path}: {field} changed")
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
        snapshots.append(
            validate_per_map_accounting(
                record,
                manifest_rows,
                condition_ids,
                bank["family_by_condition"],
            )
        )
    stage_gate.validate_panel_conditions(records, condition_ids)
    return records, snapshots


def validate_selected_development_gate(main: dict, capability: dict) -> None:
    exact = main["exact"]
    if exact["episodes"] != GATE_OVERALL[1] or exact["successes"] < GATE_OVERALL[0]:
        raise ValueError("teacher development overall exact gate failed")
    for family, required in (
        ("foundation", GATE_FOUNDATION),
        ("trench", GATE_TRENCH),
    ):
        observed = exact["by_family"][family]
        if observed["episodes"] != required[1] or observed["successes"] < required[0]:
            raise ValueError(f"teacher development {family} exact gate failed")
    if any(
        result["successes"] < GATE_CELL[0] for result in main["by_condition"].values()
    ):
        raise ValueError("teacher development per-condition exact gate failed")
    if set(capability["by_condition"]) != set(stage_gate.CAPABILITY_IDS) or any(
        result["episodes"] != GATE_CAPABILITY_CELL[1]
        or result["successes"] < GATE_CAPABILITY_CELL[0]
        for result in capability["by_condition"].values()
    ):
        raise ValueError("teacher capability-development exact gate failed")


def validate_external_evidence(receipt: dict, parent: dict, bank_root: Path) -> dict:
    qualified_path = receipt_input_path(receipt, "qualified_full_receipt")
    inventory_path = receipt_input_path(receipt, "checkpoint_inventory")
    paths = {
        name: receipt_input_path(receipt, name)
        for name in (
            "promotion",
            "development",
            "capability_promotion",
            "capability_development",
        )
    }
    qualified, qualified_info = continuation_contract.inspect_receipt(qualified_path)
    if qualified_info["arm"] != TEACHER_ARM:
        raise ValueError("teacher continuation source is not the deep+xattn arm")
    inventory = continuation_tail_eval.validate_inventory(
        inventory_path, qualified_path
    )
    if inventory.get("arm") != TEACHER_ARM or inventory.get("seed") != (
        qualified_info["seed"]
    ):
        raise ValueError("teacher inventory arm or seed changed")

    bank = stage_gate.load_bank_contract(bank_root.resolve())
    sampling = stage_gate.stage_sampling_contract(bank, "full")
    validate_fingerprint(receipt, bank, sampling, inventory["seed"])
    records = {}
    snapshots = {}
    for name, panel_group, split in (
        ("promotion", "main", "promotion"),
        ("development", "main", "development"),
        ("capability_promotion", "capability", "promotion"),
        ("capability_development", "capability", "development"),
    ):
        records[name], snapshots[name] = validate_portable_evaluation_records(
            path=paths[name],
            panel_group=panel_group,
            split=split,
            inventory=inventory,
            bank=bank,
            sampling=sampling,
        )
        if any(
            record.get("treatment_fingerprint") != receipt.get("treatment_fingerprint")
            for record in records[name]
        ):
            raise ValueError(f"teacher {name} treatment differs from receipt")

    checkpoint_validation = continuation_tail_eval.validate_selected_checkpoint_states(
        inventory, records["promotion"], sampling
    )
    declared_validation = receipt.get("selected_checkpoint_validation", {})
    if declared_validation != {
        "count": len(checkpoint_validation),
        "all_passed": True,
    }:
        raise ValueError("teacher checkpoint-validation receipt changed")

    history = []
    for index, checkpoint in enumerate(inventory["selected_checkpoints"]):
        history.append(
            {
                "update": checkpoint["update"],
                "checkpoint": {
                    "path": checkpoint["path"],
                    "sha256": checkpoint["sha256"],
                },
                "selection": checkpoint["selection"],
                "reward_gate_scheduled": checkpoint["reward_gate_scheduled"],
                "panels": {
                    "promotion": snapshots["promotion"][index],
                    "capability_promotion": snapshots["capability_promotion"][index],
                },
            }
        )
    recomputed_gate = continuation_tail_eval.build_reward_gate(
        history,
        qualified["retention"]["frozen_thresholds"],
        bank["core_ids"],
        stage_gate.CAPABILITY_IDS,
    )
    if recomputed_gate != receipt.get("gate"):
        raise ValueError("teacher reward gate differs from fixed evaluation evidence")

    matches = [
        index
        for index, checkpoint in enumerate(inventory["selected_checkpoints"])
        if {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "update": checkpoint["update"],
        }
        == parent
    ]
    if len(matches) != 1:
        raise ValueError("teacher selected parent differs from checkpoint inventory")
    selected_index = matches[0]
    validate_selected_development_gate(
        snapshots["development"][selected_index],
        snapshots["capability_development"][selected_index],
    )
    return {
        "seed": inventory["seed"],
        "selected_checkpoints_validated": len(checkpoint_validation),
        "bank_protocol_validated": True,
        "full_sampler_state_validated": True,
        "four_panel_identity_validated": True,
        "promotion_gate_recomputed": True,
    }


def inspect_teacher(receipt_path: Path, bank_root: Path) -> dict:
    receipt_path = receipt_path.resolve()
    receipt = read_json(receipt_path, dict)
    expected = {
        "schema": TEACHER_RECEIPT_SCHEMA,
        "passed": True,
        "qualified_for_reward_curriculum": True,
        "reward_launched": False,
        "arm": TEACHER_ARM,
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "training_mixture_sha256": stage_gate.TRAINING_MIXTURE_SHA256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"teacher receipt {key} must be {value!r}")
    parent = validate_promotion_gate(receipt)
    evidence = validate_external_evidence(receipt, parent, bank_root.resolve())
    return {
        "schema": "terra_v8_10m_teacher_inspection_v1",
        "passed": True,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "teacher_arm": TEACHER_ARM,
        "teacher_checkpoint": parent["path"],
        "teacher_checkpoint_sha256": parent["sha256"],
        "teacher_update": parent["update"],
        "same_distribution": True,
        "promotion_latest_three_passed": True,
        "development_gate_verified": True,
        "evidence": evidence,
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "target_architecture": TARGET_ARCHITECTURE,
    }


def probe_parameter_count() -> dict:
    import contextlib
    import io

    import jax

    from scripts.grow_checkpoint import _derive_stage_spec_from_params
    from scripts.grow_checkpoint import _dummy_env
    from scripts.grow_checkpoint import build_target_config
    from scripts.grow_checkpoint import grow_params
    from terra.actions import TrackedAction
    from utils.models import get_model_ready

    source = {
        "clip_action_maps": True,
        "maps_net_normalization_bounds": (-10, 10),
        "local_map_normalization_bounds": (-16, 16),
        "loaded_max": 100,
        "num_prev_actions": 5,
        **TEACHER_ARCHITECTURE,
    }
    teacher = build_target_config(source, {})
    target = build_target_config(
        source,
        {"resnet_stage_channels": tuple(TARGET_ARCHITECTURE["resnet_stage_channels"])},
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, teacher_params = get_model_ready(
            jax.random.PRNGKey(0), teacher, _dummy_env(64, TrackedAction)
        )
        _, target_params = get_model_ready(
            jax.random.PRNGKey(0), target, _dummy_env(64, TrackedAction)
        )
    grown_params, report = grow_params(
        teacher_params,
        target_params,
        _derive_stage_spec_from_params(teacher_params),
        _derive_stage_spec_from_params(target_params),
    )
    teacher_count = int(
        sum(value.size for value in jax.tree_util.tree_leaves(teacher_params))
    )
    observed = int(sum(value.size for value in jax.tree_util.tree_leaves(grown_params)))
    if observed != TARGET_PARAMETER_COUNT:
        raise ValueError(
            f"10M architecture must contain {TARGET_PARAMETER_COUNT:,} parameters, "
            f"got {observed:,}"
        )
    continuation_contract._require_finite_tree(grown_params, "grown 10M parameters")
    categories = {
        name: sum(entry["category"] == name for entry in report)
        for name in sorted({entry["category"] for entry in report})
    }
    if sum(categories.values()) != len(jax.tree_util.tree_leaves(grown_params)):
        raise ValueError("10M growth report does not cover every parameter leaf")
    return {
        "schema": "terra_v8_10m_parameter_probe_v1",
        "passed": True,
        "teacher_parameter_count": teacher_count,
        "parameter_count": observed,
        "architecture": TARGET_ARCHITECTURE,
        "growth_categories": categories,
        "finite_grown_parameters": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe")
    inspect = commands.add_parser("inspect-teacher")
    inspect.add_argument("--receipt", type=Path, required=True)
    inspect.add_argument("--bank-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        result = probe_parameter_count()
    else:
        result = inspect_teacher(args.receipt, args.bank_root)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
