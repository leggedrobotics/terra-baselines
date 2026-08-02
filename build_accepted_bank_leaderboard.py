#!/usr/bin/env python3
"""Build deterministic accepted-bank leaderboards from fixed evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

PANELS = ("promotion", "development")
P5_ARMS = (
    "F-ANCHOR",
    "F-SPECIALIST",
    "T-ANCHOR",
    "T-SPECIALIST",
    "G-UNIFORM",
    "G-ADAPTIVE",
)
P5_CAMPAIGN_SHA256 = "f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa"
P5_TERRA_REVISION = "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
P5_BASELINES_REVISION = "4b34668d105cf44118186d5ce49d1b78cd19a8e5"
PAIRED_CONTROLS = {
    "G-ADAPTIVE": "G-UNIFORM",
    "G-DEEP-ADAPTIVE-WARM": "G-MEDIUM-ADAPTIVE-WARM",
    "G-MEDIUM-UNIFORM-WARM": "G-MEDIUM-ADAPTIVE-WARM",
}
NEAR_COMPLETE_THRESHOLD = 0.95
FOLLOWUP_CAMPAIGN_FIELDS = (
    "phase",
    "seed",
    "updates",
    "terra_revision",
    "terra_baselines_revision",
    "parent_checkpoint_sha256",
    "teacher_checkpoint_sha256",
    "initialization",
)
MANIFEST_IDENTITY_FIELDS = (
    "candidate_sample_index",
    "environment_protocol_sha256",
    "episode_id",
    "family",
    "identity_slot_multiplicity",
    "map_id",
    "pair_slot_id",
    "primary_cell",
    "reset_seed",
    "scenario_id",
    "slot_index",
    "slot_weight",
    "source_id",
    "split",
    "stratum",
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"invalid receipt line in {path}: {line!r}")
        values[key] = value
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def followup_campaign_sha256(contract: dict[str, str]) -> str:
    missing = [field for field in FOLLOWUP_CAMPAIGN_FIELDS if not contract.get(field)]
    if missing:
        raise ValueError(f"follow-up run contract lacks campaign fields: {missing}")
    shared_contract = {field: contract[field] for field in FOLLOWUP_CAMPAIGN_FIELDS}
    return hashlib.sha256(
        json.dumps(shared_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(
            f"{label} mismatch: recomputed {actual} != recorded {expected}"
        )


def classify_condition(condition_id: str, family: str, branch_depth: str) -> dict:
    tokens = condition_id.split("-")
    expected_prefix = "fnd" if family == "foundation" else "trn"
    if tokens[0] != expected_prefix:
        raise ValueError(f"family/condition mismatch: {family} / {condition_id}")

    if family == "foundation":
        geometries = ("slab-lg", "strips", "slab", "proc")
        dumps = ("ring3x", "apron", "side1", "split")
    elif family == "trench":
        geometries = ("straight", "tee", "seg2", "seg3", "net3", "net4")
        dumps = ("side1", "side2", "altsides")
    else:
        raise ValueError(f"unknown family: {family}")

    body = condition_id[len(expected_prefix) + 1 :]
    geometry = next(
        (item for item in geometries if body == item or body.startswith(item + "-")),
        None,
    )
    if geometry is None:
        raise ValueError(f"cannot parse geometry from {condition_id}")
    remainder = body[len(geometry) :].lstrip("-")
    dump = next(
        (
            item
            for item in dumps
            if remainder == item or remainder.startswith(item + "-")
        ),
        None,
    )
    if dump is None:
        raise ValueError(f"cannot parse dump layout from {condition_id}")
    extras = remainder[len(dump) :].lstrip("-").split("-") if remainder != dump else []

    capacity = "generous"
    distance = "unspec"
    site = "clean"
    scale = "std"
    for token in extras:
        if token in {"c3x", "c2x", "c1p6", "c1p2", "tight"}:
            capacity = token
        elif token in {"near", "d12", "d16"}:
            distance = token
        elif token in {"obj1", "obj", "road"}:
            site = token
        elif token == "s":
            scale = token
        else:
            raise ValueError(f"unknown condition token {token!r} in {condition_id}")

    if branch_depth == "Anchor":
        axis = "anchor"
    elif branch_depth == "Composed":
        axis = "composed"
    elif capacity != "generous":
        axis = "capacity"
    elif distance != "unspec":
        axis = "distance"
    elif site != "clean":
        axis = "site"
    elif (family == "foundation" and dump in {"side1", "split"}) or dump == "altsides":
        axis = "dump_layout"
    else:
        axis = "geometry"

    return {
        "geometry": geometry,
        "dump_layout": dump,
        "capacity": capacity,
        "distance": distance,
        "site": site,
        "scale": scale,
        "factor_axis": axis,
    }


def load_bank(bank_root: Path) -> tuple[dict[str, dict], dict[str, list[dict]], dict]:
    dataset = read_json(bank_root / "dataset.json")
    if dataset.get("schema") != "terra_curriculum_loader_bank_v1":
        raise ValueError("unsupported accepted-bank schema")

    conditions = {}
    for entry in dataset["train"]:
        condition_id = entry["condition_id"]
        if condition_id in conditions:
            raise ValueError(f"duplicate training condition {condition_id}")
        descriptor = {
            "condition_id": condition_id,
            "family": entry["family"],
            "branch_depth": entry["branch_depth"],
            "train_map_count": int(entry["map_count"]),
        }
        descriptor.update(
            classify_condition(condition_id, entry["family"], entry["branch_depth"])
        )
        conditions[condition_id] = descriptor

    panel_manifests = {}
    panel_datasets = {}
    protocol_hashes = set()
    for panel in PANELS:
        rows = read_jsonl(bank_root / panel / "manifest.jsonl")
        panel_dataset = read_json(bank_root / panel / "dataset.json")
        panel_datasets[panel] = panel_dataset
        if len(rows) != int(panel_dataset["slot_count"]):
            raise ValueError(f"{panel} manifest count does not match dataset")
        counts = Counter(row["primary_cell"] for row in rows)
        if set(counts) != set(conditions):
            raise ValueError(f"{panel} condition support differs from train support")
        for row in rows:
            condition = conditions[row["primary_cell"]]
            if row["family"] != condition["family"] or row["split"] != panel:
                raise ValueError(f"invalid {panel} manifest row {row['episode_id']}")
            protocol_hashes.add(row["environment_protocol_sha256"])
        panel_manifests[panel] = rows

    if len(protocol_hashes) != 1:
        raise ValueError("accepted-bank panels do not share one environment protocol")
    if panel_datasets["promotion"] != panel_datasets["development"]:
        raise ValueError("promotion/development dataset identities differ")
    bank_identity = {
        "schema": dataset["schema"],
        "environment_protocol_sha256": protocol_hashes.pop(),
        "source_registry_sha256": read_json(bank_root / "promotion" / "dataset.json")[
            "source_registry_sha256"
        ],
    }
    return conditions, panel_manifests, bank_identity


def architecture_label(architecture: dict) -> str:
    critic = "-".join(
        str(value) for value in architecture.get("critic_hidden_dims", ())
    )
    fields = [
        str(architecture.get("model_size", "unknown")),
        str(architecture.get("map_encoder", "unknown")),
        str(architecture.get("model_core", "unknown")),
        f"critic-{critic or 'unknown'}",
    ]
    stage_channels = architecture.get("resnet_stage_channels")
    stage_blocks = architecture.get("resnet_blocks_per_stage")
    if stage_channels is not None:
        fields.append("channels-" + "x".join(str(value) for value in stage_channels))
    if stage_blocks is not None:
        fields.append("blocks-" + "x".join(str(value) for value in stage_blocks))
    return ":".join(fields)


def arm_metadata(arm: str, receipt: dict, contract: dict, treatment: dict) -> dict:
    defaults = {
        "F-ANCHOR": ("foundation", "anchor", "uniform"),
        "F-SPECIALIST": ("foundation", "all", "uniform"),
        "T-ANCHOR": ("trench", "anchor", "uniform"),
        "T-SPECIALIST": ("trench", "all", "uniform"),
        "G-UNIFORM": ("generalist", "all", "uniform"),
        "G-ADAPTIVE": ("generalist", "all", "adaptive"),
    }
    if arm in defaults:
        if (
            receipt["campaign_sha256"] != P5_CAMPAIGN_SHA256
            or receipt["terra_revision"] != P5_TERRA_REVISION
            or receipt["terra_baselines_revision"] != P5_BASELINES_REVISION
        ):
            raise ValueError(
                f"hard-coded P5 metadata used outside the frozen P5 campaign: {arm}"
            )
        primary_family, support, sampler = defaults[arm]
    else:
        required = (
            "primary_family",
            "training_support",
            "condition_sampler",
            "architecture",
            "accepted_bank_arm",
            "parent_checkpoint_sha256",
        )
        missing = [field for field in required if not contract.get(field)]
        if missing:
            raise ValueError(f"follow-up arm {arm} lacks run metadata: {missing}")
        primary_family = contract["primary_family"]
        support = contract["training_support"]
        sampler = contract["condition_sampler"]
    treatment_contract = treatment.get("contract", {})
    recorded_architecture = treatment_contract.get("architecture")
    recorded_sampler = treatment_contract.get("sampler", {}).get("rule")
    if not isinstance(recorded_architecture, dict) or not recorded_sampler:
        raise ValueError(
            f"{arm} evaluation lacks architecture/sampler treatment metadata"
        )
    if sampler != recorded_sampler:
        raise ValueError(f"{arm} sampler receipt/evaluation mismatch")
    architecture = architecture_label(recorded_architecture)
    if arm not in defaults and contract["architecture"] != architecture:
        raise ValueError(f"{arm} architecture receipt/evaluation mismatch")
    global_transitions = contract.get("global_transitions")
    if global_transitions is None:
        ppo = treatment_contract.get("ppo", {})
        required = ("num_devices", "num_envs_per_device", "num_steps")
        missing = [field for field in required if field not in ppo]
        if missing:
            raise ValueError(
                f"{arm} cannot derive global transitions: missing PPO {missing}"
            )
        global_transitions = (
            int(contract["updates"])
            * int(ppo["num_devices"])
            * int(ppo["num_envs_per_device"])
            * int(ppo["num_steps"])
        )
    return {
        "arm": arm,
        "campaign_sha256": receipt["campaign_sha256"],
        "seed": int(contract["seed"]),
        "global_transitions": int(global_transitions),
        "primary_family": contract.get("primary_family", primary_family),
        "training_support": contract.get("training_support", support),
        "condition_sampler": contract.get("condition_sampler", sampler),
        "architecture": architecture,
        "parent_checkpoint_sha256": contract.get("parent_checkpoint_sha256", "scratch"),
    }


def grouped_stats(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty row group")
    completions = [float(row["terminal_absolute"]) for row in rows]
    successes = sum(bool(row["success"]) for row in rows)
    steps = [int(row["steps"]) for row in rows]
    no_effect_actions = [int(row["no_effect_action_count"]) for row in rows]
    if any(value <= 0 for value in steps):
        raise ValueError("evaluation steps must be positive")
    if any(value < 0 for value in no_effect_actions):
        raise ValueError("no-effect action counts must be nonnegative")
    total_steps = sum(steps)
    total_no_effect_actions = sum(no_effect_actions)
    near_complete_count = sum(value >= NEAR_COMPLETE_THRESHOLD for value in completions)
    return {
        "episodes": len(rows),
        "exact_successes": successes,
        "exact_rate": successes / len(rows),
        "near_complete_count": near_complete_count,
        "near_complete_rate": near_complete_count / len(rows),
        "completion_mean": mean(completions),
        "completion_median": median(completions),
        "completion_p10": quantile(completions, 0.10),
        "completion_p25": quantile(completions, 0.25),
        "completion_min": min(completions),
        "completion_max": max(completions),
        "zero_completion_count": sum(value <= 1e-12 for value in completions),
        "zero_completion_rate": sum(value <= 1e-12 for value in completions)
        / len(rows),
        "total_steps": total_steps,
        "no_effect_action_count": total_no_effect_actions,
        "no_effect_action_rate": total_no_effect_actions / total_steps,
    }


def factor_axis_summaries(
    common: dict,
    per_map: list[dict],
    conditions: dict[str, dict],
    by_condition: dict[str, dict],
) -> list[dict]:
    summaries = []
    axes = sorted({descriptor["factor_axis"] for descriptor in conditions.values()})
    for axis in axes:
        axis_conditions = sorted(
            condition_id
            for condition_id, descriptor in conditions.items()
            if descriptor["factor_axis"] == axis
        )
        axis_maps = [row for row in per_map if row["primary_cell"] in axis_conditions]
        worst_condition = min(
            axis_conditions,
            key=lambda condition_id: (
                by_condition[condition_id]["completion_mean"],
                condition_id,
            ),
        )
        summaries.append(
            {
                **common,
                "factor_axis": axis,
                "families": ",".join(
                    sorted(
                        {
                            conditions[condition_id]["family"]
                            for condition_id in axis_conditions
                        }
                    )
                ),
                "condition_count": len(axis_conditions),
                "macro_completion": mean(
                    by_condition[condition_id]["completion_mean"]
                    for condition_id in axis_conditions
                ),
                "worst_condition": worst_condition,
                "worst_condition_completion": by_condition[worst_condition][
                    "completion_mean"
                ],
                **grouped_stats(axis_maps),
            }
        )
    return summaries


def _paired_metrics(
    treatment_maps: list[dict],
    control_maps: list[dict],
    condition_ids: list[str],
) -> dict:
    condition_set = set(condition_ids)
    treatment = [row for row in treatment_maps if row["primary_cell"] in condition_set]
    control = [row for row in control_maps if row["primary_cell"] in condition_set]
    treatment_stats = grouped_stats(treatment)
    control_stats = grouped_stats(control)

    def condition_macro(rows: list[dict]) -> float:
        return mean(
            grouped_stats([row for row in rows if row["primary_cell"] == condition_id])[
                "completion_mean"
            ]
            for condition_id in condition_ids
        )

    treatment_macro = condition_macro(treatment)
    control_macro = condition_macro(control)
    metrics = {
        "condition_count": len(condition_ids),
        "episodes": treatment_stats["episodes"],
        "treatment_macro_completion": treatment_macro,
        "control_macro_completion": control_macro,
        "delta_macro_completion": treatment_macro - control_macro,
    }
    for name in (
        "completion_mean",
        "exact_rate",
        "near_complete_rate",
        "zero_completion_rate",
        "no_effect_action_rate",
    ):
        metrics[f"treatment_{name}"] = treatment_stats[name]
        metrics[f"control_{name}"] = control_stats[name]
        metrics[f"delta_{name}"] = treatment_stats[name] - control_stats[name]
    metrics.update(
        treatment_exact_successes=treatment_stats["exact_successes"],
        control_exact_successes=control_stats["exact_successes"],
        delta_exact_successes=(
            treatment_stats["exact_successes"] - control_stats["exact_successes"]
        ),
        treatment_near_complete_count=treatment_stats["near_complete_count"],
        control_near_complete_count=control_stats["near_complete_count"],
        delta_near_complete_count=(
            treatment_stats["near_complete_count"]
            - control_stats["near_complete_count"]
        ),
    )
    return metrics


def paired_delta_rows(
    evaluations: dict[tuple[str, str, str, int], dict],
    conditions: dict[str, dict],
) -> list[dict]:
    rows = []
    for (campaign, arm, panel, update), treatment in sorted(evaluations.items()):
        control_arm = PAIRED_CONTROLS.get(arm)
        if control_arm is None:
            continue
        control_key = (campaign, control_arm, panel, update)
        if control_key not in evaluations:
            raise ValueError(
                f"missing paired control {control_arm} for {arm} "
                f"campaign={campaign} panel={panel} update={update}"
            )
        control = evaluations[control_key]
        treatment_ids = [row["episode_id"] for row in treatment["per_map"]]
        control_ids = [row["episode_id"] for row in control["per_map"]]
        if treatment_ids != control_ids:
            raise ValueError(
                f"paired episode sequence mismatch for {arm} <- {control_arm} "
                f"campaign={campaign} panel={panel} update={update}"
            )
        sequence_sha256 = hashlib.sha256("\n".join(treatment_ids).encode()).hexdigest()
        common = {
            "campaign_sha256": campaign,
            "arm": arm,
            "control_arm": control_arm,
            "panel": panel,
            "checkpoint_update": update,
            "treatment_checkpoint_sha256": treatment["checkpoint_sha256"],
            "control_checkpoint_sha256": control["checkpoint_sha256"],
            "treatment_seed": treatment["seed"],
            "control_seed": control["seed"],
            "episode_sequence_sha256": sequence_sha256,
        }

        scopes = [("policy", "all", "", "", "", sorted(conditions))]
        for family in ("foundation", "trench"):
            family_conditions = sorted(
                condition_id
                for condition_id, descriptor in conditions.items()
                if descriptor["family"] == family
            )
            scopes.append(("family", family, family, "", "", family_conditions))
        for axis in sorted(
            {descriptor["factor_axis"] for descriptor in conditions.values()}
        ):
            axis_conditions = sorted(
                condition_id
                for condition_id, descriptor in conditions.items()
                if descriptor["factor_axis"] == axis
            )
            scopes.append(("factor_axis", axis, "", axis, "", axis_conditions))
        for condition_id, descriptor in sorted(conditions.items()):
            scopes.append(
                (
                    "condition",
                    condition_id,
                    descriptor["family"],
                    descriptor["factor_axis"],
                    condition_id,
                    [condition_id],
                )
            )

        for scope_type, scope_id, family, axis, condition_id, condition_ids in scopes:
            rows.append(
                {
                    **common,
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "family": family,
                    "factor_axis": axis,
                    "condition_id": condition_id,
                    **_paired_metrics(
                        treatment["per_map"], control["per_map"], condition_ids
                    ),
                }
            )
    return rows


def validate_record(
    record: dict,
    panel: str,
    expected_manifest: list[dict],
    conditions: dict[str, dict],
    bank_identity: dict,
) -> None:
    if record["split"] != panel or not record["exact_manifest_enumeration"]:
        raise ValueError(f"{panel} evaluation is not an exact panel enumeration")
    if (
        record.get("policy_mode") != "deterministic"
        or record.get("deterministic") is not True
    ):
        raise ValueError(f"{panel} evaluation is not deterministic")
    if not record["summary"]["integrity"]["passed"]:
        raise ValueError(f"{panel} evaluation integrity failed")
    for key in ("environment_protocol_sha256", "source_registry_sha256", "schema"):
        if record["accepted_bank"][key] != bank_identity[key]:
            raise ValueError(f"{panel} accepted-bank {key} mismatch")

    rows = record["per_map"]
    expected_by_id = {row["episode_id"]: row for row in expected_manifest}
    expected_ids = set(expected_by_id)
    actual_ids = {row["episode_id"] for row in rows}
    if len(rows) != len(expected_manifest) or actual_ids != expected_ids:
        raise ValueError(f"{panel} per-map episode identities differ from manifest")
    if any(row["split"] != panel for row in rows):
        raise ValueError(f"{panel} per-map split mismatch")
    for row in rows:
        expected = expected_by_id[row["episode_id"]]
        for field in MANIFEST_IDENTITY_FIELDS:
            if row.get(field) != expected.get(field):
                raise ValueError(
                    f"{panel} identity mismatch for {row['episode_id']} field {field}: "
                    f"{row.get(field)!r} != {expected.get(field)!r}"
                )
    if {row["primary_cell"] for row in rows} != set(conditions):
        raise ValueError(f"{panel} per-map condition support mismatch")

    all_stats = grouped_stats(rows)
    summary = record["summary"]
    close(all_stats["exact_rate"], summary["overall"]["success_rate"], "exact rate")
    close(
        all_stats["completion_mean"], summary["graded"]["micro"]["mean"], "micro mean"
    )
    close(all_stats["completion_p10"], summary["graded"]["micro"]["p10"], "micro p10")
    condition_means = []
    for condition_id in conditions:
        selected = [row for row in rows if row["primary_cell"] == condition_id]
        stats = grouped_stats(selected)
        condition_means.append(stats["completion_mean"])
        close(
            stats["completion_mean"],
            summary["graded"]["by_primary_cell"][condition_id]["mean"],
            f"{condition_id} mean",
        )
    close(
        mean(condition_means), summary["graded"]["macro_completion"], "condition macro"
    )


def validate_checkpoint_sequences(
    records_by_panel: dict[str, list[dict]], expected_final_update: int, arm: str
) -> None:
    checkpoint_maps = {
        panel: {
            int(record["checkpoint_update"]): record["checkpoint_sha256"]
            for record in records
        }
        for panel, records in records_by_panel.items()
    }
    if checkpoint_maps["promotion"] != checkpoint_maps["development"]:
        raise ValueError(f"{arm} promotion/development checkpoint identities differ")
    if max(checkpoint_maps["promotion"]) != expected_final_update:
        raise ValueError(
            f"{arm} final checkpoint update does not match receipt: "
            f"{max(checkpoint_maps['promotion'])} != {expected_final_update}"
        )


def register_unique_arm(seen_arms: set[str], arm: str) -> None:
    if arm in seen_arms:
        raise ValueError(f"duplicate arm bundle: {arm}")
    seen_arms.add(arm)


def collect_rows(results_root: Path, bank_root: Path):
    conditions, panel_manifests, bank_identity = load_bank(bank_root)
    policy_rows = []
    family_rows = []
    factor_axis_rows = []
    condition_rows = []
    evaluations = {}
    input_files = [
        bank_root / "dataset.json",
        bank_root / "environment_protocol.json",
        bank_root / "source_registry.jsonl",
    ]
    for panel in PANELS:
        input_files.extend(
            (bank_root / panel / "dataset.json", bank_root / panel / "manifest.jsonl")
        )

    run_dirs = sorted(
        path for path in results_root.iterdir() if (path / "run_contract.env").is_file()
    )
    if not run_dirs:
        raise ValueError(f"no result bundles under {results_root}")

    seen_arms = set()
    campaign_manifests = {}
    for run_dir in run_dirs:
        contract = read_env(run_dir / "run_contract.env")
        arm = contract["arm"]
        register_unique_arm(seen_arms, arm)
        receipt_path = run_dir / "receipt.env"
        if receipt_path.is_file():
            receipt = read_env(receipt_path)
            if receipt.get("status") != "PASSED" or receipt.get("arm") != arm:
                raise ValueError(
                    f"run receipt did not pass or arm mismatched: {run_dir}"
                )
            for field in (
                "phase",
                "seed",
                "updates",
                "global_transitions",
                "terra_revision",
                "terra_baselines_revision",
            ):
                if receipt.get(field) != contract.get(field):
                    raise ValueError(f"{arm} receipt/contract {field} mismatch")
            campaign_manifest_path = run_dir / "campaign_manifest.json"
            campaign_manifest = read_json(campaign_manifest_path)
            for field in (
                "terra_revision",
                "terra_baselines_revision",
                "train_maps_per_condition",
            ):
                if str(campaign_manifest.get(field)) != str(receipt.get(field)):
                    raise ValueError(f"{arm} campaign/receipt {field} mismatch")
            campaign_sha = receipt["campaign_sha256"]
            previous_manifest = campaign_manifests.setdefault(
                campaign_sha, campaign_manifest
            )
            if previous_manifest != campaign_manifest:
                raise ValueError(f"campaign {campaign_sha} has inconsistent manifests")
            input_files.extend((receipt_path, campaign_manifest_path))
        else:
            if arm in P5_ARMS:
                raise ValueError(f"frozen P5 arm {arm} requires receipt.env")
            if contract.get("status") != "PASSED":
                raise ValueError(f"follow-up run contract did not pass: {run_dir}")
            receipt = {
                **contract,
                "campaign_sha256": followup_campaign_sha256(contract),
            }
        input_files.append(run_dir / "run_contract.env")

        records_by_panel = {}
        for panel in PANELS:
            eval_path = run_dir / "eval" / f"{panel}.json"
            input_files.append(eval_path)
            records = read_json(eval_path)
            updates = [int(record["checkpoint_update"]) for record in records]
            if updates != sorted(set(updates)):
                raise ValueError(
                    f"non-monotonic or duplicate checkpoints in {eval_path}"
                )
            records_by_panel[panel] = records

        expected_final_update = int(
            contract.get("final_checkpoint_update", contract["updates"])
        )
        validate_checkpoint_sequences(records_by_panel, expected_final_update, arm)
        fingerprints = {
            json.dumps(record.get("treatment_fingerprint"), sort_keys=True)
            for records in records_by_panel.values()
            for record in records
        }
        if len(fingerprints) != 1:
            raise ValueError(f"{arm} treatment fingerprint changed across evaluations")
        treatment = records_by_panel["promotion"][0]["treatment_fingerprint"]
        treatment_run = treatment["contract"]["run"]
        expected_bank_arm = contract.get("accepted_bank_arm", arm)
        if (
            treatment_run.get("config_name") != arm
            or treatment_run.get("accepted_bank_arm") != expected_bank_arm
            or str(treatment_run.get("seed")) != contract["seed"]
        ):
            raise ValueError(f"{arm} treatment run metadata mismatch")
        metadata = arm_metadata(arm, receipt, contract, treatment)

        for panel, records in records_by_panel.items():
            for record in records:
                validate_record(
                    record, panel, panel_manifests[panel], conditions, bank_identity
                )
                per_map = record["per_map"]
                update = int(record["checkpoint_update"])
                common = {
                    **metadata,
                    "panel": panel,
                    "checkpoint_update": update,
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "integrity_passed": True,
                }
                evaluation_key = (metadata["campaign_sha256"], arm, panel, update)
                if evaluation_key in evaluations:
                    raise ValueError(f"duplicate fixed evaluation {evaluation_key}")
                evaluations[evaluation_key] = {
                    "per_map": per_map,
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "seed": metadata["seed"],
                }

                overall = grouped_stats(per_map)
                by_condition = {}
                for condition_id, descriptor in conditions.items():
                    selected = [
                        row for row in per_map if row["primary_cell"] == condition_id
                    ]
                    stats = grouped_stats(selected)
                    by_condition[condition_id] = stats
                    condition_rows.append({**common, **descriptor, **stats})

                condition_macro = mean(
                    item["completion_mean"] for item in by_condition.values()
                )
                worst_condition = min(
                    by_condition,
                    key=lambda condition_id: (
                        by_condition[condition_id]["completion_mean"],
                        condition_id,
                    ),
                )
                family_macros = {}
                family_summaries = {}
                for family in ("foundation", "trench"):
                    family_maps = [row for row in per_map if row["family"] == family]
                    family_stats = grouped_stats(family_maps)
                    family_conditions = [
                        condition_id
                        for condition_id, descriptor in conditions.items()
                        if descriptor["family"] == family
                    ]
                    family_macro = mean(
                        by_condition[item]["completion_mean"]
                        for item in family_conditions
                    )
                    family_worst = min(
                        family_conditions,
                        key=lambda item: (by_condition[item]["completion_mean"], item),
                    )
                    family_macros[family] = family_macro
                    family_summary = {
                        **common,
                        "family": family,
                        "condition_count": len(family_conditions),
                        "macro_completion": family_macro,
                        "worst_condition": family_worst,
                        "worst_condition_completion": by_condition[family_worst][
                            "completion_mean"
                        ],
                        **family_stats,
                    }
                    family_summaries[family] = family_summary
                    family_rows.append(family_summary)

                factor_axis_rows.extend(
                    factor_axis_summaries(common, per_map, conditions, by_condition)
                )

                primary_scope = metadata["primary_family"]
                if primary_scope in family_summaries:
                    primary = family_summaries[primary_scope]
                    primary_macro = primary["macro_completion"]
                    primary_worst = primary["worst_condition"]
                    primary_worst_completion = primary["worst_condition_completion"]
                    primary_p10 = primary["completion_p10"]
                    primary_successes = primary["exact_successes"]
                    primary_near_complete = primary["near_complete_count"]
                    primary_near_complete_rate = primary["near_complete_rate"]
                    primary_no_effect_action_rate = primary["no_effect_action_rate"]
                    primary_episodes = primary["episodes"]
                else:
                    primary_scope = "all"
                    primary_macro = condition_macro
                    primary_worst = worst_condition
                    primary_worst_completion = by_condition[worst_condition][
                        "completion_mean"
                    ]
                    primary_p10 = overall["completion_p10"]
                    primary_successes = overall["exact_successes"]
                    primary_near_complete = overall["near_complete_count"]
                    primary_near_complete_rate = overall["near_complete_rate"]
                    primary_no_effect_action_rate = overall["no_effect_action_rate"]
                    primary_episodes = overall["episodes"]

                policy_rows.append(
                    {
                        **common,
                        "condition_count": len(conditions),
                        "macro_completion": condition_macro,
                        "family_macro_completion": mean(family_macros.values()),
                        "min_family_macro_completion": min(family_macros.values()),
                        "worst_condition": worst_condition,
                        "worst_condition_completion": by_condition[worst_condition][
                            "completion_mean"
                        ],
                        "primary_metric_scope": primary_scope,
                        "primary_macro_completion": primary_macro,
                        "primary_completion_p10": primary_p10,
                        "primary_worst_condition": primary_worst,
                        "primary_worst_condition_completion": primary_worst_completion,
                        "primary_exact_successes": primary_successes,
                        "primary_near_complete_count": primary_near_complete,
                        "primary_near_complete_rate": primary_near_complete_rate,
                        "primary_no_effect_action_rate": primary_no_effect_action_rate,
                        "primary_episodes": primary_episodes,
                        **overall,
                    }
                )

    groups = defaultdict(list)
    for row in condition_rows:
        key = (
            row["campaign_sha256"],
            row["arm"],
            row["panel"],
            row["checkpoint_update"],
            row["family"],
        )
        groups[key].append(row)
    for rows in groups.values():
        for rank, row in enumerate(
            sorted(
                rows, key=lambda item: (-item["completion_mean"], item["condition_id"])
            ),
            start=1,
        ):
            row["rank_within_family"] = rank

    digest = hashlib.sha256()
    for path in sorted(set(input_files), key=lambda item: str(item)):
        digest.update(
            str(
                path.relative_to(
                    bank_root if bank_root in path.parents else results_root
                )
            ).encode()
        )
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    metadata = {
        "schema": "terra_accepted_bank_leaderboard_v2",
        "input_digest_sha256": digest.hexdigest(),
        "bank_identity": bank_identity,
        "conditions": len(conditions),
        "panels": list(PANELS),
        "arms": sorted({row["arm"] for row in policy_rows}, key=arm_sort_key),
        "paired_controls": dict(PAIRED_CONTROLS),
        "near_complete_threshold": NEAR_COMPLETE_THRESHOLD,
    }
    return (
        metadata,
        policy_rows,
        family_rows,
        factor_axis_rows,
        condition_rows,
        paired_delta_rows(evaluations, conditions),
    )


def arm_sort_key(arm: str):
    return (P5_ARMS.index(arm) if arm in P5_ARMS else len(P5_ARMS), arm)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"non-rectangular rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    metadata: dict,
    policy_rows: list[dict],
    family_rows: list[dict],
    factor_axis_rows: list[dict],
    condition_rows: list[dict],
    paired_rows: list[dict],
) -> str:
    lines = [
        "# Terra Accepted-Bank Leaderboard",
        "",
        f"- Schema: `{metadata['schema']}`",
        f"- Input digest: `{metadata['input_digest_sha256']}`",
        f"- Conditions: {metadata['conditions']}",
        "- Promotion and development are reported separately; sealed is not read.",
        "- Values are terminal absolute completion. Exact is solved maps / evaluated maps.",
        f"- Near-complete means terminal absolute completion >= {metadata['near_complete_threshold']:.2f}.",
        "- No-effect rate is total no-effect actions / total evaluated action steps.",
        "- Training support and sampler are explicit. Only matched treatments may support causal claims; in P5, that is `G-UNIFORM` versus `G-ADAPTIVE`.",
    ]

    latest = {}
    for row in policy_rows:
        key = (row["arm"], row["panel"])
        if (
            key not in latest
            or row["checkpoint_update"] > latest[key]["checkpoint_update"]
        ):
            latest[key] = row

    for panel in PANELS:
        lines.extend(
            [
                "",
                f"## {panel.title()} — latest policy summary",
                "",
                "| Arm | Scope | Support | Sampler | Update | Primary macro | Min family | Foundation | Trench | Near95 | No-effect | Primary worst | Worst | Exact |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        rows = [
            latest[(arm, panel)] for arm in metadata["arms"] if (arm, panel) in latest
        ]
        family_lookup = {
            (row["arm"], row["panel"], row["checkpoint_update"], row["family"]): row
            for row in family_rows
        }
        scope_order = {"all": 0, "foundation": 1, "trench": 2}
        for row in sorted(
            rows,
            key=lambda item: (
                scope_order.get(item["primary_metric_scope"], 3),
                -item["primary_macro_completion"],
                arm_sort_key(item["arm"]),
            ),
        ):
            foundation = family_lookup[
                (row["arm"], panel, row["checkpoint_update"], "foundation")
            ]
            trench = family_lookup[
                (row["arm"], panel, row["checkpoint_update"], "trench")
            ]
            lines.append(
                f"| `{row['arm']}` | {row['primary_metric_scope']} | {row['training_support']} | "
                f"{row['condition_sampler']} | {row['checkpoint_update']} | "
                f"{row['primary_macro_completion']:.3f} | "
                f"{row['min_family_macro_completion']:.3f} | "
                f"{foundation['macro_completion']:.3f} | {trench['macro_completion']:.3f} | "
                f"{row['primary_near_complete_rate']:.3f} | "
                f"{row['primary_no_effect_action_rate']:.3f} | "
                f"`{row['primary_worst_condition']}` | "
                f"{row['primary_worst_condition_completion']:.3f} | "
                f"{row['primary_exact_successes']}/{row['primary_episodes']} |"
            )

    latest_axis = {
        (row["arm"], row["panel"], row["factor_axis"]): row
        for row in factor_axis_rows
        if row["checkpoint_update"]
        == latest[(row["arm"], row["panel"])]["checkpoint_update"]
    }
    for panel in PANELS:
        arms = [arm for arm in metadata["arms"] if (arm, panel) in latest]
        axes = sorted(
            {row["factor_axis"] for row in factor_axis_rows if row["panel"] == panel}
        )
        lines.extend(
            [
                "",
                f"## {panel.title()} — latest factor-axis matrix",
                "",
                "Each cell is `macro completion; near95 rate; no-effect rate`.",
                "",
                "| Factor axis | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
                "|---|" + "---:|" * len(arms),
            ]
        )
        for axis in axes:
            cells = []
            for arm in arms:
                row = latest_axis[(arm, panel, axis)]
                cells.append(
                    f"{row['macro_completion']:.3f}; "
                    f"{row['near_complete_rate']:.3f}; "
                    f"{row['no_effect_action_rate']:.3f}"
                )
            lines.append(f"| `{axis}` | " + " | ".join(cells) + " |")

    policy_pairs = [row for row in paired_rows if row["scope_type"] == "policy"]
    if policy_pairs:
        lines.extend(
            [
                "",
                "## Matched treatment deltas",
                "",
                "Positive completion/near95/exact deltas favor the treatment; negative no-effect deltas favor it.",
                "",
                "| Treatment <- control | Panel | Update | Delta macro | Delta near95 | Delta no-effect | Delta exact |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            policy_pairs,
            key=lambda item: (
                item["panel"],
                item["checkpoint_update"],
                arm_sort_key(item["arm"]),
            ),
        ):
            lines.append(
                f"| `{row['arm']}` <- `{row['control_arm']}` | {row['panel']} | "
                f"{row['checkpoint_update']} | {row['delta_macro_completion']:+.3f} | "
                f"{row['delta_near_complete_rate']:+.3f} | "
                f"{row['delta_no_effect_action_rate']:+.3f} | "
                f"{row['delta_exact_successes']:+d} |"
            )

    for panel in PANELS:
        lines.extend(
            [
                "",
                f"## {panel.title()} — latest condition matrix",
                "",
                "Each cell is `mean completion; exact/maps`.",
                "",
            ]
        )
        arms = [arm for arm in metadata["arms"] if (arm, panel) in latest]
        lines.append(
            "| Condition | Depth | Axis | "
            + " | ".join(f"`{arm}`" for arm in arms)
            + " |"
        )
        lines.append("|---|---|---|" + "---:|" * len(arms))
        row_lookup = {
            (
                row["arm"],
                row["panel"],
                row["checkpoint_update"],
                row["condition_id"],
            ): row
            for row in condition_rows
        }
        condition_ids = sorted(
            {row["condition_id"] for row in condition_rows},
            key=lambda item: (
                next(
                    row["family"]
                    for row in condition_rows
                    if row["condition_id"] == item
                ),
                item,
            ),
        )
        for condition_id in condition_ids:
            sample = next(
                row for row in condition_rows if row["condition_id"] == condition_id
            )
            cells = []
            for arm in arms:
                update = latest[(arm, panel)]["checkpoint_update"]
                row = row_lookup[(arm, panel, update, condition_id)]
                cells.append(
                    f"{row['completion_mean']:.3f}; {row['exact_successes']}/{row['episodes']}"
                )
            lines.append(
                f"| `{condition_id}` | {sample['branch_depth']} | {sample['factor_axis']} | "
                + " | ".join(cells)
                + " |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--bank-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    (
        metadata,
        policy_rows,
        family_rows,
        factor_axis_rows,
        condition_rows,
        paired_rows,
    ) = collect_rows(args.results_root.resolve(), args.bank_root.resolve())
    policy_rows.sort(
        key=lambda row: (
            row["panel"],
            arm_sort_key(row["arm"]),
            row["checkpoint_update"],
        )
    )
    family_rows.sort(
        key=lambda row: (
            row["panel"],
            arm_sort_key(row["arm"]),
            row["checkpoint_update"],
            row["family"],
        )
    )
    factor_axis_rows.sort(
        key=lambda row: (
            row["panel"],
            arm_sort_key(row["arm"]),
            row["checkpoint_update"],
            row["factor_axis"],
        )
    )
    condition_rows.sort(
        key=lambda row: (
            row["panel"],
            arm_sort_key(row["arm"]),
            row["checkpoint_update"],
            row["family"],
            row["condition_id"],
        )
    )
    paired_rows.sort(
        key=lambda row: (
            row["panel"],
            row["checkpoint_update"],
            arm_sort_key(row["arm"]),
            row["scope_type"],
            row["scope_id"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "leaderboard_policy.csv", policy_rows)
    write_csv(args.output_dir / "leaderboard_family.csv", family_rows)
    write_csv(args.output_dir / "leaderboard_factor_axis.csv", factor_axis_rows)
    write_csv(args.output_dir / "leaderboard_condition.csv", condition_rows)
    if paired_rows:
        write_csv(args.output_dir / "leaderboard_paired_delta.csv", paired_rows)
    payload = {
        "metadata": metadata,
        "policy": policy_rows,
        "family": family_rows,
        "factor_axis": factor_axis_rows,
        "condition": condition_rows,
        "paired_delta": paired_rows,
    }
    (args.output_dir / "leaderboard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "LEADERBOARD.md").write_text(
        markdown_report(
            metadata,
            policy_rows,
            family_rows,
            factor_axis_rows,
            condition_rows,
            paired_rows,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
