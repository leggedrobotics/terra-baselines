#!/usr/bin/env python3
"""Inventory and summarize the matched V8 compact-versus-10M screen."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from scripts.euler_v8_deep_xattn_v1 import stage_gate

CONTROL = "G-V8-XATTN-REWARM-CONTROL"
TREATMENT = "G-V8-10M-XATTN-WARM"
ARMS = (CONTROL, TREATMENT)
PANELS = (
    "promotion",
    "development",
    "capability_promotion",
    "capability_development",
)
CHECKPOINT_INTERVAL = 500
EVALUATION_INTERVAL = 2_000
TARGET_UPDATE = 20_000
ALLOWED_JOB_STATES = {"COMPLETED", "TIMEOUT"}
CHECKPOINT_RE = re.compile(r".+_update_([0-9]{6})\.pkl")
EXPECTED_PARAMETERS = {CONTROL: 2_856_685, TREATMENT: 10_257_209}
EXPECTED_ARCHITECTURE = {
    arm: {
        "model_size": "medium",
        "model_core": "mlp",
        "map_encoder": "resnet_spatial_8x8_se_xattn",
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": "float32",
        "token_mixer_residual_init_scale": 0.0,
        "critic_hidden_dims": [512, 256],
        "resnet_stage_channels": (
            [24, 48, 64, 96] if arm == CONTROL else [64, 128, 192, 256]
        ),
        "resnet_blocks_per_stage": [2, 2, 3, 3],
        "loaded_max": 100,
    }
    for arm in ARMS
}
EXPECTED_PANEL_SHAPE = {
    "promotion": (45, 720, 384, 336),
    "development": (45, 720, 384, 336),
    "capability_promotion": (2, 32, 16, 16),
    "capability_development": (2, 32, 16, 16),
}
SHARED_CONTRACT_FIELDS = (
    "phase",
    "seed",
    "absolute_target_update",
    "curriculum_stage",
    "reward_type",
    "horizon",
    "full_resets",
    "terra_revision",
    "terra_baselines_revision",
    "training_bank_release_id",
    "training_bank_archive_sha256",
    "training_bank_dataset_sha256",
    "teacher_receipt_sha256",
    "teacher_checkpoint_sha256",
    "teacher_update",
    "num_devices",
    "num_envs_per_device",
    "num_steps",
    "num_minibatches",
    "update_epochs",
    "learning_rate",
    "kickstart_kl",
    "kickstart_value",
    "checkpoint_interval",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, expected_type: type):
    value = json.loads(path.read_text())
    if not isinstance(value, expected_type):
        raise ValueError(f"{path}: expected {expected_type.__name__}")
    return value


def parse_contract(path: Path) -> dict[str, str]:
    result = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise ValueError(f"{path}:{line_number}: duplicate or empty key")
        result[key] = value
    return result


def normalize_job_state(value: str) -> str:
    state = value.split("+", 1)[0]
    if state not in ALLOWED_JOB_STATES:
        raise ValueError(
            f"scale screen ended {state!r}; only COMPLETED or TIMEOUT is evaluable"
        )
    return state


def discover_checkpoints(run_dir: Path) -> dict[int, Path]:
    paths = {}
    for path in sorted((run_dir / "checkpoints").glob("*_update_*.pkl")):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unsupported checkpoint name: {path.name}")
        update = int(match.group(1))
        if not 0 < update <= TARGET_UPDATE or update % CHECKPOINT_INTERVAL:
            raise ValueError(f"checkpoint update {update} is off the frozen schedule")
        if update in paths:
            raise ValueError(f"duplicate checkpoint at update {update}")
        paths[update] = path.resolve()
    if not paths:
        raise ValueError(f"{run_dir}: no complete periodic checkpoints")
    observed = sorted(paths)
    expected = list(range(CHECKPOINT_INTERVAL, observed[-1] + 1, CHECKPOINT_INTERVAL))
    if observed != expected:
        raise ValueError(
            f"{run_dir}: checkpoint history is not contiguous: "
            f"expected {expected[:8]}...{expected[-1]}, got {observed[:8]}...{observed[-1]}"
        )
    return paths


def selected_common_updates(control: list[int], treatment: list[int]) -> list[int]:
    common = sorted(set(control) & set(treatment))
    if not common:
        raise ValueError("the two scale arms have no common checkpoint")
    common_max = common[-1]
    if common != list(range(CHECKPOINT_INTERVAL, common_max + 1, CHECKPOINT_INTERVAL)):
        raise ValueError("common checkpoint prefix is not contiguous")
    selected = {
        update
        for update in common
        if update in (500, 1_000, 1_500) or update % EVALUATION_INTERVAL == 0
    }
    selected.add(common_max)
    return sorted(selected)


def validate_contract(arm: str, run_dir: Path) -> dict[str, str]:
    contract = parse_contract(run_dir / "run_contract.env")
    expected = {
        "arm": arm,
        "phase": "screen",
        "absolute_target_update": str(TARGET_UPDATE),
        "curriculum_stage": "full",
        "reward_type": "DENSE",
        "horizon": "450",
        "full_resets": "true",
        "num_devices": "4",
        "num_envs_per_device": "1024",
        "num_steps": "32",
        "num_minibatches": "32",
        "update_epochs": "2",
        "learning_rate": "0.0003",
        "kickstart_kl": "1.0_to_0_over_1500",
        "kickstart_value": "0.5_to_0_over_500",
        "checkpoint_interval": str(CHECKPOINT_INTERVAL),
        "model_parameter_count": str(EXPECTED_PARAMETERS[arm]),
        "resnet_stage_channels": ",".join(
            str(value) for value in EXPECTED_ARCHITECTURE[arm]["resnet_stage_channels"]
        ),
        "resnet_blocks_per_stage": "2,2,3,3",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(
                f"{arm} contract {key} must be {value!r}, got {contract.get(key)!r}"
            )
    if contract.get("status") not in (None, "PASSED"):
        raise ValueError(f"{arm} contract has unsupported status")
    return contract


def build_inventory(
    *,
    control_run: Path,
    treatment_run: Path,
    control_job_id: str,
    treatment_job_id: str,
    control_job_state: str,
    treatment_job_state: str,
    control_elapsed_seconds: int,
    treatment_elapsed_seconds: int,
    evaluator_job_id: str,
) -> dict:
    runs = {CONTROL: control_run.resolve(), TREATMENT: treatment_run.resolve()}
    states = {
        CONTROL: normalize_job_state(control_job_state),
        TREATMENT: normalize_job_state(treatment_job_state),
    }
    job_ids = {CONTROL: control_job_id, TREATMENT: treatment_job_id}
    elapsed_seconds = {
        CONTROL: int(control_elapsed_seconds),
        TREATMENT: int(treatment_elapsed_seconds),
    }
    if any(value <= 0 for value in elapsed_seconds.values()):
        raise ValueError("scale-screen elapsed seconds must be positive")
    contracts = {arm: validate_contract(arm, run_dir) for arm, run_dir in runs.items()}
    for field in SHARED_CONTRACT_FIELDS:
        values = {contracts[arm].get(field) for arm in ARMS}
        if len(values) != 1 or None in values:
            raise ValueError(f"matched scale contract differs at {field}: {values}")
    checkpoints = {arm: discover_checkpoints(runs[arm]) for arm in ARMS}
    selected_updates = selected_common_updates(
        list(checkpoints[CONTROL]), list(checkpoints[TREATMENT])
    )
    selected = {
        arm: [
            {
                "update": update,
                "path": str(checkpoints[arm][update]),
                "sha256": sha256_file(checkpoints[arm][update]),
            }
            for update in selected_updates
        ]
        for arm in ARMS
    }
    return {
        "schema": "terra_v8_10m_screen_inventory_v1",
        "passed": True,
        "arms": list(ARMS),
        "evaluation_schedule": {
            "early_updates": [500, 1_000, 1_500],
            "interval": EVALUATION_INTERVAL,
            "latest_common_always_included": True,
        },
        "selected_common_updates": selected_updates,
        "latest_common_update": selected_updates[-1],
        "evaluator_job_id": evaluator_job_id,
        "jobs": {
            arm: {
                "job_id": job_ids[arm],
                "job_state": states[arm],
                "run_dir": str(runs[arm]),
                "run_contract": contracts[arm],
                "run_contract_sha256": sha256_file(runs[arm] / "run_contract.env"),
                "latest_complete_checkpoint": max(checkpoints[arm]),
                "elapsed_seconds": elapsed_seconds[arm],
                "completed_updates_per_hour": (
                    max(checkpoints[arm]) * 3600.0 / elapsed_seconds[arm]
                ),
                "selected_checkpoints": selected[arm],
            }
            for arm in ARMS
        },
    }


def panel_snapshot(record: dict) -> dict:
    if record.get("completion_contract") != "exact_visible_dump_v1":
        raise ValueError("evaluation completion contract changed")
    if record.get("deterministic") is not True or record.get("policy_mode") != (
        "deterministic"
    ):
        raise ValueError("scale leaderboard requires deterministic evaluation")
    if record.get("exact_manifest_enumeration") is not True:
        raise ValueError("scale leaderboard requires exact manifest enumeration")
    if record.get("horizon") != 450:
        raise ValueError("evaluation horizon changed")
    if record.get("summary", {}).get("integrity", {}).get("passed") is not True:
        raise ValueError("evaluation integrity failed")
    if record.get("reset_verification", {}).get("passed") is not True:
        raise ValueError("evaluation reset verification failed")
    per_map = record.get("per_map")
    if not isinstance(per_map, list) or not per_map:
        raise ValueError("evaluation lacks per-map results")

    by_condition: dict[str, list[dict]] = {}
    for row in per_map:
        condition = row.get("primary_cell")
        family = row.get("family")
        completion = row.get("terminal_absolute")
        if (
            not isinstance(condition, str)
            or family not in ("foundation", "trench")
            or not isinstance(completion, (int, float))
            or not math.isfinite(completion)
            or not -1e-6 <= completion <= 1.0 + 1e-6
            or not isinstance(row.get("success"), bool)
        ):
            raise ValueError("invalid per-map fixed-bank result")
        by_condition.setdefault(condition, []).append(row)

    condition_results = {}
    family_conditions = {"foundation": [], "trench": []}
    for condition, rows in sorted(by_condition.items()):
        families = {row["family"] for row in rows}
        if len(families) != 1:
            raise ValueError(f"condition {condition} crosses families")
        family = families.pop()
        if len(rows) != 16:
            raise ValueError(f"condition {condition} must contain exactly 16 episodes")
        completion = np.asarray(
            [float(row["terminal_absolute"]) for row in rows], dtype=np.float64
        )
        result = {
            "family": family,
            "successes": sum(int(row["success"]) for row in rows),
            "episodes": len(rows),
            "mean_completion": float(completion.mean()),
        }
        condition_results[condition] = result
        family_conditions[family].append(condition)
    if any(not values for values in family_conditions.values()):
        raise ValueError("evaluation must contain both Terra task families")

    by_family = {}
    for family, conditions in family_conditions.items():
        by_family[family] = {
            "successes": sum(
                condition_results[item]["successes"] for item in conditions
            ),
            "episodes": sum(condition_results[item]["episodes"] for item in conditions),
            "macro_completion": float(
                np.mean(
                    [condition_results[item]["mean_completion"] for item in conditions]
                )
            ),
        }
    all_completion = np.asarray(
        [float(row["terminal_absolute"]) for row in per_map], dtype=np.float64
    )
    overall_successes = sum(
        result["successes"] for result in condition_results.values()
    )
    overall_episodes = sum(result["episodes"] for result in condition_results.values())
    declared = record["summary"]["overall"]
    if (
        declared.get("successes") != overall_successes
        or declared.get("episodes") != overall_episodes
    ):
        raise ValueError("evaluation aggregate differs from per-map accounting")
    worst = min(
        condition_results,
        key=lambda item: (condition_results[item]["mean_completion"], item),
    )
    return {
        "exact_successes": overall_successes,
        "episodes": overall_episodes,
        "macro_completion": float(
            np.mean(
                [result["mean_completion"] for result in condition_results.values()]
            )
        ),
        "micro_p10": float(np.percentile(all_completion, 10)),
        "worst_condition": worst,
        "worst_condition_completion": condition_results[worst]["mean_completion"],
        "by_family": by_family,
        "by_condition": condition_results,
        "integrity_passed": True,
    }


def validate_treatment_fingerprint(record: dict, arm: str, seed: str) -> dict:
    fingerprint = record.get("treatment_fingerprint")
    if not isinstance(fingerprint, dict) or not isinstance(
        fingerprint.get("contract"), dict
    ):
        raise ValueError(f"{arm}: evaluation lacks a treatment fingerprint")
    contract = fingerprint["contract"]
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if fingerprint.get("sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError(f"{arm}: treatment fingerprint hash mismatch")
    if contract.get("schema") != "terra_fixed_bank_treatment_v1":
        raise ValueError(f"{arm}: treatment fingerprint schema changed")
    run = contract.get("run", {})
    if (
        run.get("config_name") != "G-V8-FIXED"
        or run.get("accepted_bank_arm") != "G-UNIFORM"
        or str(run.get("seed")) != seed
    ):
        raise ValueError(f"{arm}: treatment run identity changed")
    run_name = run.get("name")
    normalized_arm = arm.lower().replace("-", "_")
    if (
        not isinstance(run_name, str)
        or normalized_arm not in run_name
        or "full" not in run_name
    ):
        raise ValueError(f"{arm}: treatment run name changed")
    accepted_bank = record.get("accepted_bank", {})
    expected_bank = {
        "terra_revision": stage_gate.TERRA_REVISION,
        "environment_protocol_sha256": accepted_bank.get("environment_protocol_sha256"),
        "source_registry_sha256": accepted_bank.get("source_registry_sha256"),
    }
    if contract.get("bank") != expected_bank:
        raise ValueError(f"{arm}: treatment bank identity changed")
    expected_ppo = {
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
    }
    if contract.get("ppo") != expected_ppo:
        raise ValueError(f"{arm}: PPO treatment changed")
    reward_action = contract.get("reward_action", {})
    levels = reward_action.get("curriculum_levels")
    expected_level_fields = {
        "max_steps_in_episode": 450,
        "rewards_type": 0,
        "apply_trench_rewards": False,
    }
    if (
        reward_action.get("agent_types") != [0]
        or reward_action.get("action_types") != [0]
        or reward_action.get("relocation_progress_mult") != 1.5
        or not isinstance(levels, list)
        or len(levels) != 47
        or len({level.get("maps_path") for level in levels}) != 47
        or any(
            any(level.get(key) != value for key, value in expected_level_fields.items())
            for level in levels
        )
    ):
        raise ValueError(f"{arm}: reward/action/map treatment changed")
    expected_sampler = {
        "enabled": True,
        "rule": "fixed",
        "update_interval": 150,
        "uniform_floor": 0.20,
        "mastery_threshold": 0.75,
        "temperature": 0.25,
        "min_episodes": 20,
        "competence_ema": 0.30,
        "max_mass": 0.15,
        "seed": int(seed),
    }
    if contract.get("sampler") != expected_sampler:
        raise ValueError(f"{arm}: full-V8 fixed sampler changed")
    if contract.get("architecture") != EXPECTED_ARCHITECTURE[arm]:
        raise ValueError(f"{arm}: evaluation architecture changed")
    return contract


def validate_evaluations(
    *, inventory: dict, paths: dict[tuple[str, str], Path]
) -> tuple[dict, dict]:
    selected_updates = inventory["selected_common_updates"]
    snapshots = {}
    records_by_key = {}
    bank_identity = None
    treatment_contracts = {}
    for arm in ARMS:
        expected = inventory["jobs"][arm]["selected_checkpoints"]
        seed = inventory["jobs"][arm]["run_contract"]["seed"]
        for panel in PANELS:
            path = paths[(arm, panel)].resolve()
            records = read_json(path, list)
            if len(records) != len(expected):
                raise ValueError(f"{arm}/{panel}: evaluation count changed")
            for record, checkpoint in zip(records, expected):
                if (
                    record.get("checkpoint_update") != checkpoint["update"]
                    or record.get("checkpoint") != checkpoint["path"]
                    or record.get("checkpoint_sha256") != checkpoint["sha256"]
                    or record.get("split") != panel.removeprefix("capability_")
                ):
                    raise ValueError(f"{arm}/{panel}: checkpoint identity mismatch")
                identity = record.get("accepted_bank")
                if (
                    not isinstance(identity, dict)
                    or identity.get("schema") != "terra_curriculum_loader_bank_v1"
                    or identity.get("terra_revision") != stage_gate.TERRA_REVISION
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(identity.get("environment_protocol_sha256", "")),
                    )
                    is None
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(identity.get("source_registry_sha256", "")),
                    )
                    is None
                    or identity.get("diagnostic_control") is not False
                    or identity.get("diagnostic_contract_sha256") is not None
                ):
                    raise ValueError(f"{arm}/{panel}: accepted-bank identity changed")
                if bank_identity is None:
                    bank_identity = identity
                elif identity != bank_identity:
                    raise ValueError("fixed evaluation bank identity changed")
                treatment = validate_treatment_fingerprint(record, arm, seed)
                if arm in treatment_contracts and treatment != treatment_contracts[arm]:
                    raise ValueError(f"{arm}: treatment changed across evaluations")
                treatment_contracts[arm] = treatment
                snapshot = panel_snapshot(record)
                (
                    expected_conditions,
                    expected_episodes,
                    expected_foundation,
                    expected_trench,
                ) = EXPECTED_PANEL_SHAPE[panel]
                observed_shape = (
                    len(snapshot["by_condition"]),
                    snapshot["episodes"],
                    snapshot["by_family"]["foundation"]["episodes"],
                    snapshot["by_family"]["trench"]["episodes"],
                )
                if observed_shape != (
                    expected_conditions,
                    expected_episodes,
                    expected_foundation,
                    expected_trench,
                ):
                    raise ValueError(
                        f"{arm}/{panel}: panel shape changed from "
                        f"{EXPECTED_PANEL_SHAPE[panel]} to {observed_shape}"
                    )
                snapshots[(arm, panel, checkpoint["update"])] = snapshot
            records_by_key[(arm, panel)] = records

    shared_treatments = []
    for arm in ARMS:
        treatment = copy.deepcopy(treatment_contracts[arm])
        treatment["run"]["name"] = "<arm-specific>"
        treatment["architecture"] = "<arm-specific>"
        shared_treatments.append(treatment)
    if shared_treatments[0] != shared_treatments[1]:
        raise ValueError("scale arms differ outside run name and architecture")

    for panel in PANELS:
        for index, update in enumerate(selected_updates):
            control_rows = records_by_key[(CONTROL, panel)][index]["per_map"]
            treatment_rows = records_by_key[(TREATMENT, panel)][index]["per_map"]
            control_identity = [
                (row.get("episode_id"), row.get("slot_index")) for row in control_rows
            ]
            treatment_identity = [
                (row.get("episode_id"), row.get("slot_index")) for row in treatment_rows
            ]
            if control_identity != treatment_identity:
                raise ValueError(
                    f"{panel}@{update}: paired episode sequence changed between arms"
                )
    return snapshots, records_by_key


def result_rows(snapshots: dict) -> tuple[list[dict], list[dict]]:
    policy_rows = []
    condition_rows = []
    for (arm, panel, update), result in sorted(snapshots.items()):
        policy_rows.append(
            {
                "arm": arm,
                "panel": panel,
                "update": update,
                "exact_successes": result["exact_successes"],
                "episodes": result["episodes"],
                "macro_completion": result["macro_completion"],
                "foundation_macro_completion": result["by_family"]["foundation"][
                    "macro_completion"
                ],
                "trench_macro_completion": result["by_family"]["trench"][
                    "macro_completion"
                ],
                "micro_p10": result["micro_p10"],
                "worst_condition": result["worst_condition"],
                "worst_condition_completion": result["worst_condition_completion"],
                "integrity_passed": True,
            }
        )
        for condition, values in sorted(result["by_condition"].items()):
            condition_rows.append(
                {
                    "arm": arm,
                    "panel": panel,
                    "update": update,
                    "condition_id": condition,
                    **values,
                }
            )
    return policy_rows, condition_rows


def paired_rows(snapshots: dict, updates: list[int]) -> list[dict]:
    rows = []
    for panel in PANELS:
        for update in updates:
            control = snapshots[(CONTROL, panel, update)]
            treatment = snapshots[(TREATMENT, panel, update)]

            def append(
                scope: str, name: str, control_value: float, treatment_value: float
            ):
                rows.append(
                    {
                        "panel": panel,
                        "update": update,
                        "scope": scope,
                        "name": name,
                        "control_completion": control_value,
                        "treatment_completion": treatment_value,
                        "treatment_minus_control": treatment_value - control_value,
                    }
                )

            append(
                "overall",
                "all",
                control["macro_completion"],
                treatment["macro_completion"],
            )
            for family in ("foundation", "trench"):
                append(
                    "family",
                    family,
                    control["by_family"][family]["macro_completion"],
                    treatment["by_family"][family]["macro_completion"],
                )
            if set(control["by_condition"]) != set(treatment["by_condition"]):
                raise ValueError("scale arms expose different condition sets")
            for condition in sorted(control["by_condition"]):
                append(
                    "condition",
                    condition,
                    control["by_condition"][condition]["mean_completion"],
                    treatment["by_condition"][condition]["mean_completion"],
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"non-rectangular rows for {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    inventory: dict,
    snapshots: dict,
    policy_rows: list[dict],
) -> str:
    updates = inventory["selected_common_updates"]
    latest = updates[-1]
    lines = [
        "# V8 compact versus 10M scale screen",
        "",
        "Deterministic full-V8 fixed-bank evaluation. Completion is terminal "
        "absolute task completion; exact is fully completed maps.",
        "",
        "## Throughput",
        "",
        "| Arm | Last complete update | Elapsed hours | Updates/hour |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        job = inventory["jobs"][arm]
        lines.append(
            f"| `{arm}` | {job['latest_complete_checkpoint']} | "
            f"{job['elapsed_seconds'] / 3600.0:.2f} | "
            f"{job['completed_updates_per_hour']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Common-checkpoint history",
            "",
            "| Panel | Update | Arm | Macro | Foundation | Trench | P10 | Worst | Exact |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        policy_rows, key=lambda item: (item["panel"], item["update"], item["arm"])
    ):
        lines.append(
            f"| {row['panel']} | {row['update']} | `{row['arm']}` | "
            f"{row['macro_completion']:.3f} | "
            f"{row['foundation_macro_completion']:.3f} | "
            f"{row['trench_macro_completion']:.3f} | {row['micro_p10']:.3f} | "
            f"{row['worst_condition_completion']:.3f} | "
            f"{row['exact_successes']}/{row['episodes']} |"
        )
    for panel in PANELS:
        control = snapshots[(CONTROL, panel, latest)]
        treatment = snapshots[(TREATMENT, panel, latest)]
        lines.extend(
            [
                "",
                f"## {panel.title()} per condition at update {latest}",
                "",
                "| Condition | Family | Compact completion; exact | 10M completion; exact | Delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for condition in sorted(control["by_condition"]):
            compact = control["by_condition"][condition]
            large = treatment["by_condition"][condition]
            lines.append(
                f"| `{condition}` | {compact['family']} | "
                f"{compact['mean_completion']:.3f}; {compact['successes']}/{compact['episodes']} | "
                f"{large['mean_completion']:.3f}; {large['successes']}/{large['episodes']} | "
                f"{large['mean_completion'] - compact['mean_completion']:+.3f} |"
            )
    lines.extend(
        [
            "",
            "The screen does not auto-promote either arm. Replication or a 120-hour "
            "continuation requires a separate recorded decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(*, inventory_path: Path, paths: dict, output_dir: Path) -> dict:
    inventory = read_json(inventory_path.resolve(), dict)
    if inventory.get("schema") != "terra_v8_10m_screen_inventory_v1":
        raise ValueError("unsupported scale-screen inventory")
    snapshots, _ = validate_evaluations(inventory=inventory, paths=paths)
    policy, conditions = result_rows(snapshots)
    paired = paired_rows(snapshots, inventory["selected_common_updates"])
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "leaderboard_policy.csv", policy)
    write_csv(output_dir / "leaderboard_condition.csv", conditions)
    write_csv(output_dir / "leaderboard_paired_delta.csv", paired)
    payload = {
        "schema": "terra_v8_10m_leaderboard_v1",
        "passed": True,
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path.resolve()),
        "selected_common_updates": inventory["selected_common_updates"],
        "policy": policy,
        "conditions": conditions,
        "paired_delta": paired,
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output_dir / "LEADERBOARD.md").write_text(
        markdown_report(inventory, snapshots, policy)
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--control-run", type=Path, required=True)
    inventory.add_argument("--treatment-run", type=Path, required=True)
    inventory.add_argument("--control-job-id", required=True)
    inventory.add_argument("--treatment-job-id", required=True)
    inventory.add_argument("--control-job-state", required=True)
    inventory.add_argument("--treatment-job-state", required=True)
    inventory.add_argument("--control-elapsed-seconds", type=int, required=True)
    inventory.add_argument("--treatment-elapsed-seconds", type=int, required=True)
    inventory.add_argument("--evaluator-job-id", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--inventory", type=Path, required=True)
    for arm in ("control", "treatment"):
        for panel in PANELS:
            summary.add_argument(f"--{arm}-{panel}", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        result = build_inventory(
            control_run=args.control_run,
            treatment_run=args.treatment_run,
            control_job_id=args.control_job_id,
            treatment_job_id=args.treatment_job_id,
            control_job_state=args.control_job_state,
            treatment_job_state=args.treatment_job_state,
            control_elapsed_seconds=args.control_elapsed_seconds,
            treatment_elapsed_seconds=args.treatment_elapsed_seconds,
            evaluator_job_id=args.evaluator_job_id,
        )
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        for arm in ARMS:
            for checkpoint in result["jobs"][arm]["selected_checkpoints"]:
                print(f"{arm}\t{checkpoint['path']}")
    else:
        paths = {
            (CONTROL, "promotion"): args.control_promotion,
            (CONTROL, "development"): args.control_development,
            (CONTROL, "capability_promotion"): args.control_capability_promotion,
            (CONTROL, "capability_development"): args.control_capability_development,
            (TREATMENT, "promotion"): args.treatment_promotion,
            (TREATMENT, "development"): args.treatment_development,
            (TREATMENT, "capability_promotion"): (args.treatment_capability_promotion),
            (TREATMENT, "capability_development"): (
                args.treatment_capability_development
            ),
        }
        result = summarize(
            inventory_path=args.inventory,
            paths=paths,
            output_dir=args.output_dir,
        )
        print(
            f"V8_10M_LEADERBOARD_PASSED latest_common_update="
            f"{result['selected_common_updates'][-1]}"
        )


if __name__ == "__main__":
    main()
