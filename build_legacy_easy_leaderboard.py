#!/usr/bin/env python3
"""Build the condition-balanced Terra Legacy-Easy v1 leaderboard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PANELS = ("promotion", "development")
POLICY_MODES = ("deterministic", "sampled")
RESULT_SCHEMA = "terra_fixed_bank_eval_v4"
LEADERBOARD_SCHEMA = "terra_legacy_easy_leaderboard_v1"
EPISODE_BANK_SCHEMA = "terra_legacy_easy_explicit_episode_bank_v1"
COMPLETION_CONTRACT = "exact_visible_dump_v1"
STRATUM = "legacy_easy_capability_floor"
HORIZON = 450
SAMPLED_SEED = 20260803
NEAR_COMPLETE_THRESHOLD = 0.95
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")
POLICY_LABEL = re.compile(r"[A-Za-z0-9_.-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty group")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"{label}: recorded {actual!r}, recomputed {expected!r}")


def grouped_stats(rows: list[dict]) -> dict:
    """Summarize one non-empty map group without changing its weighting."""
    if not rows:
        raise ValueError("cannot summarize an empty group")
    completion = [float(row["terminal_absolute"]) for row in rows]
    if any(
        not math.isfinite(value) or not -1e-6 <= value <= 1.0 + 1e-6
        for value in completion
    ):
        raise ValueError("terminal absolute completion must be finite and in [0, 1]")
    steps = [row["steps"] for row in rows]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in steps
    ):
        raise ValueError("evaluation steps must be positive integers")

    if any("no_effect_action_count" not in row for row in rows):
        raise ValueError("Legacy-Easy rows require no-effect action counts")
    counts = [row["no_effect_action_count"] for row in rows]
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > step_count
        for value, step_count in zip(counts, steps)
    ):
        raise ValueError("no-effect action counts must lie in [0, steps]")
    no_effect_count = sum(counts)
    exact = sum(int(row["success"]) for row in rows)
    near_complete = sum(value >= NEAR_COMPLETE_THRESHOLD for value in completion)
    zero_completion = sum(value <= 1e-12 for value in completion)
    return {
        "episodes": len(rows),
        "exact_completions": exact,
        "exact_completion_rate": exact / len(rows),
        "near_complete_threshold": NEAR_COMPLETE_THRESHOLD,
        "near_completions": near_complete,
        "near_completion_rate": near_complete / len(rows),
        "completion_mean": mean(completion),
        "completion_median": median(completion),
        "completion_p10": quantile(completion, 0.10),
        "completion_p25": quantile(completion, 0.25),
        "completion_min": min(completion),
        "completion_max": max(completion),
        "zero_completions": zero_completion,
        "zero_completion_rate": zero_completion / len(rows),
        "total_steps": sum(steps),
        "no_effect_action_count": no_effect_count,
        "no_effect_action_rate": no_effect_count / sum(steps),
    }


def _read_policy_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = ["policy_label", "checkpoint_path", "checkpoint_sha256"]
        if reader.fieldnames != expected:
            raise ValueError(f"policy matrix must have the exact header {expected}")
        policies = list(reader)
    if not policies:
        raise ValueError("policy matrix is empty")
    labels = []
    for row in policies:
        label = row["policy_label"]
        if not POLICY_LABEL.fullmatch(label):
            raise ValueError(f"invalid policy label: {label!r}")
        checkpoint = row["checkpoint_path"]
        if not checkpoint.startswith("/"):
            raise ValueError(f"checkpoint path must be absolute: {checkpoint!r}")
        if not SHA256.fullmatch(row["checkpoint_sha256"]):
            raise ValueError(f"invalid checkpoint SHA-256 for {label}")
        labels.append(label)
    duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
    if duplicates:
        raise ValueError(f"duplicate policy labels: {duplicates}")
    return policies


def _read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid or duplicate receipt line in {path}: {line!r}")
        values[key] = value
    return values


def _verify_output_manifest(policy_root: Path) -> str:
    manifest = policy_root / "files.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    declared = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not SHA256.fullmatch(digest):
            raise ValueError(f"invalid output manifest row: {line!r}")
        relative = relative.removeprefix("./")
        if not relative or relative in declared:
            raise ValueError(f"duplicate or empty output manifest path: {relative!r}")
        declared[relative] = digest
    actual = {
        path.relative_to(policy_root).as_posix()
        for path in policy_root.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(declared) != actual:
        raise ValueError(
            f"{policy_root}: output manifest coverage differs; "
            f"missing={sorted(actual - set(declared))}, "
            f"extra={sorted(set(declared) - actual)}"
        )
    for relative, expected in declared.items():
        actual_digest = sha256_file(policy_root / relative)
        if actual_digest != expected:
            raise ValueError(f"{policy_root / relative}: output SHA-256 mismatch")
    return sha256_file(manifest)


def _load_one_record(path: Path) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"{path}: expected exactly one checkpoint record")
    if not isinstance(records[0], dict):
        raise ValueError(f"{path}: checkpoint record must be an object")
    return records[0]


def _record_contract(record: dict) -> dict:
    bank = record["explicit_episode_bank"]
    return {
        "result_schema": record.get("schema"),
        "completion_contract": record.get("completion_contract"),
        "horizon": record.get("horizon"),
        "stratum": record.get("stratum"),
        "episode_bank_schema": bank.get("schema"),
        "release_id": bank.get("release_id"),
        "episode_bank_sha256": bank.get("episode_bank_sha256"),
        "environment_protocol_sha256": bank.get("environment_protocol_sha256"),
        "environment_protocol_file_sha256": bank.get(
            "environment_protocol_file_sha256"
        ),
        "source_registry_sha256": bank.get("source_registry_sha256"),
        "files_manifest_sha256": bank.get("files_manifest_sha256"),
        "terra_revision": bank.get("terra_revision"),
        "episode_id_schema": bank.get("episode_id_schema"),
        "initial_agent_state_schema": bank.get("initial_agent_state_schema"),
    }


def _validate_record(
    path: Path,
    record: dict,
    policy: dict[str, str],
    panel: str,
    mode: str,
) -> tuple[dict, tuple[tuple[int, str, str, str], ...], dict]:
    expected = {
        "schema": RESULT_SCHEMA,
        "completion_contract": COMPLETION_CONTRACT,
        "checkpoint": policy["checkpoint_path"],
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "split": panel,
        "stratum": STRATUM,
        "horizon": HORIZON,
        "deterministic": mode == "deterministic",
        "policy_mode": mode,
        "seed": SAMPLED_SEED,
        "exact_manifest_enumeration": True,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"{path}: expected {field}={value!r}, got {record.get(field)!r}"
            )
    bank = record.get("explicit_episode_bank")
    if not isinstance(bank, dict):
        raise ValueError(f"{path}: missing explicit episode-bank receipt")
    bank_expected = {
        "schema": EPISODE_BANK_SCHEMA,
        "diagnostic_only": True,
        "included_in_constrained_macro": False,
        "panel": panel,
        "condition_balanced": True,
    }
    for field, value in bank_expected.items():
        if bank.get(field) != value:
            raise ValueError(
                f"{path}: expected episode-bank {field}={value!r}, "
                f"got {bank.get(field)!r}"
            )
    if record.get("manifest_sha256") != bank.get("manifest_sha256"):
        raise ValueError(f"{path}: result and episode-bank manifest hashes differ")
    for field in (
        "episode_bank_sha256",
        "environment_protocol_sha256",
        "environment_protocol_file_sha256",
        "source_registry_sha256",
        "manifest_sha256",
        "initial_states_sha256",
        "files_manifest_sha256",
    ):
        if not SHA256.fullmatch(str(bank.get(field, ""))):
            raise ValueError(f"{path}: invalid episode-bank {field}")
    fingerprint = record.get("treatment_fingerprint", {}).get("sha256")
    if not SHA256.fullmatch(str(fingerprint or "")):
        raise ValueError(f"{path}: invalid treatment fingerprint SHA-256")
    checkpoint_update = record.get("checkpoint_update")
    if (
        not isinstance(checkpoint_update, int)
        or isinstance(checkpoint_update, bool)
        or checkpoint_update < 0
    ):
        raise ValueError(f"{path}: invalid checkpoint update")
    summary = record.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("integrity", {}).get("passed") is not True
    ):
        raise ValueError(f"{path}: evaluation integrity did not pass")
    if int(summary["integrity"].get("failure_count", -1)) != 0:
        raise ValueError(f"{path}: evaluation reports integrity failures")
    rows = record.get("per_map")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: per_map must be a non-empty list")
    if len(rows) != bank.get("slot_count"):
        raise ValueError(f"{path}: per-map count differs from episode-bank slot count")

    episode_identities = []
    condition_family = {}
    condition_counts = defaultdict(int)
    for row in rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{path}: missing episode_id")
        slot = row.get("slot_index")
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise ValueError(f"{path}: invalid slot_index")
        family = row.get("family")
        condition = row.get("primary_cell")
        if not isinstance(family, str) or not family:
            raise ValueError(f"{path}: missing family")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"{path}: missing primary_cell")
        previous = condition_family.setdefault(condition, family)
        if previous != family:
            raise ValueError(f"{path}: condition {condition!r} spans two families")
        condition_counts[condition] += 1
        episode_identities.append((slot, episode_id, family, condition))
        if row.get("split") != panel:
            raise ValueError(f"{path}: per-map row binds a different panel")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"{path}: success must be boolean")
        completion = float(row["terminal_absolute"])
        if row["success"] != math.isclose(completion, 1.0, abs_tol=1e-6):
            raise ValueError(f"{path}: exact success disagrees with completion")
    slots = [identity[0] for identity in episode_identities]
    if slots != list(range(1, len(rows) + 1)):
        raise ValueError(
            f"{path}: explicit episode slots are not contiguous and ordered"
        )
    episode_ids = [identity[1] for identity in episode_identities]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f"{path}: duplicate explicit episode IDs")
    if len(condition_counts) != bank.get("condition_count"):
        raise ValueError(f"{path}: observed condition count differs from bank receipt")
    if set(condition_counts.values()) != {bank.get("maps_per_condition")}:
        raise ValueError(f"{path}: panel is not condition balanced")

    by_condition = {}
    for condition in sorted(condition_counts):
        selected = [row for row in rows if row["primary_cell"] == condition]
        by_condition[condition] = {
            "family": condition_family[condition],
            **grouped_stats(selected),
        }
    by_family = {}
    for family in sorted(set(condition_family.values())):
        selected_conditions = [
            condition
            for condition, condition_value in condition_family.items()
            if condition_value == family
        ]
        family_rows = [row for row in rows if row["family"] == family]
        worst = min(
            selected_conditions,
            key=lambda condition: (
                by_condition[condition]["completion_mean"],
                condition,
            ),
        )
        by_family[family] = {
            "condition_count": len(selected_conditions),
            "macro_completion": mean(
                by_condition[condition]["completion_mean"]
                for condition in selected_conditions
            ),
            "worst_condition": worst,
            "worst_condition_completion": by_condition[worst]["completion_mean"],
            **grouped_stats(family_rows),
        }
    worst_condition = min(
        by_condition,
        key=lambda condition: (by_condition[condition]["completion_mean"], condition),
    )
    macro_completion = mean(
        condition["completion_mean"] for condition in by_condition.values()
    )
    overall = grouped_stats(rows)

    recorded_overall = summary.get("overall", {})
    if recorded_overall.get("episodes") != overall["episodes"]:
        raise ValueError(f"{path}: recorded episode count is inconsistent")
    if recorded_overall.get("successes") != overall["exact_completions"]:
        raise ValueError(f"{path}: recorded exact completion count is inconsistent")
    graded = summary.get("graded", {})
    if graded.get("available") is not True:
        raise ValueError(f"{path}: graded completion is unavailable")
    _close(graded.get("macro_completion"), macro_completion, f"{path}: macro")
    for condition, statistics in by_condition.items():
        _close(
            graded.get("by_primary_cell", {}).get(condition, {}).get("mean"),
            statistics["completion_mean"],
            f"{path}: condition {condition}",
        )
    for family, statistics in by_family.items():
        _close(
            graded.get("by_family", {}).get(family, {}).get("macro_completion"),
            statistics["macro_completion"],
            f"{path}: family {family}",
        )

    result = {
        "policy_label": policy["policy_label"],
        "panel": panel,
        "policy_mode": mode,
        "condition_count": len(by_condition),
        "family_count": len(by_family),
        "macro_completion": macro_completion,
        "family_macro_completion": mean(
            family["macro_completion"] for family in by_family.values()
        ),
        "worst_condition": worst_condition,
        "worst_condition_completion": by_condition[worst_condition]["completion_mean"],
        "overall": overall,
        "by_family": by_family,
        "by_condition": by_condition,
        "provenance": {
            "result_sha256": sha256_file(path),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_update": checkpoint_update,
            "treatment_fingerprint_sha256": fingerprint,
            "manifest_sha256": record["manifest_sha256"],
            "initial_states_sha256": bank.get("initial_states_sha256"),
            "slurm_seed": record.get("seed"),
        },
    }
    return result, tuple(episode_identities), _record_contract(record)


def build_leaderboard(results_root: Path, policy_matrix: Path) -> dict:
    """Load one complete four-result evaluation set per declared policy."""
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    policies = _read_policy_matrix(policy_matrix)
    expected_labels = {policy["policy_label"] for policy in policies}
    actual_labels = {path.name for path in results_root.iterdir() if path.is_dir()}
    if actual_labels != expected_labels:
        raise ValueError(
            "result policy directories differ from the matrix; "
            f"missing={sorted(expected_labels - actual_labels)}, "
            f"extra={sorted(actual_labels - expected_labels)}"
        )

    expected_files = {
        f"{panel}_{mode}.json" for panel in PANELS for mode in POLICY_MODES
    }
    evaluations = []
    global_contract = None
    panel_contracts = {}
    panel_episode_identities = {}
    baselines_revision = None
    terra_revision = None
    policy_provenance = []
    for policy in policies:
        label = policy["policy_label"]
        root = results_root / label
        actual_json = {path.name for path in root.glob("*.json")}
        if actual_json != expected_files:
            raise ValueError(
                f"{root}: result JSON set differs; "
                f"missing={sorted(expected_files - actual_json)}, "
                f"extra={sorted(actual_json - expected_files)}"
            )
        output_manifest_sha256 = _verify_output_manifest(root)
        receipt = _read_env(root / "receipt.env")
        receipt_expected = {
            "schema": "terra_legacy_easy_v1_euler_eval_receipt_v1",
            "status": "PASSED",
            "policy_label": label,
            "checkpoint_path": policy["checkpoint_path"],
            "checkpoint_sha256": policy["checkpoint_sha256"],
            "panels": ",".join(PANELS),
            "policy_modes": ",".join(POLICY_MODES),
            "sampled_seed": str(SAMPLED_SEED),
            "horizon": str(HORIZON),
            "completion_contract": COMPLETION_CONTRACT,
        }
        for field, value in receipt_expected.items():
            if receipt.get(field) != value:
                raise ValueError(
                    f"{root / 'receipt.env'}: expected {field}={value!r}, "
                    f"got {receipt.get(field)!r}"
                )
        for field in (
            "baselines_revision",
            "terra_revision",
            "episode_bank_json_sha256",
            "episode_bank_files_sha256",
        ):
            if not receipt.get(field):
                raise ValueError(f"{root / 'receipt.env'}: missing {field}")
        for field in ("baselines_revision", "terra_revision"):
            if not GIT_SHA.fullmatch(receipt[field]):
                raise ValueError(f"{root / 'receipt.env'}: invalid {field}")
        for field in ("episode_bank_json_sha256", "episode_bank_files_sha256"):
            if not SHA256.fullmatch(receipt[field]):
                raise ValueError(f"{root / 'receipt.env'}: invalid {field}")
        if baselines_revision is None:
            baselines_revision = receipt["baselines_revision"]
            terra_revision = receipt["terra_revision"]
        elif (baselines_revision, terra_revision) != (
            receipt["baselines_revision"],
            receipt["terra_revision"],
        ):
            raise ValueError("Legacy-Easy outputs mix code revisions")

        policy_fingerprint = None
        checkpoint_update = None
        for panel in PANELS:
            for mode in POLICY_MODES:
                path = root / f"{panel}_{mode}.json"
                result, episode_identities, contract = _validate_record(
                    path, _load_one_record(path), policy, panel, mode
                )
                if (
                    contract["episode_bank_sha256"]
                    != receipt["episode_bank_json_sha256"]
                ):
                    raise ValueError(
                        f"{path}: result and Euler receipt bind different banks"
                    )
                if (
                    contract["files_manifest_sha256"]
                    != receipt["episode_bank_files_sha256"]
                ):
                    raise ValueError(
                        f"{path}: result and Euler receipt bind different bank files"
                    )
                if contract["terra_revision"] != receipt["terra_revision"]:
                    raise ValueError(
                        f"{path}: result and Euler receipt bind different Terra revisions"
                    )
                if global_contract is None:
                    global_contract = contract
                elif global_contract != contract:
                    raise ValueError(f"{path}: Legacy-Easy evaluations mix contracts")
                panel_identity = {
                    "manifest_sha256": result["provenance"]["manifest_sha256"],
                    "initial_states_sha256": result["provenance"][
                        "initial_states_sha256"
                    ],
                }
                if (
                    panel in panel_contracts
                    and panel_contracts[panel] != panel_identity
                ):
                    raise ValueError(f"{path}: {panel} results mix panel contracts")
                panel_contracts.setdefault(panel, panel_identity)
                if (
                    panel in panel_episode_identities
                    and panel_episode_identities[panel] != episode_identities
                ):
                    raise ValueError(
                        f"{path}: {panel} results mix episode condition identities"
                    )
                panel_episode_identities.setdefault(panel, episode_identities)
                fingerprint = result["provenance"]["treatment_fingerprint_sha256"]
                update = result["provenance"]["checkpoint_update"]
                if policy_fingerprint is None:
                    policy_fingerprint = fingerprint
                    checkpoint_update = update
                elif (policy_fingerprint, checkpoint_update) != (fingerprint, update):
                    raise ValueError(f"{root}: one policy mixes checkpoint contracts")
                evaluations.append(result)
        policy_provenance.append(
            {
                "policy_label": label,
                "checkpoint_path": policy["checkpoint_path"],
                "checkpoint_sha256": policy["checkpoint_sha256"],
                "checkpoint_update": checkpoint_update,
                "treatment_fingerprint_sha256": policy_fingerprint,
                "slurm_job_id": receipt.get("slurm_job_id"),
                "output_manifest_sha256": output_manifest_sha256,
            }
        )

    if len(evaluations) != len(policies) * len(PANELS) * len(POLICY_MODES):
        raise ValueError("Legacy-Easy evaluation matrix is incomplete")
    global_contract = {
        **global_contract,
        "baselines_revision": baselines_revision,
        "terra_revision": terra_revision,
    }
    return {
        "schema": LEADERBOARD_SCHEMA,
        "policy_count": len(policies),
        "panels": list(PANELS),
        "policy_modes": list(POLICY_MODES),
        "policy_matrix_sha256": sha256_file(policy_matrix),
        "contract": global_contract,
        "panel_provenance": panel_contracts,
        "policy_provenance": policy_provenance,
        "evaluations": evaluations,
    }


def _rate(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(leaderboard: dict) -> str:
    lines = [
        "# Terra Legacy-Easy v1 leaderboard",
        "",
        (
            "Equal-condition macro completion is the primary graded diagnostic. "
            "Exact completion uses the frozen visible-dump completion contract."
        ),
        "",
        "## Policy summary",
        "",
        "| Policy | Mode | Panel | Macro | Foundation | Trench | Exact | p10 | Zero | No-effect |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard["evaluations"]:
        families = row["by_family"]
        overall = row["overall"]
        lines.append(
            "| "
            f"{row['policy_label']} | {row['policy_mode']} | {row['panel']} | "
            f"{_rate(row['macro_completion'])} | "
            f"{_rate(families['foundation']['macro_completion']) if 'foundation' in families else 'n/a'} | "
            f"{_rate(families['trench']['macro_completion']) if 'trench' in families else 'n/a'} | "
            f"{overall['exact_completions']}/{overall['episodes']} | "
            f"{_rate(overall['completion_p10'])} | "
            f"{overall['zero_completions']}/{overall['episodes']} | "
            f"{_rate(overall['no_effect_action_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Family breakdown",
            "",
            "| Policy | Mode | Panel | Family | Conditions | Macro | Exact | p10 | Worst condition | No-effect |",
            "|---|---|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in leaderboard["evaluations"]:
        for family, stats in row["by_family"].items():
            lines.append(
                "| "
                f"{row['policy_label']} | {row['policy_mode']} | {row['panel']} | "
                f"{family} | {stats['condition_count']} | "
                f"{_rate(stats['macro_completion'])} | "
                f"{stats['exact_completions']}/{stats['episodes']} | "
                f"{_rate(stats['completion_p10'])} | {stats['worst_condition']} "
                f"({_rate(stats['worst_condition_completion'])}) | "
                f"{_rate(stats['no_effect_action_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Condition breakdown",
            "",
            "| Policy | Mode | Panel | Family | Condition | Mean | p10 | Exact | Zero | No-effect |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in leaderboard["evaluations"]:
        for condition, stats in row["by_condition"].items():
            lines.append(
                "| "
                f"{row['policy_label']} | {row['policy_mode']} | {row['panel']} | "
                f"{stats['family']} | {condition} | "
                f"{_rate(stats['completion_mean'])} | {_rate(stats['completion_p10'])} | "
                f"{stats['exact_completions']}/{stats['episodes']} | "
                f"{stats['zero_completions']}/{stats['episodes']} | "
                f"{_rate(stats['no_effect_action_rate'])} |"
            )

    contract = leaderboard["contract"]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Episode bank: `{contract['episode_bank_sha256']}`",
            f"- Environment protocol: `{contract['environment_protocol_sha256']}`",
            f"- Source registry: `{contract['source_registry_sha256']}`",
            f"- Terra revision: `{contract['terra_revision']}`",
            f"- Baselines revision: `{contract['baselines_revision']}`",
            f"- Policy matrix: `{leaderboard['policy_matrix_sha256']}`",
            "",
            "| Policy | Checkpoint update | Checkpoint SHA-256 | Output manifest SHA-256 |",
            "|---|---:|---|---|",
        ]
    )
    for row in leaderboard["policy_provenance"]:
        lines.append(
            f"| {row['policy_label']} | {row['checkpoint_update']} | "
            f"`{row['checkpoint_sha256']}` | `{row['output_manifest_sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--policy-matrix", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    leaderboard = build_leaderboard(args.results_root, args.policy_matrix)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(leaderboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(leaderboard), encoding="utf-8")


if __name__ == "__main__":
    main()
