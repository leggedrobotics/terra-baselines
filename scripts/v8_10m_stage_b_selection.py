#!/usr/bin/env python3
"""Select hash-pinned Stage-B parents from corrected whole-V8 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.euler_v8_deep_xattn_v1 import stage_gate

SCHEMA = "terra_v8_10m_stage_b_selection_v1"
UPDATES = (1000, 2000, 3000, 4000)
ARMS = (
    "G-V8-XATTN-REWARM-CONTROL",
    "G-V8-10M-XATTN-WARM",
)
CAPABILITY_MIN = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path, split: str, condition_ids: tuple[str, ...]) -> list[dict]:
    records = json.loads(path.read_text())
    if (
        not isinstance(records, list)
        or tuple(record.get("checkpoint_update") for record in records) != UPDATES
    ):
        raise ValueError(f"{path}: expected checkpoints {UPDATES}")
    for record in records:
        expected = {
            "schema": stage_gate.EVAL_SCHEMA,
            "completion_contract": stage_gate.COMPLETION_CONTRACT,
            "horizon": 450,
            "deterministic": True,
            "policy_mode": "deterministic",
            "exact_manifest_enumeration": True,
            "split": split,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{path}: {key} changed")
        stage_gate.validate_panel_conditions([record], condition_ids)
        if not stage_gate._integrity_passed(record):
            raise ValueError(f"{path}: integrity failed")
    return records


def _identity(record: dict) -> tuple[str, str, int]:
    return stage_gate._checkpoint_identity(record)


def _macro(record: dict) -> float:
    value = record["summary"]["graded"]["macro_completion"]
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("evaluation lacks a finite macro completion")
    return float(value)


def _exact(record: dict) -> int:
    value = record["summary"]["overall"]["successes"]
    if not isinstance(value, int):
        raise ValueError("evaluation lacks exact successes")
    return value


def _capability_passed(record: dict) -> bool:
    counts = stage_gate._cell_counts(record, stage_gate.CAPABILITY_IDS)
    return min(counts.values()) >= CAPABILITY_MIN


def _validate_parent_path(path: str, arm: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9_./-]+", path) is None:
        raise ValueError("parent checkpoint path contains unsupported characters")
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.suffix != ".pkl":
        raise ValueError("parent checkpoint path must be an absolute pickle")
    if arm not in candidate.parts or "checkpoints" not in candidate.parts:
        raise ValueError("parent checkpoint path does not match its arm")


def select_parent(
    arm: str,
    main_promotion: list[dict],
    main_development: list[dict],
    capability_promotion: list[dict],
    capability_development: list[dict],
    core_ids: tuple[str, ...],
) -> dict:
    identities = [_identity(record) for record in main_promotion]
    for records in (
        main_development,
        capability_promotion,
        capability_development,
    ):
        if [_identity(record) for record in records] != identities:
            raise ValueError(f"{arm}: evaluation panels name different checkpoints")
    eligible = [
        index
        for index in range(len(UPDATES))
        if _capability_passed(capability_promotion[index])
        and _capability_passed(capability_development[index])
    ]
    if not eligible:
        raise ValueError(f"{arm}: no checkpoint retains both capability controls")
    # Promotion chooses the parent. Development is reported but never promotes.
    index = max(
        eligible,
        key=lambda item: (
            _exact(main_promotion[item]),
            _macro(main_promotion[item]),
            -UPDATES[item],
        ),
    )
    selected = main_promotion[index]
    path = str(selected["checkpoint"])
    _validate_parent_path(path, arm)
    sha256 = str(selected["checkpoint_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("parent checkpoint lacks a SHA-256")

    development = main_development[index]
    exact_by_condition = development["summary"]["by_primary_cell"]
    graded_by_condition = development["summary"]["graded"]["by_primary_cell"]
    weaknesses = []
    for condition_id, graded in graded_by_condition.items():
        exact = exact_by_condition[condition_id]
        weaknesses.append(
            {
                "condition_id": condition_id,
                "curriculum_stage": ("nearby" if condition_id in core_ids else "full"),
                "family": (
                    "foundation"
                    if condition_id.startswith(("fnd-", "v7-fnd-"))
                    else "trench"
                ),
                "development_exact_successes": exact["successes"],
                "development_episodes": exact["episodes"],
                "development_mean_completion": graded["mean"],
            }
        )
    weaknesses.sort(
        key=lambda row: (
            row["development_mean_completion"],
            row["development_exact_successes"],
            row["condition_id"],
        )
    )
    return {
        "path": path,
        "sha256": sha256,
        "update": UPDATES[index],
        "selection_rule": (
            "max promotion exact, then promotion macro, then earliest update; "
            "requires >=12/16 per capability condition on promotion and development"
        ),
        "promotion": {
            "exact_successes": _exact(main_promotion[index]),
            "episodes": main_promotion[index]["summary"]["overall"]["episodes"],
            "macro_completion": _macro(main_promotion[index]),
        },
        "development": {
            "exact_successes": _exact(development),
            "episodes": development["summary"]["overall"]["episodes"],
            "macro_completion": _macro(development),
        },
        "capability_promotion": stage_gate._cell_counts(
            capability_promotion[index], stage_gate.CAPABILITY_IDS
        ),
        "capability_development": stage_gate._cell_counts(
            capability_development[index], stage_gate.CAPABILITY_IDS
        ),
        "weakest_development_conditions": weaknesses[:12],
    }


def build_selection(args: argparse.Namespace) -> dict:
    bank = stage_gate.load_bank_contract(args.bank_root.resolve())
    inputs: dict[str, dict[str, str]] = {}
    parents = {}
    for slug, arm in (("compact", ARMS[0]), ("10m", ARMS[1])):
        paths = {
            "main_promotion": getattr(args, f"{slug}_main_promotion").resolve(),
            "main_development": getattr(args, f"{slug}_main_development").resolve(),
            "capability_promotion": getattr(
                args, f"{slug}_capability_promotion"
            ).resolve(),
            "capability_development": getattr(
                args, f"{slug}_capability_development"
            ).resolve(),
        }
        records = {
            "main_promotion": _records(
                paths["main_promotion"], "promotion", bank["main_ids"]
            ),
            "main_development": _records(
                paths["main_development"], "development", bank["main_ids"]
            ),
            "capability_promotion": _records(
                paths["capability_promotion"],
                "promotion",
                stage_gate.CAPABILITY_IDS,
            ),
            "capability_development": _records(
                paths["capability_development"],
                "development",
                stage_gate.CAPABILITY_IDS,
            ),
        }
        parents[arm] = select_parent(
            arm,
            records["main_promotion"],
            records["main_development"],
            records["capability_promotion"],
            records["capability_development"],
            bank["core_ids"],
        )
        for label, path in paths.items():
            inputs[f"{slug}_{label}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return {
        "schema": SCHEMA,
        "passed": True,
        "next_stage": "nearby",
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "sampler_profile": "bounded_replay25_v1",
        "stage_population": {
            "capability_replay": 0.25,
            "nearby_core": 0.75,
            "foundation": 0.5,
            "trench": 0.5,
            "maps_per_condition": 96,
            "condition_count": 15,
            "distinct_training_maps": 1440,
        },
        "retention": {
            "frozen_thresholds": {
                condition_id: CAPABILITY_MIN
                for condition_id in stage_gate.CAPABILITY_IDS
            },
            "rollback_after_consecutive_failures": 2,
        },
        "parents": parents,
        "inputs": inputs,
    }


def inspect_selection(path: Path, arm: str | None = None) -> dict:
    receipt = json.loads(path.read_text())
    expected = {
        "schema": SCHEMA,
        "passed": True,
        "next_stage": "nearby",
        "release_id": stage_gate.RELEASE_ID,
        "terra_revision": stage_gate.TERRA_REVISION,
        "bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
        "sampler_profile": "bounded_replay25_v1",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"selection receipt {key} changed")
    if set(receipt.get("parents", {})) != set(ARMS):
        raise ValueError("selection receipt parent arms changed")
    expected_thresholds = {
        condition_id: CAPABILITY_MIN for condition_id in stage_gate.CAPABILITY_IDS
    }
    if receipt.get("retention", {}).get("frozen_thresholds") != expected_thresholds:
        raise ValueError("selection receipt retention thresholds changed")
    for parent_arm, parent in receipt["parents"].items():
        _validate_parent_path(parent.get("path", ""), parent_arm)
        if re.fullmatch(r"[0-9a-f]{64}", parent.get("sha256", "")) is None:
            raise ValueError("selection receipt parent hash changed")
        if parent.get("update") not in UPDATES:
            raise ValueError("selection receipt parent update changed")
    receipt["receipt_sha256"] = sha256_file(path)
    return (
        receipt
        if arm is None
        else receipt["parents"][arm] | {"receipt_sha256": receipt["receipt_sha256"]}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--bank-root", type=Path, required=True)
    for slug in ("compact", "10m"):
        for panel in (
            "main-promotion",
            "main-development",
            "capability-promotion",
            "capability-development",
        ):
            create.add_argument(f"--{slug}-{panel}", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--receipt", type=Path, required=True)
    inspect.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "create":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = build_selection(args)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    else:
        result = inspect_selection(args.receipt.resolve(), args.arm)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
