#!/usr/bin/env python3
"""Append fixed-bank evaluation summaries to the originating W&B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from utils.wandb_human import EVAL_CONDITION_COLUMNS, fixed_eval_metrics

SPLITS = ("promotion", "development")


def load_evaluations(path: Path, split: str) -> list[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a nonempty checkpoint list")
    updates = []
    for record in records:
        if record.get("split") != split:
            raise ValueError(
                f"{path} contains split={record.get('split')!r}, expected {split!r}"
            )
        updates.append(int(record["checkpoint_update"]))
    if updates != sorted(set(updates)):
        raise ValueError(f"{path} checkpoint updates must be unique and sorted")
    return records


def build_history(paths: dict[str, Path]) -> list[dict]:
    """Return validated scalar/table payloads, one per checkpoint update."""
    by_split = {split: load_evaluations(paths[split], split) for split in SPLITS}
    updates = {
        split: [int(record["checkpoint_update"]) for record in records]
        for split, records in by_split.items()
    }
    if updates["promotion"] != updates["development"]:
        raise ValueError(
            "promotion and development must contain identical checkpoint updates"
        )

    history = []
    for index, update in enumerate(updates["promotion"]):
        scalars = {"eval/update": float(update)}
        tables = {}
        for split in SPLITS:
            metrics, rows = fixed_eval_metrics(by_split[split][index], split)
            scalars.update(metrics)
            tables[f"eval/{split}_conditions"] = rows
        history.append({"scalars": scalars, "tables": tables})
    return history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--run-id", default=os.getenv("WANDB_RUN_ID"))
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the payload without contacting W&B",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "promotion": args.promotion.resolve(),
        "development": args.development.resolve(),
    }
    if args.receipt.exists():
        raise FileExistsError(f"fixed-eval W&B receipt already exists: {args.receipt}")
    history = build_history(paths)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "updates": [
                        int(item["scalars"]["eval/update"]) for item in history
                    ],
                    "scalar_keys": sorted(history[0]["scalars"]),
                    "condition_table_columns": list(EVAL_CONDITION_COLUMNS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not args.run_id or not args.entity or not args.project:
        raise ValueError("run id, entity, and project are required to append W&B eval")

    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must",
    )
    wandb.define_metric("eval/update")
    for metric in sorted(set(history[0]["scalars"]) - {"eval/update"}):
        wandb.define_metric(metric, step_metric="eval/update")
    for item in history:
        payload = dict(item["scalars"])
        for key, rows in item["tables"].items():
            payload[key] = wandb.Table(
                columns=list(EVAL_CONDITION_COLUMNS),
                data=rows,
            )
        wandb.log(payload)
    run.finish()

    receipt = {
        "schema": "terra_wandb_fixed_eval_receipt_v1",
        "run_id": args.run_id,
        "entity": args.entity,
        "project": args.project,
        "updates": [int(item["scalars"]["eval/update"]) for item in history],
        "inputs": {
            split: {"path": str(path), "sha256": _sha256(path)}
            for split, path in paths.items()
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.receipt)


if __name__ == "__main__":
    main()
