#!/usr/bin/env python3
"""Promotion and reset-parity gates for the accepted-bank Euler campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from utils.accepted_bank import ARMS as ACCEPTED_BANK_ARMS


SCREEN_UPDATES = (500, 1000, 2000)
GENERALIST_ARMS = ("G-UNIFORM", "G-ADAPTIVE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error


def write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def slice_agent_batch(agent, index: int, batch_size: int):
    """Select one Agent slot while preserving any genuinely scalar leaves."""
    import jax
    import numpy as np

    def select(value):
        shape = np.shape(value)
        if shape and shape[0] == batch_size:
            return value[index]
        return value

    return jax.tree_util.tree_map(select, agent)


def _record_arm(record: dict) -> str | None:
    return (
        record.get("treatment_fingerprint", {})
        .get("contract", {})
        .get("run", {})
        .get("accepted_bank_arm")
    )


def _panel_identity(record: dict) -> dict:
    return {
        "schema": record.get("schema"),
        "completion_contract": record.get("completion_contract"),
        "accepted_bank": record.get("accepted_bank"),
        "split": record.get("split"),
        "stratum": record.get("stratum"),
        "manifest_sha256": record.get("manifest_sha256"),
        "horizon": record.get("horizon"),
        "deterministic": record.get("deterministic"),
        "policy_mode": record.get("policy_mode"),
        "exact_manifest_enumeration": record.get(
            "exact_manifest_enumeration"
        ),
        "episode_ids": [row.get("episode_id") for row in record.get("per_map", ())],
        "map_ids": [row.get("map_id") for row in record.get("per_map", ())],
    }


def _shared_treatment(record: dict) -> dict:
    contract = record["treatment_fingerprint"]["contract"]
    run = contract["run"]
    sampler = dict(contract.get("sampler") or {})
    return {
        "seed": run.get("seed"),
        "bank": contract.get("bank"),
        "ppo": contract.get("ppo"),
        "reward_action": contract.get("reward_action"),
        "architecture": contract.get("architecture"),
        "sampler_without_rule": {
            key: value for key, value in sampler.items() if key != "rule"
        },
    }


def load_screen(path: Path, arm: str) -> list[dict]:
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path}: evaluation output must be a JSON list")
    updates = tuple(record.get("checkpoint_update") for record in records)
    if updates != SCREEN_UPDATES:
        raise ValueError(
            f"{path}: expected checkpoint updates {SCREEN_UPDATES}, got {updates}"
        )
    for record in records:
        if record.get("schema") != "terra_fixed_bank_eval_v4":
            raise ValueError(f"{path}: unsupported evaluation schema")
        if record.get("completion_contract") != "exact_visible_dump_v1":
            raise ValueError(f"{path}: unsupported completion contract")
        if _record_arm(record) != arm:
            raise ValueError(
                f"{path}: expected treatment arm {arm}, got {_record_arm(record)}"
            )
        if record.get("split") != "promotion":
            raise ValueError(f"{path}: selection requires the promotion panel")
        if record.get("policy_mode") != "deterministic":
            raise ValueError(f"{path}: selection requires deterministic evaluation")
    return records


def _candidate(records: list[dict]) -> dict:
    reference, final = records[-2:]
    comparison = final.get("comparison_to_previous")
    if not isinstance(comparison, dict):
        raise ValueError("update-2000 record has no comparison to update 1000")
    if reference.get("checkpoint_update") != 1000:
        raise ValueError("promotion comparison reference is not update 1000")
    if comparison.get("schema") != "terra_fixed_bank_comparison_gate_v1":
        raise ValueError("promotion comparison has an unsupported schema")
    if comparison.get("reference_checkpoint") != reference.get("checkpoint"):
        raise ValueError("promotion comparison does not reference update 1000")
    final_integrity_passed = bool(final["summary"]["integrity"]["passed"])
    comparison_integrity_passed = bool(comparison.get("integrity_passed"))
    return {
        "passed": bool(
            comparison.get("passed")
            and comparison_integrity_passed
            and final_integrity_passed
        ),
        "checkpoint": final.get("checkpoint"),
        "checkpoint_sha256": final.get("checkpoint_sha256"),
        "checkpoint_update": final.get("checkpoint_update"),
        "comparison_to_previous": comparison,
        "macro_completion": float(
            final["summary"]["graded"]["macro_completion"]
        ),
        "exact_successes": int(final["summary"]["overall"]["successes"]),
        "worst_condition_completion": float(
            final["summary"]["graded"]["worst_condition_completion"]
        ),
        "integrity_passed": final_integrity_passed,
        "comparison_integrity_passed": comparison_integrity_passed,
    }


def select_promotion(
    uniform_path: Path,
    adaptive_path: Path,
) -> dict:
    uniform = load_screen(uniform_path, "G-UNIFORM")
    adaptive = load_screen(adaptive_path, "G-ADAPTIVE")
    for uniform_record, adaptive_record in zip(uniform, adaptive):
        if _panel_identity(uniform_record) != _panel_identity(adaptive_record):
            raise ValueError(
                "generalist screens do not enumerate the identical promotion panel"
            )
        if _shared_treatment(uniform_record) != _shared_treatment(
            adaptive_record
        ):
            raise ValueError(
                "generalist screens differ outside the declared sampler rule"
            )
        uniform_rule = (
            uniform_record["treatment_fingerprint"]["contract"]
            .get("sampler", {})
            .get("rule")
        )
        adaptive_rule = (
            adaptive_record["treatment_fingerprint"]["contract"]
            .get("sampler", {})
            .get("rule")
        )
        if (uniform_rule, adaptive_rule) != ("uniform", "adaptive"):
            raise ValueError(
                "generalist treatment fingerprints do not declare "
                "uniform versus adaptive sampling"
            )

    candidates = {
        "G-UNIFORM": _candidate(uniform),
        "G-ADAPTIVE": _candidate(adaptive),
    }
    passing = [
        arm for arm in GENERALIST_ARMS if candidates[arm]["passed"]
    ]
    if not passing:
        selected = None
        reason = "neither generalist passed the update-1000 to update-2000 gate"
    elif len(passing) == 1:
        selected = passing[0]
        reason = "only one generalist passed the update-1000 to update-2000 gate"
    else:
        def rank(arm: str) -> tuple[float, int, float]:
            candidate = candidates[arm]
            return (
                candidate["macro_completion"],
                candidate["exact_successes"],
                candidate["worst_condition_completion"],
            )

        uniform_rank = rank("G-UNIFORM")
        adaptive_rank = rank("G-ADAPTIVE")
        selected = (
            "G-ADAPTIVE"
            if adaptive_rank > uniform_rank
            else "G-UNIFORM"
        )
        reason = (
            "both passed; ranked by macro completion, exact successes, then "
            "worst-condition completion; exact ties choose G-UNIFORM"
        )

    return {
        "schema": "terra_accepted_bank_promotion_decision_v1",
        "promotion_passed": selected is not None,
        "selected_arm": selected,
        "reason": reason,
        "selection_panel": _panel_identity(uniform[-1]),
        "screen_updates": list(SCREEN_UPDATES),
        "inputs": {
            "G-UNIFORM": {
                "path": str(uniform_path.resolve()),
                "sha256": sha256_file(uniform_path),
            },
            "G-ADAPTIVE": {
                "path": str(adaptive_path.resolve()),
                "sha256": sha256_file(adaptive_path),
            },
        },
        "candidates": candidates,
    }


def reset_hashes(args: argparse.Namespace) -> dict:
    import jax
    import numpy as np

    from eval_fixed_bank import (
        configure_for_bank,
        load_manifest,
        manifest_reset_keys,
        sha256_file as eval_sha256_file,
        verify_exact_reset,
    )
    from train_mixed import make_mixed_agent_states
    from utils.accepted_bank import load_accepted_bank
    from utils.helpers import load_pkl_object
    from terra.benchmark_state import agent_state_sha256

    bank = load_accepted_bank(
        args.bank_root,
        "G-UNIFORM",
        args.terra_revision,
    )
    panel = next(
        panel for panel in bank.evaluation_panels if panel.name == args.panel
    )
    directory = bank.root / panel.maps_path
    rows = load_manifest(directory)
    checkpoint = load_pkl_object(str(args.checkpoint.resolve()))
    config = configure_for_bank(
        checkpoint["train_config"],
        panel.maps_path,
        len(rows),
    )
    os.environ["DATASET_PATH"] = str(bank.root)
    os.environ["DATASET_SIZE"] = str(len(rows))
    _, env, env_params, _ = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    keys = manifest_reset_keys(
        rows,
        len(rows),
        bank.environment_protocol_sha256,
    )
    verification = verify_exact_reset(
        env,
        env_params,
        keys,
        directory,
        len(rows),
    )
    state = env.reset(env_params, keys).state
    ordered_hashes = []
    for index in range(len(rows)):
        agent = slice_agent_batch(
            state.agent,
            index,
            len(rows),
        )
        ordered_hashes.append(agent_state_sha256(agent))
    return {
        "schema": "terra_accepted_bank_reset_hashes_v1",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "panel": args.panel,
        "slots": len(rows),
        "terra_revision": args.terra_revision,
        "environment_protocol_sha256": bank.environment_protocol_sha256,
        "source_registry_sha256": bank.source_registry_sha256,
        "manifest_sha256": eval_sha256_file(directory / "manifest.jsonl"),
        "checkpoint_sha256": eval_sha256_file(args.checkpoint.resolve()),
        "episode_ids": [row["episode_id"] for row in rows],
        "ordered_agent_state_sha256": ordered_hashes,
        "reset_verification": verification,
    }


def verify_smoke(
    arm: str,
    periodic_path: Path,
    final_path: Path,
) -> dict:
    import jax
    import numpy as np
    import sys

    from train import TrainConfig
    from train_mixed import MixedAgentTrainConfig
    from utils.helpers import load_pkl_object

    sys.modules["__main__"].TrainConfig = TrainConfig
    sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig
    if arm not in ACCEPTED_BANK_ARMS:
        raise ValueError(f"unsupported smoke arm {arm!r}")
    checkpoints = {
        "periodic": load_pkl_object(str(periodic_path.resolve())),
        "final": load_pkl_object(str(final_path.resolve())),
    }
    for label, checkpoint in checkpoints.items():
        if int(checkpoint.get("next_update", -1)) != 1:
            raise ValueError(f"{label} smoke checkpoint did not finish update 1")
        config = checkpoint.get("train_config")
        bank = getattr(config, "accepted_bank", None)
        if getattr(bank, "arm", None) != arm:
            raise ValueError(
                f"{label} smoke checkpoint has wrong accepted-bank arm"
            )
        for tree_name in ("model", "optimizer_state"):
            leaves = jax.tree_util.tree_leaves(checkpoint.get(tree_name))
            if not leaves:
                raise ValueError(f"{label} smoke checkpoint has no {tree_name}")
            if any(
                not np.all(np.isfinite(np.asarray(jax.device_get(leaf))))
                for leaf in leaves
            ):
                raise ValueError(
                    f"{label} smoke checkpoint has non-finite {tree_name}"
                )
        losses = checkpoint.get("loss_info")
        loss_leaves = jax.tree_util.tree_leaves(losses)
        if not loss_leaves or any(
            not np.all(np.isfinite(np.asarray(jax.device_get(leaf))))
            for leaf in loss_leaves
        ):
            raise ValueError(f"{label} smoke checkpoint has non-finite loss_info")
        integrity = checkpoint.get("transition_integrity")
        if not isinstance(integrity, dict) or any(
            int(np.asarray(value)) != 0 for value in integrity.values()
        ):
            raise ValueError(
                f"{label} smoke checkpoint failed transition integrity"
            )
    return {
        "schema": "terra_accepted_bank_smoke_validation_v1",
        "passed": True,
        "arm": arm,
        "next_update": 1,
        "periodic_checkpoint": {
            "path": str(periodic_path.resolve()),
            "sha256": sha256_file(periodic_path),
        },
        "final_checkpoint": {
            "path": str(final_path.resolve()),
            "sha256": sha256_file(final_path),
        },
    }


def compare_reset_hashes(cpu_path: Path, gpu_path: Path) -> dict:
    cpu = read_json(cpu_path)
    gpu = read_json(gpu_path)
    if cpu.get("backend") != "cpu":
        raise ValueError(f"{cpu_path}: expected CPU backend")
    if gpu.get("backend") != "gpu":
        raise ValueError(f"{gpu_path}: expected GPU backend")
    fields = (
        "panel",
        "slots",
        "terra_revision",
        "environment_protocol_sha256",
        "source_registry_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "episode_ids",
        "ordered_agent_state_sha256",
        "reset_verification",
    )
    mismatches = [field for field in fields if cpu.get(field) != gpu.get(field)]
    if mismatches:
        raise ValueError(
            "CPU/GPU reset-state parity failed for: " + ", ".join(mismatches)
        )
    return {
        "schema": "terra_accepted_bank_reset_parity_v1",
        "passed": True,
        "panel": cpu["panel"],
        "slots": cpu["slots"],
        "ordered_agent_state_sha256": cpu["ordered_agent_state_sha256"],
        "cpu_receipt": {
            "path": str(cpu_path.resolve()),
            "sha256": sha256_file(cpu_path),
        },
        "gpu_receipt": {
            "path": str(gpu_path.resolve()),
            "sha256": sha256_file(gpu_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--uniform", type=Path, required=True)
    select.add_argument("--adaptive", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    reset = subparsers.add_parser("reset-hashes")
    reset.add_argument("--checkpoint", type=Path, required=True)
    reset.add_argument("--bank-root", type=Path, required=True)
    reset.add_argument(
        "--panel",
        choices=("promotion", "development"),
        required=True,
    )
    reset.add_argument("--terra-revision", required=True)
    reset.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser("verify-smoke")
    smoke.add_argument("--arm", required=True)
    smoke.add_argument("--periodic", type=Path, required=True)
    smoke.add_argument("--final", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-reset-hashes")
    compare.add_argument("--cpu", type=Path, required=True)
    compare.add_argument("--gpu", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        payload = select_promotion(args.uniform, args.adaptive)
    elif args.command == "reset-hashes":
        payload = reset_hashes(args)
    elif args.command == "verify-smoke":
        payload = verify_smoke(args.arm, args.periodic, args.final)
    else:
        payload = compare_reset_hashes(args.cpu, args.gpu)
    write_json(args.output.resolve(), payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
