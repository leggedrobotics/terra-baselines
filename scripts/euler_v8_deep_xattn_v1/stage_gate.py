#!/usr/bin/env python3
"""Issue immutable per-arm V8 map-curriculum promotion receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "terra_v8_stage_gate_v1"
EVAL_SCHEMA = "terra_fixed_bank_eval_v4"
COMPLETION_CONTRACT = "exact_visible_dump_v1"
RELEASE_ID = "terra_v8_v6_constraints_v7_adjacent_train96_v5"
TERRA_REVISION = "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
BANK_ARCHIVE_SHA256 = "dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b"
BANK_DATASET_SHA256 = "715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798"
TRAINING_MIXTURE_SHA256 = (
    "f2a2a33556d513b46193a8a3996d37e6989534eba9373f46f52d79f956ac128e"
)
INITIAL_PARENT_SHA256 = (
    "4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc"
)
INITIAL_PARENT_PATH = (
    "/cluster/scratch/lterenzi/codex_terra_edge_runs/p5c_low_entropy_v1/"
    "3478af87950d3d35059344b078209d00785c8481/screen/s20260730/"
    "G-DEEP-UNIFORM-WARM/checkpoints/"
    "p5c_3478af87950d_screen_g_deep_uniform_warm_s20260730-euler-"
    "2026-08-03-00-39-08_update_004000.pkl"
)
REMOTE_RUN_ROOT = Path(
    "/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_deep_xattn_v1"
)
CAPABILITY_IDS = ("fnd-slab-allfree", "trn-straight-allfree")
STAGE_UPDATES = {
    "capability": tuple(range(500, 2001, 500)),
    "nearby": tuple(range(500, 4001, 500)),
    "full": tuple(range(500, 8001, 500)),
}
STAGE_SAMPLING_SHA256 = {
    "capability": "a569e04eba1bc2ed7cff9d084ff75c7a09224df6d600a4ab647a7b28c15f8633",
    "nearby": "b6e9e5d4fd672b87b4b87252b630d3243355e5d10988772f9861f3ec0cf0f245",
    "full": "989f379b038f71506a188ddf55e9789f79d94c1b537f76661e0d2d6af4653af3",
}
NEXT_STAGE = {"capability": "nearby", "nearby": "full", "full": "continuation"}
PRIOR_STAGE = {"nearby": "capability", "full": "nearby"}
EXPECTED_CONDITIONS = {"capability": 2, "nearby": 15, "full": 47}
FULL_PROGRESS_MIN_MACRO_GAIN = 0.001
FULL_GUARD_MAX_REGRESSION = 0.05
ARM_ARCHITECTURES = {
    "G-DEEP-V8-DENSE-WARM": {
        "label": "deep-se",
        "map_encoder": "resnet_spatial_8x8_se",
        "attention_compute_dtype": "encoder",
    },
    "G-DEEP-XATTN-V8-DENSE-WARM": {
        "label": "deep-se-xattn",
        "map_encoder": "resnet_spatial_8x8_se_xattn",
        "attention_compute_dtype": "float32",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _read_json(path: Path, expected_type):
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{path}: expected {expected_type.__name__}")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_remote_checkpoint_path(value: object, arm: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_./-]+", value) is None:
        raise ValueError("promoted checkpoint path contains unsupported characters")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.suffix != ".pkl":
        raise ValueError("promoted checkpoint path is not an absolute checkpoint")
    try:
        path.relative_to(REMOTE_RUN_ROOT)
    except ValueError as exc:
        raise ValueError("promoted checkpoint is outside the V8 run root") from exc
    if "checkpoints" not in path.parts or arm not in path.parts:
        raise ValueError("promoted checkpoint path does not match its V8 arm")
    return value


def parse_run_contract(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line or raw_line.startswith("#"):
            continue
        if "=" not in raw_line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = raw_line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate key {key!r}")
        values[key] = value
    return values


def load_bank_contract(bank_root: Path) -> dict:
    from terra.benchmark_protocol import canonical_json_sha256

    dataset_path = bank_root / "dataset.json"
    mixture_path = bank_root / "training_mixture.json"
    if sha256_file(dataset_path) != BANK_DATASET_SHA256:
        raise ValueError(f"{dataset_path}: frozen V8 dataset hash mismatch")
    if sha256_file(mixture_path) != TRAINING_MIXTURE_SHA256:
        raise ValueError(f"{mixture_path}: frozen V8 mixture hash mismatch")
    dataset = _read_json(dataset_path, dict)
    if dataset.get("release_id") != RELEASE_ID:
        raise ValueError(f"{dataset_path}: V8 release changed")
    if tuple(dataset.get("v6_capability_floor_condition_ids", ())) != CAPABILITY_IDS:
        raise ValueError(f"{dataset_path}: capability controls changed")
    core_ids = tuple(dataset.get("v7_core_condition_ids", ()))
    if len(core_ids) != 13 or len(set(core_ids)) != 13:
        raise ValueError(f"{dataset_path}: expected 13 unique V7 core conditions")
    main_ids = tuple(dataset.get("included_in_main_macro", ()))
    if (
        len(main_ids) != 45
        or len(set(main_ids)) != 45
        or not set(core_ids).issubset(main_ids)
    ):
        raise ValueError(f"{dataset_path}: expected 45 unique main conditions")
    family_by_condition = {
        entry["condition_id"]: entry["family"] for entry in dataset.get("train", ())
    }
    if set(core_ids) - set(family_by_condition):
        raise ValueError(f"{dataset_path}: core condition missing from train registry")
    if sum(family_by_condition[name] == "foundation" for name in core_ids) != 6:
        raise ValueError(f"{dataset_path}: expected six foundation core conditions")
    if sum(family_by_condition[name] == "trench" for name in core_ids) != 7:
        raise ValueError(f"{dataset_path}: expected seven trench core conditions")
    protocol_relative_path = dataset.get("environment_protocol")
    protocol_sha = _require_sha256(
        dataset.get("environment_protocol_sha256"),
        f"{dataset_path}: environment_protocol_sha256",
    )
    if not isinstance(protocol_relative_path, str):
        raise ValueError(f"{dataset_path}: invalid environment_protocol path")
    protocol_path = bank_root / protocol_relative_path
    protocol = _read_json(protocol_path, dict)
    protocol_payload = {
        key: value
        for key, value in protocol.items()
        if key != "environment_protocol_sha256"
    }
    if (
        protocol.get("environment_protocol_sha256") != protocol_sha
        or canonical_json_sha256(protocol_payload) != protocol_sha
    ):
        raise ValueError(f"{dataset_path}: frozen environment_protocol hash mismatch")
    if protocol.get("terra_revision") != TERRA_REVISION:
        raise ValueError(f"{dataset_path}: frozen Terra revision changed")
    registry_relative_path = dataset.get("source_registry")
    registry_sha = _require_sha256(
        dataset.get("source_registry_sha256"),
        f"{dataset_path}: source_registry_sha256",
    )
    if (
        not isinstance(registry_relative_path, str)
        or sha256_file(bank_root / registry_relative_path) != registry_sha
    ):
        raise ValueError(f"{dataset_path}: frozen source_registry hash mismatch")
    return {
        "root": bank_root,
        "dataset": dataset,
        "core_ids": core_ids,
        "main_ids": main_ids,
        "family_by_condition": family_by_condition,
        "environment_protocol_sha256": dataset["environment_protocol_sha256"],
        "source_registry_sha256": dataset["source_registry_sha256"],
    }


def panel_contract(bank: dict, panel_group: str, split: str) -> dict:
    import jax
    import numpy as np

    dataset = bank["dataset"]
    group_field = (
        "capability_floor_evaluation_panels"
        if panel_group == "capability"
        else "evaluation_panels"
    )
    try:
        panel = dataset[group_field][split]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"frozen bank lacks {panel_group} {split} panel") from exc
    directory = (bank["root"] / panel["maps_path"]).resolve()
    manifest = directory / "manifest.jsonl"
    slots = int(panel["slot_count"])
    layer_hashes = {}
    for field, subdirectory in {
        "target": "images",
        "initial_action": "actions",
        "occupancy": "occupancy",
        "dumpability": "dumpability",
        "distance": "distance",
        "metadata": "metadata",
    }.items():
        digest = hashlib.sha256()
        for index in range(1, slots + 1):
            filename = (
                f"trench_{index}.json" if field == "metadata" else f"img_{index}.npy"
            )
            digest.update((directory / subdirectory / filename).read_bytes())
        layer_hashes[field] = digest.hexdigest()
    reset_verification = {
        "passed": True,
        "slots": slots,
        "env_steps_min": 0,
        "env_steps_max": 0,
        "verified_fields": [
            "target",
            "initial_action",
            "occupancy",
            "dumpability",
            "distance",
            "trench_axes",
            "trench_type",
            "foundation_border_axes",
            "foundation_border_type",
        ],
        "layer_sha256": layer_hashes,
    }
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    seeds = np.asarray([row["reset_seed"] for row in rows], dtype=np.uint32)
    seed_keys = jax.vmap(jax.random.PRNGKey)(seeds)
    reset_verification["manifest_episode_seeds"] = {
        "passed": True,
        "map_selection_decoupled": True,
        "sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(seed_keys)).tobytes()
        ).hexdigest(),
    }
    return {
        "bank_root": str(bank["root"]),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "slots": slots,
        "conditions": int(panel["conditions"]),
        "stratum": "capability" if panel_group == "capability" else "all",
        "reset_verification": reset_verification,
    }


def stage_sampling_contract(
    bank: dict,
    stage: str,
    sampler_profile: str = "bank_v4",
) -> dict:
    import numpy as np

    from utils.accepted_bank import load_accepted_bank

    accepted = load_accepted_bank(
        bank["root"],
        "G-UNIFORM",
        TERRA_REVISION,
        curriculum_stage=stage,
        sampler_profile=sampler_profile,
    )
    conditions = [level.condition_id for level in accepted.levels]
    weights = np.asarray(accepted.sampling_probabilities, dtype=np.float64)
    probabilities = (weights / weights.sum()).tolist()
    contract = {
        "stage": stage,
        "conditions": conditions,
        "declared_weights": weights.tolist(),
        "probabilities": probabilities,
        "maps_per_condition": accepted.map_count_per_condition,
    }
    if stage != "capability":
        contract["sampler_profile"] = sampler_profile
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != STAGE_SAMPLING_SHA256[stage]:
        raise ValueError(f"frozen {stage} sampling contract changed")
    return {**contract, "sha256": digest}


def validate_run_contract(
    contract: dict[str, str], stage: str, arm: str, prior_receipt: dict | None
) -> None:
    sampler_profile = "bank_v4" if stage == "capability" else "bounded_replay25_v1"
    expected = {
        "arm": arm,
        "curriculum_stage": stage,
        "reward_stage": "dense_skill",
        "reward_type": "DENSE",
        "condition_sampler": (
            "fixed_v8_stage_weights"
            if stage == "capability"
            else "fixed_v8_bounded_replay"
        ),
        "sampler_profile": sampler_profile,
        "condition_count": str(EXPECTED_CONDITIONS[stage]),
        "phase": "screen",
        "terra_revision": TERRA_REVISION,
        "training_bank_release_id": RELEASE_ID,
        "training_bank_archive_sha256": BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": BANK_DATASET_SHA256,
        "trench_shaping": "false",
        "horizon": "450",
        "full_resets": "true",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"run contract {field} must be {value!r}, got {contract.get(field)!r}"
            )
    seed = contract.get("seed")
    if not isinstance(seed, str) or not seed.isdigit():
        raise ValueError("run contract seed must be a nonnegative integer")
    updates = STAGE_UPDATES[stage][-1]
    if contract.get("updates") != str(updates):
        raise ValueError("run contract update budget changed")
    expected_transitions = 4 * 1024 * 32 * updates
    if contract.get("global_transitions") != str(expected_transitions):
        raise ValueError("run contract transition budget changed")
    if contract.get("architecture") != ARM_ARCHITECTURES[arm]["label"]:
        raise ValueError("run contract architecture label changed")
    revision = contract.get("terra_baselines_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("run contract lacks an immutable terra-baselines revision")
    for field in (
        "parent_checkpoint_sha256",
        "initial_checkpoint_sha256",
    ):
        _require_sha256(contract.get(field), f"run contract {field}")
    if stage == "capability":
        if contract.get("parent_checkpoint_path") != INITIAL_PARENT_PATH:
            raise ValueError("Stage A did not use the frozen P5c parent path")
        if contract["parent_checkpoint_sha256"] != INITIAL_PARENT_SHA256:
            raise ValueError("Stage A did not use the frozen P5c parent")
        if contract.get("teacher_checkpoint_sha256") != INITIAL_PARENT_SHA256:
            raise ValueError("Stage A did not use the frozen P5c teacher")
        if contract.get("initialization") != "params_only_warm_fresh_optimizer":
            raise ValueError("Stage A initialization contract changed")
    else:
        assert prior_receipt is not None
        if contract.get("parent_checkpoint_path") != prior_receipt["candidate"]["path"]:
            raise ValueError(
                "later stage does not start from the prior promoted checkpoint path"
            )
        prior_sha = prior_receipt["candidate"]["checkpoint_sha256"]
        if contract["parent_checkpoint_sha256"] != prior_sha:
            raise ValueError(
                "later stage does not start from the prior promoted checkpoint"
            )
        if contract.get("teacher_checkpoint_sha256") != "none":
            raise ValueError("later map stages must disable teacher kickstart")
        if contract.get("initialization") != (
            "params_only_stage_transition_fresh_optimizer"
        ):
            raise ValueError("later-stage initialization contract changed")


def _architecture_from_record(record: dict) -> dict:
    try:
        return record["treatment_fingerprint"]["contract"]["architecture"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "evaluation record lacks its architecture fingerprint"
        ) from exc


def _validate_architecture(architecture: dict, arm: str) -> None:
    arm_contract = ARM_ARCHITECTURES[arm]
    expected = {
        "model_size": "medium",
        "model_core": "mlp",
        "map_encoder": arm_contract["map_encoder"],
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": arm_contract["attention_compute_dtype"],
        "token_mixer_residual_init_scale": 0.0,
        "critic_hidden_dims": [512, 256],
        "resnet_stage_channels": [24, 48, 64, 96],
        "resnet_blocks_per_stage": [2, 2, 3, 3],
        "loaded_max": 100,
    }
    for field, value in expected.items():
        observed = architecture.get(field)
        if isinstance(observed, tuple):
            observed = list(observed)
        if observed != value:
            raise ValueError(
                f"{arm} architecture {field} must be {value!r}, got {observed!r}"
            )


def _validate_treatment(
    record: dict,
    bank: dict,
    stage: str,
    arm: str,
    seed: int,
    sampling: dict,
) -> None:
    fingerprint = record.get("treatment_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("evaluation record lacks its treatment fingerprint")
    contract = fingerprint.get("contract")
    if not isinstance(contract, dict) or contract.get("schema") != (
        "terra_fixed_bank_treatment_v1"
    ):
        raise ValueError("evaluation treatment schema changed")
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if fingerprint.get("sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("evaluation treatment fingerprint hash mismatch")
    run = contract.get("run", {})
    expected_run = {
        "seed": seed,
        "config_name": "G-V8-FIXED",
        "accepted_bank_arm": "G-UNIFORM",
    }
    for field, value in expected_run.items():
        if run.get(field) != value:
            raise ValueError(f"evaluation treatment run {field} changed")
    normalized_arm = arm.lower().replace("-", "_")
    name = run.get("name")
    if not isinstance(name, str) or normalized_arm not in name or stage not in name:
        raise ValueError("evaluation run name does not identify its arm and stage")
    expected_bank = {
        "terra_revision": TERRA_REVISION,
        "environment_protocol_sha256": bank["environment_protocol_sha256"],
        "source_registry_sha256": bank["source_registry_sha256"],
    }
    if contract.get("bank") != expected_bank:
        raise ValueError("evaluation treatment bank identity changed")
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
        raise ValueError("evaluation PPO treatment changed")
    train_by_id = {
        entry["condition_id"]: entry for entry in bank["dataset"].get("train", ())
    }
    expected_levels = [
        {
            "maps_path": train_by_id[condition_id]["maps_path"],
            "max_steps_in_episode": 450,
            "rewards_type": 0,
            "apply_trench_rewards": False,
        }
        for condition_id in sampling["conditions"]
    ]
    expected_reward = {
        "agent_types": [0],
        "action_types": [0],
        "relocation_progress_mult": 1.5,
        "curriculum_levels": expected_levels,
    }
    if contract.get("reward_action") != expected_reward:
        raise ValueError("evaluation reward/action/map treatment changed")
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
        "seed": seed,
    }
    if contract.get("sampler") != expected_sampler:
        raise ValueError("evaluation fixed sampler treatment changed")
    _validate_architecture(contract.get("architecture", {}), arm)


def validate_evaluation(
    path: Path,
    stage: str,
    arm: str,
    bank: dict,
    run_contract: dict[str, str],
    expected_panel: dict,
    sampling: dict,
    expected_split: str = "promotion",
) -> list[dict]:
    records = _read_json(path, list)
    expected_updates = STAGE_UPDATES[stage]
    observed_updates = tuple(record.get("checkpoint_update") for record in records)
    valid_updates = (
        len(observed_updates) >= 2
        and observed_updates == expected_updates[: len(observed_updates)]
        if stage == "full"
        else observed_updates == expected_updates
    )
    if not valid_updates:
        raise ValueError(
            f"{path}: expected checkpoint prefix of {expected_updates}, got "
            f"{observed_updates}"
        )
    reference_architecture = None
    reference_name = None
    for record in records:
        if record.get("schema") != EVAL_SCHEMA:
            raise ValueError(f"{path}: unsupported fixed-evaluation schema")
        expected_fields = {
            "completion_contract": COMPLETION_CONTRACT,
            "horizon": 450,
            "deterministic": True,
            "policy_mode": "deterministic",
            "exact_manifest_enumeration": True,
            "split": expected_split,
            "stratum": expected_panel["stratum"],
            "bank_root": expected_panel["bank_root"],
            "manifest": expected_panel["manifest"],
            "manifest_sha256": expected_panel["manifest_sha256"],
            "seed": int(run_contract["seed"]),
            "reset_verification": expected_panel["reset_verification"],
        }
        for field, value in expected_fields.items():
            if record.get(field) != value:
                raise ValueError(f"{path}: {field} must be {value!r}")
        checkpoint = Path(record.get("checkpoint", ""))
        if not checkpoint.is_absolute():
            raise ValueError(f"{path}: checkpoint paths must be absolute")
        _require_sha256(record.get("checkpoint_sha256"), "evaluation checkpoint hash")
        accepted_bank = record.get("accepted_bank")
        expected_accepted_bank = {
            "schema": "terra_curriculum_loader_bank_v1",
            "terra_revision": TERRA_REVISION,
            "environment_protocol_sha256": bank["environment_protocol_sha256"],
            "source_registry_sha256": bank["source_registry_sha256"],
            "diagnostic_control": False,
            "diagnostic_contract_sha256": None,
        }
        if accepted_bank != expected_accepted_bank:
            raise ValueError(f"{path}: accepted-bank identity changed")
        _validate_treatment(
            record,
            bank,
            stage,
            arm,
            int(run_contract["seed"]),
            sampling,
        )
        architecture = _architecture_from_record(record)
        _validate_architecture(architecture, arm)
        name = (
            record.get("treatment_fingerprint", {})
            .get("contract", {})
            .get("run", {})
            .get("name")
        )
        if reference_architecture is None:
            reference_architecture = architecture
            reference_name = name
        elif architecture != reference_architecture or name != reference_name:
            raise ValueError(f"{path}: treatment changed between checkpoints")
    return records


def _cell_counts(record: dict, condition_ids: tuple[str, ...]) -> dict[str, int]:
    try:
        cells = record["summary"]["by_primary_cell"]
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation summary lacks by_primary_cell") from exc
    result = {}
    for condition_id in condition_ids:
        cell = cells.get(condition_id)
        if not isinstance(cell, dict) or cell.get("episodes") != 16:
            raise ValueError(f"{condition_id}: expected exactly 16 promotion episodes")
        successes = cell.get("successes")
        if not isinstance(successes, int) or not 0 <= successes <= 16:
            raise ValueError(f"{condition_id}: invalid exact-success count")
        result[condition_id] = successes
    return result


def validate_panel_conditions(
    records: list[dict], condition_ids: tuple[str, ...]
) -> None:
    expected = set(condition_ids)
    for record in records:
        try:
            cells = record["summary"]["by_primary_cell"]
            overall_episodes = record["summary"]["overall"]["episodes"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "evaluation summary lacks its exact panel accounting"
            ) from exc
        if set(cells) != expected:
            raise ValueError(
                "evaluation panel condition set does not match the frozen bank"
            )
        if overall_episodes != 16 * len(expected):
            raise ValueError(
                "evaluation panel episode count does not match the frozen bank"
            )


def _integrity_passed(record: dict) -> bool:
    try:
        return record["summary"]["integrity"]["passed"] is True
    except (KeyError, TypeError):
        return False


def _checkpoint_identity(record: dict) -> tuple[str, str, int]:
    return (
        record["checkpoint"],
        record["checkpoint_sha256"],
        int(record["checkpoint_update"]),
    )


def _load_checkpoint(path: Path) -> dict:
    # Current checkpoints pickle script-defined config dataclasses as __main__.
    from train import TrainConfig
    from train_mixed import MixedAgentTrainConfig

    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "TrainConfig"):
        main_module.TrainConfig = TrainConfig
    if not hasattr(main_module, "MixedAgentTrainConfig"):
        main_module.MixedAgentTrainConfig = MixedAgentTrainConfig
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: checkpoint must contain a mapping")
    return checkpoint


def _validate_sampler_state(
    state: object, sampling: dict, seed: int, checkpoint_path: Path
) -> None:
    import numpy as np

    if not isinstance(state, dict) or state.get("schema") != (
        "terra_pooled_condition_sampler_state_v1"
    ):
        raise ValueError(f"{checkpoint_path}: fixed sampler state is missing")
    if state.get("conditions") != sampling["conditions"]:
        raise ValueError(f"{checkpoint_path}: fixed sampler condition order changed")
    expected_settings = {
        "rule": "fixed",
        "update_interval": 150,
        "uniform_floor": 0.20,
        "mastery_threshold": 0.75,
        "temperature": 0.25,
        "min_episodes": 20,
        "competence_ema": 0.30,
        "max_mass": 0.15,
        "seed": seed,
    }
    if state.get("settings") != expected_settings:
        raise ValueError(f"{checkpoint_path}: fixed sampler settings changed")
    expected_maps = [sampling["maps_per_condition"]] * len(sampling["conditions"])
    if state.get("maps_per_condition") != expected_maps:
        raise ValueError(f"{checkpoint_path}: fixed sampler map counts changed")
    observed = state.get("probabilities")
    expected = sampling["probabilities"]
    if (
        not isinstance(observed, list)
        or len(observed) != len(expected)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in observed
        )
        or not np.allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError(f"{checkpoint_path}: fixed sampler probabilities changed")
    labels = state.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{checkpoint_path}: fixed sampler labels are missing")
    for condition_id, weight_value in zip(
        sampling["conditions"], sampling["declared_weights"]
    ):
        label = labels.get(condition_id)
        weight = label.get("sampling_weight") if isinstance(label, dict) else None
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not np.isclose(weight, weight_value, rtol=0.0, atol=1e-15)
        ):
            raise ValueError(
                f"{checkpoint_path}: fixed sampler weight changed for {condition_id}"
            )


def validate_candidate_checkpoint(
    record: dict,
    stage: str,
    arm: str,
    run_contract: dict[str, str] | None = None,
    sampling: dict | None = None,
) -> dict:
    path = Path(record["checkpoint"])
    if run_contract is not None:
        _validate_remote_checkpoint_path(str(path), arm)
        required_parts = {
            run_contract["terra_baselines_revision"],
            "screen",
            stage,
            f"s{run_contract['seed']}",
        }
        if not required_parts.issubset(path.parts):
            raise ValueError(f"{path}: checkpoint path changed campaign identity")
    observed_sha = sha256_file(path)
    if observed_sha != record["checkpoint_sha256"]:
        raise ValueError(f"{path}: checkpoint hash does not match fixed evaluation")
    checkpoint = _load_checkpoint(path)
    if "model" not in checkpoint:
        raise ValueError(f"{path}: checkpoint has no model parameters")
    update = int(record["checkpoint_update"])
    if checkpoint.get("next_update") != update:
        raise ValueError(f"{path}: checkpoint next_update does not match evaluation")
    config = checkpoint.get("train_config")
    if config is None:
        raise ValueError(f"{path}: checkpoint has no train_config")
    bank = _field(config, "accepted_bank")
    expected_bank = {
        "release_id": RELEASE_ID,
        "terra_revision": TERRA_REVISION,
        "curriculum_stage": stage,
        "sampler_profile": (
            "bank_v4" if stage == "capability" else "bounded_replay25_v1"
        ),
    }
    for field, value in expected_bank.items():
        observed = _field(bank, field)
        if field == "sampler_profile" and stage == "capability" and observed is None:
            # Stage A predates the named profile; bank_v4 is its exact legacy
            # probability vector and its receipt hash remains unchanged.
            continue
        if observed != value:
            raise ValueError(f"{path}: accepted-bank {field} changed")
    architecture = {
        field: _field(config, field)
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
    _validate_architecture(architecture, arm)
    if run_contract is not None:
        if sampling is None:
            raise ValueError("candidate validation requires the frozen sampler")
        from eval_fixed_bank import checkpoint_treatment_fingerprint

        if checkpoint_treatment_fingerprint(checkpoint) != record.get(
            "treatment_fingerprint"
        ):
            raise ValueError(f"{path}: checkpoint treatment differs from evaluation")
        parent_path = run_contract["parent_checkpoint_path"]
        warm_start_from = _field(config, "warm_start_from")
        teacher_checkpoint = _field(config, "teacher_checkpoint")
        if stage == "capability":
            if teacher_checkpoint != parent_path:
                raise ValueError(f"{path}: Stage-A teacher differs from its parent")
            if arm == "G-DEEP-V8-DENSE-WARM" and warm_start_from != parent_path:
                raise ValueError(f"{path}: deep Stage-A warm start changed")
        elif warm_start_from != parent_path or teacher_checkpoint is not None:
            raise ValueError(f"{path}: map-stage transition warm start changed")
        _validate_sampler_state(
            checkpoint.get("pooled_sampler_state"),
            sampling,
            int(run_contract["seed"]),
            path,
        )
    return {
        "path": str(path),
        "checkpoint_sha256": observed_sha,
        "next_update": update,
        "architecture": ARM_ARCHITECTURES[arm]["label"],
        "map_encoder": ARM_ARCHITECTURES[arm]["map_encoder"],
        "curriculum_stage": stage,
        "warm_start_from": _field(config, "warm_start_from"),
        "teacher_checkpoint": _field(config, "teacher_checkpoint"),
    }


def validate_prior_receipt(path: Path, arm: str, expected_stage: str) -> dict:
    receipt = _read_json(path, dict)
    expected = {
        "schema": SCHEMA,
        "passed": True,
        "stage": expected_stage,
        "next_stage": NEXT_STAGE[expected_stage],
        "arm": arm,
        "release_id": RELEASE_ID,
        "terra_revision": TERRA_REVISION,
        "bank_archive_sha256": BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": BANK_DATASET_SHA256,
        "training_mixture_sha256": TRAINING_MIXTURE_SHA256,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"{path}: prior receipt {field} must be {value!r}")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{path}: prior receipt lacks its candidate")
    _require_sha256(candidate.get("checkpoint_sha256"), "prior candidate hash")
    _validate_remote_checkpoint_path(candidate.get("path"), arm)
    scheduled_updates = list(STAGE_UPDATES[expected_stage])
    if receipt.get("scheduled_updates") != scheduled_updates:
        raise ValueError(f"{path}: prior scheduled updates changed")
    if expected_stage == "full":
        evaluated_updates = receipt.get("evaluated_updates")
        if (
            not isinstance(evaluated_updates, list)
            or len(evaluated_updates) < 2
            or evaluated_updates != scheduled_updates[: len(evaluated_updates)]
        ):
            raise ValueError(
                f"{path}: full receipt lacks a contiguous evaluated prefix"
            )
        candidate_update = evaluated_updates[-1]
        expected_pair = evaluated_updates[-2:]
    else:
        evaluated_updates = scheduled_updates
        candidate_update = scheduled_updates[-1]
        expected_pair = scheduled_updates[-2:]
    expected_candidate = {
        "next_update": candidate_update,
        "architecture": ARM_ARCHITECTURES[arm]["label"],
        "map_encoder": ARM_ARCHITECTURES[arm]["map_encoder"],
        "curriculum_stage": expected_stage,
    }
    for field, value in expected_candidate.items():
        if candidate.get(field) != value:
            raise ValueError(f"{path}: prior candidate {field} must be {value!r}")
    sampling = receipt.get("sampling")
    sampling_payload = (
        {key: value for key, value in sampling.items() if key != "sha256"}
        if isinstance(sampling, dict)
        else None
    )
    sampling_digest = (
        hashlib.sha256(
            json.dumps(
                sampling_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if sampling_payload is not None
        else None
    )
    if (
        not isinstance(sampling, dict)
        or sampling.get("stage") != expected_stage
        or sampling.get("sha256") != STAGE_SAMPLING_SHA256[expected_stage]
        or sampling_digest != STAGE_SAMPLING_SHA256[expected_stage]
        or sampling.get("maps_per_condition") != 96
        or len(sampling.get("conditions", ())) != EXPECTED_CONDITIONS[expected_stage]
    ):
        raise ValueError(f"{path}: prior sampling contract changed")
    if receipt.get("evaluated_pair") != expected_pair:
        raise ValueError(f"{path}: prior gate did not use the latest checkpoint pair")
    thresholds = receipt.get("retention", {}).get("frozen_thresholds")
    mastery = receipt.get("mastery", {})
    if not isinstance(thresholds, dict) or not isinstance(mastery, dict):
        raise ValueError(f"{path}: prior receipt lacks frozen mastery thresholds")
    if set(mastery) != set(sampling["conditions"]):
        raise ValueError(f"{path}: prior mastery set changed")
    if receipt.get("retention", {}).get("rollback_triggered") is not False:
        raise ValueError(f"{path}: rollback-triggered receipt cannot promote")
    if receipt.get("integrity_pair") != [True, True]:
        raise ValueError(f"{path}: prior receipt did not pass integrity twice")
    expected_thresholds = {}
    for condition_id, values in mastery.items():
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(
                not isinstance(value, int) or not 0 <= value <= 16 for value in values
            )
        ):
            raise ValueError(f"{path}: invalid mastery pair for {condition_id}")
        floor = 12 if condition_id in CAPABILITY_IDS else 11
        expected_thresholds[condition_id] = max(floor, min(values) - 1)
    if expected_stage == "full":
        inherited = receipt.get("retention", {}).get("inherited_thresholds")
        if (
            not isinstance(inherited, dict)
            or thresholds != inherited
            or not set(thresholds).issubset(mastery)
            or len(thresholds) != EXPECTED_CONDITIONS["nearby"]
        ):
            raise ValueError(f"{path}: full inherited retention thresholds changed")
        qualification = receipt.get("continuation_qualification")
        if (
            not isinstance(qualification, dict)
            or qualification.get("qualified_for_120h") is not True
            or qualification.get("candidate_update") != candidate_update
            or qualification.get("promotion", {}).get("progress_passed") is not True
            or qualification.get("promotion", {}).get("guards_passed") is not True
            or qualification.get("development", {}).get("guards_passed") is not True
            or qualification.get("integrity_passed_all_panels") is not True
            or qualification.get("inherited_retention_passed") is not True
        ):
            raise ValueError(f"{path}: full receipt is not qualified for continuation")
        evaluated_checkpoints = receipt.get("evaluated_checkpoints")
        if not isinstance(evaluated_checkpoints, list) or len(
            evaluated_checkpoints
        ) != len(evaluated_updates):
            raise ValueError(
                f"{path}: full receipt lacks evaluated checkpoint identities"
            )
        for update, checkpoint in zip(evaluated_updates, evaluated_checkpoints):
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("next_update") != update
            ):
                raise ValueError(f"{path}: full evaluated checkpoint order changed")
            _validate_remote_checkpoint_path(checkpoint.get("path"), arm)
            _require_sha256(
                checkpoint.get("checkpoint_sha256"),
                f"{path}: full evaluated checkpoint hash",
            )
        if evaluated_checkpoints[-1] != candidate:
            raise ValueError(f"{path}: full candidate is not the evaluated tail")
        checkpoints_by_update = {
            checkpoint["next_update"]: checkpoint
            for checkpoint in evaluated_checkpoints
        }
        reference_update = qualification.get("reference_update")
        if (
            reference_update not in checkpoints_by_update
            or qualification.get("reference_checkpoint_sha256")
            != checkpoints_by_update[reference_update]["checkpoint_sha256"]
            or qualification.get("candidate_checkpoint_sha256")
            != candidate["checkpoint_sha256"]
        ):
            raise ValueError(f"{path}: full continuation checkpoint identity changed")
        parent_job = receipt.get("parent_slurm_job")
        run_contract_input = receipt.get("inputs", {}).get("run_contract")
        if (
            not isinstance(parent_job, dict)
            or parent_job.get("schema") != "terra_v8_parent_slurm_job_v1"
            or parent_job.get("state") not in ("COMPLETED", "TIMEOUT")
            or parent_job.get("checkpoint_updates") != evaluated_updates
            or parent_job.get("partition") != "gpuhe.24h"
            or parent_job.get("terra_baselines_revision")
            != receipt.get("terra_baselines_revision")
            or re.fullmatch(r"[0-9]+", str(parent_job.get("evaluator_job_id"))) is None
            or not isinstance(run_contract_input, dict)
            or parent_job.get("run_contract") != run_contract_input
        ):
            raise ValueError(f"{path}: full parent Slurm identity changed")
        parent_run_dir = Path(str(parent_job.get("run_dir")))
        if run_contract_input.get("path") != str(
            parent_run_dir / "run_contract.env"
        ) or len(parent_job.get("checkpoints", ())) != len(evaluated_checkpoints):
            raise ValueError(f"{path}: full parent run-contract binding changed")
        for parent_checkpoint, evaluated in zip(
            parent_job["checkpoints"], evaluated_checkpoints
        ):
            expected_parent = {
                "update": evaluated["next_update"],
                "path": evaluated["path"],
                "sha256": evaluated["checkpoint_sha256"],
            }
            if (
                parent_checkpoint != expected_parent
                or Path(parent_checkpoint["path"]).parent
                != parent_run_dir / "checkpoints"
            ):
                raise ValueError(f"{path}: full parent checkpoint binding changed")
    elif thresholds != expected_thresholds:
        raise ValueError(f"{path}: prior retention thresholds were modified")
    return receipt


def decide_capability(records: list[dict]) -> dict:
    pair = records[-2:]
    mastery = {
        condition_id: [
            _cell_counts(record, CAPABILITY_IDS)[condition_id] for record in pair
        ]
        for condition_id in CAPABILITY_IDS
    }
    integrity = [_integrity_passed(record) for record in pair]
    passed = all(integrity) and all(min(values) >= 12 for values in mastery.values())
    thresholds = (
        {
            condition_id: max(12, min(values) - 1)
            for condition_id, values in mastery.items()
        }
        if passed
        else {}
    )
    reason = (
        "latest two checkpoints master both capability controls"
        if passed
        else "latest two checkpoints do not both master both capability controls"
    )
    return {
        "passed": passed,
        "reason": reason,
        "mastery": mastery,
        "integrity_pair": integrity,
        "inherited_thresholds": {},
        "new_thresholds": thresholds,
        "frozen_thresholds": thresholds,
        "failures": {},
        "rollback_triggered": False,
    }


def _retention_audit(
    mastery_history: dict[str, list[int]],
    inherited: dict[str, int],
    updates: list[int],
) -> dict:
    if set(mastery_history) != set(inherited):
        raise ValueError("retention history does not match inherited conditions")
    if any(len(values) != len(updates) for values in mastery_history.values()):
        raise ValueError("retention history length does not match evaluations")
    failures = {
        condition_id: [value < inherited[condition_id] for value in values]
        for condition_id, values in mastery_history.items()
    }
    failure_history = [
        any(values[index] for values in failures.values())
        for index in range(len(updates))
    ]
    rollback_index = next(
        (
            index
            for index in range(1, len(failure_history))
            if failure_history[index - 1] and failure_history[index]
        ),
        None,
    )
    return {
        "failures": failures,
        "failure_history": failure_history,
        "rollback_triggered": rollback_index is not None,
        "rollback_updates": (
            None
            if rollback_index is None
            else [updates[rollback_index - 1], updates[rollback_index]]
        ),
        "latest_pair_passed": [not failed for failed in failure_history[-2:]],
    }


def decide_nearby(
    records: list[dict],
    capability_records: list[dict],
    prior_receipt: dict,
    core_ids: tuple[str, ...],
    family_by_condition: dict[str, str],
) -> dict:
    pair = records[-2:]
    capability_pair = capability_records[-2:]
    for main, capability in zip(records, capability_records):
        if _checkpoint_identity(main) != _checkpoint_identity(capability):
            raise ValueError(
                "nearby main and capability panels name different checkpoints"
            )
    core_mastery = {
        condition_id: [_cell_counts(record, core_ids)[condition_id] for record in pair]
        for condition_id in core_ids
    }
    capability_history = {
        condition_id: [
            _cell_counts(record, CAPABILITY_IDS)[condition_id]
            for record in capability_records
        ]
        for condition_id in CAPABILITY_IDS
    }
    capability_mastery = {
        condition_id: values[-2:] for condition_id, values in capability_history.items()
    }
    inherited = prior_receipt["retention"]["frozen_thresholds"]
    retention = _retention_audit(
        capability_history,
        inherited,
        [record["checkpoint_update"] for record in capability_records],
    )
    core_pass = []
    family_totals = []
    for index in range(2):
        foundation = sum(
            values[index]
            for condition_id, values in core_mastery.items()
            if family_by_condition[condition_id] == "foundation"
        )
        trench = sum(
            values[index]
            for condition_id, values in core_mastery.items()
            if family_by_condition[condition_id] == "trench"
        )
        family_totals.append({"foundation": foundation, "trench": trench})
        core_pass.append(
            foundation >= 78
            and trench >= 91
            and all(values[index] >= 12 for values in core_mastery.values())
        )
    capability_pass = retention["latest_pair_passed"]
    integrity = [
        _integrity_passed(main) and _integrity_passed(capability)
        for main, capability in zip(pair, capability_pair)
    ]
    passed = (
        all(core_pass)
        and all(capability_pass)
        and all(integrity)
        and not retention["rollback_triggered"]
    )
    new_thresholds = (
        {
            condition_id: max(11, min(values) - 1)
            for condition_id, values in core_mastery.items()
        }
        if passed
        else {}
    )
    frozen = {**inherited, **new_thresholds} if passed else dict(inherited)
    reason = (
        "latest two checkpoints master nearby core and retain capability"
        if passed
        else "latest two checkpoints do not both master nearby core and retain capability"
    )
    return {
        "passed": passed,
        "reason": reason,
        "mastery": {**capability_mastery, **core_mastery},
        "family_totals": family_totals,
        "integrity_pair": integrity,
        "inherited_thresholds": inherited,
        "new_thresholds": new_thresholds,
        "frozen_thresholds": frozen,
        "failures": retention["failures"],
        "retention_failure_history": retention["failure_history"],
        "rollback_updates": retention["rollback_updates"],
        "rollback_triggered": retention["rollback_triggered"],
    }


def decide_full(
    records: list[dict],
    capability_records: list[dict],
    prior_receipt: dict,
    main_ids: tuple[str, ...],
    family_by_condition: dict[str, str],
) -> dict:
    for main, capability in zip(records, capability_records):
        if _checkpoint_identity(main) != _checkpoint_identity(capability):
            raise ValueError(
                "full main and capability panels name different checkpoints"
            )
    main_history = {
        condition_id: [
            _cell_counts(record, main_ids)[condition_id] for record in records
        ]
        for condition_id in main_ids
    }
    capability_history = {
        condition_id: [
            _cell_counts(record, CAPABILITY_IDS)[condition_id]
            for record in capability_records
        ]
        for condition_id in CAPABILITY_IDS
    }
    all_history = {**capability_history, **main_history}
    inherited = prior_receipt["retention"]["frozen_thresholds"]
    inherited_history = {
        condition_id: all_history[condition_id] for condition_id in inherited
    }
    updates = [record["checkpoint_update"] for record in records]
    retention = _retention_audit(inherited_history, inherited, updates)
    integrity_history = [
        _integrity_passed(main) and _integrity_passed(capability)
        for main, capability in zip(records, capability_records)
    ]
    latest_mastery = {
        condition_id: values[-2:] for condition_id, values in all_history.items()
    }
    family_totals = []
    for index in (-2, -1):
        family_totals.append(
            {
                family: sum(
                    values[index]
                    for condition_id, values in main_history.items()
                    if family_by_condition[condition_id] == family
                )
                for family in ("foundation", "trench")
            }
        )
    passed = (
        all(retention["latest_pair_passed"])
        and not retention["rollback_triggered"]
        and all(integrity_history)
    )
    return {
        "passed": passed,
        "reason": (
            "full-V8 screen retains all inherited conditions without integrity loss"
            if passed
            else "full-V8 screen violates inherited retention or integrity"
        ),
        "mastery": latest_mastery,
        "family_totals": family_totals,
        "integrity_pair": integrity_history[-2:],
        "integrity_history": integrity_history,
        "inherited_thresholds": inherited,
        "new_thresholds": {},
        "frozen_thresholds": inherited,
        "failures": retention["failures"],
        "retention_failure_history": retention["failure_history"],
        "rollback_updates": retention["rollback_updates"],
        "rollback_triggered": retention["rollback_triggered"],
    }


def _full_progress_snapshot(record: dict, v6_ids: tuple[str, ...]) -> dict:
    """Return the small set of fixed-bank metrics used to award long compute."""
    try:
        exact_cells = record["summary"]["by_primary_cell"]
        graded = record["summary"]["graded"]
        graded_cells = graded["by_primary_cell"]
        by_family = graded["by_family"]
        values = {
            "v6_exact_successes": sum(
                int(exact_cells[condition_id]["successes"]) for condition_id in v6_ids
            ),
            "v6_macro_completion": sum(
                float(graded_cells[condition_id]["mean"]) for condition_id in v6_ids
            )
            / len(v6_ids),
            "foundation_macro_completion": float(
                by_family["foundation"]["macro_completion"]
            ),
            "trench_macro_completion": float(by_family["trench"]["macro_completion"]),
            "micro_p10": float(graded["micro"]["p10"]),
            "worst_condition_completion": float(graded["worst_condition_completion"]),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("full evaluation lacks graded V6 progress metrics") from exc
    if graded.get("available") is not True or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise ValueError("full evaluation has unavailable or non-finite graded metrics")
    return values


def _full_progress_comparison(reference: dict, candidate: dict) -> dict:
    deltas = {
        key: candidate[key] - reference[key]
        for key in (
            "v6_exact_successes",
            "v6_macro_completion",
            "foundation_macro_completion",
            "trench_macro_completion",
            "micro_p10",
            "worst_condition_completion",
        )
    }
    progress_passed = (
        deltas["v6_exact_successes"] >= 1
        or deltas["v6_macro_completion"] >= FULL_PROGRESS_MIN_MACRO_GAIN
    )
    guards_passed = all(
        deltas[key] >= -FULL_GUARD_MAX_REGRESSION
        for key in (
            "foundation_macro_completion",
            "trench_macro_completion",
            "micro_p10",
            "worst_condition_completion",
        )
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "deltas": deltas,
        "required_v6_exact_gain": 1,
        "required_v6_macro_gain": FULL_PROGRESS_MIN_MACRO_GAIN,
        "max_guard_regression": FULL_GUARD_MAX_REGRESSION,
        "progress_passed": progress_passed,
        "guards_passed": guards_passed,
    }


def qualify_full_continuation(
    promotion_records: list[dict],
    development_records: list[dict],
    capability_promotion_records: list[dict],
    capability_development_records: list[dict],
    main_ids: tuple[str, ...],
    core_ids: tuple[str, ...],
) -> dict:
    """Apply a permissive, held-out, new-constraint gate for 120-hour compute."""
    record_sets = (
        promotion_records,
        development_records,
        capability_promotion_records,
        capability_development_records,
    )
    update_sequences = [
        [int(record["checkpoint_update"]) for record in records]
        for records in record_sets
    ]
    if any(sequence != update_sequences[0] for sequence in update_sequences[1:]):
        raise ValueError("full promotion/development checkpoint sequences differ")
    for record in (*promotion_records, *development_records):
        _cell_counts(record, main_ids)
    for record in (
        *capability_promotion_records,
        *capability_development_records,
    ):
        _cell_counts(record, CAPABILITY_IDS)
    for records in record_sets[1:]:
        for main, other in zip(promotion_records, records):
            if _checkpoint_identity(main) != _checkpoint_identity(other):
                raise ValueError("full evaluation panels name different checkpoints")
    v6_ids = tuple(
        condition_id for condition_id in main_ids if condition_id not in core_ids
    )
    if len(v6_ids) != 32:
        raise ValueError("full continuation gate expected exactly 32 V6 conditions")
    candidate_index = len(promotion_records) - 1
    reference_index = max(0, candidate_index - 2)
    promotion_comparison = _full_progress_comparison(
        _full_progress_snapshot(promotion_records[reference_index], v6_ids),
        _full_progress_snapshot(promotion_records[candidate_index], v6_ids),
    )
    development_comparison = _full_progress_comparison(
        _full_progress_snapshot(development_records[reference_index], v6_ids),
        _full_progress_snapshot(development_records[candidate_index], v6_ids),
    )
    integrity_passed = all(
        _integrity_passed(record) for records in record_sets for record in records
    )
    qualified = bool(
        promotion_comparison["progress_passed"]
        and promotion_comparison["guards_passed"]
        and development_comparison["guards_passed"]
        and integrity_passed
    )
    return {
        "reference_update": update_sequences[0][reference_index],
        "candidate_update": update_sequences[0][candidate_index],
        "promotion": promotion_comparison,
        "development": development_comparison,
        "integrity_passed_all_panels": integrity_passed,
        "qualified_for_120h": qualified,
    }


def validate_parent_job_receipt(
    path: Path,
    run_contract_path: Path,
    run_contract: dict[str, str],
    evaluated_updates: list[int],
) -> dict:
    receipt = _read_json(path, dict)
    expected = {
        "schema": "terra_v8_parent_slurm_job_v1",
        "job_id": run_contract.get("slurm_job_id"),
        "checkpoint_updates": evaluated_updates,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"{path}: parent job {field} must be {value!r}")
    if receipt.get("state") not in ("COMPLETED", "TIMEOUT"):
        raise ValueError(f"{path}: parent training job did not complete or time out")
    state = receipt["state"]
    exit_code = str(receipt.get("exit_code"))
    if re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is None:
        raise ValueError(f"{path}: parent job lacks a Slurm exit code")
    if state == "COMPLETED" and exit_code != "0:0":
        raise ValueError(f"{path}: completed parent has a nonzero exit code")
    if receipt.get("partition") != "gpuhe.24h":
        raise ValueError(f"{path}: parent job partition changed")
    if receipt.get("terra_baselines_revision") != run_contract.get(
        "terra_baselines_revision"
    ):
        raise ValueError(f"{path}: parent job revision changed")
    evaluator_job_id = str(receipt.get("evaluator_job_id"))
    if re.fullmatch(r"[0-9]+", evaluator_job_id) is None:
        raise ValueError(f"{path}: parent receipt lacks evaluator job identity")

    expected_run_dir = (
        REMOTE_RUN_ROOT
        / run_contract["terra_baselines_revision"]
        / "screen"
        / "full"
        / f"s{run_contract['seed']}"
        / run_contract["arm"]
    )
    run_dir = receipt.get("run_dir")
    if run_dir != str(expected_run_dir):
        raise ValueError(f"{path}: parent run directory changed")
    run_contract_path = run_contract_path.resolve()
    if run_contract_path != expected_run_dir / "run_contract.env":
        raise ValueError(f"{path}: run contract path changed campaign identity")
    expected_contract = {
        "path": str(run_contract_path),
        "sha256": sha256_file(run_contract_path),
    }
    if receipt.get("run_contract") != expected_contract:
        raise ValueError(f"{path}: parent run-contract identity changed")
    generated = receipt.get("generated_at_utc")
    if not isinstance(generated, str) or not generated:
        raise ValueError(f"{path}: parent job receipt lacks generation time")
    checkpoints = receipt.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(evaluated_updates):
        raise ValueError(f"{path}: parent job lacks its checkpoint identities")
    for update, checkpoint in zip(evaluated_updates, checkpoints):
        if not isinstance(checkpoint, dict) or checkpoint.get("update") != update:
            raise ValueError(f"{path}: parent checkpoint order changed")
        checkpoint_path = Path(str(checkpoint.get("path")))
        if checkpoint_path.parent != expected_run_dir / "checkpoints":
            raise ValueError(f"{path}: parent checkpoint left the exact run directory")
        _validate_remote_checkpoint_path(checkpoint.get("path"), run_contract["arm"])
        _require_sha256(checkpoint.get("sha256"), f"{path}: parent checkpoint hash")
    return receipt


def promote(args: argparse.Namespace) -> dict:
    bank = load_bank_contract(args.bank_root.resolve())
    run_contract_path = args.run_contract.resolve()
    run_contract = parse_run_contract(run_contract_path)
    sampling = stage_sampling_contract(
        bank,
        args.stage,
        run_contract.get("sampler_profile", "bank_v4"),
    )
    prior_receipt = None
    if args.stage == "capability":
        if args.prior_receipt is not None or args.capability is not None:
            raise ValueError("Stage A does not accept prior/capability inputs")
    else:
        if args.prior_receipt is None or args.capability is None:
            raise ValueError(
                f"{args.stage} promotion requires prior and capability inputs"
            )
        prior_receipt = validate_prior_receipt(
            args.prior_receipt.resolve(), args.arm, PRIOR_STAGE[args.stage]
        )
    validate_run_contract(run_contract, args.stage, args.arm, prior_receipt)

    promotion_path = args.promotion.resolve()
    main_group = "capability" if args.stage == "capability" else "main"
    records = validate_evaluation(
        promotion_path,
        args.stage,
        args.arm,
        bank,
        run_contract,
        panel_contract(bank, main_group, "promotion"),
        sampling,
    )
    capability_records = None
    development_records = None
    capability_development_records = None
    continuation_qualification = None
    if args.stage == "capability":
        validate_panel_conditions(records, CAPABILITY_IDS)
        decision = decide_capability(records)
    else:
        validate_panel_conditions(records, bank["main_ids"])
        capability_path = args.capability.resolve()
        capability_records = validate_evaluation(
            capability_path,
            args.stage,
            args.arm,
            bank,
            run_contract,
            panel_contract(bank, "capability", "promotion"),
            sampling,
        )
        validate_panel_conditions(capability_records, CAPABILITY_IDS)
        if args.stage == "nearby":
            decision = decide_nearby(
                records,
                capability_records,
                prior_receipt,
                bank["core_ids"],
                bank["family_by_condition"],
            )
        else:
            decision = decide_full(
                records,
                capability_records,
                prior_receipt,
                bank["main_ids"],
                bank["family_by_condition"],
            )

            if (
                getattr(args, "development", None) is None
                or getattr(args, "capability_development", None) is None
                or getattr(args, "parent_job_receipt", None) is None
            ):
                raise ValueError(
                    "full promotion requires development, capability-development, "
                    "and parent-job-receipt inputs"
                )
            development_records = validate_evaluation(
                args.development.resolve(),
                args.stage,
                args.arm,
                bank,
                run_contract,
                panel_contract(bank, "main", "development"),
                sampling,
                expected_split="development",
            )
            validate_panel_conditions(development_records, bank["main_ids"])
            capability_development_records = validate_evaluation(
                args.capability_development.resolve(),
                args.stage,
                args.arm,
                bank,
                run_contract,
                panel_contract(bank, "capability", "development"),
                sampling,
                expected_split="development",
            )
            validate_panel_conditions(capability_development_records, CAPABILITY_IDS)
            continuation_qualification = qualify_full_continuation(
                records,
                development_records,
                capability_records,
                capability_development_records,
                bank["main_ids"],
                bank["core_ids"],
            )
            decision["passed"] = bool(
                decision["passed"] and continuation_qualification["qualified_for_120h"]
            )
            decision["reason"] = (
                "full-V8 screen retains prior skills and improves V6 constraints"
                if decision["passed"]
                else "full-V8 screen failed retention, integrity, or V6 progress"
            )

    pair = records[-2:]
    records_to_validate = records if args.stage == "full" else pair
    validated_checkpoints = [
        validate_candidate_checkpoint(
            record,
            args.stage,
            args.arm,
            run_contract,
            sampling,
        )
        for record in records_to_validate
    ]
    candidate = validated_checkpoints[-1]
    if continuation_qualification is not None:
        by_update = {
            checkpoint["next_update"]: checkpoint
            for checkpoint in validated_checkpoints
        }
        reference = by_update[continuation_qualification["reference_update"]]
        continuation_qualification.update(
            {
                "reference_checkpoint_sha256": reference["checkpoint_sha256"],
                "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
                "inherited_retention_passed": bool(
                    not decision["rollback_triggered"]
                    and all(
                        not failed
                        for failed in decision["retention_failure_history"][-2:]
                    )
                ),
                "qualified_for_120h": bool(decision["passed"]),
            }
        )
    if args.stage == "capability":
        input_parent_path = candidate["teacher_checkpoint"]
    else:
        input_parent_path = candidate["warm_start_from"]
        if input_parent_path != prior_receipt["candidate"]["path"]:
            raise ValueError("stage transition did not load the prior promoted path")

    evaluated_updates = [record["checkpoint_update"] for record in records]
    parent_job = None
    if args.stage == "full":
        parent_job = validate_parent_job_receipt(
            args.parent_job_receipt.resolve(),
            run_contract_path,
            run_contract,
            evaluated_updates,
        )
        for record, checkpoint in zip(records, parent_job["checkpoints"]):
            if _checkpoint_identity(record) != (
                checkpoint["path"],
                checkpoint["sha256"],
                checkpoint["update"],
            ):
                raise ValueError(
                    "full evaluation differs from the frozen parent checkpoint prefix"
                )

    receipt = {
        "schema": SCHEMA,
        "passed": decision["passed"],
        "reason": decision["reason"],
        "stage": args.stage,
        "next_stage": NEXT_STAGE[args.stage],
        "arm": args.arm,
        "release_id": RELEASE_ID,
        "terra_revision": TERRA_REVISION,
        "terra_baselines_revision": run_contract["terra_baselines_revision"],
        "bank_archive_sha256": BANK_ARCHIVE_SHA256,
        "bank_dataset_sha256": BANK_DATASET_SHA256,
        "training_mixture_sha256": TRAINING_MIXTURE_SHA256,
        "sampling": sampling,
        "scheduled_updates": list(STAGE_UPDATES[args.stage]),
        "evaluated_updates": evaluated_updates,
        "evaluated_pair": [record["checkpoint_update"] for record in pair],
        "evaluated_checkpoints": (
            validated_checkpoints if args.stage == "full" else None
        ),
        "candidate": candidate,
        "input_parent": {
            "path": input_parent_path,
            "checkpoint_sha256": run_contract["parent_checkpoint_sha256"],
            "initial_checkpoint_sha256": run_contract["initial_checkpoint_sha256"],
        },
        "mastery": decision["mastery"],
        "family_totals": decision.get("family_totals"),
        "integrity_pair": decision["integrity_pair"],
        "integrity_history": decision.get("integrity_history"),
        "continuation_qualification": continuation_qualification,
        "parent_slurm_job": parent_job,
        "retention": {
            "inherited_thresholds": decision["inherited_thresholds"],
            "new_thresholds": decision["new_thresholds"],
            "frozen_thresholds": decision["frozen_thresholds"],
            "failures": decision["failures"],
            "failure_history": decision.get(
                "retention_failure_history", [False, False]
            ),
            "rollback_updates": decision.get("rollback_updates"),
            "rollback_triggered": decision["rollback_triggered"],
        },
        "inputs": {
            "promotion": {
                "path": str(promotion_path),
                "sha256": sha256_file(promotion_path),
            },
            "capability": (
                None
                if args.capability is None
                else {
                    "path": str(args.capability.resolve()),
                    "sha256": sha256_file(args.capability.resolve()),
                }
            ),
            "development": (
                None
                if development_records is None
                else {
                    "path": str(args.development.resolve()),
                    "sha256": sha256_file(args.development.resolve()),
                }
            ),
            "capability_development": (
                None
                if capability_development_records is None
                else {
                    "path": str(args.capability_development.resolve()),
                    "sha256": sha256_file(args.capability_development.resolve()),
                }
            ),
            "parent_job_receipt": (
                None
                if parent_job is None
                else {
                    "path": str(args.parent_job_receipt.resolve()),
                    "sha256": sha256_file(args.parent_job_receipt.resolve()),
                }
            ),
            "prior_receipt": (
                None
                if args.prior_receipt is None
                else {
                    "path": str(args.prior_receipt.resolve()),
                    "sha256": sha256_file(args.prior_receipt.resolve()),
                }
            ),
            "run_contract": {
                "path": str(run_contract_path),
                "sha256": sha256_file(run_contract_path),
            },
        },
    }
    return receipt


def inspect_receipt(args: argparse.Namespace) -> dict:
    receipt_path = args.receipt.resolve()
    receipt = validate_prior_receipt(receipt_path, args.arm, args.stage)
    candidate = receipt["candidate"]
    if args.expect_candidate is not None and candidate["path"] != str(
        args.expect_candidate
    ):
        raise ValueError("prior receipt names an unexpected promoted checkpoint")
    if (
        args.expect_candidate_sha256 is not None
        and candidate["checkpoint_sha256"] != args.expect_candidate_sha256
    ):
        raise ValueError("prior receipt names an unexpected promoted checkpoint hash")
    if args.verify_checkpoint:
        checkpoint_path = Path(candidate["path"])
        if sha256_file(checkpoint_path) != candidate["checkpoint_sha256"]:
            raise ValueError("prior receipt candidate checkpoint hash mismatch")
    return {
        "schema": "terra_v8_stage_gate_inspection_v1",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "arm": args.arm,
        "stage": args.stage,
        "next_stage": receipt["next_stage"],
        "candidate_path": candidate["path"],
        "candidate_sha256": candidate["checkpoint_sha256"],
        "candidate_update": candidate["next_update"],
    }


def check_smoke_contract(args: argparse.Namespace) -> dict:
    path = args.run_contract.resolve()
    contract = parse_run_contract(path)
    _require_sha256(args.parent_sha256, "expected smoke parent hash")
    if args.stage == "capability":
        if args.prior_gate_sha256 != "none":
            raise ValueError("capability smoke cannot have a prior gate")
    else:
        _require_sha256(args.prior_gate_sha256, "expected prior gate hash")
    expected = {
        "status": "PASSED",
        "phase": "smoke",
        "arm": args.arm,
        "architecture": ARM_ARCHITECTURES[args.arm]["label"],
        "curriculum_stage": args.stage,
        "condition_count": str(EXPECTED_CONDITIONS[args.stage]),
        "seed": str(args.seed),
        "updates": "1",
        "global_transitions": str(4 * 1024 * 32),
        "terra_revision": TERRA_REVISION,
        "training_bank_release_id": RELEASE_ID,
        "training_bank_archive_sha256": BANK_ARCHIVE_SHA256,
        "training_bank_dataset_sha256": BANK_DATASET_SHA256,
        "parent_checkpoint_sha256": args.parent_sha256,
        "prior_gate_receipt_sha256": args.prior_gate_sha256,
        "reward_stage": "dense_skill",
        "reward_type": "DENSE",
        "trench_shaping": "false",
        "horizon": "450",
        "full_resets": "true",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"smoke run contract {field} must be {value!r}, "
                f"got {contract.get(field)!r}"
            )
    if args.stage == "capability":
        if contract.get("teacher_checkpoint_sha256") != INITIAL_PARENT_SHA256:
            raise ValueError("capability smoke teacher changed")
        if contract.get("initialization") != "params_only_warm_fresh_optimizer":
            raise ValueError("capability smoke initialization changed")
    else:
        if contract.get("teacher_checkpoint_sha256") != "none":
            raise ValueError("later-stage smoke must disable teacher kickstart")
        if contract.get("initialization") != (
            "params_only_stage_transition_fresh_optimizer"
        ):
            raise ValueError("later-stage smoke initialization changed")
    _require_sha256(
        contract.get("initial_checkpoint_sha256"),
        "smoke initial checkpoint hash",
    )
    revision = contract.get("terra_baselines_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("smoke lacks immutable terra-baselines revision")
    return {
        "schema": "terra_v8_smoke_contract_check_v1",
        "passed": True,
        "run_contract": str(path),
        "run_contract_sha256": sha256_file(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("promote")
    command.add_argument("--stage", choices=tuple(STAGE_UPDATES), required=True)
    command.add_argument("--arm", choices=tuple(ARM_ARCHITECTURES), required=True)
    command.add_argument("--bank-root", type=Path, required=True)
    command.add_argument("--promotion", type=Path, required=True)
    command.add_argument("--capability", type=Path)
    command.add_argument("--development", type=Path)
    command.add_argument("--capability-development", type=Path)
    command.add_argument("--parent-job-receipt", type=Path)
    command.add_argument("--prior-receipt", type=Path)
    command.add_argument("--run-contract", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--receipt", type=Path, required=True)
    inspect.add_argument("--stage", choices=tuple(NEXT_STAGE), required=True)
    inspect.add_argument("--arm", choices=tuple(ARM_ARCHITECTURES), required=True)
    inspect.add_argument("--expect-candidate", type=Path)
    inspect.add_argument("--expect-candidate-sha256")
    inspect.add_argument("--verify-checkpoint", action="store_true")
    smoke = subparsers.add_parser("check-smoke")
    smoke.add_argument("--run-contract", type=Path, required=True)
    smoke.add_argument("--stage", choices=tuple(STAGE_UPDATES), required=True)
    smoke.add_argument("--arm", choices=tuple(ARM_ARCHITECTURES), required=True)
    smoke.add_argument("--seed", type=int, required=True)
    smoke.add_argument("--parent-sha256", required=True)
    smoke.add_argument("--prior-gate-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "promote":
        receipt = promote(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(
            f"V8_STAGE_GATE stage={args.stage} arm={args.arm} "
            f"passed={str(receipt['passed']).lower()} output={args.output}"
        )
    elif args.command == "inspect":
        print(json.dumps(inspect_receipt(args), sort_keys=True, allow_nan=False))
    elif args.command == "check-smoke":
        print(json.dumps(check_smoke_contract(args), sort_keys=True, allow_nan=False))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
