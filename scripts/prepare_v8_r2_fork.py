#!/usr/bin/env python3
"""Create the one output-preserving compact-u20 parent used by both R2 arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from terra.config import BatchConfig, MapsDimsConfig

from scripts.grow_checkpoint import _derive_action_type
from utils import helpers
from utils.accepted_bank import load_accepted_bank
from utils.models import get_model_ready
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings

PARENT_SHA256 = "0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834"
PARENT_UPDATE = 20_000
PARENT_OPTIMIZER_STEP = 1_280_000
PARENT_PARAMETER_COUNT = 2_856_685
PREPARED_PARAMETER_COUNT = 2_856_701
SCHEMA = "terra_v8_r2_prepared_fork_v1"
CARRY_KERNEL = (
    "params",
    "agent_state_net",
    "mlp_continuous",
    "layers_0",
    "kernel",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_sha256(tree) -> str:
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def set_config_value(config, name, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def parameter_count(params) -> int:
    return sum(int(np.asarray(leaf).size) for leaf in jax.tree.leaves(params))


def expand_carry_input(source_params, target_params):
    """Copy every parent leaf and append one exactly-zero carry-input row."""
    source = flatten_dict(unfreeze(source_params))
    target = flatten_dict(unfreeze(target_params))
    if set(source) != set(target):
        raise ValueError("carry expansion changed the model parameter paths")

    expanded = {}
    changed = []
    for key in sorted(target):
        old = jnp.asarray(source[key])
        new = jnp.asarray(target[key])
        if old.shape == new.shape:
            expanded[key] = old.astype(new.dtype)
            continue
        if key != CARRY_KERNEL or new.shape != (old.shape[0] + 1, old.shape[1]):
            raise ValueError(
                f"unexpected carry-expansion shape change at {key}: "
                f"{old.shape} -> {new.shape}"
            )
        grown = jnp.zeros_like(new).at[: old.shape[0], :].set(old)
        expanded[key] = grown
        changed.append(
            {"path": "/".join(key), "source": old.shape, "target": new.shape}
        )
    if len(changed) != 1:
        raise ValueError(f"expected one carry-input expansion, observed {changed}")
    return freeze(unflatten_dict(expanded)), changed[0]


def sampler_labels(bank) -> dict[str, dict]:
    return {
        level.condition_id: {
            "family": level.family,
            "branch_depth": level.branch_depth,
            "curriculum_depth": bank.curriculum_depths[index],
        }
        for index, level in enumerate(bank.levels)
    }


def migrate_sampler(source_state: dict, bank) -> tuple[dict, dict]:
    source_settings = dict(source_state.get("settings", {}))
    if source_settings.get("rule") != "continuous_banded_v1":
        raise ValueError("compact-u20 sampler is not continuous_banded_v1")
    target_settings = SamplerSettings(
        **{**source_settings, "rule": "continuous_banded_v2"}
    )
    sampler = PooledConditionSampler(
        [level.condition_id for level in bank.levels],
        target_settings,
        maps_per_condition=[level.map_count for level in bank.levels],
        labels=sampler_labels(bank),
    )
    sampler.restore_state_dict(copy.deepcopy(source_state))
    migrated = sampler.state_dict()
    if migrated["settings"]["rule"] != "continuous_banded_v2":
        raise ValueError("sampler migration did not materialize the v2 rule")
    preserved = (
        "conditions",
        "maps_per_condition",
        "labels",
        "competence",
        "current_window",
        "closed_window",
        "refresh",
        "numpy_rng",
        "mastery",
    )
    if any(migrated[key] != source_state[key] for key in preserved):
        raise ValueError(
            "sampler migration changed history instead of only probabilities"
        )
    return migrated, sampler.receipt()


def parity_observation(env, batch_size: int = 3):
    edge = env.batch_cfg.maps_dims.maps_edge_length
    angles = env.batch_cfg.agent.angles_cabin
    state_width = env.batch_cfg.agent.num_state_obs
    if state_width != 9:
        raise ValueError(
            f"R2 Terra must expose 9 agent-state fields, got {state_width}"
        )
    states = jnp.zeros((batch_size, 4, state_width), dtype=jnp.float32)
    states = states.at[:, 0, 0].set(jnp.arange(batch_size) + 7)
    states = states.at[:, 0, 1].set(jnp.arange(batch_size) + 11)
    states = states.at[:, 0, 2].set(2)
    states = states.at[:, 0, 3].set(4)
    states = states.at[:, 0, 5].set(13)
    states = states.at[:, 0, 8].set(jnp.asarray([0.0, 0.3, 0.9]))
    obs = [
        states,
        jnp.zeros((batch_size, 4), dtype=jnp.int8).at[:, 0].set(1),
        jnp.ones((batch_size,), dtype=jnp.int32),
    ]
    obs += [jnp.zeros((batch_size, angles)) for _ in range(9)]
    obs += [jnp.zeros((batch_size, edge, edge)) for _ in range(4)]
    obs += [jnp.zeros((batch_size,), dtype=jnp.int32) for _ in range(2)]
    obs += [jnp.zeros((batch_size, edge, edge)) for _ in range(3)]
    obs += [jnp.zeros((batch_size, 5), dtype=jnp.int32)]
    return obs


def validate_parent(checkpoint: dict) -> None:
    config = checkpoint.get("train_config")
    expected = {
        "next_update": PARENT_UPDATE,
        "train_state_step": PARENT_OPTIMIZER_STEP,
        "map_encoder": "resnet_spatial_8x8_se_xattn",
        "model_size": "medium",
        "model_core": "mlp",
        "encoder_compute_dtype": "bfloat16",
        "attention_compute_dtype": "float32",
        "resnet_blocks_per_stage": (2, 2, 3, 3),
        "resnet_stage_channels": (24, 48, 64, 96),
        "critic_hidden_dims": (512, 256),
        "num_prev_actions": 5,
        "ent_schedule_end": 0.02,
    }
    observed_step = int(np.asarray(checkpoint.get("train_state_step")).reshape(()))
    observed = {
        "next_update": checkpoint.get("next_update"),
        "train_state_step": observed_step,
        **{
            key: config_value(config, key)
            for key in expected
            if key not in {"next_update", "train_state_step"}
        },
    }
    if observed != expected:
        raise ValueError(f"compact-u20 parent contract changed: {observed!r}")
    if parameter_count(checkpoint["model"]) != PARENT_PARAMETER_COUNT:
        raise ValueError("compact-u20 parent parameter count changed")
    state = checkpoint.get("pooled_sampler_state")
    if not isinstance(state, dict) or len(state.get("conditions", ())) != 47:
        raise ValueError("compact-u20 parent lacks the 47-condition sampler state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--protocol-terra-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("prepared fork output and receipt must be new paths")
    if sha256_file(args.parent) != PARENT_SHA256:
        raise ValueError("selected parent bytes do not match compact-u20")

    helpers.register_checkpoint_config_classes()
    checkpoint = helpers.load_pkl_object(str(args.parent))
    validate_parent(checkpoint)

    bank = load_accepted_bank(
        args.bank_root,
        "G-UNIFORM",
        args.protocol_terra_revision,
        curriculum_stage="full",
        sampler_profile="continuous_banded_v2",
    )
    migrated_state, migrated_receipt = migrate_sampler(
        checkpoint["pooled_sampler_state"], bank
    )

    source_config = copy.deepcopy(checkpoint["train_config"])
    target_config = copy.deepcopy(checkpoint["train_config"])
    set_config_value(source_config, "carry_work_observation", False)
    set_config_value(target_config, "carry_work_observation", True)
    set_config_value(target_config, "config_name", "G-V8-CONTINUOUS-V2")
    target_sampler_config = copy.deepcopy(config_value(target_config, "pooled_sampler"))
    if not isinstance(target_sampler_config, dict):
        raise ValueError("compact-u20 parent lacks its pooled-sampler config")
    target_sampler_config["rule"] = "continuous_banded_v2"
    set_config_value(target_config, "pooled_sampler", target_sampler_config)
    set_config_value(target_config, "accepted_bank", bank)
    set_config_value(target_config, "prepared_fork_from", None)
    env = type("ModelEnv", (), {})()
    env.batch_cfg = BatchConfig(
        action_type=_derive_action_type(checkpoint),
        maps_dims=MapsDimsConfig(maps_edge_length=64),
    )
    source_model, _ = get_model_ready(jax.random.PRNGKey(1), source_config, env)
    target_model, target_init = get_model_ready(
        jax.random.PRNGKey(2), target_config, env
    )
    expanded, expansion = expand_carry_input(checkpoint["model"], target_init)
    if parameter_count(expanded) != PREPARED_PARAMETER_COUNT:
        raise ValueError("prepared fork parameter count changed")

    obs = parity_observation(env)
    source_value, source_logits = source_model.apply(checkpoint["model"], obs)
    target_value, target_logits = target_model.apply(expanded, obs)
    value_delta = float(np.max(np.abs(np.asarray(source_value - target_value))))
    logits_delta = float(np.max(np.abs(np.asarray(source_logits - target_logits))))
    if value_delta != 0.0 or logits_delta != 0.0:
        raise ValueError(
            f"carry expansion changed parent outputs: value={value_delta}, logits={logits_delta}"
        )

    sampler_state_sha = canonical_sha256(migrated_state)
    prepared = dict(checkpoint)
    prepared["model"] = expanded
    prepared["train_config"] = target_config
    prepared["pooled_sampler_state"] = migrated_state
    prepared.pop("optimizer_state", None)
    prepared.pop("train_state_step", None)
    prepared["r2_prepared_fork"] = {
        "schema": SCHEMA,
        "source_checkpoint_sha256": PARENT_SHA256,
        "source_next_update": PARENT_UPDATE,
        "source_optimizer_step": PARENT_OPTIMIZER_STEP,
        "source_parameter_count": PARENT_PARAMETER_COUNT,
        "prepared_parameter_count": PREPARED_PARAMETER_COUNT,
        "carry_observation_index": 8,
        "carry_input_expansion": expansion,
        "output_preserving": True,
        "source_sampler_rule": "continuous_banded_v1",
        "target_sampler_rule": "continuous_banded_v2",
        "target_config_name": "G-V8-CONTINUOUS-V2",
        "target_bank_sampler_profile": "continuous_banded_v2",
        "migrated_sampler_state_sha256": sampler_state_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    helpers.save_pkl_object(prepared, str(args.output))
    receipt = {
        **prepared["r2_prepared_fork"],
        "schema": SCHEMA,
        "passed": True,
        "prepared_checkpoint": str(args.output),
        "prepared_checkpoint_sha256": sha256_file(args.output),
        "prepared_model_sha256": tree_sha256(expanded),
        "migrated_sampler_receipt": migrated_receipt,
        "parity": {"max_abs_value": value_delta, "max_abs_logits": logits_delta},
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "passed",
                    "prepared_checkpoint_sha256",
                    "prepared_model_sha256",
                    "migrated_sampler_state_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
