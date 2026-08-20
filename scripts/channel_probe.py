#!/usr/bin/env python3
"""Channel-sensitivity probe (CURRICULUM_SPEC_V6.md section 5, as corrected by
REVIEW_V6 section 4).

How much does the policy actually READ a given piece of observation
INFORMATION? For a fixed batch of held-out states, re-evaluate the network with
that information removed and measure how far the policy and the value head move:

* ``policy_kl``  -- mean KL( intact || ablated ) over the action distribution
* ``value_mae``  -- mean | V_intact - V_ablated |

What the first version of this script got wrong, and what changed
--------------------------------------------------------------------------

**F4-2 / the obstacle group is now the INFORMATION, not the plane.** The
``occupancy`` group only touches ``padding_mask`` and ``local_map_obstacles``,
but 100% of obstacle cells are ALSO marked non-dumpable in ``dumpability_mask``
and flagged in ``traversability_mask``. Wiping one plane while two untouched
planes still carry the same cells removes no information, so a near-zero result
says "redundant", not "ignored". ``obstacle_info`` ablates all four planes
together: occupancy zeroed/shuffled, obstacle cells restored to dumpable and to
traversable. On M1-A the two differ by ~6x. ``occupancy`` is kept, labelled as a
single-plane redundancy measurement, and is NOT the obstacle-sensitivity metric.

**F4-1 / dilution is reported, not averaged away.** An ablation is a no-op on
any state whose channel content it does not change (an occupancy shuffle between
two obstacle-free states changes nothing), and the M1-A batch is 87.7% such
states. Every ablation now reports ``n_effective`` -- states the ablation
actually perturbed -- with the conditional mean/median/max over exactly those
states, plus ``cells_changed`` and ``kl_per_100_cells`` so KL magnitude can be
read against perturbation size (F4-9). Quote the conditional number for a
sparse channel; the batch mean is only meaningful when
``effective_fraction >= 0.5``, which the record flags per ablation.

**F4-8 / the dump-target shuffle no longer erases the dig target.** The old
implementation added the donor's positive half to ``target_map`` with the dig
half still present, and int8 ``-1 + 1`` cancelled to 0 wherever the donor's dump
cell landed on this state's dig cell -- 31% of dig cells per state. The shuffle
is now written as ``where(donor > 0, 1, where(target < 0, -1, 0))``.

**F4-11 / donors can be drawn from a plausible stratum.** ``--donor plausible``
draws the donor from the SAME condition and the SAME episode step, so the
grafted plane is geometrically compatible with the agent pose; ``--donor
sequential`` reproduces the original state ``i -> i+1`` derangement. The
``obstacle_info`` and ``occupancy`` donors are ALWAYS forced inside the
obstacle-bearing stratum, since a donor with no obstacle is a no-op.

**F4-5 / provenance.** The record carries the sha256 of this script, of the
``terra`` checkout actually imported (the venv's editable ``terra`` points at
the un-fixed main repo -- REVIEW_V6 R-8), the checkpoint, and the probe batch.

Design points that carry over:

* **The probe batch is policy-independent.** States are collected by stepping
  the env with a fixed seeded RANDOM action sequence, not with the checkpoint
  under test, so the same states are scored for every checkpoint and every arm.
  Cached to ``--probe-cache``, keyed by (manifest sha256, seed, snapshot
  schedule, size); a cache built for a different key is a hard error, because a
  probe batch is only comparable to itself (F4-4).
* **Reference ablations are mandatory.** ``action_map`` and ``target_map`` are
  ablated too. If those also read ~0 the probe is broken, not the policy. Note
  ``action_map`` is itself degenerate on an early checkpoint (all-zero on more
  than half the batch), so ``target_map`` carries the soundness criterion.
* Ablations act on the observation dict before ``obs_to_model_input``, so a
  global map plane and its per-cabin-angle local summary move together.
* Zero vs shuffle is a PER-CHANNEL question (F4-7), not a global rule: zeroing
  the mostly-True ``dumpability_mask`` is a 19x out-of-distribution artifact,
  while for ``dump_target`` zeroing reads LOWER than shuffling. Both modes are
  run and both are reported; say which one a quoted number came from.

Standalone usage:

    PYTHONPATH=$TERRA:$PWD python scripts/channel_probe.py \\
        --checkpoint checkpoints_v5m/A/..._FINAL.pkl \\
        --bank-root terra_data/curriculum_v5m --split held_out --strata all \\
        --donor plausible --output eval/channel_probe_m1a.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eval_fixed_bank import (
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    sha256_file,
)
from train import TrainConfig
from train_mixed import (
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.helpers import load_pkl_object
from utils.models import validate_model_params_match
from utils.utils_ppo import obs_to_model_input, policy, wrap_action

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

SCHEMA = "terra_channel_probe_v2"

# name -> observation keys the ablation writes. Two groups are computed rather
# than copied (see `ablate`): `dump_target` rebuilds `target_map`'s positive
# half dig-preservingly, and `obstacle_info` repairs `dumpability_mask` /
# `traversability_mask` on the obstacle cells instead of overwriting the planes.
ABLATIONS: dict[str, tuple[str, ...]] = {
    # the obstacle-sensitivity metric: every plane that carries an obstacle cell
    "obstacle_info": (
        "padding_mask",
        "local_map_obstacles",
        "dumpability_mask",
        "traversability_mask",
    ),
    # the single plane, kept only to measure how redundant it is
    "occupancy": ("padding_mask", "local_map_obstacles"),
    "dumpability": ("dumpability_mask", "local_map_dumpability"),
    "dump_target": ("target_map", "local_map_target_pos"),
    # reference ablations: these MUST move the policy
    "action_map": ("action_map", "local_map_action_neg", "local_map_action_pos"),
    "target_map": ("target_map", "local_map_target_neg", "local_map_target_pos"),
}
REFERENCE_ABLATIONS = ("action_map", "target_map")
# groups whose donor must itself carry an obstacle, or the shuffle is a no-op
OBSTACLE_ABLATIONS = ("obstacle_info", "occupancy")


def ablate(obs: dict, name: str, mode: str, donor: dict | None = None) -> dict:
    """Remove one group of information from `obs`.

    `mode="zero"` switches the channel off; `mode="shuffle"` replaces it with
    another state's copy, preserving the marginal distribution and destroying
    only the correlation with THIS state. Shuffle is the fair test of "is the
    content read"; zeroing additionally asserts something globally false, which
    for a mostly-True mask is its own out-of-distribution shock.
    """
    out = dict(obs)

    if name == "dump_target":
        target = out["target_map"]
        if mode == "shuffle" and donor is not None:
            # F4-8: dig cells stay dig cells. The old `where(t>0,0,t) +
            # where(o>0,o,0)` cancelled -1 + 1 to 0 on every collision.
            out["target_map"] = jnp.where(
                donor["target_map"] > 0, 1, jnp.where(target < 0, -1, 0)
            ).astype(target.dtype)
            out["local_map_target_pos"] = donor["local_map_target_pos"]
        else:
            out["target_map"] = jnp.where(target > 0, jnp.zeros_like(target), target)
            out["local_map_target_pos"] = jnp.zeros_like(out["local_map_target_pos"])
        return out

    if name == "obstacle_info":
        obstacle = out["padding_mask"] != 0
        if mode == "shuffle" and donor is not None:
            out["padding_mask"] = donor["padding_mask"]
            out["local_map_obstacles"] = donor["local_map_obstacles"]
        else:
            out["padding_mask"] = jnp.zeros_like(out["padding_mask"])
            out["local_map_obstacles"] = jnp.zeros_like(out["local_map_obstacles"])
        # the redundant copies: obstacle cells become dumpable and traversable
        dumpability = out["dumpability_mask"]
        out["dumpability_mask"] = jnp.where(
            obstacle, jnp.ones_like(dumpability), dumpability
        )
        traversability = out["traversability_mask"]
        out["traversability_mask"] = jnp.where(
            obstacle, jnp.zeros_like(traversability), traversability
        )
        return out

    for key in ABLATIONS[name]:
        if mode == "shuffle" and donor is not None:
            out[key] = donor[key]
        else:
            out[key] = jnp.zeros_like(out[key])
    return out


def checkpoint_paths(args) -> list[Path]:
    paths = [Path(p) for p in args.checkpoint]
    for pattern in args.checkpoint_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit("no checkpoints given (--checkpoint / --checkpoint-glob)")
    seen, unique = set(), []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def manifest_sha256(directory: Path) -> str:
    return sha256_file(directory / "manifest.jsonl")


def terra_provenance() -> dict:
    """Which `terra` did we actually import? (REVIEW_V6 R-8.)"""
    import terra

    root = Path(terra.__file__).resolve().parent
    record = {"terra_package": str(root), "terra_state_sha256": ""}
    state = root / "state.py"
    if state.exists():
        record["terra_state_sha256"] = sha256_file(state)
    try:
        record["terra_git_rev"] = subprocess.run(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        record["terra_git_rev"] = ""
    return record


def collect_probe_batch(
    env,
    env_params,
    config,
    reset_keys,
    rows: list[dict],
    *,
    seed: int,
    snapshots: tuple[int, ...],
    size: int,
) -> dict[str, np.ndarray]:
    """Fixed states, gathered under a seeded RANDOM policy (checkpoint-free)."""
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    timestep = env.reset(env_params, reset_keys)
    count = len(rows)
    prev_actions = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)
    num_actions = int(env.batch_cfg.action_type.get_num_actions())

    frames: list[dict] = []
    slot_of_frame: list[np.ndarray] = []
    step_of_frame: list[np.ndarray] = []
    for step in range(max(snapshots) + 1):
        if step in snapshots:
            frames.append({k: np.asarray(v) for k, v in timestep.observation.items()})
            frames[-1]["__prev_actions__"] = np.asarray(prev_actions)
            slot_of_frame.append(np.arange(count, dtype=np.int32))
            step_of_frame.append(np.full(count, step, dtype=np.int32))
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        actions = jax.random.randint(
            action_rng, (count,), 0, num_actions, dtype=jnp.int32
        )
        prev_actions = jnp.roll(prev_actions, shift=1, axis=1).at[:, 0].set(actions)
        step_keys = jax.random.split(step_rng, count)
        timestep = env.step(
            timestep, wrap_action(actions, env.batch_cfg.action_type), step_keys
        )

    stacked = {
        key: np.concatenate([frame[key] for frame in frames], axis=0)
        for key in frames[0]
    }
    slots = np.concatenate(slot_of_frame)
    steps = np.concatenate(step_of_frame)
    total = slots.shape[0]
    # deterministic, condition-spreading subsample: a stride keeps every
    # (slot, step) family represented instead of truncating the last snapshots
    if size < total:
        keep = np.linspace(0, total - 1, num=size).round().astype(np.int64)
        keep = np.unique(keep)
        stacked = {key: value[keep] for key, value in stacked.items()}
        slots, steps = slots[keep], steps[keep]
    stacked["__slot__"] = slots
    stacked["__step__"] = steps
    stacked["__condition__"] = np.array(
        [rows[int(slot)]["condition_id"] for slot in slots], dtype=object
    )
    return stacked


def donor_indices(batch: dict[str, np.ndarray], scheme: str) -> dict[str, np.ndarray]:
    """One donor permutation per ablation group.

    * `sequential`: state i is scored against state i+1 (the original scheme).
    * `plausible`: donor drawn from the same (condition, step) stratum, so the
      grafted global plane and the grafted egocentric summary describe a
      compatible agent pose (F4-11). A state with no partner keeps itself, which
      registers as a no-op and is excluded from `n_effective`.

    The obstacle groups always draw from the obstacle-bearing stratum: a donor
    with no obstacle turns the shuffle into a no-op (F4-1/F4-3).
    """
    total = batch["__slot__"].shape[0]
    index = np.arange(total)
    if scheme == "plausible":
        conditions = np.array([str(c) for c in batch["__condition__"]])
        steps = np.asarray(batch["__step__"])
        default = index.copy()
        for condition in np.unique(conditions):
            for step in np.unique(steps):
                group = np.where((conditions == condition) & (steps == step))[0]
                if group.size > 1:
                    default[group] = np.roll(group, 1)
    else:
        default = (index + 1) % total

    obstacle = np.where(
        (np.asarray(batch["padding_mask"]).reshape(total, -1) != 0).any(axis=1)
    )[0]
    if obstacle.size > 1:
        obstacle_donor = obstacle[(np.searchsorted(obstacle, index) + 1) % obstacle.size]
    else:
        obstacle_donor = default

    perms = {name: default for name in ABLATIONS}
    for name in OBSTACLE_ABLATIONS:
        perms[name] = obstacle_donor
    return perms


def obstacle_mask(batch: dict[str, np.ndarray]) -> np.ndarray:
    total = batch["__slot__"].shape[0]
    return (np.asarray(batch["padding_mask"]).reshape(total, -1) != 0).any(axis=1)


def ablation_strata(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The states on which an ablation is even askable.

    For the obstacle groups that is the obstacle-BEARING states: on an
    obstacle-free state there is no obstacle information to remove, so including
    it only dilutes (F4-1 -- 449 of M1-A's 512 states). Every other group is
    askable everywhere, and dilution is then detected by `cells_changed == 0`.
    """
    total = batch["__slot__"].shape[0]
    everywhere = np.ones(total, dtype=bool)
    bearing = obstacle_mask(batch)
    return {
        name: (bearing if name in OBSTACLE_ABLATIONS else everywhere)
        for name in ABLATIONS
    }


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _summary(kl, mae, flip, changed, stratum, *, is_reference: bool) -> dict:
    """Batch mean plus the conditional-on-informative statistics (F4-1/F4-9).

    `stratum` selects the states the ablation is askable on (all of them for
    most groups, the obstacle-bearing ones for the obstacle groups); a state is
    counted as EFFECTIVE only if it is in the stratum and the ablation actually
    changed at least one cell of the observation. Quote
    `policy_kl_conditional_mean` for a sparse channel and `policy_kl_mean` only
    when `batch_mean_publishable`.
    """
    total = int(kl.shape[0])
    active = np.asarray(stratum, dtype=bool) & (changed > 0)
    n_effective = int(active.sum())
    record = {
        "policy_kl_mean": float(kl.mean()),
        "policy_kl_p95": float(np.quantile(kl, 0.95)),
        "value_mae_mean": float(mae.mean()),
        "value_mae_p95": float(np.quantile(mae, 0.95)),
        "argmax_flip_rate": float(flip.mean()),
        "states": total,
        "n_stratum": int(np.asarray(stratum, dtype=bool).sum()),
        "n_effective": n_effective,
        "effective_fraction": float(n_effective / total) if total else 0.0,
        "batch_mean_publishable": bool(total and n_effective / total >= 0.5),
        "is_reference": is_reference,
    }
    if n_effective:
        record.update(
            {
                "policy_kl_conditional_mean": float(kl[active].mean()),
                "policy_kl_conditional_median": float(np.median(kl[active])),
                "policy_kl_conditional_max": float(kl[active].max()),
                "value_mae_conditional_mean": float(mae[active].mean()),
                "argmax_flip_rate_conditional": float(flip[active].mean()),
                "cells_changed_mean": float(changed[active].mean()),
                "kl_per_100_cells": float(
                    100.0 * kl[active].sum() / changed[active].sum()
                ),
                "max_kl_on_noop_states": (
                    float(kl[~active].max()) if n_effective < total else 0.0
                ),
            }
        )
    else:
        record.update(
            {
                "policy_kl_conditional_mean": float("nan"),
                "policy_kl_conditional_median": float("nan"),
                "policy_kl_conditional_max": float("nan"),
                "value_mae_conditional_mean": float("nan"),
                "argmax_flip_rate_conditional": float("nan"),
                "cells_changed_mean": 0.0,
                "kl_per_100_cells": float("nan"),
                "max_kl_on_noop_states": float(kl.max()),
            }
        )
    return record


def probe_checkpoint(
    apply_fn,
    params,
    config,
    batch: dict[str, np.ndarray],
    chunk: int,
    mode: str,
    perms: dict[str, np.ndarray],
    strata: dict[str, np.ndarray],
) -> dict:
    obs_keys = [key for key in batch if not key.startswith("__")]
    total = batch["__slot__"].shape[0]

    def forward(obs: dict, prev_chunk):
        value, pi = policy(
            apply_fn, params, obs_to_model_input(obs, prev_chunk, config)
        )
        return np.asarray(value[:, 0]), np.asarray(pi.logits)

    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        rows_in_chunk = stop - start
        prev_chunk = jnp.asarray(batch["__prev_actions__"][start:stop])
        obs_chunk = {key: jnp.asarray(batch[key][start:stop]) for key in obs_keys}

        value, logit = forward(obs_chunk, prev_chunk)
        bucket = collected.setdefault("intact", {})
        bucket.setdefault("value", []).append(value)
        bucket.setdefault("logits", []).append(logit)

        for name in ABLATIONS:
            donor_chunk = None
            if mode == "shuffle":
                rows = perms[name][start:stop]
                donor_chunk = {key: jnp.asarray(batch[key][rows]) for key in obs_keys}
            ablated = ablate(obs_chunk, name, mode, donor_chunk)
            # how much of the observation did this actually move?
            changed = np.zeros(rows_in_chunk, dtype=np.int64)
            for key in ABLATIONS[name]:
                before = np.asarray(obs_chunk[key])
                after = np.asarray(ablated[key])
                changed += (before != after).reshape(rows_in_chunk, -1).sum(axis=1)
            value, logit = forward(ablated, prev_chunk)
            bucket = collected.setdefault(name, {})
            bucket.setdefault("value", []).append(value)
            bucket.setdefault("logits", []).append(logit)
            bucket.setdefault("changed", []).append(changed)

    stacked = {
        name: {field: np.concatenate(values) for field, values in fields.items()}
        for name, fields in collected.items()
    }

    base_logp = _log_softmax(stacked["intact"]["logits"].astype(np.float64))
    base_p = np.exp(base_logp)
    base_value = stacked["intact"]["value"]
    base_argmax = stacked["intact"]["logits"].argmax(axis=-1)

    out = {
        "mode": mode,
        "states": int(total),
        "value_intact_mean": float(base_value.mean()),
        "value_intact_std": float(base_value.std()),
        "policy_entropy_intact": float((-(base_p * base_logp).sum(axis=-1)).mean()),
        "ablations": {},
    }
    for name in ABLATIONS:
        logp = _log_softmax(stacked[name]["logits"].astype(np.float64))
        out["ablations"][name] = _summary(
            (base_p * (base_logp - logp)).sum(axis=-1),
            np.abs(base_value - stacked[name]["value"]),
            base_argmax != stacked[name]["logits"].argmax(axis=-1),
            stacked[name]["changed"],
            strata[name],
            is_reference=name in REFERENCE_ABLATIONS,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--split", default="held_out")
    parser.add_argument("--strata", nargs="+", default=("all",))
    parser.add_argument("--probe-size", type=int, default=512)
    parser.add_argument(
        "--probe-steps",
        default="0,20,40,60,80",
        help="episode steps at which states are snapshotted",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("zero", "shuffle"),
        choices=("zero", "shuffle"),
    )
    parser.add_argument(
        "--donor",
        default="plausible",
        choices=("plausible", "sequential"),
        help=(
            "shuffle donor stratum: 'plausible' = same condition and same "
            "episode step (F4-11); 'sequential' = state i+1, the original "
            "scheme. Obstacle groups always draw inside the obstacle stratum."
        ),
    )
    parser.add_argument("--probe-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank_root = args.bank_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = tuple(int(v) for v in args.probe_steps.split(","))
    provenance = terra_provenance()
    provenance["script_sha256"] = sha256_file(Path(__file__).resolve())
    print(f"[channel_probe] terra = {provenance['terra_package']}")

    paths = checkpoint_paths(args)
    checkpoints = [(path, load_pkl_object(str(path))) for path in paths]
    checkpoints.sort(
        key=lambda item: (int(item[1].get("next_update", 0)), str(item[0]))
    )
    reference_train_config = checkpoints[0][1]["train_config"]
    for _, checkpoint in checkpoints:
        if "model" not in checkpoint:
            raise KeyError("checkpoint has no model parameters")
        _validate_checkpoint_architecture(checkpoint, reference_train_config)

    records = []
    for stratum in args.strata:
        relative_path = f"{args.split}/{stratum}"
        directory = bank_root / relative_path
        rows = load_manifest(directory)
        count = len(rows)
        os.environ["DATASET_PATH"] = str(bank_root)
        os.environ["DATASET_SIZE"] = str(count)
        config = configure_for_bank(reference_train_config, relative_path, count)
        # The probe config is a rewrite of the checkpoint's own train_config;
        # re-check the architecture against what actually builds the model.
        for _, checkpoint in checkpoints:
            _validate_checkpoint_architecture(checkpoint, config)
        _, env, env_params, initialized_state = make_mixed_agent_states(config)
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        for path, checkpoint in checkpoints:
            validate_model_params_match(
                initialized_state.params, checkpoint["model"], str(path)
            )
        reset_keys = exact_reset_keys(count)

        cache_key = hashlib.sha256(
            "|".join(
                [
                    manifest_sha256(directory),
                    str(args.seed),
                    args.probe_steps,
                    str(args.probe_size),
                ]
            ).encode()
        ).hexdigest()
        batch = None
        cache_path = args.probe_cache
        if cache_path is not None:
            cache_path = cache_path.resolve()
            if cache_path.exists():
                loaded = np.load(cache_path, allow_pickle=True)
                if str(loaded["__cache_key__"]) == cache_key:
                    batch = {
                        key: loaded[key]
                        for key in loaded.files
                        if key != "__cache_key__"
                    }
                    print(f"[channel_probe] reusing probe batch {cache_path}")
                else:
                    raise SystemExit(
                        f"{cache_path} was built for a different batch "
                        f"({loaded['__cache_key__']} != {cache_key}); a probe "
                        "batch is only comparable to itself (F4-4)"
                    )
        if batch is None:
            batch = collect_probe_batch(
                env,
                env_params,
                config,
                reset_keys,
                rows,
                seed=args.seed,
                snapshots=snapshots,
                size=args.probe_size,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path, __cache_key__=np.array(cache_key), **batch
                )
                print(f"[channel_probe] wrote probe batch {cache_path}")

        conditions = sorted({str(c) for c in batch["__condition__"]})
        total_states = int(batch["__slot__"].shape[0])
        obstacle_states = int(obstacle_mask(batch).sum())
        print(
            f"[channel_probe] {relative_path}: {total_states} states over "
            f"{len(conditions)} conditions, snapshots {snapshots}, "
            f"{obstacle_states} obstacle-bearing "
            f"({obstacle_states / total_states:.1%})"
        )
        model_apply = initialized_state.apply_fn
        perms = donor_indices(batch, args.donor)
        strata = ablation_strata(batch)

        for checkpoint_path, checkpoint in checkpoints:
            for mode in args.modes:
                result = probe_checkpoint(
                    model_apply,
                    checkpoint["model"],
                    config,
                    batch,
                    args.chunk,
                    mode,
                    perms,
                    strata,
                )
                record = {
                    "schema": SCHEMA,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_update": int(checkpoint.get("next_update", -1)),
                    "bank_root": str(bank_root),
                    "split_stratum": relative_path,
                    "manifest_sha256": manifest_sha256(directory),
                    "probe_seed": args.seed,
                    "probe_steps": list(snapshots),
                    "probe_cache_key": cache_key,
                    "probe_cache_path": str(cache_path) if cache_path else "",
                    "donor_scheme": args.donor,
                    "obstacle_states": obstacle_states,
                    "conditions": conditions,
                    **provenance,
                    **result,
                }
                records.append(record)
                headline = result["ablations"]
                print(f"  [{mode}] {checkpoint_path.name}:")
                for name in ABLATIONS:
                    row = headline[name]
                    print(
                        f"    {name:14s} KL={row['policy_kl_mean']:.5f} "
                        f"cond={row['policy_kl_conditional_mean']:.5f} "
                        f"n_eff={row['n_effective']:4d}/{row['n_stratum']:4d}"
                        f"(of {row['states']}) "
                        f"cells={row['cells_changed_mean']:8.1f} "
                        f"KL/100c={row['kl_per_100_cells']:.4f} "
                        f"vMAE={row['value_mae_mean']:.4f} "
                        f"flip={row['argmax_flip_rate']:.4f}"
                    )

    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"[channel_probe] wrote {output}")


if __name__ == "__main__":
    main()
