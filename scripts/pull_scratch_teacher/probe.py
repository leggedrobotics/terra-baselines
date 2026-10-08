"""Probe a frozen legacy GRU on declared training maps under new bulk rules.

This reports teacher-state DO compatibility and progress. It neither trains nor
filters the teacher candidate pool using returns. The eight-map subset retains
the source arrays and the native exact-dataset validation contract.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def prepare_subset(source_bank, recipe, selection, output):
    rows = [json.loads(line) for line in (source_bank / "manifest.jsonl").read_text().splitlines()]
    by_slot = {int(row["slot_index"]) - 1: row for row in rows}
    candidates = sorted(slot for slot, row in by_slot.items()
                        if row["family"] == "foundation" and row["split"] == "train")
    if selection["teacher_slots"] != candidates:
        raise ValueError("teacher pool must contain exactly all training foundation slots")
    if not set(selection["precision_slots"]).issubset(candidates):
        raise ValueError("precision pool is outside the training foundation pool")
    cases = recipe["probe"]["maps"]
    if len({case["source_id"] for case in cases}) != len(cases):
        raise ValueError("probe maps must have distinct excavation source IDs")
    for case in cases:
        row = by_slot[case["slot"]]
        for key in ("source_id", "scenario_id", "map_id"):
            if row[key] != case[key]:
                raise ValueError(f"source identity mismatch for slot {case['slot']}: {key}")
        if row["primary_cell"] != case["condition"] or case["slot"] not in candidates:
            raise ValueError("probe selection is not the declared training foundation case")

    output.mkdir(parents=True, exist_ok=False)
    subset = output / "native_bank"
    subset.mkdir()
    sidecars = [(name, "img_", ".npy") for name in
                ("images", "occupancy", "dumpability", "actions", "distance")]
    sidecars.append(("metadata", "trench_", ".json"))
    for folder, _, _ in sidecars:
        (subset / folder).mkdir()
    local_rows = []
    for index, case in enumerate(cases, start=1):
        source_index = case["slot"] + 1
        row = dict(by_slot[case["slot"]])
        row.update(slot_index=index, identity_slot_multiplicity=1,
                   source_manifest_slot_zero_based=case["slot"])
        local_rows.append(row)
        for folder, prefix, suffix in sidecars:
            source = (source_bank / folder / f"{prefix}{source_index}{suffix}").resolve(strict=True)
            (subset / folder / f"{prefix}{index}{suffix}").symlink_to(source)
    (subset / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in local_rows))
    metadata = json.loads((source_bank / "dataset.json").read_text())
    registry = (source_bank / metadata["source_registry"]).resolve(strict=True)
    (subset / "source_registry.jsonl").symlink_to(registry)
    metadata.update(slot_count=len(cases), unique_identity_count=len(cases),
                    source_registry="source_registry.jsonl")
    metadata.pop("train_v3_exposure_balance", None)
    (subset / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    mapping = [{"subset_slot": index, "source_slot": case["slot"],
                "source_id": case["source_id"], "scenario_id": case["scenario_id"],
                "condition": case["condition"], "reset_seeds": case["reset_seeds"]}
               for index, case in enumerate(cases)]
    (output / "source_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return subset


def run_probe(args, recipe, selection, subset):
    # Bind the tiny exact bank before importing Terra/training modules.
    os.environ["DATASET_PATH"] = str(subset.parent)
    os.environ["DATASET_SIZE"] = str(len(recipe["probe"]["maps"]))
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(HERE.parents[1]))

    import jax
    import jax.numpy as jnp

    from eval_fixed_bank import configure_for_bank, exact_reset_keys, prepare_manifest_episode_reset
    from train_mixed import make_mixed_agent_states
    from terra.actions import TrackedActionType
    from utils.helpers import checkpoint_evaluation_config, load_pkl_object, register_checkpoint_config_classes
    from utils.models import validate_model_params_match
    from utils.task_teachers import (
        CACHED_LOGITS_KEY, CACHED_VALUE_KEY, ELIGIBLE_KEY,
        bulk_teacher_observation_and_compatibility,
        recurrent_teacher_rollout_observation,
        validate_bulk_teacher_environment,
    )
    from utils.utils_ppo import initial_actor_hidden, wrap_action

    started = time.monotonic()
    register_checkpoint_config_classes()
    checkpoint = load_pkl_object(args.checkpoint)
    teacher_cfg = checkpoint_evaluation_config(checkpoint)
    if teacher_cfg.actor_core != "gru" or getattr(teacher_cfg, "action_logit_masking", False):
        raise ValueError("probe requires the unmasked recurrent teacher")
    cases = recipe["probe"]["maps"]
    starts = recipe["probe"]["starts_per_map"]
    count = len(cases) * starts
    if any(len(case["reset_seeds"]) != starts for case in cases):
        raise ValueError("each declared map must have the same number of fixed starts")
    cfg = configure_for_bank(teacher_cfg, subset.name, count, precision_mode="bulk")
    cfg.pull_direction_alignment = True
    cfg.dig_pull_min_length_m = 2.5
    cfg.enforce_foundation_border_alignment = False
    cfg.precision_required_band_observation = False
    cfg.precision_episode_fraction = 0.0
    cfg.pull_direction_training_slots = None
    cfg.recurrent_teacher = False
    cfg.teacher_bulk_compatibility = False
    _, env, env_params, model_state = make_mixed_agent_states(cfg)
    env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
    validate_model_params_match(model_state.params, checkpoint["model"], "GRU110000 probe teacher")
    if cfg.num_prev_actions != teacher_cfg.num_prev_actions:
        raise ValueError("teacher action-history width changed")

    map_indices = np.repeat(np.arange(len(cases)), starts)
    source_slots = np.repeat(np.asarray([case["slot"] for case in cases], np.int32), starts)
    reset_seeds = np.asarray([seed for case in cases for seed in case["reset_seeds"]], np.uint32)
    map_keys = exact_reset_keys(len(cases))[map_indices]
    reset_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(reset_seeds))
    print("Resetting 32 predeclared training episodes under new bulk rules.", flush=True)
    timestep, env_params, _ = prepare_manifest_episode_reset(env, env_params, map_keys, reset_keys)
    jax.block_until_ready(timestep)
    teacher_env_cfg = checkpoint["env_config"]
    validate_bulk_teacher_environment(teacher_env_cfg, timestep.env_cfg, teacher_cfg)
    legacy_axes = env.legacy_foundation_border_axes[0, map_indices]
    teacher_slots = jnp.asarray(selection["teacher_slots"], jnp.int32)
    source_slots_device = jnp.asarray(source_slots)
    target_init = np.asarray(timestep.state.world.target_map.map)
    obstacle_init = np.asarray(timestep.state.world.padding_mask.map)
    for lane, index in enumerate(map_indices):
        for folder, observed in (("images", target_init), ("occupancy", obstacle_init)):
            np.testing.assert_array_equal(observed[lane], np.load(subset / folder / f"img_{index+1}.npy"))
    if np.any(np.asarray(timestep.observation["precision_required_band"])):
        raise AssertionError("bulk probe exposed an active precision band")
    initial_agent = np.asarray(timestep.observation["agent_states"][:, 0]).copy()

    @jax.jit
    def teacher_decision(ts, previous_actions, hidden, teacher_params):
        legacy_observation, compatible = jax.vmap(
            lambda state, axes: bulk_teacher_observation_and_compatibility(
                state, teacher_env_cfg, axes,
                executable_dig_observation=bool(getattr(teacher_cfg, "executable_dig_observation", False)),
            )
        )(ts.state, legacy_axes)
        cached, next_hidden = recurrent_teacher_rollout_observation(
            ts.observation, previous_actions, hidden, model_state.apply_fn,
            teacher_params, teacher_cfg, ts.env_cfg.enforce_foundation_border_alignment,
            source_slots_device, teacher_slots, teacher_observation=legacy_observation,
            bulk_compatible=compatible,
        )
        logits = cached[CACHED_LOGITS_KEY]
        finite = (jnp.all(jnp.isfinite(logits), axis=-1)
                  & jnp.all(jnp.isfinite(next_hidden), axis=-1)
                  & jnp.all(jnp.isfinite(cached[CACHED_VALUE_KEY]).reshape(count, -1), axis=-1))
        loaded = ts.observation["agent_states"][:, 0, 5] > 0
        return jnp.argmax(logits, axis=-1), cached[ELIGIBLE_KEY], loaded, next_hidden, finite

    @jax.jit
    def preserve_inactive(previous, candidate, active):
        def choose(old, new):
            if hasattr(new, "shape") and new.ndim and new.shape[0] == count:
                return jnp.where(active.reshape((count,) + (1,) * (new.ndim - 1)), new, old)
            return new
        return jax.tree_util.tree_map(choose, previous, candidate)

    @jax.jit
    def diagnostics(ts):
        action_map = ts.state.world.action_map.map.astype(jnp.int32)
        target = ts.state.world.target_map.map
        loaded = sum(agent.loaded.astype(jnp.int32).sum(axis=-1)
                     for agent in ts.state.agent.agent_states)
        dug = jnp.minimum(jnp.maximum(-action_map, 0), jnp.maximum(-target, 0)).sum(axis=(-2, -1))
        accepted = jnp.where(target > 0, jnp.maximum(action_map, 0), 0).sum(axis=(-2, -1))
        illegal = jnp.where(target <= 0, jnp.maximum(action_map, 0), 0).sum(axis=(-2, -1))
        finite = jnp.ones((count,), dtype=jnp.bool_)
        for leaf in jax.tree_util.tree_leaves(ts.state):
            if hasattr(leaf, "shape") and leaf.ndim and leaf.shape[0] == count:
                finite &= jnp.all(jnp.isfinite(leaf), axis=tuple(range(1, leaf.ndim)))
        return dict(
            mass=action_map.sum(axis=(-2, -1)) + loaded,
            dug=dug, accepted=accepted, illegal=illegal, loaded=loaded, finite=finite,
            target_mutation=jnp.any(target != jnp.asarray(target_init), axis=(-2, -1)),
            obstacle_mutation=jnp.any(ts.state.world.padding_mask.map != jnp.asarray(obstacle_init), axis=(-2, -1)),
        )

    initial_mass = np.asarray(diagnostics(timestep)["mass"])
    required = np.maximum(-target_init.astype(np.int32), 0).sum(axis=(-2, -1))
    hidden = initial_actor_hidden(count, teacher_cfg)
    previous_actions = jnp.zeros((count, cfg.num_prev_actions), jnp.int32)
    active = np.ones(count, bool)
    success = np.zeros(count, bool)
    first_success = np.full(count, -1, np.int32)
    counts = {name: np.zeros(count, np.int32) for name in (
        "decisions", "eligible", "empty", "eligible_empty", "loaded", "eligible_loaded",
        "teacher_do", "eligible_teacher_do", "empty_teacher_do", "eligible_empty_teacher_do",
        "no_effect", "fresh_dig_events", "eligible_fresh_dig_events",
    )}
    max_mass_residual = np.zeros(count, np.int32)
    any_target_mutation = np.zeros(count, bool)
    any_obstacle_mutation = np.zeros(count, bool)
    any_nonfinite = np.zeros(count, bool)
    trace = collections.defaultdict(list)
    rng = jax.random.PRNGKey(recipe["probe"]["step_seed"])
    last_dug = np.zeros(count, np.int32)
    horizon = recipe["probe"]["horizon"]
    print("Compiling teacher decision/compatibility and native step; no optimizer or training.", flush=True)
    for frame in range(horizon):
        action, eligible, loaded, next_hidden, actor_finite = teacher_decision(
            timestep, previous_actions, hidden, checkpoint["model"],
        )
        action_host, eligible_host, loaded_host, actor_finite_host = map(np.asarray, (action, eligible, loaded, actor_finite))
        do = action_host == int(TrackedActionType.DO)
        masks = dict(decisions=active, eligible=active & eligible_host,
                     empty=active & ~loaded_host, eligible_empty=active & ~loaded_host & eligible_host,
                     loaded=active & loaded_host, eligible_loaded=active & loaded_host & eligible_host,
                     teacher_do=active & do, eligible_teacher_do=active & do & eligible_host,
                     empty_teacher_do=active & do & ~loaded_host,
                     eligible_empty_teacher_do=active & do & ~loaded_host & eligible_host)
        for name, mask in masks.items():
            counts[name] += mask
        rng, key = jax.random.split(rng)
        candidate = env.step_no_reset(timestep, wrap_action(action, env.batch_cfg.action_type),
                                      jax.random.split(key, count))
        timestep = preserve_inactive(timestep, candidate, jnp.asarray(active))
        diag = {key: np.asarray(value) for key, value in diagnostics(timestep).items()}
        max_mass_residual = np.maximum(max_mass_residual, np.abs(diag["mass"] - initial_mass))
        any_target_mutation |= diag["target_mutation"]
        any_obstacle_mutation |= diag["obstacle_mutation"]
        any_nonfinite |= ~diag["finite"] | (active & ~actor_finite_host)
        any_nonfinite |= active & ~np.isfinite(np.asarray(timestep.reward))
        counts["no_effect"] += active & ~np.asarray(timestep.info["action_had_effect"])
        fresh = active & (diag["dug"] > last_dug)
        counts["fresh_dig_events"] += fresh
        counts["eligible_fresh_dig_events"] += fresh & eligible_host
        just_succeeded = active & np.asarray(timestep.info["task_done"])
        first_success[just_succeeded & ~success] = frame + 1
        success |= just_succeeded
        for name, value in dict(action=action_host, active=active.copy(), eligible=eligible_host,
                                loaded=loaded_host, dug=diag["dug"], accepted=diag["accepted"]).items():
            trace[name].append(value)
        previous_actions = jnp.roll(previous_actions, shift=1, axis=1).at[:, 0].set(action)
        done = np.asarray(timestep.done) | success
        previous_actions = jnp.where(jnp.asarray(done)[:, None], 0, previous_actions)
        hidden = jnp.where(jnp.asarray(done)[:, None], 0, next_hidden)
        active &= ~done
        last_dug = diag["dug"]
        if frame == 0 or (frame + 1) % 50 == 0 or not active.any():
            print(json.dumps(dict(step=frame + 1, active=int(active.sum()), successes=int(success.sum()),
                                  mean_dug_fraction=float(np.mean(diag["dug"] / required)),
                                  eligible_fraction=float(counts["eligible"].sum() / counts["decisions"].sum()),
                                  elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
        if not active.any():
            break

    def summarize(indices):
        totals = {name: int(value[indices].sum()) for name, value in counts.items()}
        fractions = {}
        for label, numerator, denominator in (
            ("all", "eligible", "decisions"), ("empty", "eligible_empty", "empty"),
            ("loaded", "eligible_loaded", "loaded"),
            ("teacher_do", "eligible_teacher_do", "teacher_do"),
            ("empty_teacher_do", "eligible_empty_teacher_do", "empty_teacher_do"),
            ("fresh_dig_events", "eligible_fresh_dig_events", "fresh_dig_events"),
        ):
            fractions[label] = totals[numerator] / totals[denominator] if totals[denominator] else None
        return dict(counts=totals, eligible_fraction=fractions,
                    successes=int(success[indices].sum()),
                    mean_dug_fraction=float(np.mean(diag["dug"][indices] / required[indices])),
                    mean_accepted_fraction=float(np.mean(diag["accepted"][indices] / required[indices])))

    integrity = dict(maximum_mass_residual=max_mass_residual.tolist(),
                     target_mutation=any_target_mutation.tolist(), obstacle_mutation=any_obstacle_mutation.tolist(),
                     nonfinite_state_or_actor=any_nonfinite.tolist())
    components = timestep.info.get("reward_components", {})
    episodes = []
    for lane in range(count):
        case = cases[map_indices[lane]]
        episodes.append(dict(source_slot=int(source_slots[lane]), subset_slot=int(map_indices[lane]),
                             source_id=case["source_id"], condition=case["condition"],
                             reset_seed=int(reset_seeds[lane]), initial_agent_state=initial_agent[lane].tolist(),
                             target_units=int(required[lane]), dug_units=int(diag["dug"][lane]),
                             accepted_dump_units=int(diag["accepted"][lane]), illegal_dump_units=int(diag["illegal"][lane]),
                             final_loaded=int(diag["loaded"][lane]), first_success_step=int(first_success[lane]),
                             final_absolute_completion=float(np.asarray(components["absolute_completion"])[lane]),
                             **summarize(np.asarray([lane]))))
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), horizon=horizon,
                  teacher_candidate_pool=recipe["teacher_pool"],
                  source_bank=str(args.source_bank), subset_bank=str(subset),
                  selection="predeclared training sources and starts; no filtering using policy returns",
                  teacher_observation=recipe["probe"]["teacher_observation"],
                  gate="actual trainer helper: same DO selected mask, volume, relift flag and admission; loaded uses validated matching machine/dump settings",
                  limitation="Immediate DO compatibility does not establish future-plan competence; rollout uses the teacher's greedy actions even on ineligible states.",
                  runtime=dict(jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
                               elapsed_seconds=time.monotonic() - started),
                  aggregate=summarize(np.arange(count)), integrity=integrity, episodes=episodes,
                  by_source=[dict(source_slot=case["slot"], source_id=case["source_id"], condition=case["condition"],
                                  **summarize(np.arange(index * starts, (index + 1) * starts)))
                             for index, case in enumerate(cases)])
    np.savez_compressed(args.output / "trace.npz", **{name: np.asarray(values) for name, values in trace.items()})
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if max_mass_residual.any() or any_target_mutation.any() or any_obstacle_mutation.any() or any_nonfinite.any():
        raise AssertionError("native integrity failure; results.json retains the evidence")
    print(json.dumps(report["aggregate"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-bank", type=Path,
                        default=Path("/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/bank_balanced/train_v3_generalist_512"))
    parser.add_argument("--recipe", type=Path, default=HERE / "recipe.json")
    parser.add_argument("--selection", type=Path, default=HERE / "selection.json")
    parser.add_argument("--prepare-only", action="store_true", help="Materialize the exact subset; do not initialize JAX or execute a policy")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.source_bank = args.source_bank.resolve(strict=True)
    recipe = json.loads(args.recipe.read_text())
    selection = json.loads(args.selection.read_text())
    subset = prepare_subset(args.source_bank, recipe, selection, args.output)
    if args.prepare_only:
        print(json.dumps(dict(subset=str(subset), maps=len(recipe["probe"]["maps"]),
                              episodes=recipe["probe"]["episode_count"])), flush=True)
        return
    run_probe(args, recipe, selection, subset)


if __name__ == "__main__":
    main()
