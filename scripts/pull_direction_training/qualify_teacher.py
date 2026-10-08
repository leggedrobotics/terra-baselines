"""Qualify the frozen GRU on training maps with the new bulk cutting rule."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from eval_fixed_bank import configure_for_bank, prepare_manifest_episode_reset, selected_map_indices
from eval_mcts import rollout_episode
from train_mixed import make_mixed_agent_states
from utils.helpers import checkpoint_evaluation_config, load_pkl_object, register_checkpoint_config_classes
from utils.models import validate_model_params_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, nargs="+", help="Zero-based training slots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = Path(os.environ["DATASET_PATH"]) / "train_v3_generalist_512"
    rows = [json.loads(line) for line in (bank / "manifest.jsonl").read_text().splitlines()]
    by_slot = {int(row["slot_index"]) - 1: row for row in rows}
    if args.slots is None:
        # Eight different foundation conditions, selected before policy evaluation.
        selected = {}
        for row in rows:
            if row["family"] == "foundation" and row["split"] == "train":
                selected.setdefault(row["primary_cell"], int(row["slot_index"]) - 1)
        slots = list(selected.values())[:8]
    else:
        slots = args.slots
    assert len(set(slots)) == len(slots) and slots
    assert all(by_slot[slot]["split"] == "train" for slot in slots)
    expected = np.repeat(np.asarray(slots, dtype=np.int32), 4)
    count = len(expected)
    register_checkpoint_config_classes()
    checkpoint = load_pkl_object(args.checkpoint)
    cfg = configure_for_bank(checkpoint_evaluation_config(checkpoint), "train_v3_generalist_512", count)
    cfg.pull_direction_alignment = True
    cfg.dig_pull_min_length_m = 2.5
    cfg.enforce_foundation_border_alignment = False
    cfg.precision_required_band_observation = False
    cfg.precision_episode_fraction = 0.0
    cfg.pull_direction_training_slots = None
    cfg.recurrent_teacher = False
    _, env, params, state = make_mixed_agent_states(cfg)
    params = jax.tree_util.tree_map(lambda value: value[0], params)
    validate_model_params_match(state.params, checkpoint["model"], "GRU110000 teacher")
    found = {}
    for offset in range(0, 1_000_000, 8192):
        keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(offset, offset + 8192, dtype=jnp.uint32))
        actual = selected_map_indices(keys, len(rows))
        for key, slot in zip(np.asarray(keys), actual):
            if int(slot) in slots:
                found.setdefault(int(slot), key)
        if len(found) == len(slots):
            break
    assert len(found) == len(slots), "Could not resolve selected training maps"
    map_keys = jnp.asarray(np.stack([found[int(slot)] for slot in expected]))
    seeds = np.arange(820000, 820000 + count, dtype=np.uint32)
    reset_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    initial, params, _ = prepare_manifest_episode_reset(env, params, map_keys, reset_keys)
    rewards, stats, _ = rollout_episode(
        env, SimpleNamespace(apply=state.apply_fn), checkpoint["model"], params, cfg,
        max_frames=450, deterministic=True, seed=20261006, use_mcts=False,
        record_observations=False, record_actions=False, preserve_terminal_states=True,
        expected_slot_indices=expected, initial_timestep=initial,
    )
    jax.block_until_ready(stats)
    assert np.isfinite(rewards).all()
    integrity = stats["integrity"]
    assert bool(integrity["supported"])
    np.testing.assert_array_equal(np.asarray(integrity["slot_index_zero_based"]), expected)
    for name in ("maximum_mass_residual", "target_mutation", "obstacle_mutation", "nonfinite_state"):
        assert np.all(np.asarray(integrity[name]) == 0), name
    success = np.asarray(stats["episode_done_once"], dtype=bool)
    completion = np.asarray(stats["terminal_completion"]["absolute"])
    np.testing.assert_array_equal(success, np.isclose(completion, 1, atol=1e-6, rtol=0))
    results = []
    for index, slot in enumerate(slots):
        selection = slice(index * 4, (index + 1) * 4)
        results.append(dict(slot=slot, source_id=by_slot[slot]["source_id"],
                            condition=by_slot[slot]["primary_cell"],
                            successes=int(success[selection].sum()), starts=4,
                            completion=completion[selection].tolist()))
    qualified = [row["slot"] for row in results if row["successes"] == row["starts"]]
    report = dict(checkpoint=args.checkpoint, horizon=450,
                  observation_path="saved GRU preprocessing of current new-rule observations",
                  qualification="all four preselected starts succeed; training split only",
                  seed=seeds.tolist(), teacher_slots=qualified, results=results)
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
