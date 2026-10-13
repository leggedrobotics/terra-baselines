"""Check finite native update/resume artifacts for one action-interface arm."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from train_structured import load_checkpoint


def finite(tree, label):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise AssertionError(f"{label}: missing leaves")
    for value in leaves:
        if not np.isfinite(np.asarray(value)).all():
            raise AssertionError(f"{label}: nonfinite leaf")


def changed(before, after):
    if jax.tree.structure(before) != jax.tree.structure(after):
        raise AssertionError("resume changed the parameter/state tree")
    return any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--interface", required=True, choices=("primitive8", "structured"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--compare-initial", type=Path,
                        help="Other arm's update-zero checkpoint; compare with this arm's update zero")
    args = parser.parse_args()
    before, after = load_checkpoint(args.before), load_checkpoint(args.after)
    for name, payload in (("before", before), ("after", after)):
        for field in ("model", "optimizer_state", "runner"):
            if payload[field] is None:
                raise AssertionError(f"{name}: missing {field}")
            finite(payload[field], f"{name}/{field}")
        training = payload["training_config"]
        model = payload["model_config"]
        assert training["action_interface"] == args.interface
        assert model["action_interface"] == args.interface
        for setting in ("precision_safety", "precision_observation_v2",
                        "terminate_loaded_dead_ends", "terminate_unloaded_dead_ends"):
            assert training[setting], setting
        assert training["precision_edge_weight"] == 0.5
        assert training["advantage_normalization"] == "per_mode"
        assert training["decision_limit"] == training["num_steps"] == 4
        assert training["decisions_per_dig_unit"] == 0
        runner = payload["runner"]
        age = np.asarray(runner.state.env_steps)
        assert np.all((age >= 0) & (age < 4)), "runner escaped the smoke decision limit"
        reset = age == 0
        for value in (runner.elapsed_s, runner.previous_types, runner.previous_arguments,
                      runner.previous_outcome, runner.hidden, runner.clock.visit_open, runner.clock.moved):
            assert np.all(np.asarray(value)[reset] == 0), "reset context leaked"
    assert after["next_update"] == before["next_update"] + 1
    expected_steps = training["epochs"] * training["minibatches"]
    assert after["train_state_step"] - before["train_state_step"] == expected_steps
    assert changed(before["model"], after["model"]), "no parameter update on resume"
    assert changed(before["optimizer_state"], after["optimizer_state"]), "optimizer did not advance"
    assert changed(before["runner"].rng, after["runner"].rng), "RNG did not advance"
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    assert len(rows) == after["next_update"], "missing or duplicate smoke metrics"
    for update, row in enumerate(rows, 1):
        assert row["train/update"] == update
        assert row["episode/count"] > 0, "no terminal/reset event in cap smoke"
        for key, value in row.items():
            if key.startswith(("optimization/", "loss/")) and isinstance(value, (int, float)):
                assert np.isfinite(value), key
        if args.interface == "primitive8":
            for head in ("move", "turn", "do"):
                assert row[f"optimization/head_{head}_grad_norm"] == 0
    result = dict(action_interface=args.interface, passed=True,
                  before_update=before["next_update"], after_update=after["next_update"],
                  before_optimizer_step=before["train_state_step"],
                  after_optimizer_step=after["train_state_step"],
                  finite=["model", "optimizer_state", "runner"],
                  parameters_advanced=True, optimizer_advanced=True, rng_advanced=True,
                  scope="Numerical update/resume qualification, not learned policy quality")
    if args.evaluation:
        evaluation = json.loads(args.evaluation.read_text())
        assert len(evaluation["episodes"]) == evaluation["summary"]["starts"] > 0
        assert evaluation["summary"]["action_interface"] == args.interface
        assert evaluation["summary"]["update"] == after["next_update"]
        assert Path(evaluation["summary"]["checkpoint"]).resolve() == args.after.resolve()
        for episode in evaluation["episodes"]:
            assert 0 < episode["decisions"] <= episode["decision_limit"] == 4
            for key in ("dug_total", "dug_edge", "dug_inner", "modeled_hours"):
                assert np.isfinite(episode[key]), key
            for key in ("loaded_dead_end", "unloaded_dead_end", "loaded_at_end"):
                assert isinstance(episode[key], bool), key
        result["evaluation_starts"] = evaluation["summary"]["starts"]
    sample_path = args.after.parent / "policy_samples" / f"update_{after['next_update']:06d}.npz"
    with np.load(sample_path) as sample:
        assert str(sample["action_interface"]) == args.interface
        for name in ("action", "move", "turn", "do"):
            probs = sample[name + "_probs"]
            assert np.isfinite(probs).all()
            np.testing.assert_allclose(probs.sum(-1), 1., atol=1e-5)
    result["head_probability_snapshot_checked"] = True
    if args.compare_initial:
        initial = load_checkpoint(args.after.parent / "checkpoint_update_000000.pkl")
        other = load_checkpoint(args.compare_initial)
        assert initial["next_update"] == other["next_update"] == 0
        assert not changed(initial["model"], other["model"]), "arms have different initial parameters"
        result["identical_initial_model"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
