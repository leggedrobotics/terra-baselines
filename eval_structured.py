"""Greedy evaluation of a structured-v1 checkpoint on saved initial states.

Runs every start of a saved bank (greedy starts only when the bank records
decoders) to termination under the checkpoint's own rules and episode limits
(time budget or time-cost scale, decision limit) and reports success, completion, modeled hours
and decisions per start. Directly comparable with the oracle panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig, MapsDimsConfig
from terra.state import static_rules
from terra.structured_actions import StructuredClock, StructuredTimeConfig
from train_structured import (
    CHECKPOINT_FORMAT, ModelConfig, environment_kernels, initial_runner, load_checkpoint,
    load_initial_bank, policy_observation, rule_switches,
)
from utils.models import get_model_ready
from utils.structured_ppo import ACTION_PROTOCOL, Runner, forward_step, sample_action


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--initial-states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampled", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("action_protocol") != ACTION_PROTOCOL:
        raise ValueError("expected a structured-v1 checkpoint")
    model_config = ModelConfig(checkpoint["model_config"])
    training = checkpoint["training_config"]
    timing = StructuredTimeConfig(**training["timing"])
    with args.initial_states.open("rb") as stream:
        episodes = [e for e in pickle.load(stream).get("episodes", [])
                    if e.get("decoder", "greedy") == "greedy"]
    bank, static_cfg = load_initial_bank(args.initial_states)
    lanes = int(bank.env_steps.shape[0])
    env_shape = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=bank.world.target_map.map.shape[-1])),
        executable_dig_observation=bool(model_config.get("executable_dig_observation", False)))
    model, _ = get_model_ready(jax.random.PRNGKey(0), model_config, env_shape)
    params = jax.tree.map(jnp.asarray, checkpoint["model"])
    observe, advance = environment_kernels(
        static_cfg, model_config, time_budget_s=training["time_budget_s"],
        decision_limit=training["decision_limit"], timing=timing, allow_wait=training["allow_wait"],
        time_budget_factor=training.get("time_budget_factor", 0.),
        time_budget_offset_s=training.get("time_budget_offset_s", 0.),
        decisions_per_dig_unit=training.get("decisions_per_dig_unit", 0.),
        time_limit=training.get("time_limit", True))  # rewards are not reported

    @jax.jit
    def step(runner, finished):
        raw, masks, (budget, decision_limit) = jax.vmap(observe)(runner.state)
        obs = policy_observation(raw, runner, budget, decision_limit)
        rng, key = jax.random.split(runner.rng)
        _, logits, hidden = forward_step(model, params, obs, runner.previous_types, runner.hidden, model_config)
        action, _ = sample_action(logits, masks, key, deterministic=not args.sampled)
        (state, _, duration, elapsed, clock, done, task_done, completion, budget, outcome,
         _) = jax.vmap(advance)(runner.state, action, runner.clock, runner.elapsed_s)
        keep = lambda new, old: jnp.where(finished.reshape(finished.shape + (1,) * (new.ndim - 1)), old, new)
        arguments = jnp.stack((action.amount.astype(jnp.float32) / 6.,
                               (action.heading.astype(jnp.float32) + 1.) / 12., duration / budget), -1)
        updated = Runner(state, elapsed, clock,
                         jnp.roll(runner.previous_types, 1, -1).at[:, 0].set(action.action),
                         jnp.roll(runner.previous_arguments, 1, 1).at[:, 0].set(arguments),
                         outcome.astype(jnp.float32), hidden, rng)
        # Finished lanes stay frozen at their terminal state.
        runner = jax.tree.map(keep, updated._replace(rng=None), runner._replace(rng=None))._replace(rng=rng)
        return runner, finished | done, task_done, completion, budget, decision_limit

    runner = jax.tree.map(jnp.asarray, initial_runner(jax.device_get(bank), model_config,
                                                      jax.random.PRNGKey(args.seed)))
    finished = jnp.zeros(lanes, bool)
    results = [None] * lanes
    with static_rules(**rule_switches(static_cfg)):
        while not bool(finished.all()):
            before = np.asarray(finished)
            runner, finished, success, completion, budget, limit = step(runner, finished)
            for lane in np.flatnonzero(np.asarray(finished) & ~before):
                meta = episodes[lane] if lane < len(episodes) else {}
                results[lane] = dict(
                    lane=int(lane), source_slot=meta.get("source_slot"), mode=meta.get("mode"),
                    reset_seed=meta.get("reset_seed"), condition=meta.get("condition"),
                    success=bool(success[lane]), completion=float(completion[lane]),
                    modeled_hours=float(runner.elapsed_s[lane]) / 3600,
                    time_scale_hours=float(budget[lane]) / 3600, decisions=int(runner.state.env_steps[lane]),
                    decision_limit=int(limit[lane]))
    summary = dict(checkpoint=str(args.checkpoint), update=int(checkpoint["next_update"]),
                   starts=lanes, successes=sum(r["success"] for r in results),
                   mean_completion=float(np.mean([r["completion"] for r in results])),
                   decoder="sampled" if args.sampled else "greedy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, episodes=results), indent=1) + "\n")
    print(json.dumps(summary))
    for r in results:
        print(f"{r['source_slot']} {r['mode']} {r['reset_seed']}: success={r['success']} "
              f"completion={r['completion']:.3f} hours={r['modeled_hours']:.2f} (scale {r['time_scale_hours']:.2f}) "
              f"decisions={r['decisions']}/{r['decision_limit']}")


if __name__ == "__main__":
    main()
