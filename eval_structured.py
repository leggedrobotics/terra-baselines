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
from terra.structured_actions import StructuredClock, StructuredTimeConfig, structured_progress
from train_structured import (
    CHECKPOINT_FORMAT, ModelConfig, assert_finite, checkpoint_action_interface,
    environment_kernels, initial_runner, load_checkpoint,
    load_initial_bank, policy_observation, policy_snapshot, rule_switches,
)
from utils.models import get_model_ready, validate_model_params_match
from utils.structured_ppo import ACTION_PROTOCOL, Runner, forward_step, sample_action


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--initial-states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampled", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-interface", choices=("structured", "primitive8"),
                        help="Assert the saved interface; evaluation always uses the checkpoint's protocol")
    parser.add_argument("--policy-snapshot-interval", type=int, default=0,
                        help="Save all-lane head distributions every N decisions next to output; 0 disables")
    args = parser.parse_args(argv)
    if args.policy_snapshot_interval < 0:
        parser.error("policy-snapshot-interval must be >=0")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("action_protocol") != ACTION_PROTOCOL:
        raise ValueError("expected a structured-v1 checkpoint")
    model_config = ModelConfig(checkpoint["model_config"])
    action_interface = checkpoint_action_interface(checkpoint, args.action_interface)
    model_config["action_interface"] = action_interface
    training = checkpoint["training_config"]
    timing = StructuredTimeConfig(**training["timing"])
    with args.initial_states.open("rb") as stream:
        episodes = [e for e in pickle.load(stream).get("episodes", [])
                    if e.get("decoder", "greedy") == "greedy"]
    bank, static_cfg = load_initial_bank(args.initial_states)
    safety = training.get("precision_safety", False)
    static_cfg = static_cfg._replace(precision_safety=safety)
    bank = bank._replace(env_cfg=bank.env_cfg._replace(
        precision_safety=jnp.full(bank.env_steps.shape, safety, dtype=bool)))
    lanes = int(bank.env_steps.shape[0])
    env_shape = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=bank.world.target_map.map.shape[-1])),
        executable_dig_observation=bool(model_config.get("executable_dig_observation", False)))
    model, initialized = get_model_ready(jax.random.PRNGKey(0), model_config, env_shape)
    validate_model_params_match(initialized, checkpoint["model"], "structured evaluation")
    params = jax.tree.map(jnp.asarray, checkpoint["model"])
    assert_finite(params, "evaluation parameters")
    observe, advance = environment_kernels(
        static_cfg, model_config, time_budget_s=training["time_budget_s"],
        decision_limit=training["decision_limit"], timing=timing, allow_wait=training["allow_wait"],
        time_budget_factor=training.get("time_budget_factor", 0.),
        time_budget_offset_s=training.get("time_budget_offset_s", 0.),
        decisions_per_dig_unit=training.get("decisions_per_dig_unit", 0.),
        time_limit=training.get("time_limit", True),  # rewards are not reported
        terminate_loaded_dead_ends=training.get("terminate_loaded_dead_ends", False),
        terminate_unloaded_dead_ends=training.get("terminate_unloaded_dead_ends", False),
        precision_edge_weight=training.get("precision_edge_weight", -1.))

    @jax.jit
    def step(runner, finished):
        raw, masks, (budget, decision_limit) = jax.vmap(observe)(runner.state)
        obs = policy_observation(raw, runner, budget, decision_limit)
        rng, key = jax.random.split(runner.rng)
        _, logits, hidden = forward_step(model, params, obs, runner.previous_types, runner.hidden, model_config)
        action, _ = sample_action(logits, masks, key, deterministic=not args.sampled,
                                 action_interface=action_interface)
        (state, _, duration, elapsed, clock, done, task_done, completion, budget, outcome,
         _) = jax.vmap(advance)(runner.state, action, runner.clock, runner.elapsed_s,
                               masks["loaded_dead_end"], masks["unloaded_dead_end"])
        loaded_dead_end = masks["loaded_dead_end"] & done & ~task_done
        unloaded_dead_end = masks["unloaded_dead_end"] & done & ~task_done
        progress = jax.vmap(structured_progress)(state)
        load_before = jax.vmap(lambda s: s._get_current_agent_state().loaded[0])(runner.state)
        load_after = jax.vmap(lambda s: s._get_current_agent_state().loaded[0])(state)
        unchanged_holes = jnp.all(
            (runner.state.world.action_map.map < 0) == (state.world.action_map.map < 0),
            axis=tuple(range(1, state.world.action_map.map.ndim)))
        relift = (action.action == 6) & (load_before == 0) & (load_after > 0) & unchanged_holes
        immobile = (load_before == 0) & ~masks["action_mask"][:, :4].any(-1)
        snapshot = {}
        if args.policy_snapshot_interval:
            snapshot = policy_snapshot(logits, masks, action, runner.state, load_before,
                                       runner.state.env_cfg.enforce_foundation_border_alignment.astype(bool), lanes)
            snapshot["snapshot_active"] = ~finished
        keep = lambda new, old: jnp.where(finished.reshape(finished.shape + (1,) * (new.ndim - 1)), old, new)
        arguments = jnp.stack((action.amount.astype(jnp.float32) / 6.,
                               (action.heading.astype(jnp.float32) + 1.) / 12., duration / budget), -1)
        updated = Runner(state, elapsed, clock,
                         jnp.roll(runner.previous_types, 1, -1).at[:, 0].set(action.action),
                         jnp.roll(runner.previous_arguments, 1, 1).at[:, 0].set(arguments),
                         outcome.astype(jnp.float32), hidden, rng)
        # Finished lanes stay frozen at their terminal state.
        runner = jax.tree.map(keep, updated._replace(rng=None), runner._replace(rng=None))._replace(rng=rng)
        return (runner, finished | done, task_done, completion, budget, decision_limit,
                loaded_dead_end, unloaded_dead_end, progress, load_after > 0, relift, immobile, snapshot)

    runner = jax.tree.map(jnp.asarray, initial_runner(jax.device_get(bank), model_config,
                                                      jax.random.PRNGKey(args.seed)))
    finished = jnp.zeros(lanes, bool)
    results = [None] * lanes
    relift_counts = np.zeros(lanes, np.int32)
    immobile_counts = np.zeros(lanes, np.int32)
    samples = []
    decision = 0
    with static_rules(**rule_switches(static_cfg)):
        while not bool(finished.all()):
            before = np.asarray(finished)
            (runner, finished, success, completion, budget, limit, loaded_dead_end,
             unloaded_dead_end, progress, loaded, relift, immobile, snapshot) = step(runner, finished)
            if args.policy_snapshot_interval and decision % args.policy_snapshot_interval == 0:
                assert_finite(snapshot, "evaluation policy snapshot")
                samples.append({key.removeprefix("snapshot_"): np.asarray(value)
                                for key, value in snapshot.items()})
            decision += 1
            relift_counts += np.asarray(relift, np.int32) * ~before
            immobile_counts += np.asarray(immobile, np.int32) * ~before
            for lane in np.flatnonzero(np.asarray(finished) & ~before):
                meta = episodes[lane] if lane < len(episodes) else {}
                time_reached = bool(training.get("time_limit", True)
                                    and float(runner.elapsed_s[lane]) >= float(budget[lane]))
                decision_reached = int(runner.state.env_steps[lane]) >= int(limit[lane])
                reasons = (("success", bool(success[lane])),
                           ("loaded_dead_end", bool(loaded_dead_end[lane])),
                           ("unloaded_dead_end", bool(unloaded_dead_end[lane])),
                           ("time_limit", time_reached), ("decision_limit", decision_reached))
                reason = next((name for name, reached in reasons if reached), None)
                if reason is None:
                    raise RuntimeError(f"lane {lane} terminated without a recognized reason")
                results[lane] = dict(
                    lane=int(lane), source_slot=meta.get("source_slot"), mode=meta.get("mode"),
                    reset_seed=meta.get("reset_seed"), condition=meta.get("condition"),
                    success=bool(success[lane]), completion=float(completion[lane]),
                    loaded_dead_end=bool(loaded_dead_end[lane]),
                    unloaded_dead_end=bool(unloaded_dead_end[lane]),
                    time_limit_reached=time_reached, decision_limit_reached=decision_reached,
                    termination_reason=reason,
                    loaded_at_end=bool(loaded[lane]),
                    **{f"dug_{part}": float(value[lane]) for part, value in progress.items()},
                    relifts=int(relift_counts[lane]), immobile_empty_decisions=int(immobile_counts[lane]),
                    modeled_hours=float(runner.elapsed_s[lane]) / 3600,
                    time_scale_hours=float(budget[lane]) / 3600, decisions=int(runner.state.env_steps[lane]),
                    decision_limit=int(limit[lane]))
    summary = dict(checkpoint=str(args.checkpoint), update=int(checkpoint["next_update"]),
                   action_interface=action_interface,
                   starts=lanes, successes=sum(r["success"] for r in results),
                   loaded_dead_ends=sum(r["loaded_dead_end"] for r in results),
                   unloaded_dead_ends=sum(r["unloaded_dead_end"] for r in results),
                   time_limit_reached=sum(r["time_limit_reached"] for r in results),
                   decision_limit_reached=sum(r["decision_limit_reached"] for r in results),
                   time_limit_failure_rate=sum(r["time_limit_reached"] and not r["success"] for r in results) / lanes,
                   decision_limit_failure_rate=sum(r["decision_limit_reached"] and not r["success"] for r in results) / lanes,
                   termination_reason_counts={reason: sum(r["termination_reason"] == reason for r in results)
                                              for reason in ("success", "loaded_dead_end", "unloaded_dead_end",
                                                             "time_limit", "decision_limit")},
                   mean_completion=float(np.mean([r["completion"] for r in results])),
                   decoder="sampled" if args.sampled else "greedy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert_finite(runner, "evaluation runner")
    if samples:
        np.savez_compressed(args.output.with_suffix(".policy.npz"),
                            **{key: np.stack([row[key] for row in samples]) for key in samples[0]},
                            action_interface=action_interface,
                            argument_heads_active=action_interface == "structured",
                            decision_interval=args.policy_snapshot_interval)
    args.output.write_text(json.dumps(dict(summary=summary, episodes=results), indent=1) + "\n")
    print(json.dumps(summary))
    for r in results:
        print(f"{r['source_slot']} {r['mode']} {r['reset_seed']}: success={r['success']} "
              f"completion={r['completion']:.3f} hours={r['modeled_hours']:.2f} (scale {r['time_scale_hours']:.2f}) "
              f"decisions={r['decisions']}/{r['decision_limit']} loaded_dead_end={r['loaded_dead_end']}")


if __name__ == "__main__":
    main()
