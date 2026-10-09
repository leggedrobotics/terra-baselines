"""Opt-in solo structured-action PPO using saved native Terra initial states.

This entrypoint deliberately has its own action/reward/checkpoint protocol.
Legacy train_mixed.py and legacy checkpoints retain their original contract.
See docs/STRUCTURED_ACTIONS.md for the bounded first-update command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from functools import partial
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from terra.config import BatchConfig, MapsDimsConfig
from terra.env import TerraEnv
from terra.state import static_rules
from terra.structured_actions import (
    StructuredAction, StructuredClock, StructuredTimeConfig,
    structured_action_masks, structured_transition, structured_termination,
)
from terra.wrappers import LocalMapWrapper
from utils.models import get_model_ready, validate_model_params_match
from utils.structured_ppo import (
    ACTION_PROTOCOL, Runner, StructuredRollout, duration_gae, forward_step, policy_masks,
    ppo_update, sample_action, warm_start_params,
)
from utils.utils_ppo import obs_to_model_input


CHECKPOINT_FORMAT = "terra_structured_ppo_v1"


class ModelConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def default_model_config():
    return ModelConfig(
        clip_action_maps=True, loaded_max=127,
        local_map_normalization_bounds=(-16, 16), maps_net_normalization_bounds=(-10, 10),
        local_map_area_scale=1., model_core="mlp", actor_core="gru", actor_gru_hidden_dim=64,
        model_size="medium", map_encoder="atari", num_prev_actions=5,
        structured_actions=True, action_logit_masking=False, time_observation_mode="none",
        admissible_dig_observation=True, executable_dig_observation=True,
        native_dump_observation=True,
    )


def load_checkpoint(path):
    from utils.helpers import register_checkpoint_config_classes
    register_checkpoint_config_classes()
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_initial_bank(path, indices=None):
    with Path(path).open("rb") as stream:
        bank = pickle.load(stream)
    initial = bank["initial"]
    if indices is not None:
        initial = [initial[index] for index in indices]
    elif "episodes" in bank:
        # The diagnosis bank records the same starts for multiple decoders.
        initial = [state for episode, state in zip(bank["episodes"], initial)
                   if episode.get("decoder", "greedy") == "greedy"]
    if not initial:
        raise ValueError("initial-state bank is empty")
    first = initial[0]
    for state in initial:
        current = state._get_current_agent_state()
        if (int(state.agent.num_agents) != 1 or int(current.agent_type[0]) != 0
                or int(current.action_type[0]) != 0):
            raise ValueError("structured-v1 supports one tracked excavator only")
        if int(state.env_steps) != 0:
            raise ValueError("bank must contain initial states with env_steps == 0")
        if int(state.env_cfg.agent.angles_base) != 12 or int(state.env_cfg.agent.angles_cabin) != 12:
            raise ValueError("structured-v1 requires twelve base and cabin bins")
        if state.world.target_map.map.shape != first.world.target_map.map.shape:
            raise ValueError("initial bank map shapes must match")
        for field in first.env_cfg._fields:
            if field == "enforce_foundation_border_alignment":
                continue
            for left, right in zip(jax.tree.leaves(getattr(first.env_cfg, field)),
                                   jax.tree.leaves(getattr(state.env_cfg, field))):
                if not np.array_equal(left, right):
                    raise ValueError(f"bank rule mismatch in {field}; split the bank by rule set")
    static_cfg = jax.tree.map(lambda x: np.asarray(x).item() if np.ndim(x) == 0 else x, first.env_cfg)
    return jax.tree.map(lambda *x: jnp.stack(x), *initial), static_cfg


def make_environment_kernels(static_cfg, model_config, *, time_budget_s, decision_limit,
                             timing, allow_wait):
    """Compile batching once; preserve each row's precision-mode flag."""
    def native(state):
        return state._replace(env_cfg=static_cfg._replace(
            enforce_foundation_border_alignment=state.env_cfg.enforce_foundation_border_alignment))

    def observe_one(state):
        state = native(state)
        state = LocalMapWrapper.wrap(
            state,
            executable_dig_observation=bool(model_config.get("executable_dig_observation", False)),
            native_dump_observation=bool(model_config.get("native_dump_observation", False)),
        )
        observation = TerraEnv._state_to_obs_dict(state)
        if model_config.get("movement_feasibility_observation", False):
            observation["movement_feasibility"] = state._movement_feasibility_tracked().astype(jnp.float32)
        masks = policy_masks(structured_action_masks(state), allow_wait=allow_wait)
        return observation, masks

    def advance_one(state, action, clock, elapsed):
        result = structured_transition(native(state), action, clock=clock,
                                       timing=timing, time_budget_s=time_budget_s)
        next_state = jax.tree.map(lambda value, old: jnp.asarray(value, dtype=old.dtype),
                                 result.state._replace(env_cfg=state.env_cfg), state)
        elapsed = elapsed + result.duration_s
        terminal = structured_termination(next_state, elapsed, time_budget_s=time_budget_s,
                                           decision_limit=decision_limit)
        return (next_state, result.reward + terminal.reward, result.duration_s,
                elapsed, StructuredClock(result.info["time_visit_open"], result.info["time_moved"]),
                terminal.done, terminal.task_done,
                jnp.stack((result.info["action_had_effect"], result.info["material_or_load_changed"])),
                jnp.stack((result.info["transition_mass_residual"],
                           result.info["target_mutation"].astype(jnp.int32),
                           result.info["obstacle_mutation"].astype(jnp.int32))))

    def observe(states):
        with static_rules(pull_cone=float(static_cfg.pull_half_angle_rad) > 0,
                          tracked_move_keeps_turn=bool(static_cfg.tracked_move_keeps_turn)):
            return jax.vmap(observe_one)(states)

    def advance(states, actions, clocks, elapsed):
        with static_rules(pull_cone=float(static_cfg.pull_half_angle_rad) > 0,
                          tracked_move_keeps_turn=bool(static_cfg.tracked_move_keeps_turn)):
            return jax.vmap(advance_one)(states, actions, clocks, elapsed)

    return jax.jit(observe), jax.jit(advance)


def policy_observation(raw, runner, time_budget_s, decision_limit):
    remaining = jnp.clip(1. - runner.elapsed_s / time_budget_s, 0., 1.)
    decision_remaining = jnp.clip(1. - runner.state.env_steps / decision_limit, 0., 1.)
    return {
        **raw, "remaining_time": remaining,
        "previous_action_outcome": runner.previous_outcome,
        "structured_context": jnp.concatenate((
            remaining[:, None], decision_remaining[:, None],
            runner.clock.visit_open[:, None].astype(jnp.float32),
            runner.clock.moved[:, None].astype(jnp.float32),
            runner.previous_arguments.reshape((remaining.shape[0], -1))), axis=-1),
    }


def initial_runner(bank, config, num_envs, rng):
    rng, key = jax.random.split(rng)
    ids = jax.random.randint(key, (num_envs,), 0, bank.env_steps.shape[0])
    state = jax.tree.map(lambda x: x[ids], bank)
    return Runner(state, jnp.zeros(num_envs),
                  StructuredClock(jnp.zeros(num_envs, bool), jnp.zeros(num_envs, bool)),
                  jnp.zeros((num_envs, config["num_prev_actions"]), jnp.int32),
                  jnp.zeros((num_envs, config["num_prev_actions"], 3), jnp.float32),
                  jnp.zeros((num_envs, 2), jnp.float32),
                  jnp.zeros((num_envs, config["actor_gru_hidden_dim"] if config["actor_core"] == "gru" else 0)), rng)


def collect_rollout(train_state, runner, bank, observe, advance, select, config,
                    *, num_steps, time_budget_s, decision_limit, deterministic=False):
    transitions = []
    initial_hidden = runner.hidden
    max_integrity = jnp.zeros((3,), jnp.int32)
    for _ in range(num_steps):
        raw, masks = observe(runner.state)
        obs = policy_observation(raw, runner, time_budget_s, decision_limit)
        rng, key, reset_key = jax.random.split(runner.rng, 3)
        value, action, log_prob, next_hidden = select(
            train_state.params, obs, runner.previous_types, runner.hidden, masks, key, deterministic)
        (next_state, reward, duration, elapsed, clock, done, task_done,
         outcome, integrity) = advance(runner.state, action, runner.clock, runner.elapsed_s)
        max_integrity = jnp.maximum(max_integrity, integrity.max(axis=0))
        transitions.append(StructuredRollout(obs, masks, runner.previous_types, action,
                                            log_prob, value, reward, duration, done, task_done))
        ids = jax.random.randint(reset_key, done.shape, 0, bank.env_steps.shape[0])
        resets = jax.tree.map(lambda x: x[ids], bank)
        def reset(value, fresh):
            selector = done.reshape(done.shape + (1,) * (value.ndim - 1))
            return jnp.where(selector, fresh, value)
        next_state = jax.tree.map(reset, next_state, resets)
        previous_types = jnp.roll(runner.previous_types, 1, -1).at[:, 0].set(action.action)
        # Canonical values for inactive args avoid conditioning on sampled junk.
        args = jnp.stack((action.amount.astype(jnp.float32) / 6.,
                          (action.heading.astype(jnp.float32) + 1.) / 12.,
                          duration / time_budget_s), -1)
        previous_arguments = jnp.roll(runner.previous_arguments, 1, 1).at[:, 0].set(args)
        runner = Runner(next_state, reset(elapsed, jnp.zeros_like(elapsed)),
                        jax.tree.map(lambda x: reset(x, jnp.zeros_like(x)), clock),
                        reset(previous_types, jnp.zeros_like(previous_types)),
                        reset(previous_arguments, jnp.zeros_like(previous_arguments)),
                        reset(outcome, jnp.zeros_like(outcome)).astype(jnp.float32),
                        reset(next_hidden, jnp.zeros_like(next_hidden)), rng)
    rollout = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *transitions)
    return runner, rollout, initial_hidden, max_integrity


def assert_finite(tree, label):
    for value in jax.tree.leaves(jax.device_get(tree)):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite {label}")


def restore_runner(saved, template):
    """Restore stable batched leaves, including append-only EnvConfig defaults.

    Historical NamedTuple pickles materialize newly appended config fields as
    scalar defaults. Broadcast only a scalar env_cfg value equal to the current
    bank; reject all other shape changes instead of silently reshaping state.
    """
    def restore(path, value, expected):
        value = np.asarray(value)
        expected_host = np.asarray(expected)
        if value.shape != expected_host.shape:
            is_config = any(getattr(part, "name", None) == "env_cfg" for part in path)
            if not (is_config and value.ndim == 0 and np.all(expected_host == value)):
                raise ValueError(f"restored runner shape mismatch: {jax.tree_util.keystr(path)}")
            value = np.broadcast_to(value, expected_host.shape)
        return jnp.asarray(value, dtype=expected.dtype)
    return jax.tree_util.tree_map_with_path(restore, saved, template)


def save_checkpoint(path, train_state, runner, model_config, training_config, next_update):
    payload = dict(format=CHECKPOINT_FORMAT, action_protocol=ACTION_PROTOCOL,
                   model=jax.device_get(train_state.params),
                   optimizer_state=jax.device_get(train_state.opt_state),
                   train_state_step=int(train_state.step), runner=jax.device_get(runner),
                   model_config=dict(model_config), training_config=training_config,
                   next_update=next_update)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-states", required=True, type=Path)
    parser.add_argument("--bank-indices", help="Optional comma-separated bank indices for a bounded diagnostic")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=1, help="Absolute completed PPO update target")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--time-budget-s", type=float, default=14400.)
    parser.add_argument("--decision-limit", type=int, default=450)
    parser.add_argument("--gamma", type=float, default=1.0, help="1.0 preserves material-potential telescoping; discounting is an explicit experiment")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--discount-reference-s", type=float, default=30.)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=2.)
    parser.add_argument("--entropy-type", type=float, default=0.01)
    parser.add_argument("--entropy-move", type=float, default=0.01)
    parser.add_argument("--entropy-turn", type=float, default=0.01)
    parser.add_argument("--entropy-heading", type=float, default=0.01)
    parser.add_argument("--allow-wait", action="store_true")
    parser.add_argument("--timing-json", type=Path, help="JSON overrides for StructuredTimeConfig (modeled seconds)")
    parser.add_argument("--model-config", type=Path, help="JSON architecture/observation options; required for a different scratch trunk")
    parser.add_argument("--warm-start-from", type=Path, help="Reuse compatible legacy backbone/GRU; fresh critic, args and optimizer")
    parser.add_argument("--resume-from", type=Path, help="Native structured checkpoint, including optimizer and live environments")
    parser.add_argument("--teacher-checkpoint", type=Path, help="Reserved: currently rejected because no 8-way teacher mapping is defined")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    args = parser.parse_args(argv)
    if args.teacher_checkpoint:
        parser.error("teacher KL is unsupported for structured-v1: cabin/DO probabilities need an explicit mapping")
    if args.resume_from and args.warm_start_from:
        parser.error("choose resume or warm start")
    for name in ("updates", "num_envs", "num_steps", "epochs", "minibatches", "decision_limit", "checkpoint_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >=1")
    if args.num_envs % args.minibatches:
        parser.error("num-envs must divide evenly into whole-trajectory minibatches")
    if not (0 < args.gamma <= 1 and 0 < args.gae_lambda <= 1):
        parser.error("gamma and gae-lambda must be in (0,1]")
    for name in ("time_budget_s", "discount_reference_s", "learning_rate", "clip_eps"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    for name in ("vf_coef", "entropy_type", "entropy_move", "entropy_turn", "entropy_heading"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative and finite")
    return args


def main(argv=None):
    args = parse_args(argv)
    training = vars(args).copy()
    training = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in training.items()}
    timing_values = json.loads(args.timing_json.read_text()) if args.timing_json else {}
    timing = StructuredTimeConfig(**timing_values)
    if any(not np.isfinite(x) or x < 0 for x in timing) or timing.nav_speed_mps <= 0:
        raise ValueError("timing values must be finite and nonnegative; navigation speed must be positive")
    training["timing"] = timing._asdict()
    bank_stat = args.initial_states.stat()
    training["initial_bank_size"] = bank_stat.st_size
    training["initial_bank_mtime_ns"] = bank_stat.st_mtime_ns
    model_config = default_model_config()
    source = None
    checkpoint_path = args.resume_from or args.warm_start_from
    if checkpoint_path:
        source = load_checkpoint(checkpoint_path)
        saved_config = source.get("model_config", source.get("train_config", {}))
        if is_dataclass(saved_config):
            saved_config = asdict(saved_config)
        elif not isinstance(saved_config, dict):
            saved_config = vars(saved_config)
        model_config.update(saved_config)
        # Missing legacy options mean their original defaults, not this
        # pilot's scratch GRU/executable-observation defaults.
        for name, default in {
            "actor_core": "mlp", "model_core": "mlp", "model_size": "base",
            "map_encoder": "atari", "admissible_dig_observation": False,
            "executable_dig_observation": False, "native_dump_observation": False,
        }.items():
            model_config[name] = saved_config.get(name, default)
    if args.model_config:
        model_config.update(json.loads(args.model_config.read_text()))
    model_config.update(structured_actions=True, action_logit_masking=False)
    if model_config.get("actor_residual_head", False):
        raise ValueError("structured-v1 does not support the legacy residual actor variant")
    if args.resume_from:
        if source.get("format") != CHECKPOINT_FORMAT or source.get("action_protocol") != ACTION_PROTOCOL:
            raise ValueError("native resume requires a structured-v1 checkpoint; use warm-start-from for legacy policies")
        for key, value in source["training_config"].items():
            if key not in {"output", "updates", "checkpoint_interval", "resume_from", "warm_start_from"} and training.get(key) != value:
                raise ValueError(f"native resume protocol mismatch: {key}")
        if dict(model_config) != source["model_config"]:
            raise ValueError("native resume must preserve model configuration")
    bank, static_cfg = load_initial_bank(args.initial_states,
        [int(x) for x in args.bank_indices.split(",")] if args.bank_indices else None)
    edge = bank.world.target_map.map.shape[-1]
    env_shape = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=edge)),
        executable_dig_observation=bool(model_config.get("executable_dig_observation", False)))
    rng, model_key = jax.random.split(jax.random.PRNGKey(args.seed))
    model, params = get_model_ready(model_key, model_config, env_shape)
    if args.warm_start_from:
        params = warm_start_params(params, source["model"])
        print("Warm start: reused backbone/GRU/type head; reset critic, argument heads, optimizer and clocks", flush=True)
    # get_model_ready deliberately initializes on CPU; rollout/optimization
    # must put the whole tree on the selected training device together.
    params = jax.device_put(params, jax.devices()[0])
    train_state = TrainState.create(apply_fn=model.apply, params=params,
        tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.learning_rate, eps=1e-5)))
    runner = initial_runner(bank, model_config, args.num_envs, rng)
    start_update = 0
    if args.resume_from:
        validate_model_params_match(params, source["model"], "structured resume")
        train_state = train_state.replace(params=jax.tree.map(jnp.asarray, source["model"]),
            opt_state=jax.tree.map(jnp.asarray, source["optimizer_state"]), step=jnp.asarray(source["train_state_step"]))
        runner = restore_runner(source["runner"], runner)
        start_update = int(source["next_update"])
    train_state = jax.device_put(train_state, jax.devices()[0])
    runner = jax.device_put(runner, jax.devices()[0])
    assert_finite((train_state.params, train_state.opt_state, runner), "initial/restored state")
    observe, advance = make_environment_kernels(static_cfg, model_config,
        time_budget_s=args.time_budget_s, decision_limit=args.decision_limit,
        timing=timing, allow_wait=args.allow_wait)

    @partial(jax.jit, static_argnums=(6,))
    def select(params, observation, previous_types, hidden, masks, key, deterministic=False):
        value, logits, hidden = forward_step(model, params, observation, previous_types, hidden, model_config)
        action, log_prob = sample_action(logits, masks, key, deterministic=deterministic)
        return value, action, log_prob, hidden

    @jax.jit
    def bootstrap(params, observation, previous_types):
        inputs = obs_to_model_input(observation, previous_types, model_config)
        return model.apply(params, inputs, method="value")[..., 0]

    update = jax.jit(partial(ppo_update, model=model, config=model_config,
        clip_eps=args.clip_eps, vf_coef=args.vf_coef,
        entropy_coefs=(args.entropy_type, args.entropy_move, args.entropy_turn, args.entropy_heading)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(
        dict(training=training, model=dict(model_config), action_protocol=ACTION_PROTOCOL), indent=2, default=str) + "\n")
    print(json.dumps(dict(phase="native_rollout", bank_states=int(bank.env_steps.shape[0]),
                         devices=[str(d) for d in jax.devices()], start_update=start_update,
                         reward_protocol="material_time_v1", action_protocol=ACTION_PROTOCOL)), flush=True)
    for index in range(start_update, args.updates):
        start = time.monotonic()
        runner, rollout, hidden, residual = collect_rollout(
            train_state, runner, bank, observe, advance, select, model_config,
            num_steps=args.num_steps, time_budget_s=args.time_budget_s, decision_limit=args.decision_limit)
        raw, _ = observe(runner.state)
        obs = policy_observation(raw, runner, args.time_budget_s, args.decision_limit)
        last_value = bootstrap(train_state.params, obs, runner.previous_types)
        advantage, target = duration_gae(rollout, last_value, args.gamma, args.gae_lambda, args.discount_reference_s)
        assert_finite((rollout, advantage, target), "rollout/GAE")
        if np.any(np.asarray(residual) != 0):
            raise RuntimeError(f"native transition integrity failed (mass,target,obstacle): {np.asarray(residual)}")
        print(json.dumps(dict(phase="ppo_update", update=index+1,
                             rollout_seconds=time.monotonic()-start)), flush=True)
        for _ in range(args.epochs):
            rng, shuffle_key = jax.random.split(runner.rng)
            runner = runner._replace(rng=rng)
            ids = jax.random.permutation(shuffle_key, args.num_envs).reshape((args.minibatches, -1))
            for rows in ids:
                batch = jax.tree.map(lambda x: x[rows], rollout)
                train_state, metrics = update(train_state, rollout=batch,
                    advantages=advantage[rows], targets=target[rows], initial_hidden=hidden[rows])
                assert_finite((metrics, train_state.params, train_state.opt_state), "PPO update")
                if not bool(metrics["grads_finite"]):
                    raise FloatingPointError("nonfinite PPO gradients")
        log = {k: float(v) for k, v in jax.device_get(metrics).items()}
        completions = int(rollout.done.sum())
        log.update(update=index+1, optimizer_step=int(train_state.step),
                   transitions=(index+1)*args.num_envs*args.num_steps,
                   modeled_seconds=float(rollout.duration_s.sum()),
                   completed_episodes=completions, successes=int(rollout.task_done.sum()),
                   success_rate=float(rollout.task_done.sum()/completions) if completions else None,
                   max_mass_residual=int(residual[0]), target_mutation=bool(residual[1]),
                   obstacle_mutation=bool(residual[2]),
                   seconds=time.monotonic()-start, finite=True)
        print(json.dumps(log, sort_keys=True), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(log, sort_keys=True) + "\n")
        if (index+1) % args.checkpoint_interval == 0 or index+1 == args.updates:
            save_checkpoint(args.output / "checkpoint.pkl", train_state, runner, model_config, training, index+1)
            print(f"Saved {args.output / 'checkpoint.pkl'} next_update={index+1}", flush=True)


if __name__ == "__main__":
    main()
