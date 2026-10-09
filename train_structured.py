"""Solo structured-action PPO (terra_structured_v1) on native Terra states.

Episodes start from a saved initial-state pickle (``--initial-states``) or from
a training map level under DATASET_PATH (``--maps-path``). In map mode the
saved ``--env-template`` supplies the native rules and a fixed fraction of the
lanes on every device runs precision episodes on qualified slots. Rollout, GAE
and PPO run in one pmap over ``--num-devices``.

This entrypoint deliberately has its own action/reward/checkpoint protocol.
Legacy train_mixed.py and legacy checkpoints retain their original contract.
See docs/STRUCTURED_ACTIONS.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from functools import partial
import hashlib
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

from terra.config import BatchConfig, CurriculumGlobalConfig, MapsDimsConfig, RewardsType
from terra.env import TerraEnv
from terra.map import GridWorld
from terra.state import static_rules
from terra.structured_actions import (
    StructuredClock, StructuredTimeConfig,
    structured_action_masks, structured_transition, structured_termination,
)
from terra.wrappers import LocalMapWrapper, TraversabilityMaskWrapper
from utils.models import get_model_ready, validate_model_params_match
from utils.structured_ppo import (
    ACTION_PROTOCOL, Runner, StructuredRollout, duration_gae, forward_step, policy_masks,
    ppo_update, sample_action, warm_start_params,
)
from utils.utils_ppo import obs_to_model_input


CHECKPOINT_FORMAT = "terra_structured_ppo_v1"
AXIS = "devices"
# Observation caches that observe() recomputes from the current state; the
# transition dynamics never read them.
LOCAL_MAP_FIELDS = tuple(name for name in GridWorld._fields if name.startswith("local_map_"))


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


def rule_switches(static_cfg):
    """Compile in exactly the optional rules the bank enables (terra.state.static_rules)."""
    return dict(pull_cone=float(static_cfg.pull_half_angle_rad) > 0,
                tracked_move_keeps_turn=bool(static_cfg.tracked_move_keeps_turn))


class MapStarts(NamedTuple):
    """One training level in TerraEnvBatch order; ``precision`` holds row indices."""
    target: jax.Array
    padding: jax.Array
    trench_axes: jax.Array
    trench_types: jax.Array
    border_axes: jax.Array
    border_types: jax.Array
    dumpability: jax.Array
    action: jax.Array
    distance: jax.Array
    precision: jax.Array


def load_map_starts(maps_path, static_cfg, model_config, *, distance_protocol_id, precision_slots):
    """Load a level through TerraEnvBatch: distance sidecar and pull boundary geometry."""
    from terra.env import TerraEnvBatch

    class Level(CurriculumGlobalConfig):
        levels = [dict(maps_path=str(maps_path), max_steps_in_episode=int(static_cfg.max_steps_in_episode),
                       rewards_type=RewardsType.DENSE, apply_trench_rewards=False)]
        last_level_type = "none"

    env = TerraEnvBatch(
        batch_cfg=BatchConfig(curriculum_global=Level()),
        distance_protocol_id=distance_protocol_id,
        executable_dig_observation=bool(model_config.get("executable_dig_observation", False)),
        native_dump_observation=bool(model_config.get("native_dump_observation", False)),
        pull_direction_alignment=bool(static_cfg.pull_direction_alignment),
        **rule_switches(static_cfg),
    )
    buffer = env.maps_buffer
    if buffer.maps.shape[0] != 1:
        raise ValueError("structured map training supports exactly one level")
    slots = np.asarray(buffer.slot_indices[0])
    precision = np.flatnonzero(np.isin(slots, precision_slots or []))
    if len(precision) != len(set(precision_slots or [])):
        raise ValueError("precision slots absent from the loaded training level")
    starts = MapStarts(*(np.asarray(x[0]) for x in (
        buffer.maps, buffer.padding_mask, buffer.trench_axes, buffer.trench_types,
        buffer.foundation_border_axes, buffer.foundation_border_types,
        buffer.dumpability_masks_init, buffer.action_maps, buffer.distance_maps)),
        precision.astype(np.int32))
    edge = env.batch_cfg.maps_dims.maps_edge_length
    if int(static_cfg.maps.edge_length_px) != edge or not np.isclose(
            float(static_cfg.tile_size), env.batch_cfg.maps.edge_length_m / edge):
        raise ValueError("env template and training maps disagree on the map scale")
    return env, starts


def _cast_like(candidate, template):
    return jax.tree.map(lambda value, old: jnp.asarray(value, dtype=old.dtype), candidate, template)


def _lanes(mask, value):
    return mask.reshape(mask.shape + (1,) * (value.ndim - 1))


def sample_saved_start(bank, key, state):
    index = jax.random.randint(key, (), 0, bank.env_steps.shape[0])
    return _cast_like(jax.tree.map(lambda x: x[index], bank), state)


def map_start_sampler(static_cfg):
    """Fresh episode on a sampled map; the lane keeps its precision mode."""
    def sample(starts, key, state):
        pick, qualified, place = jax.random.split(key, 3)
        precision = state.env_cfg.enforce_foundation_border_alignment
        index = jax.random.randint(pick, (), 0, starts.target.shape[0])
        if starts.precision.shape[0]:
            chosen = starts.precision[jax.random.randint(qualified, (), 0, starts.precision.shape[0])]
            index = jnp.where(precision, chosen, index)
        cfg = static_cfg._replace(enforce_foundation_border_alignment=precision)
        fresh = state._replace(key=place)._reset(
            cfg, starts.target[index], starts.padding[index], starts.trench_axes[index],
            starts.trench_types[index], starts.border_axes[index], starts.border_types[index],
            starts.dumpability[index], starts.action[index], distance_map_override=starts.distance[index])
        fresh = TraversabilityMaskWrapper.wrap(fresh)
        # Keep the lane's local-map buffers: a fresh world has dummy shapes.
        fresh = fresh._replace(env_cfg=state.env_cfg, world=fresh.world._replace(
            **{name: getattr(state.world, name) for name in LOCAL_MAP_FIELDS}))
        return _cast_like(fresh, state)
    return sample


def environment_kernels(static_cfg, model_config, *, time_budget_s, decision_limit, timing, allow_wait,
                        time_budget_factor=0., time_budget_offset_s=0., decisions_per_dig_unit=0.):
    """Per-lane observation, transition and episode limits, traced under ``static_rules``.

    Every lane runs the bank's static rules with its own precision-mode flag.
    With ``time_budget_factor`` > 0 an episode's time budget is
    ``time_budget_offset_s`` plus that multiple of its map's dig-only time (dig
    units * tile**3 * dig_s_per_m3), at least ``time_budget_s``; with
    ``decisions_per_dig_unit`` > 0 its decision limit is
    that many decisions per dig unit, at least ``decision_limit``.
    """
    unit_s = float(static_cfg.tile_size) ** 3 * timing.dig_s_per_m3

    def limits(state):
        units = jnp.sum(jnp.maximum(-state.world.target_map.map.astype(jnp.float32), 0.))
        budget = (jnp.maximum(jnp.float32(time_budget_s),
                              time_budget_offset_s + time_budget_factor * unit_s * units)
                  if time_budget_factor else jnp.float32(time_budget_s))
        decisions = (jnp.maximum(decision_limit, jnp.ceil(decisions_per_dig_unit * units).astype(jnp.int32))
                     if decisions_per_dig_unit else jnp.int32(decision_limit))
        return budget, decisions

    def native(state):
        return state._replace(env_cfg=static_cfg._replace(
            enforce_foundation_border_alignment=state.env_cfg.enforce_foundation_border_alignment))

    def observe(state):
        state = native(state)
        state = LocalMapWrapper.wrap(
            state,
            executable_dig_observation=bool(model_config.get("executable_dig_observation", False)),
            native_dump_observation=bool(model_config.get("native_dump_observation", False)),
        )
        observation = TerraEnv._state_to_obs_dict(state)
        if model_config.get("movement_feasibility_observation", False):
            observation["movement_feasibility"] = state._movement_feasibility_tracked().astype(jnp.float32)
        return observation, policy_masks(structured_action_masks(state), allow_wait=allow_wait), limits(state)

    def advance(state, action, clock, elapsed):
        lane_budget, lane_decisions = limits(state)
        result = structured_transition(native(state), action, clock=clock,
                                       timing=timing, time_budget_s=lane_budget)
        next_state = _cast_like(result.state._replace(env_cfg=state.env_cfg), state)
        elapsed = elapsed + result.duration_s
        terminal = structured_termination(next_state, elapsed, time_budget_s=lane_budget,
                                           decision_limit=lane_decisions)
        completion = next_state._get_task_completion(
            next_state.world.action_map.map, next_state.world.target_map.map)["absolute_completion"]
        return (next_state, result.reward + terminal.reward, result.duration_s,
                elapsed, StructuredClock(result.info["time_visit_open"], result.info["time_moved"]),
                terminal.done, terminal.task_done, completion, lane_budget,
                jnp.stack((result.info["action_had_effect"], result.info["material_or_load_changed"])),
                jnp.stack((result.info["transition_mass_residual"],
                           result.info["target_mutation"].astype(jnp.int32),
                           result.info["obstacle_mutation"].astype(jnp.int32))))

    return observe, advance


def policy_observation(raw, runner, time_budget_s, decision_limit):
    """Budgets and decision limits are scalars or per-lane arrays."""
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


def initial_runner(states, config, rng):
    """Fresh clocks/history for lanes [..., lanes] holding the given states."""
    shape = np.shape(states.env_steps)
    hidden = config["actor_gru_hidden_dim"] if config["actor_core"] == "gru" else 0
    return Runner(states, np.zeros(shape, np.float32),
                  StructuredClock(np.zeros(shape, bool), np.zeros(shape, bool)),
                  np.zeros(shape + (config["num_prev_actions"],), np.int32),
                  np.zeros(shape + (config["num_prev_actions"], 3), np.float32),
                  np.zeros(shape + (2,), np.float32),
                  np.zeros(shape + (hidden,), np.float32), rng)


def make_rollout(model, model_config, observe, advance, sample, *, num_steps):
    def rollout(params, runner, data):
        initial_hidden = runner.hidden

        def step(runner, _):
            raw, masks, (budget, decision_limit) = jax.vmap(observe)(runner.state)
            obs = policy_observation(raw, runner, budget, decision_limit)
            rng, act_key, reset_key = jax.random.split(runner.rng, 3)
            value, logits, next_hidden = forward_step(
                model, params, obs, runner.previous_types, runner.hidden, model_config)
            action, log_prob = sample_action(logits, masks, act_key)
            (state, reward, duration, elapsed, clock, done, task_done, completion, budget,
             outcome, integrity) = jax.vmap(advance)(runner.state, action, runner.clock, runner.elapsed_s)
            precision = runner.state.env_cfg.enforce_foundation_border_alignment.astype(bool)
            decisions = state.env_steps
            keys = jax.random.split(reset_key, done.shape[0])
            fresh = jax.lax.cond(jnp.any(done), lambda: jax.vmap(partial(sample, data))(keys, state),
                                 lambda: state)
            state = jax.tree.map(lambda new, old: jnp.where(_lanes(done, old), new, old), fresh, state)

            def clear(value):
                return jnp.where(_lanes(done, value), jnp.zeros_like(value), value)
            previous_types = jnp.roll(runner.previous_types, 1, -1).at[:, 0].set(action.action)
            # Canonical values for inactive args avoid conditioning on sampled junk.
            arguments = jnp.stack((action.amount.astype(jnp.float32) / 6.,
                                   (action.heading.astype(jnp.float32) + 1.) / 12.,
                                   duration / budget), -1)
            previous_arguments = jnp.roll(runner.previous_arguments, 1, 1).at[:, 0].set(arguments)
            next_runner = Runner(state, clear(elapsed),
                                 StructuredClock(clear(clock.visit_open), clear(clock.moved)),
                                 clear(previous_types), clear(previous_arguments),
                                 clear(outcome.astype(jnp.float32)), clear(next_hidden), rng)
            transition = StructuredRollout(obs, masks, runner.previous_types, action, log_prob,
                                           value, reward, duration, done, task_done)
            kind = action.action
            stats = dict(
                episodes_bulk=jnp.sum(done & ~precision), episodes_precision=jnp.sum(done & precision),
                success_bulk=jnp.sum(task_done & ~precision),
                success_precision=jnp.sum(task_done & precision),
                episode_seconds=jnp.sum(jnp.where(done, elapsed, 0.)),
                episode_budget_seconds=jnp.sum(jnp.where(done, budget, 0.)),
                episode_decisions=jnp.sum(jnp.where(done, decisions, 0)),
                episode_completion=jnp.sum(jnp.where(done, completion, 0.)),
                action_types=jax.nn.one_hot(kind, 8, dtype=jnp.int32).sum(0),
                do_effective=jnp.sum((kind == 6) & outcome[:, 1]),
                move_cells=jnp.sum(jnp.where(kind < 2, action.amount, 0)),
                turn_steps=jnp.sum(jnp.where((kind >= 2) & (kind < 4), action.amount, 0)),
                modeled_seconds=jnp.sum(duration), reward=jnp.sum(reward),
                integrity=integrity.max(0),
            )
            return next_runner, (transition, stats)

        runner, (transitions, stats) = jax.lax.scan(step, runner, None, length=num_steps)
        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        stats = {key: value.max(0) if key == "integrity" else value.sum(0) for key, value in stats.items()}
        return runner, transitions, initial_hidden, stats
    return rollout


def make_iteration(model, model_config, observe, rollout, args):
    """One PPO update on one device's lanes; gradients and statistics reduce over AXIS."""
    update = partial(ppo_update, model=model, config=model_config,
                     clip_eps=args.clip_eps, vf_coef=args.vf_coef,
                     entropy_coefs=(args.entropy_type, args.entropy_move, args.entropy_turn,
                                    args.entropy_heading),
                     value_clip=args.value_clip, axis_name=AXIS)

    def iteration(train_state, runner, data):
        runner, transitions, initial_hidden, stats = rollout(train_state.params, runner, data)
        raw, _, (budget, decision_limit) = jax.vmap(observe)(runner.state)
        obs = policy_observation(raw, runner, budget, decision_limit)
        last_value = model.apply(train_state.params,
                                 obs_to_model_input(obs, runner.previous_types, model_config),
                                 method="value")[..., 0]
        advantages, targets = duration_gae(transitions, last_value, args.gamma, args.gae_lambda,
                                           args.discount_reference_s)
        rng, epoch_key = jax.random.split(runner.rng)
        runner = runner._replace(rng=rng)
        lanes = transitions.done.shape[0]

        def epoch(train_state, key):
            order = jax.random.permutation(key, lanes).reshape((args.minibatches, -1))

            def minibatch(train_state, rows):
                return update(train_state, rollout=jax.tree.map(lambda x: x[rows], transitions),
                              advantages=advantages[rows], targets=targets[rows],
                              initial_hidden=initial_hidden[rows])
            return jax.lax.scan(minibatch, train_state, order)

        train_state, metrics = jax.lax.scan(epoch, train_state, jax.random.split(epoch_key, args.epochs))
        metrics = {key: value.all() if key == "grads_finite" else value.mean()
                   for key, value in metrics.items()}
        residual = targets - transitions.value
        metrics["explained_variance"] = 1. - (jax.lax.pmean(jnp.var(residual), AXIS)
                                              / (jax.lax.pmean(jnp.var(targets), AXIS) + 1e-8))
        stats = {key: jax.lax.pmax(value, AXIS) if key == "integrity" else jax.lax.psum(value, AXIS)
                 for key, value in stats.items()}
        stats["returns_finite"] = jax.lax.pmin(
            (jnp.isfinite(advantages).all() & jnp.isfinite(targets).all()).astype(jnp.int32), AXIS)
        return train_state, runner, {**metrics, **stats}
    return iteration


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
    """``train_state`` is one device's copy; ``runner`` keeps its [device, lane] layout."""
    payload = dict(format=CHECKPOINT_FORMAT, action_protocol=ACTION_PROTOCOL,
                   model=jax.device_get(train_state.params),
                   optimizer_state=jax.device_get(train_state.opt_state),
                   train_state_step=int(train_state.step),
                   runner=None if runner is None else jax.device_get(runner),
                   model_config=dict(model_config), training_config=training_config,
                   next_update=next_update)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    os.replace(temporary, path)


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return dict(size=Path(path).stat().st_size, sha256=digest.hexdigest())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-states", type=Path, help="Saved initial-state bank (all starts)")
    source.add_argument("--maps-path", help="Training level under DATASET_PATH (map mode)")
    parser.add_argument("--bank-indices", help="Optional comma-separated bank indices for a bounded diagnostic")
    parser.add_argument("--env-template", type=Path, help="Map mode: saved state bank whose native rules every lane uses")
    parser.add_argument("--distance-protocol-id", help="Map mode: required physical distance sidecar protocol")
    parser.add_argument("--training-slots", type=Path, help="Map mode: JSON with 0-based precision_slots (teacher_slots ignored)")
    parser.add_argument("--precision-episode-fraction", type=float, default=0.,
                        help="Map mode: fraction of lanes per device running precision episodes")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=1, help="Absolute completed PPO update target")
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8, help="Lanes per device")
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatches", type=int, default=1, help="Whole-trajectory minibatches per device")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--time-budget-s", type=float, default=14400.,
                        help="Episode time budget; with --time-budget-factor the minimum budget")
    parser.add_argument("--time-budget-factor", type=float, default=0.,
                        help="Budget = offset + this multiple of the map's dig-only modeled time (0 = fixed budget)")
    parser.add_argument("--time-budget-offset-s", type=float, default=0.,
                        help="Constant part of a map-scaled budget (setups and travel)")
    parser.add_argument("--decision-limit", type=int, default=450,
                        help="Episode decision limit; with --decisions-per-dig-unit the minimum limit")
    parser.add_argument("--decisions-per-dig-unit", type=float, default=0.,
                        help="Decision limit = this many decisions per map dig unit (0 = fixed limit)")
    parser.add_argument("--gamma", type=float, default=1.0, help="1.0 preserves material-potential telescoping; discounting is an explicit experiment")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--discount-reference-s", type=float, default=30.)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=2.)
    parser.add_argument("--value-clip", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Rolling checkpoint.pkl, with live environments")
    parser.add_argument("--keep-checkpoint-every", type=int, default=0, help="Also keep checkpoint_update_NNNNNN.pkl (no environments); 0 = never")
    parser.add_argument("--finite-check-interval", type=int, default=1, help="Full parameter/optimizer finiteness check cadence")
    parser.add_argument("--wandb-project", help="Log to W&B unless WANDB_MODE=disabled")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id", help="Continue this W&B run (resume=allow)")
    args = parser.parse_args(argv)
    if args.teacher_checkpoint:
        parser.error("teacher KL is unsupported for structured-v1: cabin/DO probabilities need an explicit mapping")
    if args.resume_from and args.warm_start_from:
        parser.error("choose resume or warm start")
    for name in ("updates", "num_devices", "num_envs", "num_steps", "epochs", "minibatches",
                 "decision_limit", "checkpoint_interval", "finite_check_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >=1")
    if args.num_envs % args.minibatches:
        parser.error("num-envs must divide evenly into whole-trajectory minibatches")
    if not (0 < args.gamma <= 1 and 0 < args.gae_lambda <= 1):
        parser.error("gamma and gae-lambda must be in (0,1]")
    for name in ("time_budget_s", "discount_reference_s", "learning_rate", "clip_eps"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    for name in ("vf_coef", "entropy_type", "entropy_move", "entropy_turn", "entropy_heading",
                 "time_budget_factor", "time_budget_offset_s", "decisions_per_dig_unit"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative and finite")
    if args.maps_path is None:
        for name in ("env_template", "distance_protocol_id", "training_slots"):
            if getattr(args, name) is not None:
                parser.error(f"--{name.replace('_', '-')} applies to --maps-path only")
        if args.precision_episode_fraction:
            parser.error("saved banks carry their own precision modes")
    else:
        if args.env_template is None or args.distance_protocol_id is None:
            parser.error("--maps-path requires --env-template and --distance-protocol-id")
        if args.bank_indices:
            parser.error("--bank-indices applies to --initial-states only")
        lanes = args.num_envs * args.precision_episode_fraction
        if not (0 <= args.precision_episode_fraction <= 1 and np.isclose(lanes, round(lanes))):
            parser.error("precision-episode-fraction must give a whole number of lanes per device")
        if args.precision_episode_fraction > 0 and args.training_slots is None:
            parser.error("precision lanes require --training-slots")
    return args


def read_precision_slots(path):
    recipe = json.loads(Path(path).read_text())
    slots = recipe.get("precision_slots")
    if (not isinstance(slots, list) or not slots or any(type(x) is not int or x < 0 for x in slots)
            or len(slots) != len(set(slots))):
        raise ValueError("precision_slots must be distinct nonnegative 0-based manifest slots")
    return sorted(slots)


def episode_log(stats):
    """Host-side means over episodes and decisions completed in this update."""
    episodes_bulk, episodes_precision = int(stats["episodes_bulk"]), int(stats["episodes_precision"])
    episodes = episodes_bulk + episodes_precision
    types = np.asarray(stats["action_types"], np.float64)
    decisions = types.sum()
    moves, turns = types[0] + types[1], types[2] + types[3]
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    return {
        "episode/count": episodes,
        "episode/bulk_count": episodes_bulk,
        "episode/precision_count": episodes_precision,
        "episode/bulk_successes": int(stats["success_bulk"]),
        "episode/precision_successes": int(stats["success_precision"]),
        "episode/bulk_success_rate": ratio(stats["success_bulk"], episodes_bulk),
        "episode/precision_success_rate": ratio(stats["success_precision"], episodes_precision),
        "episode/modeled_hours": ratio(stats["episode_seconds"] / 3600., episodes),
        "episode/budget_hours": ratio(stats["episode_budget_seconds"] / 3600., episodes),
        "episode/decisions": ratio(stats["episode_decisions"], episodes),
        "episode/completion": ratio(stats["episode_completion"], episodes),
        "action/move_fraction": ratio(moves, decisions),
        "action/turn_fraction": ratio(turns, decisions),
        "action/do_fraction": ratio(types[6], decisions),
        "action/wait_fraction": ratio(types[7], decisions),
        "action/do_effective_rate": ratio(stats["do_effective"], types[6]),
        "action/mean_move_cells": ratio(stats["move_cells"], moves),
        "action/mean_turn_steps": ratio(stats["turn_steps"], turns),
        "action/modeled_seconds_per_decision": ratio(stats["modeled_seconds"], decisions),
        "reward/per_decision": ratio(stats["reward"], decisions),
    }


def main(argv=None):
    args = parse_args(argv)
    training = vars(args).copy()
    training = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in training.items()}
    for key in ("wandb_project", "wandb_entity", "wandb_name", "wandb_id"):
        training.pop(key)
    timing_values = json.loads(args.timing_json.read_text()) if args.timing_json else {}
    timing = StructuredTimeConfig(**timing_values)
    if any(not np.isfinite(x) or x < 0 for x in timing) or timing.nav_speed_mps <= 0:
        raise ValueError("timing values must be finite and nonnegative; navigation speed must be positive")
    training["timing"] = timing._asdict()
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

    devices = jax.devices()[:args.num_devices]
    if len(devices) != args.num_devices:
        raise RuntimeError(f"requested {args.num_devices} devices; JAX sees {len(jax.devices())}")
    lanes = args.num_envs
    if args.maps_path is None:
        bank, static_cfg = load_initial_bank(args.initial_states,
            [int(x) for x in args.bank_indices.split(",")] if args.bank_indices else None)
        training["initial_bank"] = file_identity(args.initial_states)
        data = bank
        sample = sample_saved_start
        env_shape = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=bank.world.target_map.map.shape[-1])),
            executable_dig_observation=bool(model_config.get("executable_dig_observation", False)))
    else:
        template_bank, static_cfg = load_initial_bank(args.env_template)
        template = jax.tree.map(lambda x: np.asarray(x[0]), template_bank)
        precision_slots = read_precision_slots(args.training_slots) if args.training_slots else []
        training["env_template"] = file_identity(args.env_template)
        training["precision_slots"] = precision_slots
        training["dataset_path"] = os.environ.get("DATASET_PATH")
        training["dataset_size"] = os.environ.get("DATASET_SIZE")
        env_shape, data = load_map_starts(
            args.maps_path, static_cfg, model_config, distance_protocol_id=args.distance_protocol_id,
            precision_slots=precision_slots)
        for name, expected, actual in (
                ("trench axes", template.world.trench_axes.shape, data.trench_axes.shape[1:]),
                ("boundary records", template.world.foundation_border_axes.shape, data.border_axes.shape[1:])):
            if tuple(expected) != tuple(actual):
                raise ValueError(f"env template and training maps disagree on {name}: {expected} vs {actual}")
        sample = map_start_sampler(static_cfg)
        print(json.dumps(dict(phase="map_bank", maps=int(data.target.shape[0]),
                              precision_maps=int(data.precision.shape[0]))), flush=True)
    if bool(model_config.get("native_dump_observation", False)) != bool(static_cfg.native_dump_observation):
        raise ValueError("model native_dump_observation must match the bank's EnvConfig")

    if args.resume_from:
        if source.get("format") != CHECKPOINT_FORMAT or source.get("action_protocol") != ACTION_PROTOCOL:
            raise ValueError("native resume requires a structured-v1 checkpoint; use warm-start-from for legacy policies")
        for key, value in source["training_config"].items():
            if (key not in {"output", "updates", "checkpoint_interval", "keep_checkpoint_every",
                            "finite_check_interval", "resume_from", "warm_start_from"}
                    and training.get(key) != value):
                raise ValueError(f"native resume protocol mismatch: {key}")
        if dict(model_config) != source["model_config"]:
            raise ValueError("native resume must preserve model configuration")

    rng, model_key, runner_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    model, params = get_model_ready(model_key, model_config, env_shape)
    if args.warm_start_from:
        params = warm_start_params(params, source["model"])
        print("Warm start: reused backbone/GRU/type head; reset critic, argument heads, optimizer and clocks", flush=True)
    train_state = TrainState.create(apply_fn=model.apply, params=params,
        tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.learning_rate, eps=1e-5)))
    start_update = 0
    if args.resume_from:
        validate_model_params_match(params, source["model"], "structured resume")
        train_state = train_state.replace(params=jax.tree.map(jnp.asarray, source["model"]),
            opt_state=jax.tree.map(jnp.asarray, source["optimizer_state"]),
            step=jnp.asarray(source["train_state_step"]))
        start_update = int(source["next_update"])
    assert_finite((train_state.params, train_state.opt_state), "initial/restored parameters")

    # Lane layout [device, lane]. Map mode starts every lane from the template
    # with its fixed precision flag and immediately samples a fresh map.
    if args.maps_path is None:
        host_bank = jax.device_get(data)
        ids = np.asarray(jax.random.randint(runner_key, (args.num_devices, lanes), 0, host_bank.env_steps.shape[0]))
        states = jax.tree.map(lambda x: x[ids], host_bank)
    else:
        count = int(round(lanes * args.precision_episode_fraction))
        flags = np.broadcast_to(np.arange(lanes) < count, (args.num_devices, lanes))
        states = jax.tree.map(lambda x: np.broadcast_to(x, (args.num_devices, lanes) + np.shape(x)).copy(), template)
        states = states._replace(env_cfg=states.env_cfg._replace(
            enforce_foundation_border_alignment=flags.astype(np.asarray(
                template.env_cfg.enforce_foundation_border_alignment).dtype)))
    runner = initial_runner(states, model_config, np.asarray(jax.random.split(runner_key, args.num_devices)))
    if args.resume_from and source.get("runner") is not None:
        runner = restore_runner(source["runner"], runner)

    def shard(tree):
        return jax.device_put_sharded([jax.tree.map(lambda x: np.asarray(x)[i], tree)
                                       for i in range(args.num_devices)], devices)
    runner = shard(jax.device_get(runner))
    train_state = jax.device_put_replicated(train_state, devices)
    data = jax.device_put_replicated(jax.device_get(data), devices)

    observe, advance = environment_kernels(static_cfg, model_config,
        time_budget_s=args.time_budget_s, decision_limit=args.decision_limit,
        timing=timing, allow_wait=args.allow_wait, time_budget_factor=args.time_budget_factor,
        time_budget_offset_s=args.time_budget_offset_s, decisions_per_dig_unit=args.decisions_per_dig_unit)
    rollout = make_rollout(model, model_config, observe, advance, sample, num_steps=args.num_steps)
    iteration = jax.pmap(make_iteration(model, model_config, observe, rollout, args),
                         axis_name=AXIS, donate_argnums=(0, 1))
    switches = rule_switches(static_cfg)
    if args.maps_path is not None and not (args.resume_from and source.get("runner") is not None):
        def reset_all(runner, data):
            rng, key = jax.random.split(runner.rng)
            states = jax.vmap(partial(sample, data))(jax.random.split(key, lanes), runner.state)
            return runner._replace(state=states, rng=rng)
        with static_rules(**switches):
            runner = jax.pmap(reset_all, donate_argnums=(0,))(runner, data)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(
        dict(training=training, model=dict(model_config), action_protocol=ACTION_PROTOCOL,
             rules=rule_switches(static_cfg)), indent=2, default=str) + "\n")
    wandb_run = None
    if args.wandb_project and os.environ.get("WANDB_MODE", "online") != "disabled":
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                               id=args.wandb_id, resume="allow" if args.wandb_id else None,
                               config=dict(training=training, model=dict(model_config),
                                           action_protocol=ACTION_PROTOCOL))
    print(json.dumps(dict(phase="native_rollout", devices=[str(d) for d in devices],
                         lanes_per_device=lanes, start_update=start_update,
                         wandb=None if wandb_run is None else wandb_run.id,
                         reward_protocol="material_time_v1", action_protocol=ACTION_PROTOCOL)), flush=True)

    decisions_per_update = args.num_devices * lanes * args.num_steps
    for index in range(start_update, args.updates):
        start = time.monotonic()
        with static_rules(**switches):
            train_state, runner, metrics = iteration(train_state, runner, data)
        metrics = jax.tree.map(lambda x: np.asarray(x)[0], jax.device_get(metrics))
        seconds = time.monotonic() - start
        integrity = metrics.pop("integrity")
        if np.any(integrity != 0):
            raise RuntimeError(f"native transition integrity failed (mass,target,obstacle): {integrity}")
        if not (bool(metrics.pop("grads_finite")) and bool(metrics.pop("returns_finite"))):
            raise FloatingPointError("nonfinite PPO gradients or returns")
        losses = {key: float(metrics[key]) for key in (
            "total_loss", "actor_loss", "value_loss", "entropy", "entropy_action", "entropy_move",
            "entropy_turn", "entropy_do", "approx_kl", "clip_fraction", "explained_variance")}
        if not all(np.isfinite(value) for value in losses.values()):
            raise FloatingPointError(f"nonfinite PPO metrics: {losses}")
        update_number = index + 1
        if update_number % args.finite_check_interval == 0:
            assert_finite((train_state.params, train_state.opt_state), "PPO parameters")
        log = {"train/update": update_number,
               "train/decisions": update_number * decisions_per_update,
               "train/optimizer_step": int(np.asarray(jax.device_get(train_state.step))[0]),
               **{f"loss/{key}": value for key, value in losses.items()},
               **episode_log(metrics),
               "perf/update_seconds": seconds,
               "perf/decisions_per_second": decisions_per_update / seconds}
        print(json.dumps(log, sort_keys=True), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(log, sort_keys=True) + "\n")
        if wandb_run is not None:
            wandb_run.log(log)
        rolling = update_number % args.checkpoint_interval == 0 or update_number == args.updates
        history = args.keep_checkpoint_every and update_number % args.keep_checkpoint_every == 0
        if rolling or history:
            assert_finite((train_state.params, train_state.opt_state), "checkpoint parameters")
            host_state = jax.tree.map(lambda x: x[0], jax.device_get(train_state))
            if history:
                save_checkpoint(args.output / f"checkpoint_update_{update_number:06d}.pkl", host_state,
                                None, model_config, training, update_number)
            if rolling:
                save_checkpoint(args.output / "checkpoint.pkl", host_state, runner,
                                model_config, training, update_number)
                print(f"Saved {args.output / 'checkpoint.pkl'} next_update={update_number}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
