#!/usr/bin/env python3
"""Benchmark native-parent versus time/actor growth on a repeated failure state.

Run on an otherwise idle GPU under an external timeout. Compilation and three
warmups are excluded from the synchronized medians. These are model kernels,
not full PPO/environment throughput or evidence of improved policy quality.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args):
    import jax
    import jax.numpy as jnp
    from terra.actions import TrackedAction
    from terra.config import BatchConfig, ImmutableAgentConfig, MapsDimsConfig
    from terra.env import TerraEnv
    from utils.actor_capacity import migrate_actor_capacity_checkpoint, without_actor_capacity
    from utils.helpers import (
        checkpoint_evaluation_config, load_pkl_object, register_checkpoint_config_classes,
    )
    from utils.models import get_model_ready, validate_model_params_match
    from utils.remaining_time import migrate_remaining_time_checkpoint
    from utils.task_teachers import (
        CACHED_LOGITS_KEY, CACHED_VALUE_KEY, FAMILY_KEY, LEGACY_DIG_KEY,
        cache_task_teacher_outputs, legacy_teacher_admissible_dig,
        load_task_teacher_checkpoint, make_task_teacher_apply_fn, option,
    )
    from utils.utils_ppo import obs_to_model_input

    register_checkpoint_config_classes()
    device = jax.devices()[0]
    if device.platform != 'gpu':
        raise RuntimeError('This benchmark requires a GPU; select JAX_PLATFORMS=cuda,cpu')
    checkpoint = load_pkl_object(str(args.checkpoint))
    old_config = checkpoint_evaluation_config(checkpoint)
    if (option(old_config, 'time_observation_mode', 'none') != 'none'
            or option(old_config, 'actor_residual_head', False)):
        raise ValueError('Use the native parent before either time or actor migration')
    stored = load_pkl_object(str(args.failure_state))
    state = stored['snapshot'].state
    previous_actions = jnp.asarray(stored['previous_actions'])
    if int(state.agent.num_agents) != 1 or int(state._get_current_agent_state().action_type[0]) != 0:
        raise ValueError('Use a saved single tracked-excavator failure state')
    edge = state.world.action_map.map.shape[-1]
    batch_cfg = BatchConfig(
        maps_dims=MapsDimsConfig(maps_edge_length=edge),
        maps=state.env_cfg.maps,
        agent=ImmutableAgentConfig(
            angles_base=int(state.env_cfg.agent.angles_base),
            angles_cabin=int(state.env_cfg.agent.angles_cabin)),
        action_type=TrackedAction,
    )
    env = SimpleNamespace(batch_cfg=batch_cfg,
                          executable_dig_observation=bool(option(old_config, 'executable_dig_observation', False)))
    old_model, old_initialized = get_model_ready(jax.random.PRNGKey(0), old_config, env)
    validate_model_params_match(old_initialized, checkpoint['model'], 'benchmark native parent')
    new_config = copy.copy(old_config)
    if isinstance(new_config, dict):
        new_config.update(time_observation_mode='remaining', actor_residual_head=True)
    else:
        new_config.time_observation_mode = 'remaining'
        new_config.actor_residual_head = True
    new_model, new_initialized = get_model_ready(jax.random.PRNGKey(1), new_config, env)
    timed = migrate_remaining_time_checkpoint(checkpoint, without_actor_capacity(new_initialized), 'remaining')
    grown = migrate_actor_capacity_checkpoint(timed, new_initialized)

    @jax.jit
    def observe(candidate):
        wrapped = TerraEnv.wrap_state(candidate, update_reachability=False,
                                      executable_dig_observation=env.executable_dig_observation)
        return TerraEnv._state_to_obs_dict(wrapped)

    raw = observe(state)
    repeat = lambda x: jnp.broadcast_to(x, (args.batchsize,) + x.shape)
    raw_batch = jax.tree.map(repeat, raw)
    history = repeat(previous_actions)
    old_inputs = obs_to_model_input(raw_batch, history, old_config)
    new_inputs = obs_to_model_input(raw_batch, history, new_config)
    old_params, new_params, old_inputs, new_inputs = jax.device_put(
        (checkpoint['model'], grown['model'], old_inputs, new_inputs), device)

    compiled = {}
    report = {
        'status': 'INCOMPLETE',
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_update': int(checkpoint['next_update']),
        'failure_state': str(args.failure_state.resolve()),
        'snapshot_age': int(state.env_steps),
        'batchsize': args.batchsize, 'repeats': args.repeats, 'warmups': 3,
        'device': str(device), 'device_kind': device.device_kind,
        'jax_version': jax.__version__,
        'parameter_counts': {
            'parent': sum(int(x.size) for x in jax.tree.leaves(old_params)),
            'combined': sum(int(x.size) for x in jax.tree.leaves(new_params)),
        },
        'timings': {},
        'limitations': [
            'One physical state and action history repeated across the batch.',
            'Forward and PPO-like value-and-gradient kernels exclude optimizer, environment and reset work.',
            'Compilation and warmup are excluded; each timed output is synchronized.',
            'These timings do not establish an end-to-end training speedup or policy-quality improvement.',
        ],
    }

    def write_report():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def prepare(name, function, arguments):
        arguments = jax.device_put(arguments, device)
        start = time.perf_counter()
        executable = jax.jit(function).lower(*arguments).compile()
        compilation_seconds = time.perf_counter() - start
        for _ in range(3):
            jax.block_until_ready(executable(*arguments))
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(executable(*arguments))
            durations.append(1000 * (time.perf_counter() - start))
        result = executable(*arguments)
        jax.block_until_ready(result)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(jax.device_get(result))):
            raise FloatingPointError(f'Nonfinite output or gradient in {name}')
        compiled[name] = (executable, arguments)
        report['timings'][name] = {
            'compilation_seconds': compilation_seconds,
            'median_ms': statistics.median(durations),
            'p10_ms': float(np.percentile(durations, 10)),
            'p90_ms': float(np.percentile(durations, 90)),
            'samples_ms': durations,
        }
        write_report()
        print(f'{name}: {statistics.median(durations):.3f} ms warmed median; '
              f'{compilation_seconds:.1f} s compile', flush=True)
        return result

    parent_outputs = prepare('parent_forward', old_model.apply, (old_params, old_inputs))
    combined_outputs = prepare('combined_forward', new_model.apply, (new_params, new_inputs))
    parity = {}
    for name, expected, actual in zip(('value', 'logits'), parent_outputs, combined_outputs):
        expected, actual = np.asarray(expected), np.asarray(actual)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        parity[f'{name}_max_abs_difference'] = float(np.max(np.abs(actual - expected)))
    np.testing.assert_array_equal(np.argmax(np.asarray(parent_outputs[1]), axis=-1),
                                  np.argmax(np.asarray(combined_outputs[1]), axis=-1))
    report['initial_output_parity'] = {**parity, 'atol': 1e-5, 'rtol': 1e-5,
                                      'same_greedy_actions': True}
    values, logits = parent_outputs
    actions = jax.random.categorical(jax.random.PRNGKey(7), logits)
    old_logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1)[:, 0]
    advantages = jnp.linspace(-1, 1, args.batchsize)
    target_values = values[:, 0] + advantages

    def loss_for(model):
        def loss(parameters, inputs, selected, reference_logp, advantage, targets, reference_value):
            values, logits = model.apply(parameters, inputs)
            values = values[:, 0]
            logp = jax.nn.log_softmax(logits, axis=-1)
            selected_logp = jnp.take_along_axis(logp, selected[:, None], axis=-1)[:, 0]
            ratio = jnp.exp(selected_logp - reference_logp)
            actor = -jnp.minimum(advantage * ratio, advantage * jnp.clip(ratio, .8, 1.2)).mean()
            clipped_values = reference_value + jnp.clip(values - reference_value, -.2, .2)
            critic = .5 * jnp.maximum((values - targets)**2, (clipped_values - targets)**2).mean()
            entropy = -(jnp.exp(logp) * logp).sum(axis=-1).mean()
            return actor + .5 * critic - .02 * entropy
        return jax.value_and_grad(loss)

    loss_args = (actions, old_logp, advantages, target_values, values[:, 0])
    prepare('parent_value_and_grad', loss_for(old_model), (old_params, old_inputs, *loss_args))
    prepare('combined_value_and_grad', loss_for(new_model), (new_params, new_inputs, *loss_args))
    report['combined_over_parent'] = {
        name: report['timings'][f'combined_{name}']['median_ms']
        / report['timings'][f'parent_{name}']['median_ms']
        for name in ('forward', 'value_and_grad')
    }

    if args.teachers_dir is not None:
        teacher_models, teacher_params, teacher_configs = {}, {}, {}
        for role in ('foundation', 'trench'):
            path = args.teachers_dir / f'{role}_teacher.pkl'
            identity_field = ('teacher_checkpoint_sha256' if role == 'foundation'
                              else 'trench_teacher_checkpoint_sha256')
            expected_identity = option(old_config, identity_field)
            if expected_identity is None:
                raise ValueError(f'Parent has no recorded {identity_field}')
            saved = load_task_teacher_checkpoint(path, expected_identity)
            teacher_config = checkpoint_evaluation_config(saved)
            teacher_env = SimpleNamespace(batch_cfg=batch_cfg,
                executable_dig_observation=bool(option(teacher_config, 'executable_dig_observation', False)))
            model, initialized = get_model_ready(jax.random.PRNGKey(2), teacher_config, teacher_env)
            validate_model_params_match(initialized, saved['model'], str(path))
            teacher_models[role], teacher_params[role], teacher_configs[role] = model, saved['model'], teacher_config
        teacher_params = jax.device_put(teacher_params, device)
        family_ids = option(old_config, 'task_teacher_family_ids')
        if not isinstance(family_ids, dict) or set(family_ids) != {'foundation', 'trench'}:
            raise ValueError('Parent must record its dual-teacher family routing')
        teacher = make_task_teacher_apply_fn(
            teacher_models['foundation'].apply, teacher_models['trench'].apply,
            teacher_configs['foundation'], teacher_configs['trench'], family_ids)
        teacher_raw = dict(raw_batch)
        # Alternate labels only to exercise both routing branches numerically;
        # this is still one repeated physical state, not a task-balanced cohort.
        teacher_raw[FAMILY_KEY] = jnp.where(jnp.arange(args.batchsize) % 2,
            family_ids['trench'], family_ids['foundation'])
        teacher_raw[LEGACY_DIG_KEY] = repeat(jax.jit(legacy_teacher_admissible_dig)(state))
        prepare('dual_teacher_live_minibatch', teacher, (teacher_params, teacher_raw, history))
        # Two production-sized minibatches test the actual cache path without
        # storing a full 32-step rollout or evaluating it as one huge batch.
        rollout_raw = jax.tree.map(lambda x: jnp.stack((x, x)), teacher_raw)
        rollout_history = jnp.stack((history, history))

        def cache(parameters, raw, previous):
            result = cache_task_teacher_outputs(raw, previous, teacher, parameters, 2)
            return result[CACHED_VALUE_KEY], result[CACHED_LOGITS_KEY]

        cached = prepare('dual_teacher_cache_two_minibatches', cache,
                         (teacher_params, rollout_raw, rollout_history))
        chunks = jax.tree.map(lambda x: x.swapaxes(0, 1).reshape((2, args.batchsize) + x.shape[2:]),
                              (rollout_raw, rollout_history))
        direct = [compiled['dual_teacher_live_minibatch'][0](teacher_params,
                  jax.tree.map(lambda x: x[index], chunks[0]), chunks[1][index]) for index in range(2)]
        live = jax.tree.map(lambda *x: jnp.stack(x).reshape((args.batchsize, 2) + x[0].shape[1:]).swapaxes(0, 1), *direct)
        differences = {}
        for name, expected, actual in zip(('value', 'logits'), live, cached):
            expected, actual = np.asarray(expected), np.asarray(actual)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            differences[f'{name}_max_abs_difference'] = float(np.max(np.abs(actual - expected)))
        report['teacher_cache_parity'] = {
            **differences, 'atol': 1e-5, 'rtol': 1e-5,
            'minibatches': 2, 'samples_per_minibatch': args.batchsize,
            'routing': 'Alternating foundation/trench labels on the repeated physical state',
            'teacher_files': {role: str((args.teachers_dir / f'{role}_teacher.pkl').resolve())
                              for role in ('foundation', 'trench')},
        }
    else:
        report['teacher_cache_parity'] = {'status': 'not run; pass --teachers-dir'}

    if args.geometry:
        def refresh(candidate):
            wrapped = TerraEnv.wrap_state(candidate, update_reachability=False,
                executable_dig_observation=env.executable_dig_observation)
            return TerraEnv._state_to_obs_dict(wrapped)
        prepare('observation_refresh_no_reachability', jax.vmap(refresh),
                (jax.tree.map(lambda x: repeat(jnp.asarray(x)), state),))

    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(args.profile_dir), create_perfetto_trace=True):
            for name, (executable, arguments) in compiled.items():
                with jax.profiler.TraceAnnotation(name):
                    for _ in range(3):
                        jax.block_until_ready(executable(*arguments))
        report['profile_dir'] = str(args.profile_dir.resolve())
    report['status'] = 'PASS'
    write_report()
    print(json.dumps({'parameter_counts': report['parameter_counts'],
                      'combined_over_parent': report['combined_over_parent'],
                      'output': str(args.output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--failure-state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batchsize', type=int, choices=(128, 256), default=128)
    parser.add_argument('--repeats', type=int, choices=range(10, 21), default=15)
    parser.add_argument('--teachers-dir', type=Path)
    parser.add_argument('--geometry', action='store_true', help='Also time the batched observation refresh')
    parser.add_argument('--profile-dir', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    run(args)


if __name__ == '__main__':
    main()
