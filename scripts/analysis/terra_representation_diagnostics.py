#!/usr/bin/env python3
"""Cheap checkpoint sensitivity probes on saved failure states; no training.

Counterfactual cell changes refresh Terra's observation wrappers, but are not
legal soil transitions and do not establish that a policy can finish the map.
They test whether the visible representation and policy distinguish a residual,
pile amount, or remaining episode budget before paying for a larger encoder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def tree_changed_entries(left, right):
    """Count changed input values, not just changed observation channels."""
    import jax

    return int(sum(np.count_nonzero(np.asarray(a) != np.asarray(b))
                   for a, b in zip(jax.tree_util.tree_leaves(left),
                                   jax.tree_util.tree_leaves(right))))


def output_difference(reference, candidate):
    """Report behavioral and feature sensitivity without a pass/fail claim."""
    value, logits, embedding = (np.asarray(x, dtype=np.float64)
                                for x in reference)
    other_value, other_logits, other_embedding = (
        np.asarray(x, dtype=np.float64) for x in candidate)

    def log_softmax(x):
        x = x - np.max(x)
        return x - np.log(np.exp(x).sum())

    log_p, log_q = log_softmax(logits), log_softmax(other_logits)
    p, q = np.exp(log_p), np.exp(log_q)
    return {
        'value_delta': float((other_value - value).reshape(-1)[0]),
        'logit_linf': float(np.max(np.abs(other_logits - logits))),
        'policy_total_variation': float(np.abs(p - q).sum() / 2),
        'policy_kl_reference_to_variant': float(np.sum(p * (log_p - log_q))),
        'greedy_action_changed': bool(np.argmax(logits) != np.argmax(other_logits)),
        'reference_action': int(np.argmax(logits)),
        'variant_action': int(np.argmax(other_logits)),
        'map_embedding_l2': float(np.linalg.norm(other_embedding - embedding)),
        'map_embedding_relative_l2': float(
            np.linalg.norm(other_embedding - embedding)
            / max(np.linalg.norm(embedding), 1e-12)),
    }


def counterfactual_states(state):
    """Return two single-cell residual probes, a remote pile, and age probes."""
    import jax.numpy as jnp

    terrain = np.asarray(state.world.action_map.map)
    target = np.asarray(state.world.target_map.map)
    position = np.asarray(state._get_current_agent_state().pos_base)
    variants = []

    def with_cell(row, col, height):
        action_map = state.world.action_map._replace(
            map=jnp.asarray(terrain).at[row, col].set(height))
        return state._replace(world=state.world._replace(action_map=action_map))

    residual = np.argwhere((target < 0) & (terrain > target))
    if len(residual):
        distances = np.linalg.norm(residual - position, axis=1)
        indices = [('nearest_residual', int(np.argmin(distances)))]
        if len(residual) > 1:
            indices.append(('furthest_residual', int(np.argmax(distances))))
        for name, index in indices:
            row, col = (int(x) for x in residual[index])
            variants.append((name, with_cell(row, col, int(target[row, col])),
                             dict(cell=[row, col], old_height=int(terrain[row, col]),
                                  new_height=int(target[row, col]))))

    piles = np.argwhere(terrain > 0)
    if len(piles):
        # Furthest positive cell is most likely outside every local workspace;
        # input comparison, rather than this distance heuristic, decides aliasing.
        distances = np.linalg.norm(piles - position, axis=1)
        index = int(np.argmax(distances))
        row, col = (int(x) for x in piles[index])
        old_height = int(terrain[row, col])
        variants.append(('pile_height_plus_one', with_cell(row, col, old_height + 1),
                         dict(cell=[row, col], old_height=old_height,
                              new_height=old_height + 1,
                              distance_from_base_cells=float(distances[index]))))

    horizon = int(state.env_cfg.max_steps_in_episode)
    for age in (min(50, horizon - 1), max(horizon - 10, 0), horizon - 1):
        variants.append((f'episode_age_{age}', state._replace(env_steps=jnp.int32(age)),
                         dict(age=age, remaining_actions=horizon - age)))
    return variants


def run(args):
    import jax
    import jax.numpy as jnp
    from terra.config import BatchConfig, MapsDimsConfig
    from terra.env import TerraEnv
    from train import TrainConfig
    from train_mixed import MixedAgentTrainConfig
    from utils.helpers import checkpoint_evaluation_config
    from utils.models import MapsNet, get_model_ready, validate_model_params_match
    from utils.utils_ppo import obs_to_model_input

    # Historical checkpoints were pickled by a script run as __main__.
    sys.modules['__main__'].TrainConfig = TrainConfig
    sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig
    with args.checkpoint.open('rb') as stream:
        checkpoint = pickle.load(stream)
    config = checkpoint_evaluation_config(checkpoint)
    paths = sorted(args.audit_dir.glob('slot_*/states.pkl'))
    if args.slots:
        chosen = set(args.slots)
        paths = [p for p in paths if int(p.parent.name[5:]) in chosen]
        if {int(p.parent.name[5:]) for p in paths} != chosen:
            raise ValueError('Not every requested slot has saved states')
    paths = paths[:args.max_cases]
    if not paths:
        raise ValueError('No saved failure states selected')

    with paths[0].open('rb') as stream:
        example = pickle.load(stream)['snapshot']
    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(
            maps_edge_length=example.state.world.action_map.map.shape[-1])),
        executable_dig_observation=bool(config.executable_dig_observation),
    )
    model, initialized = get_model_ready(jax.random.PRNGKey(0), config, env)
    validate_model_params_match(initialized, checkpoint['model'], str(args.checkpoint))

    @jax.jit
    def observe(state):
        wrapped = TerraEnv.wrap_state(
            state, executable_dig_observation=bool(config.executable_dig_observation))
        return TerraEnv._state_to_obs_dict(wrapped)

    @jax.jit
    def infer(inputs):
        (values, logits), collected = model.apply(
            checkpoint['model'], inputs,
            capture_intermediates=lambda module, method: (
                isinstance(module, MapsNet) and method == '__call__'),
            mutable=['intermediates'])
        embedding = collected['intermediates']['maps_net']['__call__'][0]
        return values[0], logits[0], embedding[0]

    rows = []
    for path in paths:
        with path.open('rb') as stream:
            stored = pickle.load(stream)
        state = stored['snapshot'].state
        history = jnp.asarray(stored['previous_actions'])[None]

        def prepare(candidate):
            observation = observe(candidate)
            batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], observation)
            return observation, obs_to_model_input(batch, history, config)

        observation, inputs = prepare(state)
        reference = jax.device_get(infer(inputs))
        for name, candidate, metadata in counterfactual_states(state):
            other_observation, other_inputs = prepare(candidate)
            result = jax.device_get(infer(other_inputs))
            rows.append(dict(slot=int(path.parent.name[5:]),
                             snapshot_age=int(state.env_steps), probe=name,
                             **metadata,
                             raw_observation_values_changed=tree_changed_entries(
                                 observation, other_observation),
                             model_input_values_changed=tree_changed_entries(
                                 inputs, other_inputs),
                             **output_difference(reference, result)))
        print(f'probed {path.parent.name}: {len(rows)} counterfactuals so far', flush=True)

    grouped = {}
    for prefix in ('nearest_residual', 'furthest_residual', 'pile_height', 'episode_age'):
        selected = [row for row in rows if row['probe'].startswith(prefix)]
        grouped[prefix] = dict(
            cases=len(selected),
            input_collisions=sum(row['model_input_values_changed'] == 0 for row in selected),
            action_changes=sum(row['greedy_action_changed'] for row in selected),
            mean_policy_total_variation=(float(np.mean([row['policy_total_variation']
                                                        for row in selected])) if selected else None),
            mean_abs_value_delta=(float(np.mean([abs(row['value_delta'])
                                                 for row in selected])) if selected else None),
        )
    report = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_update=int(checkpoint.get('next_update', checkpoint.get('update', -1))),
        audit_dir=str(args.audit_dir.resolve()),
        jax_version=jax.__version__, platforms=[device.platform for device in jax.devices()],
        source='snapshot physical states and previous-action history from bounded failure audit',
        parameter_count=sum(int(x.size) for x in jax.tree_util.tree_leaves(checkpoint['model'])),
        clip_action_maps=bool(config.clip_action_maps),
        time_observation_mode=getattr(config, 'time_observation_mode', 'none'),
        limitations=[
            'Counterfactual cell edits are representation probes, not legal mass-conserving transitions.',
            'Sensitivity does not establish correct action selection or map-completion ability.',
            'Input collisions cannot be repaired by increasing model capacity alone.',
            'The selected failures are purposeful cases, not a prevalence sample.',
        ], summary=grouped, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(grouped, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slots', type=int, nargs='+')
    parser.add_argument('--max-cases', type=int, default=12)
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error('--max-cases must be positive')
    # Explicitly leave platform choice to the caller (CPU for cheap probes).
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    run(args)


if __name__ == '__main__':
    main()
