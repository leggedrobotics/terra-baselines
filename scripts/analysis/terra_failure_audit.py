#!/usr/bin/env python3
"""Bounded u5000 failure replay; no training and no claim of infeasibility.

Run --select-only first, then replay the same report with a free GPU. A replay
captures complete initial agents, previous-action histories, and a physical
state at the last material change at/before action 400. An optional bounded
beam enumerates real Terra successors from those states. It is a witness
finder, not an exhaustive planner or proof that a residual cannot be finished.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ACTION_NAMES = ('forward', 'backward', 'clock', 'anticlock', 'cabin_clock',
                'cabin_anticlock', 'do', 'wait')


def read_report(path):
    value = json.loads(Path(path).read_text())
    report = value[0] if isinstance(value, list) and len(value) == 1 else value
    if not isinstance(report, dict) or len(report['per_map']) != 608:
        raise ValueError('Expected the complete 608-slot generalist fixed report')
    if report['checkpoint_update'] != 5000 or report['horizon'] != 450:
        raise ValueError('This audit is for the frozen u5000, 450-action report')
    if not report['deterministic'] or not report['reset_verification']['passed']:
        raise ValueError('Expected deterministic replay with verified fixed resets')
    return report


def select_cases(report):
    """Purposeful failure diversity, explicitly not a prevalence estimate."""
    failed = [row for row in report['per_map'] if not row['success']]
    chosen = []

    def take(label, predicate, key, reverse=True):
        used = {item['slot_index'] for item in chosen}
        rows = [row for row in failed if row['slot_index'] not in used and predicate(row)]
        if not rows:
            raise ValueError(f'No unused failure for {label}')
        row = sorted(rows, key=lambda r: (key(r), -r['slot_index']), reverse=reverse)[0]
        chosen.append(dict(row, audit_group=label))

    for cell in ('fnd-slab-side1-obj', 'fnd-proc-ring3x',
                 'v7-fnd-bearing-walls-adjacent', 'fnd-slab-apron-d12'):
        take('foundation', lambda r, cell=cell: r['primary_cell'] == cell,
             lambda r: r['longest_material_stall_steps'])
    road = lambda r: r['family'] == 'trench' and 'road' in r['primary_cell']
    take('road_no_excavation', lambda r: road(r) and r['dig_fraction'] == 0,
         lambda r: r['longest_material_stall_steps'])
    take('road_stalled', lambda r: road(r) and 0 < r['dig_fraction'] < 1,
         lambda r: r['longest_material_stall_steps'])
    take('road_material_cycling', lambda r: road(r) and 0 < r['dig_fraction'] < 1,
         lambda r: r['longest_material_stall_steps'], reverse=False)
    take('junction_small_residual', lambda r: 'tee' in r['primary_cell'],
         lambda r: r['dig_fraction'])
    take('junction_stalled', lambda r: 'tee' in r['primary_cell'],
         lambda r: r['longest_material_stall_steps'])
    take('junction_network', lambda r: 'net4' in r['primary_cell'] and not road(r),
         lambda r: r['longest_material_stall_steps'])
    take('foundation_cleanup', lambda r: r['family'] == 'foundation' and r['dig_fraction'] == 1,
         lambda r: r['terminal_soil_fraction'])
    take('road_cleanup', lambda r: road(r) and r['dig_fraction'] == 1,
         lambda r: r['terminal_soil_fraction'])
    return chosen


def plain(value):
    if hasattr(value, '_asdict'):
        return {k: plain(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return np.asarray(value).tolist()


def compare_episode_ages(state, config, previous_actions):
    """Only env_steps changes: physical state and the full history stay fixed."""
    import jax
    import jax.numpy as jnp
    from terra.env import TerraEnv
    from utils.utils_ppo import obs_to_model_input

    ages = (50, 440, 449, 450)
    states = [state._replace(env_steps=jnp.int32(age)) for age in ages]
    observations = [TerraEnv._state_to_obs_dict(item) for item in states]
    inputs = [obs_to_model_input(jax.tree_util.tree_map(lambda x: x[None], obs),
                                previous_actions[None], config)
              for obs in observations]
    equal_observations = all(
        np.array_equal(np.asarray(observations[0][key]), np.asarray(obs[key]))
        for obs in observations[1:] for key in observations[0])
    equal_inputs = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for other in inputs[1:] for a, b in zip(inputs[0], other))
    done = [bool(item._is_done(item.world.action_map.map, item.world.target_map.map)[0])
            for item in states]
    # State comparisons exclude precisely the one deliberately changed field.
    identical_other_state = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for item in states[1:]
        for a, b in zip(jax.tree_util.tree_leaves(states[0]._replace(env_steps=0)),
                        jax.tree_util.tree_leaves(item._replace(env_steps=0))))
    return dict(ages=ages, remaining_actions=[450 - age for age in ages],
                identical_physics_history_and_other_state=identical_other_state,
                raw_observations_identical=equal_observations,
                model_inputs_identical=equal_inputs, done=done,
                observation_keys=sorted(observations[0]),
                conclusion='Time is aliased; this does not measure its learning effect.')


def pose_diagnostics(state):
    """Call actual eligibility/filter methods; counts are diagnostic, not vetoes."""
    import jax.numpy as jnp
    cone = state._build_dig_dump_cone().astype(jnp.bool_)
    target = state.world.target_map.map.reshape(-1)
    action = state.world.action_map.map.reshape(-1)
    occupied = state._active_base_footprint_mask().reshape(-1)
    eligible = state._mask_out_wrong_dig_tiles(cone)
    mask, volume, relift, admitted = state._dig_eligibility(cone)
    dump = cone
    counts = {'cone': jnp.sum(cone)}
    for name, method in (
        ('not_holes', state._exclude_dig_tiles_from_dump_mask),
        ('dumpability', state._exclude_dumpability_mask_tiles_from_dump_mask),
        ('traversability', state._exclude_traversability_mask_tiles_from_dump_mask),
        ('anti_cycle', state._exclude_just_moved_tiles_from_dump_mask),
    ):
        dump = method(dump).astype(jnp.bool_)
        counts[name] = jnp.sum(dump)
    dump &= (state.world.padding_mask.map.reshape(-1) == 0) & ~occupied
    counts['outside_obstacle_and_chassis'] = jnp.sum(dump)
    counts['legal_disposal_cells'] = jnp.sum(dump & state._accepted_dump_mask().reshape(-1))
    cur = state._get_current_agent_state()
    return dict(
        loaded=cur.loaded[0], base_pose=jnp.concatenate((cur.pos_base, cur.angle_base)),
        cabin=cur.angle_cabin[0],
        fresh_target_cells_in_cone=jnp.sum(cone & (target < 0) & (action == 0)),
        fresh_cells_under_chassis=jnp.sum(cone & (target < 0) & (action == 0) & occupied),
        last_work_overlap_cells=jnp.sum(cone & state.world.last_dig_mask.map.reshape(-1)),
        before_trench_filter_cells=jnp.sum(eligible), after_trench_filter_cells=jnp.sum(mask),
        whole_cone_obstacle_veto=jnp.any(cone & (state.world.padding_mask.map.reshape(-1) == 1)),
        empty_agent_dig_admitted=admitted, selected_volume=volume, selected_is_relift=relift,
        fresh_volume_by_cabin=state._executable_fresh_dig_counts(),
        dump_candidate_counts=counts, dump_free_space_veto=state._dump_cone_lacks_free_space(dump),
        current_chassis_on_nonzero_soil=jnp.any(occupied & (action != 0)),
        # These immediate quantities do not establish reachability or escape.
        remaining_target_cells=jnp.sum((target < 0) & (action > target)),
    )


def physical_key(timestep):
    state = timestep.state
    return (np.asarray(state.world.action_map.map).tobytes(),
            np.asarray(state.world.last_dig_mask.map).tobytes(),
            json.dumps(plain(state.agent), sort_keys=True))


def bounded_suffix(env, snapshot, max_depth=12, width=8):
    """Beam search through real step_no_reset transitions, preserving the age.

    The beam prefers excavation plus accepted material and deduplicates terrain,
    last-work masks and agent states. Other history can be pruned, and this
    heuristic can miss valid detours. Bounds and
    unsuccessful searches must always accompany witness claims.
    """
    import jax
    import jax.numpy as jnp
    from utils.utils_ppo import wrap_action

    def measure(ts):
        s = ts.state
        target = np.asarray(s.world.target_map.map, dtype=np.int32)
        action = np.asarray(s.world.action_map.map, dtype=np.int32)
        need = np.maximum(-target, 0).sum()
        dug = np.minimum(np.maximum(-action, 0), np.maximum(-target, 0)).sum()
        accepted = (target > 0) & (np.asarray(s.world.padding_mask.map) == 0)
        soil = np.maximum(action, 0)
        return np.array([dug, soil[accepted].sum(), soil[~accepted].sum(),
                         int(s.agent.agent_states[0].loaded[0])], dtype=float), float(need)

    if int(snapshot.state.agent.num_agents) != 1 or int(snapshot.state.agent.current_agent) != 0:
        raise ValueError('This bounded audit supports the current single-excavator bank only')
    start, _ = measure(snapshot)
    frontier = [(snapshot, [], False, False, False)]
    seen = {physical_key(snapshot)}
    witnesses = {}
    first_successor_outcomes = []
    tried = 0
    action_ids = jnp.tile(jnp.arange(8, dtype=jnp.int32), width)
    wrapped = wrap_action(action_ids, env.batch_cfg.action_type)
    keys = jax.random.split(jax.random.PRNGKey(20260916), width * 8)
    depth_used = 0
    for depth in range(1, min(max_depth, 450 - int(snapshot.state.env_steps)) + 1):
        depth_used = depth
        padded = frontier + [frontier[-1]] * (width - len(frontier))
        parent_batch = jax.tree_util.tree_map(
            lambda *xs: jnp.repeat(jnp.stack(xs), 8, axis=0), *(x[0] for x in padded))
        candidates = env.step_no_reset(parent_batch, wrapped, keys)
        jax.block_until_ready(candidates.done)
        pool = []
        for i in range(len(frontier) * 8):
            tried += 1
            ts = jax.tree_util.tree_map(lambda x: x[i], candidates)
            parent, path, worked, disposed, escaped = frontier[i // 8]
            action_id = i % 8
            path = path + [action_id]
            current, _ = measure(ts)
            previous, _ = measure(parent)
            for key in ('transition_mass_residual', 'target_mutation', 'obstacle_mutation'):
                if np.any(np.asarray(ts.info[key])):
                    raise RuntimeError(f'Suffix integrity failure: {key}')
            if depth == 1:
                first_successor_outcomes.append(dict(action=ACTION_NAMES[action_id],
                    action_had_effect=bool(ts.info['action_had_effect']),
                    material_delta_units=(current - previous).tolist()))
            if not bool(ts.info['action_had_effect']):
                continue
            worked |= bool(current[0] > previous[0])
            disposed |= bool(worked and current[1] > previous[1] and current[3] < previous[3])
            moved = not np.array_equal(np.asarray(ts.state.agent.agent_states[0].pos_base),
                                       np.asarray(parent.state.agent.agent_states[0].pos_base))
            escaped |= bool(worked and disposed and moved)
            success = bool(ts.info['task_done'])
            for name, found in (
                ('successful_suffix', success),
                ('material_progress_suffix', bool(current[0] > start[0] or current[1] > start[1])),
                ('fresh_work_then_disposal_then_translation', worked and disposed and escaped),
            ):
                if found and name not in witnesses:
                    witnesses[name] = dict(actions=[ACTION_NAMES[a] for a in path],
                                           action_ids=path, progress_units=current.tolist(),
                                           end_step=int(ts.state.env_steps),
                                           remaining_target_cells=int(np.sum((np.asarray(ts.state.world.target_map.map) < 0) &
                                               (np.asarray(ts.state.world.action_map.map) > np.asarray(ts.state.world.target_map.map)))))
            if success:
                return dict(witnesses=witnesses, first_successor_outcomes=first_successor_outcomes, candidate_transitions=tried, depth=depth,
                            beam_width=width, exhaustive=False, all_remaining_access_proven=False)
            if bool(ts.done):
                continue
            key = physical_key(ts)
            if key in seen:
                continue
            seen.add(key)
            # Keep a useful mix of work and orientation states. No model or
            # hypothetical teleportation is used to create a witness.
            remaining = np.argwhere((np.asarray(ts.state.world.target_map.map) < 0) &
                                    (np.asarray(ts.state.world.action_map.map) >= 0))
            pos = np.asarray(ts.state.agent.agent_states[0].pos_base)
            distance = float(np.linalg.norm(remaining - pos, axis=1).min()) if len(remaining) else 0.0
            score = (2 * current[0] + current[1] - current[2], -distance, -current[3])
            pool.append((score, ts, path, worked, disposed, escaped))
        if not pool:
            break
        pool.sort(key=lambda item: item[0], reverse=True)
        frontier = [(ts, path, w, d, e) for _, ts, path, w, d, e in pool[:width]]
    return dict(witnesses=witnesses, first_successor_outcomes=first_successor_outcomes, candidate_transitions=tried, depth=depth_used,
                beam_width=width, exhaustive=False, all_remaining_access_proven=False,
                conclusion='No complete suffix found within this bound; infeasibility is not established.')


def replay(report, cases, output, search_depth):
    import jax
    import jax.numpy as jnp
    from eval_fixed_bank import (configure_for_bank, exact_reset_keys, load_manifest,
                                 manifest_environment_keys, prepare_manifest_episode_reset)
    from train import TrainConfig
    from train_mixed import MixedAgentTrainConfig, make_mixed_agent_states
    from utils.helpers import load_pkl_object, checkpoint_evaluation_config
    from utils.models import validate_model_params_match
    from utils.utils_ppo import obs_to_model_input, wrap_action
    sys.modules['__main__'].TrainConfig = TrainConfig
    sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig
    jax.config.update('jax_threefry_partitionable', True)

    manifest_path = Path(report['manifest'])
    rows = load_manifest(manifest_path.parent)
    slots = np.array([case['slot_index'] - 1 for case in cases], dtype=int)
    for case, slot in zip(cases, slots):
        for field in ('episode_id', 'source_id', 'scenario_id', 'reset_seed', 'slot_index'):
            if case[field] != rows[slot][field]:
                raise ValueError(f'Report/manifest identity mismatch: {field}, slot {slot + 1}')
    os.environ['DATASET_PATH'] = report['bank_root']
    os.environ['DATASET_SIZE'] = str(len(rows))
    checkpoint = load_pkl_object(report['checkpoint'])
    config = configure_for_bank(checkpoint_evaluation_config(checkpoint),
                                str(manifest_path.parent.relative_to(report['bank_root'])), len(cases))
    _, env, env_params, initialized = make_mixed_agent_states(config)
    validate_model_params_match(initialized.params, checkpoint['model'], report['checkpoint'])
    env_params = jax.tree_util.tree_map(lambda x: x[0], env_params)
    map_keys = exact_reset_keys(len(rows))[slots]
    state_keys = manifest_environment_keys(rows, len(rows), report['accepted_bank']['environment_protocol_sha256'])[slots]
    ts, _, _ = prepare_manifest_episode_reset(env, env_params, map_keys, state_keys)
    count = len(cases)
    history = jnp.zeros((count, config.num_prev_actions), dtype=jnp.int32)
    initial = jax.device_get(ts)
    # Compare actual reset terrain to its original slot, never a renumbered subset.
    for i, slot in enumerate(slots):
        for directory, value in [('images', ts.state.world.target_map.map),
                                 ('actions', ts.state.world.action_map.map),
                                 ('occupancy', ts.state.world.padding_mask.map),
                                 ('dumpability', ts.state.world.dumpability_mask_init.map),
                                 ('distance', ts.state.world.relocation_distance_map)]:
            np.testing.assert_allclose(np.asarray(value[i]).squeeze(),
                np.load(manifest_path.parent / directory / f'img_{slot + 1}.npy'), rtol=0, atol=1e-7)
    snapshots = [jax.tree_util.tree_map(lambda x: x[i], ts) for i in range(count)]
    snapshot_histories = [history[i] for i in range(count)]
    records = {name: [] for name in ('actions', 'loaded', 'positions', 'base_angles', 'cabin_angles',
                                    'dig_fraction', 'accepted_fraction', 'action_had_effect')}
    done = np.zeros(count, dtype=bool)
    started = time.monotonic()
    # Match the existing evaluator's 120-row policy kernel, with deterministic
    # row padding. Model layers do not mix examples across the batch dimension.
    @jax.jit
    def policy(params, observation, previous_actions):
        model_input = obs_to_model_input(observation, previous_actions, config)
        padded = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], 120 - count, axis=0)]), model_input)
        values, logits = initialized.apply_fn(params, padded)
        return jnp.argmax(logits[:count], axis=-1), values[:count]

    for step in range(450):
        actions, _ = policy(checkpoint['model'], ts.observation, history)
        history = jnp.roll(history, 1, axis=1).at[:, 0].set(actions)
        old = ts
        ts = env.step_no_reset(ts, wrap_action(actions, env.batch_cfg.action_type), state_keys)
        ts = jax.tree_util.tree_map(lambda a, b: jnp.where(jnp.asarray(done).reshape((count,) + (1,) * (b.ndim-1)), a, b), old, ts)
        for key in ('transition_mass_residual', 'target_mutation', 'obstacle_mutation'):
            if np.any(np.asarray(ts.info[key])):
                raise RuntimeError(f'Replay integrity failure: {key}, action {step + 1}')
        if not np.isfinite(np.asarray(ts.reward)).all():
            raise RuntimeError(f'Nonfinite replay reward at action {step + 1}')
        material_changed = np.any(np.asarray(ts.state.world.action_map.map) != np.asarray(old.state.world.action_map.map), axis=(1, 2))
        loads = np.array([np.asarray(a.loaded).reshape(count) for a in ts.state.agent.agent_states]).T
        old_loads = np.array([np.asarray(a.loaded).reshape(count) for a in old.state.agent.agent_states]).T
        material_changed |= np.any(loads != old_loads, axis=1)
        if step + 1 <= 400:
            for i in np.flatnonzero(material_changed):
                snapshots[i] = jax.tree_util.tree_map(lambda x: x[i], ts)
                snapshot_histories[i] = history[i]
        history = jnp.where(ts.done[:, None], jnp.zeros_like(history), history)
        actor = ts.state.agent.agent_states[0]
        records['actions'].append(np.asarray(actions))
        records['loaded'].append(loads[:, 0])
        records['positions'].append(np.asarray(actor.pos_base))
        records['base_angles'].append(np.asarray(actor.angle_base).reshape(count))
        records['cabin_angles'].append(np.asarray(actor.angle_cabin).reshape(count))
        target = np.asarray(ts.state.world.target_map.map, dtype=np.int32)
        terrain = np.asarray(ts.state.world.action_map.map, dtype=np.int32)
        required = np.maximum(-target, 0).sum(axis=(1, 2))
        dug = np.minimum(np.maximum(-terrain, 0), np.maximum(-target, 0)).sum(axis=(1, 2))
        accepted_mask = (target > 0) & (np.asarray(ts.state.world.padding_mask.map) == 0)
        accepted = np.where(accepted_mask, np.maximum(terrain, 0), 0).sum(axis=(1, 2))
        records['dig_fraction'].append(dug / required)
        records['accepted_fraction'].append(accepted / required)
        records['action_had_effect'].append(np.asarray(ts.info['action_had_effect']))
        done |= np.asarray(ts.done)
        if step % 100 == 0:
            print(f'replay action {step + 1}/450, elapsed {time.monotonic()-started:.1f}s', flush=True)
    np.savez_compressed(output / 'traces.npz', **{k: np.asarray(v) for k, v in records.items()})
    results = []
    diagnose = jax.jit(pose_diagnostics)
    for i, case in enumerate(cases):
        snapshot = snapshots[i]
        terminal = jax.tree_util.tree_map(lambda x: x[i], ts)
        start = jax.tree_util.tree_map(lambda x: x[i], initial)
        case_dir = output / f"slot_{case['slot_index']:03d}"
        case_dir.mkdir()
        with (case_dir / 'states.pkl').open('wb') as stream:
            pickle.dump(jax.device_get(dict(initial=start, snapshot=snapshot,
                        previous_actions=snapshot_histories[i], terminal=terminal)), stream)
        (case_dir / 'initial_agent.json').write_text(json.dumps(plain(start.state.agent), indent=2) + '\n')
        observed_dig = float(records['dig_fraction'][-1][i])
        observed_accepted = float(records['accepted_fraction'][-1][i])
        matched = (not bool(terminal.info['task_done'])
                   and abs(observed_dig - case['dig_fraction']) < 1e-6
                   and abs(observed_accepted - case['terminal_soil_fraction']) < 1e-6)
        result = dict(slot_index=case['slot_index'], episode_id=case['episode_id'], audit_group=case['audit_group'],
                      reproduced_failure_and_excavation=matched, observed_dig_fraction=observed_dig,
                      match_scope='Terminal failure, excavation and accepted material match; original action-trace identity is not established.',
                      original_dig_fraction=case['dig_fraction'],
                      observed_accepted_fraction=observed_accepted,
                      original_accepted_fraction=case['terminal_soil_fraction'],
                      snapshot_step=int(snapshot.state.env_steps),
                      snapshot_rule='Latest material change at or before action 400; full physical state/history retained',
                      pose_diagnostics=plain(diagnose(snapshot.state)),
                      time_observability=compare_episode_ages(snapshot.state, config, snapshot_histories[i]))
        if not matched:
            result['suffix_search'] = {'not_run': 'Replay differed; do not attribute this new trace to the old report.'}
        (case_dir / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        print(f"slot {case['slot_index']}: matched={matched}, snapshot={result['snapshot_step']}", flush=True)
    # Preserve every replay before starting the separately charged searches.
    (output / 'replay.json').write_text(json.dumps(dict(results=results, replay_actions=450*count), indent=2) + '\n')
    if search_depth:
        for i, result in enumerate(results):
            if result['reproduced_failure_and_excavation']:
                result['suffix_search'] = bounded_suffix(env, snapshots[i], search_depth)
                case_dir = output / f"slot_{result['slot_index']:03d}"
                (case_dir / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
                print(f"slot {result['slot_index']}: witnesses {list(result['suffix_search']['witnesses'])}", flush=True)
    summary = dict(checkpoint=report['checkpoint'], cases=len(cases), matched=sum(r['reproduced_failure_and_excavation'] for r in results),
                   replay_actions=450*count, suffix_candidate_transitions=sum(r.get('suffix_search',{}).get('candidate_transitions',0) for r in results),
                   successful_suffix_witnesses=sum('successful_suffix' in r.get('suffix_search',{}).get('witnesses',{}) for r in results),
                   material_progress_witnesses=sum('material_progress_suffix' in r.get('suffix_search',{}).get('witnesses',{}) for r in results),
                   elapsed_seconds=time.monotonic()-started, results=results,
                   interpretation='Selected failures are not representative frequencies. Matching endpoints does not establish identical original action traces. No witness found is not proof of infeasibility. Escaping once does not prove access to every remaining target.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--search-depth', type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.search_depth <= 12:
        raise ValueError('The audit budget is at most 12 successor actions, beam width 8')
    report = read_report(args.report)
    cases = select_cases(report)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        raise FileExistsError(args.output / 'summary.json')
    selection = dict(source_report=str(args.report.resolve()), checkpoint=report['checkpoint'],
                     checkpoint_update=5000, bank_root=report['bank_root'], manifest=report['manifest'],
                     selection='Purposeful 12-case failure coverage, not a population sample', cases=cases)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    if args.select_only:
        print(json.dumps([dict(slot=c['slot_index'], group=c['audit_group'], cell=c['primary_cell'],
                              dig=c['dig_fraction'], accepted=c['terminal_soil_fraction']) for c in cases], indent=2))
        return
    replay(report, cases, args.output, args.search_depth)


if __name__ == '__main__':
    main()
