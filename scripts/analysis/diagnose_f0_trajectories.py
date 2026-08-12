#!/usr/bin/env python3
"""Replay the four decision-relevant F0 checkpoints with compact policy traces.

This is deliberately a one-off recovery diagnostic, not a new evaluation
framework.  It replays the same 32-reset batches as ``eval_f0_identity.py`` so
that transition RNG and batching stay identical, validates every replayed row
against the sealed evaluator JSON, and retains detailed traces only for six
preregistered reset/checkpoint pairs.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from eval_f0_identity import (
    HORIZON,
    RESET_SEEDS,
    configure_for_identity,
    declared_reset_keys,
    sha256_file,
    verify_production_checkpoint,
    verify_single_identity_reset,
)
from train import TrainConfig
from train_mixed import (
    MixedAgentTrainConfig,
    _validate_checkpoint_architecture,
    make_mixed_agent_states,
)
from utils.helpers import load_pkl_object
from utils.utils_ppo import obs_to_model_input, wrap_action

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

SCHEMA = "terra_f0_trajectory_diagnosis_v1"
ROLLOUT_SEED = RESET_SEEDS[0]
ACTION_NAMES = (
    "FORWARD",
    "BACKWARD",
    "CLOCK",
    "ANTICLOCK",
    "CABIN_CLOCK",
    "CABIN_ANTICLOCK",
    "DO",
    "DO_NOTHING",
)
TRACE_SELECTION = {
    ("foundation", 900): (2026072600,),
    ("foundation", 1000): (2026072600,),
    ("trench", 900): (2026072600, 2026072601),
    ("trench", 1000): (2026072602, 2026072611),
}
COMPLETION_COMPONENTS = {
    "absolute": "absolute_completion",
    "dig": "dig_completion_total",
    "dump_purity": "dump_completion_action_map",
    "dump_volume": "total_dig_dump_completion",
    "unloaded": "unloaded_completion",
    "accepted_dump_volume": "accepted_dump_volume",
    "illegal_dump_volume": "illegal_dump_volume",
}
FLOAT_VALIDATION_ATOL = 2e-5


def run_length_encode(values: list[Any]) -> list[dict[str, Any]]:
    """Compress a sequence while retaining one-based inclusive step bounds."""
    if not values:
        return []
    runs = []
    start = 1
    previous = values[0]
    for index, value in enumerate(values[1:], start=2):
        if value == previous:
            continue
        runs.append(
            {
                "start_step": start,
                "end_step": index - 1,
                "count": index - start,
                "value": previous,
            }
        )
        start = index
        previous = value
    runs.append(
        {
            "start_step": start,
            "end_step": len(values),
            "count": len(values) - start + 1,
            "value": previous,
        }
    )
    return runs


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _digest_arrays(named_arrays: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def observation_digest(
    observation: dict[str, jax.Array],
    previous_actions: jax.Array,
    index: int,
) -> str:
    arrays = [
        (name, np.asarray(value[index]))
        for name, value in sorted(observation.items())
        if hasattr(value, "shape")
        and value.shape
        and value.shape[0] == len(RESET_SEEDS)
    ]
    arrays.append(("previous_actions", np.asarray(previous_actions[index])))
    return _digest_arrays(arrays)


def state_digest(observation: dict[str, jax.Array], index: int) -> str:
    return _digest_arrays(
        [
            ("action_map", np.asarray(observation["action_map"][index])),
            ("agent_states", np.asarray(observation["agent_states"][index])),
            ("agent_active", np.asarray(observation["agent_active"][index])),
        ]
    )


def map_metrics(observation: dict[str, jax.Array], index: int) -> dict[str, Any]:
    target = np.asarray(observation["target_map"][index], dtype=np.float64)
    action = np.asarray(observation["action_map"][index], dtype=np.float64)
    agent_states = np.asarray(observation["agent_states"][index], dtype=np.float64)
    active = np.asarray(observation["agent_active"][index], dtype=bool)
    required = np.where(target < 0, -target, 0.0)
    dug = np.where(
        target < 0,
        np.clip(-action, a_min=0.0, a_max=required),
        0.0,
    )
    positive = np.clip(action, a_min=0.0, a_max=None)
    accepted = float(np.where(target > 0, positive, 0.0).sum())
    total_positive = float(positive.sum())
    loaded = float(np.maximum(agent_states[active, 5], 0.0).sum())
    required_volume = float(required.sum())
    return {
        "required_dig_volume": required_volume,
        "dug_required_volume": float(dug.sum()),
        "remaining_dig_volume": float(required_volume - dug.sum()),
        "accepted_dump_volume": accepted,
        "illegal_dump_volume": float(total_positive - accepted),
        "loaded_volume": loaded,
        "world_signed_mass": float(action.sum()),
        "conserved_mass": float(action.sum() + loaded),
        "active_agent_state": agent_states[0].tolist(),
    }


def _component_row(
    components: dict[str, jax.Array],
    index: int,
) -> dict[str, float]:
    row = {}
    for output_name, component_name in COMPLETION_COMPONENTS.items():
        row[output_name] = float(np.asarray(components[component_name])[index])
    for name in (
        "terminal",
        "trench",
        "existence",
        "remaining_edge_dig_tiles",
        "remaining_inner_dig_tiles",
    ):
        row[name] = float(np.asarray(components[name])[index])
    row["agent_reward"] = float(np.asarray(components["agent_rewards"])[index].sum())
    return row


def _preserve_inactive(
    previous: jax.Array,
    candidate: jax.Array,
    active: jax.Array,
) -> jax.Array:
    if not hasattr(candidate, "shape"):
        return candidate
    if candidate.ndim == 0 or candidate.shape[0] != len(RESET_SEEDS):
        return candidate
    mask = active.reshape((active.shape[0],) + (1,) * (candidate.ndim - 1))
    return jnp.where(mask, candidate, previous)


def _checkpoint_path(run_root: Path, identity: str, update: int) -> Path:
    matches = sorted(
        (run_root / identity / "checkpoints").glob(f"*_update_{update:06d}.pkl")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {identity} update-{update} checkpoint, got {matches}"
        )
    return matches[0]


def _eval_records(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "terra_f0_identity_eval_v1":
        raise RuntimeError(f"unexpected F0 evaluator schema in {path}")
    return {int(record["checkpoint_update"]): record for record in payload["records"]}


def _expected_by_seed(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["reset_seed"]): row for row in record["summary"]["per_reset"]}


def validate_replay(
    expected_record: dict[str, Any],
    observed: dict[str, np.ndarray],
    action_sequence: np.ndarray,
) -> dict[str, Any]:
    """Require exact discrete parity and float32-scale metric parity."""
    expected = _expected_by_seed(expected_record)
    if set(expected) != set(RESET_SEEDS):
        raise RuntimeError("sealed evaluator reset grid changed")
    maximum_float_error = 0.0
    for index, seed in enumerate(RESET_SEEDS):
        row = expected[seed]
        discrete_pairs = {
            "success": (bool(observed["success"][index]), bool(row["success"])),
            "terminated": (
                bool(observed["terminated"][index]),
                bool(row["terminated"]),
            ),
            "steps": (int(observed["length"][index]), int(row["steps"])),
            "no_effect_action_count": (
                int(observed["no_effect"][index]),
                int(row["no_effect_action_count"]),
            ),
        }
        for name, (actual, sealed) in discrete_pairs.items():
            if actual != sealed:
                raise RuntimeError(
                    f"replay mismatch update={expected_record['checkpoint_update']} "
                    f"seed={seed} field={name}: {actual} != {sealed}"
                )
        float_pairs = {
            "return": (observed["return"][index], row["return"]),
            "absolute": (
                observed["absolute"][index],
                row["terminal_absolute_completion"],
            ),
            "dig": (
                observed["dig"][index],
                row["terminal_dig_completion"],
            ),
            "dump_purity": (
                observed["dump_purity"][index],
                row["terminal_dump_purity"],
            ),
            "dump_volume": (
                observed["dump_volume"][index],
                row["terminal_dump_volume"],
            ),
            "unloaded": (
                observed["unloaded"][index],
                row["terminal_unloaded_completion"],
            ),
            "accepted_dump_volume": (
                observed["accepted_dump_volume"][index],
                row["accepted_dump_volume"],
            ),
            "illegal_dump_volume": (
                observed["illegal_dump_volume"][index],
                row["illegal_dump_volume"],
            ),
        }
        for name, (actual, sealed) in float_pairs.items():
            error = abs(float(actual) - float(sealed))
            maximum_float_error = max(maximum_float_error, error)
            if error > FLOAT_VALIDATION_ATOL:
                raise RuntimeError(
                    f"replay mismatch update={expected_record['checkpoint_update']} "
                    f"seed={seed} field={name}: {actual} != {sealed}"
                )

    saved_trajectory = expected_record["summary"].get("successful_action_trajectory")
    trajectory_checked = False
    if saved_trajectory is not None:
        seed = int(saved_trajectory["reset_seed"])
        index = RESET_SEEDS.index(seed)
        steps = int(saved_trajectory["steps"])
        observed_actions = action_sequence[:steps, index].tolist()
        if observed_actions != saved_trajectory["actions"]:
            raise RuntimeError("saved successful action trajectory did not replay")
        trajectory_checked = True
    return {
        "passed": True,
        "rows_checked": len(RESET_SEEDS),
        "float_atol": FLOAT_VALIDATION_ATOL,
        "maximum_float_error": maximum_float_error,
        "saved_successful_trajectory_checked": trajectory_checked,
    }


def summarize_trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [step["action_name"] for step in steps]
    effects = [bool(step["action_had_effect"]) for step in steps]
    loaded = [step["pre"]["loaded_volume"] > 0 for step in steps]
    action_counts = Counter(actions)
    effect_counts = Counter(
        step["action_name"] for step in steps if step["action_had_effect"]
    )
    observation_counts = Counter(step["observation_digest"] for step in steps)
    state_counts = Counter(step["state_digest"] for step in steps)
    do_opportunities = [
        step
        for step in steps
        if step["counterfactual_effect_mask"][6] and step["action"] != 6
    ]
    counterfactuals = [
        step["counterfactual_do"]
        for step in do_opportunities
        if step["counterfactual_do"] is not None
    ]
    terrain_events = [
        {
            "step": step["step"],
            "action": step["action_name"],
            "reward": step["reward"],
            "dig_before": step["pre"]["dug_required_volume"],
            "dig_after": step["post"]["dug_required_volume"],
            "accepted_before": step["pre"]["accepted_dump_volume"],
            "accepted_after": step["post"]["accepted_dump_volume"],
            "loaded_before": step["pre"]["loaded_volume"],
            "loaded_after": step["post"]["loaded_volume"],
            "absolute_after": step["reward_components"]["absolute"],
        }
        for step in steps
        if (
            step["pre"]["dug_required_volume"] != step["post"]["dug_required_volume"]
            or step["pre"]["accepted_dump_volume"]
            != step["post"]["accepted_dump_volume"]
            or step["pre"]["loaded_volume"] != step["post"]["loaded_volume"]
        )
    ]
    do_ranks = [
        int(step["do_logit_rank"])
        for step in steps
        if step["counterfactual_effect_mask"][6]
    ]
    return {
        "steps": len(steps),
        "action_counts": dict(sorted(action_counts.items())),
        "effective_action_counts": dict(sorted(effect_counts.items())),
        "effective_steps": int(sum(effects)),
        "no_effect_steps": int(len(effects) - sum(effects)),
        "loaded_steps": int(sum(loaded)),
        "unloaded_steps": int(len(loaded) - sum(loaded)),
        "effect_possible_counts": {
            ACTION_NAMES[action]: int(
                sum(step["counterfactual_effect_mask"][action] for step in steps)
            )
            for action in range(len(ACTION_NAMES))
        },
        "do_effect_possible_steps": len(do_ranks),
        "do_chosen_when_effect_possible": int(
            sum(
                step["action"] == 6 and step["counterfactual_effect_mask"][6]
                for step in steps
            )
        ),
        "do_logit_rank_when_effect_possible": {
            "minimum": min(do_ranks) if do_ranks else None,
            "maximum": max(do_ranks) if do_ranks else None,
            "mean": float(np.mean(do_ranks)) if do_ranks else None,
        },
        "unchosen_do_opportunities": len(do_opportunities),
        "counterfactual_do": {
            "records": len(counterfactuals),
            "positive_immediate_advantage": int(
                sum(row["reward_advantage"] > 0 for row in counterfactuals)
            ),
            "mean_immediate_advantage": (
                float(np.mean([row["reward_advantage"] for row in counterfactuals]))
                if counterfactuals
                else None
            ),
            "would_change_terrain_or_load": int(
                sum(row["action_had_effect"] for row in counterfactuals)
            ),
        },
        "unique_policy_observations": len(observation_counts),
        "maximum_policy_observation_repetitions": max(
            observation_counts.values(),
            default=0,
        ),
        "unique_physical_states": len(state_counts),
        "maximum_physical_state_repetitions": max(
            state_counts.values(),
            default=0,
        ),
        "action_phase_runs": run_length_encode(
            [
                {
                    "action": action,
                    "effect": effect,
                    "loaded": is_loaded,
                }
                for action, effect, is_loaded in zip(
                    actions,
                    effects,
                    loaded,
                    strict=True,
                )
            ]
        ),
        "terrain_or_load_events": terrain_events,
        "final": {
            "return": float(sum(step["reward"] for step in steps)),
            "absolute_completion": steps[-1]["reward_components"]["absolute"],
            "dig_completion": steps[-1]["reward_components"]["dig"],
            "dump_purity": steps[-1]["reward_components"]["dump_purity"],
            "dump_volume_completion": steps[-1]["reward_components"]["dump_volume"],
            "accepted_dump_volume": steps[-1]["post"]["accepted_dump_volume"],
            "illegal_dump_volume": steps[-1]["post"]["illegal_dump_volume"],
            "loaded_volume": steps[-1]["post"]["loaded_volume"],
        },
    }


def _counterfactual_effect_function(env):
    """Return whether each action would change pose, orientation, or load."""
    dummy_action = env.batch_cfg.action_type.do_nothing()

    @jax.jit
    def effect_possible(state):
        mask = jax.vmap(lambda item: item._get_action_mask(dummy_action))(state)
        do_nothing = jnp.zeros((mask.shape[0], 1), dtype=jnp.bool_)
        return jnp.concatenate((mask, do_nothing), axis=1)

    return effect_possible


def replay_checkpoint(
    *,
    env,
    env_params,
    config,
    apply_fn,
    checkpoint: dict[str, Any],
    expected_record: dict[str, Any],
    selected_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reset_keys = declared_reset_keys()
    timestep = env.reset(env_params, reset_keys)
    previous_actions = jnp.zeros(
        (len(RESET_SEEDS), config.num_prev_actions),
        dtype=jnp.int32,
    )
    rng = jax.random.PRNGKey(ROLLOUT_SEED)
    rng, _ = jax.random.split(rng)
    terminated = jnp.zeros(len(RESET_SEEDS), dtype=jnp.bool_)
    succeeded = jnp.zeros_like(terminated)
    lengths = jnp.zeros(len(RESET_SEEDS), dtype=jnp.int32)
    returns = jnp.zeros(len(RESET_SEEDS), dtype=jnp.float32)
    no_effect = jnp.zeros(len(RESET_SEEDS), dtype=jnp.int32)
    terminal = {
        name: jnp.zeros(len(RESET_SEEDS), dtype=jnp.float32)
        for name in COMPLETION_COMPONENTS
    }
    traces = {seed: [] for seed in selected_seeds}
    action_rows = []
    counterfactual_effect = _counterfactual_effect_function(env)

    for step_index in range(1, HORIZON + 1):
        active = ~terminated
        pre_observation = timestep.observation
        pre_previous_actions = previous_actions
        model_input = obs_to_model_input(
            pre_observation,
            pre_previous_actions,
            config,
        )
        value, logits = apply_fn(checkpoint["model"], model_input)
        action = jnp.argmax(logits, axis=-1)
        action_rows.append(np.asarray(action, dtype=np.int32))
        effect_possible = counterfactual_effect(timestep.state)

        rng, _, rng_step = jax.random.split(rng, 3)
        previous_actions = jnp.roll(previous_actions, shift=1, axis=1)
        previous_actions = previous_actions.at[:, 0].set(action)
        step_keys = jax.random.split(rng_step, len(RESET_SEEDS))
        candidate = env.step_no_reset(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            step_keys,
        )
        preserve_active = functools.partial(_preserve_inactive, active=active)
        timestep_next = jax.tree_util.tree_map(
            preserve_active,
            timestep,
            candidate,
        )
        previous_actions = jnp.where(
            timestep_next.done[:, None],
            jnp.zeros_like(previous_actions),
            previous_actions,
        )

        chosen_reward = jnp.where(active, timestep_next.reward, 0.0)
        changed_action_map = jnp.any(
            timestep_next.observation["action_map"] != pre_observation["action_map"],
            axis=(-2, -1),
        )
        changed_agent = jnp.any(
            timestep_next.observation["agent_states"]
            != pre_observation["agent_states"],
            axis=(-2, -1),
        )
        no_effect += (active & ~changed_action_map & ~changed_agent).astype(jnp.int32)
        returns += chosen_reward
        lengths += active.astype(jnp.int32)
        components = timestep_next.info["reward_components"]
        for output_name, component_name in COMPLETION_COMPONENTS.items():
            terminal[output_name] = jnp.where(
                active & timestep_next.done,
                components[component_name],
                terminal[output_name],
            )
        succeeded |= active & timestep_next.info["task_done"]
        terminated |= active & timestep_next.done

        selected_indices = [RESET_SEEDS.index(seed) for seed in selected_seeds]
        need_counterfactual = any(
            bool(np.asarray(effect_possible)[index, 6])
            and int(np.asarray(action)[index]) != 6
            and bool(np.asarray(active)[index])
            for index in selected_indices
        )
        do_candidate = None
        if need_counterfactual:
            do_action = jnp.full((len(RESET_SEEDS),), 6, dtype=jnp.int32)
            do_candidate = env.step_no_reset(
                timestep,
                wrap_action(do_action, env.batch_cfg.action_type),
                step_keys,
            )

        logits_host = np.asarray(logits)
        value_host = np.asarray(value).reshape(len(RESET_SEEDS), -1)
        action_host = np.asarray(action, dtype=np.int32)
        effect_possible_host = np.asarray(effect_possible, dtype=bool)
        active_host = np.asarray(active, dtype=bool)
        effect_host = np.asarray(
            timestep_next.info["action_had_effect"],
            dtype=bool,
        )
        reward_host = np.asarray(chosen_reward, dtype=np.float64)
        for seed, index in zip(selected_seeds, selected_indices, strict=True):
            if not active_host[index]:
                continue
            probabilities = _softmax(logits_host[index])
            chosen_action = int(action_host[index])
            sorted_actions = np.argsort(-logits_host[index], kind="stable")
            do_rank = int(np.flatnonzero(sorted_actions == 6)[0]) + 1
            counterfactual = None
            if (
                do_candidate is not None
                and effect_possible_host[index, 6]
                and chosen_action != 6
            ):
                do_reward = float(np.asarray(do_candidate.reward)[index])
                do_components = do_candidate.info["reward_components"]
                counterfactual = {
                    "reward": do_reward,
                    "reward_advantage": do_reward - float(reward_host[index]),
                    "action_had_effect": bool(
                        np.asarray(do_candidate.info["action_had_effect"])[index]
                    ),
                    "post": map_metrics(do_candidate.observation, index),
                    "reward_components": _component_row(do_components, index),
                }
            traces[seed].append(
                {
                    "step": step_index,
                    "action": chosen_action,
                    "action_name": ACTION_NAMES[chosen_action],
                    "action_had_effect": bool(effect_host[index]),
                    "counterfactual_effect_mask": effect_possible_host[index].tolist(),
                    "policy_value": float(value_host[index, 0]),
                    "logits": logits_host[index].astype(float).tolist(),
                    "probabilities": probabilities.tolist(),
                    "top_logit_margin": float(
                        logits_host[index, sorted_actions[0]]
                        - logits_host[index, sorted_actions[1]]
                    ),
                    "do_logit_rank": do_rank,
                    "reward": float(reward_host[index]),
                    "reward_components": _component_row(components, index),
                    "pre": map_metrics(pre_observation, index),
                    "post": map_metrics(timestep_next.observation, index),
                    "observation_digest": observation_digest(
                        pre_observation,
                        pre_previous_actions,
                        index,
                    ),
                    "state_digest": state_digest(pre_observation, index),
                    "counterfactual_do": counterfactual,
                }
            )

        timestep = timestep_next
        if bool(jnp.all(terminated).item()):
            break

    action_sequence = np.stack(action_rows)
    observed = {
        "success": np.asarray(succeeded),
        "terminated": np.asarray(terminated),
        "length": np.asarray(lengths),
        "return": np.asarray(returns),
        "no_effect": np.asarray(no_effect),
        **{name: np.asarray(value) for name, value in terminal.items()},
    }
    validation = validate_replay(
        expected_record,
        observed,
        action_sequence,
    )
    records = []
    expected_rows = _expected_by_seed(expected_record)
    for seed in selected_seeds:
        steps = traces[seed]
        expected = expected_rows[seed]
        summary = summarize_trace(steps)
        if summary["steps"] != int(expected["steps"]):
            raise RuntimeError(f"selected trace length mismatch for seed {seed}")
        if not math.isclose(
            summary["final"]["return"],
            float(expected["return"]),
            rel_tol=0.0,
            abs_tol=FLOAT_VALIDATION_ATOL,
        ):
            raise RuntimeError(f"selected trace return mismatch for seed {seed}")
        records.append(
            {
                "reset_seed": seed,
                "sealed_evaluator_row": expected,
                "summary": summary,
                "steps": steps,
            }
        )
    return validation, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": SCHEMA,
        "observer_only": True,
        "completion_contract": "exact_visible_dump_v1",
        "reward_contract": "corrected_dense_v1",
        "run_root": str(run_root),
        "rollout_seed": ROLLOUT_SEED,
        "reset_seeds": list(RESET_SEEDS),
        "horizon": HORIZON,
        "trace_selection": {
            f"{identity}_update_{update}": list(seeds)
            for (identity, update), seeds in TRACE_SELECTION.items()
        },
        "inputs": {},
        "reset_verification": {},
        "replays": [],
    }

    os.environ["DATASET_PATH"] = str(run_root / "bank")
    os.environ["DATASET_SIZE"] = "1"
    for identity in ("foundation", "trench"):
        eval_path = run_root / identity / "eval.json"
        eval_records = _eval_records(eval_path)
        payload["inputs"][f"{identity}_eval"] = {
            "path": str(eval_path),
            "sha256": sha256_file(eval_path),
        }
        checkpoints = {
            update: (
                _checkpoint_path(run_root, identity, update),
                None,
            )
            for update in (900, 1000)
        }
        for update, (path, _) in checkpoints.items():
            checkpoint = load_pkl_object(str(path))
            checkpoints[update] = (path, checkpoint)
            payload["inputs"][f"{identity}_checkpoint_{update}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_gate": verify_production_checkpoint(
                    checkpoint,
                    identity,
                    update,
                ),
            }

        reference_config = checkpoints[900][1]["train_config"]
        config = configure_for_identity(reference_config, identity)
        _, env, env_params, initialized_state = make_mixed_agent_states(config)
        env_params = jax.tree_util.tree_map(lambda value: value[0], env_params)
        payload["reset_verification"][identity] = verify_single_identity_reset(
            env,
            env_params,
            declared_reset_keys(),
            run_root / "bank" / identity,
            json.loads(
                (run_root / "bank" / identity / "manifest.jsonl").read_text().strip()
            ),
        )
        for update in (900, 1000):
            path, checkpoint = checkpoints[update]
            _validate_checkpoint_architecture(checkpoint, config)
            validation, records = replay_checkpoint(
                env=env,
                env_params=env_params,
                config=config,
                apply_fn=initialized_state.apply_fn,
                checkpoint=checkpoint,
                expected_record=eval_records[update],
                selected_seeds=TRACE_SELECTION[(identity, update)],
            )
            payload["replays"].append(
                {
                    "identity": identity,
                    "checkpoint_update": update,
                    "checkpoint": str(path),
                    "validation": validation,
                    "traces": records,
                }
            )
            print(
                f"{identity} update {update}: replay parity passed; "
                f"traces={len(records)}",
                flush=True,
            )

    payload["gate"] = {
        "passed": all(replay["validation"]["passed"] for replay in payload["replays"]),
        "replays": len(payload["replays"]),
        "selected_traces": sum(len(replay["traces"]) for replay in payload["replays"]),
        "validated_evaluator_rows": sum(
            replay["validation"]["rows_checked"] for replay in payload["replays"]
        ),
        "bounded_transitions": len(payload["replays"]) * len(RESET_SEEDS) * HORIZON,
        "gradient_updates": 0,
    }
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"F0_TRAJECTORY_DIAGNOSIS_GATE={payload['gate']['passed']}", flush=True)


if __name__ == "__main__":
    main()
