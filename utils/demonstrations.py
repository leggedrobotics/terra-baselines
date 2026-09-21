"""Small training-only trajectory bank for actor imitation alongside PPO."""

import copy
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils.utils_ppo import obs_to_model_input


DEMONSTRATION_GROUPS = ("foundation", "trench", "expert", "recovery")


def validate_demonstration_config(config):
    if not math.isfinite(config.demonstration_coef) or config.demonstration_coef < 0:
        raise ValueError("demonstration_coef must be finite and nonnegative")
    if type(config.demonstration_fade_transitions) is not int or config.demonstration_fade_transitions < 0:
        raise ValueError("demonstration_fade_transitions must be a nonnegative integer")
    if type(config.demonstration_batch_size) is not int or config.demonstration_batch_size < 1:
        raise ValueError("demonstration_batch_size must be a positive integer")
    if config.demonstration_npz or config.demonstration_coef or config.demonstration_fade_transitions:
        if config.actor_core != "mlp":
            raise ValueError("demonstrations require a feedforward actor with saved action history")
        if tuple(config.agent_types_override or ()) != (0,) or tuple(config.action_types_override or ()) != (0,):
            raise ValueError("demonstrations require one tracked excavator")
        if config.warm_start_from or config.resume_update is not None:
            raise ValueError("demonstrations require fresh training or native optimizer/clock resume")
        if not config.demonstration_npz:
            raise ValueError("demonstration settings require explicit --demonstration_npz")


def demonstration_coefficient(state, global_transitions):
    if state is None:
        return 0.0
    elapsed = max(0, global_transitions - state["origin_transitions"])
    return state["initial_coefficient"] * max(0.0, 1.0 - elapsed / state["fade_transitions"])


def demonstration_global_transitions(state, update, global_batch):
    if state is None:
        return update * global_batch
    return state["origin_transitions"] + (update - state["origin_update"]) * global_batch


def _field(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def restore_demonstration_schedule(config, checkpoint, checkpoint_mode, resume_update):
    """Anchor one fade to absolute experience; resubmission cannot restart it."""
    saved = checkpoint.get("demonstration_state") if checkpoint and checkpoint_mode == "resume" else None
    previous = checkpoint.get("train_config") if checkpoint else None
    if (checkpoint_mode == "resume" and _field(previous, "demonstration_npz")
            and saved is None):
        raise ValueError("demonstration checkpoint is missing its saved schedule")
    if saved is None and not config.demonstration_npz:
        return None
    batch = config.env_steps_per_update
    origin = resume_update * batch
    if checkpoint_mode == "resume":
        parent_batch = (_field(previous, "num_steps") * _field(previous, "num_envs_per_device")
                        * _field(previous, "num_devices"))
        if ("optimizer_state" not in checkpoint or checkpoint.get("next_update") != resume_update
                or config.resume_update is not None):
            raise ValueError("demonstrations require native Adam and clock resume")
        origin = resume_update * parent_batch
    requested_paths = [str(Path(p).resolve()) for p in config.demonstration_npz or ()]
    if len(set(requested_paths)) != len(requested_paths):
        raise ValueError("demonstration input paths must not be repeated")
    if saved is not None:
        now = demonstration_global_transitions(saved, resume_update, batch)
        if (saved["origin_transitions"] < 0 or saved["origin_update"] > resume_update
                or saved["fade_transitions"] <= 0 or saved["transitions_per_update"] != batch
                or not math.isfinite(saved["initial_coefficient"]) or saved["initial_coefficient"] <= 0):
            raise ValueError("invalid saved demonstration schedule")
        if not requested_paths and demonstration_coefficient(saved, now) > 0:
            raise ValueError("active demonstration resume requires explicit --demonstration_npz")
        if requested_paths and requested_paths != saved["npz_paths"]:
            raise ValueError("demonstration inputs changed across native resume")
        if config.demonstration_coef not in (0.0, saved["initial_coefficient"]):
            raise ValueError("demonstration coefficient changed across native resume")
        if config.demonstration_fade_transitions not in (0, saved["fade_transitions"]):
            raise ValueError("demonstration fade changed across native resume")
        if config.demonstration_batch_size != saved["batch_size"]:
            raise ValueError("demonstration batch size changed across native resume")
        state = copy.deepcopy(saved)
    else:
        if config.demonstration_coef <= 0 or config.demonstration_fade_transitions <= 0:
            raise ValueError("starting demonstrations requires positive coefficient and fade transitions")
        state = dict(origin_transitions=origin, origin_update=resume_update,
                     fade_transitions=config.demonstration_fade_transitions,
                     initial_coefficient=config.demonstration_coef, transitions_per_update=batch,
                     batch_size=config.demonstration_batch_size, npz_paths=requested_paths)
    config.demonstration_npz = list(state["npz_paths"])
    config.demonstration_coef = state["initial_coefficient"]
    config.demonstration_fade_transitions = state["fade_transitions"]
    return state


def load_demonstrations(paths, config, map_edge_length):
    """Read full episodes, explicit mixture weights and eligible supervision rows.

    ``group``, ``condition`` and ``source_id`` have one string per episode.
    ``group_names``/``group_weights`` declare the same mixture in every file.
    Foundation and trench retention rows use cached ``parent_probs[N, 8]``
    from the selected parent; expert and recovery rows use executed actions.
    ``supervision_mask[N]`` selects labels, never history.
    Physics and held-out identity exclusion are checked during collection.
    """
    episodes, observations, actions, histories, starts, lengths = [], [], [], [], [], []
    metadata = {key: [] for key in ("group", "condition", "source_id")}
    parent_probs, eligible_indices, eligible_starts, eligible_lengths = [], [], [], []
    mixture = None
    eligible_rows = 0
    rows = 0
    keys = None
    for path in paths:
        with np.load(path, allow_pickle=False) as bank:
            if "split" not in bank or bank["split"].shape != () or bank["split"].item() != "train":
                raise ValueError(f"demonstration input must explicitly declare split='train': {path}")
            required = set(metadata) | {"group_names", "group_weights", "supervision_mask", "parent_probs"}
            if missing := required - set(bank.files):
                raise ValueError(f"demonstrations missing grouped supervision fields: {sorted(missing)}")
            names, weights = bank["group_names"], bank["group_weights"]
            if (names.ndim != 1 or names.dtype.kind not in "US" or not names.size
                    or len(set(names.astype(str).tolist())) != len(names)
                    or not set(names.astype(str).tolist()) <= set(DEMONSTRATION_GROUPS)
                    or weights.shape != names.shape or weights.dtype.kind not in "fiu" or not np.isfinite(weights).all()
                    or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-6, rtol=0)):
                raise ValueError("explicit demonstration group weights must be positive and sum to one")
            file_mixture = dict(zip(names.astype(str).tolist(), map(float, weights)))
            if mixture is not None and mixture != file_mixture:
                raise ValueError("demonstration files must declare the same group weights")
            mixture = file_mixture
            action = bank["actions"]
            episode_id = bank["episode_id"]
            history = bank["previous_actions"]
            count = len(action)
            if (action.shape != (count,) or not count or action.dtype.kind not in "iu"
                    or np.any(action < 0) or np.any(action >= 8)):
                raise ValueError("demonstrations require nonempty tracked-excavator action IDs 0..7")
            if episode_id.shape != (count,) or episode_id.dtype.kind not in "iu":
                raise ValueError("demonstrations require one integer episode_id per action")
            if history.shape != (count, config.num_prev_actions) or history.dtype.kind not in "iu":
                raise ValueError("demonstration previous_actions width/type differs from policy history")
            boundaries = np.r_[0, np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1, count]
            ids = episode_id[boundaries[:-1]]
            if len(np.unique(ids)) != len(ids):
                raise ValueError("demonstration episodes must be contiguous")
            file_metadata = {key: bank[key] for key in metadata}
            for key, value in file_metadata.items():
                if value.shape != ids.shape or value.dtype.kind not in "US" or np.any(value.astype(str) == ""):
                    raise ValueError(f"demonstration {key} requires one nonempty string per episode")
                file_metadata[key] = value.astype(str)
                metadata[key].append(file_metadata[key])
            mask, probabilities = bank["supervision_mask"], bank["parent_probs"]
            if mask.shape != (count,) or mask.dtype.kind != "b":
                raise ValueError("demonstration supervision_mask must be boolean, one per action")
            if (probabilities.shape != (count, 8) or probabilities.dtype != np.float32
                    or not np.isfinite(probabilities).all() or np.any(probabilities < 0)):
                raise ValueError("demonstration parent_probs must be finite nonnegative float32 [N, 8]")
            for group in ("foundation", "trench"):
                retention_rows = np.repeat(file_metadata["group"] == group, np.diff(boundaries)) & mask
                if not np.allclose(probabilities[retention_rows].sum(axis=-1), 1., atol=1e-6, rtol=0):
                    raise ValueError(
                        f"eligible {group} rows require cached selected-parent probabilities "
                        "summing to one; rebuild legacy zero-target retention banks"
                    )
            parent_probs.append(probabilities)
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                expected = np.zeros(config.num_prev_actions, dtype=np.int32)
                for i in range(lo, hi):
                    if not np.array_equal(history[i], expected):
                        raise ValueError("demonstration history must be pre-action, newest action first")
                    expected = np.roll(expected, 1)
                    expected[0] = action[i]
                starts.append(rows + int(lo))
                lengths.append(int(hi - lo))
                eligible = rows + lo + np.flatnonzero(mask[lo:hi])
                if not len(eligible):
                    raise ValueError("every demonstration episode needs at least one eligible step")
                eligible_indices.append(eligible.astype(np.int32))
                eligible_starts.append(eligible_rows)
                eligible_lengths.append(len(eligible))
                eligible_rows += len(eligible)
            obs = {key[4:]: bank[key] for key in bank.files if key.startswith("obs/")}
            if not obs or (keys is not None and set(obs) != keys):
                raise ValueError("demonstration files must contain the same raw observation fields")
            keys = set(obs)
            for key, value in obs.items():
                if value.shape[:1] != (count,) or not np.isfinite(value).all():
                    raise ValueError(f"demonstration observation {key} has invalid rows or nonfinite values")
            if obs["action_map"].shape[1:] != (map_edge_length, map_edge_length):
                raise ValueError("demonstration map dimensions differ from the live environment")
            if not np.all(obs["num_agents"] == 1):
                raise ValueError("demonstrations currently support one tracked excavator")
            observations.append(obs)
            actions.append(action.astype(np.int32))
            histories.append(history.astype(np.int32))
            episodes.append(len(ids))
            rows += count
    if not observations:
        raise ValueError("no demonstration input files")
    metadata = {key: np.concatenate(value) for key, value in metadata.items()}
    if set(metadata["group"].tolist()) != set(mixture):
        raise ValueError("demonstration groups must exactly match the declared mixture; no rescaling")
    # Retention balances conditions first; hard corrections balance sources first
    # so extra disposal conditions cannot multiply a geometry's sampling mass.
    # Group weights are applied here only, never multiplied into the loss again.
    episode_probability = np.zeros(sum(episodes), dtype=np.float64)
    group_counts = {}
    for group, weight in mixture.items():
        group_mask = metadata["group"] == group
        conditions = np.unique(metadata["condition"][group_mask])
        first_key, second_key = (("source_id", "condition") if group in ("expert", "recovery")
                                 else ("condition", "source_id"))
        first_values = np.unique(metadata[first_key][group_mask])
        for first in first_values:
            first_mask = group_mask & (metadata[first_key] == first)
            second_values = np.unique(metadata[second_key][first_mask])
            for second in second_values:
                selected = first_mask & (metadata[second_key] == second)
                episode_probability[selected] = weight / len(first_values) / len(second_values) / selected.sum()
        group_counts[group] = dict(weight=weight, episodes=int(group_mask.sum()),
                                   conditions=len(conditions), sources=len(np.unique(metadata["source_id"][group_mask])),
                                   eligible_steps=int(np.asarray(eligible_lengths)[group_mask].sum()))
    data = dict(obs={key: np.concatenate([obs[key] for obs in observations]) for key in keys},
                actions=np.concatenate(actions), previous_actions=np.concatenate(histories),
                starts=np.asarray(starts, dtype=np.int32), lengths=np.asarray(lengths, dtype=np.int32),
                parent_probs=np.concatenate(parent_probs),
                group_ids=np.asarray([DEMONSTRATION_GROUPS.index(group) for group in metadata["group"]], np.int32),
                episode_log_probs=np.log(episode_probability).astype(np.float32),
                eligible_indices=np.concatenate(eligible_indices),
                eligible_starts=np.asarray(eligible_starts, dtype=np.int32),
                eligible_lengths=np.asarray(eligible_lengths, dtype=np.int32))
    # Exercise the exact current feature contract before tracing a training update.
    example = {key: jnp.asarray(value[:1]) for key, value in data["obs"].items()}
    obs_to_model_input(example, jnp.asarray(data["previous_actions"][:1]), config)
    return data, dict(episodes=sum(episodes), transitions=rows, episodes_per_file=episodes,
                      eligible_steps=eligible_rows, groups=group_counts)


def sample_demonstrations(data, rng, batch_size):
    """Sample the declared balanced mixture, then a uniform eligible step."""
    episode_key, step_key = jax.random.split(rng)
    episode = jax.random.categorical(episode_key, data["episode_log_probs"], shape=(batch_size,))
    step = jax.random.randint(step_key, (batch_size,), 0, data["eligible_lengths"][episode])
    index = data["eligible_indices"][data["eligible_starts"][episode] + step]
    return dict(obs=jax.tree_util.tree_map(lambda x: x[index], data["obs"]),
                actions=data["actions"][index], previous_actions=data["previous_actions"][index],
                group_id=data["group_ids"][episode], parent_probs=data["parent_probs"][index])
