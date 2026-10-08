"""Two frozen teachers, routed by the family of the pre-action episode.

This deliberately supports one tracked excavator at one shared resolution.
Teacher-only raw features never enter the student's observation list.
"""

import hashlib
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils.utils_ppo import obs_to_model_input


FAMILY_KEY = "teacher_episode_family_id"
LEGACY_DIG_KEY = "teacher_legacy_local_map_admissible_dig"
CACHED_LOGITS_KEY = "teacher_cached_logits"
CACHED_VALUE_KEY = "teacher_cached_value"
ELIGIBLE_KEY = "teacher_eligible"
PRECISION_KEY = "episode_precision_required"
CANDIDATE_KEY = "teacher_bulk_candidate"
IDENTITY_FIELDS = (
    "teacher_checkpoint_sha256",
    "trench_teacher_checkpoint_sha256",
    "task_teacher_family_ids",
)


def option(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def validate_recurrent_teacher_mode(config):
    if option(config, "teacher_bulk_compatibility", False):
        if not option(config, "recurrent_teacher", False):
            raise ValueError("teacher_bulk_compatibility requires recurrent_teacher")
        if option(config, "warm_start_from") is not None:
            raise ValueError("teacher_bulk_compatibility starts a scratch student; omit warm_start_from")
    if not option(config, "recurrent_teacher", False):
        return
    if option(config, "teacher_checkpoint") is None:
        raise ValueError("recurrent_teacher requires teacher_checkpoint")
    if option(config, "actor_core", "mlp") != "gru":
        raise ValueError("recurrent_teacher requires a GRU student")
    if option(config, "trench_teacher_checkpoint") is not None:
        raise ValueError("recurrent_teacher cannot be combined with dual task teachers")
    if float(option(config, "kickstart_value_coef", 0.5)) != 0.0:
        raise ValueError("recurrent_teacher requires kickstart_value_coef=0 (policy KL only)")
    if option(config, "cache_teacher_outputs", False):
        raise ValueError("recurrent_teacher caches each rollout step; omit cache_teacher_outputs")
    if int(option(config, "teacher_obs_downsample", 1)) != 1:
        raise ValueError("recurrent_teacher requires teacher_obs_downsample=1")
    if option(config, "action_logit_masking", False):
        raise ValueError("recurrent_teacher requires an unmasked policy")


def validate_bulk_teacher_environment(teacher_env_cfg, student_env_cfg, teacher_config):
    """The legacy view changes dig rules, never the physical machine or dump.

    Both configs must describe the effective reset-time geometry. In particular,
    the student's width, height and tile size must already have been resolved.
    """
    if teacher_env_cfg is None or np.asarray(teacher_env_cfg.tile_size).ndim != 0:
        raise ValueError("bulk teacher requires a saved single-environment EnvConfig")
    if bool(teacher_env_cfg.pull_direction_alignment):
        raise ValueError("bulk teacher must use the legacy pull_direction_alignment=False rules")
    for cfg in (teacher_env_cfg, student_env_cfg):
        for field in ("agent_types", "action_types"):
            raw = getattr(cfg, field)
            values = np.asarray(raw)
            count = len(raw) if isinstance(raw, (tuple, list)) else values.shape[-1]
            if count != 1 or not np.all(values == 0):
                raise ValueError("bulk teacher supports exactly one tracked excavator")
    if not np.all(np.asarray(student_env_cfg.pull_direction_alignment)):
        raise ValueError("bulk compatibility requires the student's pull-direction rules")
    for field in teacher_env_cfg.agent._fields:
        if field == "random_init_state":
            continue
        if not np.allclose(np.asarray(getattr(teacher_env_cfg.agent, field)),
                           np.asarray(getattr(student_env_cfg.agent, field)), rtol=1e-6, atol=1e-7):
            raise ValueError(f"bulk teacher/student machine setting differs: agent.{field}")
    for field in ("tile_size", "foundation_dump_min_free_fraction"):
        if not np.allclose(np.asarray(getattr(teacher_env_cfg, field)),
                           np.asarray(getattr(student_env_cfg, field)), rtol=1e-6, atol=1e-7):
            raise ValueError(f"bulk teacher/student setting differs: {field}")
    for field in teacher_env_cfg.maps._fields:
        if not np.allclose(np.asarray(getattr(teacher_env_cfg.maps, field)),
                           np.asarray(getattr(student_env_cfg.maps, field)), rtol=1e-6, atol=1e-7):
            raise ValueError(f"bulk teacher/student map geometry differs: maps.{field}")
    for field in ("movement_feasibility_observation", "previous_outcome_observation", "action_logit_masking"):
        if option(teacher_config, field, False):
            raise ValueError(f"bulk teacher legacy observation view does not support {field}")


def legacy_teacher_state_view(state, teacher_env_cfg, legacy_foundation_border_axes):
    """Same live terrain/pose/history with the teacher's saved rules and axes."""
    return state._replace(
        env_cfg=teacher_env_cfg,
        world=state.world._replace(foundation_border_axes=legacy_foundation_border_axes),
    )


def bulk_teacher_do_compatible(state, legacy_state):
    """Immediate DO agreement, not future workspace or navigation competence.

    Loaded DO is identical because startup checks all machine/dump settings.
    Empty DO must select exactly the same cells, volume, relift and admission;
    two allowed digs with different selected cells are deliberately excluded.
    """
    def compare_dig():
        current = state._dig_eligibility(state._build_dig_dump_cone())
        legacy = legacy_state._dig_eligibility(legacy_state._build_dig_dump_cone())
        return jnp.all(jnp.stack([jnp.all(a == b) for a, b in zip(current, legacy)]))

    return jax.lax.cond(
        state._get_current_agent_state().loaded[0] > 0,
        lambda: jnp.bool_(True), compare_dig,
    )


def bulk_teacher_observation_and_compatibility(
    state, teacher_env_cfg, legacy_foundation_border_axes, *, executable_dig_observation,
):
    """Single-lane pre-action teacher view; vmap this over the rollout lanes."""
    from terra.env import TerraEnv
    from terra.wrappers import LocalMapWrapper

    legacy_state = legacy_teacher_state_view(state, teacher_env_cfg, legacy_foundation_border_axes)
    compatible = bulk_teacher_do_compatible(state, legacy_state)
    legacy_state = LocalMapWrapper.wrap(
        legacy_state, executable_dig_observation=executable_dig_observation,
    )
    return TerraEnv._state_to_obs_dict(legacy_state), compatible


def recurrent_teacher_rollout_observation(
    observation, prev_actions, hidden, apply_fn, params, teacher_config,
    precision_required, slot_id, eligible_slots, *, teacher_observation=None,
    bulk_compatible=None,
):
    """Advance the frozen teacher on the current student-state observation.

    This is called before the student's action and before any native auto-reset.
    The teacher uses its own preprocessing and its own recurrent carry. Cached
    outputs and the pre-action eligibility label follow the PPO sequence shuffle.
    """
    teacher_input = obs_to_model_input(
        observation if teacher_observation is None else teacher_observation,
        prev_actions, teacher_config,
    )
    value, logits, next_hidden = apply_fn(
        params, teacher_input, hidden, method="actor_step",
    )
    eligible = (~jnp.asarray(precision_required, dtype=jnp.bool_)) & jnp.any(
        jnp.asarray(slot_id)[..., None] == jnp.asarray(eligible_slots), axis=-1,
    )
    result = dict(observation)
    if bulk_compatible is not None:
        result[CANDIDATE_KEY] = jax.lax.stop_gradient(eligible)
        eligible &= jnp.asarray(bulk_compatible, dtype=jnp.bool_)
    result[CACHED_VALUE_KEY] = jax.lax.stop_gradient(value.astype(jnp.float32))
    result[CACHED_LOGITS_KEY] = jax.lax.stop_gradient(logits.astype(jnp.float32))
    result[ELIGIBLE_KEY] = jax.lax.stop_gradient(eligible)
    return result, jax.lax.stop_gradient(next_hidden)


def reset_recurrent_teacher_hidden(hidden, done):
    return jnp.where(jnp.asarray(done)[..., None], jnp.zeros_like(hidden), hidden)


def masked_teacher_kl(per_row_kl, eligible, *, axis_name=None):
    """Mean over selected rows, including unequal exposure across devices.

    PPO pmeans gradients after this loss. Multiplying each local numerator by
    the device count makes that pmean equal the pooled selected-row gradient.
    Globally empty selections have exactly zero loss and gradient.
    """
    numerator = jnp.sum(jnp.where(eligible, per_row_kl, jnp.zeros_like(per_row_kl)))
    count = jnp.sum(jnp.asarray(eligible, dtype=jnp.float32))
    if axis_name is not None:
        count = jax.lax.psum(count, axis_name)
        numerator = numerator * jax.lax.psum(jnp.float32(1), axis_name)
    return numerator / jnp.maximum(count, 1)


def validate_task_teacher_mode(config):
    if option(config, "trench_teacher_checkpoint") is None:
        return
    if option(config, "teacher_checkpoint") is None:
        raise ValueError("trench_teacher_checkpoint requires the foundation teacher_checkpoint")
    # Teachers stay feed-forward; a GRU student is scored row by row against them.
    actor_core = option(config, "actor_core", "mlp")
    if actor_core not in ("mlp", "gru"):
        raise ValueError("task teachers require an mlp or gru student actor")
    if actor_core == "gru" and option(config, "cache_teacher_outputs", False):
        raise ValueError("a gru student scores teachers per minibatch; disable cache_teacher_outputs")
    if int(option(config, "teacher_obs_downsample", 1)) != 1:
        raise ValueError("task teachers require teacher_obs_downsample=1")
    if float(option(config, "kickstart_value_coef", 0.5)) != 0.0:
        raise ValueError("task teachers require kickstart_value_coef=0 (policy KL only)")
    if option(config, "action_logit_masking", False):
        raise ValueError("task teachers do not support action_logit_masking")
    if not option(config, "executable_dig_observation", False):
        raise ValueError("task teachers require the student's executable_dig_observation")
    if not option(config, "admissible_dig_observation", False):
        raise ValueError("task teachers require admissible_dig_observation")
    for field in ("agent_types_override", "action_types_override"):
        value = option(config, field)
        if value is not None and tuple(value) != (0,):
            raise ValueError(f"task teachers require {field}=(0,)")


def bind_task_teacher_checkpoints(config):
    """Bind file contents, allowing native continuation at relocated paths."""
    validate_task_teacher_mode(config)
    if option(config, "trench_teacher_checkpoint") is None:
        return
    for path_field, hash_field in (
        ("teacher_checkpoint", "teacher_checkpoint_sha256"),
        ("trench_teacher_checkpoint", "trench_teacher_checkpoint_sha256"),
    ):
        digest = hashlib.sha256()
        with Path(option(config, path_field)).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        expected = option(config, hash_field)
        if expected is not None and expected != actual:
            raise ValueError(f"task teacher bytes differ from saved {hash_field}")
        setattr(config, hash_field, actual)


def load_task_teacher_checkpoint(path, expected_sha256):
    """Check exactly the bytes deserialized, not a previous read of the path."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("task teacher bytes changed after checkpoint binding")
    return pickle.loads(data)


def validate_task_teacher_resume(checkpoint, config, *, require_families=False):
    saved = checkpoint.get("train_config", {})
    saved_dual = option(saved, "trench_teacher_checkpoint") is not None
    current_dual = option(config, "trench_teacher_checkpoint") is not None
    if not saved_dual and not current_dual:
        return
    # Adding teachers to a teacher-free parent is an existing supported start.
    if not saved_dual and option(saved, "teacher_checkpoint") is None:
        return
    if saved_dual != current_dual:
        raise ValueError("native resume must retain single/dual task teacher mode")
    for field in IDENTITY_FIELDS:
        expected, actual = option(saved, field), option(config, field)
        if field == "task_teacher_family_ids" and actual is None and not require_families:
            continue
        if expected is None or actual != expected:
            raise ValueError(f"task teacher native resume must retain {field}")


def resolve_task_teacher_families(family_names, used_family_ids):
    names = tuple(family_names)
    if len(set(names)) != len(names):
        raise ValueError("task teacher family names must be unique")
    if "foundation" not in names or "trench" not in names:
        raise ValueError("task teachers require foundation and trench provenance")
    route = {name: names.index(name) for name in ("foundation", "trench")}
    used = set(np.asarray(used_family_ids, dtype=np.int64).reshape(-1).tolist())
    if used != set(route.values()):
        raise ValueError(f"task teacher map provenance is unknown or incomplete: {sorted(used)}")
    return route


def legacy_teacher_admissible_dig(state):
    from terra.wrappers import LocalMapWrapper

    return LocalMapWrapper.wrap(
        state, executable_dig_observation=False,
    ).world.local_map_admissible_dig.map


def task_teacher_rollout_observation(observation, state, active_family_id):
    """Called before step/reset; all returned leaves follow the PPO shuffle."""
    result = dict(observation)
    result[FAMILY_KEY] = jnp.asarray(active_family_id, dtype=jnp.int32)
    result[LEGACY_DIG_KEY] = jax.vmap(legacy_teacher_admissible_dig)(state)
    return result


def cache_task_teacher_outputs(observation, prev_actions, apply_fn, params, num_minibatches):
    """Evaluate frozen teachers once on pre-action rows, before PPO shuffling.

    Input/output use [time, env, ...]. Inference uses the same flattened batch
    size as PPO, avoiding a rollout-sized convolution and its memory cost.
    The resulting leaves follow exactly the ordinary observation shuffle.
    """
    steps, envs = prev_actions.shape[:2]
    if (steps * envs) % num_minibatches:
        raise ValueError("teacher cache requires complete PPO minibatches")

    def chunks(x):
        return x.swapaxes(0, 1).reshape((num_minibatches, -1) + x.shape[2:])

    rows = jax.tree_util.tree_map(chunks, (observation, prev_actions))
    values, logits = jax.lax.map(lambda batch: apply_fn(params, *batch), rows)

    def restore(x):
        x = x.reshape((envs, steps) + x.shape[2:]).swapaxes(0, 1)
        return jax.lax.stop_gradient(x.astype(jnp.float32))

    result = dict(observation)
    result[CACHED_VALUE_KEY] = restore(values)
    result[CACHED_LOGITS_KEY] = restore(logits)
    # No later teacher call needs these relatively large native-only features.
    result.pop(LEGACY_DIG_KEY, None)
    return result


def validate_task_teacher_configs(student_config, foundation_config, trench_config):
    validate_task_teacher_mode(student_config)
    for role, config in (("foundation", foundation_config), ("trench", trench_config)):
        if option(config, "actor_core", "mlp") != "mlp":
            raise ValueError(f"{role} teacher must use actor_core='mlp'")
        if option(config, "action_logit_masking", False):
            raise ValueError(f"{role} teacher action_logit_masking is unsupported")
        if option(config, "num_prev_actions") != option(student_config, "num_prev_actions"):
            raise ValueError(f"{role} teacher num_prev_actions mismatch")
        for field in ("agent_types_override", "action_types_override"):
            value = option(config, field)
            if value is not None and tuple(value) != (0,):
                raise ValueError(f"{role} teacher requires {field}=(0,)")


def native_task_teacher_obs(raw_obs, prev_actions, teacher_config):
    # Reconstruct the teacher's semantics BEFORE its own clipping/area scaling
    # and optional-input ordering. No hard-coded model-list feature index.
    obs = dict(raw_obs)
    if (option(teacher_config, "admissible_dig_observation", False)
            and not option(teacher_config, "executable_dig_observation", False)):
        obs["local_map_admissible_dig"] = raw_obs[LEGACY_DIG_KEY]
    # Earlier teachers use nine local maps, plus carry credit in agent_states
    # and the environment's latched reset context. Those raw values are always
    # exported by Terra, even when the student model does not consume them.
    return obs_to_model_input(obs, prev_actions, teacher_config)


def make_task_teacher_apply_fn(foundation_apply_fn, trench_apply_fn,
                               foundation_config, trench_config, family_ids, shared=False):
    """``shared``: both roles bind the same checkpoint bytes, so evaluate it once."""
    def apply_fn(params, raw_obs, prev_actions):
        params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
        foundation_value, foundation_logits = foundation_apply_fn(
            params["foundation"], native_task_teacher_obs(raw_obs, prev_actions, foundation_config),
        )
        if shared:
            trench_value, trench_logits = foundation_value, foundation_logits
        else:
            trench_value, trench_logits = trench_apply_fn(
                params["trench"], native_task_teacher_obs(raw_obs, prev_actions, trench_config),
            )
        if foundation_logits.shape != trench_logits.shape or foundation_logits.shape[-1] != 8:
            raise ValueError("task teachers must share the eight-action tracked interface")
        family = raw_obs[FAMILY_KEY]
        if family.shape != foundation_logits.shape[:-1]:
            raise ValueError("task teacher family labels must match the PPO sample axes")
        foundation = family == family_ids["foundation"]
        trench = family == family_ids["trench"]
        # Unknown provenance also fails the existing finite-diagnostic checks if
        # it somehow reaches a traced rollout despite the host map validation.
        value = jnp.where(foundation[..., None], foundation_value,
                          jnp.where(trench[..., None], trench_value, jnp.nan))
        logits = jnp.where(foundation[..., None], foundation_logits,
                           jnp.where(trench[..., None], trench_logits, jnp.nan))
        return jax.lax.stop_gradient(value), jax.lax.stop_gradient(logits)
    return apply_fn


def task_teacher_kl_stats(per_row_kl, family, family_ids):
    """Unnormalized sums/counts; reduce before forming conditional KL means."""
    result = {}
    for name in ("foundation", "trench"):
        mask = family == family_ids[name]
        result[f"kickstart/{name}_kl_sum"] = jnp.where(mask, per_row_kl, 0.0).sum()
        result[f"kickstart/{name}_selected_count"] = mask.astype(jnp.float32).sum()
    return result


def finalize_task_teacher_metrics(loss_info, num_minibatches):
    """Convert epoch/minibatch-averaged sums to conditional means and counts."""
    for family in ("foundation", "trench"):
        count_key = f"kickstart/{family}_selected_count"
        count = loss_info[count_key]
        loss_info[f"kickstart/{family}_kl"] = (
            loss_info[f"kickstart/{family}_kl_sum"] / jnp.where(count > 0, count, 1.0)
        )
        # The epoch mean removes repeated PPO exposure. psum in the loss
        # already includes every device; recover one count per rollout sample.
        loss_info[count_key] = count * num_minibatches
    return loss_info


def clear_task_teacher_config(config):
    for field in ("trench_teacher_checkpoint", *IDENTITY_FIELDS):
        if isinstance(config, dict):
            config[field] = None
        else:
            setattr(config, field, None)
    if isinstance(config, dict):
        config["foundation_teacher_release_updates"] = 0
        config["cache_teacher_outputs"] = False
    else:
        config.foundation_teacher_release_updates = 0
        config.cache_teacher_outputs = False
