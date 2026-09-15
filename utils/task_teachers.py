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
IDENTITY_FIELDS = (
    "teacher_checkpoint_sha256",
    "trench_teacher_checkpoint_sha256",
    "task_teacher_family_ids",
)


def option(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def validate_task_teacher_mode(config):
    if option(config, "trench_teacher_checkpoint") is None:
        return
    if option(config, "teacher_checkpoint") is None:
        raise ValueError("trench_teacher_checkpoint requires the foundation teacher_checkpoint")
    if option(config, "actor_core", "mlp") != "mlp":
        raise ValueError("task teachers require actor_core='mlp'")
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


def validate_task_teacher_configs(student_config, foundation_config, trench_config):
    validate_task_teacher_mode(student_config)
    for role, config in (("foundation", foundation_config), ("trench", trench_config)):
        if option(config, "actor_core", "mlp") != "mlp":
            raise ValueError(f"{role} teacher must use actor_core='mlp'")
        if option(config, "action_logit_masking", False):
            raise ValueError(f"{role} teacher action_logit_masking is unsupported")
        if not option(config, "admissible_dig_observation", False):
            raise ValueError(f"{role} teacher must have admissible_dig_observation")
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
    if not option(teacher_config, "executable_dig_observation", False):
        obs["local_map_admissible_dig"] = raw_obs[LEGACY_DIG_KEY]
    return obs_to_model_input(obs, prev_actions, teacher_config)


def make_task_teacher_apply_fn(foundation_apply_fn, trench_apply_fn,
                               foundation_config, trench_config, family_ids):
    def apply_fn(params, raw_obs, prev_actions):
        params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
        foundation_value, foundation_logits = foundation_apply_fn(
            params["foundation"], native_task_teacher_obs(raw_obs, prev_actions, foundation_config),
        )
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
