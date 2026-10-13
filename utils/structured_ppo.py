"""Masked hierarchical PPO for Terra structured-v1 actions.

An action has the legacy type ID and only the argument that type consumes.
Masks are captured at sampling and replayed unchanged during optimization.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
from terra.structured_actions import StructuredAction, StructuredClock
from utils.utils_ppo import obs_to_model_input


ACTION_PROTOCOL = "terra_structured_v1"


class Runner(NamedTuple):
    """Stable pickle module path, shared by training and checkpoint tools."""
    state: object
    elapsed_s: jax.Array
    clock: StructuredClock
    previous_types: jax.Array
    previous_arguments: jax.Array
    previous_outcome: jax.Array
    hidden: jax.Array
    rng: jax.Array


def _validate_interface(action_interface):
    if action_interface not in ("structured", "primitive8"):
        raise ValueError(f"unknown action interface: {action_interface}")


def policy_masks(masks, allow_wait=False, *, action_interface="structured", current_heading=None):
    """Exact executable types for the selected interface.

    Primitive DO uses the current chassis-relative cabin index. Keep the full
    native argument masks unchanged: loaded-dead-end detection needs all twelve
    unload headings, even if the current heading cannot unload.
    """
    _validate_interface(action_interface)
    action = jnp.asarray(masks["action_mask"], bool)
    if action_interface == "primitive8":
        if current_heading is None:
            raise ValueError("primitive8 masks require current_heading")
        current_heading = jnp.asarray(current_heading, jnp.int32)
        if current_heading.shape != action.shape[:-1]:
            raise ValueError("current_heading must match the action mask batch shape")
        move = masks["move_mask"][..., :, 4]
        turn = masks["turn_mask"][..., :, 0]
        work = jnp.take_along_axis(masks["do_mask"], current_heading[..., None], -1)[..., 0]
    else:
        action = action.at[..., 4:6].set(False)
        move, turn, work = (masks[name].any(-1) for name in ("move_mask", "turn_mask", "do_mask"))
    action = action.at[..., :2].set(action[..., :2] & move)
    action = action.at[..., 2:4].set(action[..., 2:4] & turn)
    action = action.at[..., 6].set(action[..., 6] & work)
    action = action.at[..., 7].set(allow_wait | ~action[..., :7].any(-1))
    return {**masks, "action_mask": action}


def _log_probs(logits, mask):
    # Unused rows can be empty (e.g. all moves while loaded). Give them a
    # deterministic dummy argument so entropy and gradients stay finite.
    fallback = jax.nn.one_hot(jnp.zeros(mask.shape[:-1], jnp.int32), mask.shape[-1], dtype=bool)
    mask = jnp.where(mask.any(-1, keepdims=True), mask, fallback)
    return jax.nn.log_softmax(jnp.where(mask, logits, -1e9), axis=-1)


def distributions(logits, masks, *, action_interface="structured"):
    _validate_interface(action_interface)
    result = {"action": _log_probs(logits["action"], masks["action_mask"])}
    if action_interface == "primitive8":
        return result
    return {**result,
        "move": _log_probs(logits["move"], masks["move_mask"]),
        "turn": _log_probs(logits["turn"], masks["turn_mask"]),
        "do": _log_probs(logits["do"], masks["do_mask"]),
    }


def _take(log_probs, index):
    return jnp.take_along_axis(log_probs, index[..., None], axis=-1)[..., 0]


def joint_log_prob(logits, masks, action, *, action_interface="structured"):
    lp = distributions(logits, masks, action_interface=action_interface)
    typ = jnp.asarray(action.action, jnp.int32)
    if action_interface == "primitive8":
        return _take(lp["action"], typ)
    amount = jnp.asarray(action.amount, jnp.int32) - 1
    move_row = jnp.take_along_axis(lp["move"], jnp.clip(typ, 0, 1)[..., None, None], -2)[..., 0, :]
    turn_row = jnp.take_along_axis(lp["turn"], jnp.clip(typ - 2, 0, 1)[..., None, None], -2)[..., 0, :]
    move_lp = _take(move_row, jnp.clip(amount, 0, 4))
    turn_lp = _take(turn_row, jnp.clip(amount, 0, 5))
    do_lp = _take(lp["do"], jnp.clip(action.heading, 0, 11))
    return (_take(lp["action"], typ) + jnp.where(typ < 2, move_lp, 0.)
            + jnp.where((typ >= 2) & (typ < 4), turn_lp, 0.)
            + jnp.where(typ == 6, do_lp, 0.))


def entropy_components(logits, masks, *, action_interface="structured"):
    """Exact joint entropy components, averaged over the chosen type law."""
    lp = distributions(logits, masks, action_interface=action_interface)
    entropy = lambda value: -(jnp.exp(value) * value).sum(-1)
    if action_interface == "primitive8":
        action = entropy(lp["action"])
        return {"action": action, **{name: jnp.zeros_like(action) for name in ("move", "turn", "do")}}
    probs = jnp.exp(lp["action"])
    return {
        "action": entropy(lp["action"]),
        "move": (probs[..., :2] * entropy(lp["move"])).sum(-1),
        "turn": (probs[..., 2:4] * entropy(lp["turn"])).sum(-1),
        "do": probs[..., 6] * entropy(lp["do"]),
    }


def sample_action(logits, masks, rng, deterministic=False, *, action_interface="structured"):
    lp = distributions(logits, masks, action_interface=action_interface)
    keys = jax.random.split(rng, 4)
    choose = lambda values, key: (jnp.argmax(values, -1) if deterministic
                                  else jax.random.categorical(key, values, axis=-1))
    typ = choose(lp["action"], keys[0])
    if action_interface == "primitive8":
        action = StructuredAction(typ.astype(jnp.int32), jnp.where(typ < 2, 5, 1).astype(jnp.int32),
                                  jnp.full_like(typ, -1, dtype=jnp.int32))
        return action, _take(lp["action"], typ)
    move_row = jnp.take_along_axis(lp["move"], jnp.clip(typ, 0, 1)[..., None, None], -2)[..., 0, :]
    turn_row = jnp.take_along_axis(lp["turn"], jnp.clip(typ - 2, 0, 1)[..., None, None], -2)[..., 0, :]
    amount = jnp.where(typ < 2, choose(move_row, keys[1]) + 1,
                      jnp.where((typ >= 2) & (typ < 4), choose(turn_row, keys[2]) + 1, 1))
    heading = jnp.where(typ == 6, choose(lp["do"], keys[3]), -1)
    action = StructuredAction(typ.astype(jnp.int32), amount.astype(jnp.int32), heading.astype(jnp.int32))
    return action, joint_log_prob(logits, masks, action)


def policy_diagnostics(logits, masks, *, action_interface="structured", axis_name=None):
    """Policy-weighted conditional head statistics, not gradient attribution.

    Argument statistics condition on that head being selected; direction rows
    are weighted by their action-type probability. Normalized entropy is zero
    for zero/one valid choice. Primitive argument heads are inactive and zero.
    Under pmap, pool weighted sums before dividing, including absent heads.
    """
    lp = distributions(logits, masks, action_interface=action_interface)
    type_prob = jnp.exp(lp["action"])
    samples = jnp.asarray(type_prob[..., 0].size, jnp.float32)
    heads = (("action", 8, 0), ("move", 5, 1), ("turn", 6, 1), ("do", 12, 0))
    sums = [samples[None]]
    for name, width, _ in heads:
        if action_interface == "primitive8" and name != "action":
            sums.append(jnp.zeros(4 + width, jnp.float32))
            continue
        weight = {"action": jnp.ones_like(type_prob[..., 0]), "move": type_prob[..., :2],
                  "turn": type_prob[..., 2:4], "do": type_prob[..., 6]}[name]
        probability = jnp.exp(lp[name])
        count = masks[f"{name}_mask"].sum(-1)
        entropy = -(probability * lp[name]).sum(-1)
        normalized = jnp.where(count > 1, entropy / jnp.log(jnp.maximum(count, 2)), 0.)
        sums.append(jnp.concatenate((jnp.stack((weight.sum(), (weight * entropy).sum(),
            (weight * normalized).sum(), (weight * count).sum())),
            (weight[..., None] * probability).reshape((-1, width)).sum(0))))
    # One packed collective per minibatch, not a small all-reduce per metric.
    pooled = jnp.concatenate(sums)
    if axis_name is not None:
        pooled = jax.lax.psum(pooled, axis_name)
    metrics, offset = {}, 1
    for name, width, first in heads:
        prefix = f"head_{name}_"
        mass = pooled[offset]
        denominator = jnp.maximum(mass, 1e-8)
        metrics[prefix + "mass"] = mass
        metrics[prefix + "probability"] = mass / pooled[0]
        for index, suffix in enumerate(("conditional_entropy", "normalized_entropy", "valid_count"), 1):
            metrics[prefix + suffix] = pooled[offset + index] / denominator
        probabilities = pooled[offset + 4:offset + 4 + width] / denominator
        metrics.update({prefix + f"prob_{i + first}": probabilities[i] for i in range(width)})
        offset += 4 + width
    return metrics


def head_gradient_norms(grads, actor_core):
    """Norms of output-head subtrees, after device averaging, before Optax.

    These use the existing total-loss gradient (including entropy), not a
    second backward pass or a decomposition of shared encoder gradients.
    """
    params = grads["params"]
    type_name = "actor_post_gru" if actor_core == "gru" else "mlp_pi"
    groups = {"action": params[type_name], **params["structured_heads"]}
    return {f"head_{name}_grad_norm": jnp.sqrt(sum(jnp.sum(value.astype(jnp.float32) ** 2)
                for value in jax.tree.leaves(tree))) for name, tree in groups.items()}


@struct.dataclass
class StructuredRollout:
    # PPO minibatches are whole recurrent trajectories [batch, time, ...].
    obs: dict
    masks: dict
    prev_actions: jax.Array
    action: StructuredAction
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    duration_s: jax.Array
    done: jax.Array
    task_done: jax.Array


def duration_gae(rollout, last_value, gamma=1.0, gae_lambda=0.95, reference_s=30.):
    """Semi-Markov GAE; gamma AND lambda are per reference_s seconds.

    Time budgets and the decision guard are true episode boundaries in this
    finite-horizon objective, so none of them bootstrap across reset.
    """
    discounts = jnp.power(gamma, rollout.duration_s / reference_s)
    trace_discounts = jnp.power(gae_lambda, rollout.duration_s / reference_s)

    def step(carry, values):
        advantage, next_value = carry
        reward, value, done, discount, trace_discount = values
        live = 1. - done.astype(jnp.float32)
        delta = reward + discount * live * next_value - value
        advantage = delta + discount * trace_discount * live * advantage
        return (advantage, value), advantage

    arrays = tuple(x.swapaxes(0, 1) for x in (
        rollout.reward, rollout.value, rollout.done, discounts, trace_discounts))
    _, advantages = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value), arrays, reverse=True)
    advantages = advantages.swapaxes(0, 1)
    return advantages, advantages + rollout.value


def forward_step(model, params, observation, previous_actions, hidden, config):
    inputs = obs_to_model_input(observation, previous_actions, config)
    if config["actor_core"] == "gru":
        value, logits, next_hidden = model.apply(params, inputs, hidden, method="actor_step")
    else:
        value, logits = model.apply(params, inputs)
        next_hidden = hidden
    return value[..., 0], logits, next_hidden


def _masked_moments(values, mask, axis_name=None):
    """Global sample moments; an absent group has count and moments zero."""
    count = mask.sum(dtype=jnp.float32)
    total = jnp.where(mask, values, 0.).sum()
    absolute = jnp.where(mask, jnp.abs(values), 0.).sum()
    if axis_name is not None:
        count, total, absolute = jax.lax.psum((count, total, absolute), axis_name)
    denominator = jnp.maximum(count, 1.)
    mean = total / denominator
    squared = jnp.where(mask, (values - mean) ** 2, 0.).sum()
    if axis_name is not None:
        squared = jax.lax.psum(squared, axis_name)
    return count, mean, jnp.sqrt(squared / denominator), absolute / denominator


def normalize_advantages(advantages, precision_mode=None, *, mode="global", axis_name=None):
    """Normalize actor coefficients, preserving raw critic targets at the caller.

    These are advantage statistics, not measurements of policy gradients.
    Mode masks describe each transition, including across episode resets.
    """
    if mode not in ("global", "per_mode"):
        raise ValueError(f"unknown advantage normalization: {mode}")
    if precision_mode is None and mode == "per_mode":
        raise ValueError("per_mode advantage normalization requires precision_mode")
    if precision_mode is not None and precision_mode.shape != advantages.shape:
        raise ValueError("precision_mode must have the same [batch, time] shape as advantages")
    masks = ({} if precision_mode is None else
             {"bulk": ~precision_mode.astype(bool), "precision": precision_mode.astype(bool)})
    moments = {name: _masked_moments(advantages, mask, axis_name) for name, mask in masks.items()}
    if mode == "global":
        if axis_name is None:
            mean, std = advantages.mean(), advantages.std()
        else:
            mean = jax.lax.pmean(advantages.mean(), axis_name)
            std = jnp.sqrt(jax.lax.pmean(((advantages - mean) ** 2).mean(), axis_name))
    else:
        mean = jnp.where(masks["precision"], moments["precision"][1], moments["bulk"][1])
        std = jnp.where(masks["precision"], moments["precision"][2], moments["bulk"][2])
    normalized = jnp.where(std > 1e-8, (advantages - mean) / (std + 1e-8), 0.)
    metrics = {}
    for name, mask in masks.items():
        count, mean, std, absolute = moments[name]
        _, norm_mean, norm_std, norm_absolute = _masked_moments(normalized, mask, axis_name)
        metrics.update({f"advantage_{name}_{key}": value for key, value in (
            ("count", count), ("raw_mean", mean), ("raw_std", std), ("raw_abs_mean", absolute),
            ("normalized_mean", norm_mean), ("normalized_std", norm_std),
            ("normalized_abs_mean", norm_absolute))})
    return normalized, metrics


def ppo_update(train_state, model, rollout, advantages, targets, initial_hidden,
               config, *, clip_eps=0.2, vf_coef=2., entropy_coefs=(0.01, 0.01, 0.01, 0.01),
               value_clip=True, axis_name=None):
    """One minibatch; the probability ratio uses the full sampled action.

    Under pmap, normalization uses global minibatch moments, optionally per
    lane mode. Gradients and metrics are averaged across devices.
    """
    precision_mode = rollout.obs.get("precision_mode")
    action_interface = config.get("action_interface", "structured")
    _validate_interface(action_interface)
    norm_adv, advantage_metrics = normalize_advantages(
        advantages, precision_mode, mode=config.get("advantage_normalization", "global"),
        axis_name=axis_name)

    def loss(params):
        inputs = obs_to_model_input(rollout.obs, rollout.prev_actions, config)
        if config["actor_core"] == "gru":
            values, logits, _ = model.apply(params, inputs, initial_hidden, rollout.done, method="actor_sequence")
        else:
            shape = rollout.done.shape
            inputs = [x.reshape((-1,) + x.shape[2:]) for x in inputs]
            values, logits = model.apply(params, inputs)
            values = values.reshape(shape + (1,))
            logits = jax.tree.map(lambda x: x.reshape(shape + x.shape[1:]), logits)
        values = values[..., 0]
        log_prob = joint_log_prob(logits, rollout.masks, rollout.action, action_interface=action_interface)
        ratio = jnp.exp(log_prob - rollout.log_prob)
        actor_loss = -jnp.minimum(ratio * norm_adv, jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * norm_adv).mean()
        if value_clip:
            clipped_value = rollout.value + jnp.clip(values - rollout.value, -clip_eps, clip_eps)
            value_loss = 0.5 * jnp.maximum((values-targets)**2, (clipped_value-targets)**2).mean()
        else:
            value_loss = 0.5 * ((values-targets)**2).mean()
        components = entropy_components(logits, rollout.masks, action_interface=action_interface)
        entropy_bonus = sum(coef * components[name].mean() for coef, name in zip(entropy_coefs, ("action", "move", "turn", "do")))
        total = actor_loss + vf_coef * value_loss - entropy_bonus
        metrics = {
            "total_loss": total, "actor_loss": actor_loss, "value_loss": value_loss,
            "entropy": sum(x.mean() for x in components.values()),
            "approx_kl": ((ratio - 1.) - (log_prob - rollout.log_prob)).mean(),
            "clip_fraction": (jnp.abs(ratio-1.) > clip_eps).mean(),
            **{f"entropy_{k}": v.mean() for k, v in components.items()},
            **advantage_metrics,
            **policy_diagnostics(logits, rollout.masks, action_interface=action_interface, axis_name=axis_name),
        }
        if precision_mode is not None:
            clipped = jnp.abs(ratio - 1.) > clip_eps
            for name, mask in (("bulk", ~precision_mode.astype(bool)),
                               ("precision", precision_mode.astype(bool))):
                count = advantage_metrics[f"advantage_{name}_count"]
                clipped_count = (clipped & mask).sum(dtype=jnp.float32)
                if axis_name is not None:
                    clipped_count = jax.lax.psum(clipped_count, axis_name)
                metrics[f"clip_fraction_{name}"] = clipped_count / jnp.maximum(count, 1.)
        return total, metrics
    (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(train_state.params)
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name)
        # Head diagnostics have already pooled their numerators and masses.
        heads = {key: value for key, value in metrics.items() if key.startswith("head_")}
        metrics = jax.lax.pmean({key: value for key, value in metrics.items()
                                if not key.startswith("head_")}, axis_name)
        metrics.update(heads)
    metrics.update(head_gradient_norms(grads, config["actor_core"]))
    train_state = train_state.apply_gradients(grads=grads)
    metrics["grads_finite"] = jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]).all()
    return train_state, metrics


def warm_start_params(initialized, source, *, reset_critic=True):
    """Reuse a same-architecture legacy encoder/GRU; new action args stay fresh.

    Unlike native resume this intentionally resets optimizer and, by default,
    the value head because the return and time horizon have changed.
    """
    target, old = flatten_dict(initialized), flatten_dict(source)
    allowed_new = ("structured_heads", "structured_context")
    for key, value in old.items():
        if key not in target or target[key].shape != value.shape:
            raise ValueError(f"warm start architecture mismatch at {'/'.join(key)}")
    for key in target:
        is_new = any(name in key for name in allowed_new)
        if key not in old and not is_new:
            raise ValueError(f"warm start missing existing model leaf {'/'.join(key)}")
        if key in old and not is_new and not (reset_critic and "mlp_v" in key):
            target[key] = old[key]
    # Start navigation near native behavior: five cells, one base step. Finite
    # preference retains exploration and allows shorter unblocked requests.
    target[("params", "structured_heads", "move", "bias")] = jnp.tile(jnp.array([0., 0., 0., 0., 3.]), 2)
    target[("params", "structured_heads", "turn", "bias")] = jnp.tile(jnp.array([3., 0., 0., 0., 0., 0.]), 2)
    return unflatten_dict(target)
