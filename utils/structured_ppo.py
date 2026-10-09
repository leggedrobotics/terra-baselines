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


def policy_masks(masks, allow_wait=False):
    """Remove explicit cabin turns; retain wait only as all-invalid fallback."""
    action = jnp.asarray(masks["action_mask"], bool).at[..., 4:6].set(False)
    # Every type must have at least one executable argument.
    action = action.at[..., :2].set(action[..., :2] & masks["move_mask"].any(-1))
    action = action.at[..., 2:4].set(action[..., 2:4] & masks["turn_mask"].any(-1))
    action = action.at[..., 6].set(action[..., 6] & masks["do_mask"].any(-1))
    action = action.at[..., 7].set(allow_wait | ~action[..., :7].any(-1))
    return {**masks, "action_mask": action}


def _log_probs(logits, mask):
    # Unused rows can be empty (e.g. all moves while loaded). Give them a
    # deterministic dummy argument so entropy and gradients stay finite.
    fallback = jax.nn.one_hot(jnp.zeros(mask.shape[:-1], jnp.int32), mask.shape[-1], dtype=bool)
    mask = jnp.where(mask.any(-1, keepdims=True), mask, fallback)
    return jax.nn.log_softmax(jnp.where(mask, logits, -1e9), axis=-1)


def distributions(logits, masks):
    return {
        "action": _log_probs(logits["action"], masks["action_mask"]),
        "move": _log_probs(logits["move"], masks["move_mask"]),
        "turn": _log_probs(logits["turn"], masks["turn_mask"]),
        "do": _log_probs(logits["do"], masks["do_mask"]),
    }


def _take(log_probs, index):
    return jnp.take_along_axis(log_probs, index[..., None], axis=-1)[..., 0]


def joint_log_prob(logits, masks, action):
    lp = distributions(logits, masks)
    typ = jnp.asarray(action.action, jnp.int32)
    amount = jnp.asarray(action.amount, jnp.int32) - 1
    move_row = jnp.take_along_axis(lp["move"], jnp.clip(typ, 0, 1)[..., None, None], -2)[..., 0, :]
    turn_row = jnp.take_along_axis(lp["turn"], jnp.clip(typ - 2, 0, 1)[..., None, None], -2)[..., 0, :]
    move_lp = _take(move_row, jnp.clip(amount, 0, 4))
    turn_lp = _take(turn_row, jnp.clip(amount, 0, 5))
    do_lp = _take(lp["do"], jnp.clip(action.heading, 0, 11))
    return (_take(lp["action"], typ) + jnp.where(typ < 2, move_lp, 0.)
            + jnp.where((typ >= 2) & (typ < 4), turn_lp, 0.)
            + jnp.where(typ == 6, do_lp, 0.))


def entropy_components(logits, masks):
    """Exact joint entropy components, averaged over the chosen type law."""
    lp = distributions(logits, masks)
    entropy = lambda value: -(jnp.exp(value) * value).sum(-1)
    probs = jnp.exp(lp["action"])
    return {
        "action": entropy(lp["action"]),
        "move": (probs[..., :2] * entropy(lp["move"])).sum(-1),
        "turn": (probs[..., 2:4] * entropy(lp["turn"])).sum(-1),
        "do": probs[..., 6] * entropy(lp["do"]),
    }


def sample_action(logits, masks, rng, deterministic=False):
    lp = distributions(logits, masks)
    keys = jax.random.split(rng, 4)
    choose = lambda values, key: (jnp.argmax(values, -1) if deterministic
                                  else jax.random.categorical(key, values, axis=-1))
    typ = choose(lp["action"], keys[0])
    move_row = jnp.take_along_axis(lp["move"], jnp.clip(typ, 0, 1)[..., None, None], -2)[..., 0, :]
    turn_row = jnp.take_along_axis(lp["turn"], jnp.clip(typ - 2, 0, 1)[..., None, None], -2)[..., 0, :]
    amount = jnp.where(typ < 2, choose(move_row, keys[1]) + 1,
                      jnp.where((typ >= 2) & (typ < 4), choose(turn_row, keys[2]) + 1, 1))
    heading = jnp.where(typ == 6, choose(lp["do"], keys[3]), -1)
    action = StructuredAction(typ.astype(jnp.int32), amount.astype(jnp.int32), heading.astype(jnp.int32))
    return action, joint_log_prob(logits, masks, action)


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


def ppo_update(train_state, model, rollout, advantages, targets, initial_hidden,
               config, *, clip_eps=0.2, vf_coef=2., entropy_coefs=(0.01, 0.01, 0.01, 0.01)):
    """One minibatch; the probability ratio uses the full sampled action."""
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
        log_prob = joint_log_prob(logits, rollout.masks, rollout.action)
        ratio = jnp.exp(log_prob - rollout.log_prob)
        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = -jnp.minimum(ratio * norm_adv, jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * norm_adv).mean()
        clipped_value = rollout.value + jnp.clip(values - rollout.value, -clip_eps, clip_eps)
        value_loss = 0.5 * jnp.maximum((values-targets)**2, (clipped_value-targets)**2).mean()
        components = entropy_components(logits, rollout.masks)
        entropy_bonus = sum(coef * components[name].mean() for coef, name in zip(entropy_coefs, ("action", "move", "turn", "do")))
        total = actor_loss + vf_coef * value_loss - entropy_bonus
        metrics = {
            "total_loss": total, "actor_loss": actor_loss, "value_loss": value_loss,
            "entropy": sum(x.mean() for x in components.values()),
            "approx_kl": ((ratio - 1.) - (log_prob - rollout.log_prob)).mean(),
            "clip_fraction": (jnp.abs(ratio-1.) > clip_eps).mean(),
            **{f"entropy_{k}": v.mean() for k, v in components.items()},
        }
        return total, metrics
    (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(train_state.params)
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
