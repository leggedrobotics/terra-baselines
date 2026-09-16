"""Occasional shared-encoder PPO/KL diagnostics on an actual training minibatch.

This is deliberately separate from the normal update: four reverse passes are
too expensive per minibatch. Inputs must retain the production minibatch's
already-normalized advantages and selected frozen teacher logits. A failure
snapshot, synthetic return, or renormalized family subset is not a substitute.
"""

import jax
import jax.numpy as jnp


def task_gradient_diagnostics(
    apply_fn, params, model_inputs, *, actions, old_log_prob, old_values,
    targets, normalized_advantages, family, teacher_logits, family_ids,
    clip_eps, vf_coef, ent_coef, foundation_kl_coef, trench_kl_coef,
    use_value_clip=True, axis_name=None,
):
    """Return family gradient norms/cosines for the current unmasked MLP recipe.

    All sample arrays are flat [N], except teacher logits [N, actions]. Global
    minibatch normalization happens BEFORE calling this function. With pmap,
    pass its axis_name: gradient contributions/counts are reduced across devices
    before calculating a norm, cosine, or conditional gradient. Without pmap,
    provide the complete actual minibatch from all devices. This mirrors PPO's
    clipped actor/value loss and entropy term, with no aux or value-distillation
    objective. It never updates parameters or changes task sampling.
    """
    actions = jnp.asarray(actions).reshape(-1)
    old_log_prob = jnp.asarray(old_log_prob).reshape(-1)
    old_values = jnp.asarray(old_values).reshape(-1)
    targets = jnp.asarray(targets).reshape(-1)
    advantages = jax.lax.stop_gradient(jnp.asarray(normalized_advantages).reshape(-1))
    family = jnp.asarray(family).reshape(-1)
    count = actions.shape[0]
    if not count or any(x.shape != (count,) for x in
                        (old_log_prob, old_values, targets, advantages, family)):
        raise ValueError('task gradient diagnostics require matching flat sample arrays')
    if teacher_logits.ndim != 2 or teacher_logits.shape[0] != count:
        raise ValueError('teacher logits must have [samples, actions] shape')
    masks = jnp.stack([family == family_ids[name] for name in ('foundation', 'trench')])
    counts = masks.sum(axis=1).astype(jnp.float32)
    teacher_logp = jax.nn.log_softmax(jax.lax.stop_gradient(teacher_logits), axis=-1)
    teacher_p = jnp.exp(teacher_logp)

    def objectives(encoder_params):
        changed = {**params, 'params': {**params['params'], 'maps_net': encoder_params}}
        value, logits = apply_fn(changed, model_inputs)
        value = value[..., 0]
        logp = jax.nn.log_softmax(logits, axis=-1)
        chosen_logp = jnp.take_along_axis(logp, actions[:, None], axis=-1)[:, 0]
        ratio = jnp.exp(chosen_logp - old_log_prob)
        actor = -jnp.minimum(advantages * ratio,
                             advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps))
        squared = jnp.square(value - targets)
        if use_value_clip:
            clipped = old_values + jnp.clip(value - old_values, -clip_eps, clip_eps)
            squared = jnp.maximum(squared, jnp.square(clipped - targets))
        entropy = -jnp.sum(jnp.exp(logp) * logp, axis=-1)
        ppo = actor + .5 * vf_coef * squared - ent_coef * entropy
        kl = jnp.sum(teacher_p * (teacher_logp - logp), axis=-1)
        # Global-mean contributions preserve training's transition weighting.
        # They are converted to conditional means after multi-device reduction.
        return jnp.stack([
            jnp.where(masks[0], ppo, 0).mean(),
            jnp.where(masks[1], ppo, 0).mean(),
            jnp.where(masks[0], kl, 0).mean(),
            jnp.where(masks[1], kl, 0).mean(),
        ])

    derivatives = jax.jacrev(objectives)(params['params']['maps_net'])
    if axis_name is not None:
        derivatives = jax.lax.pmean(derivatives, axis_name)
        counts = jax.lax.psum(counts, axis_name)
        global_count = jax.lax.psum(jnp.float32(count), axis_name)
    else:
        global_count = jnp.float32(count)
    gradients = jnp.concatenate([leaf.reshape(4, -1)
                                 for leaf in jax.tree.leaves(derivatives)], axis=1)
    fractions = counts / global_count
    coefficients = jnp.array([foundation_kl_coef, trench_kl_coef], jnp.float32)

    def norm(vector):
        return jnp.linalg.norm(vector)

    def cosine(left, right):
        denominator = norm(left) * norm(right)
        return jnp.where(denominator > 0, jnp.dot(left, right) / denominator, jnp.nan)

    metrics = {}
    for i, name in enumerate(('foundation', 'trench')):
        ppo, kl = gradients[i], gradients[i + 2]
        metrics[f'{name}/samples'] = counts[i]
        metrics[f'{name}/transition_fraction'] = fractions[i]
        metrics[f'{name}/ppo_contribution_norm'] = norm(ppo)
        metrics[f'{name}/ppo_conditional_norm'] = jnp.where(
            counts[i] > 0, norm(ppo) / fractions[i], jnp.nan)
        metrics[f'{name}/kl_contribution_norm'] = norm(kl)
        metrics[f'{name}/kl_conditional_norm'] = jnp.where(
            counts[i] > 0, norm(kl) / fractions[i], jnp.nan)
        metrics[f'{name}/weighted_kl_contribution_norm'] = coefficients[i] * norm(kl)
        metrics[f'{name}/ppo_kl_cosine'] = cosine(ppo, kl)
    metrics['foundation_trench_ppo_cosine'] = cosine(gradients[0], gradients[1])
    metrics['foundation_trench_kl_cosine'] = cosine(gradients[2], gradients[3])
    metrics['combined_ppo_weighted_kl_cosine'] = cosine(
        gradients[0] + gradients[1],
        coefficients[0] * gradients[2] + coefficients[1] * gradients[3])
    metrics['recognized_sample_fraction'] = counts.sum() / global_count
    return metrics
