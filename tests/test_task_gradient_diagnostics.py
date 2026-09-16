"""Cross-device family imbalance must not turn interference into a norm artifact."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.task_gradient_diagnostics import task_gradient_diagnostics


def test_distributed_gradients_match_complete_global_minibatch():
    if jax.local_device_count() < 4:
        pytest.skip('run with XLA_FLAGS=--xla_force_host_platform_device_count=4')
    params = {'params': {'maps_net': {'weights': jnp.array([[.2, -.1, .1], [.4, .1, -.2]])}}}
    inputs = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 10

    def apply(parameters, observations):
        hidden = observations @ parameters['params']['maps_net']['weights']
        logits = hidden @ jnp.array([[.2, -.1], [-.4, .5], [.1, .3]])
        return (hidden ** 2).sum(axis=-1, keepdims=True), logits

    values, logits = apply(params, inputs)
    actions = jnp.arange(8) % 2
    log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1)[:, 0]
    advantages = jnp.array([.7, -.2, .4, .2, -.7, -.4, .8, -.8])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch = dict(model_inputs=inputs, actions=actions, old_log_prob=log_prob,
                 old_values=values[:, 0], targets=values[:, 0] + advantages,
                 normalized_advantages=advantages,
                 family=jnp.array([0, 0, 0, 0, 0, 1, 1, 0]),
                 teacher_logits=-logits)

    def diagnostic(data, axis_name):
        return task_gradient_diagnostics(
            apply, params, **data, family_ids={'foundation': 0, 'trench': 1},
            clip_eps=.2, vf_coef=2., ent_coef=.02,
            foundation_kl_coef=.2, trench_kl_coef=.7, axis_name=axis_name)

    complete = diagnostic(batch, None)
    split = jax.tree.map(lambda x: x.reshape((4, 2) + x.shape[1:]), batch)
    distributed = jax.pmap(lambda x: diagnostic(x, 'devices'), axis_name='devices')(split)
    for key, expected in complete.items():
        np.testing.assert_allclose(distributed[key], jnp.repeat(expected[None], 4),
                                   rtol=1e-5, atol=1e-7, err_msg=key)
    assert float(complete['recognized_sample_fraction']) == 1.
    assert float(complete['foundation/samples']) == 6
    assert float(complete['trench/samples']) == 2
    for name, fraction in [('foundation', .75), ('trench', .25)]:
        np.testing.assert_allclose(complete[f'{name}/ppo_contribution_norm'],
                                  complete[f'{name}/ppo_conditional_norm'] * fraction)
        np.testing.assert_allclose(complete[f'{name}/kl_contribution_norm'],
                                  complete[f'{name}/kl_conditional_norm'] * fraction)


def test_missing_family_is_unavailable_not_zero_agreement():
    params = {'params': {'maps_net': {'weight': jnp.float32(.2)}}}

    def apply(parameters, observations):
        value = observations * parameters['params']['maps_net']['weight']
        return value[:, None], jnp.stack((value, -value), axis=-1)

    result = task_gradient_diagnostics(
        apply, params, jnp.ones(2), actions=jnp.array([0, 1]),
        old_log_prob=jnp.zeros(2), old_values=jnp.zeros(2), targets=jnp.ones(2),
        normalized_advantages=jnp.array([1., -1.]), family=jnp.zeros(2, jnp.int32),
        teacher_logits=jnp.zeros((2, 2)), family_ids={'foundation': 0, 'trench': 1},
        clip_eps=.2, vf_coef=2., ent_coef=.02,
        foundation_kl_coef=.2, trench_kl_coef=.7)
    assert float(result['trench/samples']) == 0
    assert np.isnan(float(result['trench/ppo_conditional_norm']))
    assert np.isnan(float(result['foundation_trench_ppo_cosine']))
