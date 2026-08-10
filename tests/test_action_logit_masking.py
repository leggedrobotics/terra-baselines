"""D3 action-logit masking: distribution, obs-list, and consistency contracts."""

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from utils.utils_ppo import _masked_logits


def test_masked_sampling_never_selects_invalid():
    logits = jnp.array([[2.0, 1.0, 0.5, -0.5, 3.0, 0.0, 1.5, -1.0]] * 64)
    mask = jnp.array([[False, True, False, True, False, True, False, True]] * 64)
    pi = tfp.distributions.Categorical(logits=_masked_logits(logits, mask))
    actions = pi.sample(seed=jax.random.PRNGKey(0))
    assert bool(jnp.all(mask[jnp.arange(64), actions]))


def test_masked_log_prob_matches_manual_softmax():
    logits = jnp.array([[0.3, -0.7, 1.2, 0.0, 2.0, -2.0, 0.4, 0.9]])
    mask = jnp.array([[True, False, True, True, False, True, True, True]])
    pi = tfp.distributions.Categorical(logits=_masked_logits(logits, mask))
    manual = jax.nn.log_softmax(jnp.where(mask, logits, jnp.float32(-1e9)))
    for action in (0, 2, 3):
        np.testing.assert_allclose(
            float(pi.log_prob(jnp.array([action]))[0]),
            float(manual[0, action]),
            rtol=1e-6,
        )
    # Entropy is computed on the masked support only.
    assert float(pi.entropy()[0]) <= float(np.log(6)) + 1e-5


def test_mask_none_is_identity():
    logits = jnp.array([[0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(_masked_logits(logits, None), logits)


def test_do_nothing_guard_makes_all_invalid_impossible():
    # The env appends DO_NOTHING=True to every mask row, so a row of False
    # for the 7 simulated handlers still leaves one valid action.
    handler_mask = jnp.zeros((4, 7), dtype=bool)
    env_mask = jnp.concatenate(
        [handler_mask, jnp.ones((4, 1), dtype=bool)], axis=-1
    )
    logits = jnp.zeros((4, 8))
    pi = tfp.distributions.Categorical(logits=_masked_logits(logits, env_mask))
    actions = pi.sample(seed=jax.random.PRNGKey(1))
    assert bool(jnp.all(actions == 7))
    assert bool(jnp.all(jnp.isfinite(pi.log_prob(actions))))
