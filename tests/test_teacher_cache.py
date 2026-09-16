"""Caching must retain row provenance and the actual PPO update."""
import jax
import jax.numpy as jnp
import numpy as np

from test_foundation_teacher_release import _actual_update
from utils.task_teachers import (
    CACHED_LOGITS_KEY, CACHED_VALUE_KEY, FAMILY_KEY, LEGACY_DIG_KEY,
    cache_task_teacher_outputs,
)


def test_cache_preserves_time_environment_rows_and_frozen_outputs():
    steps, envs = 3, 4
    rows = jnp.arange(steps * envs).reshape(steps, envs)
    obs = {FAMILY_KEY: rows % 2, "row": rows,
           LEGACY_DIG_KEY: jnp.ones((steps, envs, 12))}
    history = (rows + 100)[..., None]

    def teacher(params, raw, prev):
        x = params * raw["row"] + prev[:, 0]
        return x[:, None], x[:, None] + jnp.arange(8)

    result = jax.jit(lambda p: cache_task_teacher_outputs(obs, history, teacher, p, 3))(2.)
    expected = 3 * rows + 100
    np.testing.assert_array_equal(result[CACHED_VALUE_KEY][..., 0], expected)
    np.testing.assert_array_equal(result[CACHED_LOGITS_KEY], expected[..., None] + jnp.arange(8))
    np.testing.assert_array_equal(result[FAMILY_KEY], rows % 2)
    assert LEGACY_DIG_KEY not in result
    assert float(jax.grad(lambda p: cache_task_teacher_outputs(
        obs, history, teacher, p, 3)[CACHED_LOGITS_KEY].sum())(2.)) == 0.


def test_cached_outputs_preserve_actual_ppo_gradient_and_global_family_metrics():
    uncached = _actual_update(foundation_coef=.2, trench_coef=.7)
    cached = _actual_update(foundation_coef=.2, trench_coef=.7, cached=True)
    for expected, actual in zip(jax.tree_util.tree_leaves(uncached), jax.tree_util.tree_leaves(cached)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
