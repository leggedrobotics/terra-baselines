"""Machine-work observation: the one-way migration keeps the team policy."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from test_team_policy import A, B, ENV, _config, _team_obs
from utils.machine_work import KERNEL_PATH, migrate_machine_work_observation_checkpoint
from utils.models import get_model_ready
from utils.utils_ppo import joint_obs_to_model_input, random_agent_order


def _with_work_column(obs, rng):
    work = jax.random.uniform(rng, obs["agent_states"].shape[:-1] + (2,))
    return {**obs, "agent_states": jnp.concatenate((obs["agent_states"], work), axis=-1)}


def test_migration_adds_a_zero_input_and_keeps_policy_and_value():
    base_cfg = _config(agent_types_override=(0,) * A)
    grown_cfg = _config(agent_types_override=(0,) * A, machine_work_observation=True)
    base_model, base_params = get_model_ready(jax.random.PRNGKey(1), base_cfg, ENV)
    grown_model, grown_init = get_model_ready(jax.random.PRNGKey(2), grown_cfg, ENV)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    checkpoint = {
        "model": base_params, "optimizer_state": optimizer.init(base_params),
        "train_state_step": 0, "next_update": 0, "train_config": base_cfg,
    }
    migrated = migrate_machine_work_observation_checkpoint(checkpoint, grown_init)

    kernel, base_kernel = migrated["model"], base_params
    for key in KERNEL_PATH:
        kernel, base_kernel = kernel[key], base_kernel[key]
    assert kernel.shape == (base_kernel.shape[0] + 2, base_kernel.shape[1])
    np.testing.assert_array_equal(kernel[:-2], base_kernel)
    assert not np.any(np.asarray(kernel[-2:]))
    assert migrated["train_config"]["machine_work_observation"] is True
    assert jax.tree_util.tree_structure(migrated["optimizer_state"]) == jax.tree_util.tree_structure(
        optimizer.init(grown_init))
    for got, want in zip(jax.tree_util.tree_leaves(migrated["optimizer_state"]),
                         jax.tree_util.tree_leaves(optimizer.init(grown_init))):
        assert np.shape(got) == np.shape(want)

    obs = _with_work_column(_team_obs(jax.random.PRNGKey(3)), jax.random.PRNGKey(4))
    prev = jax.random.randint(jax.random.PRNGKey(5), (B, A, 5), 0, 8)
    order = random_agent_order(jax.random.PRNGKey(6), B, A)
    actions = jax.random.randint(jax.random.PRNGKey(7), (B, A), 0, 8)
    base_values, base_logits = base_model.apply(
        base_params, joint_obs_to_model_input(obs, prev, base_cfg, A), actions, order,
        method="joint_policy",
    )
    values, logits = grown_model.apply(
        migrated["model"], joint_obs_to_model_input(obs, prev, grown_cfg, A), actions, order,
        method="joint_policy",
    )
    np.testing.assert_allclose(logits, base_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(values, base_values, rtol=1e-6, atol=1e-6)

    # The grown model does read the new column once its weights move.
    moved = jax.tree_util.tree_map(lambda x: x, migrated["model"])
    node = moved
    for key in KERNEL_PATH[:-1]:
        node = node[key]
    node[KERNEL_PATH[-1]] = node[KERNEL_PATH[-1]].at[-1].set(1.0)
    _, moved_logits = grown_model.apply(
        moved, joint_obs_to_model_input(obs, prev, grown_cfg, A), actions, order,
        method="joint_policy",
    )
    assert not np.allclose(moved_logits, base_logits)
