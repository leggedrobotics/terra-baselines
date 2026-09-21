"""D3 action-logit masking: distribution, obs-list, and consistency contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.jax_utils import replicate
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp

from train import Transition, ppo_update_networks
from train_mixed import MixedAgentTrainConfig
from utils.utils_ppo import _masked_logits, obs_to_model_input, select_action_ppo


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


def _optional_feature_case(alignment, masking):
    config = MixedAgentTrainConfig(
        name="mask-input-test", num_devices=1, num_envs_per_device=4,
        num_steps=1, num_minibatches=1, num_prev_actions=5,
        agent_types_override=(0,), action_types_override=(0,),
        flat_minibatch_shuffle=True, trench_alignment_observation=alignment,
        action_logit_masking=masking, time_observation_mode="none",
    )
    local = ("action_neg", "action_pos", "target_neg", "target_pos",
             "dumpability", "obstacles", "border_workspace",
             "edge_alignment_error", "border_diggable")
    maps = ("traversability_mask", "reachability_mask", "action_map", "target_map",
            "padding_mask", "dumpability_mask", "interaction_mask")
    raw = {"local_map_" + key: jnp.zeros((4, 12)) for key in local}
    raw.update({key: jnp.zeros((4, 2, 2)) for key in maps})
    raw.update(agent_states=jnp.zeros((4, 4, 9)), agent_active=jnp.ones((4, 4)),
               num_agents=jnp.ones(4), agent_width=jnp.ones(4), agent_height=jnp.ones(4),
               fresh_trench_dig_alignment_valid=jnp.ones(4),
               fresh_trench_dig_yaw_error=jnp.zeros(4),
               fresh_trench_dig_standoff_error=jnp.zeros(4),
               action_mask=jnp.tile(jnp.array([False, True] * 4), (4, 1)))
    history = jnp.zeros((4, 5), dtype=jnp.int32)

    def model(params, obs):
        batch = obs[0].shape[0]
        return (jnp.broadcast_to(params["value"], (batch, 1)),
                jnp.broadcast_to(params["logits"], (batch, 8)))

    state = TrainState.create(apply_fn=model, tx=optax.adam(0.01),
        params=dict(value=jnp.float32(0),
                    logits=jnp.array([.2, -.1, .7, 0., 1.2, -.2, .3, 1.])))
    return config, raw, history, state


@pytest.mark.parametrize("alignment", [False, True])
@pytest.mark.parametrize("masking", [False, True])
def test_rollout_uses_eight_action_mask_after_optional_features(alignment, masking):
    config, raw, history, state = _optional_feature_case(alignment, masking)
    model_obs = obs_to_model_input(raw, history, config)
    if alignment:
        assert model_obs[22].shape == (4, 3)
    if masking:
        np.testing.assert_array_equal(model_obs[-1], raw["action_mask"])
    rng = jax.random.PRNGKey(9)
    actions, log_prob, value, distribution = select_action_ppo(
        state, raw, history, rng, config,
    )
    logits = jnp.broadcast_to(state.params["logits"], (4, 8))
    expected = tfp.distributions.Categorical(logits=jnp.where(
        raw["action_mask"], logits, -1e9) if masking else logits)
    np.testing.assert_array_equal(actions, expected.sample(seed=rng))
    np.testing.assert_array_equal(log_prob, expected.log_prob(actions))
    np.testing.assert_array_equal(distribution.probs_parameter(), expected.probs_parameter())
    np.testing.assert_array_equal(value, jnp.zeros(4))


def _ppo_and_demo_update(alignment, masking):
    config, raw, history, state = _optional_feature_case(alignment, masking)
    live_actions = jnp.array([1, 3, 5, 7])
    logits = jnp.broadcast_to(state.params["logits"], (4, 8))
    live_distribution = tfp.distributions.Categorical(logits=jnp.where(
        raw["action_mask"], logits, -1e9) if masking else logits)
    transition = Transition(done=jnp.zeros(4, bool), task_done=jnp.zeros(4, bool),
        action=live_actions, value=jnp.zeros(4), reward=jnp.zeros(4),
        log_prob=live_distribution.log_prob(live_actions), obs=raw,
        prev_actions=history, prev_reward=jnp.zeros(4))
    # Distinct demo support catches accidentally reusing the live-rollout mask.
    demo_raw = dict(raw, action_mask=jnp.logical_not(raw["action_mask"]))
    demo_actions = jnp.array([0, 2, 4, 6])
    demo = dict(obs=demo_raw, previous_actions=history, actions=demo_actions,
                group_id=jnp.full(4, 2, dtype=jnp.int32), parent_probs=jnp.zeros((4, 8)))

    def update(state, transition, demo):
        return ppo_update_networks(state, transition, jnp.arange(4.), jnp.ones(4),
            config, demonstration_batch=demo, demonstration_coef=jnp.float32(.1))

    result = jax.pmap(update, axis_name="devices")(
        replicate(state, jax.local_devices()[:1]),
        jax.tree_util.tree_map(lambda x: x[None], transition),
        jax.tree_util.tree_map(lambda x: x[None], demo))
    updated, info = result
    expected_demo = tfp.distributions.Categorical(logits=jnp.where(
        demo_raw["action_mask"], logits, -1e9) if masking else logits)
    np.testing.assert_allclose(info["imitation/cross_entropy"],
                               -expected_demo.log_prob(demo_actions).mean(), rtol=1e-6)
    np.testing.assert_allclose(info["approx_kl"], 0., atol=1e-7)
    np.testing.assert_array_equal(updated.step, [1])
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(result))
    return updated.params, updated.opt_state, info


@pytest.mark.parametrize("masking", [False, True])
def test_ppo_and_demo_preserve_base_mask_and_unmasked_update(masking):
    baseline = _ppo_and_demo_update(False, masking)
    with_alignment = _ppo_and_demo_update(True, masking)
    for before, after in zip(jax.tree_util.tree_leaves(baseline),
                             jax.tree_util.tree_leaves(with_alignment)):
        np.testing.assert_array_equal(before, after)
