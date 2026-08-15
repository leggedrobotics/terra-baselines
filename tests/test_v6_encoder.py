"""V6 readout block: flatten shrink, latent queries, aux decoder, aux loss."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from terra.config import BatchConfig, MapsDimsConfig

from train import Transition, aux_decoder_loss, ppo_update_networks
from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input, policy, policy_with_intermediates


class Config(dict):
    __getattr__ = dict.__getitem__


# Frozen compact V8 architecture used by the architecture comparison.
COMPACT = dict(
    map_encoder="resnet_spatial_8x8_se_xattn",
    resnet_stage_channels=(24, 48, 64, 96),
    resnet_blocks_per_stage=(2, 2, 3, 3),
)
# Frozen spatial-v6 architecture used by the architecture comparison.
V6_3M = dict(
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    resnet_stage_channels=(24, 48, 64, 96),
    resnet_blocks_per_stage=(3, 3, 2, 2),
    token_mixer_residual_init_scale=0.1,
    flatten_reduce_channels=32,
    attn_latent_queries=8,
    aux_coef=0.25,
)
COMPACT_PARAMETERS = 2_856_685
V6_3M_PARAMETERS = 2_134_755


def _config(**extra):
    return Config(
        clip_action_maps=True,
        loaded_max=100,
        local_map_normalization_bounds=(-16, 16),
        maps_net_normalization_bounds=(-10, 10),
        model_core="mlp",
        model_size="medium",
        num_prev_actions=5,
        critic_hidden_dims=(512, 256),
        encoder_compute_dtype="bfloat16",
        attention_compute_dtype="float32",
        **extra,
    )


def _env():
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )


def _obs(batch, env, seed=0):
    """Model-input obs list with nontrivial maps (get_model_ready's layout)."""
    edge = env.batch_cfg.maps_dims.maps_edge_length
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    obs = [
        jnp.zeros((batch, 4, env.batch_cfg.agent.num_state_obs)),
        jnp.zeros((batch, 4), dtype=jnp.int8).at[:, 0].set(1),
        jnp.ones((batch,), dtype=jnp.int32),
    ]
    obs += [jnp.zeros((batch, env.batch_cfg.agent.angles_cabin)) for _ in range(9)]
    obs += [jnp.zeros((batch, edge, edge)) for _ in range(4)]
    obs += [jnp.zeros((batch,), dtype=jnp.int32) for _ in range(2)]
    obs += [jnp.zeros((batch, edge, edge)) for _ in range(3)]
    obs += [jnp.zeros((batch, 5), dtype=jnp.int32)]
    # traversability, action, target, dumpability: enough structure for the
    # derived channels and the aux targets to be non-degenerate.
    obs[12] = (jax.random.uniform(keys[0], (batch, edge, edge)) < 0.1).astype(
        jnp.float32
    )
    obs[14] = jnp.clip(jax.random.normal(keys[1], (batch, edge, edge)), -1.0, 1.0)
    obs[15] = jnp.sign(jax.random.normal(keys[2], (batch, edge, edge)))
    obs[19] = (jax.random.uniform(keys[3], (batch, edge, edge)) < 0.3).astype(
        jnp.float32
    )
    return obs


def _param_shapes(params):
    return {
        jax.tree_util.keystr(path): tuple(leaf.shape)
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    }


class V6ParamTreeTest(unittest.TestCase):
    def test_v4_tree_is_identical_under_the_new_defaults(self):
        """New knobs must not perturb the deployed compact architecture."""
        _, params = get_model_ready(jax.random.PRNGKey(0), _config(**COMPACT), _env())
        shapes = _param_shapes(params)
        self.assertEqual(
            sum(x.size for x in jax.tree.leaves(params)), COMPACT_PARAMETERS
        )
        # Explicit defaults must reproduce the same tree as omitting them.
        _, defaults_params = get_model_ready(
            jax.random.PRNGKey(0),
            _config(flatten_reduce_channels=None, attn_latent_queries=4, **COMPACT),
            _env(),
        )
        self.assertEqual(shapes, _param_shapes(defaults_params))
        self.assertEqual(
            [p for p in shapes if "aux_decoder" in p or "LayerNorm_3" in p], []
        )
        self.assertIn(
            ("['params']['maps_net']['cnn']['attn_latent_queries']", (4, 96)),
            shapes.items(),
        )
        # 8*8*96 flatten rows: the readout still reads the full-width grid.
        self.assertEqual(
            shapes["['params']['maps_net']['cnn']['Dense_0']['kernel']"],
            (8 * 8 * 96, 192),
        )

    def test_v6_3m_parameter_count_and_shapes(self):
        model, params = get_model_ready(jax.random.PRNGKey(0), _config(**V6_3M), _env())
        count = sum(x.size for x in jax.tree.leaves(params))
        print(f"[V6] spatial_v6_3m = {count:,} parameters")
        self.assertEqual(count, V6_3M_PARAMETERS)
        shapes = _param_shapes(params)
        # Flatten shrink: 96 -> 32 channels, so the Dense reads 8*8*32 rows.
        self.assertEqual(
            shapes["['params']['maps_net']['cnn']['Conv_1']['kernel']"], (1, 1, 96, 32)
        )
        self.assertEqual(
            shapes["['params']['maps_net']['cnn']['Dense_0']['kernel']"],
            (8 * 8 * 32, 192),
        )
        self.assertEqual(
            shapes["['params']['maps_net']['cnn']['attn_latent_queries']"], (8, 96)
        )
        # Aux head is always in the tree when it is trained: 8x8x96 -> 32x32x4.
        aux = {
            k.split("['aux_decoder']")[1]: v
            for k, v in shapes.items()
            if "aux_decoder" in k
        }
        self.assertEqual(len(aux), 6)
        self.assertEqual(aux["['Conv_0']['kernel']"], (1, 1, 96, 64))
        self.assertEqual(aux["['Conv_1']['kernel']"], (3, 3, 64, 32))
        self.assertEqual(aux["['Conv_2']['kernel']"], (1, 1, 32, 4))
        self.assertEqual(model.map_encoder, "resnet_spatial_8x8_se_sa_xattn")


class V6ForwardTest(unittest.TestCase):
    def test_forward_is_finite_and_aux_logits_need_the_mutable_request(self):
        env = _env()
        model, params = get_model_ready(jax.random.PRNGKey(0), _config(**V6_3M), env)
        obs = _obs(2, env)

        value, xpi = model.apply(params, obs)
        self.assertEqual(value.shape, (2, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(value))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(xpi))))

        value_m, dist, intermediates = policy_with_intermediates(
            model.apply, params, obs
        )
        aux_logits = intermediates["maps_net"]["cnn"]["aux_logits"][0]
        self.assertEqual(aux_logits.shape, (2, 32, 32, 4))
        self.assertEqual(aux_logits.dtype, jnp.float32)
        self.assertTrue(bool(jnp.all(jnp.isfinite(aux_logits))))
        # Sowing must not perturb the policy outputs.
        legacy_value, legacy_dist = policy(model.apply, params, obs)
        self.assertEqual(float(jnp.max(jnp.abs(value_m - legacy_value))), 0.0)
        self.assertEqual(
            float(
                jnp.max(
                    jnp.abs(dist.logits_parameter() - legacy_dist.logits_parameter())
                )
            ),
            0.0,
        )

    def test_aux_head_is_dropped_from_the_rollout_graph(self):
        """The compiled rollout program must not contain the decoder."""
        env = _env()
        obs = _obs(2, env)
        compiled = []
        for aux_coef in (0.0, 0.25):
            model, params = get_model_ready(
                jax.random.PRNGKey(0),
                _config(**{**V6_3M, "aux_coef": aux_coef}),
                env,
            )
            compiled.append(jax.jit(model.apply).lower(params, obs).compile().as_text())
        self.assertEqual(
            compiled[0].count("convolution("), compiled[1].count("convolution(")
        )


class V6AuxLossTest(unittest.TestCase):
    def test_targets_are_pooled_masked_and_bce_is_finite(self):
        env = _env()
        obs = _obs(3, env, seed=1)
        # Pad the top half of the map: those cells must not be supervised.
        obs[18] = jnp.zeros_like(obs[18]).at[:, :32, :].set(1.0)
        logits = jnp.zeros((3, 32, 32, 4))
        loss = aux_decoder_loss(logits, obs)
        self.assertTrue(bool(jnp.isfinite(loss)))
        # Zero logits => BCE = log 2 per cell regardless of the (soft) target
        # (float32 summation over 12k terms costs the last digits).
        self.assertAlmostEqual(float(loss), float(jnp.log(2.0)), places=4)

        # A head that reproduces the pooled targets beats chance; inverting it
        # is worse than chance. Targets are rebuilt here independently.
        def pooled(binary):
            return binary.astype(jnp.float32).reshape(3, 32, 2, 32, 2).mean(axis=(2, 4))

        targets = jnp.stack(
            (
                pooled((obs[15] < 0) & (obs[14] > obs[15])),  # remaining dig
                pooled((obs[15] > 0) & (obs[14] <= 0)),  # dump deficit
                pooled(obs[19] != 0),  # dumpability
                pooled(obs[12] != 0),  # obstacle
            ),
            axis=-1,
        )
        good = jnp.clip(jnp.log(targets / (1.0 - targets)), -6.0, 6.0)
        self.assertLess(float(aux_decoder_loss(good, obs)), float(loss))
        self.assertGreater(float(aux_decoder_loss(-good, obs)), float(loss))

    def test_masking_ignores_padded_cells(self):
        env = _env()
        obs = _obs(2, env, seed=2)
        obs[18] = jnp.zeros_like(obs[18]).at[:, :32, :].set(1.0)
        logits = jax.random.normal(jax.random.PRNGKey(3), (2, 32, 32, 4))
        baseline = aux_decoder_loss(logits, obs)
        # Anything inside the padded half is irrelevant to the loss.
        perturbed = logits.at[:, :16, :, :].add(50.0)
        self.assertEqual(float(aux_decoder_loss(perturbed, obs)), float(baseline))


class V6UpdateTest(unittest.TestCase):
    """The aux term reaches the PPO update without disturbing aux_coef=0."""

    def _train_state(self, aux_coef):
        env = _env()
        model, params = get_model_ready(
            jax.random.PRNGKey(0),
            _config(**{**V6_3M, "aux_coef": aux_coef}),
            env,
        )
        state = TrainState.create(
            apply_fn=model.apply, params=params, tx=optax.adam(1e-4)
        )
        return state, env

    def _transitions(self, env, batch):
        edge = env.batch_cfg.maps_dims.maps_edge_length
        keys = jax.random.split(jax.random.PRNGKey(11), 3)
        grid = lambda: jnp.zeros((batch, edge, edge))
        local = lambda: jnp.zeros((batch, env.batch_cfg.agent.angles_cabin))
        obs = {
            "agent_states": jnp.zeros((batch, 4, env.batch_cfg.agent.num_state_obs)),
            "agent_active": jnp.zeros((batch, 4), dtype=jnp.int8).at[:, 0].set(1),
            "num_agents": jnp.ones((batch,), dtype=jnp.int32),
            "local_map_action_neg": local(),
            "local_map_action_pos": local(),
            "local_map_target_neg": local(),
            "local_map_target_pos": local(),
            "local_map_dumpability": local(),
            "local_map_obstacles": local(),
            "local_map_border_workspace": local(),
            "local_map_edge_alignment_error": local(),
            "local_map_border_diggable": local(),
            "traversability_mask": grid(),
            "reachability_mask": grid(),
            "action_map": jnp.clip(
                jax.random.normal(keys[0], (batch, edge, edge)), -1.0, 1.0
            ),
            "target_map": jnp.sign(jax.random.normal(keys[1], (batch, edge, edge))),
            "agent_width": jnp.zeros((batch,), dtype=jnp.int32),
            "agent_height": jnp.zeros((batch,), dtype=jnp.int32),
            "padding_mask": grid(),
            "dumpability_mask": (
                jax.random.uniform(keys[2], (batch, edge, edge)) < 0.3
            ).astype(jnp.float32),
            "interaction_mask": grid(),
        }
        return Transition(
            done=jnp.zeros((batch,), dtype=jnp.bool_),
            task_done=jnp.zeros((batch,), dtype=jnp.bool_),
            action=jnp.zeros((batch,), dtype=jnp.int32),
            value=jnp.zeros((batch,), dtype=jnp.float32),
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            log_prob=jnp.zeros((batch,), dtype=jnp.float32),
            obs=obs,
            prev_actions=jnp.zeros((batch, 5), dtype=jnp.int32),
            prev_reward=jnp.zeros((batch,), dtype=jnp.float32),
        )

    def _update(self, config, state, transitions):
        advantages = jax.random.normal(jax.random.PRNGKey(5), (4,))
        targets = jax.random.normal(jax.random.PRNGKey(6), (4,))
        add = lambda x: jnp.asarray(x)[None]
        _, info = jax.vmap(
            lambda ts, tr, adv, tgt: ppo_update_networks(ts, tr, adv, tgt, config),
            axis_name="devices",
        )(
            jax.tree_util.tree_map(add, state),
            jax.tree_util.tree_map(add, transitions),
            advantages[None],
            targets[None],
        )
        return jax.tree_util.tree_map(lambda x: x[0], info)

    def _ppo_config(self, aux_coef):
        return Config(
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            clip_action_maps=True,
            use_value_clip=False,
            flat_minibatch_shuffle=True,
            teacher_obs_downsample=1,
            aux_coef=aux_coef,
        )

    def test_aux_term_is_logged_and_absent_when_disabled(self):
        state, env = self._train_state(0.25)
        transitions = self._transitions(env, 4)

        info = self._update(self._ppo_config(0.25), state, transitions)
        self.assertIn("aux_loss", info)
        self.assertTrue(bool(jnp.isfinite(info["aux_loss"])))
        self.assertGreater(float(info["aux_loss"]), 0.0)
        self.assertTrue(bool(jnp.isfinite(info["total_loss"])))
        self.assertEqual(float(info["diagnostics/grads_all_finite"]), 1.0)

        # aux_coef = 0 keeps the legacy loss exactly: same model, same batch,
        # only the aux term removed.
        off = self._update(self._ppo_config(0.0), state, transitions)
        self.assertNotIn("aux_loss", off)
        expected = float(info["total_loss"]) - 0.25 * float(info["aux_loss"])
        self.assertAlmostEqual(float(off["total_loss"]), expected, places=5)
        for key in ("value_loss", "actor_loss", "entropy", "approx_kl"):
            self.assertAlmostEqual(float(off[key]), float(info[key]), places=6, msg=key)

    def test_aux_gradient_reaches_the_encoder_trunk(self):
        state, env = self._train_state(0.25)
        transitions = self._transitions(env, 4)
        obs = obs_to_model_input(
            transitions.obs, transitions.prev_actions, self._ppo_config(0.25)
        )

        def loss_fn(params):
            _, _, intermediates = policy_with_intermediates(state.apply_fn, params, obs)
            return aux_decoder_loss(
                intermediates["maps_net"]["cnn"]["aux_logits"][0], obs
            )

        grads = jax.grad(loss_fn)(state.params)
        flat = {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]
        }
        stem = flat["['params']['maps_net']['cnn']['Conv_0']['kernel']"]
        head = flat["['params']['maps_net']['cnn']['aux_decoder']['Conv_2']['kernel']"]
        self.assertGreater(float(jnp.linalg.norm(head)), 0.0)
        self.assertGreater(float(jnp.linalg.norm(stem)), 0.0)
        # The PPO heads are not touched by the aux loss alone.
        self.assertEqual(
            float(jnp.linalg.norm(flat["['params']['mlp_pi']['layers_0']['kernel']"])),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
