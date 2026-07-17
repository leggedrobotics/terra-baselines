import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import DelayedDownsampleResNet, MapsNet, get_model_ready


class DelayedDownsampleResNetTest(unittest.TestCase):
    def test_output_shape_and_backward_pass(self):
        model = DelayedDownsampleResNet()
        inputs = jax.random.normal(
            jax.random.PRNGKey(1), (2, 64, 64, 7), dtype=jnp.float32
        )
        params = model.init(jax.random.PRNGKey(0), inputs)

        output = model.apply(params, inputs)
        self.assertEqual(output.shape, (2, 128))
        self.assertTrue(bool(jnp.all(jnp.isfinite(output))))

        loss, gradients = jax.jit(
            jax.value_and_grad(
                lambda variables: jnp.mean(model.apply(variables, inputs) ** 2)
            )
        )(params)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(
            all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gradients))
        )

    def test_maps_net_selects_residual_encoder(self):
        model = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_delayed")
        maps = [jnp.zeros((2, 64, 64), dtype=jnp.float32) for _ in range(7)]
        params = model.init(jax.random.PRNGKey(0), maps)

        output = model.apply(params, maps)
        self.assertEqual(output.shape, (2, 128))

    def test_maps_net_keeps_atari_encoder_as_default(self):
        model = MapsNet(map_min_max=(-1, 1))
        maps = [jnp.zeros((2, 64, 64), dtype=jnp.float32) for _ in range(7)]
        params = model.init(jax.random.PRNGKey(0), maps)

        output = model.apply(params, maps)
        self.assertEqual(output.shape, (2, 32))

    def test_full_policy_initializes_with_residual_encoder(self):
        class Config(dict):
            __getattr__ = dict.__getitem__

        config = Config(
            clip_action_maps=False,
            loaded_max=6,
            local_map_normalization_bounds=(-1, 1),
            map_encoder="resnet_delayed",
            maps_net_normalization_bounds=(-1, 1),
            model_core="mlp",
            model_size="base",
            num_prev_actions=5,
        )
        env = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
        )

        model, params = get_model_ready(jax.random.PRNGKey(0), config, env)
        self.assertEqual(model.map_encoder, "resnet_delayed")
        self.assertGreater(sum(x.size for x in jax.tree.leaves(params)), 0)


if __name__ == "__main__":
    unittest.main()
