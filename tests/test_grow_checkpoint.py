import os
import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig, MapsDimsConfig
from utils.models import Spatial8x8MapResNet, get_model_ready
import utils.helpers as helpers
from scripts.grow_checkpoint import (
    build_target_config,
    grow_params,
    _leaf_map,
    _param_count,
    _update_out_train_config,
)


class _Config(dict):
    __getattr__ = dict.__getitem__


def _full_model_config(map_encoder, model_size="base", **extra):
    return _Config(
        clip_action_maps=False,
        loaded_max=6,
        local_map_normalization_bounds=(-1, 1),
        map_encoder=map_encoder,
        maps_net_normalization_bounds=(-1, 1),
        model_core="mlp",
        model_size=model_size,
        num_prev_actions=5,
        **extra,
    )


def _dummy_env():
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )


class GrowFunctionPreservationTest(unittest.TestCase):
    def test_v2_deepened_reproduces_source_output(self):
        # Only depth is added (one extra block appended to the last stage), so
        # zero-init of the added block's second conv makes it an exact identity.
        source = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 2))
        target = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 3))
        inputs = jax.random.normal(jax.random.PRNGKey(5), (3, 64, 64, 7))
        source_params = source.init(jax.random.PRNGKey(0), inputs)
        target_params = target.init(jax.random.PRNGKey(1), inputs)

        grown_params, report = grow_params(source_params, target_params)

        source_out = source.apply(source_params, inputs)
        grown_out = target.apply(grown_params, inputs)
        self.assertLess(
            float(jnp.max(jnp.abs(source_out - grown_out))), 1e-5
        )

        # The added block's second conv kernel is exactly zero.
        grown_leaves = _leaf_map(grown_params)
        added_key = "['params']['ResidualMapBlock_6']['Conv_1']['kernel']"
        self.assertIn(added_key, grown_leaves)
        self.assertTrue(bool(jnp.all(grown_leaves[added_key] == 0)))
        # Exactly one zero-init leaf (the added block's second conv).
        zero_leaves = [e for e in report if e["category"] == "zero-init"]
        self.assertEqual(len(zero_leaves), 1)

        # Existing shape-matched leaves are verbatim copies.
        source_leaves = _leaf_map(source_params)
        copied = [e for e in report if e["category"] == "copied"]
        self.assertGreater(len(copied), 0)
        for entry in copied:
            np.testing.assert_array_equal(
                np.asarray(grown_leaves[entry["key"]]),
                np.asarray(source_leaves[entry["key"]]),
            )

    def test_deepened_added_block_is_identity_on_positive_input(self):
        # Sanity on the exactness of the identity claim.
        source = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 2))
        target = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 3))
        inputs = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (2, 64, 64, 7)))
        sp = source.init(jax.random.PRNGKey(0), inputs)
        tp = target.init(jax.random.PRNGKey(1), inputs)
        grown, _ = grow_params(sp, tp)
        np.testing.assert_allclose(
            np.asarray(source.apply(sp, inputs)),
            np.asarray(target.apply(grown, inputs)),
            atol=1e-6,
        )


class GrowFullModelTest(unittest.TestCase):
    def test_base_v2_to_medium_v3_grows_without_errors(self):
        _, source_params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_8x8", "base"),
            _dummy_env(),
        )
        _, target_params = get_model_ready(
            jax.random.PRNGKey(1),
            _full_model_config("resnet_spatial_v3", "medium"),
            _dummy_env(),
        )

        grown_params, report = grow_params(source_params, target_params)

        # Structure and total param count follow the target exactly.
        self.assertEqual(
            jax.tree_util.tree_structure(grown_params),
            jax.tree_util.tree_structure(target_params),
        )
        self.assertEqual(_param_count(grown_params), _param_count(target_params))

        # Exact copies where shapes matched.
        source_leaves = _leaf_map(source_params)
        grown_leaves = _leaf_map(grown_params)
        for entry in report:
            if entry["category"] == "copied":
                np.testing.assert_array_equal(
                    np.asarray(grown_leaves[entry["key"]]),
                    np.asarray(source_leaves[entry["key"]]),
                )

        # Added-depth block: second conv kernel zeroed.
        zero_leaves = [e for e in report if e["category"] == "zero-init"]
        self.assertEqual(len(zero_leaves), 1)
        for entry in zero_leaves:
            self.assertIn("ResidualMapBlock", entry["key"])
            self.assertIn("Conv_1", entry["key"])
            self.assertTrue(bool(jnp.all(grown_leaves[entry["key"]] == 0)))

        # v3 stem conv reads the 11 assembled channels; it is a sliced copy that
        # keeps the source's 7 input channels in the leading slice.
        sliced = [e for e in report if e["category"] == "sliced"]
        self.assertGreater(len(sliced), 0)

    def test_output_checkpoint_is_consumable(self):
        # Build a source checkpoint, grow it, and confirm the output pkl rebuilds
        # into the target architecture (as --resume_from / --teacher_checkpoint do).
        src_config = _full_model_config("resnet_spatial_8x8", "base")
        _, source_params = get_model_ready(
            jax.random.PRNGKey(0), src_config, _dummy_env()
        )
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.pkl")
            out_path = os.path.join(tmp, "grown.pkl")
            helpers.save_pkl_object(
                {"model": source_params, "train_config": dict(src_config), "env_config": None},
                src_path,
            )
            checkpoint = helpers.load_pkl_object(src_path)

            overrides = {
                "map_encoder": "resnet_spatial_v3",
                "model_size": "medium",
                "model_core": None,
                "critic_hidden_dims": (512, 256),
                "encoder_compute_dtype": None,
            }
            target_config = build_target_config(checkpoint["train_config"], overrides)
            _, target_params = get_model_ready(
                jax.random.PRNGKey(1), target_config, _dummy_env()
            )
            grown_params, _ = grow_params(checkpoint["model"], target_params)

            out_checkpoint = dict(checkpoint)
            out_checkpoint["model"] = grown_params
            out_checkpoint["train_config"] = _update_out_train_config(
                checkpoint["train_config"], target_config
            )
            helpers.save_pkl_object(out_checkpoint, out_path)

            # Reload and rebuild the model from the stored train_config.
            reloaded = helpers.load_pkl_object(out_path)
            self.assertEqual(reloaded["train_config"]["map_encoder"], "resnet_spatial_v3")
            self.assertEqual(reloaded["train_config"]["model_size"], "medium")
            self.assertEqual(
                tuple(reloaded["train_config"]["critic_hidden_dims"]), (512, 256)
            )
            rebuild_config = build_target_config(reloaded["train_config"], {})
            _, rebuilt_params = get_model_ready(
                jax.random.PRNGKey(2), rebuild_config, _dummy_env()
            )
            self.assertEqual(
                jax.tree_util.tree_structure(reloaded["model"]),
                jax.tree_util.tree_structure(rebuilt_params),
            )


if __name__ == "__main__":
    unittest.main()
