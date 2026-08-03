import collections
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import numpy as np

import terra.config
from scripts.transplant_legacy_checkpoint import RECEIPT_SCHEMA
from scripts.transplant_legacy_checkpoint import _load_historical
from scripts.transplant_legacy_checkpoint import model_content_sha256
from scripts.transplant_legacy_checkpoint import transplant_checkpoint
from train_mixed import _validate_checkpoint_architecture
from utils import helpers

E8_SOURCE = Path(
    "/home/lorenzo/moleworks/.artifacts/terra_e8_checkpoint_20260802/"
    "terra-sv3-E8-multitask-ks-euler-2026-07-22-13-54-23_FINAL.pkl"
)
E8_SOURCE_SHA256 = "f364a5dbfe3329542317273819b65cf5fc12a7329fef2d9126d8e3f251a9f674"
E8_SKELETON = Path(
    "/home/lorenzo/moleworks/.artifacts/terra_e8_checkpoint_20260802/"
    "e8_params_in_current_eval_skeleton.pkl"
)
E8_SKELETON_SHA256 = "1a417c8844cad8822cb826b6ed9f8edfffdf2197581ebca2299ac6925ba27c86"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leaves(model):
    return {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(model)[0]
    }


class LegacyCheckpointTransplantTest(unittest.TestCase):
    def test_synthetic_env_config_arity_and_state_discard(self):
        old_env_config = collections.namedtuple(
            "EnvConfig", [f"field_{index}" for index in range(32)]
        )
        old_env_config.__module__ = "terra.config"
        source_model = {
            "params": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)}
        }
        skeleton_model = {"params": {"kernel": np.zeros((2, 3), dtype=np.float32)}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "historical.pkl"
            skeleton = root / "skeleton.pkl"
            output = root / "current.pkl"
            with mock.patch.object(terra.config, "EnvConfig", old_env_config):
                source.write_bytes(
                    pickle.dumps(
                        {
                            "model": source_model,
                            "next_update": 73,
                            "env_config": old_env_config(*range(32)),
                            "train_config": {"historical": True},
                            "optimizer_state": {"must": "be discarded"},
                            "pooled_sampler_state": {"must": "be discarded"},
                        },
                        pickle.HIGHEST_PROTOCOL,
                    )
                )
            skeleton.write_bytes(
                pickle.dumps(
                    {
                        "checkpoint_version": 2,
                        "model": skeleton_model,
                        "train_config": {"current": True},
                        "env_config": {"current": True},
                        "optimizer_state": {"skeleton": "also omitted"},
                    },
                    pickle.HIGHEST_PROTOCOL,
                )
            )

            source_sha256 = _file_sha256(source)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                transplant_checkpoint(
                    source=source,
                    expected_source_sha256="0" * 64,
                    skeleton=skeleton,
                    output=output,
                    source_revision="old",
                    skeleton_revision="current",
                    role="synthetic",
                )
            self.assertFalse(output.exists())

            receipt = transplant_checkpoint(
                source=source,
                expected_source_sha256=source_sha256,
                skeleton=skeleton,
                output=output,
                source_revision="old",
                skeleton_revision="current",
                role="synthetic",
            )
            loaded = helpers.load_pkl_object(str(output))
            self.assertEqual(
                set(loaded),
                {
                    "checkpoint_version",
                    "train_config",
                    "env_config",
                    "model",
                    "next_update",
                    "compatibility_transplant",
                },
            )
            self.assertEqual(loaded["next_update"], 73)
            self.assertEqual(loaded["train_config"], {"current": True})
            self.assertEqual(loaded["env_config"], {"current": True})
            np.testing.assert_array_equal(
                loaded["model"]["params"]["kernel"],
                source_model["params"]["kernel"],
            )
            self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
            self.assertEqual(receipt["source_checkpoint"]["raw_sha256"], source_sha256)
            self.assertEqual(receipt["source_checkpoint"]["next_update"], 73)
            self.assertTrue(receipt["historical_state_discarded"])
            with self.assertRaises(FileExistsError):
                transplant_checkpoint(
                    source=source,
                    expected_source_sha256=source_sha256,
                    skeleton=skeleton,
                    output=output,
                    source_revision="old",
                    skeleton_revision="current",
                    role="synthetic",
                )

    @unittest.skipUnless(
        E8_SOURCE.is_file() and E8_SKELETON.is_file(),
        "local historical E8 artifacts are unavailable",
    )
    def test_actual_e8_transplant_preserves_model_and_inputs(self):
        self.assertEqual(_file_sha256(E8_SOURCE), E8_SOURCE_SHA256)
        self.assertEqual(_file_sha256(E8_SKELETON), E8_SKELETON_SHA256)
        source_bytes = E8_SOURCE.read_bytes()
        source_checkpoint = _load_historical(source_bytes)
        source_model_sha256, _, _ = model_content_sha256(source_checkpoint["model"])
        source_before = _file_sha256(E8_SOURCE)
        skeleton_before = _file_sha256(E8_SKELETON)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "e8_current_compat.pkl"
            receipt = transplant_checkpoint(
                source=E8_SOURCE,
                expected_source_sha256=E8_SOURCE_SHA256,
                skeleton=E8_SKELETON,
                output=output,
                source_revision="spatial-v3-e8",
                skeleton_revision="current-legacy-easy-eval",
                role="legacy-easy-e8-current-compat",
            )
            helpers.register_checkpoint_config_classes()
            loaded = helpers.load_pkl_object(str(output))
            _validate_checkpoint_architecture(loaded, loaded["train_config"])

        self.assertEqual(_file_sha256(E8_SOURCE), source_before)
        self.assertEqual(_file_sha256(E8_SKELETON), skeleton_before)
        self.assertEqual(loaded["next_update"], source_checkpoint["next_update"])
        self.assertEqual(receipt["model_content_sha256"], source_model_sha256)
        self.assertEqual(
            receipt["evaluation_skeleton"]["raw_sha256"], E8_SKELETON_SHA256
        )
        source_leaves = _leaves(source_checkpoint["model"])
        output_leaves = _leaves(loaded["model"])
        self.assertEqual(source_leaves.keys(), output_leaves.keys())
        for path, source_leaf in source_leaves.items():
            self.assertEqual(source_leaf.dtype, output_leaves[path].dtype, path)
            np.testing.assert_array_equal(
                source_leaf, output_leaves[path], err_msg=path
            )


if __name__ == "__main__":
    unittest.main()
