import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.verify_b0_training import (
    assert_equal_aggregate_prefix,
    assert_equal_trees,
    manifest_digest,
)


class VerifyB0TrainingTest(unittest.TestCase):
    def test_equal_trees_require_exact_structure_and_values(self):
        first = {"model": np.array([1.0, 2.0], dtype=np.float32)}
        second = {"model": np.array([1.0, 2.0], dtype=np.float32)}
        self.assertEqual(
            assert_equal_trees(first, second, "checkpoint"),
            1,
        )
        second["model"][1] = 4.0
        with self.assertRaisesRegex(RuntimeError, "leaf 0 differs"):
            assert_equal_trees(first, second, "checkpoint")

    def test_manifest_digest_is_ordered_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.write_text("one\n")
            second.write_text("two\n")
            digest = manifest_digest([first, second])
            self.assertEqual(
                digest,
                manifest_digest([first, second]),
            )
            self.assertNotEqual(
                digest,
                manifest_digest([second, first]),
            )

    def test_aggregate_prefix_ignores_only_run_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            reference.write_text(
                json.dumps({"update": 1, "run_name": "short", "totals": {"done": 2}})
            )
            candidate.write_text(
                json.dumps({"update": 1, "run_name": "long", "totals": {"done": 2}})
            )
            self.assertEqual(
                assert_equal_aggregate_prefix([reference], [candidate]),
                1,
            )
            candidate.write_text(
                json.dumps({"update": 1, "run_name": "long", "totals": {"done": 3}})
            )
            with self.assertRaisesRegex(RuntimeError, "prefix differs at update 1"):
                assert_equal_aggregate_prefix([reference], [candidate])


if __name__ == "__main__":
    unittest.main()
