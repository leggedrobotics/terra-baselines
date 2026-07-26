import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.verify_f0_training import assert_equal_trees, manifest_digest


class VerifyF0TrainingTest(unittest.TestCase):
    def test_equal_trees_require_exact_structure_and_values(self):
        first = {
            "model": (
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3], dtype=np.int32),
            )
        }
        second = {
            "model": (
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3], dtype=np.int32),
            )
        }

        self.assertEqual(assert_equal_trees(first, second, "checkpoint"), 2)
        second["model"][0][1] = 4.0
        with self.assertRaisesRegex(RuntimeError, "leaf 0 differs"):
            assert_equal_trees(first, second, "checkpoint")
        with self.assertRaisesRegex(RuntimeError, "tree definitions differ"):
            assert_equal_trees(first, {"other": first["model"]}, "checkpoint")

    def test_manifest_digest_is_ordered_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.write_text("one\n")
            second.write_text("two\n")

            digest = manifest_digest([first, second])
            self.assertEqual(digest, manifest_digest([first, second]))
            self.assertNotEqual(digest, manifest_digest([second, first]))
            second.write_text("changed\n")
            self.assertNotEqual(digest, manifest_digest([first, second]))


if __name__ == "__main__":
    unittest.main()
