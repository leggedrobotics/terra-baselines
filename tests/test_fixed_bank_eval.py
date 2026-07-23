import unittest

import numpy as np

from eval_fixed_bank import (
    exact_reset_keys,
    grouped_results,
    selected_map_indices,
)


class FixedBankEvalTest(unittest.TestCase):
    def test_reset_keys_enumerate_every_slot_once(self):
        keys = exact_reset_keys(64)
        np.testing.assert_array_equal(selected_map_indices(keys, 64), np.arange(64))

    def test_grouping_preserves_family_and_primary_cells(self):
        rows = []
        for family in ("foundation", "trench"):
            for cell_index in range(4):
                for episode in range(8):
                    rows.append(
                        {
                            "slot_index": len(rows) + 1,
                            "map_id": (f"{family}-{cell_index}-{episode}"),
                            "family": family,
                            "primary_cell": f"{family}-{cell_index}",
                        }
                    )
        successes = np.ones(64, dtype=bool)
        terminations = np.ones(64, dtype=bool)
        lengths = np.arange(64, dtype=np.int32) + 1
        per_map, summary = grouped_results(rows, successes, terminations, lengths)
        self.assertEqual(len(per_map), 64)
        self.assertEqual(summary["by_family"]["foundation"]["successes"], 32)
        self.assertTrue(summary["mastery_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
