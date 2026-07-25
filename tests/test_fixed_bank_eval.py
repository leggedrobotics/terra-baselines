import unittest
from unittest.mock import patch

import numpy as np

from eval_fixed_bank import (
    LEGACY_COMPLETION_CONTRACT,
    environment_completion_contract,
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

    def test_grouping_separates_success_timeout_and_completion(self):
        rows = [
            {
                "slot_index": index + 1,
                "map_id": f"map-{index}",
                "family": "foundation",
                "primary_cell": "easy",
            }
            for index in range(4)
        ]
        successes = np.array([True, False, True, False])
        terminations = np.array([True, True, True, False])
        lengths = np.array([10, 450, 450, 450])
        absolute = np.array([1.0, 0.5, 1.0, 0.25], dtype=np.float32)
        per_map, summary = grouped_results(
            rows,
            successes,
            terminations,
            lengths,
            horizon=450,
            completion_metrics={"terminal_absolute": absolute},
        )

        self.assertEqual(
            [row["termination_reason"] for row in per_map],
            [
                "task_done",
                "timeout",
                "task_done_and_timeout",
                "horizon_censored",
            ],
        )
        self.assertEqual(per_map[1]["terminal_absolute"], 0.5)
        self.assertEqual(summary["termination_reasons"]["timeout"], 1)

    def test_environment_contract_is_explicit(self):
        self.assertEqual(
            environment_completion_contract(),
            "exact_visible_dump_v1",
        )
        with patch.dict("sys.modules", {"terra.state": None}):
            self.assertEqual(
                environment_completion_contract(),
                LEGACY_COMPLETION_CONTRACT,
            )


if __name__ == "__main__":
    unittest.main()
