import unittest

import numpy as np

from scripts.diagnose_b0_collapse import action_diagnostics, run_length_encode


class DiagnoseB0CollapseTest(unittest.TestCase):
    def test_run_length_encode(self):
        self.assertEqual(
            run_length_encode(np.array([0, 0, 6, 6, 1], dtype=np.int32)),
            [
                {"action": 0, "name": "FORWARD", "count": 2},
                {"action": 6, "name": "DO", "count": 2},
                {"action": 1, "name": "BACKWARD", "count": 1},
            ],
        )

    def test_action_diagnostics_counts_do_and_effects(self):
        actions = np.array([[0], [6], [6], [1]], dtype=np.int32)
        effects = np.array([[True], [False], [True], [True]], dtype=bool)
        rows = [{"map_id": "map-0", "primary_cell": "cell"}]
        task_per_map = [
            {
                "slot_index_zero_based": 0,
                "terminal_absolute": 0.5,
                "terminal_dig": 0.75,
                "terminal_dump_volume": 0.5,
                "success": False,
            }
        ]
        result = action_diagnostics(
            actions,
            effects,
            np.array([4], dtype=np.int32),
            rows,
            task_per_map,
        )
        cell = result["cells"]["cell"]
        self.assertEqual(cell["maps_with_do"], 1)
        self.assertEqual(cell["maps_with_effective_do"], 1)
        self.assertEqual(cell["median_first_do_step"], 1.0)
        self.assertEqual(cell["action_counts"]["DO"], 2)
        self.assertEqual(cell["effective_action_counts"]["DO"], 1)
        self.assertEqual(cell["dominant_action"], "DO")


if __name__ == "__main__":
    unittest.main()
