import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from eval_fixed_bank import (
    LEGACY_COMPLETION_CONTRACT,
    environment_completion_contract,
    exact_reset_keys,
    grouped_results,
    selected_map_indices,
    verify_exact_reset,
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

    def test_integrity_failure_blocks_mastery(self):
        rows = []
        for family in ("foundation", "trench"):
            for cell_index in range(4):
                for episode in range(8):
                    rows.append(
                        {
                            "slot_index": len(rows) + 1,
                            "map_id": f"{family}-{cell_index}-{episode}",
                            "family": family,
                            "primary_cell": f"{family}-{cell_index}",
                        }
                    )
        count = len(rows)
        integrity = {
            "maximum_mass_residual": np.zeros(count, dtype=np.int32),
            "target_mutation": np.zeros(count, dtype=bool),
            "obstacle_mutation": np.zeros(count, dtype=bool),
            "nonfinite_state": np.zeros(count, dtype=bool),
            "termination_disagreement": np.zeros(count, dtype=bool),
            "slot_index_disagreement": np.zeros(count, dtype=bool),
        }
        integrity["maximum_mass_residual"][3] = 1
        _, summary = grouped_results(
            rows,
            np.ones(count, dtype=bool),
            np.ones(count, dtype=bool),
            np.ones(count, dtype=np.int32),
            integrity_metrics=integrity,
        )
        self.assertTrue(summary["mastery_gate"]["performance_passed"])
        self.assertFalse(summary["mastery_gate"]["integrity_passed"])
        self.assertFalse(summary["mastery_gate"]["passed"])
        self.assertEqual(summary["integrity"]["failure_count"], 1)

    def test_exact_reset_verifies_all_layers_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays = {
                "images": np.array([[0, -1], [1, 0]], dtype=np.int8),
                "actions": np.array([[0, 0], [0, 0]], dtype=np.int8),
                "occupancy": np.array([[0, 0], [0, 1]], dtype=np.int8),
                "dumpability": np.array([[1, 1], [1, 0]], dtype=np.bool_),
                "distance": np.array([[0.2, 0.1], [0.0, 1.0]], dtype=np.float32),
            }
            for name, array in arrays.items():
                (directory / name).mkdir()
                np.save(directory / name / "img_1.npy", array)
            (directory / "metadata").mkdir()
            (directory / "metadata" / "trench_1.json").write_text(
                json.dumps({"axes_ABC": []}) + "\n"
            )
            trench_axes = np.full((3, 3), -97.0)
            foundation_axes = np.full((64, 3), -97.0)
            world = SimpleNamespace(
                target_map=SimpleNamespace(map=arrays["images"][None]),
                action_map=SimpleNamespace(map=arrays["actions"][None]),
                padding_mask=SimpleNamespace(map=arrays["occupancy"][None]),
                dumpability_mask_init=SimpleNamespace(map=arrays["dumpability"][None]),
                relocation_distance_map=arrays["distance"][None],
                trench_axes=trench_axes[None],
                trench_type=np.array([-1]),
                foundation_border_axes=foundation_axes[None],
                foundation_border_type=np.array([-1]),
            )
            timestep = SimpleNamespace(
                state=SimpleNamespace(
                    world=world,
                    env_steps=np.array([0], dtype=np.int32),
                )
            )
            env = SimpleNamespace(
                reset=lambda *_: timestep,
                maps_buffer=SimpleNamespace(
                    trench_axes=trench_axes[None, None],
                    trench_types=np.array([[-1]]),
                    foundation_border_axes=foundation_axes[None, None],
                    foundation_border_types=np.array([[-1]]),
                ),
            )
            receipt = verify_exact_reset(
                env,
                None,
                None,
                directory,
                1,
            )
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["env_steps_max"], 0)
            self.assertEqual(len(receipt["layer_sha256"]), 6)


if __name__ == "__main__":
    unittest.main()
