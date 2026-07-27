import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.diagnose_b0_policy_cross import (
    TRACE_COMPONENTS,
    aggregate_cross,
    payload_for,
    selected_failed_traces,
    write_partial,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SBATCH_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "euler_curriculum_recovery_v1"
    / "diagnose_b0_policy_cross.sbatch"
)


def record(split, mode, seed, successes, episodes, cell_successes):
    return {
        "split": split,
        "policy_mode": mode,
        "seed": seed,
        "summary": {
            "overall": {"successes": successes, "episodes": episodes},
            "cells": {
                "cell": {
                    "successes": cell_successes,
                    "episodes": episodes,
                }
            },
        },
    }


class DiagnoseB0PolicyCrossTest(unittest.TestCase):
    def test_aggregates_crossed_rates(self):
        records = [
            record("train", "deterministic", 0, 6, 8, 6),
            record("train", "sampled", 1, 8, 8, 8),
            record("train", "sampled", 2, 6, 8, 6),
            record("development", "deterministic", 0, 0, 8, 0),
            record("development", "sampled", 1, 4, 8, 4),
            record("development", "sampled", 2, 2, 8, 2),
        ]
        summary = aggregate_cross(records)
        self.assertEqual(summary["train"]["deterministic"]["success_rate"], 0.75)
        self.assertEqual(summary["train"]["sampled"]["success_rate"], 0.875)
        self.assertEqual(
            summary["development"]["sampled"]["per_cell"]["cell"],
            {"successes": 6, "episodes": 16, "success_rate": 0.375},
        )
        self.assertEqual(
            summary["rate_differences"]["train_minus_development_sampled"],
            0.5,
        )

    def test_requires_every_cross_cell(self):
        with self.assertRaisesRegex(ValueError, "development requires"):
            aggregate_cross(
                [
                    record("train", "deterministic", 0, 0, 8, 0),
                    record("train", "sampled", 1, 0, 8, 0),
                ]
            )

    def test_failed_trace_selection_deduplicates_high_and_zero(self):
        rows = [
            {
                "map_id": "map-0",
                "source_id": "source-0",
                "primary_cell": "cell",
                "slot_index": 1,
            }
        ]
        summary = {
            "per_map": [
                {
                    **rows[0],
                    "success": False,
                    "steps": 3,
                    "terminal_absolute": 0.0,
                }
            ]
        }
        zeros = np.zeros((3, 1), dtype=np.float32)
        stats = {
            "action_sequence": np.zeros((3, 1), dtype=np.int32),
            "action_had_effect_sequence": np.ones((3, 1), dtype=bool),
            "completion_sequence": {name: zeros for name in TRACE_COMPONENTS},
        }
        selected = selected_failed_traces(rows, summary, stats)["cell"]
        self.assertEqual(len(selected["traces"]), 1)
        self.assertEqual(
            selected["traces"][0]["selection"],
            ["high_completion", "zero_progress"],
        )

    def test_partial_payload_is_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cross.json"
            partial = write_partial(output, {"checkpoint": 3800})
            self.assertEqual(json.loads(partial.read_text()), {"checkpoint": 3800})
            self.assertFalse(Path(f"{partial}.tmp").exists())

    def test_single_checkpoint_payload_retains_legacy_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pkl"
            checkpoint_path.write_bytes(b"checkpoint")
            records = [
                {
                    **record(split, mode, seed, 1, 1, 1),
                    "checkpoint_update": 900,
                }
                for split in ("train", "development")
                for mode, seed in (("deterministic", 0), ("sampled", 1))
            ]
            payload = payload_for(
                SimpleNamespace(panel="foundation_distance"),
                Path(directory),
                [(checkpoint_path, {"next_update": 900})],
                {"train": {"passed": True}, "development": {"passed": True}},
                records,
                status="complete",
            )

            self.assertEqual(payload["checkpoint"], str(checkpoint_path))
            self.assertEqual(payload["checkpoint_update"], 900)
            self.assertEqual(
                payload["checkpoint_sha256"],
                payload["checkpoints"][0]["sha256"],
            )
            self.assertEqual(
                payload["cross_summary"],
                payload["cross_summary_by_checkpoint"]["900"],
            )

    def test_sbatch_preserves_sealed_source_and_single_artifact_contracts(self):
        source = SBATCH_PATH.read_text()
        self.assertIn(
            'PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"',
            source,
        )
        self.assertIn('if [[ "$PYTHONDONTWRITEBYTECODE" != 1 ]]; then', source)
        self.assertLess(
            source.index("export PYTHONDONTWRITEBYTECODE"),
            source.index('SITE_PACKAGES="$("$VENV/bin/python"'),
        )
        self.assertIn("if ((${#CHECKPOINT_UPDATES[@]} == 1)); then", source)
        self.assertIn(
            'OUTPUT_BASENAME="policy_cross_update_${CHECKPOINT_LABEL}.json"',
            source,
        )
        self.assertIn(
            'COMPLETE_MARKER_BASENAME="POLICY_CROSS_UPDATE_'
            '${CHECKPOINT_LABEL}_COMPLETE"',
            source,
        )
        self.assertIn(
            'OUTPUT_BASENAME="policy_cross_updates_${CHECKPOINT_LABEL}.json"',
            source,
        )
        self.assertIn('OUTPUT="$PANEL_ROOT/$OUTPUT_BASENAME"', source)
        self.assertIn('touch "$PANEL_ROOT/$COMPLETE_MARKER_BASENAME"', source)
        self.assertIn('echo "$COMPLETE_MESSAGE"', source)


if __name__ == "__main__":
    unittest.main()
