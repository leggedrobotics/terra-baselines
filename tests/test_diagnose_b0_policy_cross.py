import unittest

from scripts.diagnose_b0_policy_cross import aggregate_cross


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


if __name__ == "__main__":
    unittest.main()
