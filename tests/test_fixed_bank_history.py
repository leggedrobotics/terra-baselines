import unittest

from aggregate_fixed_bank_history import aggregate_history


def record(
    update,
    successes=(8, 8, 8, 8),
    *,
    integrity_passed=True,
    complete=True,
):
    condition_results = {
        f"foundation-{index}": {
            "successes": successes_here,
            "episodes": 8,
            "success_rate": successes_here / 8,
        }
        for index, successes_here in enumerate(successes)
    }
    family_successes = sum(successes)
    performance_passed = family_successes >= 26 and all(
        successes_here >= 6 for successes_here in successes
    )
    return {
        "split": "development",
        "stratum": "starter",
        "policy_mode": "deterministic",
        "checkpoint_update": update,
        "checkpoint_sha256": f"{update:064x}",
        "exact_manifest_enumeration": complete,
        "reset_verification": {"passed": True},
        "summary": {
            "overall": {
                "successes": family_successes,
                "episodes": 32,
                "success_rate": family_successes / 32,
            },
            "mastery_gate": {
                "performance_passed": performance_passed,
                "integrity_passed": integrity_passed,
                "passed": performance_passed and integrity_passed,
            },
            "integrity": {"passed": integrity_passed},
            "by_family": {
                "foundation": {
                    "successes": family_successes,
                    "episodes": 32,
                    "success_rate": family_successes / 32,
                },
            },
            "by_primary_cell": condition_results,
        },
    }


class FixedBankHistoryTest(unittest.TestCase):
    def test_single_checkpoint_is_not_consecutive_mastery(self):
        result = aggregate_history([record(100)])
        history = result["histories"][0]
        self.assertFalse(history["two_consecutive_mastery"])
        self.assertIsNone(history["first_mastery_update"])

    def test_references_use_lower_of_two_passing_counts(self):
        result = aggregate_history(
            [
                record(100),
                record(200, (7, 7, 7, 6)),
            ]
        )
        history = result["histories"][0]
        self.assertTrue(history["two_consecutive_mastery"])
        self.assertEqual(history["first_mastery_update"], 200)
        self.assertEqual(
            history["mastery_references"]["families"]["foundation"],
            {
                "successes": 27,
                "episodes": 32,
                "retention_threshold": 26,
            },
        )
        self.assertEqual(
            history["mastery_references"]["conditions"]["foundation-0"],
            {
                "successes": 7,
                "episodes": 8,
                "retention_threshold": 6,
            },
        )

    def test_n8_and_n32_retention_thresholds_are_count_based(self):
        result = aggregate_history(
            [
                record(100),
                record(200),
                record(300, (7, 8, 8, 8)),
                record(400, (6, 8, 8, 8)),
            ]
        )
        history = result["histories"][0]
        first, second = history["retention_checks"]
        self.assertTrue(first["passed"])
        self.assertEqual(
            first["panels"]["conditions"]["foundation-0"]["threshold"],
            7,
        )
        self.assertEqual(
            first["panels"]["families"]["foundation"]["threshold"],
            31,
        )
        self.assertFalse(second["passed"])
        self.assertFalse(history["rollback_triggered"])

    def test_invalid_and_incomplete_evaluations_do_not_update_streaks(self):
        result = aggregate_history(
            [
                record(100),
                record(200),
                record(300, (6, 8, 8, 8)),
                record(
                    400,
                    (0, 0, 0, 0),
                    integrity_passed=False,
                ),
                record(500, complete=False),
                record(600, (6, 8, 8, 8)),
            ]
        )
        history = result["histories"][0]
        self.assertEqual(
            [check["passed"] for check in history["retention_checks"]],
            [False, None, None, False],
        )
        self.assertTrue(history["rollback_triggered"])
        self.assertEqual(history["rollback_trigger"]["checkpoint_update"], 600)

    def test_valid_pass_resets_retention_failure_streak(self):
        result = aggregate_history(
            [
                record(100),
                record(200),
                record(300, (6, 8, 8, 8)),
                record(400, (7, 8, 8, 8)),
                record(500, (6, 8, 8, 8)),
            ]
        )
        history = result["histories"][0]
        self.assertFalse(history["rollback_triggered"])
        self.assertEqual(
            history["retention_failure_streaks"]["conditions"]["foundation-0"],
            1,
        )

    def test_invalid_evaluation_does_not_break_mastery_streak(self):
        result = aggregate_history(
            [
                record(100),
                record(200, integrity_passed=False),
                record(300),
            ]
        )
        history = result["histories"][0]
        self.assertTrue(history["two_consecutive_mastery"])
        self.assertEqual(history["first_mastery_update"], 300)


if __name__ == "__main__":
    unittest.main()
