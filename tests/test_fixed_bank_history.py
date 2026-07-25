import unittest

from aggregate_fixed_bank_history import aggregate_history


def record(update, passed, foundation_rate=0.9, trench_rate=0.9):
    return {
        "split": "development",
        "stratum": "starter",
        "policy_mode": "deterministic",
        "checkpoint_update": update,
        "checkpoint_sha256": f"{update:064x}",
        "summary": {
            "mastery_gate": {"passed": passed},
            "integrity": {"passed": True},
            "by_family": {
                "foundation": {"success_rate": foundation_rate},
                "trench": {"success_rate": trench_rate},
            },
        },
    }


class FixedBankHistoryTest(unittest.TestCase):
    def test_single_checkpoint_is_not_consecutive_mastery(self):
        result = aggregate_history([record(100, True)])
        history = result["histories"][0]
        self.assertFalse(history["two_consecutive_mastery"])
        self.assertIsNone(history["first_mastery_update"])

    def test_two_passes_establish_mastery_and_later_regression_fails_retention(
        self,
    ):
        result = aggregate_history(
            [
                record(100, True),
                record(200, True),
                record(300, True, foundation_rate=0.80),
            ]
        )
        history = result["histories"][0]
        self.assertTrue(history["two_consecutive_mastery"])
        self.assertEqual(history["first_mastery_update"], 200)
        self.assertFalse(history["retention_passed"])
        self.assertTrue(
            history["retention_checks"][0]["family_regressions"]["foundation"]
        )


if __name__ == "__main__":
    unittest.main()
