import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.trench_benchmark_gallery import (
    _load_benchmark,
    safe_slug,
    select_representatives,
    trench_geometry,
)


def row(slot, condition, success, dig, steps):
    return {
        "slot_index": slot,
        "primary_cell": condition,
        "family": "trench",
        "success": success,
        "dig_fraction": dig,
        "steps": steps,
        "no_effect_action_count": 0,
    }


class TrenchBenchmarkGalleryTest(unittest.TestCase):
    def test_geometry_parser(self):
        self.assertEqual(trench_geometry("trn-straight-side2"), "straight")
        self.assertEqual(trench_geometry("trn-net4-side2-s"), "net4")
        with self.assertRaises(ValueError):
            trench_geometry("fnd-slab-ring3x")

    def test_selection_balances_success_and_failure(self):
        rows = [
            row(1, "trn-tee-side2", True, 1.0, 60),
            row(2, "trn-tee-side2", True, 1.0, 90),
            row(3, "trn-tee-side2", False, 0.2, 450),
            row(4, "trn-tee-side2", False, 0.8, 450),
        ]
        selected = select_representatives(rows, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({item["slot_index"] for item in selected}), 3)
        self.assertTrue(any(item["success"] for item in selected))
        self.assertTrue(any(not item["success"] for item in selected))
        self.assertTrue(
            any(item["gallery_role"] == "highest-progress stall" for item in selected)
        )

    def test_net4_is_marked_as_diagnostic(self):
        selected = select_representatives(
            [row(8, "trn-net4-side2", False, 0.3, 450)], 1
        )
        self.assertTrue(selected[0]["structural_exclusion"])

    def test_safe_slug(self):
        self.assertEqual(safe_slug("Tee / slot 17"), "tee-slot-17")

    def test_benchmark_contract_is_bound_to_requested_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pkl"
            checkpoint.write_bytes(b"checkpoint")
            receipt = root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    [
                        {
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                            "deterministic": True,
                            "exact_manifest_enumeration": True,
                            "per_map": [row(1, "trn-straight-side2", True, 1.0, 10)],
                            "split": "development",
                            "accepted_bank": {
                                "evaluation_panel_family": "gate_main",
                                "terra_revision": "protocol-revision",
                            },
                        }
                    ]
                )
            )
            record = _load_benchmark(
                receipt,
                checkpoint,
                panel_family="gate_main",
                accepted_panel="development",
                terra_revision="protocol-revision",
            )
            self.assertEqual(record["split"], "development")
            with self.assertRaises(ValueError):
                _load_benchmark(
                    receipt,
                    checkpoint,
                    panel_family="main",
                    accepted_panel="development",
                    terra_revision="protocol-revision",
                )


if __name__ == "__main__":
    unittest.main()
