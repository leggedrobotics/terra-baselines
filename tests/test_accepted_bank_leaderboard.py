import copy
import unittest

from build_accepted_bank_leaderboard import (
    architecture_label,
    classify_condition,
    factor_axis_summaries,
    followup_campaign_sha256,
    grouped_stats,
    paired_delta_rows,
    quantile,
    register_unique_arm,
    validate_checkpoint_sequences,
    validate_record,
)


def _manifest_row(index, condition):
    return {
        "candidate_sample_index": index,
        "environment_protocol_sha256": "p" * 64,
        "episode_id": f"episode-{index}",
        "family": "foundation",
        "identity_slot_multiplicity": 1,
        "map_id": f"map-{index}",
        "pair_slot_id": f"pair-{index}",
        "primary_cell": condition,
        "reset_seed": index,
        "scenario_id": f"scenario-{index}",
        "slot_index": index,
        "slot_weight": 1.0,
        "source_id": f"source-{index}",
        "split": "promotion",
        "stratum": "test",
    }


def _valid_record():
    manifest = [_manifest_row(1, "fnd-a"), _manifest_row(2, "fnd-b")]
    per_map = []
    for terminal, row in zip((0.2, 0.8), manifest):
        result = dict(row)
        result.update(
            success=False,
            terminal_absolute=terminal,
            steps=10,
            no_effect_action_count=0,
        )
        per_map.append(result)
    record = {
        "split": "promotion",
        "exact_manifest_enumeration": True,
        "policy_mode": "deterministic",
        "deterministic": True,
        "accepted_bank": {
            "environment_protocol_sha256": "p" * 64,
            "source_registry_sha256": "s" * 64,
            "schema": "terra_curriculum_loader_bank_v1",
        },
        "per_map": per_map,
        "summary": {
            "integrity": {"passed": True},
            "overall": {"success_rate": 0.0},
            "graded": {
                "micro": {"mean": 0.5, "p10": 0.26},
                "macro_completion": 0.5,
                "by_primary_cell": {
                    "fnd-a": {"mean": 0.2},
                    "fnd-b": {"mean": 0.8},
                },
            },
        },
    }
    conditions = {
        "fnd-a": {"family": "foundation"},
        "fnd-b": {"family": "foundation"},
    }
    identity = {
        "environment_protocol_sha256": "p" * 64,
        "source_registry_sha256": "s" * 64,
        "schema": "terra_curriculum_loader_bank_v1",
    }
    return record, manifest, conditions, identity


class AcceptedBankLeaderboardTest(unittest.TestCase):
    def test_followup_campaign_identity_ignores_treatment_only_fields(self):
        shared = {
            "phase": "screen",
            "seed": "7",
            "updates": "2000",
            "terra_revision": "terra",
            "terra_baselines_revision": "baselines",
            "parent_checkpoint_sha256": "parent",
            "teacher_checkpoint_sha256": "teacher",
            "initialization": "params_only_warm",
        }
        adaptive = {
            **shared,
            "arm": "G-MEDIUM-ADAPTIVE-WARM",
            "condition_sampler": "adaptive",
            "architecture": "medium",
            "initial_checkpoint_sha256": "parent",
        }
        deep = {
            **shared,
            "arm": "G-DEEP-ADAPTIVE-WARM",
            "condition_sampler": "adaptive",
            "architecture": "deep",
            "initial_checkpoint_sha256": "grown",
        }
        self.assertEqual(
            followup_campaign_sha256(adaptive), followup_campaign_sha256(deep)
        )
        self.assertNotEqual(
            followup_campaign_sha256(adaptive),
            followup_campaign_sha256({**adaptive, "seed": "8"}),
        )

    def test_grouped_stats_reports_near_complete_and_no_effect_rates(self):
        rows = [
            {
                "terminal_absolute": 0.95,
                "success": False,
                "steps": 10,
                "no_effect_action_count": 2,
            },
            {
                "terminal_absolute": 0.94,
                "success": False,
                "steps": 30,
                "no_effect_action_count": 6,
            },
        ]
        stats = grouped_stats(rows)
        self.assertEqual(stats["near_complete_count"], 1)
        self.assertEqual(stats["near_complete_rate"], 0.5)
        self.assertEqual(stats["total_steps"], 40)
        self.assertEqual(stats["no_effect_action_count"], 8)
        self.assertEqual(stats["no_effect_action_rate"], 0.2)

    def test_architecture_label_distinguishes_resnet_depth(self):
        common = {
            "model_size": "medium",
            "map_encoder": "resnet_spatial_8x8_se",
            "model_core": "mlp",
            "critic_hidden_dims": (512, 256),
        }
        baseline = architecture_label(common)
        self.assertEqual(
            baseline,
            "medium:resnet_spatial_8x8_se:mlp:critic-512-256",
        )
        deep = architecture_label(
            {
                **common,
                "resnet_stage_channels": (24, 48, 64, 96),
                "resnet_blocks_per_stage": (2, 2, 3, 3),
            }
        )
        self.assertNotEqual(baseline, deep)
        self.assertIn("blocks-2x2x3x3", deep)

    def test_quantile_matches_linear_interpolation(self):
        self.assertAlmostEqual(quantile([0.0, 1.0, 2.0, 3.0], 0.10), 0.3)
        self.assertAlmostEqual(quantile([0.0, 1.0, 2.0, 3.0], 0.25), 0.75)

    def test_foundation_composed_condition(self):
        self.assertEqual(
            classify_condition("fnd-proc-side1-road", "foundation", "Composed"),
            {
                "geometry": "proc",
                "dump_layout": "side1",
                "capacity": "generous",
                "distance": "unspec",
                "site": "road",
                "scale": "std",
                "factor_axis": "composed",
            },
        )

    def test_foundation_distance_condition(self):
        result = classify_condition("fnd-slab-apron-d16", "foundation", "One-axis")
        self.assertEqual(result["geometry"], "slab")
        self.assertEqual(result["dump_layout"], "apron")
        self.assertEqual(result["distance"], "d16")
        self.assertEqual(result["factor_axis"], "distance")

    def test_trench_mini_geometry_condition(self):
        result = classify_condition("trn-net4-side2-s", "trench", "One-axis")
        self.assertEqual(result["geometry"], "net4")
        self.assertEqual(result["dump_layout"], "side2")
        self.assertEqual(result["scale"], "s")
        self.assertEqual(result["factor_axis"], "geometry")

    def test_unknown_token_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown condition token"):
            classify_condition("trn-straight-side2-mystery", "trench", "One-axis")

    def test_swapped_episode_ids_fail_identity_validation(self):
        record, manifest, conditions, identity = _valid_record()
        record = copy.deepcopy(record)
        record["per_map"][0]["episode_id"], record["per_map"][1]["episode_id"] = (
            record["per_map"][1]["episode_id"],
            record["per_map"][0]["episode_id"],
        )
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_record(record, "promotion", manifest, conditions, identity)

    def test_sampled_policy_fails_validation(self):
        record, manifest, conditions, identity = _valid_record()
        record["policy_mode"] = "sampled"
        record["deterministic"] = False
        with self.assertRaisesRegex(ValueError, "not deterministic"):
            validate_record(record, "promotion", manifest, conditions, identity)

    def test_cross_panel_checkpoint_mismatch_fails(self):
        records = {
            "promotion": [{"checkpoint_update": 2000, "checkpoint_sha256": "a" * 64}],
            "development": [{"checkpoint_update": 2000, "checkpoint_sha256": "b" * 64}],
        }
        with self.assertRaisesRegex(ValueError, "checkpoint identities differ"):
            validate_checkpoint_sequences(records, 2000, "G-TEST")

    def test_duplicate_arm_fails(self):
        seen = set()
        register_unique_arm(seen, "G-TEST")
        with self.assertRaisesRegex(ValueError, "duplicate arm"):
            register_unique_arm(seen, "G-TEST")

    def test_factor_axis_summary_uses_condition_macro(self):
        common = {"arm": "G-TEST", "panel": "promotion", "checkpoint_update": 1}
        conditions = {
            "fnd-a": {"family": "foundation", "factor_axis": "distance"},
            "fnd-b": {"family": "foundation", "factor_axis": "distance"},
        }
        per_map = [
            {
                "primary_cell": "fnd-a",
                "terminal_absolute": 0.2,
                "success": False,
                "steps": 10,
                "no_effect_action_count": 2,
            },
            {
                "primary_cell": "fnd-b",
                "terminal_absolute": 0.8,
                "success": False,
                "steps": 30,
                "no_effect_action_count": 3,
            },
        ]
        by_condition = {
            condition_id: grouped_stats(
                [row for row in per_map if row["primary_cell"] == condition_id]
            )
            for condition_id in conditions
        }
        summaries = factor_axis_summaries(common, per_map, conditions, by_condition)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["factor_axis"], "distance")
        self.assertEqual(summaries[0]["macro_completion"], 0.5)
        self.assertEqual(summaries[0]["no_effect_action_rate"], 0.125)

    def test_paired_deltas_require_exact_episode_sequence(self):
        conditions = {
            "fnd-a": {"family": "foundation", "factor_axis": "distance"},
            "trn-a": {"family": "trench", "factor_axis": "geometry"},
        }

        def evaluation(values):
            return {
                "checkpoint_sha256": "a" * 64,
                "seed": 7,
                "per_map": [
                    {
                        "episode_id": "one",
                        "primary_cell": "fnd-a",
                        "terminal_absolute": values[0],
                        "success": False,
                        "steps": 10,
                        "no_effect_action_count": 2,
                    },
                    {
                        "episode_id": "two",
                        "primary_cell": "trn-a",
                        "terminal_absolute": values[1],
                        "success": values[1] == 1.0,
                        "steps": 20,
                        "no_effect_action_count": 1,
                    },
                ],
            }

        key = ("campaign", "G-UNIFORM", "promotion", 500)
        treatment_key = ("campaign", "G-ADAPTIVE", "promotion", 500)
        evaluations = {
            key: evaluation((0.2, 0.8)),
            treatment_key: evaluation((0.4, 1.0)),
        }
        rows = paired_delta_rows(evaluations, conditions)
        policy = next(row for row in rows if row["scope_type"] == "policy")
        self.assertAlmostEqual(policy["delta_macro_completion"], 0.2)
        self.assertEqual(policy["delta_near_complete_count"], 1)
        self.assertEqual(policy["delta_exact_successes"], 1)
        distance = next(
            row
            for row in rows
            if row["scope_type"] == "factor_axis" and row["scope_id"] == "distance"
        )
        self.assertAlmostEqual(distance["delta_macro_completion"], 0.2)

        evaluations[treatment_key]["per_map"].reverse()
        with self.assertRaisesRegex(ValueError, "paired episode sequence mismatch"):
            paired_delta_rows(evaluations, conditions)

    def test_paired_deltas_fail_when_same_campaign_control_is_missing(self):
        evaluations = {
            ("campaign-a", "G-ADAPTIVE", "promotion", 500): {
                "checkpoint_sha256": "a" * 64,
                "seed": 7,
                "per_map": [],
            },
            ("campaign-b", "G-UNIFORM", "promotion", 500): {
                "checkpoint_sha256": "b" * 64,
                "seed": 7,
                "per_map": [],
            },
        }
        with self.assertRaisesRegex(ValueError, "missing paired control"):
            paired_delta_rows(evaluations, {})


if __name__ == "__main__":
    unittest.main()
