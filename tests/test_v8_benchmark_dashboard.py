import json
import hashlib
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.build_v8_benchmark_dashboard import (
    aligned_rows,
    behavior_comparison,
    build_dashboard_data,
    choose_review_rows,
    issue_tags,
    render_html,
    validate_pair,
    write_episode_csv,
)
from scripts.render_v8_fixed_panel_gifs import (
    longest_equal_run,
    longest_true_run,
    summarize_trace,
    terminal_cycle,
    verify_manifest_reset_receipt,
    verify_prepared_manifest_reset,
)


def row(slot, *, success, no_effect, steps=100, terminal=1.0, stall=0.0,
        off_zone=0.0, loaded=0.0, dig=1.0, carry=0.0):
    return {
        "slot_index": slot,
        "episode_id": f"episode-{slot}",
        "map_id": f"map-{slot}",
        "primary_cell": "fnd-a" if slot <= 2 else "trn-b",
        "family": "foundation" if slot <= 2 else "trench",
        "reset_seed": 100 + slot,
        "success": success,
        "steps": steps,
        "no_effect_action_count": no_effect,
        "terminal_soil_fraction": terminal,
        "dig_fraction": dig,
        "off_zone_staged_soil_fraction": off_zone,
        "loaded_soil_fraction": loaded,
        "stall_age_saturated_decision_fraction": stall,
        "maximum_carry_work_normalized": carry,
        "productive_workspace_cycles": 3,
        "integrity_failure": False,
    }


def record(rows, sha):
    return {
        "schema": "terra_fixed_bank_eval_v4",
        "split": "promotion",
        "stratum": "all",
        "deterministic": True,
        "policy_mode": "deterministic",
        "completion_contract": "exact_visible_dump_v1",
        "r2_protocol_receipt": {"schema": "r2"},
        "accepted_bank": {"environment_protocol_sha256": "environment"},
        "reset_verification": {"passed": True},
        "summary": {"integrity": {"passed": True}},
        "manifest_sha256": "manifest",
        "horizon": 450,
        "seed": 7,
        "checkpoint_sha256": sha,
        "checkpoint_update": 10,
        "per_map": rows,
    }


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.reference = record(
            [
                row(1, success=False, no_effect=90, terminal=0.8, stall=0.7),
                row(2, success=True, no_effect=1),
                row(3, success=False, no_effect=70, terminal=0.4),
                row(4, success=True, no_effect=2, carry=0.2),
            ],
            "a" * 64,
        )
        self.candidate = record(
            [
                row(1, success=True, no_effect=2, steps=80),
                row(2, success=False, no_effect=90, terminal=0.9, loaded=0.1),
                row(3, success=False, no_effect=80, terminal=0.3, off_zone=0.2),
                row(4, success=True, no_effect=1, steps=70, carry=0.3),
            ],
            "b" * 64,
        )

    def test_alignment_and_diagnostic_labels(self):
        validate_pair(self.reference, self.candidate)
        rows = aligned_rows(self.reference, self.candidate)
        self.assertEqual(
            [item["outcome"] for item in rows],
            ["conversion", "regression", "persistent failure", "persistent success"],
        )
        self.assertIn("loaded endpoint", issue_tags(self.candidate["per_map"][1]))
        self.assertIn("staged-soil residue", issue_tags(self.candidate["per_map"][2]))
        selected = choose_review_rows(rows, 4)
        self.assertEqual({item["slot"] for item in selected}, {1, 2, 3, 4})

    def test_behavior_deltas_use_only_paired_common_successes(self):
        for record_, travel in ((self.reference, [10000.0, 1000.0, 4000.0, 20.0]),
                                (self.candidate, [100.0, 10000.0, 5000.0, 10.0])):
            for episode, distance in zip(record_["per_map"], travel):
                episode.update(behavior_metrics_available=True, base_travel_m=distance,
                               productive_base_stances=2, mean_workspace_dig_area_m2=5.0,
                               lateral_dig_volume_fraction=0.25)
        self.reference["per_map"][3]["mean_workspace_dig_area_m2"] = float("nan")
        comparison = behavior_comparison(aligned_rows(self.reference, self.candidate))
        self.assertEqual(comparison["successes"]["reference"]["metrics"]["base_travel_m"]["mean"], 510.0)
        self.assertEqual(comparison["successes"]["candidate"]["metrics"]["base_travel_m"]["mean"], 55.0)
        paired = comparison["paired_common_successes"]
        self.assertEqual(paired["episodes"], 1)
        self.assertEqual(paired["metrics"]["base_travel_m"]["delta"],
                         {"count": 1, "mean": -10.0, "median": -10.0, "p90": -10.0})
        self.assertEqual(paired["metrics"]["mean_workspace_dig_area_m2"]["delta"]["count"], 0)
        self.assertIsNone(paired["metrics"]["mean_workspace_dig_area_m2"]["delta"]["mean"])
        self.assertEqual(comparison["all_episodes"]["reference"]["episodes"], 4)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "episodes.csv"
            write_episode_csv({"maps": aligned_rows(self.reference, self.candidate)}, csv_path)
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["paired_common_success_delta_base_travel_m"], "")
            self.assertEqual(rows[3]["paired_common_success_delta_base_travel_m"], "-10.0")
            self.assertEqual(rows[3]["reference_mean_workspace_dig_area_m2"], "")
            self.assertEqual(rows[3]["candidate_productive_base_stances"], "2.0")
            self.assertEqual(rows[3]["candidate_lateral_dig_volume_fraction"], "0.25")

    def test_legacy_behavior_is_unavailable_and_does_not_reuse_workspace_cycles(self):
        rows = aligned_rows(self.reference, self.candidate)
        self.assertEqual(rows[3]["reference"]["workspace_cycles"], 3.0)
        self.assertIsNone(rows[3]["reference"]["productive_base_stances"])
        self.assertIsNone(rows[3]["reference"]["base_travel_m"])
        comparison = behavior_comparison(rows)
        self.assertEqual(comparison["all_episodes"]["reference"]["available_episodes"], 0)
        self.assertIsNone(comparison["paired_common_successes"]["metrics"]["base_travel_m"]["delta"]["mean"])
        self.candidate["per_map"][3]["success"] = False
        comparison = behavior_comparison(aligned_rows(self.reference, self.candidate))
        self.assertEqual(comparison["paired_common_successes"]["episodes"], 0)
        self.assertIsNone(comparison["paired_common_successes"]["metrics"]["base_travel_m"]["reference"]["mean"])

    def test_identity_mismatch_fails(self):
        self.candidate["per_map"][0]["episode_id"] = "different"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_pair(self.reference, self.candidate)

    def test_stale_media_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            reference_path.write_text(json.dumps([self.reference]))
            candidate_path.write_text(json.dumps([self.candidate]))
            selection = choose_review_rows(
                aligned_rows(self.reference, self.candidate), 4
            )
            media = root / "media"
            for label, path, checkpoint in (
                ("ff", reference_path, "a" * 64),
                ("gru", candidate_path, "b" * 64),
            ):
                label_dir = media / label
                label_dir.mkdir(parents=True)
                receipt = {
                    "full_panel_terminal_parity_verified": True,
                    "full_panel_no_effect_count_parity_verified": True,
                    "checkpoint_sha256": checkpoint,
                    "fixed_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "manifest_sha256": "stale-manifest",
                    "panel_maps": 4,
                    "horizon": 450,
                    "seed": 7,
                    "deterministic": True,
                    "canonical_forward_chunk": 120,
                    "selected_slots": [item["slot"] for item in selection],
                    "episodes": [],
                }
                (label_dir / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "manifest_sha256 mismatch"):
                build_dashboard_data(
                    self.reference,
                    self.candidate,
                    reference_label="ff",
                    candidate_label="gru",
                    reference_path=reference_path,
                    candidate_path=candidate_path,
                    media_dir=media,
                    output_dir=root / "dashboard",
                    review_limit=4,
                )

    def test_portable_html_and_json_are_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            reference_path.write_text(json.dumps([self.reference]))
            candidate_path.write_text(json.dumps([self.candidate]))
            data = build_dashboard_data(
                self.reference,
                self.candidate,
                reference_label="ff",
                candidate_label="gru",
                reference_path=reference_path,
                candidate_path=candidate_path,
                media_dir=None,
                output_dir=root / "dashboard",
                review_limit=4,
            )
            encoded = json.dumps(data, allow_nan=False)
            document = render_html(data)
            self.assertEqual(data["summary"]["exact_delta"], 0)
            self.assertIn("Terra benchmark: gru vs ff", document)
            self.assertIn(encoded[:20], json.dumps(data, allow_nan=False))
            self.assertIn("episode-dot", document)
            self.assertIn("maps both policies solved", document)
            self.assertIn("unavailable", document)
            self.assertIn("Legacy productive workspace cycles", document)
            self.assertIn("not physical safety", document)
            self.assertIn("Area per productive stance", document)
            self.assertIn("by_family", data["behavior"])
            self.assertIn("fnd-a", data["behavior"]["by_primary_cell"])
            self.assertNotIn("NaN", document)
            csv_path = root / "episodes.csv"
            write_episode_csv(data, csv_path)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("candidate_no_effect_rate", csv_text)
            self.assertIn("candidate_base_travel_m", csv_text)
            self.assertIn("candidate_productive_base_stances", csv_text)
            self.assertIn("candidate_mean_workspace_dig_area_m2", csv_text)
            self.assertIn("candidate_lateral_dig_volume_fraction", csv_text)
            self.assertIn("paired_common_success_delta_base_travel_m", csv_text)
            with csv_path.open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(csv_rows[0]["candidate_base_travel_m"], "")
            self.assertIn("episode-1", csv_text)

    def test_trace_summary_surfaces_no_effect_and_terminal_cycle(self):
        trace = []
        for step in range(1, 13):
            trace.append(
                {
                    "step": step,
                    "input_hash": f"hash-{step % 2}",
                    "action": step % 2,
                    "action_had_effect": step < 3,
                    "material_changed": step == 2,
                }
            )
        self.assertEqual(longest_true_run([False, True, True, False]), 2)
        self.assertEqual(longest_equal_run([1, 1, 2, 2, 2]), 3)
        cycle = terminal_cycle(trace)
        self.assertEqual(cycle["period"], 2)
        summary = summarize_trace(trace)
        self.assertEqual(summary["maximum_no_effect_streak"], 10)
        self.assertEqual(summary["last_material_change_step"], 2)
        self.assertEqual(summary["terminal_observation_action_cycle"]["period"], 2)
        self.assertIsNone(summary["terminal_recurrent_state_action_cycle"])

    def test_recurrent_hidden_state_is_part_of_policy_cycle(self):
        trace = []
        for step in range(1, 13):
            trace.append(
                {
                    "step": step,
                    "input_hash": "same-input",
                    "hidden_hash": f"hidden-{step}",
                    "action": 1,
                    "action_had_effect": False,
                    "material_changed": False,
                }
            )
        summary = summarize_trace(trace)
        self.assertEqual(summary["terminal_observation_action_cycle"]["period"], 1)
        self.assertIsNone(summary["terminal_recurrent_state_action_cycle"])

    def test_renderer_verifies_input_seed_receipt_not_advanced_state_key(self):
        state_keys = np.asarray([[0, 101], [0, 202]], dtype=np.uint32)
        base_receipt = {"passed": True, "slots": 2, "env_steps_min": 0}
        seed_receipt = {
            "passed": True,
            "map_selection_decoupled": True,
            "sha256": hashlib.sha256(
                np.ascontiguousarray(state_keys).tobytes()
            ).hexdigest(),
        }
        fixed_record = {
            "reset_verification": {
                **base_receipt,
                "manifest_episode_seeds": seed_receipt,
            }
        }
        self.assertEqual(
            verify_manifest_reset_receipt(
                fixed_record,
                base_receipt,
                state_keys,
            ),
            fixed_record["reset_verification"],
        )

        stale = json.loads(json.dumps(fixed_record))
        stale["reset_verification"]["manifest_episode_seeds"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "reset receipt"):
            verify_manifest_reset_receipt(stale, base_receipt, state_keys)

        with patch(
            "scripts.render_v8_fixed_panel_gifs.verify_exact_reset",
            return_value=base_receipt,
        ) as reset_check:
            observed = verify_prepared_manifest_reset(
                object(),
                object(),
                Path("promotion"),
                2,
                object(),
                state_keys,
                fixed_record,
            )
        self.assertEqual(observed, fixed_record["reset_verification"])
        self.assertNotIn("expected_state_keys", reset_check.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
