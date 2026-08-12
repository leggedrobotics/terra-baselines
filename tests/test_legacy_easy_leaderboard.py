import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_legacy_easy_leaderboard import build_leaderboard
from build_legacy_easy_leaderboard import render_markdown

CHECKPOINT_SHA = "c" * 64
BANK_SHA = "b" * 64
BANK_FILES_SHA = "f" * 64
PROTOCOL_SHA = "6" * 64
SOURCE_SHA = "7" * 64
TERRA_REVISION = "8" * 40
BASELINES_REVISION = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values):
    exact = sum(value == 1.0 for value in values)
    return {
        "mean": sum(values) / len(values),
        "macro_completion": sum(values) / len(values),
        "successes": exact,
        "episodes": len(values),
        "success_rate": exact / len(values),
    }


def _record(panel: str, mode: str) -> dict:
    rows = []
    conditions = {
        "foundation-all-around": ("foundation", (1.0, 0.8)),
        "trench-both-sides": ("trench", (0.4, 0.0)),
    }
    for condition, (family, completion_values) in conditions.items():
        for index, completion in enumerate(completion_values):
            rows.append(
                {
                    "slot_index": len(rows) + 1,
                    "episode_id": f"{panel}-{condition}-{index}",
                    "split": panel,
                    "family": family,
                    "primary_cell": condition,
                    "success": completion == 1.0,
                    "terminal_absolute": completion,
                    "steps": 10,
                    "no_effect_action_count": index + 1,
                }
            )
    condition_stats = {
        condition: _stats(values) for condition, (_, values) in conditions.items()
    }
    family_stats = {
        family: {
            **_stats(values),
            "macro_completion": sum(values) / len(values),
        }
        for family, values in (
            ("foundation", (1.0, 0.8)),
            ("trench", (0.4, 0.0)),
        )
    }
    all_values = [row["terminal_absolute"] for row in rows]
    return {
        "schema": "terra_fixed_bank_eval_v4",
        "completion_contract": "exact_visible_dump_v1",
        "checkpoint": "/checkpoints/policy.pkl",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_update": 4000,
        "treatment_fingerprint": {"sha256": "d" * 64},
        "split": panel,
        "stratum": "legacy_easy_capability_floor",
        "manifest": f"/bank/{panel}/manifest.jsonl",
        "manifest_sha256": ("1" if panel == "promotion" else "2") * 64,
        "horizon": 450,
        "deterministic": mode == "deterministic",
        "policy_mode": mode,
        "seed": 20260803,
        "exact_manifest_enumeration": True,
        "explicit_episode_bank": {
            "schema": "terra_legacy_easy_explicit_episode_bank_v1",
            "name": "Terra Legacy-Easy v1",
            "release_id": "terra-legacy-easy-v1-current-episodes-v1",
            "diagnostic_only": True,
            "included_in_constrained_macro": False,
            "panel": panel,
            "maps_path": panel,
            "slot_count": 4,
            "condition_count": 2,
            "maps_per_condition": 2,
            "condition_balanced": True,
            "terra_revision": TERRA_REVISION,
            "environment_protocol_sha256": PROTOCOL_SHA,
            "source_registry_sha256": SOURCE_SHA,
            "episode_bank_sha256": BANK_SHA,
            "environment_protocol_file_sha256": "e" * 64,
            "manifest_sha256": ("1" if panel == "promotion" else "2") * 64,
            "initial_states_sha256": ("3" if panel == "promotion" else "4") * 64,
            "files_manifest_sha256": BANK_FILES_SHA,
            "episode_id_schema": "episode-v1",
            "initial_agent_state_schema": "agent-v1",
        },
        "summary": {
            "overall": {
                "successes": 1,
                "episodes": 4,
                "success_rate": 0.25,
                "terminations": 4,
            },
            "integrity": {"passed": True, "failure_count": 0},
            "graded": {
                "available": True,
                "macro_completion": sum(
                    stats["mean"] for stats in condition_stats.values()
                )
                / len(condition_stats),
                "micro": {"mean": sum(all_values) / len(all_values)},
                "by_primary_cell": condition_stats,
                "by_family": family_stats,
            },
        },
        "per_map": rows,
    }


def _write_manifest(policy_root: Path) -> None:
    manifest = policy_root / "files.sha256"
    lines = []
    for path in sorted(path for path in policy_root.iterdir() if path != manifest):
        if path.is_file():
            lines.append(f"{_sha256(path)}  ./{path.name}\n")
    manifest.write_text("".join(lines), encoding="utf-8")


def _write_policy(policy_root: Path) -> None:
    policy_root.mkdir()
    for panel in ("promotion", "development"):
        for mode in ("deterministic", "sampled"):
            (policy_root / f"{panel}_{mode}.json").write_text(
                json.dumps([_record(panel, mode)]), encoding="utf-8"
            )
            (policy_root / f"{panel}_{mode}.log").write_text(
                "evaluation passed\n", encoding="utf-8"
            )
    (policy_root / "receipt.env").write_text(
        "\n".join(
            (
                "schema=terra_legacy_easy_v1_euler_eval_receipt_v1",
                "status=PASSED",
                f"policy_label={policy_root.name}",
                "checkpoint_path=/checkpoints/policy.pkl",
                f"checkpoint_sha256={CHECKPOINT_SHA}",
                f"baselines_revision={BASELINES_REVISION}",
                f"terra_revision={TERRA_REVISION}",
                f"episode_bank_json_sha256={BANK_SHA}",
                f"episode_bank_files_sha256={BANK_FILES_SHA}",
                "panels=promotion,development",
                "policy_modes=deterministic,sampled",
                "sampled_seed=20260803",
                "horizon=450",
                "completion_contract=exact_visible_dump_v1",
                "slurm_job_id=1234",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(policy_root)


def _write_matrix(path: Path, labels=("policy-a",)) -> None:
    rows = ["policy_label\tcheckpoint_path\tcheckpoint_sha256\n"]
    rows.extend(
        f"{label}\t/checkpoints/policy.pkl\t{CHECKPOINT_SHA}\n" for label in labels
    )
    path.write_text("".join(rows), encoding="utf-8")


class LegacyEasyLeaderboardTest(unittest.TestCase):
    def test_reports_condition_balanced_policy_family_and_tail_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            _write_policy(results / "policy-a")
            matrix = root / "policies.tsv"
            _write_matrix(matrix)

            leaderboard = build_leaderboard(results, matrix)
            self.assertEqual(leaderboard["policy_count"], 1)
            self.assertEqual(len(leaderboard["evaluations"]), 4)
            evaluation = leaderboard["evaluations"][0]
            self.assertAlmostEqual(evaluation["macro_completion"], 0.55)
            self.assertAlmostEqual(
                evaluation["by_family"]["foundation"]["macro_completion"], 0.9
            )
            self.assertEqual(evaluation["overall"]["exact_completions"], 1)
            self.assertAlmostEqual(evaluation["overall"]["completion_p10"], 0.12)
            self.assertAlmostEqual(evaluation["overall"]["no_effect_action_rate"], 0.15)
            self.assertEqual(
                leaderboard["contract"]["environment_protocol_sha256"],
                PROTOCOL_SHA,
            )
            markdown = render_markdown(leaderboard)
            self.assertIn("## Condition breakdown", markdown)
            self.assertIn("foundation-all-around", markdown)
            self.assertIn(CHECKPOINT_SHA, markdown)

    def test_rejects_missing_policy_or_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            _write_policy(results / "policy-a")
            matrix = root / "policies.tsv"
            _write_matrix(matrix, ("policy-a", "policy-b"))
            with self.assertRaisesRegex(ValueError, "missing=.*policy-b"):
                build_leaderboard(results, matrix)

            _write_matrix(matrix)
            (results / "policy-a" / "development_sampled.json").unlink()
            _write_manifest(results / "policy-a")
            with self.assertRaisesRegex(ValueError, "result JSON set differs"):
                build_leaderboard(results, matrix)

    def test_rejects_duplicate_labels_and_episode_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            _write_policy(results / "policy-a")
            matrix = root / "policies.tsv"
            _write_matrix(matrix, ("policy-a", "policy-a"))
            with self.assertRaisesRegex(ValueError, "duplicate policy labels"):
                build_leaderboard(results, matrix)

            _write_matrix(matrix)
            path = results / "policy-a" / "promotion_sampled.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            records[0]["per_map"][1]["episode_id"] = records[0]["per_map"][0][
                "episode_id"
            ]
            path.write_text(json.dumps(records), encoding="utf-8")
            _write_manifest(results / "policy-a")
            with self.assertRaisesRegex(ValueError, "duplicate explicit episode IDs"):
                build_leaderboard(results, matrix)

    def test_rejects_mixed_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            _write_policy(results / "policy-a")
            matrix = root / "policies.tsv"
            _write_matrix(matrix)
            path = results / "policy-a" / "development_sampled.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            records[0]["explicit_episode_bank"]["environment_protocol_sha256"] = (
                "9" * 64
            )
            path.write_text(json.dumps(records), encoding="utf-8")
            _write_manifest(results / "policy-a")
            with self.assertRaisesRegex(ValueError, "mix contracts"):
                build_leaderboard(results, matrix)

    def test_rejects_seed_metric_or_condition_identity_drift(self):
        def change_seed(record):
            record["seed"] = 7

        def remove_no_effect(record):
            for row in record["per_map"]:
                row.pop("no_effect_action_count")

        def rename_condition(record):
            old = "foundation-all-around"
            new = "foundation-renamed"
            for row in record["per_map"]:
                if row["primary_cell"] == old:
                    row["primary_cell"] = new
            graded = record["summary"]["graded"]
            graded["by_primary_cell"][new] = graded["by_primary_cell"].pop(old)

        cases = (
            (change_seed, "expected seed=20260803"),
            (remove_no_effect, "require no-effect action counts"),
            (rename_condition, "mix episode condition identities"),
        )
        for mutate, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                results = root / "results"
                results.mkdir()
                _write_policy(results / "policy-a")
                matrix = root / "policies.tsv"
                _write_matrix(matrix)
                path = results / "policy-a" / "promotion_sampled.json"
                records = json.loads(path.read_text(encoding="utf-8"))
                mutate(records[0])
                path.write_text(json.dumps(records), encoding="utf-8")
                _write_manifest(results / "policy-a")
                with self.assertRaisesRegex(ValueError, message):
                    build_leaderboard(results, matrix)


if __name__ == "__main__":
    unittest.main()
