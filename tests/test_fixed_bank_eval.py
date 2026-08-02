import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np

from eval_fixed_bank import (
    LEGACY_COMPLETION_CONTRACT,
    checkpoint_treatment_fingerprint,
    comparison_gate,
    environment_completion_contract,
    exact_reset_keys,
    grouped_results,
    manifest_reset_keys,
    selected_map_indices,
    validate_checkpoint_sequence,
    verify_exact_reset,
)


class FixedBankEvalTest(unittest.TestCase):
    def test_reset_keys_enumerate_every_slot_once(self):
        keys = exact_reset_keys(64)
        np.testing.assert_array_equal(selected_map_indices(keys, 64), np.arange(64))

    def test_reset_keys_use_the_frozen_partitionable_threefry_mapping(self):
        self.assertEqual(str(jax.config.jax_default_prng_impl), "threefry2x32")
        self.assertTrue(jax.config.jax_threefry_partitionable)
        keys = exact_reset_keys(4)
        self.assertEqual([int(key[1]) for key in np.asarray(keys)], [1, 3, 0, 2])
        np.testing.assert_array_equal(selected_map_indices(keys, 4), np.arange(4))

    def test_manifest_reset_keys_reject_a_runtime_prng_mode_mismatch(self):
        try:
            jax.config.update("jax_threefry_partitionable", False)
            with self.assertRaisesRegex(RuntimeError, "reset PRNG contract"):
                manifest_reset_keys([], 0, "a" * 64)
        finally:
            jax.config.update("jax_threefry_partitionable", True)

    def test_manifest_reset_keys_pin_the_protocol_and_slot(self):
        keys = exact_reset_keys(4)
        seeds = [int(key[1]) for key in np.asarray(keys)]
        for seed, expected in zip(seeds, np.asarray(keys)):
            np.testing.assert_array_equal(jax.random.PRNGKey(seed), expected)
        protocol = "a" * 64
        rows = [
            {
                "slot_index": index + 1,
                "reset_seed": seed,
                "environment_protocol_sha256": protocol,
            }
            for index, seed in enumerate(seeds)
        ]
        observed = manifest_reset_keys(rows, 4, protocol)
        np.testing.assert_array_equal(selected_map_indices(observed, 4), np.arange(4))
        rows[0]["environment_protocol_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "stale protocol"):
            manifest_reset_keys(rows, 4, protocol)

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
        self.assertTrue(summary["integrity"]["passed"])
        self.assertFalse(summary["graded"]["available"])

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
        self.assertEqual(summary["integrity"]["failure_count"], 1)
        self.assertFalse(summary["integrity"]["passed"])

    def test_graded_metrics_are_condition_macro_not_map_weighted(self):
        rows = [
            {
                "slot_index": index + 1,
                "map_id": f"easy-{index}",
                "family": "foundation",
                "primary_cell": "easy",
            }
            for index in range(9)
        ]
        rows.append(
            {
                "slot_index": 10,
                "map_id": "hard-0",
                "family": "trench",
                "primary_cell": "hard",
            }
        )
        completion = np.asarray([1.0] * 9 + [0.0], dtype=np.float32)
        _, summary = grouped_results(
            rows,
            completion == 1.0,
            np.ones(10, dtype=bool),
            np.ones(10, dtype=np.int32),
            completion_metrics={"terminal_absolute": completion},
        )
        graded = summary["graded"]
        self.assertEqual(graded["micro"]["mean"], 0.9)
        self.assertEqual(graded["macro_completion"], 0.5)
        self.assertEqual(graded["family_macro_completion"], 0.5)
        self.assertEqual(graded["by_family"]["foundation"]["macro_completion"], 1.0)
        self.assertEqual(graded["by_family"]["trench"]["macro_completion"], 0.0)
        self.assertEqual(graded["worst_condition"], "hard")
        self.assertEqual(graded["worst_condition_completion"], 0.0)

    def test_comparison_gate_uses_one_map_or_continuous_macro_progress(self):
        rows = []
        for condition in ("a", "b"):
            for index in range(5):
                rows.append(
                    {
                        "slot_index": len(rows) + 1,
                        "map_id": f"{condition}-{index}",
                        "family": "foundation",
                        "primary_cell": condition,
                    }
                )

        def summary(completion):
            completion = np.asarray(completion, dtype=np.float32)
            _, result = grouped_results(
                rows,
                completion == 1.0,
                np.ones(10, dtype=bool),
                np.ones(10, dtype=np.int32),
                completion_metrics={"terminal_absolute": completion},
            )
            return result

        reference = summary([0.50] * 10)
        macro_candidate = summary([0.52] * 10)
        gate = comparison_gate(reference, macro_candidate)
        self.assertEqual(gate["exact_map_gain"], 0)
        self.assertAlmostEqual(gate["exact_rate_quantum"], 0.1)
        self.assertTrue(gate["progress_passed"])
        self.assertTrue(gate["passed"])

        exact_candidate = summary([1.0] + [0.50] * 9)
        gate = comparison_gate(reference, exact_candidate)
        self.assertEqual(gate["exact_map_gain"], 1)
        self.assertTrue(gate["passed"])

    def test_comparison_gate_blocks_lower_tail_regression(self):
        rows = [
            {
                "slot_index": index + 1,
                "map_id": f"map-{index}",
                "family": "foundation",
                "primary_cell": f"condition-{index // 2}",
            }
            for index in range(8)
        ]

        def summary(completion):
            completion = np.asarray(completion, dtype=np.float32)
            _, result = grouped_results(
                rows,
                completion == 1.0,
                np.ones(8, dtype=bool),
                np.ones(8, dtype=np.int32),
                completion_metrics={"terminal_absolute": completion},
            )
            return result

        reference = summary([0.5] * 8)
        candidate = summary([1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0])
        gate = comparison_gate(reference, candidate)
        self.assertTrue(gate["progress_passed"])
        self.assertFalse(gate["guards_passed"])
        self.assertFalse(gate["passed"])

    def test_comparison_gate_requires_reference_and_candidate_integrity(self):
        rows = [
            {
                "slot_index": index + 1,
                "map_id": f"map-{index}",
                "family": "foundation",
                "primary_cell": "condition",
            }
            for index in range(4)
        ]

        def summary(completion):
            completion = np.asarray(completion, dtype=np.float32)
            _, result = grouped_results(
                rows,
                completion == 1.0,
                np.ones(4, dtype=bool),
                np.ones(4, dtype=np.int32),
                completion_metrics={"terminal_absolute": completion},
            )
            return result

        reference = summary([0.5] * 4)
        candidate = summary([0.6] * 4)
        reference["integrity"]["passed"] = False
        gate = comparison_gate(reference, candidate)
        self.assertFalse(gate["reference_integrity_passed"])
        self.assertTrue(gate["candidate_integrity_passed"])
        self.assertFalse(gate["integrity_passed"])
        self.assertFalse(gate["passed"])

    def test_treatment_fingerprint_rejects_run_or_protocol_drift(self):
        bank = SimpleNamespace(
            arm="G-UNIFORM",
            terra_revision="terra-rev",
            environment_protocol_sha256="a" * 64,
            source_registry_sha256="b" * 64,
        )
        config = SimpleNamespace(
            name="run",
            seed=7,
            config_name="G-UNIFORM",
            accepted_bank=bank,
            num_devices=4,
            num_envs_per_device=1024,
            num_steps=32,
            update_epochs=2,
            num_minibatches=32,
            lr=3e-4,
            relocation_progress_mult=1.5,
            agent_types_override=(0,),
            action_types_override=(0,),
            curriculum_levels_override=[
                {
                    "maps_path": "train/a",
                    "max_steps_in_episode": 450,
                    "rewards_type": 0,
                    "apply_trench_rewards": False,
                }
            ],
        )
        baseline = checkpoint_treatment_fingerprint(
            {"train_config": config}
        )
        same = checkpoint_treatment_fingerprint(
            {"train_config": config}
        )
        self.assertEqual(baseline, same)

        changed_run = SimpleNamespace(**vars(config))
        changed_run.seed = 8
        self.assertNotEqual(
            baseline["sha256"],
            checkpoint_treatment_fingerprint(
                {"train_config": changed_run}
            )["sha256"],
        )
        changed_protocol = SimpleNamespace(**vars(bank))
        changed_protocol.environment_protocol_sha256 = "c" * 64
        changed_bank_config = SimpleNamespace(**vars(config))
        changed_bank_config.accepted_bank = changed_protocol
        self.assertNotEqual(
            baseline["sha256"],
            checkpoint_treatment_fingerprint(
                {"train_config": changed_bank_config}
            )["sha256"],
        )

        changed_depth = SimpleNamespace(**vars(config))
        changed_depth.resnet_stage_channels = (24, 48, 64, 96)
        changed_depth.resnet_blocks_per_stage = (2, 2, 3, 3)
        self.assertNotEqual(
            baseline["sha256"],
            checkpoint_treatment_fingerprint(
                {"train_config": changed_depth}
            )["sha256"],
        )

    def test_checkpoint_sequence_rejects_duplicate_updates_and_mixed_runs(self):
        config = SimpleNamespace(name="run-a", seed=1)
        checkpoints = [
            (Path("a.pkl"), {"next_update": 500, "train_config": config}),
            (Path("b.pkl"), {"next_update": 500, "train_config": config}),
        ]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_checkpoint_sequence(checkpoints)

        changed = SimpleNamespace(name="run-b", seed=1)
        checkpoints[1] = (
            Path("b.pkl"),
            {"next_update": 1000, "train_config": changed},
        )
        with self.assertRaisesRegex(ValueError, "mixes treatment"):
            validate_checkpoint_sequence(checkpoints)

    def test_checkpoint_sequence_accepts_explicit_update_zero_only(self):
        config = SimpleNamespace(name="initialization", seed=1)
        validate_checkpoint_sequence(
            [(Path("initial.pkl"), {"next_update": 0, "train_config": config})]
        )
        with self.assertRaisesRegex(ValueError, "must declare next_update"):
            validate_checkpoint_sequence(
                [(Path("missing.pkl"), {"train_config": config})]
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_checkpoint_sequence(
                [
                    (
                        Path("negative.pkl"),
                        {"next_update": -1, "train_config": config},
                    )
                ]
            )

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
