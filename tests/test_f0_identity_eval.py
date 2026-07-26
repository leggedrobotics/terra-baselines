import unittest
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from configs.training_configs import get_config
from eval_f0_identity import (
    HORIZON,
    PRODUCTION_TIMESTEPS,
    PRODUCTION_UPDATES,
    RESET_SEEDS,
    configure_for_identity,
    consecutive_mastery,
    declared_reset_keys,
    summarize_rollout,
    treatment_spec,
    verify_production_checkpoint,
)


def _rollout_fixture(success_count=29, mass_failure=False):
    count = len(RESET_SEEDS)
    successes = np.arange(count) < success_count
    lengths = np.where(successes, 100, HORIZON).astype(np.int32)
    completion = np.where(successes, 1.0, 0.5).astype(np.float32)
    mass_residual = np.zeros(count, dtype=np.int32)
    if mass_failure:
        mass_residual[0] = 1
    stats = {
        "episode_done_once": successes,
        "episode_terminated_once": np.ones(count, dtype=bool),
        "episode_length": lengths,
        "terminal_completion": {
            "absolute": completion,
            "dig": completion,
            "dump_purity": completion,
            "dump_volume": completion,
            "unloaded": np.ones(count, dtype=np.float32),
            "dump_mask_integrity": np.ones(count, dtype=np.float32),
            "accepted_dump_volume": completion,
            "illegal_dump_volume": np.zeros(count, dtype=np.float32),
        },
        "integrity": {
            "supported": True,
            "slot_index_zero_based": np.zeros(count, dtype=np.int32),
            "maximum_mass_residual": mass_residual,
            "no_effect_action_count": np.zeros(count, dtype=np.int32),
            "target_mutation": np.zeros(count, dtype=bool),
            "obstacle_mutation": np.zeros(count, dtype=bool),
            "nonfinite_state": np.zeros(count, dtype=bool),
        },
        "action_sequence": np.full(
            (HORIZON, count),
            7,
            dtype=np.int32,
        ),
        "action_had_effect_sequence": np.zeros(
            (HORIZON, count),
            dtype=bool,
        ),
    }
    cumulative_rewards = np.zeros((HORIZON, count), dtype=np.float32)
    return stats, cumulative_rewards


def _production_checkpoint(
    identity="foundation",
    integrity_failure=False,
    treatment="corrected_dense_v1",
):
    reward_spec = treatment_spec(identity, treatment)
    config = SimpleNamespace(
        config_name=reward_spec["config_name"],
        seed=2026072601 if identity == "foundation" else 2026072602,
        num_devices=4,
        num_envs_per_device=1024,
        num_steps=32,
        num_updates=PRODUCTION_UPDATES,
        actual_total_timesteps=PRODUCTION_TIMESTEPS,
        lr=3e-4,
        update_epochs=2,
        num_minibatches=32,
        ent_schedule_start=0.15,
        ent_schedule_end=0.005,
        ent_schedule_steps=950,
        model_size="base",
        model_core="mlp",
        map_encoder="resnet_spatial_8x8",
        encoder_compute_dtype="float32",
        use_value_clip=False,
        flat_minibatch_shuffle=True,
        fail_on_nonfinite=True,
        finite_check_interval=1,
        resume_from=None,
        warm_start_from=None,
        teacher_checkpoint=None,
        curriculum_levels_override=[
            {
                "maps_path": identity,
                "max_steps_in_episode": HORIZON,
                "rewards_type": 0,
                "apply_trench_rewards": reward_spec["apply_trench_rewards"],
            }
        ],
    )
    return {
        "next_update": 100,
        "train_config": config,
        "model": {"weight": np.ones((2,), dtype=np.float32)},
        "optimizer_state": {"moment": np.zeros((2,), dtype=np.float32)},
        "transition_integrity": {
            "maximum_mass_residual": int(integrity_failure),
            "target_mutation_count": 0,
            "obstacle_mutation_count": 0,
        },
    }


class F0IdentityEvalTest(unittest.TestCase):
    def test_identity_presets_freeze_family_specific_dense_config(self):
        foundation = get_config("f0_foundation_identity_v1")
        trench = get_config("f0_trench_identity_v1")
        trench_repair = get_config("f0_trench_identity_shaping_off_v1")

        self.assertEqual(foundation.agent_types, (0,))
        self.assertEqual(foundation.maps[0].maps_path, "foundation")
        self.assertEqual(foundation.maps[0].max_steps_in_episode, HORIZON)
        self.assertFalse(foundation.maps[0].apply_trench_rewards)

        self.assertEqual(trench.agent_types, (0,))
        self.assertEqual(trench.maps[0].maps_path, "trench")
        self.assertEqual(trench.maps[0].max_steps_in_episode, HORIZON)
        self.assertTrue(trench.maps[0].apply_trench_rewards)

        self.assertEqual(trench_repair.agent_types, (0,))
        self.assertEqual(trench_repair.maps[0].maps_path, "trench")
        self.assertEqual(trench_repair.maps[0].max_steps_in_episode, HORIZON)
        self.assertFalse(trench_repair.maps[0].apply_trench_rewards)

        trench_fields = asdict(trench)
        repair_fields = asdict(trench_repair)
        for fields in (trench_fields, repair_fields):
            fields.pop("name")
            fields.pop("description")
        trench_fields["maps"][0]["apply_trench_rewards"] = False
        self.assertEqual(trench_fields, repair_fields)

    def test_declared_reset_seeds_are_frozen_and_distinct(self):
        self.assertEqual(len(RESET_SEEDS), 32)
        self.assertEqual(len(set(RESET_SEEDS)), 32)
        self.assertEqual(np.asarray(declared_reset_keys()).shape, (32, 2))

    def test_29_of_32_with_integrity_and_trajectory_passes(self):
        stats, rewards = _rollout_fixture()
        summary = summarize_rollout(stats, rewards)
        self.assertEqual(summary["successes"], 29)
        self.assertTrue(summary["performance_passed"])
        self.assertTrue(summary["integrity_passed"])
        self.assertTrue(summary["passed"])
        self.assertTrue(
            summary["successful_action_trajectory"]["all_actions_in_discrete_range"]
        )

    def test_integrity_failure_blocks_per_checkpoint_gate(self):
        stats, rewards = _rollout_fixture(mass_failure=True)
        summary = summarize_rollout(stats, rewards)
        self.assertTrue(summary["performance_passed"])
        self.assertFalse(summary["integrity_passed"])
        self.assertFalse(summary["passed"])

    def test_two_adjacent_passing_checkpoints_are_required(self):
        passing_summary = summarize_rollout(*_rollout_fixture())
        records = [
            {"checkpoint_update": 100, "summary": passing_summary},
            {"checkpoint_update": 200, "summary": passing_summary},
        ]
        gate = consecutive_mastery(records)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passing_update_pairs"], [[100, 200]])

        records[1]["checkpoint_update"] = 300
        self.assertFalse(consecutive_mastery(records)["passed"])

    def test_production_checkpoint_gate_freezes_config_and_integrity(self):
        gate = verify_production_checkpoint(
            _production_checkpoint(),
            "foundation",
            100,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["model_leaf_count"], 1)
        self.assertEqual(gate["optimizer_leaf_count"], 1)

        with self.assertRaisesRegex(RuntimeError, "transition integrity"):
            verify_production_checkpoint(
                _production_checkpoint(integrity_failure=True),
                "foundation",
                100,
            )

    def test_trench_repair_changes_only_registered_reward_treatment(self):
        treatment = "corrected_dense_v1_trench_absolute_off"
        checkpoint = _production_checkpoint(
            identity="trench",
            treatment=treatment,
        )
        gate = verify_production_checkpoint(
            checkpoint,
            "trench",
            100,
            treatment,
        )
        self.assertTrue(gate["passed"])
        config = configure_for_identity(
            checkpoint["train_config"],
            "trench",
            treatment,
        )
        self.assertFalse(config.curriculum_levels_override[0]["apply_trench_rewards"])
        with self.assertRaisesRegex(ValueError, "unsupported F0"):
            treatment_spec("foundation", treatment)


if __name__ == "__main__":
    unittest.main()
