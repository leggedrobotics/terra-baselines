import unittest
from types import SimpleNamespace

import numpy as np

from configs.training_configs import get_config
from eval_b0_panel import (
    INITIAL_UPDATES,
    PANEL_SPECS,
    TRANSITIONS_PER_UPDATE,
    configure_for_panel,
    consecutive_cell_witnesses,
    decision,
    slight_improvement,
    verify_b0_checkpoint,
)


def _checkpoint(panel="foundation_geometry", integrity_failure=False):
    spec = PANEL_SPECS[panel]
    config = SimpleNamespace(
        config_name=spec["config_name"],
        seed=spec["seed"],
        num_devices=4,
        num_envs_per_device=1024,
        num_envs=4096,
        num_steps=32,
        env_steps_per_update=TRANSITIONS_PER_UPDATE,
        num_updates=INITIAL_UPDATES,
        total_timesteps=INITIAL_UPDATES * TRANSITIONS_PER_UPDATE,
        actual_total_timesteps=(INITIAL_UPDATES * TRANSITIONS_PER_UPDATE),
        lr=3e-4,
        clip_eps=0.2,
        gamma=0.9984,
        gae_lambda=0.95,
        update_epochs=2,
        num_minibatches=32,
        vf_coef=2.0,
        max_grad_norm=0.5,
        log_train_interval=1,
        log_eval_interval=0,
        eval_episodes=32,
        checkpoint_interval=100,
        keep_checkpoint_history=True,
        cache_clear_interval=1000,
        ent_schedule_start=0.15,
        ent_schedule_end=0.005,
        ent_schedule_steps=950,
        agent_types_override=(0,),
        action_types_override=(0,),
        dump_bonus_mult=0.5,
        excavator_relocate_dumped_mult=1.5,
        excavator_relocate_dug_dirt_mult=1.5,
        transport_relocate_mult=1.5,
        curriculum_increase_level_threshold=20,
        curriculum_decrease_level_threshold=80,
        curriculum_last_level_type="none",
        single_map_path=None,
        replay_map_count=0,
        target_map_repeat=0,
        model_size="base",
        model_core="mlp",
        map_encoder="resnet_spatial_8x8",
        encoder_compute_dtype="float32",
        attention_compute_dtype="encoder",
        token_mixer_residual_init_scale=0.0,
        critic_hidden_dims=None,
        use_value_clip=False,
        flat_minibatch_shuffle=True,
        fail_on_nonfinite=True,
        finite_check_interval=1,
        resume_from=None,
        warm_start_from=None,
        load_env_from_checkpoint=False,
        teacher_checkpoint=None,
        teacher_obs_downsample=1,
        curriculum_levels_override=[
            {
                "maps_path": f"panels/train/{panel}",
                "max_steps_in_episode": 450,
                "rewards_type": 0,
                "apply_trench_rewards": False,
            }
        ],
    )
    return {
        "next_update": 100,
        "train_config": config,
        "model": {"weight": np.ones(2, dtype=np.float32)},
        "optimizer_state": {"moment": np.zeros(2, dtype=np.float32)},
        "transition_integrity": {
            "maximum_mass_residual": int(integrity_failure),
            "target_mutation_count": 0,
            "obstacle_mutation_count": 0,
        },
    }


def _record(update, cell_values):
    return {
        "checkpoint_update": update,
        "summary": {
            "cells": {
                cell: {
                    "successes": successes,
                    "median_terminal_absolute_completion": completion,
                    "integrity_failure_count": 0,
                    "trajectory_saved": successes > 0,
                    "passed": successes >= 6,
                }
                for cell, (successes, completion) in cell_values.items()
            }
        },
    }


class B0PanelEvalTest(unittest.TestCase):
    def test_presets_match_declared_panels_and_disable_absolute_trench_term(
        self,
    ):
        for panel, spec in PANEL_SPECS.items():
            config = get_config(spec["config_name"])
            self.assertEqual(config.agent_types, (0,))
            self.assertEqual(
                config.maps[0].maps_path,
                f"panels/train/{panel}",
            )
            self.assertEqual(config.maps[0].max_steps_in_episode, 450)
            self.assertFalse(config.maps[0].apply_trench_rewards)

    def test_evaluator_uses_largest_valid_inherited_minibatch_divisor(
        self,
    ):
        train_config = _checkpoint("trench_topology")["train_config"]
        configured = configure_for_panel(
            train_config,
            "trench_topology",
            48,
        )
        self.assertEqual(configured.num_minibatches, 16)
        self.assertEqual(configured.num_envs_per_device, 48)

    def test_checkpoint_gate_freezes_initial_panel_treatment(self):
        gate = verify_b0_checkpoint(
            _checkpoint(),
            "foundation_geometry",
            100,
        )
        self.assertTrue(gate["passed"])
        with self.assertRaisesRegex(RuntimeError, "transition integrity"):
            verify_b0_checkpoint(
                _checkpoint(integrity_failure=True),
                "foundation_geometry",
                100,
            )
        changed_reward = _checkpoint()
        changed_reward["train_config"].dump_bonus_mult = 0.6
        with self.assertRaisesRegex(RuntimeError, "config mismatch"):
            verify_b0_checkpoint(
                changed_reward,
                "foundation_geometry",
                100,
            )

    def test_each_cell_requires_two_adjacent_passing_checkpoints(self):
        records = [
            _record(100, {"a": (6, 1.0), "b": (5, 0.9)}),
            _record(200, {"a": (7, 1.0), "b": (6, 1.0)}),
            _record(300, {"a": (7, 1.0), "b": (6, 1.0)}),
        ]
        gate = consecutive_cell_witnesses(records)
        self.assertTrue(gate["passed"])
        self.assertEqual(
            gate["cells"]["a"]["passing_update_pairs"], [[100, 200], [200, 300]]
        )
        self.assertEqual(gate["cells"]["b"]["passing_update_pairs"], [[200, 300]])

    def test_one_success_or_one_percent_completion_authorizes_continuation(
        self,
    ):
        records = [
            _record(100, {"a": (0, 0.20), "b": (0, 0.30)}),
            _record(200, {"a": (1, 0.20), "b": (0, 0.30)}),
            _record(300, {"a": (1, 0.205), "b": (0, 0.305)}),
            _record(400, {"a": (2, 0.21), "b": (0, 0.305)}),
            _record(500, {"a": (2, 0.21), "b": (0, 0.315)}),
        ]
        progress = slight_improvement(records, ["a", "b"])
        self.assertTrue(progress["passed"])
        self.assertTrue(progress["cells"]["a"]["passed"])
        self.assertTrue(progress["cells"]["b"]["passed"])
        self.assertEqual(
            progress["cells"]["a"]["last_improvement_update"],
            400,
        )
        self.assertEqual(
            progress["cells"]["b"]["last_improvement_update"],
            500,
        )
        self.assertEqual(
            decision(records)["decision"],
            "continue_same_panel",
        )

    def test_five_flat_evaluations_stop_after_older_improvement(self):
        records = [
            _record(100, {"a": (0, 0.20)}),
            _record(200, {"a": (1, 0.20)}),
            _record(300, {"a": (1, 0.20)}),
            _record(400, {"a": (1, 0.20)}),
            _record(500, {"a": (1, 0.20)}),
            _record(600, {"a": (1, 0.20)}),
            _record(700, {"a": (1, 0.20)}),
        ]
        progress = slight_improvement(records, ["a"])
        self.assertFalse(progress["passed"])
        self.assertEqual(
            progress["window_updates"],
            [300, 400, 500, 600, 700],
        )
        self.assertEqual(
            decision(records)["decision"],
            "stop_and_diagnose_panel",
        )

    def test_flat_mixed_panel_requests_only_failed_cell_isolates(self):
        records = [
            _record(100, {"a": (6, 1.0), "b": (0, 0.30)}),
            _record(200, {"a": (6, 1.0), "b": (0, 0.30)}),
            _record(300, {"a": (6, 1.0), "b": (0, 0.30)}),
            _record(400, {"a": (6, 1.0), "b": (0, 0.30)}),
            _record(500, {"a": (6, 1.0), "b": (0, 0.30)}),
        ]
        result = decision(records)
        self.assertEqual(result["decision"], "conditional_cell_isolates")
        self.assertEqual(result["conditional_cell_isolates"], ["b"])


if __name__ == "__main__":
    unittest.main()
