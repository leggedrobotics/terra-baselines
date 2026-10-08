"""Pull alignment must survive checkpoint/eval reconstruction and explicit overrides."""

import collections
import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from configs.training_configs import get_config
from eval_fixed_bank import checkpoint_treatment_fingerprint, configure_for_bank
from terra.config import EnvConfig
from train_mixed import (
    MixedAgentTrainConfig, PULL_DIRECTION_RULE_FIELDS, apply_pull_direction_rules,
    make_mixed_agent_states, pull_direction_rules_effective, restore_pull_direction_rules,
    _preflight_trench_alignment_metadata,
    _validate_checkpoint_architecture,
)
from utils.helpers import (
    PULL_DIRECTION_RULE_DEFAULTS, checkpoint_evaluation_config,
    checkpoint_pull_direction_rules,
)


REPO = Path(__file__).resolve().parents[1]
TREATMENT = {
    **PULL_DIRECTION_RULE_DEFAULTS,
    "pull_direction_alignment": True,
    "edge_band_width_m": 0.8,
    "edge_pull_tolerance_rad": 0.4,
    "trench_pull_tolerance_rad": 0.2,
    "dig_pull_min_length_m": 2.0,
}


def checkpoint(*, train_fields=True):
    return {
        "train_config": SimpleNamespace(name="pull", seed=1, **(TREATMENT if train_fields else {})),
        "env_config": EnvConfig()._replace(**TREATMENT),
    }


class PullDirectionPlumbingTest(unittest.TestCase):
    def test_omitted_fields_leave_default_and_checkpoint_env_unchanged(self):
        for env in (EnvConfig(), checkpoint()["env_config"]):
            for config in (SimpleNamespace(), SimpleNamespace(**dict.fromkeys(PULL_DIRECTION_RULE_FIELDS))):
                self.assertIs(apply_pull_direction_rules(env, config), env)
        self.assertEqual(pull_direction_rules_effective(EnvConfig()), PULL_DIRECTION_RULE_DEFAULTS)

    def test_explicit_overrides_preserve_other_environment_settings(self):
        env = checkpoint()["env_config"]._replace(enforce_trench_dig_alignment=True)
        config = SimpleNamespace(pull_direction_alignment=False, edge_band_width_m=1.0)
        actual = apply_pull_direction_rules(env, config)
        self.assertEqual(pull_direction_rules_effective(actual), {
            **TREATMENT, "pull_direction_alignment": False, "edge_band_width_m": 1.0,
        })
        self.assertTrue(actual.enforce_trench_dig_alignment)
        self.assertEqual(actual.agent, env.agent)
        self.assertEqual(actual.rewards, env.rewards)

    def test_resume_restores_omitted_fields_and_honors_explicit_disable(self):
        for enabled in (None, False, True):
            config = SimpleNamespace(pull_direction_alignment=enabled, edge_band_width_m=1.0)
            restore_pull_direction_rules(config, checkpoint(train_fields=False))
            self.assertEqual(config.pull_direction_alignment, True if enabled is None else enabled)
            self.assertEqual(config.edge_band_width_m, 1.0)
            self.assertEqual(config.trench_pull_tolerance_rad, 0.2)

    def test_saved_env_only_and_train_only_treatments_restore_for_evaluation(self):
        for saved in (checkpoint(train_fields=False), {"train_config": checkpoint()["train_config"]}):
            restored = checkpoint_evaluation_config(saved)
            evaluated = configure_for_bank(restored, "evaluation/all", 2)
            self.assertEqual({name: getattr(evaluated, name) for name in TREATMENT}, TREATMENT)
        self.assertEqual(
            checkpoint_treatment_fingerprint(checkpoint(train_fields=False)),
            checkpoint_treatment_fingerprint({"train_config": checkpoint()["train_config"]}),
        )

    def test_mixed_evaluation_requires_explicit_mode_and_preserves_precision_architecture(self):
        saved = MixedAgentTrainConfig(
            name="mixed", num_devices=1, actor_core="gru", teacher_checkpoint="frozen.pkl",
            num_steps=32, num_minibatches=32,
            recurrent_teacher=True, kickstart_value_coef=0, pull_direction_alignment=True,
            precision_episode_fraction=.5, precision_required_band_observation=True,
            pull_direction_training_slots="/training-only/slots.json",
            precision_slots=[1, 7], teacher_slots=[9],
        )
        with self.assertRaisesRegex(ValueError, "--precision-mode"):
            configure_for_bank(saved, "evaluation/all", 2)
        for mode in ("bulk", "precision"):
            actual = configure_for_bank(saved, "evaluation/all", 2, precision_mode=mode)
            self.assertEqual(actual.enforce_foundation_border_alignment, mode == "precision")
            self.assertTrue(actual.precision_required_band_observation)
            self.assertTrue(actual.pull_direction_alignment)
            self.assertFalse(actual.recurrent_teacher)
            self.assertEqual(actual.precision_episode_fraction, 0)
            for field in ("teacher_checkpoint", "pull_direction_training_slots", "precision_slots", "teacher_slots"):
                self.assertIsNone(getattr(actual, field))
            _validate_checkpoint_architecture({"train_config": saved}, actual)
        contract = checkpoint_treatment_fingerprint({"train_config": saved})["contract"]
        self.assertTrue(contract["architecture"]["precision_required_band_observation"])
        self.assertEqual(saved.precision_episode_fraction, .5)
        self.assertEqual(saved.precision_slots, [1, 7])

    def test_old_dataclass_class_defaults_do_not_hide_saved_environment(self):
        config = MixedAgentTrainConfig(name="old", num_devices=1)
        for name in PULL_DIRECTION_RULE_FIELDS:
            del vars(config)[name]
        self.assertIsNone(config.pull_direction_alignment)
        self.assertEqual(checkpoint_pull_direction_rules({
            "train_config": config, "env_config": checkpoint()["env_config"],
        }), TREATMENT)

    def test_old_positional_env_pickle_preserves_all_existing_field_positions(self):
        legacy_fields = tuple(name for name in EnvConfig._fields if name not in PULL_DIRECTION_RULE_FIELDS)
        self.assertEqual(EnvConfig._fields[:len(legacy_fields)], legacy_fields)
        original = EnvConfig()._replace(
            foundation_border_width_tiles=7, terminal_reward_mix=0.42,
            enforce_trench_dig_alignment=True, retained_work_turn_cost=0.13,
        )

        class LegacySavedEnv:
            def __reduce__(self):
                # NamedTuple pickles construct their class from positional values.
                return EnvConfig, tuple(getattr(original, name) for name in legacy_fields)

        restored = pickle.loads(pickle.dumps(LegacySavedEnv()))
        for name in legacy_fields:
            self.assertEqual(getattr(restored, name), getattr(original, name), name)
        self.assertEqual(pull_direction_rules_effective(restored), PULL_DIRECTION_RULE_DEFAULTS)

    def test_saved_disagreements_and_nonuniform_settings_are_rejected(self):
        for name, default in PULL_DIRECTION_RULE_DEFAULTS.items():
            with self.subTest(name=name):
                saved = checkpoint()
                setattr(saved["train_config"], name, default)
                with self.assertRaisesRegex(ValueError, f"{name} mismatch"):
                    checkpoint_evaluation_config(saved)
        saved = checkpoint(train_fields=False)
        saved["env_config"] = saved["env_config"]._replace(edge_band_width_m=np.asarray([0.6, 1.0]))
        with self.assertRaisesRegex(ValueError, "must be uniform"):
            checkpoint_pull_direction_rules(saved)

    def test_runtime_without_the_feature_refuses_requested_mode(self):
        env = collections.namedtuple("OldEnv", ["seed"])(1)
        self.assertIs(apply_pull_direction_rules(env, SimpleNamespace()), env)
        with self.assertRaisesRegex(RuntimeError, "no EnvConfig.pull_direction_alignment"):
            apply_pull_direction_rules(env, SimpleNamespace(pull_direction_alignment=True))

    def test_invalid_geometry_settings_fail_before_environment_construction(self):
        for name, values in (
            ("edge_band_width_m", (0.0, -0.1, np.inf, np.nan)),
            ("dig_pull_min_length_m", (0.0, -0.1, np.inf, np.nan)),
            ("edge_pull_tolerance_rad", (-0.1, np.pi / 2, np.inf, np.nan)),
            ("trench_pull_tolerance_rad", (-0.1, np.pi / 2, np.inf, np.nan)),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    apply_pull_direction_rules(EnvConfig(), SimpleNamespace(**{name: value}))

    def test_constructor_mode_is_resolved_from_checkpoint_before_geometry_loading(self):
        for override, expected in ((None, True), (False, False), (True, True)):
            config = MixedAgentTrainConfig(name="constructor", num_devices=1, pull_direction_alignment=override)
            with patch("train_mixed.TerraEnvBatch", side_effect=RuntimeError("intercept constructor")) as constructor:
                with self.assertRaisesRegex(RuntimeError, "intercept constructor"):
                    make_mixed_agent_states(config, env_params_override=checkpoint()["env_config"])
            self.assertEqual(constructor.call_args.kwargs["pull_direction_alignment"], expected)
            self.assertEqual(config.pull_direction_alignment, expected)
            self.assertEqual(config.edge_band_width_m, 0.8)

    def test_cutting_space_mode_skips_saved_legacy_trench_metadata_preflight(self):
        env = EnvConfig()._replace(
            enforce_trench_dig_alignment=True, pull_direction_alignment=True,
        )
        with patch.dict(os.environ, {"DATASET_PATH": ""}):
            # This preflight is invoked when the saved training flag requires
            # trench metadata. Arbitrary raster geometry needs no such axes.
            _preflight_trench_alignment_metadata(None, env, [])
            with self.assertRaisesRegex(RuntimeError, "needs DATASET_PATH"):
                _preflight_trench_alignment_metadata(
                    None, env._replace(pull_direction_alignment=False), [],
                )

    def test_feature_off_keeps_historical_fingerprint_and_enabled_tolerances_bind(self):
        legacy = {"train_config": SimpleNamespace(name="pull", seed=1)}
        explicit = copy.deepcopy(legacy)
        explicit["env_config"] = EnvConfig()
        for name, value in PULL_DIRECTION_RULE_DEFAULTS.items():
            setattr(explicit["train_config"], name, value)
        self.assertEqual(checkpoint_treatment_fingerprint(legacy), checkpoint_treatment_fingerprint(explicit))
        self.assertNotIn("pull_direction_rules", checkpoint_treatment_fingerprint(legacy)["contract"])
        treatment = {"train_config": checkpoint()["train_config"]}
        original = checkpoint_treatment_fingerprint(treatment)
        self.assertNotEqual(original, checkpoint_treatment_fingerprint(legacy))
        treatment["train_config"].edge_pull_tolerance_rad = 0.3
        self.assertNotEqual(original, checkpoint_treatment_fingerprint(treatment))

    def test_opt_in_preset_only_changes_pull_rules(self):
        base = get_config("gru_generalist_512_machine_rules")
        selected = get_config("gru_generalist_512_pull_direction")
        for name in vars(base):
            if name not in {"name", "description", *PULL_DIRECTION_RULE_FIELDS}:
                self.assertEqual(getattr(selected, name), getattr(base, name), name)
        self.assertTrue(selected.pull_direction_alignment)
        self.assertEqual(selected.edge_band_width_m, 0.6)
        self.assertEqual(selected.dig_pull_min_length_m, 2.5)
        for preset in (base, get_config("gru_generalist_512")):
            for name in PULL_DIRECTION_RULE_FIELDS:
                self.assertIsNone(getattr(preset, name))

    def test_training_and_evaluation_command_lines_expose_all_flags(self):
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
        for script, separator in (
            ("train_mixed.py", "_"), ("eval_fixed_bank.py", "-"),
            ("scripts/euler_gru_generalist_512/eval_known_starts.py", "-"),
        ):
            output = subprocess.run(
                [sys.executable, str(REPO / script), "--help"], env=env, cwd=REPO,
                capture_output=True, text=True, check=True,
            ).stdout
            for name in PULL_DIRECTION_RULE_FIELDS:
                self.assertIn("--" + name.replace("_", separator), output, script)
            self.assertIn("--no-" + "pull_direction_alignment".replace("_", separator), output, script)


if __name__ == "__main__":
    unittest.main()
