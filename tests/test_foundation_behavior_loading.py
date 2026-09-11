"""Checkpoint behavior metadata must agree with the executable observation path."""

import ast
import copy
from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import eval_mcts
from eval_fixed_bank import checkpoint_treatment_fingerprint, configure_for_bank
from terra.config import BatchConfig, EnvConfig, MapsDimsConfig
from utils.helpers import (
    FOUNDATION_BEHAVIOR_DEFAULTS,
    checkpoint_evaluation_config,
    checkpoint_foundation_behavior,
    overlay_foundation_behavior,
    replicate_checkpoint_env_config,
    validate_foundation_behavior_env,
)
from utils.models import get_model_ready, validate_model_params_match
from utils.utils_ppo import obs_to_model_input


@dataclass
class SavedConfig:
    admissible_dig_observation: bool = True
    executable_dig_observation: bool = False
    lateral_dig_cost: float = 0.0
    base_travel_cost: float = 0.0
    base_turn_cost: float = 0.0


def treated_checkpoint():
    config = SavedConfig(True, True, 0.125, 0.25, 0.5)
    return {
        "train_config": config,
        "env_config": EnvConfig()._replace(**{
            name: getattr(config, name) for name in FOUNDATION_BEHAVIOR_DEFAULTS
        }),
        "model": {},
    }


class ModelConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def model_config(executable):
    return ModelConfig(
        clip_action_maps=True,
        local_map_area_scale=2.0,
        loaded_max=100,
        local_map_normalization_bounds=(-16, 16),
        model_core="mlp",
        model_size="base",
        map_encoder="atari",
        num_prev_actions=5,
        admissible_dig_observation=True,
        executable_dig_observation=executable,
    )


def observation():
    obs = {
        "agent_states": jnp.zeros((1, 4, 9)),
        "agent_active": jnp.ones((1, 4), dtype=jnp.int8),
        "num_agents": jnp.ones((1,), dtype=jnp.int32),
        "agent_width": jnp.ones((1,), dtype=jnp.int32),
        "agent_height": jnp.ones((1,), dtype=jnp.int32),
    }
    for name in (
        "local_map_action_neg", "local_map_action_pos", "local_map_target_neg",
        "local_map_target_pos", "local_map_dumpability", "local_map_obstacles",
        "local_map_border_workspace", "local_map_edge_alignment_error",
        "local_map_border_diggable", "local_map_admissible_dig",
    ):
        obs[name] = jnp.ones((1, 12), dtype=jnp.int16) * 4
    for name in (
        "traversability_mask", "reachability_mask", "action_map", "target_map",
        "padding_mask", "dumpability_mask", "interaction_mask",
    ):
        obs[name] = jnp.zeros((1, 64, 64))
    return obs


class FoundationBehaviorLoadingTest(unittest.TestCase):
    def test_legacy_checkpoint_keeps_defaults_and_fingerprint(self):
        legacy = {"train_config": SimpleNamespace(name="old", seed=1)}
        explicit = copy.deepcopy(legacy)
        for name, value in FOUNDATION_BEHAVIOR_DEFAULTS.items():
            setattr(explicit["train_config"], name, value)
        explicit["env_config"] = EnvConfig()
        self.assertEqual(checkpoint_foundation_behavior(legacy), FOUNDATION_BEHAVIOR_DEFAULTS)
        self.assertEqual(
            checkpoint_treatment_fingerprint(legacy),
            checkpoint_treatment_fingerprint(explicit),
        )
        self.assertNotIn("foundation_behavior", checkpoint_treatment_fingerprint(legacy)["contract"])

    def test_env_recovers_fields_absent_from_old_dataclass_pickle(self):
        checkpoint = treated_checkpoint()
        config = checkpoint["train_config"]
        for name in FOUNDATION_BEHAVIOR_DEFAULTS:
            del config.__dict__[name]
        self.assertFalse(config.executable_dig_observation)  # inherited new default
        restored = checkpoint_evaluation_config(checkpoint)
        self.assertTrue(restored.executable_dig_observation)
        self.assertEqual(restored.base_travel_cost, 0.25)
        self.assertNotIn("executable_dig_observation", vars(config))
        bank_config = configure_for_bank(restored, "eval/bank", 2)
        self.assertTrue(bank_config.executable_dig_observation)
        self.assertEqual(bank_config.base_turn_cost, 0.5)

    def test_explicit_checkpoint_conflicts_are_rejected(self):
        for name in FOUNDATION_BEHAVIOR_DEFAULTS:
            with self.subTest(name=name):
                checkpoint = treated_checkpoint()
                setattr(checkpoint["train_config"], name, FOUNDATION_BEHAVIOR_DEFAULTS[name])
                with self.assertRaisesRegex(ValueError, f"{name} mismatch"):
                    checkpoint_evaluation_config(checkpoint)

    def test_float32_roundoff_is_accepted_but_zero_is_not_missing(self):
        checkpoint = treated_checkpoint()
        checkpoint["train_config"].base_travel_cost = 0.1
        checkpoint["env_config"] = checkpoint["env_config"]._replace(
            base_travel_cost=jnp.float32(0.1)
        )
        self.assertAlmostEqual(checkpoint_foundation_behavior(checkpoint)["base_travel_cost"], 0.1)
        checkpoint["train_config"].base_travel_cost = 0.0
        with self.assertRaisesRegex(ValueError, "base_travel_cost mismatch"):
            checkpoint_foundation_behavior(checkpoint)

    def test_fingerprint_does_not_depend_on_float_storage_precision(self):
        config = SavedConfig(base_travel_cost=0.1)
        train_only = {"train_config": config}
        env_only = {
            "train_config": copy.deepcopy(config),
            "env_config": EnvConfig(base_travel_cost=jnp.float32(0.1)),
        }
        del env_only["train_config"].__dict__["base_travel_cost"]
        self.assertEqual(
            checkpoint_treatment_fingerprint(train_only),
            checkpoint_treatment_fingerprint(env_only),
        )

    def test_nonuniform_and_invalid_values_are_rejected(self):
        for name, value in (
            ("executable_dig_observation", [True, False]),
            ("executable_dig_observation", 2),
            ("base_travel_cost", [0.25, 0.5]),
            ("lateral_dig_cost", float("nan")),
            ("base_turn_cost", float("inf")),
            ("base_turn_cost", -1.0),
        ):
            with self.subTest(name=name, value=value):
                checkpoint = treated_checkpoint()
                checkpoint["env_config"] = checkpoint["env_config"]._replace(**{name: value})
                with self.assertRaisesRegex(ValueError, name):
                    checkpoint_foundation_behavior(checkpoint)

    def test_overlay_preserves_batch_axes_agent_vectors_and_other_fields(self):
        checkpoint = treated_checkpoint()
        original = replicate_checkpoint_env_config(EnvConfig(tile_size=0.75), 3)
        overlaid = overlay_foundation_behavior(original, checkpoint_foundation_behavior(checkpoint))
        validate_foundation_behavior_env(checkpoint["train_config"], overlaid)
        self.assertEqual(overlaid.executable_dig_observation.shape, (3,))
        self.assertEqual(overlaid.executable_dig_observation.dtype, jnp.bool_)
        np.testing.assert_array_equal(overlaid.base_travel_cost, [0.25] * 3)
        np.testing.assert_array_equal(overlaid.agent_types, original.agent_types)
        np.testing.assert_array_equal(overlaid.tile_size, original.tile_size)
        np.testing.assert_array_equal(original.base_travel_cost, [0.0] * 3)

    def test_old_terra_env_cannot_accept_an_active_treatment(self):
        legacy_env = SimpleNamespace()
        self.assertIs(overlay_foundation_behavior(legacy_env, FOUNDATION_BEHAVIOR_DEFAULTS), legacy_env)
        with self.assertRaisesRegex(ValueError, "no EnvConfig.executable_dig_observation"):
            overlay_foundation_behavior(legacy_env, {"executable_dig_observation": True})

    def test_fingerprint_binds_each_behavior_setting_without_new_architecture(self):
        baseline = checkpoint_treatment_fingerprint({"train_config": SavedConfig()})
        for name in FOUNDATION_BEHAVIOR_DEFAULTS:
            with self.subTest(name=name):
                config = SavedConfig()
                setattr(config, name, True if name == "executable_dig_observation" else 0.25)
                treatment = checkpoint_treatment_fingerprint({"train_config": config})
                self.assertNotEqual(baseline["sha256"], treatment["sha256"])
                self.assertEqual(baseline["contract"]["architecture"], treatment["contract"]["architecture"])
                self.assertEqual(treatment["contract"]["foundation_behavior"][name], getattr(config, name))

    def test_model_and_obs_require_admissible_input(self):
        config = model_config(True)
        config["admissible_dig_observation"] = False
        with self.assertRaisesRegex(ValueError, "requires admissible_dig_observation"):
            obs_to_model_input({}, None, config)
        with self.assertRaisesRegex(ValueError, "requires admissible_dig_observation"):
            get_model_ready(None, config, None)

    def test_static_selector_and_saved_env_must_agree_before_reset(self):
        checkpoint = treated_checkpoint()
        config = checkpoint["train_config"]
        with self.assertRaisesRegex(ValueError, "static observation selector"):
            get_model_ready(None, config, SimpleNamespace())
        env = SimpleNamespace(executable_dig_observation=True)
        with self.assertRaisesRegex(ValueError, "executable_dig_observation mismatch"):
            eval_mcts.rollout_episode(env, None, None, EnvConfig(), config, 1, True, 0)
        with self.assertRaisesRegex(ValueError, "static observation selector"):
            eval_mcts.rollout_episode(SimpleNamespace(), None, None, checkpoint["env_config"], config, 1, True, 0)

    def test_executable_input_keeps_width_order_and_area_scale(self):
        obs = observation()
        previous = jnp.zeros((1, 5), dtype=jnp.int32)
        legacy = obs_to_model_input(obs, previous, model_config(False))
        executable = obs_to_model_input(obs, previous, model_config(True))
        self.assertEqual(len(executable), 23)
        self.assertEqual(executable[-1].shape, (1, 12))
        np.testing.assert_array_equal(executable[-1], [[2.0] * 12])
        for left, right in zip(legacy, executable):
            np.testing.assert_array_equal(left, right)
        obs["local_map_admissible_dig"] = jnp.zeros((1, 11))
        with self.assertRaisesRegex(ValueError, "width 12"):
            obs_to_model_input(obs, previous, model_config(True))

    def test_semantics_change_preserves_parameter_tree_and_initial_values(self):
        env = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)),
            executable_dig_observation=False,
        )
        _, old = get_model_ready(jax.random.PRNGKey(0), model_config(False), env)
        env.executable_dig_observation = True
        _, new = get_model_ready(jax.random.PRNGKey(0), model_config(True), env)
        validate_model_params_match(new, old, "same-width executable semantics")
        for left, right in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new)):
            np.testing.assert_array_equal(left, right)

    def test_eval_mcts_main_forwards_recovered_metadata_and_static_selector(self):
        checkpoint = treated_checkpoint()
        for name in FOUNDATION_BEHAVIOR_DEFAULTS:
            del checkpoint["train_config"].__dict__[name]
        with (
            patch("sys.argv", ["eval_mcts.py", "--run_name", "test.pkl", "--n_envs", "3"]),
            patch.object(eval_mcts, "load_pkl_object", return_value=checkpoint),
            patch.object(eval_mcts, "TerraEnvBatch") as env_factory,
            patch.object(eval_mcts, "load_neural_network_for_checkpoint") as model_loader,
            patch.object(eval_mcts, "rollout_episode", return_value=([], {}, [])) as rollout,
            patch.object(eval_mcts, "print_stats"),
        ):
            eval_mcts.main()
        self.assertIs(env_factory.call_args.kwargs["executable_dig_observation"], True)
        restored = model_loader.call_args.args[0]
        env_cfgs = rollout.call_args.args[3]
        self.assertTrue(restored.executable_dig_observation)
        self.assertEqual(env_cfgs.executable_dig_observation.shape, (3,))
        validate_foundation_behavior_env(restored, env_cfgs)
        self.assertNotIn("executable_dig_observation", vars(checkpoint["train_config"]))

    def test_playback_main_loaders_forward_settings_before_environment_creation(self):
        class StopBeforeEnvironmentCreation(Exception):
            pass

        for module_name in (
            "eval", "visualize_paths", "visualize_mixed",
            "inference.inference_single_map",
        ):
            for executable in (False, True):
                with self.subTest(module=module_name, executable=executable):
                    module = importlib.import_module(module_name)
                    checkpoint = treated_checkpoint() if executable else {
                        "train_config": SavedConfig(), "env_config": EnvConfig(),
                    }
                    # Model a checkpoint whose new dataclass fields were not saved.
                    for name in FOUNDATION_BEHAVIOR_DEFAULTS:
                        del checkpoint["train_config"].__dict__[name]
                    namespace = dict(vars(module))
                    namespace["load_pkl_object"] = lambda path: checkpoint

                    def capture_environment(**kwargs):
                        self.assertIs(kwargs["executable_dig_observation"], executable)
                        validate_foundation_behavior_env(
                            namespace["config"], namespace["env_cfgs"],
                            env=SimpleNamespace(executable_dig_observation=executable),
                        )
                        raise StopBeforeEnvironmentCreation

                    namespace["TerraEnvBatch"] = capture_environment
                    path = Path(module.__file__)
                    main = next(
                        node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
                    )
                    # Run the actual CLI setup body, including its batching and
                    # overrides. Stop before maps, model init, or rendering.
                    code = compile(ast.Module(body=main.body, type_ignores=[]), str(path), "exec")
                    argv = [str(path), "--policy" if "single_map" in module_name else "--run_name", "test.pkl"]
                    with patch("sys.argv", argv), self.assertRaises(StopBeforeEnvironmentCreation):
                        exec(code, namespace)


if __name__ == "__main__":
    unittest.main()
