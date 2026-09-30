"""Terra's machine working rules in train_mixed, the presets and the evaluators."""

import collections
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from terra.config import EnvConfig

from configs.training_configs import get_config
from train_mixed import MACHINE_RULE_FIELDS, apply_machine_rules, machine_rules_effective

REPO = Path(__file__).resolve().parents[1]
OFF = dict(
    dig_min_radius_m=0.0, dump_min_radius_m=0.0, dug_clearance_m=0.0,
    dump_min_dug_distance_m=0.0, centre_chassis_on_base=False,
)


class MachineRulesPlumbingTest(unittest.TestCase):
    def test_unset_rules_leave_the_env_config(self):
        env = EnvConfig()
        # An old checkpoint's train_config has none of the fields.
        self.assertIs(apply_machine_rules(env, SimpleNamespace()), env)
        self.assertIs(apply_machine_rules(env, SimpleNamespace(**dict.fromkeys(MACHINE_RULE_FIELDS))), env)
        self.assertEqual(machine_rules_effective(env), OFF)

    def test_set_rules_override_a_checkpoint_env_config(self):
        base = EnvConfig()
        checkpoint_env = base._replace(agent=base.agent._replace(dug_clearance_m=0.6, dump_max_radius_m=5.5))
        config = SimpleNamespace(dig_min_radius_m=4, dug_clearance_m=0.57, centre_chassis_on_base=True)
        env = apply_machine_rules(checkpoint_env, config)
        self.assertEqual(machine_rules_effective(env), {
            **OFF, "dig_min_radius_m": 4.0, "dug_clearance_m": 0.57, "centre_chassis_on_base": True,
        })
        self.assertIsInstance(env.agent.dig_min_radius_m, float)
        self.assertEqual(env.agent.dump_max_radius_m, 5.5)  # dump reach is set separately

    def test_a_terra_without_the_rules_refuses_them(self):
        agent = collections.namedtuple("Agent", ["dump_max_radius_m"])(0.0)
        env = collections.namedtuple("Env", ["agent"])(agent)
        self.assertEqual(machine_rules_effective(env), {})
        self.assertIs(apply_machine_rules(env, SimpleNamespace(dug_clearance_m=None)), env)
        with self.assertRaisesRegex(RuntimeError, "no agent.dug_clearance_m"):
            apply_machine_rules(env, SimpleNamespace(dug_clearance_m=0.6))

    def test_machine_rules_preset_is_the_generalist_bank_with_the_rules(self):
        base = get_config("gru_generalist_512")
        rules = get_config("gru_generalist_512_machine_rules")
        self.assertEqual(
            (rules.dig_min_radius_m, rules.dump_max_radius_m, rules.centre_chassis_on_base),
            (4.0, 6.0, True),
        )
        self.assertIsNone(rules.dug_clearance_m)  # the launcher's parameter
        self.assertIsNone(rules.dump_min_radius_m)
        self.assertIsNone(rules.dump_min_dug_distance_m)
        for name in ("dump_max_radius_m", *MACHINE_RULE_FIELDS):
            self.assertIsNone(getattr(base, name), name)
        for name in (
            "agent_types", "action_types", "maps", "curriculum", "pooled_sampler",
            "relocation_progress_mult", "trench_alignment_observation",
            "require_trench_alignment_metadata", "enforce_trench_dig_alignment",
            "trench_dig_standoff_enforced", "trench_dig_max_offset_m",
        ):
            self.assertEqual(getattr(rules, name), getattr(base, name), name)

    def test_command_lines_accept_the_rule_flags(self):
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
        for script, flags in (
            ("train_mixed.py", ("--dig_min_radius_m", "--dump_min_radius_m", "--dug_clearance_m",
                                "--dump_min_dug_distance_m", "--centre_chassis_on_base",
                                "--no-centre_chassis_on_base")),
            ("eval_fixed_bank.py", ("--dig-min-radius-m", "--dump-min-radius-m", "--dug-clearance-m",
                                    "--dump-min-dug-distance-m", "--centre-chassis-on-base",
                                    "--no-centre-chassis-on-base")),
            ("scripts/euler_gru_generalist_512/eval_known_starts.py",
             ("--dig-min-radius-m", "--dug-clearance-m", "--centre-chassis-on-base")),
        ):
            text = subprocess.run(
                [sys.executable, str(REPO / script), "--help"], env=env, cwd=REPO,
                capture_output=True, text=True, check=True,
            ).stdout
            for flag in flags:
                self.assertIn(flag, text, (script, flag))


if __name__ == "__main__":
    unittest.main()
