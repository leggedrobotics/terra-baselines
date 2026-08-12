"""rv2p1_scratch: the reward_v2_scratch baseline plus exactly two changes."""

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig

from configs.training_configs import get_config
from utils.models import get_model_ready
from utils.pooled_sampler import CONTINUOUS_MAX_MASS, CONTINUOUS_RULES

ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = ROOT / "scripts" / "run_v8_r2_reward_v2.sh"
RUNNER = ROOT / "scripts" / "run_v8_reward_timing_pilot.sh"
SBATCH = ROOT / "scripts" / "euler_v8_reward_timing_pilot" / "run.sbatch"
SUBMIT = ROOT / "scripts" / "euler_v8_reward_timing_pilot" / "submit.sh"

TERRA_REVISION = "46b5a1ddcd3b0e3a0d9e637af2e4ea94af51b4c8"
# The compact baseline trunk, unchanged: reward timing costs no parameters.
PARAMETERS = 2_856_701
PARAMETERS_WITHOUT_CARRY_WORK = 2_856_685


class AttrDict(dict):
    """``get_model_ready`` reads its config by attribute as well as by key."""

    __getattr__ = dict.__getitem__


def train_flags(path: Path) -> dict[str, str | None]:
    """Every ``train_mixed.py`` flag a launcher passes, value or None."""
    body = path.read_text().split("train_mixed.py", 1)[1]
    flags: dict[str, str | None] = {}
    for line in body.splitlines():
        line = line.strip().rstrip("\\").strip()
        match = re.match(r"^(--[a-z0-9_-]+)(?:\s+(.*))?$", line)
        if match is None:
            continue
        value = match.group(2)
        flags[match.group(1)] = value.strip('"') if value else None
    return flags


class RewardTimingLauncherTest(unittest.TestCase):
    def test_runner_is_the_baseline_plus_timing_and_sampler(self):
        baseline = train_flags(BASELINE_RUNNER)
        treatment = train_flags(RUNNER)
        self.assertEqual(
            set(treatment) - set(baseline), {"--reward_v2_timing_variant"}
        )
        self.assertEqual(set(baseline) - set(treatment), set())
        differing = {
            flag: (baseline[flag], treatment[flag])
            for flag in baseline
            if baseline[flag] != treatment[flag]
        }
        self.assertEqual(
            differing,
            {
                "--config": ("G-V8-CONTINUOUS-V2", "G-V8-CONTINUOUS-V3"),
                "--accepted-bank-sampler-profile": (
                    "continuous_banded_v2",
                    "continuous_banded_v3",
                ),
            },
        )
        self.assertEqual(
            treatment["--reward_v2_timing_variant"], "$REWARD_V2_TIMING_VARIANT"
        )
        self.assertIn(
            'REWARD_V2_TIMING_VARIANT="${REWARD_V2_TIMING_VARIANT:-1}"',
            RUNNER.read_text(),
        )

    def test_sbatch_pins_the_arm_contract(self):
        body = SBATCH.read_text()
        for assignment in (
            "ARM_NAME=rv2p1_scratch",
            f"EXPECTED_RUNTIME_TERRA_REVISION={TERRA_REVISION}",
            f"EXPECTED_PARAMETERS={PARAMETERS}",
            f"EXPECTED_PARAMETERS_WITHOUT_CARRY_WORK={PARAMETERS_WITHOUT_CARRY_WORK}",
            "SAMPLER_PROFILE=continuous_banded_v3",
            "REWARD_V2_TIMING_VARIANT=1",
            "PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4",
        ):
            self.assertIn(assignment, body)
        for contract_line in (
            '"reward_v2_timing=gamma1_stepcost_3.6"',
            '"shaping_gamma=1.0"',
            '"step_cost_total=3.6"',
            '"reward_v2_step_cost_total=3.6"',
            '"sampler_profile=$SAMPLER_PROFILE"',
            '"reward_v2_timing_variant=$REWARD_V2_TIMING_VARIANT"',
            '"arm=$ARM_NAME"',
        ):
            self.assertIn(contract_line, body)
        # Main-run shape and the promotion panel spacing.
        self.assertIn("UPDATES=14000", body)
        self.assertIn("CHECKPOINT_INTERVAL=500", body)
        self.assertIn("EVAL_INTERVAL=1000", body)
        self.assertIn("NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512", body)
        self.assertIn("--accepted-panel promotion", body)
        self.assertIn("--accepted-panel \"$PANEL\"", body)
        self.assertIn("select_promotion.py", body)

    def test_submit_pins_the_same_revision_and_gates_the_smoke(self):
        body = SUBMIT.read_text()
        self.assertIn(f"EXPECTED_RUNTIME_TERRA_REVISION={TERRA_REVISION}", body)
        self.assertIn("ARM_NAME=rv2p1_scratch", body)
        self.assertIn("SAMPLER_PROFILE=continuous_banded_v3", body)
        self.assertIn("REWARD_V2_TIMING_VARIANT=1", body)
        self.assertIn("smoke) PARTITION=gpuhe.4h", body)
        self.assertIn("phase1) PARTITION=gpuhe.24h", body)
        self.assertIn("WALLTIME=23:45:00", body)
        self.assertIn("--gpus='$GPU_TYPE:4'", body)
        self.assertIn("test \"$SMOKE_STATE\" = COMPLETED", body)
        self.assertIn(
            "scripts/euler_v8_reward_timing_pilot/run.sbatch", body
        )

    def test_launcher_is_account_aware(self):
        """Every account-owned path derives from the selected Euler account."""
        submit = SUBMIT.read_text()
        # The shared resolver, not a hardcoded /cluster/home/<user>.
        self.assertIn("source \"$REPO/cluster/euler_account.sh\"", submit)
        self.assertIn('terra_euler_configure "${TERRA_EULER_USER:-lterenzi}"', submit)
        self.assertIn('REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"', submit)
        for derived in (
            "${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/",
            "${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/",
        ):
            self.assertIn(derived, submit)
        # The venv is a shared /cluster/project/rsl tree: account-independent.
        self.assertIn("/cluster/project/rsl/lterenzi/", submit)
        # The quota gate reads the selected account's home, via the shared parser.
        self.assertIn(
            'remote lquota | "$REPO/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT"',
            submit,
        )
        # phase1 refuses to burn a GPU slot without W&B credentials.
        self.assertIn("api.wandb.ai", submit)

        sbatch = SBATCH.read_text()
        self.assertIn('test "$(id -un)" = "$TERRA_EULER_USER"', sbatch)
        self.assertIn('test "$HOME" = "$TERRA_EULER_HOME_ROOT"', sbatch)
        self.assertIn(
            'lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT"',
            sbatch,
        )
        self.assertIn('"euler_user=$TERRA_EULER_USER"', sbatch)
        # W&B routing arrives from the launcher instead of being hardcoded.
        self.assertNotIn("WANDB_ENTITY=aless-weber-eth", sbatch)
        self.assertIn("export WANDB_ENTITY WANDB_PROJECT", sbatch)
        self.assertIn("WANDB_ENTITY=$WANDB_ENTITY", submit)
        # The smoke that gates phase1 must belong to the same account.
        self.assertIn(
            'test "$(stat -c %U "$SMOKE_RUN/run_contract.env")" = "$TERRA_EULER_USER"',
            sbatch,
        )

    def test_v3_preset_selects_the_capped_sampler(self):
        preset = get_config("G-V8-CONTINUOUS-V3")
        self.assertEqual(preset.accepted_bank_arm, "G-UNIFORM")
        sampler = preset.pooled_sampler
        self.assertTrue(sampler.enabled)
        self.assertEqual(sampler.rule, "continuous_banded_v3")
        self.assertIn("continuous_banded_v3", CONTINUOUS_RULES)
        self.assertEqual(CONTINUOUS_MAX_MASS, 0.15)
        # The frozen continuous settings the sampler validates on construction.
        self.assertEqual(sampler.update_interval, 150)
        self.assertEqual(sampler.mastery_threshold, 0.80)
        self.assertEqual(sampler.min_episodes, 32)
        self.assertEqual(sampler.competence_ema, 0.30)
        self.assertEqual(sampler.max_mass, CONTINUOUS_MAX_MASS)

    def test_parameter_count_is_the_compact_baseline(self):
        environment = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
        )
        counts = {}
        for carry in (False, True):
            _, params = get_model_ready(
                jax.random.PRNGKey(0),
                AttrDict(
                    clip_action_maps=True,
                    loaded_max=100,
                    local_map_normalization_bounds=(-16, 16),
                    maps_net_normalization_bounds=(-10, 10),
                    model_core="mlp",
                    model_size="medium",
                    num_prev_actions=5,
                    critic_hidden_dims=(512, 256),
                    encoder_compute_dtype="bfloat16",
                    attention_compute_dtype="float32",
                    map_encoder="resnet_spatial_8x8_se_xattn",
                    resnet_stage_channels=(24, 48, 64, 96),
                    resnet_blocks_per_stage=(2, 2, 3, 3),
                    carry_work_observation=carry,
                ),
                environment,
            )
            counts[carry] = sum(
                int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params)
            )
        self.assertEqual(counts[True], PARAMETERS)
        self.assertEqual(counts[False], PARAMETERS_WITHOUT_CARRY_WORK)


if __name__ == "__main__":
    unittest.main()
