"""v6_3m_yolo_rv2: paired with reward_v2_scratch except the seven treatment flags."""

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input

ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = ROOT / "scripts" / "run_v8_r2_reward_v2.sh"
TREATMENT_RUNNER = ROOT / "scripts" / "run_v8_v6_yolo_rv2.sh"
SBATCH = ROOT / "scripts" / "euler_v8_v6_yolo_rv2" / "run.sbatch"
SUBMIT = ROOT / "scripts" / "euler_v8_v6_yolo_rv2" / "submit.sh"
VERIFY_SMOKE = ROOT / "scripts" / "euler_v8_v6_yolo_rv2" / "verify_smoke.py"

# Frozen: the v6 readout block consuming terra's 9-wide (carry-work) agent state.
V6_3M_RV2_PARAMETERS = 2_134_771
V6_3M_PARAMETERS_WITHOUT_CARRY_WORK = 2_134_755


class AttrDict(dict):
    """``get_model_ready`` reads its config by attribute as well as by key."""

    __getattr__ = dict.__getitem__

# Two flags the baseline also passes, with a different value.
RETUNED_FLAGS = {
    "--map_encoder": ("resnet_spatial_8x8_se_xattn", "resnet_spatial_8x8_se_sa_xattn"),
    "--resnet_blocks_per_stage": ("2,2,3,3", "3,3,2,2"),
}
# Five flags the baseline does not pass at all. D3 masking is passed via the
# MASK_ARGS conditional (default ON, MASK_VARIANT=nomask disables it), so it
# is asserted textually below rather than through the flag parser.
ADDED_FLAGS = {
    "--token_mixer_residual_init_scale": "0.1",
    "--flatten_reduce_channels": "32",
    "--attn_latent_queries": "8",
    "--aux_coef": "0.25",
    "--vf_coef": "0.5",
}


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


class PairedLauncherTest(unittest.TestCase):
    def test_treatment_differs_from_the_baseline_only_in_the_declared_flags(self):
        baseline = train_flags(BASELINE_RUNNER)
        treatment = train_flags(TREATMENT_RUNNER)
        self.assertGreater(len(baseline), 30)
        # Nothing the baseline passes is dropped.
        self.assertEqual(set(baseline) - set(treatment), set())
        self.assertEqual(set(treatment) - set(baseline), set(ADDED_FLAGS))
        for name, value in ADDED_FLAGS.items():
            self.assertEqual(treatment[name], value, name)
        differing = {name for name in baseline if baseline[name] != treatment[name]}
        self.assertEqual(differing, set(RETUNED_FLAGS))
        for name, (before, after) in RETUNED_FLAGS.items():
            self.assertEqual(baseline[name], before, name)
            self.assertEqual(treatment[name], after, name)
        # D3 masking rides in the MASK_ARGS conditional: default on, and the
        # no-mask ablation is selected by ACTION_LOGIT_MASKING=0.
        text = TREATMENT_RUNNER.read_text()
        self.assertIn('ACTION_LOGIT_MASKING="${ACTION_LOGIT_MASKING:-1}"', text)
        self.assertIn("MASK_ARGS=(--action_logit_masking)", text)
        self.assertIn('"${MASK_ARGS[@]}"', text)

    def test_reward_v2_contract_flags_survive_the_port(self):
        treatment = train_flags(TREATMENT_RUNNER)
        self.assertEqual(treatment["--config"], "G-V8-CONTINUOUS-V2")
        self.assertEqual(
            treatment["--accepted-bank-sampler-profile"], "continuous_banded_v2"
        )
        self.assertEqual(treatment["--reward_stage"], "reward_v2")
        self.assertEqual(
            treatment["--distance_protocol_id"],
            "obstacle_geodesic_8_physical_global_v1",
        )
        self.assertEqual(treatment["--distance_sidecar_sha256"], "$SIDECAR_SHA256")
        self.assertIn("--carry_work_observation", treatment)
        self.assertIn("--no_value_clip", treatment)
        self.assertIn("--flat_minibatch_shuffle", treatment)
        self.assertIn("--fail_on_nonfinite", treatment)
        self.assertEqual(treatment["--lr"], "3e-4")
        self.assertEqual(treatment["--critic_hidden_dims"], "512,256")
        self.assertEqual(treatment["--resnet_stage_channels"], "24,48,64,96")
        self.assertEqual(treatment["--encoder_compute_dtype"], "bfloat16")
        self.assertEqual(treatment["--attention_compute_dtype"], "float32")
        self.assertEqual(treatment["--ent_schedule_start"], "0.15")
        self.assertEqual(treatment["--ent_schedule_end"], "0.02")
        for forbidden in ("--warm_start_from", "--teacher_checkpoint",
                          "--resume_from", "--prepared_fork_from"):
            self.assertNotIn(forbidden, treatment)

    def test_cluster_shape_matches_the_baseline_run(self):
        sbatch = SBATCH.read_text()
        baseline_sbatch = (
            ROOT / "scripts" / "euler_v8_r2_reward_v2" / "run.sbatch"
        ).read_text()
        for shared in (
            "UPDATES=14000",
            "export ENTROPY_SCHEDULE_STEPS=20000",
            "export NUM_DEVICES=4 NUM_ENVS_PER_DEVICE=512 NUM_STEPS=32 NUM_MINIBATCHES=32",
            "PROTOCOL_TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4",
            'test "${#GPU_NAMES[@]}" -eq 4',
            "#SBATCH --gpus=rtx_4090:4",
            "#SBATCH --partition=gpuhe.24h",
            "used < 45.0",
            "EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5",
        ):
            self.assertIn(shared, sbatch, shared)
            self.assertIn(shared, baseline_sbatch, shared)
        # Runtime terra diverges BY DESIGN: the treatment runs the D3-mask
        # terra (reward-v2 base + obs['action_mask']); the baseline stays on
        # the reward-v2 revision it launched with.
        self.assertIn(
            "EXPECTED_RUNTIME_TERRA_REVISION="
            "04c67bbafce2cb3d1a1de35384dfde477d244349",
            sbatch,
        )
        self.assertIn(
            "EXPECTED_RUNTIME_TERRA_REVISION="
            "3051054bc4c713d95905d3f954e6eabf55d6a85a",
            baseline_sbatch,
        )
        self.assertIn("scripts/run_v8_v6_yolo_rv2.sh", sbatch)
        self.assertIn(f"model_parameter_count={V6_3M_RV2_PARAMETERS}", sbatch)
        self.assertIn(
            "model_parameter_count_without_carry_work="
            f"{V6_3M_PARAMETERS_WITHOUT_CARRY_WORK}",
            sbatch,
        )
        self.assertIn(f"assert counts[True] == {V6_3M_RV2_PARAMETERS:_}", sbatch)
        self.assertIn(
            f"PARAMETER_COUNT = {V6_3M_RV2_PARAMETERS:_}", VERIFY_SMOKE.read_text()
        )

        submit = SUBMIT.read_text()
        self.assertLess(
            submit.index('if [ "$SUBMIT" = 0 ]'), submit.index('ssh "$REMOTE_HOST"')
        )
        self.assertIn("--gpus='$GPU_TYPE:4'", submit)
        self.assertIn("scripts/euler_v8_v6_yolo_rv2/run.sbatch", submit)
        self.assertIn("CAMPAIGN=terra_v8_v6_yolo_rv2", submit)


class CarryWorkObservationContractTest(unittest.TestCase):
    """The port hazard: carry-work must not move the aux decoder's obs indices."""

    def test_carry_work_widens_agent_states_not_the_obs_list(self):
        batch_cfg = BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
        # reward-v2 terra appends normalized carry work at agent-state index 8.
        self.assertEqual(batch_cfg.agent.num_state_obs, 9)

        maps = {
            key: jnp.zeros((2, 64, 64))
            for key in (
                "traversability_mask", "reachability_mask", "action_map",
                "target_map", "padding_mask", "dumpability_mask",
                "interaction_mask",
            )
        }
        obs = {
            "agent_states": jnp.zeros((2, 4, batch_cfg.agent.num_state_obs)),
            "agent_active": jnp.zeros((2, 4)),
            "num_agents": jnp.zeros((2,)),
            **{
                key: jnp.zeros((2, 12))
                for key in (
                    "local_map_action_neg", "local_map_action_pos",
                    "local_map_target_neg", "local_map_target_pos",
                    "local_map_dumpability", "local_map_obstacles",
                    "local_map_border_workspace",
                    "local_map_edge_alignment_error",
                    "local_map_border_diggable",
                )
            },
            "agent_width": jnp.zeros((2,)),
            "agent_height": jnp.zeros((2,)),
            **maps,
        }
        entries = obs_to_model_input(
            dict(obs), jnp.zeros((2, 5), dtype=jnp.int32),
            {"clip_action_maps": False, "local_map_area_scale": 1.0},
        )
        self.assertEqual(len(entries), 22)
        self.assertEqual(entries[0].shape[-1], 9)
        # train.aux_decoder_loss reads exactly these five positions.
        for index, name in (
            (12, "traversability_mask"), (14, "action_map"), (15, "target_map"),
            (18, "padding_mask"), (19, "dumpability_mask"),
        ):
            self.assertIs(entries[index], obs[name], f"[{index}] != {name}")

    def test_carry_work_costs_sixteen_weights_and_keeps_the_aux_head(self):
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
                    map_encoder="resnet_spatial_8x8_se_sa_xattn",
                    resnet_stage_channels=(24, 48, 64, 96),
                    resnet_blocks_per_stage=(3, 3, 2, 2),
                    token_mixer_residual_init_scale=0.1,
                    flatten_reduce_channels=32,
                    attn_latent_queries=8,
                    aux_coef=0.25,
                    carry_work_observation=carry,
                ),
                environment,
            )
            counts[carry] = sum(
                int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params)
            )
            aux = [
                jax.tree_util.keystr(path)
                for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]
                if "aux_decoder" in jax.tree_util.keystr(path)
            ]
            self.assertEqual(len(aux), 6)
        self.assertEqual(counts[False], V6_3M_PARAMETERS_WITHOUT_CARRY_WORK)
        self.assertEqual(counts[True], V6_3M_RV2_PARAMETERS)
        self.assertEqual(counts[True] - counts[False], 16)


if __name__ == "__main__":
    unittest.main()
