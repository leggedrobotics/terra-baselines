"""Task transfer, equal-data launch arguments, and explicit reward comparisons."""

import ast
import copy
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from configs.training_configs import get_config
from scripts.build_v8_benchmark_dashboard import validate_pair
import train_mixed


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/foundation_reward_sweep/train.sh"


def config(**changes):
    values = dict(
        reward_stage="reward_v2", gamma=0.9984,
        distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        distance_sidecar_sha256="a" * 64, update_epochs=2, num_minibatches=32,
        admissible_dig_observation=True, executable_dig_observation=False,
        lateral_dig_cost=0.0, base_travel_cost=0.0, base_turn_cost=0.0,
        finetune_foundation_behavior=False, finetune_task_bank=False,
        resume_from="parent.pkl", warm_start_from=None, resume_update=None,
        load_env_from_checkpoint=False, pooled_sampler=None,
        partial_reset_root=None, partial_reset_bank_sha256=None,
    )
    return SimpleNamespace(**(values | changes))


def checkpoint(saved_config=None):
    saved = saved_config or config()
    return dict(
        train_config=saved, r2_protocol_receipt=train_mixed._r2_protocol_receipt(saved),
        optimizer_state={"sentinel": np.array([1.5, 2.5])},
        train_state_step=np.array(320000), next_update=5000,
    )


def validate(saved, current):
    train_mixed._validate_r2_resume_checkpoint(saved, train_mixed._r2_protocol_receipt(current), current)


def test_task_transfer_changes_only_bank_identity_and_preserves_native_state():
    saved = checkpoint()
    current = config(distance_sidecar_sha256="b" * 64)
    with pytest.raises(ValueError, match="protocol receipt"):
        validate(saved, current)
    current.finetune_task_bank = True
    original_optimizer = saved["optimizer_state"]
    validate(saved, current)
    assert saved["optimizer_state"] is original_optimizer
    np.testing.assert_array_equal(original_optimizer["sentinel"], [1.5, 2.5])
    assert int(saved["train_state_step"]) == 320000
    assert saved["next_update"] == 5000


def test_reward_and_observation_changes_require_the_separate_opt_in():
    saved = checkpoint()
    current = config(distance_sidecar_sha256="b" * 64, finetune_task_bank=True,
                     executable_dig_observation=True, lateral_dig_cost=0.25)
    with pytest.raises(ValueError, match="protocol receipt"):
        validate(saved, current)
    current.finetune_foundation_behavior = True
    validate(saved, current)
    train_mixed._validate_checkpoint_architecture(saved, current)


@pytest.mark.parametrize("field,value", [
    ("distance_protocol_id", "other-distance"),
    ("reward_protocol_id", "other-reward"),
    ("reward_v2_timing_variant", 1),
    ("constants", {"potential_gamma": 1.0}),
])
def test_task_transfer_does_not_relax_other_r2_fields(field, value):
    saved = checkpoint()
    saved["r2_protocol_receipt"][field] = value
    current = config(distance_sidecar_sha256="b" * 64, finetune_task_bank=True,
                     finetune_foundation_behavior=True)
    with pytest.raises(ValueError, match="protocol receipt"):
        validate(saved, current)


@pytest.mark.parametrize("changes", [
    {"load_env_from_checkpoint": True}, {"warm_start_from": "other.pkl"},
    {"resume_from": None}, {"resume_update": 2}, {"reward_stage": "dense_skill"},
    {"pooled_sampler": {"enabled": True}}, {"partial_reset_root": "/old/partial"},
    {"partial_reset_bank_sha256": "c" * 64},
])
def test_task_transfer_rejects_current_state_carryover(changes):
    with pytest.raises(ValueError, match="finetune_task_bank"):
        train_mixed._validate_task_bank_transfer(config(finetune_task_bank=True, **changes))


@pytest.mark.parametrize("field,value", [
    ("pooled_sampler_state", {}), ("partial_reset_curriculum", {}),
    ("train_config", None),
])
def test_task_transfer_rejects_saved_state_carryover(field, value):
    saved = checkpoint()
    saved[field] = value
    with pytest.raises(ValueError, match="finetune_task_bank"):
        validate(saved, config(finetune_task_bank=True))


@pytest.mark.parametrize("changes", [
    {"pooled_sampler": {"enabled": True}}, {"partial_reset_root": "/old/partial"},
    {"partial_reset_bank_sha256": "c" * 64}, {"reward_stage": "dense_skill"},
])
def test_task_transfer_rejects_saved_sampler_or_partial_config(changes):
    saved = checkpoint()
    saved["train_config"] = config(**changes)
    with pytest.raises(ValueError, match="finetune_task_bank"):
        validate(saved, config(finetune_task_bank=True))


def test_transfer_still_requires_the_native_optimizer_clock_and_receipt():
    current = config(finetune_task_bank=True)
    saved = checkpoint()
    saved["train_state_step"] = np.array(319999)
    with pytest.raises(ValueError, match="optimizer clock"):
        validate(saved, current)
    saved = checkpoint()
    del saved["optimizer_state"]
    with pytest.raises(ValueError, match="optimizer_state"):
        validate(saved, current)
    saved = checkpoint()
    del saved["r2_protocol_receipt"]["distance_sidecar_sha256"]
    with pytest.raises(ValueError, match="sidecar hashes"):
        validate(saved, current)
    with pytest.raises(ValueError, match="current R2"):
        train_mixed._validate_r2_resume_checkpoint(checkpoint(), None, current)


def test_new_bank_checkpoint_continues_normally_after_transfer():
    transferred = config(distance_sidecar_sha256="b" * 64, finetune_task_bank=True,
                         executable_dig_observation=True, lateral_dig_cost=0.25,
                         finetune_foundation_behavior=True)
    saved = checkpoint(transferred)
    current = copy.copy(transferred)
    current.finetune_task_bank = False
    current.finetune_foundation_behavior = False
    current.load_env_from_checkpoint = True
    validate(saved, current)
    train_mixed._validate_checkpoint_architecture(saved, current)


@pytest.fixture
def launch_env(tmp_path):
    env = os.environ.copy()
    env.update(DATASET_PATH=str(tmp_path / "bank"), DISTANCE_SIDECAR_SHA="b" * 64,
               RUN_DIR=str(tmp_path / "run"), RUN_NAME="B_exec", RESUME_FROM=str(tmp_path / "parent.pkl"),
               START_UPDATE="5000", TARGET_UPDATE="7000", EXECUTABLE_DIG_OBSERVATION="1",
               SEED="20260907", BANK_TRANSFER="1", LATERAL_DIG_COST="0", BASE_TRAVEL_COST="0",
               BASE_TURN_COST="0", NUM_DEVICES="1", BEHAVIOR_COST_RAMP_UPDATES="0",
               BEHAVIOR_FINETUNE="0")
    return env


def script_args(env):
    return shlex.split(subprocess.check_output(["bash", str(SCRIPT), "--print-args"], env=env, text=True))


@pytest.mark.parametrize("family,config,maps_path", [
    ("foundation", "foundation_reward_sweep", "train/all"),
    ("trench", "trench_align_v2_specialist_spec", "train_v2_pooled_trench15"),
])
def test_scratch_never_imports_checkpoint_or_transfer_state(launch_env, family, config, maps_path):
    launch_env.update(INITIALIZATION="scratch", START_UPDATE="0", BANK_TRANSFER="0", TASK_FAMILY=family)
    launch_env.pop("RESUME_FROM")
    args = script_args(launch_env)
    assert args[args.index("--config") + 1] == config
    preset = get_config(config)
    assert len(preset.maps) == 1 and preset.maps[0].maps_path == maps_path
    assert preset.maps[0].max_steps_in_episode == 450
    assert args[args.index("--total_timesteps") + 1] == str(7000 * 16384)
    for flag in ("--resume_from", "--warm_start_from", "--resume_update", "--load_env_from_checkpoint",
                 "--finetune_task_bank", "--finetune_foundation_behavior"):
        assert flag not in args
    for changes in ({"RESUME_FROM": "parent.pkl"}, {"START_UPDATE": "5000"}, {"BANK_TRANSFER": "1"}):
        result = subprocess.run(["bash", str(SCRIPT), "--print-args"], env=launch_env | changes,
                                capture_output=True, text=True)
        assert result.returncode != 0
    treatment = script_args(launch_env | {"LATERAL_DIG_COST": "0.5", "BASE_TRAVEL_COST": "0.01",
                                       "BASE_TURN_COST": "0.04"})
    expected = args.copy()
    for flag, value in (("--lateral_dig_cost", "0.5"), ("--base_travel_cost", "0.01"),
                        ("--base_turn_cost", "0.04")):
        expected[expected.index(flag) + 1] = value
    assert treatment == expected


def test_sourceable_and_executable_wrappers_share_the_same_arguments(launch_env):
    rendered = script_args(launch_env)
    sourced = subprocess.check_output(
        ["bash", "-c", 'source "$1"; printf "%s\\0" "${FOUNDATION_TRAIN_ARGS[@]}"', "_", str(SCRIPT)],
        env=launch_env,
    ).decode().rstrip("\0").split("\0")
    assert sourced == rendered
    # Resolve the actual CLI and preset into the dataclass without starting training.
    module = ast.parse((REPO / "train_mixed.py").read_text())
    main = next(node for node in module.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
    stop = next(index for index, node in enumerate(main.body)
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "config"
                                                        for target in node.targets))
    namespace = dict(vars(train_mixed))
    with patch.object(sys, "argv", ["train_mixed.py", *rendered]), patch.dict(os.environ, launch_env):
        exec(compile(ast.Module(body=main.body[:stop + 1], type_ignores=[]), "train_mixed.py", "exec"), namespace)
    args = namespace["args"]
    assert not namespace["unknown"]
    assert (args.num_devices, args.num_envs_per_device, args.num_steps) == (1, 512, 32)
    assert (args.update_epochs, args.num_minibatches) == (2, 32)
    assert args.total_timesteps == 7000 * 16384
    assert args.log_eval_interval == 0
    assert args.finetune_task_bank and args.finetune_foundation_behavior
    assert args.executable_dig_observation and not args.load_env_from_checkpoint
    preset = get_config(args.config)
    assert len(preset.maps) == 1 and preset.maps[0].maps_path == "train/all"
    assert preset.maps[0].max_steps_in_episode == 450
    assert preset.trench_alignment_observation
    assert preset.pooled_sampler is None or not preset.pooled_sampler.enabled
    # Frozen values independently compared to the actual generalist u5000 parent.
    resolved = namespace["config"]
    expected = dict(
        gamma=0.9984, gae_lambda=0.95, clip_eps=0.2, max_grad_norm=0.5,
        lr=3e-4, vf_coef=2.0, use_value_clip=False, flat_minibatch_shuffle=False,
        ent_schedule_start=0.15, ent_schedule_end=0.02, ent_schedule_steps=20000,
        model_size="medium", model_core="mlp", actor_core="mlp",
        map_encoder="resnet_spatial_8x8_se_sa_xattn", encoder_compute_dtype="bfloat16",
        attention_compute_dtype="float32", critic_hidden_dims=(512, 256),
        resnet_stage_channels=(24, 48, 64, 96), resnet_blocks_per_stage=(2, 2, 3, 3),
        flatten_reduce_channels=32, attn_latent_queries=8, aux_coef=0.0,
        clip_action_maps=True, local_map_normalization_bounds=(-16, 16),
        maps_net_normalization_bounds=(-10, 10), loaded_max=100, local_map_area_scale=1.0,
        action_logit_masking=False, carry_work_observation=True,
        admissible_dig_observation=True, relocation_distance_observation=True,
        trench_alignment_observation=True, stall_age_observation=False,
        movement_feasibility_observation=False, previous_outcome_observation=False,
        agent_types_override=(0,), action_types_override=(0,),
        enforce_trench_dig_alignment=True, trench_dig_standoff_enforced=False,
        enforce_foundation_border_alignment=None, reward_v2_timing_variant=0,
        agent_move_tiles=None, dig_radius_tiles=None, truck_capacity=None,
        skidsteer_capacity=None, reward_normalizer=None, relocation_progress_mult=1.5,
    )
    assert {key: getattr(resolved, key) for key in expected} == expected
    assert 5 * len(resolved.agent_types_override) == 5  # runtime history width


def test_continuation_wrapper_drops_transfer_flags_and_restores_env(launch_env):
    launch_env.update(BANK_TRANSFER="0", START_UPDATE="7000", TARGET_UPDATE="10000")
    args = script_args(launch_env)
    assert "--load_env_from_checkpoint" in args
    assert "--finetune_task_bank" not in args
    assert "--finetune_foundation_behavior" not in args
    assert "--no-load-env-from-checkpoint" not in args


@pytest.mark.parametrize("devices,envs", [(1, 512), (2, 256), (4, 128)])
def test_gpu_layout_keeps_global_batch_and_optimizer_work(launch_env, devices, envs):
    launch_env["NUM_DEVICES"] = str(devices)
    args = script_args(launch_env)
    assert args[args.index("--num_devices") + 1] == str(devices)
    assert args[args.index("--num_envs_per_device") + 1] == str(envs)
    assert devices * envs * int(args[args.index("--num_steps") + 1]) == 16384
    assert args[args.index("--total_timesteps") + 1] == str(7000 * 16384)
    assert int(args[args.index("--update_epochs") + 1]) * int(args[args.index("--num_minibatches") + 1]) == 64


@pytest.mark.parametrize("devices", ["0", "3", "8", "1.5", "many"])
def test_gpu_layout_rejects_unsupported_values(launch_env, devices):
    result = subprocess.run(["bash", str(SCRIPT), "--print-args"], env=launch_env | {"NUM_DEVICES": devices},
                            capture_output=True, text=True)
    assert result.returncode == 2 and "NUM_DEVICES must be 1, 2 or 4" in result.stderr


def test_behavior_ramp_uses_native_same_bank_resume_and_can_restore_when_omitted(launch_env):
    launch_env.update(BANK_TRANSFER="0", BEHAVIOR_FINETUNE="1", BEHAVIOR_COST_RAMP_UPDATES="2500")
    args = script_args(launch_env)
    assert args[args.index("--behavior_cost_ramp_updates") + 1] == "2500"
    assert "--finetune_foundation_behavior" in args and "--load_env_from_checkpoint" in args
    assert "--finetune_task_bank" not in args and "--resume_update" not in args
    launch_env.update(BEHAVIOR_COST_RAMP_UPDATES="0", BEHAVIOR_FINETUNE="0")
    assert "--behavior_cost_ramp_updates" not in script_args(launch_env)


@pytest.mark.parametrize("changes", [
    {"BEHAVIOR_COST_RAMP_UPDATES": "-1"}, {"BEHAVIOR_COST_RAMP_UPDATES": "1.5"},
    {"BEHAVIOR_COST_RAMP_UPDATES": "2500", "BEHAVIOR_FINETUNE": "0", "BANK_TRANSFER": "0"},
    {"BEHAVIOR_COST_RAMP_UPDATES": "2500", "BEHAVIOR_FINETUNE": "1", "BANK_TRANSFER": "1"},
])
def test_behavior_ramp_rejects_invalid_duration_or_transfer(launch_env, changes):
    result = subprocess.run(["bash", str(SCRIPT), "--print-args"], env=launch_env | changes,
                            capture_output=True, text=True)
    assert result.returncode == 2 and ("RAMP_UPDATES" in result.stderr or "behavior-cost ramp requires" in result.stderr)


def test_checkpoint_interval_override_supports_callback_smoke(launch_env):
    launch_env["CHECKPOINT_INTERVAL"] = "2"
    args = script_args(launch_env)
    assert args[args.index("--checkpoint_interval") + 1] == "2"
    launch_env["CHECKPOINT_INTERVAL"] = "0"
    result = subprocess.run(["bash", str(SCRIPT), "--print-args"], env=launch_env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "CHECKPOINT_INTERVAL must be a positive integer" in result.stderr


def test_legacy_control_preserves_old_affordance(launch_env):
    launch_env["EXECUTABLE_DIG_OBSERVATION"] = "0"
    args = script_args(launch_env)
    assert "--executable_dig_observation" not in args
    assert "--admissible_dig_observation" in args


def test_train_entry_override_and_budget_receipt_are_used(launch_env, tmp_path):
    (tmp_path / "bank/train/all").mkdir(parents=True)
    (tmp_path / "parent.pkl").touch()
    python_stub = tmp_path / "python"
    python_stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MOCK_OUTPUT"\n')
    python_stub.chmod(0o755)
    launch_env.update(TERRA_PYTHON=str(python_stub), TRAIN_ENTRY=str(tmp_path / "entry.py"),
                      MOCK_OUTPUT=str(tmp_path / "args"))
    subprocess.run(["bash", str(SCRIPT)], env=launch_env, check=True)
    assert (tmp_path / "args").read_text().splitlines()[:2] == ["-u", str(tmp_path / "entry.py")]
    budget = (tmp_path / "run/training_budget.env").read_text()
    assert "additional_transitions=32768000" in budget
    assert "adam_steps_per_update=64" in budget


def test_fixed_eval_wrapper_uses_same_full_450_step_panel(tmp_path):
    python_stub = tmp_path / "python"
    python_stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MOCK_OUTPUT"\n'
                           'printf "%s\\n" "$EVAL_FORWARD_CHUNK" > "$MOCK_OUTPUT.chunk"\n')
    python_stub.chmod(0o755)
    env = dict(os.environ, TERRA_PYTHON=str(python_stub), BANK_ROOT=str(tmp_path / "bank"),
               MOCK_OUTPUT=str(tmp_path / "args"))
    env.pop("EVAL_FORWARD_CHUNK", None)
    subprocess.run(["bash", str(SCRIPT.with_name("eval.sh")), "checkpoint.pkl", "result.json", "validation"],
                   env=env, check=True)
    args = (tmp_path / "args").read_text().splitlines()
    assert args[args.index("--horizon") + 1] == "450"
    assert args[args.index("--split") + 1] == "validation"
    assert args[args.index("--strata") + 1] == "all"
    assert "--stochastic" not in args
    assert (tmp_path / "args.chunk").read_text().strip() == "32"


def test_eval_forward_chunk_override_splits_64_episodes_exactly(monkeypatch):
    import eval_mcts
    monkeypatch.delenv("EVAL_FORWARD_CHUNK", raising=False)
    assert eval_mcts._configured_eval_forward_chunk() == 120
    monkeypatch.setenv("EVAL_FORWARD_CHUNK", "32")
    monkeypatch.setattr(eval_mcts, "EVAL_FORWARD_CHUNK", eval_mcts._configured_eval_forward_chunk())
    batches = []

    class Model:
        def apply(self, params, obs):
            batches.append(obs[0].shape[0])
            return obs[0][:, 0] + params, obs[0] * 2

    obs = (np.arange(64 * 3, dtype=np.float32).reshape(64, 3),)
    values, logits = eval_mcts._apply_in_batch_chunks(Model(), 0.5, obs)
    assert batches == [32, 32]
    np.testing.assert_array_equal(values, obs[0][:, 0] + 0.5)
    np.testing.assert_array_equal(logits, obs[0] * 2)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many", ""])
def test_eval_forward_chunk_rejects_invalid_override(monkeypatch, value):
    import eval_mcts
    monkeypatch.setenv("EVAL_FORWARD_CHUNK", value)
    with pytest.raises(ValueError, match="EVAL_FORWARD_CHUNK must be a positive integer"):
        eval_mcts._configured_eval_forward_chunk()


def eval_record(settings=None):
    receipt = {"schema": "terra_v8_r2_reward_protocol_v1", "distance_sidecar_sha256": "b" * 64,
               "constants": {"potential_gamma": 0.9984}}
    if settings is not None:
        receipt["foundation_behavior"] = settings
    return dict(deterministic=True, reset_verification={"passed": True},
                summary={"integrity": {"passed": True}}, r2_protocol_receipt=receipt,
                manifest_sha256="same", horizon=450, seed=20260907, per_map=[])


def test_dashboard_requires_explicit_cost_comparison_and_keeps_other_guards():
    before = eval_record()
    after = eval_record(dict(executable_dig_observation=True, lateral_dig_cost=0.25,
                             base_travel_cost=0.005, base_turn_cost=0.02))
    with pytest.raises(ValueError, match="r2_protocol_receipt"):
        validate_pair(before, after)
    validate_pair(before, after, compare_foundation_rewards=True)
    for field, value in (("distance_sidecar_sha256", "a" * 64),
                         ("constants", {"potential_gamma": 1.0})):
        wrong = copy.deepcopy(after)
        wrong["r2_protocol_receipt"][field] = value
        with pytest.raises(ValueError, match="r2_protocol_receipt"):
            validate_pair(before, wrong, compare_foundation_rewards=True)
    wrong = copy.deepcopy(after)
    wrong["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="manifests"):
        validate_pair(before, wrong, compare_foundation_rewards=True)
