"""Synthetic gate boundaries; these fixtures are never campaign evidence."""
import copy
import hashlib
from pathlib import Path
import shlex
import subprocess

import pytest

from scripts.foundation_reward_sweep.finetune_after_completion import (
    COST_KEYS, FULL_COSTS, EVALUATION_PANELS, completion_gate, launch_environment,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/foundation_reward_sweep/train.sh"


def report(family, successes, update, fraction=0):
    count, total = (64, 64) if family == "foundation" else (224, 608)
    behavior = dict(zip(COST_KEYS, (v * fraction for v in FULL_COSTS)))
    behavior["executable_dig_observation"] = True
    return {
        "deterministic": True, "horizon": 450, "completion_contract": "exact_visible_dump_v1",
        **EVALUATION_PANELS[family],
        "stratum": "all", "schema": "terra_fixed_bank_eval_v4", "policy_mode": "deterministic",
        "exact_manifest_enumeration": True,
        "accepted_bank": None, "checkpoint_update": update, "checkpoint_sha256": f"synthetic-{update}",
        "reset_verification": {"passed": True, "env_steps_min": 0, "env_steps_max": 0, "slots": total},
        "summary": {"integrity": {"passed": True, "unavailable": 0, "failure_count": 0}},
        "treatment_fingerprint": {"contract": {
            "foundation_behavior": behavior,
            "run": {"name": f"synthetic-{fraction}", "seed": 20260909,
                    "config_name": "foundation_reward_sweep" if family == "foundation"
                                   else "trench_align_v2_specialist_spec"},
            "reward_action": {"distance_sidecar_sha256": "a" * 64},
            "ppo": {"clip_eps": 0.2, "ent_schedule_end": 0.02, "ent_schedule_start": 0.15,
                    "ent_schedule_steps": 20000, "flat_minibatch_shuffle": False, "gae_lambda": 0.95,
                    "gamma": 0.9984, "lr": 0.0003, "max_grad_norm": 0.5, "num_devices": 1,
                    "num_envs_per_device": 512, "num_minibatches": 32, "num_steps": 32,
                    "update_epochs": 2, "use_value_clip": False, "vf_coef": 2.0},
        }},
        "r2_protocol_receipt": {"foundation_behavior": copy.deepcopy(behavior)},
        "per_map": [{"episode_id": str(i), "map_id": str(i), "source_id": str(i),
                     "scenario_id": str(i), "reset_seed": i, "slot_index": i,
                     "family": family if i < count else "foundation", "primary_cell": "synthetic",
                     "success": i < successes, "integrity_failure": False, "integrity_unavailable": False}
                    for i in range(total)],
    }


@pytest.mark.parametrize("family,required", [("foundation", 58), ("trench", 202)])
def test_exact_completion_boundaries_and_full_panel_integrity(family, required):
    previous, latest = report(family, required, 10000), report(family, required, 12500)
    result = completion_gate(family, previous, latest, latest)
    assert result["ready"] and result["required_successes"] == required
    assert result["next_costs"] == dict(lateral_dig_cost=.125, base_travel_cost=.0025, base_turn_cost=.01)
    previous["per_map"][required - 1]["success"] = False
    assert not completion_gate(family, previous, latest, latest)["ready"]
    # For trenches, the final row is outside the reported trench family.
    previous["per_map"][-1]["integrity_failure"] = True
    with pytest.raises(ValueError, match="episode result"):
        completion_gate(family, previous, latest, latest)


def test_gate_rejects_incomparable_or_repeated_evidence():
    previous, latest = report("foundation", 60, 10000), report("foundation", 60, 12500)
    for field, value in (("deterministic", False), ("horizon", 900), ("split", "test"),
                         ("seed", 10), ("checkpoint_update", 12500),
                         ("checkpoint_sha256", latest["checkpoint_sha256"])):
        changed = copy.deepcopy(previous)
        changed[field] = value
        with pytest.raises(ValueError):
            completion_gate("foundation", changed, latest, latest)
    changed = copy.deepcopy(previous)
    changed["per_map"][1]["episode_id"] = changed["per_map"][0]["episode_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        completion_gate("foundation", changed, latest, latest)
    with pytest.raises(ValueError, match="same cost stage"):
        completion_gate("foundation", report("foundation", 60, 10000, .25), latest, latest)


def test_stage_ramp_retains_a_fixed_zero_cost_reference():
    reference = report("foundation", 64, 7500)
    for current, following in ((.25, .5), (.5, 1.0)):
        previous = report("foundation", 63, 10000, current)
        latest = report("foundation", 63, 12500, current)
        parent_stage = {"family": "foundation", "reference_sha256": reference["checkpoint_sha256"],
                        "reference_successes": 64,
                        "next_fraction_of_combined_2x": current, "parent_update": 7500,
                        "run_name": latest["treatment_fingerprint"]["contract"]["run"]["name"]}
        result = completion_gate("foundation", previous, latest, reference, parent_stage)
        assert result["ready"] and result["required_successes"] == 63
        assert result["next_fraction_of_combined_2x"] == following
        latest["per_map"][62]["success"] = False
        # Still above 90%, but the loss from the frozen reference exceeds 3 pp.
        assert not completion_gate("foundation", previous, latest, reference, parent_stage)["ready"]
        with pytest.raises(ValueError, match="original zero-cost reference"):
            completion_gate("foundation", previous, latest, report("foundation", 58, 7000), parent_stage)
        with pytest.raises(ValueError, match="penalty_stage.json"):
            completion_gate("foundation", previous, latest, reference)
    with pytest.raises(ValueError, match="final cost stage"):
        completion_gate("foundation", report("foundation", 64, 10000, 1),
                        report("foundation", 64, 12500, 1), reference)


def test_native_launch_binds_parent_and_preserves_bank_optimizer_and_schedule_flags(tmp_path, monkeypatch):
    previous, latest = report("foundation", 60, 10000), report("foundation", 60, 12500)
    parent = tmp_path / "synthetic-parent.pkl"
    parent.write_bytes(b"unit-test data only; never load as a checkpoint")
    latest["checkpoint_sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
    decision = completion_gate("foundation", previous, latest, latest)
    monkeypatch.setenv("DATASET_PATH", str(tmp_path / "bank"))
    monkeypatch.setenv("WANDB_RUN_ID", "unrelated-parent-history")
    monkeypatch.setenv("WANDB_DIR", "/unrelated/wandb")
    env = launch_environment(decision, latest, parent, tmp_path / "new-stage", tmp_path / "bank")
    assert env["START_UPDATE"] == "12500" and env["TARGET_UPDATE"] == "17500"
    assert env["SEED"] == "20260909" and "WANDB_RUN_ID" not in env
    assert env["WANDB_DIR"] == str(tmp_path / "new-stage/wandb")
    args = shlex.split(subprocess.check_output(["bash", str(SCRIPT), "--print-args"], env=env, text=True))
    assert "--finetune_foundation_behavior" in args and "--load_env_from_checkpoint" in args
    assert "--finetune_task_bank" not in args and "--resume_update" not in args and "--warm_start_from" not in args
    assert args[args.index("--total_timesteps") + 1] == str(17500 * 16384)
    assert args[args.index("--resume_from") + 1] == str(parent)
    assert "--executable_dig_observation" in args
    assert not (tmp_path / "new-stage").exists()
    parent.write_bytes(b"different parent")
    with pytest.raises(ValueError, match="checkpoint differs"):
        launch_environment(decision, latest, parent, tmp_path / "new-stage", tmp_path / "bank")
    env.update(INITIALIZATION="scratch", START_UPDATE="0")
    env.pop("RESUME_FROM")
    failure = subprocess.run(["bash", str(SCRIPT), "--print-args"], env=env, capture_output=True, text=True)
    assert failure.returncode == 2 and "BEHAVIOR_FINETUNE requires" in failure.stderr
