"""Reload the courtyard fine-tune's one-update smoke checkpoint and check its settings and finite state.

verify_smoke.py CHECKPOINT OUT_JSON. As scripts/pull_direction_training/verify_smoke.py, for the courtyard recipe:
every lane a precision lane, no teacher, warm start from GRU110000, pull cone, perpendicular precision pulls.
"""
import json
from pathlib import Path
import sys

import numpy as np
from train_mixed import _assert_finite_loss_info, _assert_finite_tree, _assert_transition_integrity
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(sys.argv[1])
cfg = checkpoint["train_config"]
env = checkpoint["env_config"]


def scalar(value):
    return np.asarray(value).reshape(-1)[0]


assert int(checkpoint["next_update"]) == 1
assert int(np.asarray(checkpoint["train_state_step"]).reshape(())) == 2
assert cfg.actor_core == "gru" and cfg.actor_gru_hidden_dim == 64
assert cfg.precision_required_band_observation
assert cfg.precision_episode_fraction == 1.0
assert not cfg.recurrent_teacher
assert cfg.warm_start_from.endswith(".pkl")
assert cfg.resume_from is None
assert bool(scalar(env.pull_direction_alignment))
assert bool(np.all(np.asarray(env.enforce_foundation_border_alignment))), "every lane must be a precision lane"
assert bool(scalar(env.edge_pull_perpendicular_ok))
assert abs(float(scalar(env.pull_half_angle_rad)) - np.pi / 6) < 1e-6
assert bool(scalar(env.tracked_move_keeps_turn))
assert not bool(scalar(env.native_dump_observation))
assert bool(scalar(env.precision_safety))
assert abs(float(scalar(env.dig_pull_min_length_m)) - 2.5) < 1e-6
_assert_finite_loss_info(checkpoint["loss_info"], 0)
_assert_finite_tree(checkpoint["model"], "saved smoke model")
_assert_finite_tree(checkpoint["optimizer_state"], "saved smoke optimizer")
_assert_transition_integrity(checkpoint["transition_integrity"])
for name, values in checkpoint["loss_info"].items():
    if name.startswith("diagnostics/") and name.endswith("finite_fraction"):
        assert np.all(np.asarray(values) == 1), name
report = dict(next_update=1, optimizer_steps=2, devices=cfg.num_devices, envs_per_device=cfg.num_envs_per_device,
              precision_episode_fraction=1.0, pull_cone_rad=float(scalar(env.pull_half_angle_rad)),
              edge_pull_perpendicular_ok=True, precision_safety=True,
              pull_trench_gate=bool(scalar(getattr(env, "pull_trench_gate", False))),
              trench_dig_max_offset_m=float(scalar(env.trench_dig_max_offset_m)),
              warm_start_from=cfg.warm_start_from, finite_losses_rollout_model_optimizer=True,
              transition_integrity_passed=True, policy_quality_claim=False)
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
