"""Reload the one-update smoke checkpoint and check finite saved state."""
import json
from pathlib import Path
import sys

import numpy as np
from train_mixed import _assert_finite_loss_info, _assert_finite_tree, _assert_transition_integrity
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(sys.argv[1])
cfg = checkpoint["train_config"]
assert int(checkpoint["next_update"]) == 1
assert int(np.asarray(checkpoint["train_state_step"]).reshape(())) == 2
assert cfg.actor_core == "gru" and cfg.actor_gru_hidden_dim == 64
assert cfg.precision_required_band_observation
assert cfg.precision_episode_fraction == 0.5
assert cfg.recurrent_teacher
assert cfg.kickstart_value_coef == 0
assert cfg.warm_start_from.endswith("gru_rules_ft_c057_s20260930_update_110000.pkl")
assert cfg.resume_from is None
assert bool(np.asarray(checkpoint["env_config"].pull_direction_alignment).reshape(()))
_assert_finite_loss_info(checkpoint["loss_info"], 0)
np.testing.assert_array_equal(
    np.asarray(checkpoint["loss_info"]["kickstart/eligible_transition_fraction"]),
    0.5,
)
_assert_finite_tree(checkpoint["model"], "saved smoke model")
_assert_finite_tree(checkpoint["optimizer_state"], "saved smoke optimizer")
_assert_transition_integrity(checkpoint["transition_integrity"])
for name, values in checkpoint["loss_info"].items():
    if name.startswith("diagnostics/") and name.endswith("finite_fraction"):
        assert np.all(np.asarray(values) == 1), name
report = dict(next_update=1, optimizer_steps=2, devices=cfg.num_devices,
              envs_per_device=cfg.num_envs_per_device, precision_episode_fraction=0.5,
              finite_losses_rollout_model_optimizer=True,
              transition_integrity_passed=True, policy_quality_claim=False)
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
