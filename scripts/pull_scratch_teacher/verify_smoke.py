"""Verify the saved update from the scratch / compatible-bulk-teacher smoke."""
import json
from pathlib import Path
import sys

import numpy as np

from train_mixed import _assert_finite_loss_info, _assert_finite_tree, _assert_transition_integrity
from utils.helpers import load_pkl_object, register_checkpoint_config_classes

register_checkpoint_config_classes()
checkpoint = load_pkl_object(sys.argv[1])
cfg = checkpoint["train_config"]
initial = json.loads((Path(sys.argv[1]).parent.parent / "scratch_initialization.json").read_text())
assert initial["initialization"] == "scratch"
assert initial["optimizer_step"] == 0
assert initial["actor_and_critic_distinct_from_teacher"]
assert int(checkpoint["next_update"]) == 1
assert int(np.asarray(checkpoint["train_state_step"]).reshape(())) == 2
assert cfg.actor_core == "gru" and cfg.actor_gru_hidden_dim == 64
assert cfg.warm_start_from is None and cfg.resume_from is None
assert cfg.precision_required_band_observation
assert cfg.precision_episode_fraction == 0.5
assert cfg.recurrent_teacher and cfg.teacher_bulk_compatibility
assert cfg.teacher_checkpoint.endswith("gru_rules_ft_c057_s20260930_update_110000.pkl")
assert cfg.kickstart_kl_coef == 1 and cfg.kickstart_value_coef == 0
assert bool(np.asarray(checkpoint["env_config"].pull_direction_alignment).reshape(()))
_assert_finite_loss_info(checkpoint["loss_info"], 0)
_assert_finite_tree(checkpoint["model"], "saved scratch model")
_assert_finite_tree(checkpoint["optimizer_state"], "saved scratch optimizer")
_assert_transition_integrity(checkpoint["transition_integrity"])
fraction = float(np.asarray(checkpoint["loss_info"]["kickstart/eligible_transition_fraction"]))
assert 0 < fraction <= 0.5, f"smoke must exercise the compatible bulk teacher: {fraction}"
teacher_kl = float(np.asarray(checkpoint["loss_info"]["kickstart/kl"]))
assert np.isfinite(teacher_kl) and teacher_kl > 0, f"scratch smoke must exercise teacher KL: {teacher_kl}"
for name, values in checkpoint["loss_info"].items():
    if name.startswith("diagnostics/") and name.endswith("finite_fraction"):
        assert np.all(np.asarray(values) == 1), name
report = dict(
    next_update=1, optimizer_steps=2, devices=cfg.num_devices,
    envs_per_device=cfg.num_envs_per_device, initialization="scratch",
    initial_actor_and_critic_distinct_from_teacher=True,
    teacher_eligible_transition_fraction=fraction,
    teacher_policy_kl=teacher_kl,
    finite_losses_rollout_model_optimizer=True, transition_integrity_passed=True,
    policy_quality_claim=False,
)
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
