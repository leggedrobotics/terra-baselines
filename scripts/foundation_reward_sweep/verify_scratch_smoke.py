"""Verify two native updates began at zero with the intended scratch treatment."""
import argparse
import json

import jax
import numpy as np

from train_mixed import _assert_finite_loss_info, _assert_finite_tree
from utils.helpers import checkpoint_foundation_behavior, load_pkl_object, register_checkpoint_config_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first_checkpoint")
    parser.add_argument("second_checkpoint")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cost-multiplier", type=int, choices=[0, 2], required=True)
    args = parser.parse_args()
    register_checkpoint_config_classes()
    checkpoints = [load_pkl_object(path) for path in (args.first_checkpoint, args.second_checkpoint)]
    for update, checkpoint in enumerate(checkpoints, start=1):
        assert checkpoint["next_update"] == update
        assert int(np.asarray(checkpoint["train_state_step"])) == update * 64
        for field in ("model", "optimizer_state"):
            _assert_finite_tree(checkpoint[field], field)
        _assert_finite_loss_info(checkpoint["loss_info"], update - 1)
        assert all(int(np.asarray(value)) == 0 for value in checkpoint["transition_integrity"].values())
        counts = [int(np.asarray(value)) for value in jax.tree.leaves(checkpoint["optimizer_state"])
                  if np.shape(value) == () and np.asarray(value).dtype.kind in "iu"]
        assert counts == [update * 64], counts
        cfg = checkpoint["train_config"]
        assert cfg.config_name == "foundation_reward_sweep" and cfg.seed == args.seed
        assert cfg.resume_from is None and cfg.warm_start_from is None and cfg.resume_update is None
        assert not cfg.finetune_task_bank and not cfg.finetune_foundation_behavior
        assert not cfg.flat_minibatch_shuffle
        assert (cfg.num_devices, cfg.num_envs_per_device, cfg.num_steps,
                cfg.update_epochs, cfg.num_minibatches) == (1, 512, 32, 2, 32)
        assert cfg.cache_clear_interval == 0
        assert cfg.distance_sidecar_sha256 == "6b2675998403ed2d6125d955fca446404fbdf260e0a0c2cf7b9864cbdd1fb2bf"
        assert (cfg.ent_schedule_start, cfg.ent_schedule_end, cfg.ent_schedule_steps) == (.15, .02, 20000)
        behavior = checkpoint_foundation_behavior(checkpoint)
        for key, value in {"executable_dig_observation": True,
                           "lateral_dig_cost": .25 * args.cost_multiplier,
                           "base_travel_cost": .005 * args.cost_multiplier,
                           "base_turn_cost": .02 * args.cost_multiplier}.items():
            assert np.isclose(behavior[key], value), (key, behavior[key])
    before, after = [checkpoint["model"] for checkpoint in checkpoints]
    assert jax.tree.structure(before) == jax.tree.structure(after)
    assert sum(np.size(value) for value in jax.tree.leaves(after)) == 2311701
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)))
    print(json.dumps({"status": "PASS", "initialization": "scratch", "seed": args.seed,
                      "cost_multiplier": args.cost_multiplier, "next_update": 2,
                      "adam_step": 128, "new_transitions": 32768,
                      "finite_model_optimizer_loss": True, "transition_integrity_zero": True}, indent=2))


if __name__ == "__main__":
    main()
