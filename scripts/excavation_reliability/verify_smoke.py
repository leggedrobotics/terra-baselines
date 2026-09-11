"""Check native optimizer continuity and the full-bank 2x training contract."""
import argparse
import json

import jax
import numpy as np

from train_mixed import _assert_finite_loss_info, _assert_finite_tree
from utils.helpers import (
    checkpoint_foundation_behavior,
    load_pkl_object,
    register_checkpoint_config_classes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent")
    parser.add_argument("checkpoint")
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    register_checkpoint_config_classes()
    parent = load_pkl_object(args.parent)
    checkpoint = load_pkl_object(args.checkpoint)
    expected_update = int(parent["next_update"]) + args.updates
    expected_adam = int(np.asarray(parent["train_state_step"])) + args.updates * 64
    assert checkpoint["next_update"] == expected_update
    assert int(np.asarray(checkpoint["train_state_step"])) == expected_adam
    for field in ("model", "optimizer_state"):
        _assert_finite_tree(checkpoint[field], field)
    _assert_finite_loss_info(checkpoint["loss_info"], expected_update - 1)
    assert all(int(np.asarray(v)) == 0 for v in checkpoint["transition_integrity"].values())
    optimizer_counts = [int(np.asarray(x)) for x in jax.tree.leaves(checkpoint["optimizer_state"])
                        if np.shape(x) == () and np.asarray(x).dtype.kind in "iu"]
    assert optimizer_counts == [expected_adam], optimizer_counts
    assert jax.tree.structure(parent["model"]) == jax.tree.structure(checkpoint["model"])
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(parent["model"]), jax.tree.leaves(checkpoint["model"])))
    parameters = sum(np.size(x) for x in jax.tree.leaves(checkpoint["model"]))
    assert parameters == 2311701, parameters
    behavior = checkpoint_foundation_behavior(checkpoint)
    for key, value in {"lateral_dig_cost": .5, "base_travel_cost": .01,
                       "base_turn_cost": .04, "executable_dig_observation": True}.items():
        assert np.isclose(behavior[key], value), (key, behavior[key])
    cfg = checkpoint["train_config"]
    assert cfg.config_name == "trench_align_v2_generalist_gen"
    assert cfg.load_env_from_checkpoint and not cfg.finetune_task_bank
    assert not cfg.flat_minibatch_shuffle
    assert (cfg.num_devices, cfg.num_envs_per_device, cfg.num_steps,
            cfg.update_epochs, cfg.num_minibatches) == (1, 512, 32, 2, 32)
    assert cfg.cache_clear_interval == 0
    assert cfg.distance_sidecar_sha256 == "f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980"
    assert (cfg.ent_schedule_start, cfg.ent_schedule_end, cfg.ent_schedule_steps) == (.15, .02, 20000)
    print(json.dumps({"status": "PASS", "next_update": expected_update,
                      "adam_step": expected_adam, "parameters": parameters,
                      "new_transitions": args.updates * 16384,
                      "finite_model_optimizer_loss": True, "transition_integrity_zero": True,
                      "same_bank_native_resume": True, "behavior_2x": True}, indent=2))


if __name__ == "__main__":
    main()
