"""Native resume must preserve effective rewards, target costs and PPO clocks."""
import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.config import EnvConfig
from train_mixed import _r2_protocol_receipt, _validate_r2_resume_checkpoint, _validate_foundation_behavior_config
from utils.behavior_cost_ramp import COST_KEYS, ramp_costs, restore_behavior_cost_ramp
from utils.helpers import checkpoint_evaluation_config, checkpoint_foundation_behavior, overlay_foundation_behavior


def config(**changes):
    return SimpleNamespace(**(dict(
        reward_stage="reward_v2", gamma=0.9984,
        distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        distance_sidecar_sha256="a" * 64, update_epochs=2, num_minibatches=32,
        admissible_dig_observation=True, executable_dig_observation=True,
        lateral_dig_cost=0.125, base_travel_cost=0.0025, base_turn_cost=0.01,
        finetune_foundation_behavior=True, resume_from="parent.pkl", warm_start_from=None,
        resume_update=None, load_env_from_checkpoint=True, behavior_cost_ramp_updates=2500,
    ) | changes))


def parent():
    cfg = config(**{key: 0.0 for key in COST_KEYS}, behavior_cost_ramp_updates=0)
    return checkpoint(cfg, None, 10000)


def checkpoint(cfg, state, update):
    values = {key: getattr(cfg, key) for key in COST_KEYS} if state is None else ramp_costs(state, update)
    cp = dict(train_config=cfg, next_update=update, train_state_step=np.array(update * 64),
              optimizer_state={"count": np.array(update * 64)},
              env_config=overlay_foundation_behavior(EnvConfig(), {**values, "executable_dig_observation": True}),
              r2_protocol_receipt=_r2_protocol_receipt(cfg))
    if state is not None:
        cp["behavior_cost_ramp_state"] = copy.deepcopy(state)
    return cp


def new_ramp():
    cp = parent()
    return restore_behavior_cost_ramp(config(), cp, "resume", 10000, checkpoint_foundation_behavior(cp))


def test_native_mid_ramp_resume_keeps_the_absolute_ramp_and_adam_clock():
    state = new_ramp()
    cfg = config(finetune_foundation_behavior=False)
    cp = checkpoint(cfg, state, 11250)
    optimizer = cp["optimizer_state"]
    resume = config(behavior_cost_ramp_updates=0, finetune_foundation_behavior=False)
    _validate_r2_resume_checkpoint(cp, _r2_protocol_receipt(resume), resume)
    restored = restore_behavior_cost_ramp(resume, cp, "resume", 11250, checkpoint_foundation_behavior(cp))
    assert restored == state
    assert cp["optimizer_state"] is optimizer
    assert int(optimizer["count"]) == 720000
    for update, fraction in ((10000, 0), (10001, 1 / 2500), (11250, .5), (12500, 1), (15000, 1)):
        for key, value in ramp_costs(restored, update).items():
            assert value == pytest.approx(getattr(cfg, key) * fraction)


def test_evaluation_uses_effective_costs_without_rewriting_declared_target():
    cfg = config()
    cp = checkpoint(cfg, new_ramp(), 11250)
    evaluated = checkpoint_evaluation_config(cp)
    assert evaluated.lateral_dig_cost == pytest.approx(.0625)
    assert cp["train_config"].lateral_dig_cost == .125
    assert cp["r2_protocol_receipt"]["foundation_behavior"]["lateral_dig_cost"] == .125
    for mutate in (
        lambda c: c["env_config"]._replace(lateral_dig_cost=jnp.float32(.125)),
        lambda c: c["env_config"]._replace(base_travel_cost=jnp.float32(0.0)),
    ):
        bad = copy.deepcopy(cp)
        bad["env_config"] = mutate(bad)
        with pytest.raises(ValueError, match="mismatch"):
            checkpoint_foundation_behavior(bad)


def test_missing_or_corrupt_ramp_cannot_load_as_a_fixed_cost_checkpoint():
    for modify in (
        lambda cp: cp.pop("behavior_cost_ramp_state"),
        lambda cp: cp["behavior_cost_ramp_state"].update(duration_updates=0),
        lambda cp: cp["behavior_cost_ramp_state"].update(start_update=20000),
        lambda cp: cp["behavior_cost_ramp_state"]["target_costs"].update(lateral_dig_cost=.25),
        lambda cp: cp["behavior_cost_ramp_state"]["start_costs"].update(lateral_dig_cost=float("nan")),
        lambda cp: cp.pop("env_config"),
    ):
        cp = checkpoint(config(), new_ramp(), 11250)
        modify(cp)
        with pytest.raises(ValueError):
            checkpoint_foundation_behavior(cp)


def test_ordinary_resume_cannot_freeze_at_the_intermediate_cost_or_change_duration():
    cp = checkpoint(config(), new_ramp(), 11250)
    frozen = config(**ramp_costs(new_ramp(), 11250), finetune_foundation_behavior=False)
    with pytest.raises(ValueError, match="protocol receipt"):
        _validate_r2_resume_checkpoint(cp, _r2_protocol_receipt(frozen), frozen)
    with pytest.raises(ValueError, match="duration changed"):
        restore_behavior_cost_ramp(config(behavior_cost_ramp_updates=5000), cp, "resume", 11250,
                                  checkpoint_foundation_behavior(cp))
    with pytest.raises(ValueError, match="unfinished"):
        restore_behavior_cost_ramp(config(lateral_dig_cost=.25), cp, "resume", 11250,
                                  checkpoint_foundation_behavior(cp))


def test_next_stage_starts_at_the_accepted_plateau_cost():
    cp = checkpoint(config(), new_ramp(), 15000)
    next_config = config(**{key: 2 * getattr(config(), key) for key in COST_KEYS})
    state = restore_behavior_cost_ramp(next_config, cp, "resume", 15000, checkpoint_foundation_behavior(cp))
    assert state["start_update"] == 15000
    assert state["start_costs"]["lateral_dig_cost"] == .125
    assert ramp_costs(state, 17500)["lateral_dig_cost"] == .25


def test_parameters_only_warm_start_keeps_the_fresh_objective_contract():
    cp = checkpoint(config(), new_ramp(), 11250)
    fresh = config(behavior_cost_ramp_updates=0, warm_start_from="parent.pkl", resume_from=None)
    assert restore_behavior_cost_ramp(fresh, cp, "warm_start", 0, None) is None
    with pytest.raises(ValueError, match="native"):
        restore_behavior_cost_ramp(config(), cp, "warm_start", 0, None)


def test_resuming_an_existing_ramp_cannot_change_digging_observations():
    cp = checkpoint(config(), new_ramp(), 11250)
    with pytest.raises(ValueError, match="observations"):
        restore_behavior_cost_ramp(config(executable_dig_observation=False), cp, "resume", 11250,
                                  checkpoint_foundation_behavior(cp))


def test_new_ramp_requires_explicit_finetune_and_unchanged_observations_bank_and_clock():
    cp = parent()
    for changes in (
        {"finetune_foundation_behavior": False}, {"executable_dig_observation": False},
        {"finetune_task_bank": True}, {"load_env_from_checkpoint": False},
        {"resume_update": 10000},
    ):
        with pytest.raises(ValueError):
            restore_behavior_cost_ramp(config(**changes), cp, "resume", 10000, checkpoint_foundation_behavior(cp))
    for duration in (-1, 1.5, True):
        with pytest.raises(ValueError):
            _validate_foundation_behavior_config(config(behavior_cost_ramp_updates=duration))


def test_reward_arrays_change_without_retracing_or_changing_other_environment_fields():
    cfg = EnvConfig()._replace(
        lateral_dig_cost=jnp.zeros((2, 3), dtype=jnp.float32),
        base_travel_cost=jnp.zeros((2, 3), dtype=jnp.float32),
        base_turn_cost=jnp.zeros((2, 3), dtype=jnp.float32),
    )
    traces = []
    @jax.jit
    def reward(env_cfg):
        traces.append(1)
        return -(env_cfg.lateral_dig_cost + 2 * env_cfg.base_travel_cost + env_cfg.base_turn_cost)
    state = new_ramp()
    for update in (10000, 10001, 11250, 12500, 15000):
        values = ramp_costs(state, update)
        overlaid = overlay_foundation_behavior(cfg, {**values, "executable_dig_observation": False})
        result = reward(overlaid)
        np.testing.assert_allclose(result, -(values[COST_KEYS[0]] + 2 * values[COST_KEYS[1]] + values[COST_KEYS[2]]), rtol=1e-6)
        for key in cfg._fields:
            if key not in (*COST_KEYS, "executable_dig_observation"):
                assert getattr(overlaid, key) is getattr(cfg, key)
    assert len(traces) == 1
