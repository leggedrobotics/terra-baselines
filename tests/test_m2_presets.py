"""The M2 presets must actually turn reward-v2 on, and prove it end to end.

REVIEW_V6 R-1 was a launch blocker of exactly this shape: reward-v2 was
mechanically complete in `terra` but INERT in every launchable preset, because
all three M1 presets ship `excavator_relocate_dumped_mult ==
excavator_relocate_dug_dirt_mult == 1.5`, which collapses the newly-reachable
discount branch back onto the full rate. The A/B row measured bit-identical to
reward-v1. R-6's fix is this file:

* the three M2 presets are loaded through `configs.training_configs.get_config`
  and asserted to have `dumped < dug` and a clean
  `terra.config.check_relocation_multipliers`;
* the three M1 presets are asserted to be PINNED at 1.5 / 1.5, because that is
  what Euler jobs 9046454/9046455/9046456 ran and the M1 readout is written
  against it — an edit on either side turns this file red;
* and the staging loop M1-B actually farmed is driven end to end through real
  `_handle_dig` / `_handle_dump` / `_handle_rewards_*` calls, with no manual
  flag patching, under each M2 preset's own multipliers. Under reward-v1 staging
  costs -2.32 against the direct loop; under reward-v2 it costs -49.56. The
  assertion is on the sign AND on the magnitude, so a preset that silently
  reverts to 1.5 / 1.5 cannot pass.

Run it against the screen checkout, not the venv's editable `terra` (REVIEW_V6
R-8): the main repo has no reward-v2 and the end-to-end test will fail with
`terra.__file__` in the message.

    PYTHONPATH=/home/lorenzo/moleworks/.worktrees/terra_v5m_screen_20260730:$PWD \\
        JAX_PLATFORMS=cpu python -m pytest tests/test_m2_presets.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import terra
from scipy import ndimage

from configs.training_configs import get_config
from terra.config import BatchConfig, EnvConfig, MapsDimsConfig, check_relocation_multipliers
from terra.env import TerraEnvBatch
from terra.state import State

M2_PRESETS = ("m2_wave_long", "m2_dose", "m2_dose_fast")
M1_PRESETS = ("curriculum_v5m_a_t0", "curriculum_v5m_b_uniform", "curriculum_v5m_c_waves")

REWARD_V2_DUMPED = 0.2
REWARD_V2_DUG = 1.5

SEED = 20260730
SHAPE = (64, 64)
# Measured on the reward-v2 worktree, geometry below (REVIEW_V6 section 3):
#   reward-v1 (1.5/1.5): direct +76.6159, staging +74.2993, delta -2.32
#   reward-v2 (0.2/1.5): direct +76.6159, staging +27.0513, delta -49.56
DIRECT_TOTAL = 76.6159
REWARD_V1_STAGING_PENALTY = -2.32
REWARD_V2_STAGING_PENALTY = -49.56


def _env_config(dumped_mult: float, dug_mult: float) -> EnvConfig:
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=SHAPE[0])
    )
    base = EnvConfig()
    batched = base._replace(
        agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
    )
    updated = batch_env.update_env_cfgs(batched)
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(
            edge_length_px=int(np.asarray(updated.maps.edge_length_px)[0])
        ),
        agent_types=(0,),  # single excavator: reward-v1 could not reach the discount
        action_types=(0,),
        foundation_dump_min_free_fraction=0.0,
        excavator_relocate_dumped_mult=dumped_mult,
        excavator_relocate_dug_dirt_mult=dug_mult,
    )


def _state(target, action, distance, cfg, *, angle_cabin=0) -> State:
    padding = np.zeros(SHAPE, dtype=np.int8)
    dumpability = np.ones(SHAPE, dtype=np.bool_)
    state = State.new(
        jax.random.PRNGKey(SEED),
        cfg,
        target,
        padding,
        -97.0 * np.ones((3, 3), np.float32),
        np.int32(-1),
        -97.0 * np.ones((SHAPE[0], 3), np.float32),
        np.int32(-1),
        dumpability,
        action,
        distance_map_override=distance,
    )
    current = state._get_current_agent_state()._replace(
        pos_base=jnp.array([32, 32], jnp.int16),
        angle_base=jnp.array([0], jnp.int8),
        angle_cabin=jnp.array([angle_cabin], jnp.int8),
        loaded=jnp.array([0], jnp.int8),
    )
    return state._set_current_agent_state(current)


def _cone_cells(cfg, angle_cabin):
    zeros = np.zeros(SHAPE, dtype=np.int8)
    ones = np.ones(SHAPE, dtype=np.float32)
    state = _state(zeros, zeros.copy(), ones, cfg, angle_cabin=angle_cabin)
    mask = np.asarray(state._build_dig_dump_cone()).reshape(SHAPE)
    return np.argwhere(mask)


def _maps(cfg):
    """dig zone in cone(0), designated dump in cone(6), cone(3) = legal bare ground."""
    target = np.zeros(SHAPE, dtype=np.int8)
    for y, x in _cone_cells(cfg, 0):
        target[y, x] = -1
    for y, x in _cone_cells(cfg, 6):
        target[y, x] = 1
    distance = ndimage.distance_transform_edt(target <= 0)
    return target, (distance / distance.max()).astype(np.float32)


def _turn(state, angle):
    return state._set_current_agent_state(
        state._get_current_agent_state()._replace(angle_cabin=jnp.array([angle], jnp.int8))
    )


def _dig(state):
    next_state = state._handle_dig()
    return next_state, float(state._handle_rewards_dig(next_state, None))


def _dump(state):
    next_state = state._handle_dump()
    return next_state, float(state._handle_rewards_dump(next_state, None))


def _direct_total(cfg, target, distance) -> float:
    state = _state(target, np.zeros(SHAPE, np.int8), distance, cfg)
    state, dig_reward = _dig(state)
    state = _turn(state, 6)
    _, dump_reward = _dump(state)
    return dig_reward + dump_reward


def _staging_total(cfg, target, distance) -> float:
    """dig cone(0) -> stage on legal cone(3) -> RE-DIG cone(3) -> dump cone(6)."""
    state = _state(target, np.zeros(SHAPE, np.int8), distance, cfg)
    total = 0.0
    state, reward = _dig(state)
    total += reward
    state = _turn(state, 3)
    state, reward = _dump(state)
    total += reward
    state, reward = _dig(state)
    total += reward
    assert bool(state.agent.moving_dumped_dirt), (
        "re-digging a staged pile did not set moving_dumped_dirt, so reward-v2 is "
        f"not in force. terra = {terra.__file__}"
    )
    state = _turn(state, 6)
    _, reward = _dump(state)
    return total + reward


@pytest.fixture(scope="module")
def geometry():
    cfg = _env_config(REWARD_V2_DUMPED, REWARD_V2_DUG)
    return _maps(cfg)


@pytest.mark.parametrize("name", M2_PRESETS)
def test_m2_preset_enables_the_redig_discount(name):
    preset = get_config(name)
    multipliers = preset.reward_multipliers
    assert multipliers.excavator_relocate_dumped_mult == REWARD_V2_DUMPED
    assert multipliers.excavator_relocate_dug_dirt_mult == REWARD_V2_DUG
    assert (
        multipliers.excavator_relocate_dumped_mult
        < multipliers.excavator_relocate_dug_dirt_mult
    )
    env_cfg = _env_config(
        multipliers.excavator_relocate_dumped_mult,
        multipliers.excavator_relocate_dug_dirt_mult,
    )
    assert check_relocation_multipliers(env_cfg) == ""


@pytest.mark.parametrize("name", M1_PRESETS)
def test_m1_preset_stays_pinned_to_reward_v1(name):
    """M1 ran at 1.5 / 1.5. Changing that would silently rewrite the M1 record."""
    multipliers = get_config(name).reward_multipliers
    assert multipliers.excavator_relocate_dumped_mult == 1.5
    assert multipliers.excavator_relocate_dug_dirt_mult == 1.5
    env_cfg = _env_config(1.5, 1.5)
    assert "reward-v2 WARNING" in check_relocation_multipliers(env_cfg)


def test_m2_promotion_rules_and_level_paths():
    wave = get_config("m2_wave_long")
    dose = get_config("m2_dose")
    fast = get_config("m2_dose_fast")

    assert [level.maps_path for level in wave.maps] == [
        "train/L0",
        "train/L1",
        "train/L2",
        "train/L3",
    ]
    for arm in (dose, fast):
        assert [level.maps_path for level in arm.maps] == [
            "train/L0p",
            "train/L1p",
            "train/L2p",
            "train/L3p",
        ]

    # 3-consecutive-exact on wave and dose; 2-consecutive-exact on dose-fast.
    # The spec's "OR rolling graded >= 0.8 over the last 3 episodes" disjunct is
    # DESCOPED (REVIEW_V6 M2-1): CurriculumConfig carries no graded history.
    assert wave.curriculum.increase_level_threshold == 3
    assert dose.curriculum.increase_level_threshold == 3
    assert fast.curriculum.increase_level_threshold == 2
    for arm in (wave, dose, fast):
        assert arm.curriculum.decrease_level_threshold == 3
        assert arm.curriculum.last_level_type == "none"
        assert all(level.max_steps_in_episode == 450 for level in arm.maps)
        assert arm.agent_types == (0,)


@pytest.mark.parametrize("name", M2_PRESETS)
def test_staging_pays_strictly_less_than_direct_under_each_m2_preset(name, geometry):
    """The end-to-end test R-6 found missing: staging vs direct, on reward."""
    target, distance = geometry
    multipliers = get_config(name).reward_multipliers
    cfg = _env_config(
        multipliers.excavator_relocate_dumped_mult,
        multipliers.excavator_relocate_dug_dirt_mult,
    )
    direct = _direct_total(cfg, target, distance)
    staging = _staging_total(cfg, target, distance)
    delta = staging - direct

    # The INTENDED loop is bit-for-bit what reward-v1 paid: reward-v2 only
    # changes which branch a re-dig reaches, never the fresh-dig rate.
    assert direct == pytest.approx(DIRECT_TOTAL, abs=1e-3)
    assert staging < direct, f"{name}: staging {staging} >= direct {direct}"
    # -2.32 is the reward-v1 number. Anything near it means the discount is off.
    assert delta < 0.5 * REWARD_V2_STAGING_PENALTY, (
        f"{name}: staging - direct = {delta:.4f}, which is the reward-v1 regime "
        f"({REWARD_V1_STAGING_PENALTY}), not reward-v2 "
        f"({REWARD_V2_STAGING_PENALTY}). terra = {terra.__file__}"
    )
