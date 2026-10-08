"""Immediate native DO agreement and the saved teacher's observation semantics."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.agent import Agent, AgentState
from terra.config import EnvConfig
from terra.dig_direction import boundary_records_from_mask
from terra.env import TerraEnv
from terra.state import State
from terra.wrappers import LocalMapWrapper
from utils.task_teachers import (
    CANDIDATE_KEY, ELIGIBLE_KEY, bulk_teacher_do_compatible,
    bulk_teacher_observation_and_compatibility, legacy_teacher_state_view,
    recurrent_teacher_rollout_observation, reset_recurrent_teacher_hidden,
    validate_bulk_teacher_environment,
)


def fixture(rows=(4, 29), *, legacy=False, position=(16, 8), cabin=0):
    target = np.ones((32, 32), np.int8)
    target[rows[0]:rows[1], 13:31] = -1
    cfg = EnvConfig()
    cfg = cfg._replace(
        maps=cfg.maps._replace(edge_length_px=32, edge_length_m=32 * 40 / 70),
        tile_size=np.float32(40 / 70), agent_types=(0,), action_types=(0,),
        agent=cfg.agent._replace(width=7, height=11, dig_min_radius_m=4.,
                                dump_max_radius_m=6., dug_clearance_m=.57,
                                centre_chassis_on_base=True),
        pull_direction_alignment=not legacy, enforce_foundation_border_alignment=False,
        enforce_trench_dig_alignment=True, executable_dig_observation=True,
        dig_pull_min_length_m=2.5,
    )
    original_axes = jnp.asarray([[0, 1, -rows[0]], [1, 0, -30],
                                 [0, 1, -(rows[1] - 1)], [1, 0, -13]], jnp.float32)
    axes = original_axes if legacy else jnp.asarray(boundary_records_from_mask(
        target < 0, max_segments=16,
    )[0])
    machine = AgentState(
        pos_base=jnp.asarray(position, jnp.int16), angle_base=jnp.array([0], jnp.int8),
        angle_cabin=jnp.array([cabin], jnp.int8), wheel_angle=jnp.array([0], jnp.int8),
        loaded=jnp.array([0], jnp.int8), agent_type=jnp.array([0], jnp.int8),
        action_type=jnp.array([0], jnp.int8), shovel_lifted=jnp.array([0], jnp.int8),
    )
    agent = Agent(width=7, height=11, agent_states=(machine,) * 4,
                  agent_active=jnp.array([1, 0, 0, 0], bool), num_agents=1,
                  current_agent=jnp.int32(0))
    state = State.new(
        jax.random.PRNGKey(0), cfg, jnp.asarray(target), jnp.zeros((32, 32), jnp.int8),
        jnp.full((4, 8), -97., jnp.float32), jnp.int32(-1), axes, jnp.int32(4),
        jnp.asarray(target > 0), jnp.zeros((32, 32), jnp.int8),
        distance_map_override=jnp.zeros((32, 32), jnp.float32), initial_agent=agent,
    )
    return state, original_axes


def test_native_gate_accepts_equal_dig_and_rejects_two_allowed_different_digs():
    @jax.jit
    def compare(current, old):
        current_do = current._dig_eligibility(current._build_dig_dump_cone())
        legacy_do = old._dig_eligibility(old._build_dig_dump_cone())
        return bulk_teacher_do_compatible(current, old), current_do, legacy_do

    for rows, expected in (((4, 29), True), ((14, 19), False)):
        current, _ = fixture(rows)
        legacy, _ = fixture(rows, legacy=True)
        validate_bulk_teacher_environment(legacy.env_cfg, current.env_cfg, {})
        batched_cfg = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.asarray(x)[None], 2, 0), current.env_cfg)
        validate_bulk_teacher_environment(legacy.env_cfg, batched_cfg, {})
        compatible, new_do, old_do = compare(current, legacy)
        assert bool(new_do[3]) and bool(old_do[3]), "both DO actions must be admitted"
        assert bool(compatible) == expected
        assert np.array_equal(new_do[0], old_do[0]) == expected
        if not expected:
            assert int(new_do[1]) < int(old_do[1])
            loaded = current._set_current_agent_state(current._get_current_agent_state()._replace(
                loaded=jnp.array([7], jnp.int8),
            ))
            old_loaded = legacy._set_current_agent_state(loaded._get_current_agent_state())
            assert bool(compare(loaded, old_loaded)[0]), "unchanged loaded DO remains compatible"
    with pytest.raises(ValueError, match="dump_max_radius_m"):
        validate_bulk_teacher_environment(legacy.env_cfg, current.env_cfg._replace(
            agent=current.env_cfg.agent._replace(dump_max_radius_m=5.5),
        ), {})


def test_teacher_view_matches_independent_native_legacy_state_with_original_axes():
    current, original_axes = fixture(position=(16, 13), cabin=3)
    reference, _ = fixture(legacy=True, position=(16, 13), cabin=3)
    teacher_view = legacy_teacher_state_view(current, reference.env_cfg, original_axes)
    np.testing.assert_array_equal(teacher_view.world.foundation_border_axes, original_axes)
    np.testing.assert_array_equal(teacher_view.world.action_map.map, current.world.action_map.map)

    view = jax.jit(lambda s, axes: bulk_teacher_observation_and_compatibility(
        s, reference.env_cfg, axes, executable_dig_observation=True,
    ))(current, original_axes)[0]
    raw = jax.jit(lambda s: TerraEnv._state_to_obs_dict(LocalMapWrapper.wrap(
        s, executable_dig_observation=True,
    )))
    expected = raw(reference)
    student = raw(current)
    keys = [key for key in expected if key.startswith("local_map_") or key.startswith("fresh_trench_")]
    for key in keys:
        np.testing.assert_array_equal(view[key], expected[key], err_msg=key)
    assert np.any(np.asarray(view["local_map_border_workspace"]) != np.asarray(student["local_map_border_workspace"]))


def test_compatibility_mask_changes_kl_selection_without_skipping_teacher_carry():
    raw = {"signal": jnp.array([[100.], [200.], [300.]])}
    old_view = {"signal": jnp.array([[1.], [2.], [3.]])}
    hidden = jnp.array([[4.], [9.], [14.]])
    def apply(_, obs, carry, method):
        assert method == "actor_step"
        nxt = carry + obs
        return nxt, jnp.concatenate((nxt, -nxt), -1), nxt
    with mock.patch("utils.task_teachers.obs_to_model_input", side_effect=lambda obs, *_: obs["signal"]):
        cached, advanced = recurrent_teacher_rollout_observation(
            raw, jnp.zeros((3, 5), jnp.int32), hidden, apply, None, {},
            jnp.array([False, False, True]), jnp.array([7, 7, 7]), jnp.array([7]),
            teacher_observation=old_view, bulk_compatible=jnp.array([True, False, True]),
        )
    np.testing.assert_array_equal(advanced, [[5.], [11.], [17.]])
    np.testing.assert_array_equal(cached[CANDIDATE_KEY], [True, True, False])
    np.testing.assert_array_equal(cached[ELIGIBLE_KEY], [True, False, False])
    np.testing.assert_array_equal(cached["signal"], raw["signal"])
    np.testing.assert_array_equal(reset_recurrent_teacher_hidden(advanced, jnp.array([False, True, False])),
                                  [[5.], [0.], [17.]])
