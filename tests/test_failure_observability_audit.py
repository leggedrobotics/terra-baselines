"""Two load-bearing checks for the bounded diagnostic, without policy inference."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from scripts.analysis.terra_failure_audit import compare_episode_ages, select_cases
from terra.agent import Agent, AgentState
from terra.config import EnvConfig
from terra.state import State
from terra.env import TerraEnv


def physical_state():
    cfg = EnvConfig()._replace(max_steps_in_episode=450, tile_size=np.float32(4/7),
                              agent_types=(0,), action_types=(0,))
    cfg = cfg._replace(agent=cfg.agent._replace(width=7, height=11),
                       maps=cfg.maps._replace(edge_length_px=64))
    zero = AgentState(pos_base=jnp.zeros(2, dtype=jnp.int16),
                      **{name: jnp.zeros(1, dtype=jnp.int8) for name in
                         ('angle_base', 'angle_cabin', 'wheel_angle', 'loaded',
                          'agent_type', 'action_type', 'shovel_lifted')})
    agent = Agent(width=jnp.int32(7), height=jnp.int32(11),
                  agent_states=(zero._replace(pos_base=jnp.array([32,32],dtype=jnp.int16)),zero,zero,zero),
                  agent_active=jnp.array([1,0,0,0],dtype=jnp.int8),
                  num_agents=jnp.int32(1), current_agent=jnp.int32(0))
    target = np.zeros((64,64),dtype=np.int8)
    target[32,40:44] = -1
    target[32,20:24] = 1
    state = State.new(jax.random.PRNGKey(7),cfg,jnp.asarray(target),jnp.zeros_like(target),
                     -97*jnp.ones((4,8)),jnp.int32(0),-97*jnp.ones((64,3)),jnp.int32(0),
                     jnp.ones((64,64),dtype=bool),jnp.zeros_like(target),
                     distance_map_override=jnp.ones((64,64)), initial_agent=agent)
    return TerraEnv.wrap_state(state, update_reachability=False, executable_dig_observation=True)


class FailureObservabilityAuditTest(unittest.TestCase):
    def test_legacy_policy_aliases_age_until_time_input_is_enabled(self):
        config = SimpleNamespace(clip_action_maps=True, executable_dig_observation=True,
                                 carry_work_observation=True, trench_alignment_observation=True,
                                 relocation_distance_observation=True, admissible_dig_observation=True)
        history = jnp.array([7,6,4,2,0], dtype=jnp.int32)
        result = compare_episode_ages(physical_state(),config,history)
        assert result['identical_physics_history_and_other_state']
        assert not result['raw_observations_identical']
        assert result['model_inputs_identical']
        assert result['done'] == [False,False,False,True]
        config.time_observation_mode = "remaining"
        timed_result = compare_episode_ages(physical_state(),config,history)
        assert not timed_result['model_inputs_identical']
        config.time_observation_mode = "constant"
        assert compare_episode_ages(physical_state(),config,history)['model_inputs_identical']
        state = physical_state()
        for age, expected in ((0,1.0),(50,400/450),(440,10/450),(450,0.0),(500,0.0)):
            observation = TerraEnv._state_to_obs_dict(state._replace(env_steps=age))
            np.testing.assert_allclose(observation['remaining_time'],expected,atol=1e-7)


    def test_selection_covers_distinct_failed_cases_and_preserves_starts(self):
        # Use source-independent rows, preserving their complete identity fields.
        rows=[]
        cells=['fnd-slab-side1-obj','fnd-proc-ring3x','v7-fnd-bearing-walls-adjacent',
               'fnd-slab-apron-d12','trn-net3-side1-road','trn-net3-side1-road',
               'trn-net4-side1-road','trn-tee-side2-s','trn-tee-side2','trn-net4-side2-s',
               'fnd-slab-apron-c1p6','trn-net3-side1-road']
        for i,cell in enumerate(cells):
            rows.append(dict(slot_index=i+1,episode_id=f'episode-{i}',reset_seed=700+i,
                             source_id=f'source-{i}',success=False,primary_cell=cell,
                             family='foundation' if cell.startswith(('fnd','v7-fnd')) else 'trench',
                             dig_fraction=0 if i==4 else 1 if i>=10 else 0.5,
                             terminal_soil_fraction=0.4,longest_material_stall_steps=100+i))
        selected=select_cases({'per_map':rows})
        assert len(selected)==len({r['slot_index'] for r in selected})==12
        for row in selected:
            original=rows[row['slot_index']-1]
            assert all(row[k]==original[k] for k in ('episode_id','source_id','reset_seed'))


if __name__ == "__main__":
    unittest.main()
