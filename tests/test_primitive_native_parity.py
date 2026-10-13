"""The two interfaces charge the same physical cabin swing and excavation."""
from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.structured_actions import StructuredAction, StructuredClock, structured_transition
from terra.tests import test_structured_actions as fixtures


@partial(jax.jit, static_argnums=(2,))
def compare_aim_then_work(state, heading, config):
    state = state._replace(env_cfg=config)
    direct = structured_transition(state, StructuredAction(jnp.int32(6), heading=heading),
                                   precision_edge_weight=.5)
    aimed = structured_transition(state, StructuredAction(jnp.int32(5)), precision_edge_weight=.5)
    clock = StructuredClock(aimed.info['time_visit_open'], aimed.info['time_moved'])
    primitive = structured_transition(aimed.state, StructuredAction(jnp.int32(6), heading=jnp.int32(-1)),
                                      clock=clock, precision_edge_weight=.5)
    return direct, aimed, primitive


class PrimitiveNativeParityTests(unittest.TestCase):
    def test_explicit_cabin_and_current_heading_work_match_direct_aimed_work(self):
        fixtures.StructuredActionTest.setUpClass()
        state = fixtures.StructuredActionTest.pose(fixtures.StructuredActionTest.state, cabin=11)
        config = state.env_cfg._replace(precision_safety=True)
        direct, aimed, primitive = compare_aim_then_work(state._replace(env_cfg=None), jnp.int32(0), config)
        self.assertEqual(int(aimed.state._get_current_agent_state().angle_cabin[0]), 0)
        self.assertGreater(int(primitive.state._get_current_agent_state().loaded[0]), 0)
        np.testing.assert_array_equal(direct.state.world.action_map.map, primitive.state.world.action_map.map)
        np.testing.assert_array_equal(direct.state._get_current_agent_state().loaded,
                                      primitive.state._get_current_agent_state().loaded)
        np.testing.assert_allclose(direct.duration_s, aimed.duration_s + primitive.duration_s, rtol=1e-6)
        np.testing.assert_allclose(direct.reward, aimed.reward + primitive.reward, atol=1e-6)
        self.assertEqual(int(direct.state.env_steps), 1)
        self.assertEqual(int(primitive.state.env_steps), 2)
        for transition in (direct, aimed, primitive):
            self.assertEqual(float(transition.info['transition_mass_residual']), 0.)


if __name__ == '__main__':
    unittest.main()
