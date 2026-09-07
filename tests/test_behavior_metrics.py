import unittest
from types import SimpleNamespace as NS

import numpy as np

from utils.behavior_metrics import EpisodeBehaviorMetrics


def configuration(tile_size=(1.0,)):
    return NS(
        tile_size=np.array(tile_size),
        agent=NS(angles_base=np.array([12]), angles_cabin=np.array([12])),
    )


def timestep(
    *, positions=(((0, 0),),), base=None, cabin=None, terrain=None,
    target=None, acting=None, types=None, active=None, done=None,
):
    positions = np.asarray(positions)
    count, slots = positions.shape[:2]
    base = np.zeros((count, slots)) if base is None else np.asarray(base)
    cabin = np.zeros((count, slots)) if cabin is None else np.asarray(cabin)
    types = np.zeros((count, slots)) if types is None else np.asarray(types)
    active = np.ones((count, slots)) if active is None else np.asarray(active)
    terrain = np.zeros((count, 2, 3)) if terrain is None else np.asarray(terrain)
    target = -np.ones_like(terrain) if target is None else np.asarray(target)
    return NS(
        done=np.zeros(count, dtype=bool) if done is None else np.asarray(done),
        state=NS(
            agent=NS(
                agent_states=tuple(
                    NS(
                        pos_base=positions[:, slot],
                        angle_base=base[:, slot, None],
                        angle_cabin=cabin[:, slot, None],
                        agent_type=types[:, slot, None],
                    )
                    for slot in range(slots)
                ),
                agent_active=active,
                current_agent=np.zeros(count, dtype=int) if acting is None else acting,
            ),
            world=NS(action_map=NS(map=terrain), target_map=NS(map=target)),
        ),
    )


class BehaviorMetricsTest(unittest.TestCase):
    def test_travel_includes_first_and_terminal_move_with_per_map_scale(self):
        tracker = EpisodeBehaviorMetrics(
            timestep(positions=(((0, 0),), ((0, 0),))),
            configuration((0.5, 1.0)), preserve_terminal_states=True,
        )
        tracker.update(timestep(positions=(((3, 4),), ((3, 4),))), [True, True])
        tracker.update(
            timestep(positions=(((5, 4),), ((3, 6),)), done=[True, False]),
            [True, True],
        )
        tracker.update(
            timestep(positions=(((99, 99),), ((3, 7),)), done=[True, True]),
            [False, True],
        )
        result = tracker.result()
        np.testing.assert_allclose(result["base_travel_m"], [3.5, 8.0])
        np.testing.assert_array_equal(result["base_reposition_count"], [2, 3])

    def test_agent_switch_never_becomes_travel_between_different_machines(self):
        positions = (((0, 0), (100, 0)),)
        tracker = EpisodeBehaviorMetrics(
            timestep(positions=positions, acting=[0]), configuration(),
            preserve_terminal_states=True,
        )
        tracker.update(timestep(positions=positions, acting=[1]), [True])
        tracker.update(
            timestep(positions=(((0, 0), (102, 0)),), acting=[0]), [True]
        )
        self.assertEqual(tracker.result()["base_travel_m"][0], 2.0)

    def test_workspace_stances_keep_cabin_swings_and_count_revisits(self):
        tracker = EpisodeBehaviorMetrics(
            timestep(), configuration(), preserve_terminal_states=True
        )
        terrain = np.zeros((1, 2, 3))
        terrain[0, 0, :2] = -1
        tracker.update(timestep(terrain=terrain), [True])
        # Swing, then dig sideways from the same base stance.
        tracker.update(timestep(terrain=terrain, cabin=[[3]]), [True])
        terrain[0, 0, 2] = -1
        tracker.update(timestep(terrain=terrain, cabin=[[3]]), [True])
        tracker.update(
            timestep(terrain=terrain, cabin=[[3]], positions=(((1, 0),),)), [True]
        )
        terrain[0, 1, :2] = -1
        tracker.update(
            timestep(terrain=terrain, cabin=[[3]], positions=(((1, 0),),)), [True]
        )
        tracker.update(timestep(terrain=terrain, cabin=[[6]]), [True])
        terrain[0, 1, 2] = -1
        tracker.update(timestep(terrain=terrain, cabin=[[6]], done=[True]), [True])
        result = tracker.result()
        self.assertEqual(result["productive_dig_actions"][0], 4)
        self.assertEqual(result["productive_base_stances"][0], 3)
        self.assertEqual(result["unique_productive_base_poses"][0], 2)
        self.assertEqual(result["mean_workspace_dig_area_m2"][0], 2)
        self.assertAlmostEqual(result["p10_workspace_dig_area_m2"][0], 1.2)
        self.assertEqual(result["newly_dug_area_m2"][0], 6)
        self.assertEqual(result["lateral_dig_volume_fraction"][0], 0.5)
        self.assertAlmostEqual(result["mean_dig_lateral_score"][0], 0.5)

    def test_turns_wrap_and_base_turn_starts_another_workspace(self):
        tracker = EpisodeBehaviorMetrics(
            timestep(base=[[11]], cabin=[[11]]), configuration(),
            preserve_terminal_states=True,
        )
        terrain = np.zeros((1, 2, 3))
        terrain[0, 0, 0] = -1
        tracker.update(timestep(base=[[11]], cabin=[[11]], terrain=terrain), [True])
        tracker.update(timestep(base=[[0]], cabin=[[0]], terrain=terrain), [True])
        terrain[0, 0, 1] = -1
        tracker.update(timestep(base=[[0]], cabin=[[0]], terrain=terrain), [True])
        result = tracker.result()
        self.assertAlmostEqual(result["base_heading_change_deg"][0], 30)
        self.assertAlmostEqual(result["cabin_swing_deg"][0], 30)
        self.assertEqual(result["productive_base_stances"][0], 2)

    def test_partial_reset_and_redigging_do_not_inflate_fresh_work(self):
        initial = np.zeros((1, 2, 3))
        initial[0, 0, 0] = -1
        tracker = EpisodeBehaviorMetrics(
            timestep(terrain=initial), configuration(), preserve_terminal_states=True
        )
        tracker.update(timestep(terrain=np.zeros_like(initial)), [True])
        tracker.update(timestep(terrain=initial), [True])
        self.assertEqual(tracker.result()["productive_dig_actions"][0], 0)
        next_map = initial.copy()
        next_map[0, 0, 1] = -1
        tracker.update(timestep(terrain=next_map), [True])
        self.assertEqual(tracker.result()["newly_dug_area_m2"][0], 1)

    def test_lateral_fraction_weights_volume_and_excludes_dump_and_relift(self):
        target = -np.ones((1, 2, 3))
        target[0, 0, 0] = -3
        tracker = EpisodeBehaviorMetrics(
            timestep(target=target), configuration(), preserve_terminal_states=True
        )
        terrain = np.zeros((1, 2, 3))
        terrain[0, 0, 0] = -3
        tracker.update(timestep(target=target, terrain=terrain), [True])
        tracker.update(timestep(target=target, terrain=terrain, cabin=[[3]]), [True])
        terrain[0, 0, 1] = -1
        tracker.update(timestep(target=target, terrain=terrain, cabin=[[3]]), [True])
        terrain[0, 1, 2] = 3  # Dump and then rehandle an above-ground pile.
        tracker.update(timestep(target=target, terrain=terrain, cabin=[[3]]), [True])
        terrain[0, 1, 2] = 0
        tracker.update(timestep(target=target, terrain=terrain, cabin=[[3]]), [True])
        self.assertEqual(tracker.result()["lateral_dig_volume_fraction"][0], 0.25)
        self.assertEqual(tracker.result()["productive_dig_actions"][0], 2)

    def test_autoreset_terminal_jump_and_missing_geometry_are_unavailable(self):
        tracker = EpisodeBehaviorMetrics(
            timestep(), configuration(), preserve_terminal_states=False
        )
        tracker.update(timestep(positions=(((1, 0),),)), [True])
        tracker.update(timestep(positions=(((90, 0),),), done=[True]), [True])
        result = tracker.result()
        self.assertFalse(result["behavior_metrics_available"][0])
        self.assertTrue(np.isnan(result["base_travel_m"][0]))
        self.assertEqual(tracker.values["base_travel_m"][0], 1)
        missing = EpisodeBehaviorMetrics(
            NS(done=np.array([False]), state=0), configuration(),
            preserve_terminal_states=True,
        ).result()
        self.assertFalse(missing["behavior_metrics_available"][0])
        self.assertTrue(np.isnan(missing["base_travel_m"][0]))

    def test_no_work_and_zero_distance_ratios_are_undefined(self):
        tracker = EpisodeBehaviorMetrics(
            timestep(), configuration(), preserve_terminal_states=True
        )
        result = tracker.result()
        self.assertEqual(result["base_travel_m"][0], 0)
        self.assertEqual(result["productive_base_stances"][0], 0)
        for field in (
            "mean_workspace_dig_area_m2", "dig_area_per_travel_m",
            "mean_dig_area_m2", "lateral_dig_volume_fraction",
        ):
            self.assertTrue(np.isnan(result[field][0]), field)


if __name__ == "__main__":
    unittest.main()
