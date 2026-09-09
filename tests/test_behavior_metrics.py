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
    target=None, acting=None, types=None, active=None, done=None, loaded=None,
    padding=None,
):
    positions = np.asarray(positions)
    count, slots = positions.shape[:2]
    base = np.zeros((count, slots)) if base is None else np.asarray(base)
    cabin = np.zeros((count, slots)) if cabin is None else np.asarray(cabin)
    types = np.zeros((count, slots)) if types is None else np.asarray(types)
    active = np.ones((count, slots)) if active is None else np.asarray(active)
    terrain = np.zeros((count, 2, 3)) if terrain is None else np.asarray(terrain)
    target = -np.ones_like(terrain) if target is None else np.asarray(target)
    loaded = np.zeros((count, slots)) if loaded is None else np.asarray(loaded)
    padding = np.zeros_like(terrain) if padding is None else np.asarray(padding)
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
                        loaded=loaded[:, slot, None],
                    )
                    for slot in range(slots)
                ),
                agent_active=active,
                current_agent=np.zeros(count, dtype=int) if acting is None else acting,
            ),
            world=NS(action_map=NS(map=terrain), target_map=NS(map=target),
                     padding_mask=NS(map=padding)),
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
        tracker.update(timestep(positions=positions, acting=[1]), [True], actions=[6])
        tracker.update(
            timestep(positions=(((0, 0), (102, 0)),), acting=[0]), [True], actions=[6]
        )
        self.assertEqual(tracker.result()["base_travel_m"][0], 2.0)
        self.assertEqual(tracker.result()["longest_action_pattern_steps"][0], 0)

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
        legacy = timestep()
        del legacy.state.agent.agent_states[0].loaded
        del legacy.state.world.padding_mask
        partial = EpisodeBehaviorMetrics(
            legacy, configuration(), preserve_terminal_states=True
        ).result()
        self.assertTrue(partial["behavior_metrics_available"][0])
        self.assertEqual(partial["base_travel_m"][0], 0)
        self.assertTrue(np.isnan(partial["relifted_volume_units"][0]))
        self.assertTrue(np.isnan(partial["new_accepted_disposal_volume_units"][0]))

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

    def test_relift_loop_changes_material_without_new_task_progress(self):
        target = np.zeros((1, 2, 3), dtype=np.int8)
        target[0, 0] = [-3, 0, 1]  # target, off-zone staging, accepted disposal
        terrain = np.zeros_like(target)
        tracker = EpisodeBehaviorMetrics(
            timestep(target=target, terrain=terrain), configuration((2.0,)),
            preserve_terminal_states=True,
        )
        # Reuse the same mutable terrain array to exercise snapshot ownership.
        for values, load, terminal in (
            ([-3, 0, 0], 3, False), ([-3, 3, 0], 0, False),
            ([-3, 0, 0], 3, False), ([-3, 0, 3], 0, False),
            ([-3, 0, 0], 3, False), ([-3, 0, 3], 0, False),
            ([-3, 0, 0], 3, False), ([-3, 0, 3], 0, True),
        ):
            terrain[0, 0] = values
            self.assertEqual(terrain.sum() + load, 0)
            tracker.update(timestep(target=target, terrain=terrain,
                                    loaded=[[load]], done=[terminal]), [True], actions=[6])
        result = tracker.result()
        expected = {
            "newly_dug_volume_units": 3, "newly_dug_area_m2": 4,
            "productive_dig_actions": 1, "relifted_volume_units": 9,
            "relifted_accepted_volume_units": 6, "relift_actions": 3,
            "redig_volume_units": 0, "new_accepted_disposal_volume_units": 3,
            "net_accepted_disposal_volume_units": 3,
            "longest_material_stall_steps": 0,
            "longest_task_progress_stall_steps": 4,
            "longest_action_pattern_steps": 8, "longest_action_pattern_period": 1,
        }
        for field, value in expected.items():
            self.assertEqual(result[field][0], value, field)
        # A later reset/caller mask error must not erase the initial terminal row.
        tracker.update(timestep(positions=(((90, 90),),)), [True], actions=[0])
        for field, value in expected.items():
            self.assertEqual(tracker.result()[field][0], value, field)

    def test_initial_depth_and_refill_redig_keep_unique_volume_separate(self):
        target = np.zeros((1, 2, 3))
        target[0, 0, 0] = -3
        terrain = np.zeros_like(target)
        terrain[0, 0, 0] = -1  # Existing partial-reset excavation earns no credit.
        tracker = EpisodeBehaviorMetrics(
            timestep(target=target, terrain=terrain), configuration(),
            preserve_terminal_states=True,
        )
        for depth in (-2, 0, -2, -3):
            terrain[0, 0, 0] = depth
            tracker.update(timestep(target=target, terrain=terrain), [True])
        result = tracker.result()
        self.assertEqual(result["newly_dug_volume_units"][0], 2)
        self.assertEqual(result["newly_dug_area_m2"][0], 0)
        self.assertEqual(result["productive_dig_actions"][0], 2)
        self.assertEqual(result["redig_volume_units"][0], 2)
        self.assertEqual(result["redig_actions"][0], 1)
        self.assertEqual(result["longest_task_progress_stall_steps"][0], 2)
        self.assertTrue(np.isnan(result["longest_action_pattern_steps"][0]))

    def test_accepted_stock_and_relift_use_net_terrain_accounting(self):
        target = np.zeros((1, 2, 3))
        target[0, 0] = [-6, 1, 1]
        padding = np.zeros_like(target)
        padding[0, 0, 2] = 1
        terrain = np.zeros_like(target)
        terrain[0, 0] = [-6, 4, 0]
        tracker = EpisodeBehaviorMetrics(
            timestep(target=target, padding=padding, terrain=terrain, loaded=[[2]]),
            configuration(), preserve_terminal_states=True,
        )
        # Deliberately place soil on a positive-target obstacle to test the
        # accepted-region definition independently of legal action generation.
        for values, load in (([-6, 4, 2], 0), ([-6, 4, 0], 2), ([-6, 6, 0], 0)):
            terrain[0, 0] = values
            tracker.update(timestep(target=target, padding=padding, terrain=terrain,
                                    loaded=[[load]]), [True])
        self.assertEqual(tracker.result()["new_accepted_disposal_volume_units"][0], 2)
        terrain[0, 0, 1] = 3
        tracker.update(timestep(target=target, padding=padding, terrain=terrain,
                                loaded=[[3]]), [True])
        result = tracker.result()
        self.assertEqual(result["net_accepted_disposal_volume_units"][0], -1)
        self.assertEqual(result["new_accepted_disposal_volume_units"][0], 2)
        self.assertEqual(result["relifted_volume_units"][0], 5)
        self.assertEqual(result["relifted_accepted_volume_units"][0], 3)
        self.assertEqual(result["longest_task_progress_stall_steps"][0], 2)

        for initial, final, target, expected_relift in (
            # Four units leave one accepted cell: three stay on the ground and
            # only one reaches the bucket. Cellwise decreases would report four.
            ([[-8, 4, 4], [0, 0, 0]], [[-8, 0, 7], [0, 0, 0]],
             [[-8, 1, 1], [0, 0, 0]], 1),
            # The one-unit load is fresh excavation. Simultaneous relaxation
            # fills another target hole, reducing positive stock by one too.
            ([[-8, 4, 4], [0, -2, 2]], [[-9, 4, 4], [0, -1, 1]],
             [[-9, 1, 1], [0, -2, 1]], 0),
        ):
            with self.subTest(expected_relift=expected_relift):
                initial, final, target = (np.array([v], dtype=np.int8)
                                          for v in (initial, final, target))
                self.assertEqual(initial.sum(), final.sum() + 1)
                tracker = EpisodeBehaviorMetrics(
                    timestep(terrain=initial, target=target), configuration(),
                    preserve_terminal_states=True,
                )
                tracker.update(timestep(terrain=final, target=target, loaded=[[1]]), [True])
                result = tracker.result()
                self.assertEqual(result["relifted_volume_units"][0], expected_relift)
                self.assertEqual(result["relifted_accepted_volume_units"][0], expected_relift)
                self.assertEqual(result["relift_actions"][0], int(expected_relift > 0))

    def test_sixteen_action_pattern_and_stalls_freeze_at_first_terminal(self):
        tracker = EpisodeBehaviorMetrics(
            timestep(), configuration(), preserve_terminal_states=True
        )
        pattern = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 6]
        for action in pattern * 2:
            tracker.update(timestep(), [True], actions=[action])
        result = tracker.result()
        self.assertEqual(result["longest_action_pattern_steps"][0], 32)
        self.assertEqual(result["longest_action_pattern_period"][0], 16)
        self.assertEqual(result["longest_material_stall_steps"][0], 32)
        self.assertEqual(result["longest_task_progress_stall_steps"][0], 32)
        terrain = np.zeros((1, 2, 3))
        terrain[0, 0, 0] = -1
        tracker.update(timestep(terrain=terrain, loaded=[[1]], done=[True]),
                       [True], actions=[6])
        terminal = tracker.result()
        tracker.update(timestep(), [True], actions=[7])
        for field, value in terminal.items():
            np.testing.assert_array_equal(tracker.result()[field], value, err_msg=field)


if __name__ == "__main__":
    unittest.main()
