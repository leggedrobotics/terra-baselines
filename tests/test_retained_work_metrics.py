"""Deployment projection, terminal work, and workspace continuity contracts."""
import numpy as np

from utils.retained_work_metrics import RetainedWorkMetrics, _adjacency


def cells(*points):
    result = np.zeros((7, 7), dtype=bool)
    for point in points:
        result[point] = True
    return result


def test_discarded_excursion_keeps_setup_but_dump_pose_splits_it():
    tracker = RetainedWorkMetrics((0, 0, 0), (7, 7), .5)
    tracker.update((0, 0, 0), cells((2, 2)), 1)
    tracker.update((0, 0, 0), cells(), 0)  # dump here
    # Navigation leaves and returns; it never enters this projection.
    tracker.update((0, 0, 0), cells((2, 3)), 1)
    result = tracker.result()
    assert result["retained_work_setups"] == 1
    assert result["retained_work_straight_line_distance_m"] == 0
    assert result["unique_area_per_retained_productive_setup_m2"] == .5
    # A dump elsewhere must remain in the route, including the return to dig.
    tracker.update((2, 0, 0), cells(), 0)
    tracker.update((0, 0, 0), cells((2, 4)), 1)
    result = tracker.result()
    assert result["retained_work_setups"] == 3
    assert result["retained_productive_setups"] == 2
    assert result["retained_work_straight_line_distance_m"] == 2
    assert result["retained_work_max_inter_setup_straight_line_m"] == 1
    assert result["retained_work_exact_pose_revisits"] == 1
    assert result["retained_work_aba_pose_returns"] == 1
    assert result["fresh_union_edge_adjacency_fraction"] == 1
    assert tracker.result() == result  # reporting must not finalize twice


def test_heading_changes_and_initial_approach_are_counted():
    tracker = RetainedWorkMetrics((0, 0, 11 * np.pi / 6), (7, 7), .5)
    tracker.update((3, 4, 0), cells((1, 1)), 1)
    tracker.update((3, 4, np.pi / 6), cells((2, 2)), 1)
    result = tracker.result()
    assert result["retained_work_initial_approach_straight_line_m"] == 2.5
    assert result["retained_work_inter_setup_straight_line_m"] == 0
    assert np.isclose(result["retained_work_heading_change_lower_bound_deg"], 60)
    assert result["fresh_union_edge_adjacency_fraction"] == 0
    assert result["fresh_union_corner_adjacency_fraction"] == 1


def test_deeper_cut_counts_productive_setup_without_fake_area_or_adjacency():
    tracker = RetainedWorkMetrics((0, 0, 0), (7, 7), .5)
    tracker.update((0, 0, 0), cells(), 1)
    tracker.update((1, 0, 0), cells((2, 2)), 1)
    result = tracker.result()
    assert result["retained_productive_setups"] == 2
    assert result["productive_setup_transfer_count"] == 1
    assert result["fresh_union_relation_count"] == 0
    assert np.isnan(result["fresh_union_corner_adjacency_fraction"])
    assert result["unique_area_per_retained_productive_setup_m2"] == .125


def test_adjacency_does_not_wrap_edges():
    assert _adjacency(cells((0, 0)), cells((6, 0))) == (False, False)
    assert _adjacency(cells((0, 0)), cells((0, 1))) == (True, True)
    assert _adjacency(cells((0, 0)), cells((1, 1))) == (False, True)
