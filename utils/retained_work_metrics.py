"""Streaming projection through effective tracked-excavator DO poses.

Successive work at an identical base pose is one setup even after discarded
navigation. Dump and relift poses remain in the route. Distances are straight
line lower bounds, not Nav2 routes. Adjacency compares unions of newly excavated
cells; it says nothing about requested-cone overlap or reachability.
"""
from copy import copy

import numpy as np


RETAINED_WORK_FIELDS = (
    "retained_work_operations", "retained_work_setups", "retained_productive_setups",
    "unique_area_per_retained_productive_setup_m2",
    "retained_work_straight_line_distance_m", "retained_work_initial_approach_straight_line_m",
    "retained_work_inter_setup_straight_line_m", "retained_work_max_inter_setup_straight_line_m",
    "retained_work_heading_change_lower_bound_deg", "retained_work_exact_pose_revisits",
    "retained_work_aba_pose_returns", "productive_setup_transfer_count",
    "fresh_union_relation_count", "fresh_union_edge_adjacent_transfer_count",
    "fresh_union_corner_adjacent_transfer_count", "fresh_union_edge_adjacency_fraction",
    "fresh_union_corner_adjacency_fraction",
)


def _adjacency(first, second):
    padded = np.pad(first, 1)
    rows, cols = first.shape
    edge = first.copy()
    corner = first.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = padded[1 + dx:1 + dx + rows, 1 + dy:1 + dy + cols]
            corner |= shifted
            if abs(dx) + abs(dy) <= 1:
                edge |= shifted
    return bool(np.any(edge & second)), bool(np.any(corner & second))


class RetainedWorkMetrics:
    """One tracked excavator; keep only two workspace masks, not the rollout."""

    def __init__(self, initial_pose, shape, tile_size):
        self.initial_pose = initial_pose
        self.tile_size = tile_size
        self.pose = None
        self.current_cells = np.zeros(shape, dtype=bool)
        self.current_productive = False
        self.previous_productive_cells = None
        self.seen_poses = set()
        self.last_two_poses = []
        self.new_area = 0.0
        self.values = {name: 0.0 for name in RETAINED_WORK_FIELDS}
        self.values["retained_work_initial_approach_straight_line_m"] = np.nan
        self.values["retained_work_max_inter_setup_straight_line_m"] = np.nan

    def _finish_setup(self):
        if not self.current_productive:
            return
        self.values["retained_productive_setups"] += 1
        previous = self.previous_productive_cells
        if previous is not None:
            self.values["productive_setup_transfer_count"] += 1
            if previous.any() and self.current_cells.any():
                edge, corner = _adjacency(previous, self.current_cells)
                self.values["fresh_union_relation_count"] += 1
                self.values["fresh_union_edge_adjacent_transfer_count"] += edge
                self.values["fresh_union_corner_adjacent_transfer_count"] += corner
        self.previous_productive_cells = self.current_cells

    def update(self, pose, fresh_cells, fresh_volume):
        """Caller supplies only DO transitions with terrain or load change."""
        self.values["retained_work_operations"] += 1
        if self.pose != pose:
            self._finish_setup()
            previous = self.pose if self.pose is not None else self.initial_pose
            distance = np.linalg.norm(np.asarray(pose[:2]) - previous[:2]) * self.tile_size
            angle = abs((pose[2] - previous[2] + np.pi) % (2 * np.pi) - np.pi)
            self.values["retained_work_straight_line_distance_m"] += distance
            self.values["retained_work_heading_change_lower_bound_deg"] += np.rad2deg(angle)
            if self.pose is None:
                self.values["retained_work_initial_approach_straight_line_m"] = distance
            else:
                self.values["retained_work_inter_setup_straight_line_m"] += distance
                self.values["retained_work_max_inter_setup_straight_line_m"] = max(
                    distance, self.values["retained_work_max_inter_setup_straight_line_m"]
                )
            self.values["retained_work_exact_pose_revisits"] += pose in self.seen_poses
            self.values["retained_work_aba_pose_returns"] += (
                len(self.last_two_poses) == 2 and pose == self.last_two_poses[0]
            )
            self.seen_poses.add(pose)
            self.last_two_poses = [*self.last_two_poses[-1:], pose]
            self.values["retained_work_setups"] += 1
            self.pose = pose
            self.current_cells = np.zeros_like(self.current_cells)
            self.current_productive = False
        self.current_cells |= fresh_cells
        self.current_productive |= fresh_volume > 0
        self.new_area += fresh_cells.sum() * self.tile_size**2

    def result(self):
        # Include the last workspace without closing the live accumulator.
        final = copy(self)
        final.values = self.values.copy()
        final._finish_setup()
        result = final.values
        count = result["retained_productive_setups"]
        result["unique_area_per_retained_productive_setup_m2"] = self.new_area / count if count else np.nan
        count = result["fresh_union_relation_count"]
        for kind in ("edge", "corner"):
            result[f"fresh_union_{kind}_adjacency_fraction"] = (
                result[f"fresh_union_{kind}_adjacent_transfer_count"] / count if count else np.nan
            )
        return result
