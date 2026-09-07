"""Physical travel and productive base-stance measurements for evaluation.

These are outcome measurements, independent of reward weights. A workspace
stance keeps the base position and heading fixed; cabin swings and repeated
dig/dump cycles can belong to the same stance. Returning after a base move
starts another stance. Excavation is credited only beyond each target cell's
maximum progress since reset, so moving soil back and digging it again cannot
inflate the measurements. Area counts cells first excavated during this episode.

The lateral metrics use cabin yaw relative to the chassis longitudinal axis.
They describe geometry, not tipping margin or physical machine stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BEHAVIOR_METRIC_FIELDS = (
    "base_travel_m",
    "base_travel_per_sqrt_target_area",
    "base_heading_change_deg",
    "cabin_swing_deg",
    "base_reposition_count",
    "productive_dig_actions",
    "productive_base_stances",
    "unique_productive_base_poses",
    "newly_dug_area_m2",
    "mean_dig_area_m2",
    "mean_workspace_dig_area_m2",
    "p10_workspace_dig_area_m2",
    "dig_area_per_travel_m",
    "lateral_dig_volume_fraction",
    "mean_dig_lateral_score",
)


def _per_episode(value, count):
    return np.broadcast_to(np.asarray(value).reshape(-1), (count,))


def _angle_change(after, before):
    return np.abs((after - before + np.pi) % (2 * np.pi) - np.pi)


def _ratio(numerator, denominator):
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


@dataclass
class _Snapshot:
    positions: np.ndarray
    base_yaw: np.ndarray
    cabin_yaw: np.ndarray
    active: np.ndarray
    types: np.ndarray
    acting: np.ndarray
    action_map: np.ndarray


def _snapshot(timestep, env_cfgs):
    """Read stable agent slots, not the observation's acting-agent-first order."""
    agent = timestep.state.agent
    count = len(timestep.done)
    states = agent.agent_states
    base_bins = _per_episode(env_cfgs.agent.angles_base, count)
    cabin_bins = _per_episode(env_cfgs.agent.angles_cabin, count)
    if np.any(base_bins <= 0) or np.any(cabin_bins <= 0):
        raise ValueError("behavior metrics require positive angular resolutions")
    return _Snapshot(
        positions=np.stack(
            [np.asarray(s.pos_base, dtype=np.float64) for s in states], axis=1
        ),
        base_yaw=np.stack(
            [np.asarray(s.angle_base).reshape(count) for s in states], axis=1
        ) * (2 * np.pi / base_bins[:, None]),
        cabin_yaw=np.stack(
            [np.asarray(s.angle_cabin).reshape(count) for s in states], axis=1
        ) * (2 * np.pi / cabin_bins[:, None]),
        active=np.asarray(agent.agent_active, dtype=bool),
        types=np.stack(
            [np.asarray(s.agent_type).reshape(count) for s in states], axis=1
        ),
        acting=_per_episode(agent.current_agent, count).astype(int),
        action_map=np.asarray(timestep.state.world.action_map.map),
    )


class EpisodeBehaviorMetrics:
    """Accumulate only initial-episode transitions, including the final action.

Raw terminal states are needed for complete measurements. With an auto-reset
environment a terminal row is marked unavailable and its reset jump excluded.
Older environments without raw agent geometry are also explicitly unavailable.
"""

    def __init__(self, timestep, env_cfgs, *, preserve_terminal_states):
        self.count = len(timestep.done)
        self.env_cfgs = env_cfgs
        self.preserve_terminal_states = preserve_terminal_states
        self.available = np.zeros(self.count, dtype=bool)
        self.previous = None
        self.values = {
            name: np.zeros(self.count, dtype=np.float64)
            for name in BEHAVIOR_METRIC_FIELDS
        }
        try:
            self.previous = _snapshot(timestep, env_cfgs)
            target = np.asarray(timestep.state.world.target_map.map, dtype=np.float64)
        except AttributeError:
            self.previous = None
            return
        self.tile_size = _per_episode(env_cfgs.tile_size, self.count)
        if not np.all(np.isfinite(self.tile_size) & (self.tile_size > 0)):
            raise ValueError("behavior metrics require positive finite tile sizes")
        self.source_depth = np.maximum(-target, 0)
        self.target_area = (target < 0).sum(axis=(-2, -1)) * self.tile_size**2
        self.best_depth = self._depth(self.previous.action_map)
        self.stance_ids = np.zeros(self.previous.active.shape, dtype=int)
        self.stance_areas = [{} for _ in range(self.count)]
        self.productive_poses = [set() for _ in range(self.count)]
        self.excavator_volume = np.zeros(self.count)
        self.lateral_volume = np.zeros(self.count)
        self.lateral_weight = np.zeros(self.count)
        self.available[:] = True

    def _depth(self, action_map):
        # Cast before negation: terrain arrays may use a small integer dtype.
        return np.clip(-action_map.astype(np.float64), 0, self.source_depth)

    def update(self, timestep, active_episode_mask):
        if self.previous is None:
            return
        active_episode_mask = np.asarray(active_episode_mask, dtype=bool)
        valid = active_episode_mask & self.available
        if not self.preserve_terminal_states:
            reset = active_episode_mask & np.asarray(timestep.done, dtype=bool)
            self.available[reset] = False
            valid &= ~reset
        after = _snapshot(timestep, self.env_cfgs)
        before = self.previous
        active_agents = before.active & after.active & valid[:, None]
        delta_m = np.linalg.norm(after.positions - before.positions, axis=-1)
        delta_m *= self.tile_size[:, None]
        base_turn = _angle_change(after.base_yaw, before.base_yaw)
        cabin_turn = _angle_change(after.cabin_yaw, before.cabin_yaw)
        repositioned = ((delta_m > 1e-8) | (base_turn > 1e-8)) & active_agents
        self.values["base_travel_m"] += (delta_m * active_agents).sum(axis=1)
        self.values["base_heading_change_deg"] += np.rad2deg(
            (base_turn * active_agents).sum(axis=1)
        )
        self.values["cabin_swing_deg"] += np.rad2deg(
            (cabin_turn * active_agents).sum(axis=1)
        )
        self.values["base_reposition_count"] += repositioned.sum(axis=1)
        self.stance_ids += repositioned

        depth = self._depth(after.action_map)
        fresh_volume = np.maximum(depth - self.best_depth, 0).sum(axis=(-2, -1))
        fresh_area = ((depth > 0) & (self.best_depth == 0)).sum(axis=(-2, -1))
        fresh_area = fresh_area * self.tile_size**2
        for row in np.flatnonzero(valid & (fresh_volume > 0)):
            slot = before.acting[row]
            if not before.active[row, slot]:
                continue
            self.values["productive_dig_actions"][row] += 1
            self.values["newly_dug_area_m2"][row] += fresh_area[row]
            key = (slot, self.stance_ids[row, slot])
            areas = self.stance_areas[row]
            areas[key] = areas.get(key, 0.0) + fresh_area[row]
            self.productive_poses[row].add(
                (slot, *after.positions[row, slot], after.base_yaw[row, slot])
            )
            if before.types[row, slot] == 0:
                relative_yaw = before.cabin_yaw[row, slot]
                # Front and rear are longitudinal; 90 degrees is fully lateral.
                lateral_score = np.sin(relative_yaw) ** 2
                self.excavator_volume[row] += fresh_volume[row]
                self.lateral_volume[row] += fresh_volume[row] * (
                    lateral_score > 0.5 + 1e-8
                )
                self.lateral_weight[row] += fresh_volume[row] * lateral_score
        self.best_depth = np.where(
            valid[:, None, None], np.maximum(self.best_depth, depth), self.best_depth
        )
        self.previous = after

    def result(self):
        result = {name: values.copy() for name, values in self.values.items()}
        if self.previous is not None:
            result["base_travel_per_sqrt_target_area"] = _ratio(
                result["base_travel_m"], np.sqrt(self.target_area)
            )
            result["dig_area_per_travel_m"] = _ratio(
                result["newly_dug_area_m2"], result["base_travel_m"]
            )
            result["mean_dig_area_m2"] = _ratio(
                result["newly_dug_area_m2"], result["productive_dig_actions"]
            )
            result["lateral_dig_volume_fraction"] = _ratio(
                self.lateral_volume, self.excavator_volume
            )
            result["mean_dig_lateral_score"] = _ratio(
                self.lateral_weight, self.excavator_volume
            )
            for row, areas in enumerate(self.stance_areas):
                values = list(areas.values())
                result["productive_base_stances"][row] = len(values)
                result["unique_productive_base_poses"][row] = len(
                    self.productive_poses[row]
                )
                result["mean_workspace_dig_area_m2"][row] = (
                    np.mean(values) if values else np.nan
                )
                result["p10_workspace_dig_area_m2"][row] = (
                    np.percentile(values, 10) if values else np.nan
                )
        for value in result.values():
            value[~self.available] = np.nan
        return {"behavior_metrics_available": self.available.copy(), **result}
