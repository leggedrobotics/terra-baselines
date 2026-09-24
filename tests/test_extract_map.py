"""extract_map.py: the single-map env as the checkpoint was evaluated, the map scale, runtime coordinates."""

from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
import yaml

from terra.config import REWARD_V2_DISTANCE_BOUND, REWARD_V2_DISTANCE_REF_M, EnvConfig
from terra.env_generation.distance import (
    REWARD_V2_DISTANCE_PROTOCOL_ID,
    compute_reward_v2_distance_map,
)
from utils.helpers import replicate_checkpoint_env_config

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "extract_map", REPO_ROOT / "isaac_sim" / "extract_map.py"
)
extract_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_map)

EDGE_M = 36.5714285714  # terra.config ImmutableMapsConfig.edge_length_m
POLICY_TILE = EDGE_M / 64
SELECTOR_CONFIG = SimpleNamespace(
    executable_dig_observation=True,
    movement_feasibility_observation=False,
    previous_outcome_observation=False,
    distance_protocol_id=REWARD_V2_DISTANCE_PROTOCOL_ID,
)


def write_map(map_dir, map_json=None, terra_metadata=None, *, trench=False):
    """A 64 x 64 TerraMapMaker-format export with a dump strip and taxicab distance."""
    images = np.zeros((64, 64), dtype=np.int8)
    if trench:
        images[30:32, 12:52] = -1
    else:
        images[24:32, 24:32] = -1
    images[40:44, 10:54] = 1
    occupancy = np.zeros((64, 64), dtype=bool)
    occupancy[10:14, 10:14] = True
    rows, cols = np.indices(images.shape)
    taxicab = np.min(
        [np.abs(rows - r) + np.abs(cols - c) for r, c in np.argwhere(images == 1)], axis=0
    ).astype(np.float32) / 24.0
    layers = {
        "images": images,
        "occupancy": occupancy,
        "dumpability": np.ones((64, 64), dtype=bool),
        "actions": np.zeros((64, 64), dtype=np.int8),
        "distance": taxicab,
    }
    for name, array in layers.items():
        (map_dir / name).mkdir(parents=True, exist_ok=True)
        np.save(map_dir / name / "img_1.npy", array)
    (map_dir / "metadata").mkdir(exist_ok=True)
    if map_json is not None:
        (map_dir / "metadata" / "map.json").write_text(json.dumps(map_json))
    if terra_metadata is not None:
        (map_dir / "metadata" / "terra_metadata.yaml").write_text(yaml.safe_dump(terra_metadata))
    return map_dir, images, occupancy


def trench_json(**extra):
    return {
        "family": "trench",
        "axes_ABC": [{"A": 0.0, "B": -1.0, "C": 30.5}],
        "trench_segments_yx": [[[30.5, 11.5], [30.5, 51.5]]],
        "trench_axes_count": 1,
        "trench_half_width_tiles": 1.0,
        **extra,
    }


def gated_env_config():
    """A checkpoint-style single EnvConfig: one excavator, gate and executable dig on."""
    config = EnvConfig()._replace(
        agent_types=(0,),
        action_types=(0,),
        enforce_trench_dig_alignment=True,
        executable_dig_observation=True,
        max_steps_in_episode=450,
    )
    return replicate_checkpoint_env_config(config, 1)


def fake_env(edge_tiles):
    return SimpleNamespace(
        batch_cfg=SimpleNamespace(
            maps=SimpleNamespace(edge_length_m=EDGE_M),
            maps_dims=SimpleNamespace(maps_edge_length=edge_tiles),
        )
    )


@dataclass
class SavedConfig:
    """The train_config fields main() reads before it builds the environment."""

    admissible_dig_observation: bool = True
    executable_dig_observation: bool = True
    lateral_dig_cost: float = 0.0
    base_travel_cost: float = 0.0
    base_turn_cost: float = 0.0
    clip_action_maps: bool = True
    movement_feasibility_observation: bool = False
    previous_outcome_observation: bool = False
    distance_protocol_id: str = REWARD_V2_DISTANCE_PROTOCOL_ID


class MainWiringTest(unittest.TestCase):
    def test_main_evaluates_the_checkpoints_own_config_on_one_environment(self):
        checkpoint = {
            "train_config": SavedConfig(),
            # Saved per environment: agent_types keeps its per-agent axis.
            "env_config": EnvConfig()._replace(
                agent_types=(0,), action_types=(0,), executable_dig_observation=True
            ),
            "model": {},
        }
        captured = {}

        class StopBeforeRollout(Exception):
            pass

        def build_env(config, batch_cfg, map_path, rendering=False):
            captured["config"] = config
            return SimpleNamespace()

        def check_scale(map_path, env, env_cfgs, policy_path):
            captured["env_cfgs"] = env_cfgs
            raise StopBeforeRollout

        argv = ["extract_map.py", "--policy_path", "u100000.pkl", "--map_path", "map"]
        with (
            patch.object(extract_map, "load_pkl_object", return_value=checkpoint),
            patch.object(extract_map, "make_single_map_env", side_effect=build_env),
            patch.object(extract_map, "check_map_tile_size", side_effect=check_scale),
            patch("sys.argv", argv),
            self.assertRaises(StopBeforeRollout),
        ):
            extract_map.main()
        self.assertTrue(captured["config"].executable_dig_observation)
        env_cfgs = captured["env_cfgs"]
        self.assertEqual(np.shape(env_cfgs.agent_types), (1, 1))
        self.assertEqual(np.shape(env_cfgs.max_steps_in_episode), (1,))
        self.assertTrue(bool(np.asarray(env_cfgs.executable_dig_observation)[0]))


class MapScaleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env_cfgs = SimpleNamespace(tile_size=np.float32(POLICY_TILE))

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, name, edge_tiles=64, **files):
        map_dir = self.root / name
        (map_dir / "metadata").mkdir(parents=True)
        if "map_json" in files:
            (map_dir / "metadata" / "map.json").write_text(json.dumps(files["map_json"]))
        if "terra_metadata" in files:
            (map_dir / "metadata" / "terra_metadata.yaml").write_text(
                yaml.safe_dump(files["terra_metadata"])
            )
        return extract_map.check_map_tile_size(
            map_dir, fake_env(edge_tiles), self.env_cfgs, "u100000.pkl"
        )

    def test_map_at_another_tile_size_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"terra_metadata.yaml declares meters_per_tile=0.6875"):
            self.check("old_policy", terra_metadata={"meters_per_tile": 0.6875})
        with self.assertRaisesRegex(ValueError, r"map.json declares meters_per_tile=0.1.*64 x 64"):
            self.check("manual_plan_tiles", map_json={"meters_per_tile": 0.1})
        # Terra derives the tile size from the map edge: a 128-tile map runs at half the size.
        with self.assertRaisesRegex(ValueError, r"is 128 x 128 tiles, which Terra runs at"):
            self.check("wide", edge_tiles=128, map_json={"meters_per_tile": POLICY_TILE})

    def test_policy_scale_is_accepted(self):
        # TerraMapMaker keeps 5 decimals; a map without metadata takes the policy's tile size.
        tile = self.check("rounded", terra_metadata={"meters_per_tile": 0.57143})
        self.assertAlmostEqual(tile, POLICY_TILE, places=6)
        self.assertAlmostEqual(self.check("bare"), POLICY_TILE, places=6)


class SchemaV2Test(unittest.TestCase):
    def test_tile_corner_positions_and_the_maps_own_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_dir, images, _ = write_map(
                Path(tmp) / "map",
                {"meters_per_tile": POLICY_TILE},
                {
                    "meters_per_tile": POLICY_TILE,
                    "terra_origin_map_m": [1.5, -2.0],
                    "rotation_deg": 30.0,
                    # The source GridMap's own resolution is not the tile size.
                    "source_gridmap": {"resolution_m_per_cell": 0.1, "size_rows_cols": [600, 600]},
                },
            )
            mask = np.zeros(images.shape, dtype=bool)
            plan = [
                {
                    "step": step,
                    "agent_state": {"pos_base": [10.0, 20.0], "angle_base": 3.0},
                    "loaded_state_change": {"before": before, "after": not before},
                    "terrain_modification_mask": mask,
                    "dug_mask": mask,
                    "dump_mask": mask,
                    "traversability_mask": mask,
                }
                for step, before in ((4, False), (9, True))
            ]
            plan_json = extract_map._plan_to_schema_v2(plan, map_dir, float(np.float32(POLICY_TILE)))
        self.assertEqual(
            [waypoint["agent_state"]["pos_base"] for waypoint in plan_json["waypoints"]],
            [[10.5, 20.5], [10.5, 20.5]],
        )
        self.assertEqual(plan_json["metadata"]["pos_base_convention"], "tile_corner")
        self.assertEqual(
            plan_json["alignment"],
            {
                "meters_per_tile": POLICY_TILE,
                "origin_map_xy_m": [1.5, -2.0],
                "yaw_map_from_plan_rad": math.radians(30.0),
            },
        )


class SingleMapEnvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def env(self, name, map_json, *, trench=False):
        map_dir, images, occupancy = write_map(self.root / name, map_json, trench=trench)
        env = extract_map.make_single_map_env(SELECTOR_CONFIG, extract_map.BatchConfig(), map_dir)
        return env, images, occupancy

    def test_foundation_env_matches_the_evaluated_environment(self):
        env, images, occupancy = self.env("foundation", {"family": "foundation"})
        self.assertTrue(env.executable_dig_observation)
        self.assertEqual(env.maps_buffer.family_names[1], "foundation")
        expected = compute_reward_v2_distance_map(
            images,
            occupancy,
            tile_size_m=POLICY_TILE,
            distance_ref_m=REWARD_V2_DISTANCE_REF_M,
            distance_bound=REWARD_V2_DISTANCE_BOUND,
        )
        distance = np.asarray(env.maps_buffer.distance_maps)
        np.testing.assert_allclose(distance[0, 0], expected, rtol=0, atol=1e-7)
        # The checkpoint-style config keeps its agent axis and resets with the gate on.
        env_cfgs = gated_env_config()
        self.assertEqual(np.shape(env_cfgs.agent_types), (1, 1))
        timestep = env.reset(env_cfgs, jax.random.split(jax.random.PRNGKey(0), 1))
        np.testing.assert_allclose(
            np.asarray(timestep.observation["relocation_distance_map"])[0], expected, atol=1e-7
        )
        self.assertEqual(extract_map._terra_trench_axes(env).shape, (0, 3))

    def test_trench_needs_finite_sections_and_foundations_carry_none(self):
        env, _, _ = self.env("trench", trench_json(), trench=True)
        self.assertEqual(env.maps_buffer.family_names[1], "trench")
        env._validate_trench_alignment_metadata_requirements(gated_env_config())
        np.testing.assert_allclose(extract_map._terra_trench_axes(env), [[0.0, -1.0, 30.5]])
        # Legacy TerraMapMaker trench exports record no half width.
        legacy = trench_json()
        del legacy["trench_half_width_tiles"], legacy["trench_segments_yx"]
        env, _, _ = self.env("legacy", legacy, trench=True)
        with self.assertRaisesRegex(RuntimeError, "finite section metadata is missing"):
            env._validate_trench_alignment_metadata_requirements(gated_env_config())
        env, _, _ = self.env("unsectioned", {"family": "trench"}, trench=True)
        with self.assertRaisesRegex(RuntimeError, "trench-family maps lack axis metadata"):
            env._validate_trench_alignment_metadata_requirements(gated_env_config())
        with self.assertRaisesRegex(ValueError, "declares a foundation but carries trench axes"):
            self.env("contradiction", trench_json(family="foundation"), trench=True)


if __name__ == "__main__":
    unittest.main()
