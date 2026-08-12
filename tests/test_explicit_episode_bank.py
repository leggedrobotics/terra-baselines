import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from eval_fixed_bank import manifest_reset_keys
from eval_fixed_bank import prepare_explicit_episode_reset
from eval_fixed_bank import verify_exact_reset
from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.benchmark_protocol import frozen_environment_protocol
from terra.benchmark_state import SCHEMA as AGENT_STATE_SCHEMA
from terra.benchmark_state import agent_state_sha256
from terra.benchmark_state import agent_to_record
from terra.benchmark_state import derive_initial_state_seed
from terra.benchmark_state import sample_benchmark_initial_agent
from terra.config import BatchConfig
from terra.config import CurriculumGlobalConfig
from terra.env import TerraEnvBatch
from terra.maps_buffer import reset_array_scenario_sha256
from utils.explicit_episode_bank import BANK_SCHEMA
from utils.explicit_episode_bank import EPISODE_SCHEMA
from utils.explicit_episode_bank import INITIAL_STATE_ROW_SCHEMA
from utils.explicit_episode_bank import explicit_episode_id
from utils.explicit_episode_bank import load_explicit_episode_panel

TERRA_REVISION = "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
RELEASE_ID = "test-legacy-easy-current-episodes-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _write_files_manifest(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "files.sha256"
    )
    (root / "files.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        )
    )


def _exact_selection_seeds(count: int) -> list[int]:
    found = [None] * count
    for seed in range(10_000):
        key = jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(key)
        index = int(jax.random.randint(subkey, (), 0, count))
        if found[index] is None:
            found[index] = seed
        if all(value is not None for value in found):
            return [int(value) for value in found]
    raise RuntimeError("could not enumerate test slots")


def _build_two_episode_bank(root: Path) -> None:
    panel = root / "development"
    for folder in (
        "images",
        "occupancy",
        "dumpability",
        "actions",
        "distance",
        "metadata",
    ):
        (panel / folder).mkdir(parents=True, exist_ok=True)

    env_cfg, _ = frozen_benchmark_protocol()
    protocol = frozen_environment_protocol(TERRA_REVISION)
    protocol_hash = protocol["environment_protocol_sha256"]
    _write_json(root / "environment_protocol.json", protocol)
    selection_seeds = _exact_selection_seeds(2)
    manifest_rows = []
    state_rows = []
    registry_rows = []
    conditions = (
        ("foundation", "legacy-foundation-easy"),
        ("trench", "legacy-trench-easy"),
    )
    for slot, (family, condition) in enumerate(conditions, start=1):
        target = np.zeros((64, 64), dtype=np.int8)
        target[39 + slot, 40] = -1
        target[39 + slot, 42] = 1
        arrays = {
            "images": target,
            "occupancy": np.zeros((64, 64), dtype=np.int8),
            "dumpability": np.ones((64, 64), dtype=np.bool_),
            "actions": np.zeros((64, 64), dtype=np.int8),
            "distance": np.zeros((64, 64), dtype=np.float32),
        }
        for folder, array in arrays.items():
            np.save(panel / folder / f"img_{slot}.npy", array)
        _write_json(panel / "metadata" / f"trench_{slot}.json", {})

        map_id = f"test-map-{slot}"
        source_id = f"test-source-{slot}"
        scenario_id = reset_array_scenario_sha256(arrays)
        agent, state_seed = sample_benchmark_initial_agent(
            release_id=RELEASE_ID,
            split="development",
            source_group_id=source_id,
            state_index=0,
            env_cfg=env_cfg,
            padding_mask=arrays["occupancy"],
            action_map=arrays["actions"],
            dumpability_mask=arrays["dumpability"],
        )
        state_hash = agent_state_sha256(agent)
        episode_id = explicit_episode_id(
            scenario_id,
            state_seed["seed_uint32"],
            state_hash,
            protocol_hash,
        )
        row = {
            "slot_index": slot,
            "map_id": map_id,
            "scenario_id": scenario_id,
            "source_id": source_id,
            "split": "development",
            "family": family,
            "stratum": "legacy_easy_capability_floor",
            "primary_cell": condition,
            "slot_weight": 1.0,
            "identity_slot_multiplicity": 1,
            "slot_selection_seed": selection_seeds[slot - 1],
            "environment_reset_seed": state_seed["seed_uint32"],
            "initial_state_seed_digest_sha256": state_seed["seed_digest_sha256"],
            "state_index": 0,
            "initial_agent_state_sha256": state_hash,
            "environment_protocol_sha256": protocol_hash,
            "episode_id": episode_id,
            "episode_id_schema": EPISODE_SCHEMA,
        }
        manifest_rows.append(row)
        state_rows.append(
            {
                "schema": INITIAL_STATE_ROW_SCHEMA,
                "slot_index": slot,
                "episode_id": episode_id,
                "initial_agent_state_sha256": state_hash,
                "initial_agent_state": agent_to_record(agent),
            }
        )
        registry_rows.append(
            {
                "map_id": map_id,
                "scenario_id": scenario_id,
                "source_id": source_id,
                "split": "development",
                "family": family,
                "primary_cell": condition,
            }
        )

    _write_jsonl(root / "source_registry.jsonl", registry_rows)
    registry_sha256 = _sha256(root / "source_registry.jsonl")
    _write_jsonl(panel / "manifest.jsonl", manifest_rows)
    _write_jsonl(panel / "initial_states.jsonl", state_rows)
    _write_json(
        panel / "dataset.json",
        {
            "schema": "terra_exact_map_dataset_v1",
            "slot_count": 2,
            "unique_identity_count": 2,
            "shape": [64, 64],
            "distance_metric": "test_normalized_distance",
            "distance_normalization": "test_unit_interval",
            "dumpability_normalization": "raw_dumpability_and_not_occupancy",
            "target_normalization": "positive_only_where_raw_dumpable_and_not_occupied",
            "accepted_dump_contract": "exact_visible_dump_v1",
            "scenario_identity_contract": "terra_reset_arrays_sha256_v1",
            "source_registry": "../source_registry.jsonl",
            "source_registry_sha256": registry_sha256,
            "evaluation_only": True,
            "minimum_dump_capacity_ratio": None,
            "explicit_initial_states": "initial_states.jsonl",
            "explicit_episode_id_schema": EPISODE_SCHEMA,
        },
    )
    _write_json(
        root / "episode_bank.json",
        {
            "schema": BANK_SCHEMA,
            "name": "Test Legacy-Easy Current Episodes",
            "release_id": RELEASE_ID,
            "protocol_id": "current_runtime_compat_v1",
            "diagnostic_only": True,
            "included_in_constrained_macro": False,
            "terra_revision": TERRA_REVISION,
            "environment_protocol": "environment_protocol.json",
            "environment_protocol_sha256": protocol_hash,
            "initial_agent_state_schema": AGENT_STATE_SCHEMA,
            "episode_id_schema": EPISODE_SCHEMA,
            "max_steps_in_episode": 450,
            "foundation_border_alignment": False,
            "source_registry": "source_registry.jsonl",
            "source_registry_sha256": registry_sha256,
            "evaluation_panels": {
                "development": {
                    "maps_path": "development",
                    "slot_count": 2,
                    "conditions": 2,
                    "manifest_sha256": _sha256(panel / "manifest.jsonl"),
                    "initial_states_sha256": _sha256(panel / "initial_states.jsonl"),
                }
            },
        },
    )
    _write_files_manifest(root)


class ExplicitEpisodeBankTest(unittest.TestCase):
    def test_two_map_cpu_reset_reproduces_maps_states_and_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_two_episode_bank(root)
            panel = load_explicit_episode_panel(
                root,
                "development",
                TERRA_REVISION,
            )
            self.assertEqual(panel.slot_count, 2)
            self.assertEqual(panel.maps_per_condition, 1)

            class TestCurriculum(CurriculumGlobalConfig):
                levels = [
                    {
                        "maps_path": "development",
                        "max_steps_in_episode": 450,
                        "rewards_type": 0,
                        "apply_trench_rewards": False,
                    }
                ]

            with patch.dict(
                os.environ,
                {"DATASET_PATH": str(root), "DATASET_SIZE": "2"},
            ):
                env = TerraEnvBatch(
                    batch_cfg=BatchConfig(
                        curriculum_global=TestCurriculum(),
                    ),
                    shuffle_maps=False,
                )
            frozen_cfg, _ = frozen_benchmark_protocol()
            env_params = jax.tree_util.tree_map(
                lambda value: jnp.repeat(jnp.asarray(value)[None], 2, axis=0),
                frozen_cfg,
            )
            selection_rows = [
                {
                    "slot_index": row["slot_index"],
                    "reset_seed": row["slot_selection_seed"],
                    "environment_protocol_sha256": row["environment_protocol_sha256"],
                }
                for row in panel.manifest_rows
            ]
            map_keys = manifest_reset_keys(
                selection_rows,
                2,
                panel.environment_protocol_sha256,
            )
            timestep, prepared_cfg, state_keys = prepare_explicit_episode_reset(
                env,
                env_params,
                map_keys,
                panel,
            )
            receipt = verify_exact_reset(
                env,
                prepared_cfg,
                None,
                panel.directory,
                2,
                timestep=timestep,
                expected_initial_state_sha256=panel.initial_agent_state_sha256,
                expected_state_keys=state_keys,
            )

            self.assertTrue(receipt["explicit_initial_state"]["passed"])
            np.testing.assert_array_equal(timestep.state.key, state_keys)
            self.assertEqual(
                [
                    agent_state_sha256(
                        jax.tree_util.tree_map(
                            lambda value: value[index], timestep.state.agent
                        )
                    )
                    for index in range(2)
                ],
                list(panel.initial_agent_state_sha256),
            )

    def test_loader_rejects_semantically_tampered_initial_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_two_episode_bank(root)
            states_path = root / "development" / "initial_states.jsonl"
            states = [json.loads(line) for line in states_path.read_text().splitlines()]
            old_angle = states[0]["initial_agent_state"]["agent_states"]["angle_base"][
                0
            ]
            states[0]["initial_agent_state"]["agent_states"]["angle_base"][0] = (
                old_angle + 1
            ) % 12
            _write_jsonl(states_path, states)
            descriptor_path = root / "episode_bank.json"
            descriptor = json.loads(descriptor_path.read_text())
            descriptor["evaluation_panels"]["development"]["initial_states_sha256"] = (
                _sha256(states_path)
            )
            _write_json(descriptor_path, descriptor)
            _write_files_manifest(root)

            with self.assertRaisesRegex(ValueError, "invalid state hash"):
                load_explicit_episode_panel(root, "development", TERRA_REVISION)

    def test_loader_rejects_metadata_changed_after_manifest_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_two_episode_bank(root)
            metadata = root / "development" / "metadata" / "trench_1.json"
            _write_json(metadata, {"foundation_border_type": 2})

            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                load_explicit_episode_panel(root, "development", TERRA_REVISION)

    def test_loader_rejects_undeclared_nested_checksum_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_two_episode_bank(root)
            (root / "development" / "files.sha256").write_text("undeclared\n")

            with self.assertRaisesRegex(ValueError, "coverage differs"):
                load_explicit_episode_panel(root, "development", TERRA_REVISION)


if __name__ == "__main__":
    unittest.main()
