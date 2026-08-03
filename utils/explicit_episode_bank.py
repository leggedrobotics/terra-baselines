"""Validate and load a frozen Terra explicit-episode evaluation panel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terra.benchmark_protocol import canonical_json_sha256
from terra.benchmark_protocol import frozen_environment_protocol
from terra.benchmark_state import SCHEMA as AGENT_STATE_SCHEMA
from terra.benchmark_state import agent_from_record
from terra.benchmark_state import agent_state_sha256
from terra.benchmark_state import derive_initial_state_seed
from terra.maps_buffer import EXACT_DATASET_SCHEMA
from terra.maps_buffer import validate_exact_dataset_contract

BANK_SCHEMA = "terra_legacy_easy_explicit_episode_bank_v1"
EPISODE_SCHEMA = "terra_explicit_episode_id_v1"
INITIAL_STATE_ROW_SCHEMA = "terra_explicit_initial_state_row_v1"
PROTOCOL_ID = "current_runtime_compat_v1"
PANELS = ("train", "promotion", "development", "sealed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str, source: Path) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{source}: {field} must be a lowercase SHA-256")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected one JSON object")
        rows.append(row)
    return rows


def _safe_relative_file(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must stay inside the episode bank: {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the episode bank: {value!r}") from error
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _safe_relative_directory(root: Path, value: object, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must stay inside the episode bank: {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the episode bank: {value!r}") from error
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(path)
    return relative.as_posix(), path


def _file_manifest(root: Path) -> tuple[dict[str, str], str]:
    path = root / "files.sha256"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    declared: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: malformed hash row") from error
        _require_sha256(digest, "digest", path)
        if relative in declared:
            raise ValueError(f"{path}: duplicate entry {relative!r}")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{path}: unsafe entry {relative!r}")
        declared[relative] = digest
    return declared, _sha256_file(path)


def _verify_declared_file(
    root: Path,
    declared: dict[str, str],
    path: Path,
) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    relative = path.relative_to(root).as_posix()
    expected = declared.get(relative)
    if expected is None:
        raise ValueError(f"files.sha256 does not cover {relative}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"file hash mismatch for {relative}: expected {expected}, got {actual}"
        )
    return actual


def explicit_episode_id(
    map_scenario_id: str,
    environment_reset_seed: int,
    initial_agent_state_sha256: str,
    environment_protocol_sha256: str,
) -> str:
    """Return the episode identity defined by the explicit-bank schema."""
    return canonical_json_sha256(
        {
            "schema": EPISODE_SCHEMA,
            "map_scenario_id": map_scenario_id,
            "environment_reset_seed": environment_reset_seed,
            "initial_agent_state_sha256": initial_agent_state_sha256,
            "environment_protocol_sha256": environment_protocol_sha256,
        }
    )


def _uint32(value: object, field: str, slot: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 2**32 - 1
    ):
        raise ValueError(f"manifest slot {slot} has invalid {field}")
    return value


@dataclass(frozen=True)
class ExplicitEpisodePanel:
    root: Path
    name: str
    release_id: str
    panel_name: str
    maps_path: str
    directory: Path
    manifest_rows: tuple[dict[str, Any], ...]
    initial_agents: tuple[Any, ...]
    environment_reset_seeds: tuple[int, ...]
    slot_selection_seeds: tuple[int, ...]
    terra_revision: str
    environment_protocol_sha256: str
    source_registry_sha256: str
    episode_bank_sha256: str
    environment_protocol_file_sha256: str
    manifest_sha256: str
    initial_states_sha256: str
    files_manifest_sha256: str
    condition_count: int
    maps_per_condition: int

    @property
    def slot_count(self) -> int:
        return len(self.manifest_rows)

    @property
    def initial_agent_state_sha256(self) -> tuple[str, ...]:
        return tuple(row["initial_agent_state_sha256"] for row in self.manifest_rows)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": BANK_SCHEMA,
            "name": self.name,
            "release_id": self.release_id,
            "diagnostic_only": True,
            "included_in_constrained_macro": False,
            "panel": self.panel_name,
            "maps_path": self.maps_path,
            "slot_count": self.slot_count,
            "condition_count": self.condition_count,
            "maps_per_condition": self.maps_per_condition,
            "condition_balanced": True,
            "terra_revision": self.terra_revision,
            "environment_protocol_sha256": self.environment_protocol_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "episode_bank_sha256": self.episode_bank_sha256,
            "environment_protocol_file_sha256": (self.environment_protocol_file_sha256),
            "manifest_sha256": self.manifest_sha256,
            "initial_states_sha256": self.initial_states_sha256,
            "files_manifest_sha256": self.files_manifest_sha256,
            "episode_id_schema": EPISODE_SCHEMA,
            "initial_agent_state_schema": AGENT_STATE_SCHEMA,
        }


def load_explicit_episode_panel(
    root: str | Path,
    panel_name: str,
    terra_revision: str,
) -> ExplicitEpisodePanel:
    """Validate one panel and decode its exact initial Agent trees."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if panel_name not in PANELS:
        raise ValueError(f"panel_name must be one of {PANELS}")
    if not isinstance(terra_revision, str) or not terra_revision:
        raise ValueError("terra_revision must be a nonempty immutable revision")

    descriptor_path = root / "episode_bank.json"
    descriptor = _read_json(descriptor_path)
    if descriptor.get("schema") != BANK_SCHEMA:
        raise ValueError(f"{descriptor_path}: unsupported schema")
    if descriptor.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{descriptor_path}: unsupported protocol_id")
    if descriptor.get("diagnostic_only") is not True:
        raise ValueError("explicit Legacy-Easy bank must remain diagnostic-only")
    if descriptor.get("included_in_constrained_macro") is not False:
        raise ValueError("explicit Legacy-Easy bank cannot enter the constrained macro")
    if descriptor.get("terra_revision") != terra_revision:
        raise ValueError(
            "episode-bank Terra revision mismatch: "
            f"expected {terra_revision}, got {descriptor.get('terra_revision')}"
        )
    if descriptor.get("initial_agent_state_schema") != AGENT_STATE_SCHEMA:
        raise ValueError("episode bank has an unsupported initial-agent schema")
    if descriptor.get("episode_id_schema") != EPISODE_SCHEMA:
        raise ValueError("episode bank has an unsupported episode-ID schema")
    if descriptor.get("max_steps_in_episode") != 450:
        raise ValueError("explicit Legacy-Easy episodes require horizon 450")
    if descriptor.get("foundation_border_alignment") is not False:
        raise ValueError("explicit Legacy-Easy episodes require edge alignment off")
    release_id = descriptor.get("release_id")
    name = descriptor.get("name")
    if not isinstance(release_id, str) or not release_id:
        raise ValueError("episode bank release_id must be nonempty")
    if not isinstance(name, str) or not name:
        raise ValueError("episode bank name must be nonempty")

    declared_files, files_manifest_sha256 = _file_manifest(root)
    episode_bank_sha256 = _verify_declared_file(root, declared_files, descriptor_path)

    protocol_path = _safe_relative_file(
        root,
        descriptor.get("environment_protocol"),
        "environment_protocol",
    )
    environment_protocol_file_sha256 = _verify_declared_file(
        root, declared_files, protocol_path
    )
    protocol = _read_json(protocol_path)
    expected_protocol = frozen_environment_protocol(terra_revision)
    if protocol != expected_protocol:
        raise ValueError(
            "episode-bank environment protocol differs from imported Terra authority"
        )
    protocol_hash = _require_sha256(
        protocol.get("environment_protocol_sha256"),
        "environment_protocol_sha256",
        protocol_path,
    )
    canonical_payload = {
        key: value
        for key, value in protocol.items()
        if key != "environment_protocol_sha256"
    }
    if canonical_json_sha256(canonical_payload) != protocol_hash:
        raise ValueError("environment protocol canonical hash is invalid")
    if descriptor.get("environment_protocol_sha256") != protocol_hash:
        raise ValueError("episode-bank descriptor binds a different protocol hash")

    registry_path = _safe_relative_file(
        root,
        descriptor.get("source_registry"),
        "source_registry",
    )
    source_registry_sha256 = _verify_declared_file(root, declared_files, registry_path)
    if descriptor.get("source_registry_sha256") != source_registry_sha256:
        raise ValueError("episode-bank source-registry hash mismatch")
    registry_rows = _read_jsonl(registry_path)
    registry = {}
    for row in registry_rows:
        map_id = row.get("map_id")
        if not isinstance(map_id, str) or not map_id:
            raise ValueError("source registry contains an invalid map_id")
        if map_id in registry:
            raise ValueError(f"source registry repeats map_id {map_id!r}")
        registry[map_id] = row

    panels = descriptor.get("evaluation_panels")
    if not isinstance(panels, dict) or panel_name not in panels:
        raise ValueError(f"episode bank does not declare panel {panel_name!r}")
    panel = panels[panel_name]
    if not isinstance(panel, dict):
        raise ValueError(f"panel {panel_name!r} descriptor must be an object")
    maps_path, directory = _safe_relative_directory(
        root, panel.get("maps_path"), f"evaluation_panels.{panel_name}.maps_path"
    )
    slot_count = panel.get("slot_count")
    condition_count = panel.get("conditions")
    if (
        not isinstance(slot_count, int)
        or isinstance(slot_count, bool)
        or slot_count <= 0
    ):
        raise ValueError(f"panel {panel_name!r} has invalid slot_count")
    if (
        not isinstance(condition_count, int)
        or isinstance(condition_count, bool)
        or condition_count <= 0
    ):
        raise ValueError(f"panel {panel_name!r} has invalid condition count")

    dataset_path = directory / "dataset.json"
    manifest_path = directory / "manifest.jsonl"
    states_path = directory / "initial_states.jsonl"
    for required in (dataset_path, manifest_path, states_path):
        _verify_declared_file(root, declared_files, required)
    dataset = _read_json(dataset_path)
    if dataset.get("schema") != EXACT_DATASET_SCHEMA:
        raise ValueError(f"{dataset_path}: unsupported exact-dataset schema")
    if dataset.get("evaluation_only") is not True:
        raise ValueError("explicit episode panels must remain evaluation-only")
    if dataset.get("explicit_initial_states") != "initial_states.jsonl":
        raise ValueError("dataset must bind initial_states.jsonl")
    if dataset.get("explicit_episode_id_schema") != EPISODE_SCHEMA:
        raise ValueError("dataset has an unsupported explicit episode-ID schema")
    if dataset.get("source_registry_sha256") != source_registry_sha256:
        raise ValueError("dataset source-registry hash mismatch")

    # Terra owns reset-array, scenario-identity, and source-disjointness checks.
    terra_rows, _, _ = validate_exact_dataset_contract(directory, slot_count)
    rows = _read_jsonl(manifest_path)
    if rows != terra_rows:
        raise RuntimeError("Terra returned a different ordered manifest")
    states = _read_jsonl(states_path)
    if len(states) != slot_count:
        raise ValueError(
            f"{states_path} contains {len(states)} rows, expected {slot_count}"
        )
    manifest_sha256 = _sha256_file(manifest_path)
    initial_states_sha256 = _sha256_file(states_path)
    if panel.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"panel {panel_name!r} manifest hash mismatch")
    if panel.get("initial_states_sha256") != initial_states_sha256:
        raise ValueError(f"panel {panel_name!r} initial-state hash mismatch")

    expected_slots = list(range(1, slot_count + 1))
    if [row.get("slot_index") for row in rows] != expected_slots:
        raise ValueError("explicit episode manifest slots must be contiguous")
    if [row.get("slot_index") for row in states] != expected_slots:
        raise ValueError("explicit initial-state slots must be contiguous")

    initial_agents = []
    environment_reset_seeds = []
    slot_selection_seeds = []
    episode_ids: set[str] = set()
    condition_counts: dict[str, int] = {}
    for slot, (row, state_row) in enumerate(zip(rows, states), start=1):
        if row.get("split") != panel_name:
            raise ValueError(f"manifest slot {slot} has the wrong split")
        if row.get("environment_protocol_sha256") != protocol_hash:
            raise ValueError(f"manifest slot {slot} has a stale protocol")
        if row.get("episode_id_schema") != EPISODE_SCHEMA:
            raise ValueError(f"manifest slot {slot} has an unsupported episode schema")
        if state_row.get("schema") != INITIAL_STATE_ROW_SCHEMA:
            raise ValueError(f"initial-state slot {slot} has an unsupported schema")

        source_id = row.get("source_id")
        state_index = row.get("state_index")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"manifest slot {slot} has invalid source_id")
        state_index = _uint32(state_index, "state_index", slot)
        environment_seed = _uint32(
            row.get("environment_reset_seed"), "environment_reset_seed", slot
        )
        slot_seed = _uint32(row.get("slot_selection_seed"), "slot_selection_seed", slot)
        expected_seed, expected_seed_digest = derive_initial_state_seed(
            release_id,
            panel_name,
            source_id,
            state_index,
        )
        if environment_seed != expected_seed:
            raise ValueError(f"manifest slot {slot} has a stale environment reset seed")
        if row.get("initial_state_seed_digest_sha256") != expected_seed_digest:
            raise ValueError(f"manifest slot {slot} has a stale state-seed digest")

        if state_row.get("episode_id") != row.get("episode_id"):
            raise ValueError(f"initial-state slot {slot} binds a different episode")
        state_hash = _require_sha256(
            row.get("initial_agent_state_sha256"),
            "initial_agent_state_sha256",
            manifest_path,
        )
        if state_row.get("initial_agent_state_sha256") != state_hash:
            raise ValueError(f"initial-state slot {slot} binds a different state hash")
        agent = agent_from_record(state_row.get("initial_agent_state"))
        if agent_state_sha256(agent) != state_hash:
            raise ValueError(f"initial-state slot {slot} has invalid state hash")

        expected_episode = explicit_episode_id(
            row.get("scenario_id"),
            environment_seed,
            state_hash,
            protocol_hash,
        )
        if row.get("episode_id") != expected_episode:
            raise ValueError(f"manifest slot {slot} has invalid episode_id")
        if expected_episode in episode_ids:
            raise ValueError(f"explicit episode repeats at slot {slot}")
        episode_ids.add(expected_episode)

        registry_row = registry.get(row.get("map_id"))
        registry_identity = (
            row.get("source_id"),
            row.get("split"),
            row.get("family"),
            row.get("primary_cell"),
            row.get("scenario_id"),
        )
        if registry_row is None or registry_identity != (
            registry_row.get("source_id"),
            registry_row.get("split"),
            registry_row.get("family"),
            registry_row.get("primary_cell"),
            registry_row.get("scenario_id"),
        ):
            raise ValueError(f"manifest slot {slot} differs from the source registry")

        condition = row.get("primary_cell")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"manifest slot {slot} has invalid primary_cell")
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        initial_agents.append(agent)
        environment_reset_seeds.append(environment_seed)
        slot_selection_seeds.append(slot_seed)

    if len(condition_counts) != condition_count:
        raise ValueError(
            f"panel declares {condition_count} conditions, observed {len(condition_counts)}"
        )
    counts = set(condition_counts.values())
    if len(counts) != 1:
        raise ValueError(
            f"explicit panel is not condition-balanced: {condition_counts}"
        )
    maps_per_condition = next(iter(counts))

    return ExplicitEpisodePanel(
        root=root,
        name=name,
        release_id=release_id,
        panel_name=panel_name,
        maps_path=maps_path,
        directory=directory,
        manifest_rows=tuple(rows),
        initial_agents=tuple(initial_agents),
        environment_reset_seeds=tuple(environment_reset_seeds),
        slot_selection_seeds=tuple(slot_selection_seeds),
        terra_revision=terra_revision,
        environment_protocol_sha256=protocol_hash,
        source_registry_sha256=source_registry_sha256,
        episode_bank_sha256=episode_bank_sha256,
        environment_protocol_file_sha256=environment_protocol_file_sha256,
        manifest_sha256=manifest_sha256,
        initial_states_sha256=initial_states_sha256,
        files_manifest_sha256=files_manifest_sha256,
        condition_count=condition_count,
        maps_per_condition=maps_per_condition,
    )
