"""Load the frozen accepted-map bank used by the P5 curriculum experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "terra_curriculum_loader_bank_v1"
SCENARIO_IDENTITY_CONTRACT = "terra_reset_arrays_sha256_v1"
ARMS = ("F-ANCHOR", "T-ANCHOR", "G-UNIFORM", "G-ADAPTIVE")
FAMILIES = ("foundation", "trench")
BRANCH_DEPTHS = ("Anchor", "One-axis", "Composed")
TRAIN_MAPS_PER_CONDITION = 64
RESET_ARRAY_FOLDERS = (
    "images",
    "occupancy",
    "dumpability",
    "actions",
    "distance",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment_protocol_for_revision(terra_revision: str) -> dict:
    """Derive the protocol from imported Terra code and an explicit revision."""
    from terra.benchmark_protocol import frozen_environment_protocol

    return frozen_environment_protocol(terra_revision)


def _sha256_field(payload: dict, field: str, source: Path) -> str:
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{source}: {field} must be a lowercase SHA-256")
    return value


def _json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _safe_relative_directory(root: Path, raw_path: object) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("train entry maps_path must be a nonempty string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"train entry maps_path must stay under the bank: {raw_path}")
    directory = (root / relative).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"train entry escapes the accepted bank: {raw_path}") from exc
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return relative.as_posix(), directory


@dataclass(frozen=True)
class AcceptedLevel:
    condition_id: str
    family: str
    branch_depth: str
    maps_path: str
    map_count: int


@dataclass(frozen=True)
class AcceptedBank:
    root: Path
    arm: str
    terra_revision: str
    levels: tuple[AcceptedLevel, ...]
    evaluation_panels: tuple["AcceptedPanel", ...]
    map_count_per_condition: int
    environment_protocol_sha256: str
    source_registry_sha256: str


@dataclass(frozen=True)
class AcceptedPanel:
    name: str
    maps_path: str
    slot_count: int
    condition_count: int


def _validate_level(root: Path, entry: dict) -> AcceptedLevel:
    condition_id = entry.get("condition_id")
    family = entry.get("family")
    branch_depth = entry.get("branch_depth")
    map_count = entry.get("map_count")
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("train entry condition_id must be a nonempty string")
    if family not in FAMILIES:
        raise ValueError(f"{condition_id}: family must be one of {FAMILIES}")
    if branch_depth not in BRANCH_DEPTHS:
        raise ValueError(
            f"{condition_id}: branch_depth must be one of {BRANCH_DEPTHS}"
        )
    if not isinstance(map_count, int) or isinstance(map_count, bool) or map_count <= 0:
        raise ValueError(f"{condition_id}: map_count must be a positive integer")
    maps_path, directory = _safe_relative_directory(root, entry.get("maps_path"))

    manifest = _json_lines(directory / "manifest.jsonl")
    if len(manifest) != map_count:
        raise ValueError(
            f"{condition_id}: descriptor declares {map_count} maps but "
            f"{directory / 'manifest.jsonl'} contains {len(manifest)}"
        )
    expected_slots = list(range(1, map_count + 1))
    actual_slots = [row.get("slot_index") for row in manifest]
    if actual_slots != expected_slots:
        raise ValueError(
            f"{condition_id}: manifest slots must be contiguous 1..{map_count}"
        )
    families = {row.get("family") for row in manifest}
    cells = {row.get("primary_cell") for row in manifest}
    if families != {family}:
        raise ValueError(
            f"{condition_id}: manifest family {sorted(str(v) for v in families)} "
            f"does not match {family}"
        )
    if cells != {condition_id}:
        raise ValueError(
            f"{condition_id}: manifest primary_cell "
            f"{sorted(str(v) for v in cells)} does not match the condition"
        )
    map_ids = [row.get("map_id") for row in manifest]
    if any(not isinstance(map_id, str) or not map_id for map_id in map_ids):
        raise ValueError(f"{condition_id}: manifest has an invalid map_id")
    if len(map_ids) != len(set(map_ids)):
        raise ValueError(f"{condition_id}: manifest repeats a map_id")

    # Terra performs the full exact-dataset validation at environment creation.
    # Checking its three authority files here catches a wrong/legacy bank before
    # the expensive JAX path without duplicating maps_buffer.py's validator.
    for required in ("dataset.json", "manifest.jsonl"):
        if not (directory / required).is_file():
            raise FileNotFoundError(directory / required)

    return AcceptedLevel(
        condition_id=condition_id,
        family=family,
        branch_depth=branch_depth,
        maps_path=maps_path,
        map_count=map_count,
    )


def validate_staged_training_bank(root: str | Path) -> int:
    """Validate the complete 64-map training payload before it is uploaded."""
    root_path = Path(root).expanduser().resolve()
    index_path = root_path / "dataset.json"
    try:
        index = json.loads(index_path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{index_path}: invalid JSON: {exc}") from exc
    train = index.get("train") if isinstance(index, dict) else None
    if not isinstance(train, list) or not train:
        raise ValueError(f"{index_path}: train must be a nonempty list")

    levels = []
    for entry in train:
        if not isinstance(entry, dict):
            raise ValueError(f"{index_path}: every train entry must be an object")
        if entry.get("map_count") != TRAIN_MAPS_PER_CONDITION:
            raise ValueError(
                f"{index_path}: {entry.get('condition_id')!r} must declare "
                f"exactly {TRAIN_MAPS_PER_CONDITION} train maps"
            )
        level = _validate_level(root_path, entry)
        directory = root_path / level.maps_path

        metadata_path = directory / "dataset.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{metadata_path}: invalid JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path}: expected a JSON object")
        for count_field in ("slot_count", "num_maps"):
            count = metadata.get(count_field)
            if count_field in metadata and (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count != TRAIN_MAPS_PER_CONDITION
            ):
                raise ValueError(
                    f"{metadata_path}: {count_field} must be "
                    f"{TRAIN_MAPS_PER_CONDITION}"
                )

        expected_names = {
            f"img_{slot}.npy"
            for slot in range(1, TRAIN_MAPS_PER_CONDITION + 1)
        }
        for folder in RESET_ARRAY_FOLDERS:
            array_directory = directory / folder
            if not array_directory.is_dir() or array_directory.is_symlink():
                raise ValueError(
                    f"{array_directory}: expected a real reset-array directory"
                )
            entries = tuple(array_directory.iterdir())
            actual_names = {path.name for path in entries}
            if actual_names != expected_names or any(
                not path.is_file() or path.is_symlink() for path in entries
            ):
                raise ValueError(
                    f"{array_directory}: must contain exactly "
                    f"img_1.npy..img_{TRAIN_MAPS_PER_CONDITION}.npy"
                )
        levels.append(level)

    condition_ids = [level.condition_id for level in levels]
    maps_paths = [level.maps_path for level in levels]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"{index_path}: train repeats a condition_id")
    if len(maps_paths) != len(set(maps_paths)):
        raise ValueError(f"{index_path}: train repeats a maps_path")
    return TRAIN_MAPS_PER_CONDITION


def _validate_evaluation_panels(
    root: Path,
    panels: object,
    protocol_sha256: str,
) -> tuple[AcceptedPanel, ...]:
    required_panels = {"promotion", "development", "sealed"}
    if not isinstance(panels, dict) or set(panels) != required_panels:
        raise ValueError(
            f"{root / 'dataset.json'}: evaluation_panels must name exactly "
            f"{sorted(required_panels)}"
        )
    validated = []
    for name in sorted(required_panels):
        panel = panels[name]
        if not isinstance(panel, dict):
            raise ValueError(f"evaluation panel {name} must be an object")
        slot_count = panel.get("slot_count")
        condition_count = panel.get("conditions")
        if (
            not isinstance(slot_count, int)
            or isinstance(slot_count, bool)
            or slot_count <= 0
        ):
            raise ValueError(f"evaluation panel {name} has invalid slot_count")
        if (
            not isinstance(condition_count, int)
            or isinstance(condition_count, bool)
            or condition_count <= 0
        ):
            raise ValueError(f"evaluation panel {name} has invalid conditions")
        _, directory = _safe_relative_directory(root, panel.get("maps_path"))
        if not (directory / "dataset.json").is_file():
            raise FileNotFoundError(directory / "dataset.json")
        rows = _json_lines(directory / "manifest.jsonl")
        if len(rows) != slot_count:
            raise ValueError(
                f"evaluation panel {name} declares {slot_count} slots but "
                f"contains {len(rows)}"
            )
        if [row.get("slot_index") for row in rows] != list(
            range(1, slot_count + 1)
        ):
            raise ValueError(
                f"evaluation panel {name} slots must be contiguous "
                f"1..{slot_count}"
            )
        cells = {
            row.get("primary_cell")
            for row in rows
            if isinstance(row.get("primary_cell"), str)
        }
        if len(cells) != condition_count:
            raise ValueError(
                f"evaluation panel {name} declares {condition_count} "
                f"conditions but contains {len(cells)}"
            )
        for row in rows:
            for field in ("scenario_id", "episode_id"):
                value = row.get(field)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(
                        f"evaluation panel {name} has invalid {field}"
                    )
            reset_seed = row.get("reset_seed")
            if (
                not isinstance(reset_seed, int)
                or isinstance(reset_seed, bool)
                or not 0 <= reset_seed <= 2**32 - 1
            ):
                raise ValueError(
                    f"evaluation panel {name} has invalid reset_seed"
                )
            if row.get("environment_protocol_sha256") != protocol_sha256:
                raise ValueError(
                    f"evaluation panel {name} protocol does not match the bank"
                )
            expected_episode_id = _canonical_json_sha256(
                {
                    "schema": "terra_episode_id_v1",
                    "scenario_id": row["scenario_id"],
                    "reset_seed": reset_seed,
                    "environment_protocol_sha256": protocol_sha256,
                }
            )
            if row["episode_id"] != expected_episode_id:
                raise ValueError(
                    f"evaluation panel {name} has an invalid episode_id"
                )
        validated.append(
            AcceptedPanel(
                name=name,
                maps_path=panel["maps_path"],
                slot_count=slot_count,
                condition_count=condition_count,
            )
        )
    return tuple(validated)


def _selected(level: AcceptedLevel, arm: str) -> bool:
    if arm == "F-ANCHOR":
        return level.family == "foundation" and level.branch_depth == "Anchor"
    if arm == "T-ANCHOR":
        return level.family == "trench" and level.branch_depth == "Anchor"
    return True


def load_accepted_bank(
    root: str | Path,
    arm: str,
    terra_revision: str,
) -> AcceptedBank:
    """Validate the canonical index and select the levels owned by one arm."""
    if arm not in ARMS:
        raise ValueError(f"accepted-bank arm must be one of {ARMS}, got {arm!r}")
    if not isinstance(terra_revision, str) or not terra_revision.strip():
        raise ValueError("terra_revision must be an explicit nonempty string")
    if terra_revision != terra_revision.strip():
        raise ValueError("terra_revision must not have surrounding whitespace")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    index_path = root_path / "dataset.json"
    try:
        index = json.loads(index_path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{index_path}: invalid JSON: {exc}") from exc
    if not isinstance(index, dict) or index.get("schema") != SCHEMA:
        raise ValueError(f"{index_path} must use schema {SCHEMA!r}")
    if index.get("scenario_identity_contract") != SCENARIO_IDENTITY_CONTRACT:
        raise ValueError(
            f"{index_path}: scenario_identity_contract must be "
            f"{SCENARIO_IDENTITY_CONTRACT!r}"
        )

    protocol_sha256 = _sha256_field(
        index, "environment_protocol_sha256", index_path
    )
    if index.get("environment_protocol") != "environment_protocol.json":
        raise ValueError(
            f"{index_path}: environment_protocol must be "
            "'environment_protocol.json'"
        )
    protocol_path = root_path / "environment_protocol.json"
    try:
        protocol = json.loads(protocol_path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{protocol_path}: invalid JSON: {exc}") from exc
    if not isinstance(protocol, dict):
        raise ValueError(f"{protocol_path}: expected a JSON object")
    embedded_hash = protocol.get("environment_protocol_sha256")
    payload = {
        key: value
        for key, value in protocol.items()
        if key != "environment_protocol_sha256"
    }
    computed_hash = _canonical_json_sha256(payload)
    if embedded_hash != protocol_sha256 or computed_hash != protocol_sha256:
        raise ValueError(
            "accepted-bank environment protocol hash mismatch: "
            f"descriptor={protocol_sha256}, embedded={embedded_hash}, "
            f"computed={computed_hash}"
        )
    if protocol.get("terra_revision") != terra_revision:
        raise ValueError(
            "accepted-bank Terra revision mismatch: "
            f"bank={protocol.get('terra_revision')!r}, "
            f"runtime={terra_revision!r}"
        )
    current_protocol = _environment_protocol_for_revision(terra_revision)
    if protocol != current_protocol:
        raise ValueError(
            "accepted-bank environment protocol is stale for the imported "
            f"Terra code at frozen revision {terra_revision}"
        )

    registry_sha256 = _sha256_field(index, "source_registry_sha256", index_path)
    if index.get("source_registry") != "source_registry.jsonl":
        raise ValueError(
            f"{index_path}: source_registry must be 'source_registry.jsonl'"
        )
    registry_path = root_path / "source_registry.jsonl"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    actual_registry_sha256 = _sha256_file(registry_path)
    if actual_registry_sha256 != registry_sha256:
        raise ValueError(
            "accepted-bank source registry hash mismatch: "
            f"expected {registry_sha256}, got {actual_registry_sha256}"
        )

    train = index.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError(f"{index_path}: train must be a nonempty list")
    evaluation_panels = _validate_evaluation_panels(
        root_path,
        index.get("evaluation_panels"),
        protocol_sha256,
    )
    all_levels = [_validate_level(root_path, entry) for entry in train]
    condition_ids = [level.condition_id for level in all_levels]
    paths = [level.maps_path for level in all_levels]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"{index_path}: train repeats a condition_id")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{index_path}: train repeats a maps_path")

    selected = tuple(
        sorted(
            (level for level in all_levels if _selected(level, arm)),
            key=lambda level: level.condition_id,
        )
    )
    if not selected:
        raise ValueError(f"{index_path}: arm {arm} selects no accepted conditions")
    map_counts = {level.map_count for level in selected}
    if len(map_counts) != 1:
        raise ValueError(
            f"{index_path}: arm {arm} has unequal per-condition map counts "
            f"{sorted(map_counts)}; Terra levels must have one slot count"
        )
    map_count_per_condition = next(iter(map_counts))
    if map_count_per_condition != TRAIN_MAPS_PER_CONDITION:
        raise ValueError(
            f"{index_path}: arm {arm} must have exactly "
            f"{TRAIN_MAPS_PER_CONDITION} train maps per condition, got "
            f"{map_count_per_condition}"
        )

    return AcceptedBank(
        root=root_path,
        arm=arm,
        terra_revision=terra_revision,
        levels=selected,
        evaluation_panels=evaluation_panels,
        map_count_per_condition=map_count_per_condition,
        environment_protocol_sha256=protocol_sha256,
        source_registry_sha256=registry_sha256,
    )
