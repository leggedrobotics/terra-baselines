"""Load the frozen accepted-map bank used by the P5 curriculum experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "terra_curriculum_loader_bank_v1"
SCENARIO_IDENTITY_CONTRACT = "terra_reset_arrays_sha256_v1"
REVIEW_ADMISSION_SCHEMA = "terra-accepted-condition-set-v1"
DIAGNOSTIC_CONTROL_SCHEMA = "terra_unconstrained_control_bank_v1"
TRAIN96_RELEASE_ID = "terra_v6main_capfloor34_train96_v1"
TRAIN96_CAPABILITY_FLOOR_SCHEMA = "terra_train96_capability_floor_contract_v1"
TRAIN96_MAPS_PER_CONDITION = 96
TRAIN96_CONSTRAINED_CONDITION_COUNT = 32
TRAIN96_CAPABILITY_FLOOR_IDS = (
    "fnd-slab-allfree",
    "trn-straight-allfree",
)
REVIEW_RELEASE = "map-curriculum-diverse64-visual-review-20260730"
REVIEW_MANIFEST_SHA256 = (
    "39f7cd2e8ce565bd384de214da5f2eee5e76764cb554e149c0ba675d815d6d51"
)
REVIEW_DATA_SHA256 = "8404fcaa9a6b66949ade2b0225d3e7800968951953d2b6363aabffe38100cc0b"
ARMS = (
    "F-ANCHOR",
    "F-SPECIALIST",
    "T-ANCHOR",
    "T-SPECIALIST",
    "G-UNIFORM",
    "G-ADAPTIVE",
)
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
RESET_PRNG_CONTRACT = {
    "jax_default_prng_impl": "threefry2x32",
    "jax_threefry_partitionable": True,
}


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
    review_admission_sha256: str | None
    diagnostic_contract_sha256: str | None = None
    release_id: str | None = None
    capability_floor_contract_sha256: str | None = None
    constrained_condition_ids: tuple[str, ...] = ()
    capability_floor_condition_ids: tuple[str, ...] = ()
    capability_floor_evaluation_panels: tuple["AcceptedPanel", ...] = ()


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
        raise ValueError(f"{condition_id}: branch_depth must be one of {BRANCH_DEPTHS}")
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


def _validate_review_admission(
    root: Path,
    index: dict,
    condition_ids: list[str],
    *,
    path_field: str = "review_admission",
    sha256_field: str = "review_admission_sha256",
) -> str:
    index_path = root / "dataset.json"
    if index.get(path_field) != "review_admission.json":
        raise ValueError(f"{index_path}: {path_field} must be 'review_admission.json'")
    expected_sha256 = _sha256_field(
        index,
        sha256_field,
        index_path,
    )
    receipt_path = root / "review_admission.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError(receipt_path)
    actual_sha256 = _sha256_file(receipt_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "accepted-bank review admission hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{receipt_path}: invalid JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ValueError(f"{receipt_path}: expected a JSON object")
    if receipt.get("schema") != REVIEW_ADMISSION_SCHEMA:
        raise ValueError(f"{receipt_path}: schema must be {REVIEW_ADMISSION_SCHEMA!r}")
    pinned_identity = {
        "release": REVIEW_RELEASE,
        "manifest_sha256": REVIEW_MANIFEST_SHA256,
        "review_data_sha256": REVIEW_DATA_SHA256,
    }
    for field, expected in pinned_identity.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"{receipt_path}: {field} must match the pinned diverse-64 "
                f"review release: expected {expected!r}, got "
                f"{receipt.get(field)!r}"
            )
    accepted = receipt.get("accepted_conditions")
    if (
        not isinstance(accepted, list)
        or not all(isinstance(value, str) and value for value in accepted)
        or accepted != sorted(set(accepted))
    ):
        raise ValueError(
            f"{receipt_path}: accepted_conditions must be unique and sorted"
        )
    expected_conditions = sorted(condition_ids)
    if accepted != expected_conditions:
        raise ValueError(
            f"{receipt_path}: accepted_conditions do not match train "
            f"condition IDs: accepted={accepted}, train={expected_conditions}"
        )
    return expected_sha256


def _validate_diagnostic_control(
    root: Path,
    index: dict,
    condition_ids: list[str],
) -> str:
    """Validate an evaluation-only control bank without admitting it to training."""
    index_path = root / "dataset.json"
    if index.get("control_schema") != DIAGNOSTIC_CONTROL_SCHEMA:
        raise ValueError(
            f"{index_path}: diagnostic control schema must be "
            f"{DIAGNOSTIC_CONTROL_SCHEMA!r}"
        )
    if index.get("included_in_constrained_macro") is not False:
        raise ValueError(
            f"{index_path}: diagnostic controls must be excluded from the "
            "constrained benchmark macro"
        )
    if index.get("control_contract") != "control_contract.json":
        raise ValueError(
            f"{index_path}: control_contract must be 'control_contract.json'"
        )
    expected_sha256 = _sha256_field(
        index,
        "control_contract_sha256",
        index_path,
    )
    contract_path = root / "control_contract.json"
    if not contract_path.is_file() or contract_path.is_symlink():
        raise FileNotFoundError(contract_path)
    actual_sha256 = _sha256_file(contract_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "diagnostic control contract hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        contract = json.loads(contract_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{contract_path}: invalid JSON: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("schema") != (
        DIAGNOSTIC_CONTROL_SCHEMA
    ):
        raise ValueError(
            f"{contract_path}: schema must be {DIAGNOSTIC_CONTROL_SCHEMA!r}"
        )
    if contract.get("included_in_constrained_macro") is not False:
        raise ValueError(
            f"{contract_path}: diagnostic controls must not enter the main macro"
        )
    conditions = contract.get("conditions")
    if not isinstance(conditions, dict) or sorted(conditions) != sorted(condition_ids):
        raise ValueError(
            f"{contract_path}: conditions do not match train condition IDs"
        )
    return expected_sha256


def _sorted_unique_strings(
    payload: dict,
    field: str,
    source: Path,
) -> tuple[str, ...]:
    values = payload.get(field)
    if (
        not isinstance(values, list)
        or not all(isinstance(value, str) and value for value in values)
        or values != sorted(set(values))
    ):
        raise ValueError(f"{source}: {field} must be unique and sorted")
    return tuple(values)


def _panel_count_contract(panels: dict) -> dict:
    return {
        name: {
            "conditions": panel.get("conditions"),
            "slot_count": panel.get("slot_count"),
        }
        for name, panel in sorted(panels.items())
    }


def _validate_train96_release(
    root: Path,
    index: dict,
    levels: list[AcceptedLevel],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Validate the one named 34-condition Train-96 map-support treatment."""
    index_path = root / "dataset.json"
    if index.get("release_id") != TRAIN96_RELEASE_ID:
        raise ValueError(f"{index_path}: release_id must be {TRAIN96_RELEASE_ID!r}")
    if index.get("train_maps_per_condition") != TRAIN96_MAPS_PER_CONDITION:
        raise ValueError(
            f"{index_path}: train_maps_per_condition must be "
            f"{TRAIN96_MAPS_PER_CONDITION}"
        )
    constrained_ids = _sorted_unique_strings(
        index,
        "constrained_condition_ids",
        index_path,
    )
    capability_floor_ids = _sorted_unique_strings(
        index,
        "capability_floor_condition_ids",
        index_path,
    )
    if len(constrained_ids) != TRAIN96_CONSTRAINED_CONDITION_COUNT:
        raise ValueError(
            f"{index_path}: constrained_condition_ids must contain exactly "
            f"{TRAIN96_CONSTRAINED_CONDITION_COUNT} conditions"
        )
    if capability_floor_ids != TRAIN96_CAPABILITY_FLOOR_IDS:
        raise ValueError(
            f"{index_path}: capability_floor_condition_ids must be exactly "
            f"{list(TRAIN96_CAPABILITY_FLOOR_IDS)!r}"
        )
    if index.get("included_in_constrained_macro") != list(constrained_ids):
        raise ValueError(
            f"{index_path}: included_in_constrained_macro must list exactly "
            "the constrained conditions"
        )
    if set(constrained_ids) & set(capability_floor_ids):
        raise ValueError(f"{index_path}: condition partitions must be disjoint")
    level_ids = {level.condition_id for level in levels}
    if level_ids != set(constrained_ids) | set(capability_floor_ids):
        raise ValueError(
            f"{index_path}: condition partitions must cover all train conditions"
        )
    capability_families = {
        level.condition_id: level.family
        for level in levels
        if level.condition_id in capability_floor_ids
    }
    if capability_families != {
        "fnd-slab-allfree": "foundation",
        "trn-straight-allfree": "trench",
    }:
        raise ValueError(
            f"{index_path}: capability-floor families do not match the frozen pair"
        )

    review_sha256 = _validate_review_admission(
        root,
        index,
        list(constrained_ids),
        path_field="constrained_review_admission",
        sha256_field="constrained_review_admission_sha256",
    )

    if index.get("capability_floor_contract") != "capability_floor_contract.json":
        raise ValueError(
            f"{index_path}: capability_floor_contract must be "
            "'capability_floor_contract.json'"
        )
    expected_contract_sha256 = _sha256_field(
        index,
        "capability_floor_contract_sha256",
        index_path,
    )
    contract_path = root / "capability_floor_contract.json"
    if not contract_path.is_file() or contract_path.is_symlink():
        raise FileNotFoundError(contract_path)
    actual_contract_sha256 = _sha256_file(contract_path)
    if actual_contract_sha256 != expected_contract_sha256:
        raise ValueError(
            "Train-96 capability-floor contract hash mismatch: "
            f"expected {expected_contract_sha256}, got {actual_contract_sha256}"
        )
    try:
        contract = json.loads(contract_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{contract_path}: invalid JSON: {exc}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"{contract_path}: expected a JSON object")
    expected_contract = {
        "schema": TRAIN96_CAPABILITY_FLOOR_SCHEMA,
        "release_id": TRAIN96_RELEASE_ID,
        "included_in_constrained_macro": False,
        "constrained_condition_ids": list(constrained_ids),
        "capability_floor_condition_ids": list(capability_floor_ids),
        "train_maps_per_condition": TRAIN96_MAPS_PER_CONDITION,
        "evaluation_panels": {
            "constrained": _panel_count_contract(index.get("evaluation_panels", {})),
            "capability_floor": _panel_count_contract(
                index.get("capability_floor_evaluation_panels", {})
            ),
        },
    }
    if contract != expected_contract:
        raise ValueError(
            f"{contract_path}: contract does not match the frozen Train-96 release"
        )
    return review_sha256, constrained_ids, capability_floor_ids


def _train_maps_per_condition(index: dict, source: Path) -> int:
    """Return the one explicitly supported train-bank slot contract."""
    release_id = index.get("release_id")
    if release_id is None:
        return TRAIN_MAPS_PER_CONDITION
    if release_id != TRAIN96_RELEASE_ID:
        raise ValueError(f"{source}: unsupported release_id {release_id!r}")
    count = index.get("train_maps_per_condition")
    if count != TRAIN96_MAPS_PER_CONDITION:
        raise ValueError(
            f"{source}: {TRAIN96_RELEASE_ID} must declare "
            f"train_maps_per_condition={TRAIN96_MAPS_PER_CONDITION}"
        )
    return TRAIN96_MAPS_PER_CONDITION


def validate_staged_training_bank(
    root: str | Path,
    *,
    expected_maps_per_condition: int | None = None,
    expected_release_id: str | None = None,
) -> int:
    """Validate one complete, explicitly versioned training payload."""
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
    maps_per_condition = _train_maps_per_condition(index, index_path)
    if expected_maps_per_condition is not None and (
        maps_per_condition != expected_maps_per_condition
    ):
        raise ValueError(
            f"{index_path}: expected {expected_maps_per_condition} maps per "
            f"condition, got {maps_per_condition}"
        )
    if expected_release_id is not None and index.get("release_id") != (
        expected_release_id
    ):
        raise ValueError(
            f"{index_path}: expected release_id {expected_release_id!r}, got "
            f"{index.get('release_id')!r}"
        )

    levels = []
    for entry in train:
        if not isinstance(entry, dict):
            raise ValueError(f"{index_path}: every train entry must be an object")
        if entry.get("map_count") != maps_per_condition:
            raise ValueError(
                f"{index_path}: {entry.get('condition_id')!r} must declare "
                f"exactly {maps_per_condition} train maps"
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
        slot_count = metadata.get("slot_count")
        if (
            not isinstance(slot_count, int)
            or isinstance(slot_count, bool)
            or slot_count != maps_per_condition
        ):
            raise ValueError(
                f"{metadata_path}: slot_count must be {maps_per_condition}"
            )
        num_maps = metadata.get("num_maps")
        if "num_maps" in metadata and (
            not isinstance(num_maps, int)
            or isinstance(num_maps, bool)
            or num_maps != maps_per_condition
        ):
            raise ValueError(f"{metadata_path}: num_maps must be {maps_per_condition}")

        expected_names = {
            f"img_{slot}.npy" for slot in range(1, maps_per_condition + 1)
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
                    f"img_1.npy..img_{maps_per_condition}.npy"
                )
        levels.append(level)

    condition_ids = [level.condition_id for level in levels]
    maps_paths = [level.maps_path for level in levels]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"{index_path}: train repeats a condition_id")
    if len(maps_paths) != len(set(maps_paths)):
        raise ValueError(f"{index_path}: train repeats a maps_path")
    if index.get("release_id") == TRAIN96_RELEASE_ID:
        _validate_train96_release(root_path, index, levels)
    else:
        _validate_review_admission(root_path, index, condition_ids)
    return maps_per_condition


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
        if [row.get("slot_index") for row in rows] != list(range(1, slot_count + 1)):
            raise ValueError(
                f"evaluation panel {name} slots must be contiguous " f"1..{slot_count}"
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
                    raise ValueError(f"evaluation panel {name} has invalid {field}")
            reset_seed = row.get("reset_seed")
            if (
                not isinstance(reset_seed, int)
                or isinstance(reset_seed, bool)
                or not 0 <= reset_seed <= 2**32 - 1
            ):
                raise ValueError(f"evaluation panel {name} has invalid reset_seed")
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
                raise ValueError(f"evaluation panel {name} has an invalid episode_id")
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
    if arm == "F-SPECIALIST":
        return level.family == "foundation"
    if arm == "T-ANCHOR":
        return level.family == "trench" and level.branch_depth == "Anchor"
    if arm == "T-SPECIALIST":
        return level.family == "trench"
    return True


def load_accepted_bank(
    root: str | Path,
    arm: str,
    terra_revision: str,
    *,
    allow_diagnostic_control: bool = False,
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

    protocol_sha256 = _sha256_field(index, "environment_protocol_sha256", index_path)
    if index.get("environment_protocol") != "environment_protocol.json":
        raise ValueError(
            f"{index_path}: environment_protocol must be " "'environment_protocol.json'"
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
    if protocol.get("reset_prng") != RESET_PRNG_CONTRACT:
        raise ValueError(
            "accepted-bank reset PRNG contract mismatch: "
            f"expected {RESET_PRNG_CONTRACT!r}, got "
            f"{protocol.get('reset_prng')!r}"
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
    release_id = index.get("release_id")
    capability_floor_evaluation_panels = ()
    if release_id == TRAIN96_RELEASE_ID:
        capability_floor_evaluation_panels = _validate_evaluation_panels(
            root_path,
            index.get("capability_floor_evaluation_panels"),
            protocol_sha256,
        )
    all_levels = [_validate_level(root_path, entry) for entry in train]
    condition_ids = [level.condition_id for level in all_levels]
    paths = [level.maps_path for level in all_levels]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"{index_path}: train repeats a condition_id")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{index_path}: train repeats a maps_path")
    diagnostic_contract_sha256 = None
    capability_floor_contract_sha256 = None
    constrained_condition_ids: tuple[str, ...] = ()
    capability_floor_condition_ids: tuple[str, ...] = ()
    if allow_diagnostic_control:
        if release_id == TRAIN96_RELEASE_ID:
            raise ValueError(
                "Train-96 capability-floor conditions are training support; "
                "use the separate diagnostic control bank for diagnostic panels"
            )
        review_admission_sha256 = None
        diagnostic_contract_sha256 = _validate_diagnostic_control(
            root_path,
            index,
            condition_ids,
        )
    elif release_id == TRAIN96_RELEASE_ID:
        (
            review_admission_sha256,
            constrained_condition_ids,
            capability_floor_condition_ids,
        ) = _validate_train96_release(root_path, index, all_levels)
        capability_floor_contract_sha256 = _sha256_field(
            index,
            "capability_floor_contract_sha256",
            index_path,
        )
    else:
        review_admission_sha256 = _validate_review_admission(
            root_path,
            index,
            condition_ids,
        )

    selected = tuple(
        sorted(
            (level for level in all_levels if _selected(level, arm)),
            key=lambda level: level.condition_id,
        )
    )
    if not selected:
        raise ValueError(f"{index_path}: arm {arm} selects no accepted conditions")
    if arm in ("F-SPECIALIST", "T-SPECIALIST"):
        family = "foundation" if arm == "F-SPECIALIST" else "trench"
        anchor_ids = {
            level.condition_id
            for level in all_levels
            if level.family == family and level.branch_depth == "Anchor"
        }
        selected_ids = {level.condition_id for level in selected}
        if selected_ids == anchor_ids:
            raise ValueError(
                f"{index_path}: arm {arm} has no accepted non-anchor conditions; "
                "running it would duplicate the family anchor control"
            )
    map_counts = {level.map_count for level in selected}
    if len(map_counts) != 1:
        raise ValueError(
            f"{index_path}: arm {arm} has unequal per-condition map counts "
            f"{sorted(map_counts)}; Terra levels must have one slot count"
        )
    map_count_per_condition = next(iter(map_counts))
    expected_maps_per_condition = _train_maps_per_condition(index, index_path)
    if map_count_per_condition != expected_maps_per_condition:
        raise ValueError(
            f"{index_path}: arm {arm} must have exactly "
            f"{expected_maps_per_condition} train maps per condition, got "
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
        review_admission_sha256=review_admission_sha256,
        diagnostic_contract_sha256=diagnostic_contract_sha256,
        release_id=release_id,
        capability_floor_contract_sha256=capability_floor_contract_sha256,
        constrained_condition_ids=constrained_condition_ids,
        capability_floor_condition_ids=capability_floor_condition_ids,
        capability_floor_evaluation_panels=capability_floor_evaluation_panels,
    )
