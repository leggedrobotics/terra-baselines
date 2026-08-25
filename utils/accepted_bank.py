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
V8_RELEASE_ID = "terra_v8_v6_constraints_v7_adjacent_train96_v5"
AXIS_V2_RELEASE_ID = "terra_axis_v2_v6_constraints_v7_foundations_train96_v1"
V8_REVIEW_ADMISSION_SCHEMA = "terra_v8_review_admission_v1"
AXIS_V2_REVIEW_ADMISSION_SCHEMA = "terra_axis_v2_review_admission_v1"
V8_TRAINING_MIXTURE_SCHEMA = "terra_v8_training_mixture_v4"
V8_TRAINING_MIXTURE_SHA256 = (
    "f2a2a33556d513b46193a8a3996d37e6989534eba9373f46f52d79f956ac128e"
)
V8_AUDIT_SHA256 = "b5cc702bc049d26c1924fcb2c2bee54377b4c209518c87102be395270eb4965b"
V8_MAPS_PER_CONDITION = 96
V8_CONSTRAINT_CONDITION_COUNT = 32
V8_CORE_CONDITION_COUNT = 13
V8_MAIN_CONDITION_COUNT = 45
V8_CAPABILITY_FLOOR_IDS = TRAIN96_CAPABILITY_FLOOR_IDS
AXIS_V2_CONSTRAINT_CONDITION_COUNT = 32
AXIS_V2_CORE_CONDITION_COUNT = 6
AXIS_V2_MAIN_CONDITION_COUNT = 38
AXIS_V2_TRAIN_CONDITION_COUNT = 40
AXIS_V2_MAPS_PER_CONDITION = 96
V8_CURRICULUM_STAGES = ("capability", "nearby", "full")
V8_SAMPLER_PROFILES = (
    "bank_v4",
    "bounded_replay25_v1",
    "banded_preview15_v1",
    "continuous_banded_v3",
)
V8_CONTINUOUS_PROFILES = ("continuous_banded_v3",)
V8_CONDITION_PROFILES = (
    "full",
    "trench_aligned_37_v1",
    "axis_v2_40_v1",
)
SPARSE_CONDITION_PROFILES = ("trench_aligned_37_v1", "axis_v2_40_v1")
V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS = (
    "trn-net4-side1-road",
    "trn-net4-side2",
    "trn-net4-side2-s",
    "v7-trn-cross-adjacent",
    "v7-trn-disconnected-pair-adjacent",
    "v7-trn-dogleg-adjacent",
    "v7-trn-double-t-adjacent",
    "v7-trn-network3-adjacent",
    "v7-trn-straight-adjacent",
    "v7-trn-tee-adjacent",
)
V8_TRENCH_ALIGNED_EVALUATION_FAMILY = "gate_supported_main"
V8_CONTINUOUS_GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "v8_continuous_banded_graph_v1.json"
)
AXIS_V2_CONTINUOUS_GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "axis_v2_continuous_banded_graph_v1.json"
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
BRANCH_DEPTHS = ("Anchor", "Nearby core", "One-axis", "Composed")
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
    curriculum_stage: str | None = None
    sampler_profile: str | None = None
    sampling_probabilities: tuple[float, ...] = ()
    v6_constraint_condition_ids: tuple[str, ...] = ()
    v7_core_condition_ids: tuple[str, ...] = ()
    curriculum_depths: tuple[int, ...] = ()
    curriculum_graph_sha256: str | None = None
    condition_profile: str = "full"
    # Which evaluation/<family>/* directories `evaluation_panels` point at.
    evaluation_panel_family: str = "main"

    def __getattr__(self, name: str):
        # Checkpoints written before sampler profiles were named unpickle into
        # this class without the new field in their instance dictionary.
        if name == "sampler_profile":
            return None
        if name == "evaluation_panel_family":
            return "main"
        if name == "condition_profile":
            return "full"
        raise AttributeError(name)


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


def _unique_strings(payload: dict, field: str, source: Path) -> tuple[str, ...]:
    values = payload.get(field)
    if (
        not isinstance(values, list)
        or not all(isinstance(value, str) and value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{source}: {field} must contain unique nonempty strings")
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


def _validate_v8_review_admission(
    root: Path,
    index: dict,
    condition_ids: list[str],
    *,
    release_id: str = V8_RELEASE_ID,
    schema: str = V8_REVIEW_ADMISSION_SCHEMA,
) -> str:
    index_path = root / "dataset.json"
    if index.get("review_admission") != "review_admission.json":
        raise ValueError(
            f"{index_path}: review_admission must be 'review_admission.json'"
        )
    expected_sha256 = _sha256_field(
        index,
        "review_admission_sha256",
        index_path,
    )
    receipt_path = root / "review_admission.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError(receipt_path)
    actual_sha256 = _sha256_file(receipt_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "V8 review admission hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{receipt_path}: invalid JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ValueError(f"{receipt_path}: expected a JSON object")
    expected_fields = {
        "schema": schema,
        "release_id": release_id,
        "decision": "accept",
        "decision_source": "explicit_user_instruction",
    }
    for field, expected in expected_fields.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"{receipt_path}: {field} must be {expected!r}, "
                f"got {receipt.get(field)!r}"
            )
    accepted = receipt.get("accepted_conditions")
    if accepted != sorted(condition_ids):
        raise ValueError(
            f"{receipt_path}: accepted_conditions do not match V8 training support"
        )
    _sha256_field(receipt, "candidate_dataset_sha256", receipt_path)
    return expected_sha256


def _validate_v8_release(
    root: Path,
    index: dict,
    levels: list[AcceptedLevel],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], dict]:
    """Validate the frozen V8 support, review, audit, and stage contract."""
    index_path = root / "dataset.json"
    if index.get("release_id") != V8_RELEASE_ID:
        raise ValueError(f"{index_path}: release_id must be {V8_RELEASE_ID!r}")
    if index.get("train_maps_per_condition") != V8_MAPS_PER_CONDITION:
        raise ValueError(
            f"{index_path}: train_maps_per_condition must be "
            f"{V8_MAPS_PER_CONDITION}"
        )

    constraint_ids = _sorted_unique_strings(
        index,
        "v6_constraint_condition_ids",
        index_path,
    )
    capability_ids = _unique_strings(
        index,
        "v6_capability_floor_condition_ids",
        index_path,
    )
    core_ids = _unique_strings(index, "v7_core_condition_ids", index_path)
    if len(constraint_ids) != V8_CONSTRAINT_CONDITION_COUNT:
        raise ValueError(
            f"{index_path}: V8 must contain {V8_CONSTRAINT_CONDITION_COUNT} "
            "V6 constraint conditions"
        )
    if capability_ids != V8_CAPABILITY_FLOOR_IDS:
        raise ValueError(
            f"{index_path}: V8 capability controls must be "
            f"{list(V8_CAPABILITY_FLOOR_IDS)!r}"
        )
    if len(core_ids) != V8_CORE_CONDITION_COUNT:
        raise ValueError(
            f"{index_path}: V8 must contain {V8_CORE_CONDITION_COUNT} V7 core "
            "conditions"
        )
    partitions = (set(constraint_ids), set(capability_ids), set(core_ids))
    if any(
        left & right
        for i, left in enumerate(partitions)
        for right in partitions[i + 1 :]
    ):
        raise ValueError(f"{index_path}: V8 condition partitions must be disjoint")
    level_ids = {level.condition_id for level in levels}
    if level_ids != set().union(*partitions):
        raise ValueError(f"{index_path}: V8 partitions must cover all train levels")
    main_ids = tuple(sorted(set(constraint_ids) | set(core_ids)))
    if len(main_ids) != V8_MAIN_CONDITION_COUNT:
        raise ValueError(
            f"{index_path}: V8 main macro must contain {V8_MAIN_CONDITION_COUNT} "
            "conditions"
        )
    if index.get("included_in_main_macro") != list(main_ids):
        raise ValueError(
            f"{index_path}: included_in_main_macro must be the sorted V8 main set"
        )

    review_sha256 = _validate_v8_review_admission(
        root,
        index,
        [level.condition_id for level in levels],
    )
    if index.get("audit_receipt") != "audit_receipt.json":
        raise ValueError(f"{index_path}: audit_receipt must be 'audit_receipt.json'")
    audit_sha256 = _sha256_field(index, "audit_receipt_sha256", index_path)
    audit_path = root / "audit_receipt.json"
    if audit_sha256 != V8_AUDIT_SHA256 or _sha256_file(audit_path) != audit_sha256:
        raise ValueError(f"{audit_path}: V8 audit receipt hash mismatch")

    if index.get("training_mixture") != "training_mixture.json":
        raise ValueError(
            f"{index_path}: training_mixture must be 'training_mixture.json'"
        )
    mixture_sha256 = _sha256_field(index, "training_mixture_sha256", index_path)
    mixture_path = root / "training_mixture.json"
    if (
        mixture_sha256 != V8_TRAINING_MIXTURE_SHA256
        or _sha256_file(mixture_path) != mixture_sha256
    ):
        raise ValueError(f"{mixture_path}: V8 training mixture hash mismatch")
    try:
        mixture = json.loads(mixture_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{mixture_path}: invalid JSON: {exc}") from exc
    if not isinstance(mixture, dict) or mixture.get("schema") != (
        V8_TRAINING_MIXTURE_SCHEMA
    ):
        raise ValueError(
            f"{mixture_path}: schema must be {V8_TRAINING_MIXTURE_SCHEMA!r}"
        )
    expected_protocol = {
        "accepted_dump_contract": "exact_visible_dump_v1",
        "apply_trench_rewards": False,
        "full_resets": True,
        "max_steps_in_episode": 450,
        "rewards_type": "DENSE",
    }
    if mixture.get("fixed_protocol") != expected_protocol:
        raise ValueError(f"{mixture_path}: V8 fixed protocol changed")
    stages = mixture.get("stages")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages] != [
        "capability_anchors",
        "nearby_geometry_core",
        "constraint_branches",
    ]:
        raise ValueError(f"{mixture_path}: V8 stage graph changed")
    if stages[0].get("new_conditions") != list(capability_ids):
        raise ValueError(f"{mixture_path}: capability stage support changed")
    if stages[1].get("new_conditions") != list(core_ids):
        raise ValueError(f"{mixture_path}: nearby stage support changed")
    if stages[2].get("new_conditions") != list(constraint_ids):
        raise ValueError(f"{mixture_path}: constraint stage support changed")
    return review_sha256, constraint_ids, capability_ids, core_ids, mixture


def _validate_axis_v2_release(
    root: Path,
    index: dict,
    levels: list[AcceptedLevel],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], dict]:
    """Validate the exact 25-foundation + 15-trench axis-v2 bank."""

    index_path = root / "dataset.json"
    if index.get("release_id") != AXIS_V2_RELEASE_ID:
        raise ValueError(
            f"{index_path}: release_id must be {AXIS_V2_RELEASE_ID!r}"
        )
    if index.get("train_maps_per_condition") != AXIS_V2_MAPS_PER_CONDITION:
        raise ValueError(
            f"{index_path}: train_maps_per_condition must be "
            f"{AXIS_V2_MAPS_PER_CONDITION}"
        )

    constraint_ids = _sorted_unique_strings(
        index, "v6_constraint_condition_ids", index_path
    )
    capability_ids = _unique_strings(
        index, "v6_capability_floor_condition_ids", index_path
    )
    core_ids = _unique_strings(index, "v7_core_condition_ids", index_path)
    if len(constraint_ids) != AXIS_V2_CONSTRAINT_CONDITION_COUNT:
        raise ValueError(
            f"{index_path}: axis-v2 must contain "
            f"{AXIS_V2_CONSTRAINT_CONDITION_COUNT} V6 conditions"
        )
    if capability_ids != V8_CAPABILITY_FLOOR_IDS:
        raise ValueError(
            f"{index_path}: axis-v2 capability controls must be "
            f"{list(V8_CAPABILITY_FLOOR_IDS)!r}"
        )
    if len(core_ids) != AXIS_V2_CORE_CONDITION_COUNT or any(
        not condition_id.startswith("v7-fnd-") for condition_id in core_ids
    ):
        raise ValueError(
            f"{index_path}: axis-v2 must contain exactly six V7 foundation "
            "core conditions"
        )

    partitions = (set(constraint_ids), set(capability_ids), set(core_ids))
    if any(
        left & right
        for index_, left in enumerate(partitions)
        for right in partitions[index_ + 1 :]
    ):
        raise ValueError(f"{index_path}: axis-v2 partitions must be disjoint")
    level_ids = {level.condition_id for level in levels}
    if level_ids != set().union(*partitions):
        raise ValueError(
            f"{index_path}: axis-v2 partitions must cover all train levels"
        )
    family_counts = {
        family: sum(level.family == family for level in levels)
        for family in FAMILIES
    }
    if len(levels) != AXIS_V2_TRAIN_CONDITION_COUNT or family_counts != {
        "foundation": 25,
        "trench": 15,
    }:
        raise ValueError(
            f"{index_path}: axis-v2 must contain 25 foundation and 15 trench "
            f"conditions; got {family_counts}"
        )
    main_ids = tuple(sorted(set(constraint_ids) | set(core_ids)))
    if len(main_ids) != AXIS_V2_MAIN_CONDITION_COUNT:
        raise ValueError(
            f"{index_path}: axis-v2 main macro must contain "
            f"{AXIS_V2_MAIN_CONDITION_COUNT} conditions"
        )
    if index.get("included_in_main_macro") != list(main_ids):
        raise ValueError(
            f"{index_path}: included_in_main_macro must be the sorted axis-v2 set"
        )

    review_sha256 = _validate_v8_review_admission(
        root,
        index,
        [level.condition_id for level in levels],
        release_id=AXIS_V2_RELEASE_ID,
        schema=AXIS_V2_REVIEW_ADMISSION_SCHEMA,
    )

    if index.get("audit_receipt") != "audit_receipt.json":
        raise ValueError(f"{index_path}: audit_receipt must be 'audit_receipt.json'")
    audit_path = root / "audit_receipt.json"
    audit_sha256 = _sha256_field(index, "audit_receipt_sha256", index_path)
    if _sha256_file(audit_path) != audit_sha256:
        raise ValueError(f"{audit_path}: axis-v2 audit receipt hash mismatch")
    try:
        audit = json.loads(audit_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{audit_path}: invalid JSON: {exc}") from exc
    expected_audit = {
        "schema": "terra_axis_v2_generalist_bank_audit_v1",
        "accepted": True,
        "owner_contract": "generator_owner_bits_v1",
        "failed_maps": 0,
        "foundation_conditions": 25,
        "trench_conditions": 15,
    }
    for field, expected in expected_audit.items():
        if audit.get(field) != expected:
            raise ValueError(
                f"{audit_path}: {field} must be {expected!r}, "
                f"got {audit.get(field)!r}"
            )

    if index.get("training_mixture") != "training_mixture.json":
        raise ValueError(
            f"{index_path}: training_mixture must be 'training_mixture.json'"
        )
    mixture_path = root / "training_mixture.json"
    mixture_sha256 = _sha256_field(index, "training_mixture_sha256", index_path)
    if _sha256_file(mixture_path) != mixture_sha256:
        raise ValueError(f"{mixture_path}: axis-v2 training mixture hash mismatch")
    try:
        mixture = json.loads(mixture_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{mixture_path}: invalid JSON: {exc}") from exc
    if not isinstance(mixture, dict) or mixture.get("schema") != (
        "terra_axis_v2_training_mixture_v1"
    ):
        raise ValueError(
            f"{mixture_path}: schema must be 'terra_axis_v2_training_mixture_v1'"
        )
    expected_protocol = {
        "accepted_dump_contract": "exact_visible_dump_v1",
        "apply_trench_rewards": False,
        "full_resets": True,
        "max_steps_in_episode": 450,
        "rewards_type": "DENSE",
    }
    if mixture.get("fixed_protocol") != expected_protocol:
        raise ValueError(f"{mixture_path}: axis-v2 fixed protocol changed")
    stages = mixture.get("stages")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages] != [
        "capability_anchors",
        "nearby_geometry_core",
        "constraint_branches",
    ]:
        raise ValueError(f"{mixture_path}: axis-v2 stage graph changed")
    if stages[0].get("new_conditions") != list(capability_ids):
        raise ValueError(f"{mixture_path}: capability stage support changed")
    if stages[1].get("new_conditions") != list(core_ids):
        raise ValueError(f"{mixture_path}: nearby stage support changed")
    if stages[2].get("new_conditions") != list(constraint_ids):
        raise ValueError(f"{mixture_path}: constraint stage support changed")
    return review_sha256, constraint_ids, capability_ids, core_ids, mixture


def _v8_stage_selection(
    levels: list[AcceptedLevel],
    stage: str,
    constraint_ids: tuple[str, ...],
    capability_ids: tuple[str, ...],
    core_ids: tuple[str, ...],
    mixture: dict,
    sampler_profile: str = "bank_v4",
    condition_profile: str = "full",
) -> tuple[tuple[AcceptedLevel, ...], tuple[float, ...]]:
    if stage not in V8_CURRICULUM_STAGES:
        raise ValueError(
            f"V8 curriculum_stage must be one of {V8_CURRICULUM_STAGES}, "
            f"got {stage!r}"
        )
    if sampler_profile not in V8_SAMPLER_PROFILES:
        raise ValueError(
            f"V8 sampler_profile must be one of {V8_SAMPLER_PROFILES}, "
            f"got {sampler_profile!r}"
        )
    if condition_profile not in V8_CONDITION_PROFILES:
        raise ValueError(
            f"V8 condition_profile must be one of {V8_CONDITION_PROFILES}, "
            f"got {condition_profile!r}"
        )
    if condition_profile != "full" and (
        stage != "full" or sampler_profile not in V8_CONTINUOUS_PROFILES
    ):
        raise ValueError(
            f"{condition_profile} requires the full stage and a continuous sampler"
        )
    by_id = {level.condition_id: level for level in levels}
    if sampler_profile in V8_CONTINUOUS_PROFILES and stage != "full":
        raise ValueError(
            f"{sampler_profile} requires full support; use "
            "curriculum_stage='full' only as a bank-selection compatibility flag"
        )
    if stage == "capability":
        selected_ids = set(capability_ids)
        slice_mass = {"capability": 1.0, "core": 0.0, "constraint": 0.0}
    elif stage == "nearby":
        if sampler_profile == "bank_v4":
            selected_ids = set(capability_ids) | set(core_ids)
            slice_mass = {"capability": 0.5, "core": 0.5, "constraint": 0.0}
        elif sampler_profile == "bounded_replay25_v1":
            selected_ids = set(capability_ids) | set(core_ids)
            slice_mass = {"capability": 0.25, "core": 0.75, "constraint": 0.0}
        else:
            selected_ids = set(capability_ids) | set(core_ids) | set(constraint_ids)
            slice_mass = {"capability": 0.10, "core": 0.75, "constraint": 0.15}
    else:
        if sampler_profile == "banded_preview15_v1":
            raise ValueError(
                "banded_preview15_v1 is a nearby-stage treatment; use "
                "bounded_replay25_v1 for the full stage"
            )
        selected_ids = set(capability_ids) | set(core_ids) | set(constraint_ids)
        if sampler_profile == "bank_v4":
            slice_mass = {"capability": 0.25, "core": 0.25, "constraint": 0.5}
        elif sampler_profile in V8_CONTINUOUS_PROFILES:
            # Selection is full-support; the live sampler owns the dynamic
            # probability vector and therefore consumes no frozen weights.
            slice_mass = {"capability": 1.0, "core": 1.0, "constraint": 1.0}
        else:
            slice_mass = {"capability": 0.0625, "core": 0.1875, "constraint": 0.75}
    if condition_profile == "trench_aligned_37_v1":
        excluded = set(V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS)
        missing = sorted(excluded - selected_ids)
        if missing:
            raise ValueError(
                "trench-aligned profile exclusions are absent from the canonical "
                f"V8 support: {missing}"
            )
        selected_ids -= excluded
    selected = tuple(
        sorted((by_id[name] for name in selected_ids), key=lambda x: x.condition_id)
    )
    if condition_profile == "trench_aligned_37_v1":
        family_counts = {
            family: sum(level.family == family for level in selected)
            for family in FAMILIES
        }
        if len(selected) != 37 or family_counts != {
            "foundation": 25,
            "trench": 12,
        }:
            raise ValueError(
                "trench-aligned profile must select 25 foundation and 12 trench "
                f"conditions; got {family_counts}"
            )
    elif condition_profile == "axis_v2_40_v1":
        family_counts = {
            family: sum(level.family == family for level in selected)
            for family in FAMILIES
        }
        if len(selected) != AXIS_V2_TRAIN_CONDITION_COUNT or family_counts != {
            "foundation": 25,
            "trench": 15,
        }:
            raise ValueError(
                "axis-v2 profile must select 25 foundation and 15 trench "
                f"conditions; got {family_counts}"
            )

    geometry_mass = mixture.get("v7_geometry_mass_within_family")
    if not isinstance(geometry_mass, dict):
        raise ValueError("V8 training mixture lacks geometry masses")
    for family in FAMILIES:
        values = geometry_mass.get(family)
        if not isinstance(values, dict) or abs(sum(values.values()) - 1.0) > 1e-12:
            raise ValueError(f"V8 {family} geometry masses must sum to one")

    constraint_family_counts = {
        family: sum(
            by_id[condition_id].family == family for condition_id in constraint_ids
        )
        for family in FAMILIES
    }

    def probability(level: AcceptedLevel) -> float:
        family_mass = 0.5
        if level.condition_id in capability_ids:
            return slice_mass["capability"] * family_mass
        if level.condition_id in core_ids:
            prefix = "v7-fnd-" if level.family == "foundation" else "v7-trn-"
            if not level.condition_id.startswith(
                prefix
            ) or not level.condition_id.endswith("-adjacent"):
                raise ValueError(f"unexpected V8 core condition {level.condition_id!r}")
            geometry = level.condition_id[len(prefix) : -len("-adjacent")].replace(
                "-", "_"
            )
            try:
                within_family = float(geometry_mass[level.family][geometry])
            except KeyError as exc:
                raise ValueError(
                    f"V8 geometry mass missing for {level.condition_id!r}"
                ) from exc
            return slice_mass["core"] * family_mass * within_family
        if level.condition_id in constraint_ids:
            return (
                slice_mass["constraint"]
                * family_mass
                / constraint_family_counts[level.family]
            )
        raise ValueError(f"V8 stage selected unknown condition {level.condition_id!r}")

    if sampler_profile in V8_CONTINUOUS_PROFILES:
        return selected, ()
    probabilities = tuple(probability(level) for level in selected)
    if abs(sum(probabilities) - 1.0) > 1e-12 or any(
        value <= 0.0 for value in probabilities
    ):
        raise ValueError(
            f"V8 {stage} sampling probabilities must be positive and sum to one"
        )
    return selected, probabilities


def _v8_continuous_graph(
    levels: tuple[AcceptedLevel, ...],
    capability_ids: tuple[str, ...],
    core_ids: tuple[str, ...],
    constraint_ids: tuple[str, ...],
    release_id: str = V8_RELEASE_ID,
) -> tuple[tuple[int, ...], str]:
    """Validate the executable coarse-depth graph against the frozen V8 bank."""
    if release_id == V8_RELEASE_ID:
        graph_path = V8_CONTINUOUS_GRAPH_PATH
        graph_schema = "terra_v8_continuous_banded_graph_v1"
        allow_empty_family_depth = False
    elif release_id == AXIS_V2_RELEASE_ID:
        graph_path = AXIS_V2_CONTINUOUS_GRAPH_PATH
        graph_schema = "terra_axis_v2_continuous_banded_graph_v1"
        allow_empty_family_depth = True
    else:
        raise ValueError(f"unsupported continuous graph release {release_id!r}")
    graph = json.loads(graph_path.read_text())
    if set(graph) != {"schema", "release_id", "siblings_ordered", "depths"}:
        raise ValueError(f"{graph_path}: unexpected graph fields")
    if graph["schema"] != graph_schema:
        raise ValueError(f"{graph_path}: unsupported graph schema")
    if graph["release_id"] != release_id or graph["siblings_ordered"] is not False:
        raise ValueError(f"{graph_path}: release or sibling-order contract changed")
    depths = graph["depths"]
    if not isinstance(depths, dict) or set(depths) != {"0", "1", "2"}:
        raise ValueError(f"{graph_path}: depths must be exactly 0, 1, and 2")
    expected = {
        0: set(capability_ids),
        1: set(core_ids),
        2: set(constraint_ids),
    }
    by_id = {level.condition_id: level for level in levels}
    depth_by_id: dict[str, int] = {}
    for depth in range(3):
        family_rows = depths[str(depth)]
        if not isinstance(family_rows, dict) or set(family_rows) != set(FAMILIES):
            raise ValueError(f"{graph_path}: depth {depth} must name both families")
        observed: set[str] = set()
        for family in FAMILIES:
            condition_ids = family_rows[family]
            if not isinstance(condition_ids, list) or (
                not condition_ids and not allow_empty_family_depth
            ):
                raise ValueError(
                    f"{graph_path}: depth {depth} {family} must be a valid list"
                )
            for condition_id in condition_ids:
                if condition_id in depth_by_id:
                    raise ValueError(
                        f"{graph_path}: repeated condition {condition_id!r}"
                    )
                if condition_id not in by_id or by_id[condition_id].family != family:
                    raise ValueError(
                        f"{graph_path}: {condition_id!r} is not a V8 {family} condition"
                    )
                depth_by_id[condition_id] = depth
                observed.add(condition_id)
        if observed != expected[depth]:
            raise ValueError(f"{graph_path}: depth {depth} does not match the V8 bank")
    if set(depth_by_id) != set(by_id):
        raise ValueError(f"{graph_path}: graph does not cover all selected conditions")
    return tuple(depth_by_id[level.condition_id] for level in levels), _sha256_file(
        graph_path
    )


def _train_maps_per_condition(index: dict, source: Path) -> int:
    """Return the one explicitly supported train-bank slot contract."""
    release_id = index.get("release_id")
    if release_id is None:
        return TRAIN_MAPS_PER_CONDITION
    named_counts = {
        TRAIN96_RELEASE_ID: TRAIN96_MAPS_PER_CONDITION,
        V8_RELEASE_ID: V8_MAPS_PER_CONDITION,
        AXIS_V2_RELEASE_ID: AXIS_V2_MAPS_PER_CONDITION,
    }
    if release_id not in named_counts:
        raise ValueError(f"{source}: unsupported release_id {release_id!r}")
    count = index.get("train_maps_per_condition")
    expected = named_counts[release_id]
    if count != expected:
        raise ValueError(
            f"{source}: {release_id} must declare "
            f"train_maps_per_condition={expected}"
        )
    return expected


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
    elif index.get("release_id") == V8_RELEASE_ID:
        _validate_v8_release(root_path, index, levels)
    elif index.get("release_id") == AXIS_V2_RELEASE_ID:
        _validate_axis_v2_release(root_path, index, levels)
    else:
        _validate_review_admission(root_path, index, condition_ids)
    return maps_per_condition


def _validate_evaluation_panels(
    root: Path,
    panels: object,
    protocol_sha256: str,
    *,
    expected_condition_ids: tuple[str, ...] | None = None,
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
        if expected_condition_ids is not None and cells != set(expected_condition_ids):
            raise ValueError(
                f"evaluation panel {name} conditions do not match the frozen set"
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


EVALUATION_PANEL_FAMILY_DEFAULT = "main"


def _substituted_panel_family(root: Path, panels: object, family: str) -> dict:
    """Repoint the declared evaluation panels at a sibling panel family.

    A derived bank (e.g. the fresh-trench finite-metadata enrichment) ships
    ``evaluation/<family>/{development,promotion,sealed}`` next to the frozen
    ``evaluation/main/*`` and deliberately leaves the root ``dataset.json``
    byte-identical to the receipt it was derived from. Each substituted panel
    directory is self-describing, so its own ``dataset.json`` slot_count and
    manifest supply the declaration that the root would otherwise carry; the
    panel contract itself is then enforced by ``_validate_evaluation_panels``
    exactly as for the declared family.
    """
    if not isinstance(panels, dict):
        raise ValueError("evaluation_panels must be an object to substitute a family")
    substituted = {}
    for name, panel in panels.items():
        if not isinstance(panel, dict):
            raise ValueError(f"evaluation panel {name} must be an object")
        declared = panel.get("maps_path")
        expected_prefix = f"evaluation/{EVALUATION_PANEL_FAMILY_DEFAULT}/"
        if not isinstance(declared, str) or not declared.startswith(expected_prefix):
            raise ValueError(
                f"evaluation panel {name} declares {declared!r}; a panel family "
                f"substitution requires a path under {expected_prefix!r}"
            )
        maps_path = f"evaluation/{family}/" + declared[len(expected_prefix) :]
        _, directory = _safe_relative_directory(root, maps_path)
        index = json.loads((directory / "dataset.json").read_text())
        slot_count = index.get("slot_count")
        rows = _json_lines(directory / "manifest.jsonl")
        conditions = {
            row.get("primary_cell")
            for row in rows
            if isinstance(row.get("primary_cell"), str)
        }
        substituted[name] = {
            "maps_path": maps_path,
            "slot_count": slot_count,
            "conditions": len(conditions),
        }
    return substituted


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
    curriculum_stage: str | None = None,
    sampler_profile: str | None = None,
    evaluation_panel_family: str = EVALUATION_PANEL_FAMILY_DEFAULT,
    condition_profile: str = "full",
) -> AcceptedBank:
    """Validate the canonical index and select the levels owned by one arm."""
    if arm not in ARMS:
        raise ValueError(f"accepted-bank arm must be one of {ARMS}, got {arm!r}")
    if not isinstance(terra_revision, str) or not terra_revision.strip():
        raise ValueError("terra_revision must be an explicit nonempty string")
    if terra_revision != terra_revision.strip():
        raise ValueError("terra_revision must not have surrounding whitespace")
    if condition_profile not in V8_CONDITION_PROFILES:
        raise ValueError(
            f"condition_profile must be one of {V8_CONDITION_PROFILES}, got "
            f"{condition_profile!r}"
        )
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
    release_id = index.get("release_id")
    expected_main_ids = None
    expected_capability_ids = None
    if release_id == V8_RELEASE_ID:
        if condition_profile == "axis_v2_40_v1":
            raise ValueError(
                f"condition_profile={condition_profile!r} applies only to "
                f"{AXIS_V2_RELEASE_ID}"
            )
        expected_main_ids = tuple(index.get("included_in_main_macro", ()))
        expected_capability_ids = tuple(
            index.get("v6_capability_floor_condition_ids", ())
        )
        if condition_profile == "trench_aligned_37_v1":
            expected_main_ids = tuple(
                condition_id
                for condition_id in expected_main_ids
                if condition_id not in V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS
            )
            if evaluation_panel_family == EVALUATION_PANEL_FAMILY_DEFAULT:
                evaluation_panel_family = V8_TRENCH_ALIGNED_EVALUATION_FAMILY
    elif release_id == AXIS_V2_RELEASE_ID:
        if condition_profile != "axis_v2_40_v1":
            raise ValueError(
                f"{AXIS_V2_RELEASE_ID} requires "
                "condition_profile='axis_v2_40_v1'"
            )
        expected_main_ids = tuple(index.get("included_in_main_macro", ()))
        expected_capability_ids = tuple(
            index.get("v6_capability_floor_condition_ids", ())
        )
    elif condition_profile != "full":
        raise ValueError(
            f"condition_profile={condition_profile!r} does not apply to "
            f"release {release_id!r}"
        )
    declared_panels = index.get("evaluation_panels")
    if evaluation_panel_family == EVALUATION_PANEL_FAMILY_DEFAULT:
        panels_to_validate = declared_panels
    else:
        # A substituted family is a strict subset of the declared macro (slots
        # are renumbered and unusable conditions dropped), so the root's frozen
        # condition list cannot be required verbatim; require containment.
        panels_to_validate = _substituted_panel_family(
            root_path, declared_panels, evaluation_panel_family
        )
        if expected_main_ids is not None:
            for name, panel in sorted(panels_to_validate.items()):
                _, directory = _safe_relative_directory(root_path, panel["maps_path"])
                observed = {
                    row.get("primary_cell")
                    for row in _json_lines(directory / "manifest.jsonl")
                }
                unknown = sorted(observed - set(expected_main_ids))
                if unknown:
                    raise ValueError(
                        f"evaluation panel {name} in family "
                        f"{evaluation_panel_family!r} introduces conditions "
                        f"outside the frozen macro: {unknown}"
                    )
        if condition_profile == "full":
            expected_main_ids = None
    evaluation_panels = _validate_evaluation_panels(
        root_path,
        panels_to_validate,
        protocol_sha256,
        expected_condition_ids=expected_main_ids,
    )
    capability_floor_evaluation_panels = ()
    if release_id in (TRAIN96_RELEASE_ID, V8_RELEASE_ID, AXIS_V2_RELEASE_ID):
        capability_floor_evaluation_panels = _validate_evaluation_panels(
            root_path,
            index.get("capability_floor_evaluation_panels"),
            protocol_sha256,
            expected_condition_ids=expected_capability_ids,
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
    v6_constraint_condition_ids: tuple[str, ...] = ()
    v7_core_condition_ids: tuple[str, ...] = ()
    v8_mixture = None
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
    elif release_id == V8_RELEASE_ID:
        (
            review_admission_sha256,
            v6_constraint_condition_ids,
            capability_floor_condition_ids,
            v7_core_condition_ids,
            v8_mixture,
        ) = _validate_v8_release(root_path, index, all_levels)
        constrained_condition_ids = tuple(
            sorted(set(v6_constraint_condition_ids) | set(v7_core_condition_ids))
        )
    elif release_id == AXIS_V2_RELEASE_ID:
        (
            review_admission_sha256,
            v6_constraint_condition_ids,
            capability_floor_condition_ids,
            v7_core_condition_ids,
            v8_mixture,
        ) = _validate_axis_v2_release(root_path, index, all_levels)
        constrained_condition_ids = tuple(
            sorted(set(v6_constraint_condition_ids) | set(v7_core_condition_ids))
        )
    else:
        review_admission_sha256 = _validate_review_admission(
            root_path,
            index,
            condition_ids,
        )

    sampling_probabilities: tuple[float, ...] = ()
    if release_id in (V8_RELEASE_ID, AXIS_V2_RELEASE_ID):
        if allow_diagnostic_control:
            raise ValueError("V8 is a training release, not a diagnostic control")
        if arm != "G-UNIFORM":
            raise ValueError("the first V8 curriculum supports only G-UNIFORM")
        if curriculum_stage is None:
            raise ValueError(f"{index_path}: V8 requires an explicit curriculum_stage")
        if sampler_profile is None:
            sampler_profile = "bank_v4"
        selected, sampling_probabilities = _v8_stage_selection(
            all_levels,
            curriculum_stage,
            v6_constraint_condition_ids,
            capability_floor_condition_ids,
            v7_core_condition_ids,
            v8_mixture,
            sampler_profile,
            condition_profile,
        )
    else:
        if curriculum_stage is not None:
            raise ValueError(
                f"{index_path}: curriculum_stage applies only to {V8_RELEASE_ID}"
            )
        if sampler_profile is not None:
            raise ValueError(
                f"{index_path}: sampler_profile applies only to {V8_RELEASE_ID}"
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
    curriculum_depths: tuple[int, ...] = ()
    curriculum_graph_sha256 = None
    if sampler_profile in V8_CONTINUOUS_PROFILES:
        all_depths, curriculum_graph_sha256 = _v8_continuous_graph(
            tuple(all_levels),
            capability_floor_condition_ids,
            v7_core_condition_ids,
            v6_constraint_condition_ids,
            release_id,
        )
        depth_by_id = {
            level.condition_id: all_depths[index]
            for index, level in enumerate(all_levels)
        }
        curriculum_depths = tuple(
            depth_by_id[level.condition_id] for level in selected
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
        curriculum_stage=curriculum_stage,
        sampler_profile=sampler_profile,
        sampling_probabilities=sampling_probabilities,
        v6_constraint_condition_ids=v6_constraint_condition_ids,
        v7_core_condition_ids=v7_core_condition_ids,
        curriculum_depths=curriculum_depths,
        curriculum_graph_sha256=curriculum_graph_sha256,
        evaluation_panel_family=evaluation_panel_family,
        condition_profile=condition_profile,
    )
