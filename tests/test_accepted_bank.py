import hashlib
import json
from pathlib import Path

import pytest

from utils import accepted_bank as accepted_bank_module
from utils.accepted_bank import (
    AcceptedLevel,
    SCENARIO_IDENTITY_CONTRACT,
    SCHEMA,
    load_accepted_bank,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_level(
    root: Path,
    condition: str,
    family: str,
    depth: str,
    map_count: int,
) -> dict:
    relative = Path("train") / condition
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "dataset.json").write_text("{}\n")
    rows = [
        {
            "slot_index": index,
            "map_id": f"{condition}-{index}",
            "family": family,
            "primary_cell": condition,
        }
        for index in range(1, map_count + 1)
    ]
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return {
        "condition_id": condition,
        "family": family,
        "branch_depth": depth,
        "maps_path": relative.as_posix(),
        "map_count": map_count,
    }


def _protocol(revision="terra-test-revision") -> dict:
    payload = {
        "schema": "terra_environment_protocol_v1",
        "terra_revision": revision,
        "reset_prng": accepted_bank_module.RESET_PRNG_CONTRACT,
    }
    return {
        **payload,
        "environment_protocol_sha256": (
            accepted_bank_module._canonical_json_sha256(payload)
        ),
    }


@pytest.fixture(autouse=True)
def _freeze_current_protocol(monkeypatch):
    monkeypatch.setattr(
        accepted_bank_module,
        "_environment_protocol_for_revision",
        _protocol,
    )


def _write_bank(root: Path, map_count: int = 64) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    (root / "environment_protocol.json").write_text(json.dumps(protocol) + "\n")
    registry = root / "source_registry.jsonl"
    registry.write_text('{"source_id": "s0", "split": "train"}\n')
    train = [
        _write_level(root, "f-anchor", "foundation", "Anchor", map_count),
        _write_level(root, "t-anchor", "trench", "Anchor", map_count),
        _write_level(root, "f-axis", "foundation", "One-axis", map_count),
        _write_level(root, "t-composed", "trench", "Composed", map_count),
    ]
    review_admission = root / "review_admission.json"
    review_admission.write_text(
        json.dumps(
            {
                "schema": "terra-accepted-condition-set-v1",
                "release": "map-curriculum-diverse64-visual-review-20260730",
                "manifest_sha256": (
                    "39f7cd2e8ce565bd384de214da5f2eee5e76764cb554e149c0ba675d815d6d51"
                ),
                "review_data_sha256": (
                    "8404fcaa9a6b66949ade2b0225d3e7800968951953d2b6363aabffe38100cc0b"
                ),
                "accepted_conditions": sorted(entry["condition_id"] for entry in train),
            }
        )
        + "\n"
    )
    evaluation_panels = {}
    for panel_name in ("promotion", "development", "sealed"):
        directory = root / panel_name
        directory.mkdir()
        (directory / "dataset.json").write_text("{}\n")
        rows = []
        for slot, entry in enumerate(train, start=1):
            scenario_id = f"{slot:064x}"
            episode_id = accepted_bank_module._canonical_json_sha256(
                {
                    "schema": "terra_episode_id_v1",
                    "scenario_id": scenario_id,
                    "reset_seed": slot,
                    "environment_protocol_sha256": (
                        protocol["environment_protocol_sha256"]
                    ),
                }
            )
            rows.append(
                {
                    "slot_index": slot,
                    "map_id": f"{panel_name}-{slot}",
                    "family": entry["family"],
                    "primary_cell": entry["condition_id"],
                    "scenario_id": scenario_id,
                    "reset_seed": slot,
                    "episode_id": episode_id,
                    "environment_protocol_sha256": (
                        protocol["environment_protocol_sha256"]
                    ),
                }
            )
        (directory / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        evaluation_panels[panel_name] = {
            "maps_path": panel_name,
            "slot_count": len(rows),
            "conditions": len(rows),
        }
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "scenario_identity_contract": SCENARIO_IDENTITY_CONTRACT,
                "environment_protocol": "environment_protocol.json",
                "environment_protocol_sha256": (
                    protocol["environment_protocol_sha256"]
                ),
                "source_registry": "source_registry.jsonl",
                "source_registry_sha256": _sha256(registry),
                "review_admission": "review_admission.json",
                "review_admission_sha256": _sha256(review_admission),
                "train": train,
                "evaluation_panels": evaluation_panels,
            }
        )
        + "\n"
    )
    return root


@pytest.mark.parametrize(
    ("arm", "conditions"),
    [
        ("F-ANCHOR", ["f-anchor"]),
        ("F-SPECIALIST", ["f-anchor", "f-axis"]),
        ("T-ANCHOR", ["t-anchor"]),
        ("T-SPECIALIST", ["t-anchor", "t-composed"]),
        ("G-UNIFORM", ["f-anchor", "f-axis", "t-anchor", "t-composed"]),
        ("G-ADAPTIVE", ["f-anchor", "f-axis", "t-anchor", "t-composed"]),
    ],
)
def test_arm_selection_is_explicit(tmp_path, arm, conditions):
    bank = load_accepted_bank(
        _write_bank(tmp_path),
        arm,
        "terra-test-revision",
    )
    assert [level.condition_id for level in bank.levels] == conditions
    assert bank.map_count_per_condition == 64
    assert bank.terra_revision == "terra-test-revision"
    assert bank.review_admission_sha256 == _sha256(tmp_path / "review_admission.json")


def test_v8_stage_selection_is_family_balanced_and_checkpoint_bounded():
    controls = [
        AcceptedLevel("fnd-slab-allfree", "foundation", "Anchor", "f", 96),
        AcceptedLevel("trn-straight-allfree", "trench", "Anchor", "t", 96),
    ]
    foundation_geometry = {
        "slab": 0.25,
        "irregular": 0.15,
        "courtyard": 0.15,
        "bearing_walls": 0.20,
        "pads": 0.15,
        "courtyard_pads": 0.10,
    }
    trench_geometry = {
        "straight": 0.15,
        "dogleg": 0.15,
        "tee": 0.20,
        "cross": 0.10,
        "double_t": 0.20,
        "network3": 0.15,
        "disconnected_pair": 0.05,
    }
    core = [
        AcceptedLevel(
            f"v7-fnd-{name.replace('_', '-')}-adjacent",
            "foundation",
            "Nearby core",
            name,
            96,
        )
        for name in foundation_geometry
    ] + [
        AcceptedLevel(
            f"v7-trn-{name.replace('_', '-')}-adjacent",
            "trench",
            "Nearby core",
            name,
            96,
        )
        for name in trench_geometry
    ]
    constraints = [
        AcceptedLevel(f"f-{index}", "foundation", "One-axis", f"f{index}", 96)
        for index in range(18)
    ] + [
        AcceptedLevel(f"t-{index}", "trench", "One-axis", f"t{index}", 96)
        for index in range(14)
    ]
    levels = controls + core + constraints
    mixture = {
        "v7_geometry_mass_within_family": {
            "foundation": foundation_geometry,
            "trench": trench_geometry,
        }
    }
    for stage, expected_count in (("capability", 2), ("nearby", 15), ("full", 47)):
        selected, probabilities = accepted_bank_module._v8_stage_selection(
            levels,
            stage,
            tuple(level.condition_id for level in constraints),
            tuple(level.condition_id for level in controls),
            tuple(level.condition_id for level in core),
            mixture,
        )
        assert len(selected) == expected_count
        assert sum(probabilities) == pytest.approx(1.0)
        for family in ("foundation", "trench"):
            assert sum(
                probability
                for probability, level in zip(probabilities, selected)
                if level.family == family
            ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("arm", "family"),
    (("F-SPECIALIST", "foundation"), ("T-SPECIALIST", "trench")),
)
def test_specialist_rejects_anchor_only_family(tmp_path, arm, family):
    root = _write_bank(tmp_path)
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    index["train"] = [
        entry
        for entry in index["train"]
        if entry["family"] != family or entry["branch_depth"] == "Anchor"
    ]
    receipt_path = root / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["accepted_conditions"] = sorted(
        entry["condition_id"] for entry in index["train"]
    )
    receipt_path.write_text(json.dumps(receipt) + "\n")
    index["review_admission_sha256"] = _sha256(receipt_path)
    index_path.write_text(json.dumps(index) + "\n")

    with pytest.raises(ValueError, match="duplicate the family anchor control"):
        load_accepted_bank(root, arm, "terra-test-revision")


def test_legacy_or_unfrozen_roots_fail_loudly(tmp_path):
    root = _write_bank(tmp_path)
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    index["schema"] = "terra_curriculum_m3_training_dataset_v1"
    index_path.write_text(json.dumps(index) + "\n")
    with pytest.raises(ValueError, match="terra_curriculum_loader_bank_v1"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")


@pytest.mark.parametrize("value", [None, "terra_legacy_map_id_v0"])
def test_root_requires_reset_array_scenario_identity(tmp_path, value):
    root = _write_bank(tmp_path)
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    if value is None:
        index.pop("scenario_identity_contract")
    else:
        index["scenario_identity_contract"] = value
    index_path.write_text(json.dumps(index) + "\n")
    with pytest.raises(ValueError, match="terra_reset_arrays_sha256_v1"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")


def test_stale_environment_protocol_is_rejected(tmp_path):
    root = _write_bank(tmp_path)
    with pytest.raises(ValueError, match="Terra revision mismatch"):
        load_accepted_bank(root, "G-UNIFORM", "different-terra-revision")


def test_reset_prng_contract_is_required(tmp_path):
    root = _write_bank(tmp_path)
    protocol_path = root / "environment_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.pop("reset_prng")
    payload = {
        key: value
        for key, value in protocol.items()
        if key != "environment_protocol_sha256"
    }
    protocol["environment_protocol_sha256"] = (
        accepted_bank_module._canonical_json_sha256(payload)
    )
    protocol_path.write_text(json.dumps(protocol) + "\n")
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    index["environment_protocol_sha256"] = protocol["environment_protocol_sha256"]
    index_path.write_text(json.dumps(index) + "\n")

    with pytest.raises(ValueError, match="reset PRNG contract mismatch"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")


def test_archive_without_git_uses_explicit_frozen_revision(tmp_path, monkeypatch):
    archive = tmp_path / "source-archive"
    root = _write_bank(archive / "accepted-bank")
    assert not (archive / ".git").exists()
    monkeypatch.chdir(archive)
    bank = load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")
    assert (
        bank.environment_protocol_sha256 == _protocol()["environment_protocol_sha256"]
    )


def test_registry_hash_and_manifest_condition_are_verified(tmp_path):
    root = _write_bank(tmp_path)
    (root / "source_registry.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="registry hash mismatch"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")

    root = _write_bank(tmp_path / "second")
    manifest = root / "train" / "f-anchor" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["primary_cell"] = "wrong"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="primary_cell"):
        load_accepted_bank(root, "F-ANCHOR", "terra-test-revision")


def test_review_admission_schema_hash_and_conditions_are_verified(tmp_path):
    root = _write_bank(tmp_path / "schema")
    receipt_path = root / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["schema"] = "wrong"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    index["review_admission_sha256"] = _sha256(receipt_path)
    index_path.write_text(json.dumps(index) + "\n")
    with pytest.raises(ValueError, match="terra-accepted-condition-set-v1"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")

    root = _write_bank(tmp_path / "hash")
    (root / "review_admission.json").write_text("{}\n")
    with pytest.raises(ValueError, match="review admission hash mismatch"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")

    root = _write_bank(tmp_path / "conditions")
    receipt_path = root / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["accepted_conditions"] = receipt["accepted_conditions"][:-1]
    receipt_path.write_text(json.dumps(receipt) + "\n")
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    index["review_admission_sha256"] = _sha256(receipt_path)
    index_path.write_text(json.dumps(index) + "\n")
    with pytest.raises(ValueError, match="do not match train condition IDs"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")


def _convert_to_diagnostic_control(root: Path) -> Path:
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    conditions = {
        entry["condition_id"]: {"family": entry["family"]} for entry in index["train"]
    }
    contract_path = root / "control_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema": "terra_unconstrained_control_bank_v1",
                "included_in_constrained_macro": False,
                "conditions": conditions,
            }
        )
        + "\n"
    )
    index.pop("review_admission")
    index.pop("review_admission_sha256")
    (root / "review_admission.json").unlink()
    index.update(
        {
            "control_schema": "terra_unconstrained_control_bank_v1",
            "control_contract": "control_contract.json",
            "control_contract_sha256": _sha256(contract_path),
            "included_in_constrained_macro": False,
        }
    )
    index_path.write_text(json.dumps(index) + "\n")
    return root


def _convert_to_train96_capability_floor(root: Path) -> Path:
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    constrained_ids = sorted(entry["condition_id"] for entry in index["train"])
    controls = [
        _write_level(root, "fnd-slab-allfree", "foundation", "Anchor", 96),
        _write_level(root, "trn-straight-allfree", "trench", "Anchor", 96),
    ]
    for offset, entry in enumerate(controls, start=len(index["train"])):
        entry["level_index"] = offset
    index["train"].extend(controls)

    protocol_sha256 = index["environment_protocol_sha256"]
    capability_panels = {}
    for panel_name in ("promotion", "development", "sealed"):
        directory = root / f"capability_floor_{panel_name}"
        directory.mkdir()
        (directory / "dataset.json").write_text("{}\n")
        rows = []
        for slot, entry in enumerate(controls, start=1):
            scenario_id = f"{100 + slot:064x}"
            episode_id = accepted_bank_module._canonical_json_sha256(
                {
                    "schema": "terra_episode_id_v1",
                    "scenario_id": scenario_id,
                    "reset_seed": 100 + slot,
                    "environment_protocol_sha256": protocol_sha256,
                }
            )
            rows.append(
                {
                    "slot_index": slot,
                    "map_id": f"capability-{panel_name}-{slot}",
                    "family": entry["family"],
                    "primary_cell": entry["condition_id"],
                    "scenario_id": scenario_id,
                    "reset_seed": 100 + slot,
                    "episode_id": episode_id,
                    "environment_protocol_sha256": protocol_sha256,
                }
            )
        (directory / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        capability_panels[panel_name] = {
            "maps_path": directory.name,
            "slot_count": len(rows),
            "conditions": len(rows),
        }

    index.pop("review_admission")
    index.pop("review_admission_sha256")
    index.update(
        {
            "release_id": accepted_bank_module.TRAIN96_RELEASE_ID,
            "included_in_constrained_macro": constrained_ids,
            "train_maps_per_condition": 96,
            "constrained_condition_ids": constrained_ids,
            "capability_floor_condition_ids": list(
                accepted_bank_module.TRAIN96_CAPABILITY_FLOOR_IDS
            ),
            "constrained_review_admission": "review_admission.json",
            "constrained_review_admission_sha256": _sha256(
                root / "review_admission.json"
            ),
            "capability_floor_evaluation_panels": capability_panels,
        }
    )
    contract = {
        "schema": accepted_bank_module.TRAIN96_CAPABILITY_FLOOR_SCHEMA,
        "release_id": accepted_bank_module.TRAIN96_RELEASE_ID,
        "included_in_constrained_macro": False,
        "constrained_condition_ids": constrained_ids,
        "capability_floor_condition_ids": list(
            accepted_bank_module.TRAIN96_CAPABILITY_FLOOR_IDS
        ),
        "train_maps_per_condition": 96,
        "evaluation_panels": {
            "constrained": accepted_bank_module._panel_count_contract(
                index["evaluation_panels"]
            ),
            "capability_floor": accepted_bank_module._panel_count_contract(
                capability_panels
            ),
        },
    }
    contract_path = root / "capability_floor_contract.json"
    contract_path.write_text(json.dumps(contract) + "\n")
    index["capability_floor_contract"] = "capability_floor_contract.json"
    index["capability_floor_contract_sha256"] = _sha256(contract_path)
    index_path.write_text(json.dumps(index) + "\n")
    return root


def test_train96_release_is_explicit_and_keeps_control_scores_separate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        accepted_bank_module,
        "TRAIN96_CONSTRAINED_CONDITION_COUNT",
        4,
    )
    root = _convert_to_train96_capability_floor(_write_bank(tmp_path, map_count=96))
    bank = load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")

    assert bank.release_id == accepted_bank_module.TRAIN96_RELEASE_ID
    assert bank.map_count_per_condition == 96
    assert len(bank.levels) == 6
    assert bank.constrained_condition_ids == (
        "f-anchor",
        "f-axis",
        "t-anchor",
        "t-composed",
    )
    assert bank.capability_floor_condition_ids == (
        "fnd-slab-allfree",
        "trn-straight-allfree",
    )
    assert bank.capability_floor_contract_sha256 == _sha256(
        root / "capability_floor_contract.json"
    )
    assert {panel.condition_count for panel in bank.evaluation_panels} == {4}
    assert {
        panel.condition_count for panel in bank.capability_floor_evaluation_panels
    } == {2}
    assert bank.diagnostic_contract_sha256 is None


def test_train96_capability_floor_contract_is_hash_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(
        accepted_bank_module,
        "TRAIN96_CONSTRAINED_CONDITION_COUNT",
        4,
    )
    root = _convert_to_train96_capability_floor(_write_bank(tmp_path, map_count=96))
    (root / "capability_floor_contract.json").write_text("{}\n")
    with pytest.raises(ValueError, match="capability-floor contract hash mismatch"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")


def test_diagnostic_control_requires_explicit_opt_in(tmp_path):
    root = _convert_to_diagnostic_control(_write_bank(tmp_path))
    with pytest.raises(ValueError, match="review_admission"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")

    bank = load_accepted_bank(
        root,
        "G-UNIFORM",
        "terra-test-revision",
        allow_diagnostic_control=True,
    )
    assert bank.review_admission_sha256 is None
    assert bank.diagnostic_contract_sha256 == _sha256(root / "control_contract.json")


def test_diagnostic_control_contract_is_hash_verified(tmp_path):
    root = _convert_to_diagnostic_control(_write_bank(tmp_path))
    (root / "control_contract.json").write_text("{}\n")
    with pytest.raises(ValueError, match="control contract hash mismatch"):
        load_accepted_bank(
            root,
            "G-UNIFORM",
            "terra-test-revision",
            allow_diagnostic_control=True,
        )


@pytest.mark.parametrize(
    "field",
    ["release", "manifest_sha256", "review_data_sha256"],
)
def test_stale_review_admission_identity_is_rejected(tmp_path, field):
    root = _write_bank(tmp_path)
    receipt_path = root / "review_admission.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = "stale"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    index["review_admission_sha256"] = _sha256(receipt_path)
    index_path.write_text(json.dumps(index) + "\n")
    with pytest.raises(
        ValueError,
        match=rf"{field} must match the pinned diverse-64 review release",
    ):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")


def test_unequal_level_counts_are_rejected(tmp_path):
    root = _write_bank(tmp_path)
    index_path = root / "dataset.json"
    index = json.loads(index_path.read_text())
    t_level = root / "train" / "t-composed"
    rows = [
        json.loads(line)
        for line in (t_level / "manifest.jsonl").read_text().splitlines()
    ]
    rows.append(
        {
            "slot_index": 65,
            "map_id": "t-composed-65",
            "family": "trench",
            "primary_cell": "t-composed",
        }
    )
    (t_level / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    next(entry for entry in index["train"] if entry["condition_id"] == "t-composed")[
        "map_count"
    ] = 65
    index_path.write_text(json.dumps(index) + "\n")
    with pytest.raises(ValueError, match="unequal per-condition map counts"):
        load_accepted_bank(root, "G-ADAPTIVE", "terra-test-revision")


def test_wrong_train_maps_per_condition_is_rejected(tmp_path):
    root = _write_bank(tmp_path, map_count=63)
    with pytest.raises(ValueError, match="exactly 64 train maps per condition"):
        load_accepted_bank(root, "G-UNIFORM", "terra-test-revision")
