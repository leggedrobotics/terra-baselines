import json
from pathlib import Path
import subprocess

import pytest

from configs.training_configs import get_config
from scripts import build_trench_aligned_runtime_bank
from utils.accepted_bank import AcceptedLevel
from utils.accepted_bank import V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS
from utils.accepted_bank import _v8_stage_selection


def test_gate_on_generalist_scope_is_exactly_the_supported_v8_subset():
    config = get_config("trench_align_generalist_v1")
    paths = [level.maps_path for level in config.maps]

    assert len(paths) == 37
    assert len(set(paths)) == 37
    assert config.enforce_trench_dig_alignment is True
    assert config.require_trench_alignment_metadata is True
    assert config.trench_alignment_observation is True
    assert config.pooled_sampler.rule == "continuous_banded_v3"
    assert all(level.max_steps_in_episode == 450 for level in config.maps)
    assert all(level.rewards_type == "DENSE" for level in config.maps)
    assert all(not level.apply_trench_rewards for level in config.maps)
    assert not any("trn-net4" in path for path in paths)
    assert not any("v7-trn" in path for path in paths)


def test_partial_recipe_requires_the_named_37_condition_view():
    config = get_config("trench_align_generalist_partial_v1")
    assert config.maps == []
    assert config.accepted_bank_arm == "G-UNIFORM"
    assert config.accepted_bank_condition_profile == "trench_aligned_37_v1"
    assert config.requires_partial_reset is True
    assert config.enforce_trench_dig_alignment is True
    assert config.require_trench_alignment_metadata is True
    assert config.trench_alignment_observation is True
    assert config.pooled_sampler.rule == "continuous_banded_v3"


def test_trench_aligned_view_is_exactly_25_foundation_plus_12_trench():
    graph_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "v8_continuous_banded_graph_v1.json"
    )
    graph = json.loads(graph_path.read_text())
    by_depth = {}
    levels = []
    branch_names = ("Anchor", "Nearby core", "Composed")
    for depth in range(3):
        by_depth[depth] = []
        for family in ("foundation", "trench"):
            for condition_id in graph["depths"][str(depth)][family]:
                by_depth[depth].append(condition_id)
                levels.append(
                    AcceptedLevel(
                        condition_id=condition_id,
                        family=family,
                        branch_depth=branch_names[depth],
                        maps_path=f"train/{condition_id}",
                        map_count=96,
                    )
                )
    selected, probabilities = _v8_stage_selection(
        levels,
        "full",
        tuple(by_depth[2]),
        tuple(by_depth[0]),
        tuple(by_depth[1]),
        {
            "v7_geometry_mass_within_family": {
                "foundation": {"fixture": 1.0},
                "trench": {"fixture": 1.0},
            }
        },
        "continuous_banded_v3",
        "trench_aligned_37_v1",
    )
    assert probabilities == ()
    assert len(selected) == 37
    assert sum(level.family == "foundation" for level in selected) == 25
    assert sum(level.family == "trench" for level in selected) == 12
    assert not (
        {level.condition_id for level in selected}
        & set(V8_TRENCH_ALIGNED_EXCLUDED_CONDITION_IDS)
    )


def test_partial_generalist_launcher_is_smoke_gated_and_resume_bounded():
    root = Path(__file__).resolve().parents[1]
    submit = (
        root
        / "scripts/euler_trench_align_generalist_partial_v1/submit.sh"
    ).read_text()
    sbatch = (
        root
        / "scripts/euler_trench_align_generalist_partial_v1/run.sbatch"
    ).read_text()
    runner = (
        root / "scripts/run_trench_align_generalist_partial_v1.sh"
    ).read_text()

    assert submit.index('if [ "$SUBMIT" = 0 ]') < submit.index(
        "remote() { ssh"
    )
    assert "0|stage|smoke|1" in submit
    assert "PARTIAL_BANK_SHA=f25398d3" in submit
    assert "EXPECTED_PARTIAL_CONDITIONS=35" in submit
    assert "EXPECTED_PARTIAL_TRIPLETS=238" in submit
    assert "EXPECTED_RELAY_TRIPLETS=85" in submit
    assert "EXPECTED_IN_ZONE_TRIPLETS=153" in submit
    assert "EXPECTED_TRENCH_AUDIT_SIDECARS=255" in submit
    assert "--partition='gpuhe.4h'" in submit
    assert "RUN_ROLE=smoke" in submit
    assert "RUN_ROLE=$RUN_ROLE,TARGET_UPDATE=1" in submit
    assert "status=COMPLETE" in submit
    assert "checkpoint_validation.json" in submit
    assert "RUN_ROLE=phase1,TARGET_UPDATE=75000" in submit
    assert "RUN_ROLE=phase2,TARGET_UPDATE=100000" in submit
    assert "--dependency='afterok:$JOB1_ID'" in submit
    assert "--exclude='eu-g6-064'" in submit

    assert "len(jax.devices()) == devices" in sbatch
    assert "lax.conv_general_dilated" in sbatch
    assert "partial_reset_bank_sha256(partial_root)" in sbatch
    assert 'audit["accepted"] is True' in sbatch
    assert 'checkpoint["next_update"] == 75000' in sbatch
    assert 'checkpoint["next_update"] == int(os.environ["TARGET_UPDATE"])' in sbatch
    assert "optimizer_finite" in sbatch

    assert "--config trench_align_generalist_partial_v1" in runner
    assert "--accepted-bank-condition-profile trench_aligned_37_v1" in runner
    assert "--partial-reset-root" in runner
    assert "--reward-v2-reset-context-observation" in runner
    assert "--stall_age_observation" in runner


def test_runtime_bank_builder_binds_the_imported_clean_terra_checkout(
    tmp_path,
    monkeypatch,
):
    terra_root = tmp_path / "terra-runtime"
    package = terra_root / "terra"
    package.mkdir(parents=True)
    imported_file = package / "__init__.py"
    imported_file.write_text("# fixture\n")
    subprocess.run(["git", "init", "-q", str(terra_root)], check=True)
    subprocess.run(["git", "-C", str(terra_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(terra_root),
            "-c",
            "user.name=Terra Test",
            "-c",
            "user.email=terra-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(terra_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    monkeypatch.setattr(
        build_trench_aligned_runtime_bank.terra,
        "__file__",
        str(imported_file),
    )

    build_trench_aligned_runtime_bank._validate_terra_checkout(
        terra_root,
        revision,
    )
    with pytest.raises(RuntimeError, match="revision mismatch"):
        build_trench_aligned_runtime_bank._validate_terra_checkout(
            terra_root,
            "0" * 40,
        )
    imported_file.write_text("# dirty fixture\n")
    with pytest.raises(RuntimeError, match="committed and clean"):
        build_trench_aligned_runtime_bank._validate_terra_checkout(
            terra_root,
            revision,
        )
