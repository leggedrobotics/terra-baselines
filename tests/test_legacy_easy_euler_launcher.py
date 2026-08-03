from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_legacy_easy_v1"


def test_sbatch_resolves_helpers_from_exported_installed_launcher() -> None:
    sbatch = (LAUNCHER / "run.sbatch").read_text()

    assert ': "${LEGACY_EASY_LAUNCHER_DIR:?missing LEGACY_EASY_LAUNCHER_DIR}"' in sbatch
    assert 'source "$LEGACY_EASY_LAUNCHER_DIR/common.sh"' in sbatch
    assert '"$LEGACY_EASY_LAUNCHER_DIR/verify_result.py"' in sbatch
    assert 'source "$SCRIPT_DIR/common.sh"' not in sbatch


def test_submit_exports_exact_installed_launcher_directory() -> None:
    submit = (LAUNCHER / "submit.sh").read_text()

    assert "LEGACY_EASY_LAUNCHER_DIR=$SCRIPT_DIR" in submit

