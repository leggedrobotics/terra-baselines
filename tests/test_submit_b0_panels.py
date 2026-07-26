import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "euler_curriculum_recovery_v1" / "submit_b0_panels.sh"
)


class SubmitB0PanelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        script_root = (
            self.root
            / "source"
            / "terra-baselines"
            / "scripts"
            / "euler_curriculum_recovery_v1"
        )
        script_root.mkdir(parents=True)
        for name in ("run_b0_panel.sbatch", "eval_b0_panel.sbatch"):
            path = script_root / name
            path.write_text("#!/usr/bin/env bash\n")
            path.chmod(0o755)
        manifests = self.root / "manifests"
        manifests.mkdir()
        (manifests / "source_files.sha256").touch()
        (manifests / "bank_files.sha256").touch()

        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        fake_sbatch = fake_bin / "sbatch"
        fake_sbatch.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "counter=0\n"
            'if [[ -f "$FAKE_SBATCH_COUNTER" ]]; then '
            'counter="$(< "$FAKE_SBATCH_COUNTER")"; fi\n'
            "counter=$((counter + 1))\n"
            'printf "%s\\n" "$counter" > "$FAKE_SBATCH_COUNTER"\n'
            'printf "%s\\n" "$*" >> "$FAKE_SBATCH_LOG"\n'
            'printf "%s\\n" "$((9000 + counter))"\n'
        )
        fake_sbatch.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{fake_bin}:{self.environment['PATH']}",
                "RUN_ROOT": str(self.root),
                "B0_UPDATES": "1000",
                "EXCLUDE_NODES": "eu-g6-064",
                "FAKE_SBATCH_COUNTER": str(self.root / "sbatch.counter"),
                "FAKE_SBATCH_LOG": str(self.root / "sbatch.log"),
            }
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_submit(self, *panels, **environment):
        process_environment = self.environment.copy()
        process_environment.update(environment)
        return subprocess.run(
            [str(SUBMIT_SCRIPT), *panels],
            check=False,
            capture_output=True,
            env=process_environment,
            text=True,
        )

    def test_diversity_variant_is_validated_exported_and_recorded(self):
        result = self.run_submit(
            "foundation_distance",
            B0_TRAIN_VARIANT="foundation_distance_diversity_v1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        receipt = (self.root / "submitted_jobs.txt").read_text()
        self.assertIn("train_variant=foundation_distance_diversity_v1\n", receipt)
        self.assertIn("python_dont_write_bytecode=1\n", receipt)
        self.assertIn("foundation_distance_train_job=9001\n", receipt)
        self.assertIn("foundation_distance_eval_job=9002\n", receipt)

        calls = (self.root / "sbatch.log").read_text().splitlines()
        self.assertEqual(len(calls), 2)
        expected_export = (
            f"--export=ALL,PANEL=foundation_distance,RUN_ROOT={self.root},"
            "B0_UPDATES=1000,"
            "B0_TRAIN_VARIANT=foundation_distance_diversity_v1,"
            "PYTHONDONTWRITEBYTECODE=1"
        )
        self.assertIn(expected_export, calls[0])
        self.assertIn(expected_export, calls[1])
        self.assertIn("--dependency=afterok:9001", calls[1])

    def test_diversity_variant_rejects_the_wrong_panel_before_submission(self):
        result = self.run_submit(
            "foundation_distance",
            B0_TRAIN_VARIANT="trench_side_diversity_v1",
        )
        self.assertEqual(result.returncode, 4)
        self.assertIn(
            "requires exactly PANEL=trench_side",
            result.stderr,
        )
        self.assertFalse((self.root / "sbatch.log").exists())
        self.assertFalse((self.root / "submitted_jobs.txt").exists())

    def test_bytecode_writes_cannot_be_enabled(self):
        result = self.run_submit(
            "foundation_distance",
            B0_TRAIN_VARIANT="base_v1",
            PYTHONDONTWRITEBYTECODE="0",
        )
        self.assertEqual(result.returncode, 5)
        self.assertIn(
            "require PYTHONDONTWRITEBYTECODE=1",
            result.stderr,
        )
        self.assertFalse((self.root / "sbatch.log").exists())


if __name__ == "__main__":
    unittest.main()
