from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_gru_fixed_benchmark_v1"


class V8GruFixedBenchmarkLauncherTest(unittest.TestCase):
    def test_dry_run_is_local_and_pins_all_four_checkpoints(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "SUBMIT": "0",
                "REMOTE_HOST": "must-not-be-contacted.invalid",
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER / "submit.sh")],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("SUBMIT=0: contract printed; no SSH", result.stdout)
        self.assertIn("update_040000.pkl", result.stdout)
        self.assertIn("update_044000.pkl", result.stdout)
        self.assertIn("update_086000.pkl", result.stdout)
        for expected in (
            "25f855db3d913fd638c4e56b1740437a2b7122ca",
            "33d26213327d66921b66753a5a6018a37d6f2e81",
            "2778766683fb8a0a53a761385fae05cf9396dda9",
            "9eb032308b07a8bb43a44bb01993f8e1aaa439d70eb8e14c2047c6469d6091fd",
            "0985b6338fb02f866b7aadbf065431cd667954a6f9b1a457e3eae9213533569d",
            "64ea0270dba0faf744eb15066232f1f137f9391c5aaf166ccbd57f00e329c623",
            "2fe5d23c86cc7702b188d33ca1ca9a42066a9a2515150e8795f8c640bbbeb4af",
            "b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725",
        ):
            self.assertIn(expected, result.stdout)

    def test_stage_exits_before_creating_a_result_directory(self) -> None:
        source = (LAUNCHER / "submit.sh").read_text(encoding="utf-8")
        stage_branch = source.index('if [ "$SUBMIT" = stage ]; then')
        stage_exit = source.index("exit 0", stage_branch)
        result_parent_creation = source.index('remote "mkdir -p \'$RESULT_PARENT\'"')
        slurm_submission = source.index("sbatch --parsable")
        self.assertIn("'$TERRA_EULER_USER|es_hutter'", source)
        self.assertNotIn("'$TERRA_EULER_USER|es_hutter|'", source)
        self.assertLess(stage_branch, stage_exit)
        self.assertLess(stage_exit, result_parent_creation)
        self.assertLess(result_parent_creation, slurm_submission)

    def test_runner_separates_architecture_fingerprints(self) -> None:
        source = (LAUNCHER / "run.sbatch").read_text(encoding="utf-8")
        self.assertIn("evaluate_treatment", source)
        self.assertIn(
            '"$GRU_U40000" "$GRU_U44000"',
            source,
        )
        self.assertIn(
            '"$FF_U44000" "$FF_U86000"',
            source,
        )
        self.assertNotIn(
            '"$GRU_U44000" "$FF_U44000"',
            source,
        )
        self.assertEqual(source.count("--accepted-panel promotion"), 1)
        self.assertIn("--horizon 450", source)
        self.assertIn("--require-productive-workspace-cycles", source)
        self.assertIn("NVIDIA GeForce RTX 4090", source)
        self.assertIn("assert EVAL_FORWARD_CHUNK == 120", source)
        self.assertIn("WANDB_MODE=disabled", source)
        self.assertIn("gru_training_source_revision=$GRU_SOURCE_REVISION", source)
        self.assertIn("ff_training_source_revision=$FF_SOURCE_REVISION", source)
        self.assertIn("lquota_home_used_gb.sh", source)
        self.assertIn("status=FAILED", source)
        self.assertIn("status=COMPLETED", source)
        self.assertEqual(
            source.count(
                '"$PYTHON" "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py"'
            ),
            2,
        )
        self.assertEqual(
            source.count(
                '"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py"'
            ),
            4,
        )
        self.assertIn("--review-limit 40", source)
        self.assertIn("dashboard_ff_u44_vs_gru_u44", source)
        self.assertIn("dashboard_gru_u40_vs_u44", source)
        self.assertIn("dashboard_ff_u86_vs_gru_u44", source)
        self.assertIn("--frame-stride 10", source)


if __name__ == "__main__":
    unittest.main()
