from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "euler_v8_gru_media_recovery_v1"


class V8GruMediaRecoveryLauncherTest(unittest.TestCase):
    def test_dry_run_is_local_and_pins_completed_inputs(self) -> None:
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
        self.assertIn("SUBMIT=0: recovery contract printed; no SSH", result.stdout)
        for expected in (
            "11303967",
            "74f72a65659353a6b4b2d163904dcbf60987805c",
            "83440b8f1b01f5d4d3b217da4e8c08a5bc7c60ab1b76483680f78cf6c5e576e2",
            "c0c53f54ee2d282c8cd5e4151e52ac3910c449cc909c5ee70c16a20965e800e5",
            "675436d00ed6a156bfa1a00a325141c6fde98f52f09da85b02e21c7df9f93070",
        ):
            self.assertIn(expected, result.stdout)

    def test_runner_is_media_only_and_fail_closed(self) -> None:
        source = (LAUNCHER / "run.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("eval_fixed_bank.py", source)
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
            3,
        )
        self.assertIn("fixed_evaluations_rerun=false", source)
        self.assertIn('grep -Fx "status=FAILED"', source)
        self.assertIn("assert EVAL_FORWARD_CHUNK == 120", source)
        self.assertIn("NVIDIA GeForce RTX 4090", source)
        self.assertIn('receipt["reset_verification"]', source)
        self.assertIn('"dashboard_ff_u44_vs_gru_u44": 22', source)
        self.assertIn('"dashboard_gru_u40_vs_u44": -3', source)
        self.assertIn('"dashboard_ff_u86_vs_gru_u44": 4', source)

    def test_stage_exits_before_result_directory_creation(self) -> None:
        source = (LAUNCHER / "submit.sh").read_text(encoding="utf-8")
        stage_branch = source.index('if [ "$SUBMIT" = stage ]; then')
        stage_exit = source.index("exit 0", stage_branch)
        result_parent_creation = source.index('remote "mkdir -p \'$RESULT_PARENT\'"')
        slurm_submission = source.index("sbatch --parsable")
        self.assertLess(stage_branch, stage_exit)
        self.assertLess(stage_exit, result_parent_creation)
        self.assertLess(result_parent_creation, slurm_submission)


if __name__ == "__main__":
    unittest.main()
