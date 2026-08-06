import json
from pathlib import Path

from scripts.v8_full_v8_report import build_report, write_report


def _record(update: int, completion: float) -> dict:
    rows = []
    for family, condition in (
        ("foundation", "f-condition"),
        ("trench", "t-condition"),
    ):
        for index in range(16):
            value = 1.0 if index == 0 else completion
            rows.append(
                {
                    "primary_cell": condition,
                    "family": family,
                    "terminal_absolute": value,
                    "success": value == 1.0,
                }
            )
    return {
        "checkpoint_update": update,
        "completion_contract": "exact_visible_dump_v1",
        "deterministic": True,
        "policy_mode": "deterministic",
        "exact_manifest_enumeration": True,
        "horizon": 450,
        "summary": {
            "integrity": {"passed": True},
            "overall": {"successes": 2, "episodes": 32},
        },
        "reset_verification": {"passed": True},
        "per_map": rows,
    }


def test_whole_v8_report_keeps_exact_macro_family_and_condition_views(tmp_path: Path):
    paths = {}
    for policy, completion in (("compact", 0.5), ("10m", 0.6)):
        for panel in ("promotion", "development"):
            path = tmp_path / f"{policy}_{panel}.json"
            path.write_text(json.dumps([_record(4000, completion)]))
            paths[(policy, panel)] = path

    report = build_report(paths)
    assert report["primary_metric"] == "fixed/development/exact_success_rate"
    assert len(report["policy_rows"]) == 4
    assert len(report["condition_rows"]) == 8
    treatment = next(
        row
        for row in report["policy_rows"]
        if row["policy"] == "10m" and row["panel"] == "development"
    )
    assert treatment["exact_success_rate"] == 2 / 32
    assert (
        treatment["foundation_macro_completion"] == treatment["trench_macro_completion"]
    )

    output = tmp_path / "report"
    write_report(report, output)
    assert "active-stage diagnostic" in (output / "LEADERBOARD.md").read_text()
    assert (output / "per_condition.csv").is_file()
