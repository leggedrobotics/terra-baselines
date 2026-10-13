"""Evaluate score-independent checkpoints after both 24-hour training arms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_structured import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=13500)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    source = Path(__file__).resolve().parents[2]
    output = args.campaign / "evaluation"
    output.mkdir(exist_ok=True)
    histories, selected = {}, {}
    for arm in ("primitive8", "structured"):
        folder = args.campaign / arm / "production"
        history = {int(path.stem.rsplit("_", 1)[1]): path
                   for path in folder.glob("checkpoint_update_*.pkl")}
        histories[arm] = history
        final = folder / "checkpoint.pkl"
        payload = load_checkpoint(final)
        final_update = int(payload["next_update"])
        assert payload["training_config"]["action_interface"] == arm
        del payload
        selected[arm] = {"final": (final_update, final)}
        if 0 in history:
            midpoint_time = (history[0].stat().st_mtime + final.stat().st_mtime) / 2
            middle = [u for u in history if 0 < u < final_update]
            if middle:
                update = min(middle, key=lambda u: abs(history[u].stat().st_mtime - midpoint_time))
                selected[arm]["midpoint"] = (update, history[update])
            selected[arm]["initial"] = (0, history[0])
    common = (set(histories["primitive8"]) & set(histories["structured"])) - {0}
    if common:
        for arm in selected:
            update = max(common)
            selected[arm]["common_update"] = (update, histories[arm][update])
    report = {"selection": {arm: {label: {"update": u, "checkpoint": str(p)}
                for label, (u, p) in picks.items()} for arm, picks in selected.items()}, "results": []}
    report_path = output / "evaluation_status.json"
    seen = {}
    # Preserve the final comparison if the bounded allocation cannot fit all
    # secondary panels. No checkpoint is selected using an evaluation score.
    tasks = [(label, decoder, arm)
             for label in ("final", "midpoint", "common_update", "initial")
             for decoder in ("greedy", "sampled")
             for arm in ("primitive8", "structured") if label in selected[arm]]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for label, decoder, arm in tasks:
        update, checkpoint = selected[arm][label]
        key = (arm, update, decoder)
        result = dict(arm=arm, label=label, update=update, decoder=decoder)
        if key in seen:
            result.update(status="reused", output=seen[key])
            report["results"].append(result)
            continue
        remaining = deadline - time.monotonic()
        if remaining < 60:
            result["status"] = "budget_exhausted"
            report["results"].append(result)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            return 2
        destination = output / arm / label / f"{decoder}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(source / "eval_structured.py"),
                   "--checkpoint", str(checkpoint), "--initial-states", str(args.campaign / "inputs/eval_states.pkl"),
                   "--action-interface", arm, "--seed", "20261013",
                   "--policy-snapshot-interval", "32", "--output", str(destination)]
        if decoder == "sampled":
            command.append("--sampled")
        print(json.dumps({**result, "phase": "starting"}), flush=True)
        try:
            with destination.with_suffix(".log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=remaining - 30, env=os.environ.copy())
            result.update(status="complete" if completed.returncode == 0 else "failed",
                          exit_code=completed.returncode, output=str(destination))
        except subprocess.TimeoutExpired:
            result.update(status="timeout", output=str(destination))
        report["results"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if result["status"] != "complete":
            return 1
        seen[key] = str(destination)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
