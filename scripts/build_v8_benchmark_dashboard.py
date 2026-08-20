#!/usr/bin/env python3
"""Build one self-contained review dashboard from two fixed-bank evaluations.

The fixed-bank JSON remains the scientific record.  This script only aligns the
two records, derives diagnostic labels, chooses representative replay slots,
and renders a portable HTML review surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "terra_v8_benchmark_dashboard_v1"
FIXED_SCHEMA = "terra_fixed_bank_eval_v4"
V8_TRAINING_CONDITION_COUNT = 47
PROMOTION_PANEL_OMISSIONS = ("fnd-slab-allfree", "trn-straight-allfree")


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_record(path: Path, index: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty fixed-evaluation list")
    try:
        record = payload[index]
    except IndexError as exc:
        raise ValueError(f"{path}: record index {index} is out of range") from exc
    if record.get("schema") != FIXED_SCHEMA:
        raise ValueError(f"{path}: expected schema {FIXED_SCHEMA}")
    rows = record.get("per_map")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: fixed-evaluation record has no per-map rows")
    return record


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["slot_index"]),
        row["episode_id"],
        row["map_id"],
        row.get("scenario_id"),
        row.get("source_id"),
        row["primary_cell"],
        row["family"],
        int(row["reset_seed"]),
        row.get("pair_slot_id"),
        row.get("environment_protocol_sha256"),
    )


def validate_pair(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    for label, record in (("reference", reference), ("candidate", candidate)):
        if record.get("deterministic") is not True:
            raise ValueError(f"{label} fixed evaluation is not deterministic")
        if not record.get("reset_verification", {}).get("passed"):
            raise ValueError(f"{label} reset verification did not pass")
        if not record.get("summary", {}).get("integrity", {}).get("passed"):
            raise ValueError(f"{label} integrity verification did not pass")
    for name in (
        "split",
        "stratum",
        "policy_mode",
        "completion_contract",
        "r2_protocol_receipt",
        "accepted_bank",
    ):
        if reference.get(name) != candidate.get(name):
            raise ValueError(f"fixed evaluations use different {name}")
    if reference.get("manifest_sha256") != candidate.get("manifest_sha256"):
        raise ValueError("fixed evaluations use different manifests")
    if reference.get("horizon") != candidate.get("horizon"):
        raise ValueError("fixed evaluations use different horizons")
    if reference.get("seed") != candidate.get("seed"):
        raise ValueError("fixed evaluations use different policy seeds")
    reference_rows = reference["per_map"]
    candidate_rows = candidate["per_map"]
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("fixed evaluations contain different numbers of maps")
    for offset, (before, after) in enumerate(zip(reference_rows, candidate_rows)):
        if _identity(before) != _identity(after):
            raise ValueError(f"fixed-panel identity mismatch at row {offset + 1}")
        if before.get("integrity_failure") or after.get("integrity_failure"):
            raise ValueError(f"integrity failure at fixed-panel slot {offset + 1}")


def _metric(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    return _finite(row.get(name, default), name)


def no_effect_rate(row: dict[str, Any]) -> float:
    return _metric(row, "no_effect_action_count") / max(int(row["steps"]), 1)


def issue_tags(row: dict[str, Any]) -> list[str]:
    """Return descriptive, non-causal issue tags in review priority order."""
    if bool(row["success"]):
        return []
    tags: list[str] = []
    if no_effect_rate(row) >= 0.50:
        tags.append("high no-effect")
    if _metric(row, "stall_age_saturated_decision_fraction") >= 0.50:
        tags.append("stall saturated")
    if _metric(row, "loaded_soil_fraction") > 0.01:
        tags.append("loaded endpoint")
    if _metric(row, "off_zone_staged_soil_fraction") > 0.01:
        tags.append("staged-soil residue")
    if _metric(row, "dig_fraction") >= 0.95:
        tags.append("near-finish cleanup")
    if _metric(row, "terminal_soil_fraction") < 0.50:
        tags.append("low terminal progress")
    return tags or ["other incomplete"]


def _outcome(before: dict[str, Any], after: dict[str, Any]) -> str:
    pair = bool(before["success"]), bool(after["success"])
    return {
        (False, True): "conversion",
        (True, False): "regression",
        (False, False): "persistent failure",
        (True, True): "persistent success",
    }[pair]


def aligned_rows(reference: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for before, after in zip(reference["per_map"], candidate["per_map"]):
        before_no_effect = no_effect_rate(before)
        after_no_effect = no_effect_rate(after)
        before_terminal = _metric(before, "terminal_soil_fraction")
        after_terminal = _metric(after, "terminal_soil_fraction")
        before_steps = int(before["steps"])
        after_steps = int(after["steps"])
        tags = issue_tags(after)
        result.append(
            {
                "slot": int(after["slot_index"]),
                "episode_id": after["episode_id"],
                "map_id": after["map_id"],
                "condition": after["primary_cell"],
                "family": after["family"],
                "outcome": _outcome(before, after),
                "issue": tags[0] if tags else "none",
                "issue_tags": tags,
                "reference": {
                    "success": bool(before["success"]),
                    "steps": before_steps,
                    "terminal_soil": before_terminal,
                    "dig": _metric(before, "dig_fraction"),
                    "off_zone": _metric(before, "off_zone_staged_soil_fraction"),
                    "loaded": _metric(before, "loaded_soil_fraction"),
                    "no_effect_rate": before_no_effect,
                    "stall_saturation": _metric(
                        before, "stall_age_saturated_decision_fraction"
                    ),
                    "workspace_cycles": before.get("productive_workspace_cycles"),
                    "max_carry": _metric(before, "maximum_carry_work_normalized"),
                },
                "candidate": {
                    "success": bool(after["success"]),
                    "steps": after_steps,
                    "terminal_soil": after_terminal,
                    "dig": _metric(after, "dig_fraction"),
                    "off_zone": _metric(after, "off_zone_staged_soil_fraction"),
                    "loaded": _metric(after, "loaded_soil_fraction"),
                    "no_effect_rate": after_no_effect,
                    "stall_saturation": _metric(
                        after, "stall_age_saturated_decision_fraction"
                    ),
                    "workspace_cycles": after.get("productive_workspace_cycles"),
                    "max_carry": _metric(after, "maximum_carry_work_normalized"),
                },
                "delta": {
                    "terminal_soil": after_terminal - before_terminal,
                    "steps": after_steps - before_steps,
                    "no_effect_rate": after_no_effect - before_no_effect,
                },
            }
        )
    return result


def condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        reference_exact = sum(row["reference"]["success"] for row in selected)
        candidate_exact = sum(row["candidate"]["success"] for row in selected)
        result.append(
            {
                "condition": condition,
                "family": selected[0]["family"],
                "maps": len(selected),
                "reference_exact": reference_exact,
                "candidate_exact": candidate_exact,
                "exact_delta": candidate_exact - reference_exact,
                "candidate_terminal_soil": sum(
                    row["candidate"]["terminal_soil"] for row in selected
                )
                / len(selected),
                "candidate_no_effect_rate": sum(
                    row["candidate"]["no_effect_rate"] for row in selected
                )
                / len(selected),
                "episode_outcomes": [
                    {
                        "slot": row["slot"],
                        "map_id": row["map_id"],
                        "outcome": row["outcome"],
                    }
                    for row in sorted(selected, key=lambda row: row["slot"])
                ],
            }
        )
    return result


def choose_review_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Choose behaviorally informative slots, not merely the lowest scores."""
    buckets = {
        "regression": [row for row in rows if row["outcome"] == "regression"],
        "persistent failure": [
            row for row in rows if row["outcome"] == "persistent failure"
        ],
        "conversion": [row for row in rows if row["outcome"] == "conversion"],
        "high-carry success": [
            row
            for row in rows
            if row["candidate"]["success"]
            and row["candidate"]["max_carry"] > 0.01
        ],
    }
    buckets["regression"].sort(
        key=lambda row: (
            -row["candidate"]["no_effect_rate"],
            -row["candidate"]["stall_saturation"],
            row["candidate"]["terminal_soil"],
            row["slot"],
        )
    )
    buckets["persistent failure"].sort(
        key=lambda row: (
            -row["candidate"]["no_effect_rate"],
            -row["candidate"]["stall_saturation"],
            row["candidate"]["terminal_soil"],
            row["slot"],
        )
    )
    buckets["conversion"].sort(
        key=lambda row: (
            -row["reference"]["no_effect_rate"],
            -row["reference"]["stall_saturation"],
            row["candidate"]["steps"],
            row["slot"],
        )
    )
    buckets["high-carry success"].sort(
        key=lambda row: (
            -row["candidate"]["max_carry"],
            row["candidate"]["steps"],
            row["slot"],
        )
    )
    high_carry_sorted = buckets["high-carry success"]
    high_carry_diverse = []
    high_carry_repeat = []
    high_carry_conditions: set[str] = set()
    for row in high_carry_sorted:
        if row["condition"] in high_carry_conditions:
            high_carry_repeat.append(row)
        else:
            high_carry_conditions.add(row["condition"])
            high_carry_diverse.append(row)
    buckets["high-carry success"] = high_carry_diverse + high_carry_repeat
    quotas = {
        "regression": max(2, limit // 4),
        "persistent failure": max(4, limit // 3),
        "conversion": max(3, limit // 4),
        "high-carry success": max(2, limit // 6),
    }
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for reason in (
        "regression",
        "persistent failure",
        "conversion",
        "high-carry success",
    ):
        for row in buckets[reason][: quotas[reason]]:
            if row["slot"] in seen:
                continue
            chosen.append(
                {
                    "slot": row["slot"],
                    "map_id": row["map_id"],
                    "condition": row["condition"],
                    "family": row["family"],
                    "reason": reason,
                    "outcome": row["outcome"],
                    "issue_tags": row["issue_tags"],
                }
            )
            seen.add(row["slot"])
            if len(chosen) >= limit:
                return chosen
    if len(chosen) < limit:
        remaining = sorted(
            (row for row in rows if row["slot"] not in seen),
            key=lambda row: (
                row["candidate"]["success"],
                -row["candidate"]["no_effect_rate"],
                row["candidate"]["terminal_soil"],
                row["slot"],
            ),
        )
        for row in remaining[: limit - len(chosen)]:
            chosen.append(
                {
                    "slot": row["slot"],
                    "map_id": row["map_id"],
                    "condition": row["condition"],
                    "family": row["family"],
                    "reason": "coverage fill",
                    "outcome": row["outcome"],
                    "issue_tags": row["issue_tags"],
                }
            )
    return chosen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dashboard_data(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
    reference_path: Path,
    candidate_path: Path,
    media_dir: Path | None,
    output_dir: Path,
    review_limit: int,
) -> dict[str, Any]:
    validate_pair(reference, candidate)
    rows = aligned_rows(reference, candidate)
    conditions = condition_rows(rows)
    selection = choose_review_rows(rows, review_limit)
    media: dict[str, dict[str, str]] = {}
    trace_summaries: dict[str, dict[str, Any]] = {
        "reference": {},
        "candidate": {},
    }
    media_timing: dict[str, tuple[int, int]] = {}
    if media_dir is not None:
        labels = {"reference": reference_label, "candidate": candidate_label}
        expected_checkpoint = {
            "reference": reference["checkpoint_sha256"],
            "candidate": candidate["checkpoint_sha256"],
        }
        for role, label in labels.items():
            receipt_path = media_dir / label / "receipt.json"
            if not receipt_path.is_file():
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not receipt.get("full_panel_terminal_parity_verified"):
                raise ValueError(f"{receipt_path}: media replay lacks terminal parity")
            if not receipt.get("full_panel_no_effect_count_parity_verified"):
                raise ValueError(f"{receipt_path}: media replay lacks no-effect-count parity")
            if receipt.get("checkpoint_sha256") != expected_checkpoint[role]:
                raise ValueError(f"{receipt_path}: media checkpoint mismatch")
            expected_result_path = (
                reference_path if role == "reference" else candidate_path
            )
            expected_record = reference if role == "reference" else candidate
            required_receipt_values = {
                "fixed_json_sha256": _sha256(expected_result_path),
                "manifest_sha256": expected_record["manifest_sha256"],
                "panel_maps": len(expected_record["per_map"]),
                "horizon": int(expected_record["horizon"]),
                "seed": int(expected_record["seed"]),
                "deterministic": True,
                "canonical_forward_chunk": 120,
                "selected_slots": [item["slot"] for item in selection],
            }
            for name, expected in required_receipt_values.items():
                if receipt.get(name) != expected:
                    raise ValueError(
                        f"{receipt_path}: media {name} mismatch: "
                        f"{receipt.get(name)!r} != {expected!r}"
                    )
            media_timing[role] = (
                int(receipt["frame_cadence_steps"]),
                int(receipt["frames_per_episode"]),
            )
            trace_summaries[role] = {
                str(row["slot"]): row["trace_summary"]
                for row in receipt.get("episodes", [])
            }
        if len(media_timing) == 2 and len(set(media_timing.values())) != 1:
            raise ValueError(
                f"reference/candidate media timing mismatch: {media_timing!r}"
            )
        for item in selection:
            slot = item["slot"]
            paths = {
                role: media_dir / label / f"slot_{slot:04d}.gif"
                for role, label in labels.items()
            }
            found = {
                key: os.path.relpath(path, start=output_dir).replace(os.sep, "/")
                for key, path in paths.items()
                if path.is_file()
            }
            if found:
                media[str(slot)] = found
    for row in rows:
        slot = str(row["slot"])
        row["reference"]["trace"] = trace_summaries["reference"].get(slot)
        row["candidate"]["trace"] = trace_summaries["candidate"].get(slot)
    reference_successes = sum(row["reference"]["success"] for row in rows)
    candidate_successes = sum(row["candidate"]["success"] for row in rows)
    counts = {
        name: sum(row["outcome"] == name for row in rows)
        for name in (
            "conversion",
            "regression",
            "persistent failure",
            "persistent success",
        )
    }
    return {
        "schema": SCHEMA,
        "labels": {"reference": reference_label, "candidate": candidate_label},
        "contract": {
            "manifest_sha256": reference["manifest_sha256"],
            "horizon": int(reference["horizon"]),
            "seed": int(reference["seed"]),
            "maps": len(rows),
            "panel_conditions": len(conditions),
            "training_conditions": V8_TRAINING_CONDITION_COUNT,
            "omitted_training_conditions": (
                list(PROMOTION_PANEL_OMISSIONS)
                if len(conditions) == V8_TRAINING_CONDITION_COUNT - 2
                else []
            ),
            "reference_checkpoint_sha256": reference["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "reference_checkpoint_update": reference.get("checkpoint_update"),
            "candidate_checkpoint_update": candidate.get("checkpoint_update"),
            "reference_result_sha256": _sha256(reference_path),
            "candidate_result_sha256": _sha256(candidate_path),
        },
        "summary": {
            "reference_successes": reference_successes,
            "candidate_successes": candidate_successes,
            "exact_delta": candidate_successes - reference_successes,
            **counts,
        },
        "conditions": conditions,
        "maps": rows,
        "review_selection": selection,
        "media": media,
    }


def write_episode_csv(data: dict[str, Any], path: Path) -> None:
    fields = (
        "slot",
        "episode_id",
        "map_id",
        "condition",
        "family",
        "outcome",
        "issue_tags",
        "reference_success",
        "candidate_success",
        "reference_steps",
        "candidate_steps",
        "reference_terminal_soil",
        "candidate_terminal_soil",
        "reference_dig",
        "candidate_dig",
        "reference_off_zone",
        "candidate_off_zone",
        "reference_loaded",
        "candidate_loaded",
        "reference_no_effect_rate",
        "candidate_no_effect_rate",
        "reference_stall_saturation",
        "candidate_stall_saturation",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in data["maps"]:
            writer.writerow(
                {
                    "slot": row["slot"],
                    "episode_id": row["episode_id"],
                    "map_id": row["map_id"],
                    "condition": row["condition"],
                    "family": row["family"],
                    "outcome": row["outcome"],
                    "issue_tags": ";".join(row["issue_tags"]),
                    "reference_success": row["reference"]["success"],
                    "candidate_success": row["candidate"]["success"],
                    "reference_steps": row["reference"]["steps"],
                    "candidate_steps": row["candidate"]["steps"],
                    "reference_terminal_soil": row["reference"]["terminal_soil"],
                    "candidate_terminal_soil": row["candidate"]["terminal_soil"],
                    "reference_dig": row["reference"]["dig"],
                    "candidate_dig": row["candidate"]["dig"],
                    "reference_off_zone": row["reference"]["off_zone"],
                    "candidate_off_zone": row["candidate"]["off_zone"],
                    "reference_loaded": row["reference"]["loaded"],
                    "candidate_loaded": row["candidate"]["loaded"],
                    "reference_no_effect_rate": row["reference"]["no_effect_rate"],
                    "candidate_no_effect_rate": row["candidate"]["no_effect_rate"],
                    "reference_stall_saturation": row["reference"]["stall_saturation"],
                    "candidate_stall_saturation": row["candidate"]["stall_saturation"],
                }
            )


def render_html(data: dict[str, Any]) -> str:
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    title = html.escape(
        f"Terra benchmark: {data['labels']['candidate']} vs {data['labels']['reference']}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f6f7; --fg:#182026; --muted:#66717a; --surface:#fff; --line:#d8dde1; --good:#16794a; --bad:#b42318; --accent:#175cd3; --warn:#a15c00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11161a; --fg:#edf1f3; --muted:#a5afb7; --surface:#192126; --line:#354149; --good:#5fd39a; --bad:#ff8b80; --accent:#82adff; --warn:#ffc067; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }}
main {{ max-width:1500px; margin:auto; padding:20px; }}
h1,h2 {{ font-weight:600; margin:0 0 12px; }}
h1 {{ font-size:24px; }} h2 {{ font-size:18px; margin-top:24px; }}
.sub {{ color:var(--muted); margin-bottom:16px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.stat,.detail {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.stat b {{ display:block; font-size:24px; }}
.good {{ color:var(--good); }} .bad {{ color:var(--bad); }} .warn {{ color:var(--warn); }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
input,select,button {{ color:var(--fg); background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:8px; }}
button {{ cursor:pointer; }} button.selected {{ outline:2px solid var(--accent); }}
.condition-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:6px; }}
.condition-card {{ background:var(--surface); border:1px solid var(--line); border-radius:7px; padding:7px; }}
.condition {{ width:100%; text-align:left; border:0; padding:2px; }} .condition span {{ display:block; color:var(--muted); }}
.episode-dots {{ display:grid; grid-template-columns:repeat(8,1fr); gap:3px; margin-top:6px; }}
.episode-dot {{ min-width:0; height:12px; padding:0; border:0; border-radius:2px; }}
.dot-persistent-success {{ background:#70c99b; }} .dot-conversion {{ background:#4d91ff; }}
.dot-regression {{ background:#e45c51; }} .dot-persistent-failure {{ background:#9ba3a8; }}
.legend {{ display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); margin-bottom:8px; }}
.legend i {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:4px; }}
.layout {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(320px,1fr); gap:14px; align-items:start; }}
.table-wrap {{ overflow:auto; max-height:720px; border:1px solid var(--line); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; white-space:nowrap; }}
th {{ position:sticky; top:0; background:var(--surface); z-index:1; }} tr {{ cursor:pointer; }} tr:hover {{ background:color-mix(in srgb,var(--accent) 8%,transparent); }}
.detail {{ position:sticky; top:12px; }} .metrics {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }}
.metrics div {{ border-bottom:1px solid var(--line); padding:5px 0; }} .metrics b {{ display:block; }}
.tag {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; margin:2px 3px 2px 0; }}
.media {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }} .media img {{ width:100%; border:1px solid var(--line); }}
code {{ overflow-wrap:anywhere; }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .detail {{ position:static; }} }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<div class="sub" id="contract"></div>
<div class="stats" id="stats"></div>
<h2>Condition overview</h2>
<div class="legend"><span><i class="dot-persistent-success"></i>both exact</span><span><i class="dot-conversion"></i>candidate conversion</span><span><i class="dot-regression"></i>candidate regression</span><span><i class="dot-persistent-failure"></i>both fail</span></div>
<div class="condition-grid" id="conditions"></div>
<h2>Map explorer</h2>
<div class="controls">
  <input id="search" type="search" placeholder="slot, map, or condition">
  <select id="family"><option value="">all families</option></select>
  <select id="outcome"><option value="">all outcomes</option></select>
  <select id="issue"><option value="">all issue tags</option></select>
  <select id="sort"><option value="review">review priority</option><option value="slot">slot</option><option value="terminal">candidate terminal soil</option><option value="noeffect">candidate no-effect</option></select>
  <button id="clear" type="button">clear filters</button>
</div>
<div class="layout"><div class="table-wrap"><table><thead><tr><th>slot</th><th>condition</th><th>outcome</th><th>issue</th><th>exact R→C</th><th>terminal R→C</th><th>no-effect R→C</th><th>steps R→C</th></tr></thead><tbody id="maps"></tbody></table></div><aside class="detail" id="detail">Select a map.</aside></div>
<script>
const DATA={encoded};
const $=id=>document.getElementById(id);
const pct=x=>(100*x).toFixed(1)+'%';
const signed=x=>(x>0?'+':'')+x;
const reviewRank=new Map(DATA.review_selection.map((x,i)=>[x.slot,i]));
let selectedCondition='';
function options(id, values){{ for(const value of [...new Set(values)].sort()){{const o=document.createElement('option');o.value=value;o.textContent=value;$(id).appendChild(o);}} }}
options('family',DATA.maps.map(x=>x.family)); options('outcome',DATA.maps.map(x=>x.outcome)); options('issue',DATA.maps.flatMap(x=>x.issue_tags));
$('contract').textContent=`${{DATA.contract.maps}} maps · ${{DATA.contract.panel_conditions}}/${{DATA.contract.training_conditions}} training conditions · horizon ${{DATA.contract.horizon}} · seed ${{DATA.contract.seed}} · manifest ${{DATA.contract.manifest_sha256.slice(0,12)}}…${{DATA.contract.omitted_training_conditions.length?' · omitted: '+DATA.contract.omitted_training_conditions.join(', '):''}}`;
const s=DATA.summary; const n=DATA.contract.maps;
$('stats').innerHTML=`<div class="stat"><span>${{DATA.labels.reference}}</span><b>${{s.reference_successes}}/${{n}}</b></div><div class="stat"><span>${{DATA.labels.candidate}}</span><b>${{s.candidate_successes}}/${{n}}</b></div><div class="stat"><span>net exact</span><b class="${{s.exact_delta>=0?'good':'bad'}}">${{signed(s.exact_delta)}}</b></div><div class="stat"><span>conversions / regressions</span><b>${{s.conversion}} / ${{s.regression}}</b></div>`;
function drawConditions(){{const root=$('conditions');root.innerHTML='';for(const c of DATA.conditions){{const card=document.createElement('div');card.className='condition-card';const b=document.createElement('button');b.type='button';b.className='condition'+(selectedCondition===c.condition?' selected':'');b.innerHTML=`<b>${{c.condition}}</b><span>${{c.candidate_exact}}/${{c.maps}} vs ${{c.reference_exact}}/${{c.maps}} · Δ ${{signed(c.exact_delta)}}</span>`;b.onclick=()=>{{selectedCondition=selectedCondition===c.condition?'':c.condition;drawConditions();drawMaps();}};card.appendChild(b);const dots=document.createElement('div');dots.className='episode-dots';for(const e of c.episode_outcomes){{const d=document.createElement('button');d.type='button';d.className='episode-dot dot-'+e.outcome.replaceAll(' ','-');d.title=`slot ${{e.slot}} · ${{e.map_id}} · ${{e.outcome}}`;d.onclick=()=>{{const row=DATA.maps.find(x=>x.slot===e.slot);if(row)drawDetail(row);}};dots.appendChild(d);}}card.appendChild(dots);root.appendChild(card);}}}}
function filtered(){{const q=$('search').value.trim().toLowerCase();let rows=DATA.maps.filter(x=>(!selectedCondition||x.condition===selectedCondition)&&(!$('family').value||x.family===$('family').value)&&(!$('outcome').value||x.outcome===$('outcome').value)&&(!$('issue').value||x.issue_tags.includes($('issue').value))&&(!q||`${{x.slot}} ${{x.map_id}} ${{x.condition}}`.toLowerCase().includes(q)));const sort=$('sort').value;rows.sort((a,b)=>sort==='slot'?a.slot-b.slot:sort==='terminal'?a.candidate.terminal_soil-b.candidate.terminal_soil:sort==='noeffect'?b.candidate.no_effect_rate-a.candidate.no_effect_rate:(reviewRank.get(a.slot)??9999)-(reviewRank.get(b.slot)??9999)||a.slot-b.slot);return rows;}}
function outcomeClass(x){{return x==='conversion'?'good':x==='regression'?'bad':x==='persistent failure'?'warn':'';}}
function drawMaps(){{const body=$('maps');body.innerHTML='';for(const x of filtered()){{const tr=document.createElement('tr');tr.innerHTML=`<td>${{x.slot}}</td><td>${{x.condition}}</td><td class="${{outcomeClass(x.outcome)}}">${{x.outcome}}</td><td>${{x.issue}}</td><td>${{x.reference.success?'✓':'×'}}→${{x.candidate.success?'✓':'×'}}</td><td>${{pct(x.reference.terminal_soil)}}→${{pct(x.candidate.terminal_soil)}}</td><td>${{pct(x.reference.no_effect_rate)}}→${{pct(x.candidate.no_effect_rate)}}</td><td>${{x.reference.steps}}→${{x.candidate.steps}}</td>`;tr.onclick=()=>drawDetail(x);body.appendChild(tr);}}}}
function metric(label,a,b,format=pct){{return `<div><span>${{label}}</span><b>${{format(a)}}</b><small>reference</small></div><div><span>${{label}}</span><b>${{format(b)}}</b><small>candidate</small></div><div><span>delta</span><b>${{format(b-a)}}</b><small>${{label}}</small></div>`;}}
function drawDetail(x){{
  const m=DATA.media[String(x.slot)]||{{}};
  const tags=x.issue_tags.length?x.issue_tags.map(t=>`<span class="tag">${{t}}</span>`).join(''):'<span class="tag">success</span>';
  const media=(m.reference||m.candidate)?`<div class="media">${{m.reference?`<div><b>${{DATA.labels.reference}}</b><img src="${{m.reference}}" alt="reference rollout for slot ${{x.slot}}"></div>`:''}}${{m.candidate?`<div><b>${{DATA.labels.candidate}}</b><img src="${{m.candidate}}" alt="candidate rollout for slot ${{x.slot}}"></div>`:''}}</div>`:'<p class="sub">No rendered trace yet. This slot remains in review_selection.json.</p>';
  const trace=(role,label)=>{{const t=x[role].trace;if(!t)return '';const obsCycle=t.terminal_observation_action_cycle;const stateCycle=t.terminal_recurrent_state_action_cycle;return `<p><b>${{label}} trace:</b> no-effect streak ${{t.maximum_no_effect_streak}}, repeated instantaneous inputs ${{t.repeated_instantaneous_input_decisions}}, last material change ${{t.last_material_change_step}}${{obsCycle?`, observation/action cycle p${{obsCycle.period}} for ${{obsCycle.decisions}} decisions`:''}}${{stateCycle?`, full recurrent-state cycle p${{stateCycle.period}} for ${{stateCycle.decisions}} decisions`:''}}</p>`;}};
  $('detail').innerHTML=`<h2>slot ${{x.slot}} · ${{x.map_id}}</h2><p>${{x.condition}} · <span class="${{outcomeClass(x.outcome)}}">${{x.outcome}}</span></p><p>${{tags}}</p><div class="metrics">${{metric('terminal soil',x.reference.terminal_soil,x.candidate.terminal_soil)}}${{metric('no-effect',x.reference.no_effect_rate,x.candidate.no_effect_rate)}}${{metric('stall saturation',x.reference.stall_saturation,x.candidate.stall_saturation)}}${{metric('off-zone soil',x.reference.off_zone,x.candidate.off_zone)}}${{metric('loaded soil',x.reference.loaded,x.candidate.loaded)}}${{metric('steps',x.reference.steps,x.candidate.steps,v=>String(Math.round(v)))}}</div>${{trace('reference',DATA.labels.reference)}}${{trace('candidate',DATA.labels.candidate)}}${{media}}<p class="sub"><code>${{x.episode_id}}</code></p>`;
}}
for(const id of ['search','family','outcome','issue','sort']) $(id).addEventListener(id==='search'?'input':'change',drawMaps);
$('clear').onclick=()=>{{selectedCondition='';for(const id of ['search','family','outcome','issue'])$(id).value='';$('sort').value='review';drawConditions();drawMaps();}};
drawConditions();drawMaps();if(DATA.review_selection.length){{const first=DATA.maps.find(x=>x.slot===DATA.review_selection[0].slot);if(first)drawDetail(first);}}
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, default=-1)
    parser.add_argument("--candidate-index", type=int, default=-1)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--review-limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.review_limit < 1:
        raise ValueError("--review-limit must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    reference_path = args.reference_json.resolve()
    candidate_path = args.candidate_json.resolve()
    reference = load_record(reference_path, args.reference_index)
    candidate = load_record(candidate_path, args.candidate_index)
    data = build_dashboard_data(
        reference,
        candidate,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        reference_path=reference_path,
        candidate_path=candidate_path,
        media_dir=args.media_dir.resolve() if args.media_dir else None,
        output_dir=output_dir,
        review_limit=args.review_limit,
    )
    output_dir.mkdir(parents=True)
    (output_dir / "dashboard_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review_selection.json").write_text(
        json.dumps(data["review_selection"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_episode_csv(data, output_dir / "episodes.csv")
    (output_dir / "index.html").write_text(render_html(data), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
