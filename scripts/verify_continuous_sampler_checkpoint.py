#!/usr/bin/env python3
"""Fail closed unless a checkpoint contains the complete V8 banded sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils import helpers
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings


def verify_sampler_state(state: object) -> dict:
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks pooled_sampler_state")
    if state.get("schema") != "terra_continuous_banded_sampler_state_v1":
        raise ValueError("checkpoint has the wrong sampler schema")
    settings = SamplerSettings(**state.get("settings", {}))
    if settings.rule != "continuous_banded_v1":
        raise ValueError("checkpoint has the wrong sampler rule")
    conditions = state.get("conditions")
    labels = state.get("labels")
    if not isinstance(conditions, list) or len(conditions) != 47:
        raise ValueError("checkpoint must contain exactly 47 conditions")
    if not isinstance(labels, dict) or set(labels) != set(conditions):
        raise ValueError("checkpoint sampler labels do not match its conditions")
    depths = [labels[name].get("curriculum_depth") for name in conditions]
    families = [labels[name].get("family") for name in conditions]
    if Counter(depths) != {0: 2, 1: 13, 2: 32}:
        raise ValueError("checkpoint sampler has the wrong depth graph")
    if Counter(families) != {"foundation": 25, "trench": 22}:
        raise ValueError("checkpoint sampler has the wrong family graph")
    for family in ("foundation", "trench"):
        if {
            depth
            for depth, observed_family in zip(depths, families)
            if observed_family == family
        } != {0, 1, 2}:
            raise ValueError(f"checkpoint sampler lacks a {family} depth")

    sampler = PooledConditionSampler(
        conditions,
        settings,
        maps_per_condition=state.get("maps_per_condition"),
        labels=labels,
    )
    sampler.restore_state_dict(state)
    probabilities = sampler.probabilities
    if not np.all(probabilities > 0.0):
        raise ValueError("checkpoint sampler lost positive condition support")
    return {
        "schema": "terra_continuous_banded_smoke_validation_v1",
        "passed": True,
        "condition_count": 47,
        "family_counts": dict(Counter(families)),
        "depth_counts": {str(key): value for key, value in Counter(depths).items()},
        "minimum_probability": float(probabilities.min()),
        "probability_sum": float(probabilities.sum()),
        "sampler_state_schema": state["schema"],
        "sampler_rule": settings.rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    helpers.register_checkpoint_config_classes()
    records = []
    for path in args.checkpoints:
        checkpoint = helpers.load_pkl_object(str(path))
        if checkpoint.get("next_update") != 1:
            raise ValueError(f"{path}: smoke checkpoint next_update must be 1")
        receipt = verify_sampler_state(checkpoint.get("pooled_sampler_state"))
        receipt.update(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        records.append(receipt)
    output = {
        "schema": "terra_continuous_banded_smoke_validation_v1",
        "passed": True,
        "checkpoints": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
