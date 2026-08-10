"""Float32-aware lift-conservation diagnostics for the D4a replay."""

from __future__ import annotations

from typing import Any

import numpy as np

LIFT_ABSOLUTE_FLOOR = 1e-6
# H before and after are independently grouped float32 reductions followed by
# float32 carry additions.  Four representable spacings cover that small
# reduction/addition rounding budget without granting a fixed large tolerance
# to low-work traces.
LIFT_ULP_MULTIPLIER = 4.0


def lift_conservation_diagnostic(
    h_before: float,
    h_after: float,
    *,
    slot_index: int,
    step: int,
    targeted_label: str | None,
) -> dict[str, Any]:
    """Describe one lift residual and apply the scale-aware four-ULP gate."""
    before = np.float32(h_before)
    after = np.float32(h_after)
    absolute_residual = abs(float(before) - float(after))
    spacing_before = abs(float(np.spacing(before)))
    spacing_after = abs(float(np.spacing(after)))
    max_spacing = max(spacing_before, spacing_after)
    tolerance = max(LIFT_ABSOLUTE_FLOOR, LIFT_ULP_MULTIPLIER * max_spacing)
    ulp_residual = absolute_residual / max_spacing if max_spacing > 0 else 0.0
    magnitude = max(abs(float(before)), abs(float(after)))
    relative_residual = absolute_residual / magnitude if magnitude > 0 else 0.0
    return {
        "slot_index": slot_index,
        "step": step,
        "targeted_label": targeted_label,
        "dtype": "float32",
        "h_before": float(before),
        "h_after": float(after),
        "absolute_residual": absolute_residual,
        "relative_residual": relative_residual,
        "spacing_before": spacing_before,
        "spacing_after": spacing_after,
        "max_spacing": max_spacing,
        "ulp_residual": ulp_residual,
        "tolerance": tolerance,
        "passed": absolute_residual <= tolerance,
    }
