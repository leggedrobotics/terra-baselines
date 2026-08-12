# continuous_banded_v3: pooled frontier with a mass cap (design, 2026-08-12)

One sampler rule change, motivated by two defects measured on the
`reward_v2_scratch` phase1 run. Implementation lands on
`experiment/v8-v6-yolo-rv2-20260810`; no launcher is wired to it yet —
adoption happens per new run contract, never mid-run.

## The two problems

**P1 — the family-pin monopoly (measured).** v1/v2 pin assignment mass 50/50
between foundation and trench before any frontier logic runs. Graduated
conditions drop to the replay floor, and their mass is redistributed only
*within their family*. When a family is nearly mastered, its entire half of
the budget funnels onto its last stragglers: by u13.4k the baseline had
mastered all trenches except `trn-net4-side1-road`, which then received
**45.2% of all sampling**, while 23 unmastered foundations split ~3.4% each.
Target-distribution ESS collapsed 34.5 → 10.3 and panel conversion flattened
+69/1k → +11/1k in the same window, with every reward term flat (the
attribution is by elimination with timing; see the reward-terms analysis of
2026-08-11). The family split is allocation ceremony inherited from the
staged-gate era (family-level pass thresholds needed family-balanced
exposure); per-condition mastery already governs exposure, so the boundary
does nothing except create this failure mode. It is the mirror image of the
v1 depth-pinning defect that motivated v2.

**P2 — end-game waste on unlearnable stragglers (projected from data).** Any
pure frontier rule concentrates mass on whatever remains unmastered. If the
remainder is effectively unlearnable at the current horizon/reward — five
foundation conditions produced **zero successes in 13.4k updates** — the
sampler pours the frontier mass into walls indefinitely. Removing the family
pin alone (P1 fix) makes this *worse* at the very end: three stragglers left
would take ~30% each.

## The rule

```
w_c   = DEPTH_PRIORITY_BASE ** (2 - depth_c)          # base 2.0, as v2
p_c   = 0.10 / N                                       # uniform floor, all N conditions
      + 0.90 * w_c / Σ_{unmastered} w                  # pooled frontier, NO family split
then: cap every p_c at MAX_CONDITION_MASS = 0.15;
      redistribute excess proportionally over the remaining uncapped
      unmastered conditions; iterate (bounded); if all unmastered are
      capped, spill the remainder uniformly over the conditions that are
      not capped (the mastered replay set), so the cap always holds; if
      nothing is left to spill onto (0.15 * N < 1), fall back to uniform.
all conditions mastered -> uniform over all (maintenance regime).
```

Unchanged from v2: graduation (EMA ≥ 0.80, ≥ 32 episodes, any depth),
demotion (< 0.65 rejoins the frontier — this remains the forgetting
recovery), refresh interval, windows, the 10% floor, depth priority. Family
becomes reporting metadata only (receipts/telemetry keep it; allocation
ignores it).

## What each mechanism buys

- **Pooling (P1):** in the measured end-state, 45.2%-on-one-cell becomes
  ~3–4% across the 24 open conditions — mass follows need, not accounting.
- **The 15% cap (P1 + P2):** bounds any single condition under *any*
  mastery configuration (would have prevented the monopoly even with the
  family pin), and bounds end-game waste: unlearnable stragglers absorb at
  most 15% each, the remainder returning to maintenance replay of the
  mastered set. Chosen over learnability/progress-rate weighting (PLR-style)
  deliberately: one constant, no estimator, no new failure modes; the
  fancier rule stays in reserve if capping proves insufficient.
- **Uniform-at-completion:** the natural trigger point for reward-fade /
  stopping decisions; unchanged semantics from v2.

## Migration and compatibility

- One-way v2→v3 checkpoint migration, exactly analogous to v1→v2: stored
  probabilities validated under the stored rule, recomputed under v3;
  mastery/windows/RNG carried over.
- v1 and v2 behavior byte-identical (regression-guarded in
  `tests/test_continuous_banded_sampler.py`); the cap exists only under v3.

## Deployment discipline

- Enters through new contracts only: the reward pilot, v6.1 follow-ups, and
  (Lorenzo's explicit call) the baseline phase2 continuation — switching
  there trades continuation comparability for not spending ~45% of compute
  on a mastered-adjacent cell; if switched, it is declared as part of the
  phase2 treatment.
- Arms currently in flight stay on v2 so their pairing remains internally
  fair (they share the defect equally).

## Open questions

- `MAX_CONDITION_MASS = 0.15` is an engineering constant, not tuned; the
  monopoly evidence says "well below 45%", the end-game argument says "well
  above the ~2% uniform share". Revisit only on evidence.
- Whether the five never-succeeded foundations are walls (horizon/reward)
  or merely late is the D1-style question the cap makes cheap to defer.
