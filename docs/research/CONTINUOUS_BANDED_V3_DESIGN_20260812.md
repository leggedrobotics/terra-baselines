# continuous_banded_v3: v2 under a per-condition mass cap (design, 2026-08-12)

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
**45.2% of all sampling**. The other 23 open conditions were not uniform
either — the five unmastered depth-1 foundations took 3.41% each and the
eighteen depth-2 foundations 1.81% each — so one cell outran the whole
open frontier by more than an order of magnitude. Target-distribution ESS
at that state is **4.62** of 47 conditions. Panel conversion flattened
+69/1k → +11/1k in the same window with every reward term flat, which is
*consistent with* the allocation collapse but not yet causally attributed
(the argument is elimination plus timing; see the reward-terms analysis of
2026-08-11). One cell taking 45% is a defect on its own terms regardless of
how that window's conversion is finally explained. It is the mirror image
of the v1 depth-pinning defect that motivated v2.

**P2 — end-game waste on unlearnable stragglers (projected from data).** Any
frontier rule concentrates mass on whatever remains unmastered. If the
remainder is effectively unlearnable at the current horizon/reward — five
foundation conditions produced **zero successes in 13.4k updates** — the
sampler pours the frontier mass into walls indefinitely.

## Why pooling was rejected

The first draft of this design removed the family split entirely and pooled
the frontier across families. External review rejected it on a decisive
counterexample: pooling fixes the *per-cell* symptom while making the
*collective* one worse. With the five never-succeeded foundations as the
only open conditions, the family halves hold their joint mass to **46%**,
because the mastered trench family keeps its own half for replay. Pooling
hands that half to the same five walls, and with a 15% per-cell cap they
take **75%** jointly — the sampler spends three quarters of the population
on conditions it has never once solved. Pooling also swings the family
split hard: on the u13.5k state it yields **92.2/7.8** foundation/trench,
abandoning trench maintenance, versus 77.6/22.4 under the capped rule.

The family halves are therefore doing real work: they are a persistent
category budget, not staged-gate residue. That is the standard shape in
this literature — AlphaStar's league mixes 35/50/15 across main, league,
and exploiter channels, and OpenAI Five holds 80/20 self-play to past
opponents — persistent category budgets with capping *within* a category,
which is exactly what v3 does.

## The rule

v3 is v2 plus one cap. The whole rule:

```
p     = the v2 distribution (family halves, 10% floor, depth priority)
if p.max() <= settings.max_mass:  return p unchanged      # v3 == v2 here
else: cap the offenders at max_mass and redistribute the excess
      proportionally over the uncapped conditions, ignoring family
      boundaries when redistributing; iterate until nothing exceeds
      the cap (water-fill, bounded by the condition count).
```

`max_mass` is the serialized sampler setting, frozen at 0.15 for every
continuous rule, so a checkpoint carries the cap it was trained under
rather than inheriting whatever a later constant says. An infeasible
combination (`max_mass * N < 1`) is rejected at sampler construction and
raises inside the water-fill; it never silently returns an over-cap
vector.

Everything else is v2 verbatim: family halves, the 10% floor, depth
priority, graduation (EMA ≥ 0.80, ≥ 32 episodes, any depth), demotion
(< 0.65 rejoins the frontier), refresh interval, windows, and
uniform-per-family at completion.

## What each mechanism buys

- **The cap (P1):** on the measured u13.5k state, 45.2% on one cell
  becomes exactly 15.0%, ESS 4.62 → **19.62** of 47, and the excess crosses
  into the family that actually has open work (77.6/22.4 foundation/trench
  instead of the pinned 50/50). Below the cap v3 *is* v2, so the rule only
  acts in the regime it was added for.
- **The cap again (P2):** unlearnable stragglers absorb at most 15% each,
  and the family halves keep their collective share bounded (46%, not the
  75% pooling would give). Chosen over learnability/progress-rate weighting
  (PLR-style) deliberately: one constant, no estimator, no new failure
  modes; the fancier rule stays in reserve if capping proves insufficient.
- **Global redistribution:** the excess is population mass, not family
  mass, so it is water-filled proportionally across all uncapped
  conditions rather than returned to the family that produced it.

## Migration and compatibility

- One-way v2→v3 checkpoint migration, analogous to v1→v2 (and v1→v3 by the
  same path): stored probabilities validated under the stored rule,
  recomputed under v3; mastery/windows/RNG carried over.
- Migration is permitted **only at an empty window boundary**. Windows drive
  graduation, so a window must never mix exposure taken under two rules; a
  mid-window migration raises instead of averaging the two regimes.
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

- `max_mass = 0.15` is an engineering constant, not tuned; the monopoly
  evidence says "well below 45%", the end-game argument says "well above the
  ~2% uniform share". Revisit only on evidence.
- Whether the five never-succeeded foundations are walls (horizon/reward)
  or merely late is the D1-style question the cap makes cheap to defer.
