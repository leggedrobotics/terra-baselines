# Continuous Banded v3 (design, 2026-08-12)

Continuous Banded v3 is the one supported V8 curriculum. It removes semantic
foundation/trench quotas and combines an open-condition frontier with a small
global replay pool:

```text
p = 0.80 * depth_weighted(all open conditions)
  + 0.20 * uniform(all mastered conditions)
```

Open-condition depth weights are `4:2:1` for immutable curriculum depths
0/1/2. A 15% per-condition water-fill cap prevents any single condition from
monopolizing the population. Foundation and trench remain reporting slices;
they do not affect assignment probability.

## Why this rule

The reward-v2 phase-1 sampler exposed two allocation failures:

1. Near update 13,500, one remaining trench condition received 45.2% of all
   assignment mass. Target-distribution effective sample size fell to 4.62 of
   47. A per-condition cap is warranted even without attributing the
   contemporaneous learning slowdown to the sampler.
2. At the selected v6.1 update-14,000 checkpoint, all 22 trench conditions and
   only 7 of 25 foundation conditions were mastered. A fixed family split
   nevertheless spent half the population on trenches, putting 51.4% of total
   mass on mastered conditions while all 18 open conditions were foundations.
   The maximum condition mass was only 4.11%, so a cap alone could not repair
   this completed-family waste.

The final rule addresses both defects directly: the open/replay split releases
mass from a completed semantic family, while the cap bounds a single
straggler. It does not add a learned teacher, progress model, or mixture-model
framework.

## Exact state transition and allocation

A condition becomes mastered only after at least 32 completed training
episodes and an exact-success EMA of at least 0.80. A mastered condition
returns to the open pool if its EMA falls below 0.65. The EMA coefficient is
0.30 and the sampler refreshes every 150 PPO updates. Held-out evaluation never
updates sampler state.

At a refresh:

1. update mastery from exact completed training episodes;
2. give open conditions 80% total mass with depth weights `4:2:1`;
3. give mastered conditions 20% total mass uniformly;
4. if either pool is empty, give all mass to the other pool;
5. water-fill any probability above 0.15 across uncapped conditions.

All 47 conditions retain positive support. If every condition is mastered,
the distribution is uniform. An infeasible cap raises rather than silently
returning an invalid distribution.

At the exact update-14,000 state, v3 assigns 80.0% to the 18 open foundations,
4.83% to seven mastered foundations, and 15.17% to 22 mastered trenches. Each
open depth-1 condition receives 6.96%, each open depth-2 condition 3.48%, and
each mastered condition 0.69%. The maximum is 6.96% and effective sample size
is 24.21.

## Checkpoint conversion used by the continuation

The selected update-14,000 checkpoint was saved 50 updates into a 150-update
sampler window. Its one-off offline materializer validates that source state,
preserves mastery, competence, the closed window, refresh schedule and NumPy
RNG, clears only the unfinished window, and writes a native v3 checkpoint.
Runtime training accepts only native v3 state; it has no sampler-migration
mode.

Clearing the partial window discards 102,400 assignments, 13,725 completed
episodes, 13,725 reset exposures, 3,276,800 transitions and 8,810 exact
successes from sampler bookkeeping only. It does not alter policy parameters,
Adam state, optimizer/update clocks, or previously closed curriculum history.
The next refresh remains update 14,100.

## Evidence scope

The 80/20 split and 15% cap are provisional engineering constants, not values
copied from the literature and not claimed optima. The allocation replay proves
the intended geometry, not improved policy learning. The v6.1 continuation is
a practical combined observation-and-curriculum run, not a clean sampler
ablation; fixed source-disjoint panels decide whether v3 improves learning and
retains trench capability.

If v3 still parks on flat conditions, the next evidence-driven extension would
be a per-condition learning-progress or staleness score. That is deliberately
not implemented before a v3 trajectory demonstrates the need.

## Literature basis

Related automatic-curriculum work supports two broad principles: prioritize
tasks with current learning value, and retain a coverage or revisitation
channel. It does not prescribe Terra's constants.

- Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. [Prioritized Level
  Replay](https://proceedings.mlr.press/v139/jiang21b.html). ICML, 2021.
- Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. [Teacher
  algorithms for curriculum learning of Deep RL in continuously parameterized
  environments](https://proceedings.mlr.press/v100/portelas20a.html). CoRL,
  2020.
- Pascal Klink, Carlo D'Eramo, Jan Peters, and Joni Pajarinen. [Self-Paced Deep
  Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html).
  NeurIPS, 2020.
