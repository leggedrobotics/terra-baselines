# P5 Accepted-Bank Results Analysis

- Date: 2026-08-02
- Campaign: `f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa`
- Terra: `a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4`
- terra-baselines: `4b34668d105cf44118186d5ce49d1b78cd19a8e5`
- Result bundle:
  `/home/lorenzo/moleworks/.artifacts/terra_p5_results_20260802_f8aac348`
- Standard leaderboard:
  `/home/lorenzo/moleworks/.artifacts/terra_p5_leaderboard_20260802/LEADERBOARD.md`
- Follow-up goal: [`P5_FOLLOWUP_GOAL.md`](P5_FOLLOWUP_GOAL.md)

## 1. Evidence integrity

All six jobs completed with exit code zero, a `PASSED` final receipt, 2,000 PPO
updates, and 262,144,000 transitions. Both fixed panels contain 512 exact
manifest episodes. CPU/GPU reset parity, panel enumeration, scenario identity,
mass conservation, target/obstacle immutability, finite-state, slot-index, and
termination checks pass for every evaluation.

The leaderboard recomputes metrics from all per-map results, checks every
episode identity against the frozen manifest, requires deterministic policy
mode, and requires promotion/development to name identical checkpoint hashes.
Sealed evaluation was not opened.

## 2. What learned

The full-family specialists are the credible family feasibility parents:

| Family | Anchor control P / D | Full specialist P / D | Final exact P / D |
|---|---:|---:|---:|
| Foundation | 0.361 / 0.371 | **0.566 / 0.559** | 0/288 / 0/288 |
| Trench | 0.277 / 0.295 | **0.450 / 0.445** | 1/224 / 2/224 |

On only the actual anchor cells, the full specialists also win despite much
less anchor exposure: foundation `0.668/0.693` versus `0.517/0.546`, and trench
`0.467/0.490` versus `0.299/0.312` on promotion/development. Broader
within-family diversity therefore appears helpful rather than dilutive. This
is descriptive because support and exposure both changed and there is one
seed.

The generalists are stronger than the specialists in aggregate. At update
2,000, uniform/adaptive score `0.574/0.588` on promotion and `0.577/0.574` on
development. Their family macros are approximately `0.53--0.55` for
foundations and `0.62--0.64` for trenches. A separate large specialist screen
is therefore not the highest-value capacity test.

Exact completion remains almost absent. Dense completion shows useful digging
and soil relocation, but does not establish reliable full-task cleanup or
arbitrary constrained planning.

The standardized near-complete view separates cleanup from dead-start failure.
`G-ADAPTIVE` reaches at least `0.95` completion on `20/512` promotion and
`18/512` development maps, while solving only `0/512` and `1/512` exactly.
Its no-effect-action rate is `0.125/0.131`; the uniform policy is lower at
`0.078/0.084` but has only `4/512` and `5/512` near-complete maps. The next
screen therefore reports both tails rather than treating exact zero as the
only outcome.

## 3. Curriculum result

The preregistered promotion selector chooses `G-ADAPTIVE`:

- adaptive passes the update-1,000 to update-2,000 gate on both panels;
- uniform passes development but fails promotion because the worst-condition
  delta is `-0.063`; and
- the formal decision receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_p5_results_20260802_f8aac348/generalist_selection.json`.

This is not evidence that adaptive sampling is generally superior. At the
final checkpoint adaptive is only `+0.014` on promotion and `-0.003` on
development macro. The policies trade condition strengths.

The adaptive rule is a competence frontier, not hard-example mining. Across
585,841 realized resets it allocated only `1.45%` to `d16`, `1.75%` to `d12`,
and `1.97%` to procedural-side1-road, versus about `3.125%` under uniform.
It concentrated about `4.1--4.25%` on short network/tee trenches. This aligns
with adaptive losing to uniform on development d12/d16 while improving several
site and mini-topology conditions. The scheduler treatment should remain
explicit and the remote-foundation cells must not disappear into a pooled
score.

## 4. Failure structure

The condition tables reject a single scalar difficulty ladder.

- **Foundation tail:** remote `d16`, then `d12`, procedural side1+road, and
  side1+objects. Remote cases also show lower dump purity, illegal volume, and
  more no-effect actions.
- **Foundation body:** nearby/capacity aprons, clean rings, and light site
  constraints learn well in graded completion, but still time out rather than
  finishing exactly.
- **Trench tail:** alternating banks, straight one-side/tight, and one-side+road
  networks. Some visually complex both-side networks outperform the nominally
  simple one-side anchor.
- **Only repeated exact cell:** short tee `trn-tee-side2-s` for the trench
  specialists; adaptive's exact development successes occur on different maps
  at updates 500 and 2,000 and are not retained identity-level mastery.

Final zero-completion mass remains large: roughly 17--18% for the foundation
specialist, 13--14% for the trench specialist, and 14--16% for the generalists.
Micro p10 is consequently zero everywhere.

## 5. Capacity and sampler beliefs

The medium model is still improving sharply from update 1,000 to 2,000, so the
data do not demonstrate a capacity plateau. A scratch-large run would conflate
optimization, capacity, and initialization. The first widened-large growth
candidate also fails admission directly: before PPO, its promotion macro is
`0.071` versus the parent's `0.588`. It is rejected rather than trained back to
the starting policy.

The replacement depth candidate retains stage channels `(24, 48, 64, 96)` and
the 512/256 critic, adds three identity-initialized residual blocks, and grows
from `2,441,223` to `2,699,117` parameters (`+10.6%`). Its update-zero
admission is exact when the untouched parent and depth-grown checkpoint are
evaluated back-to-back with the same frozen Terra, evaluator, and local GPU:
all 1,024 promotion/development trajectories match in terminal completion,
success, steps, termination, soil accounting, and no-effect count. Promotion
macro is `0.5877105316` and development is `0.5740648174` for both. The earlier
two-map difference against the archived Euler development evaluation is a
cross-runtime replay difference, not transplant damage. This treatment asks
only whether more residual depth helps.

Sampler exposure is the other supported question. Adaptive and uniform finish
nearly tied overall, while adaptive allocates less mass to `d12` and `d16` and
loses on several remote/tight cells. A uniform parameters-only warm start from
the same parent tests suspected tail starvation without new sampler code.

The selected P5 checkpoint predates pooled-sampler state in checkpoints. A
literal adaptive continuation cannot be reconstructed exactly from it. All
follow-up arms deliberately use parameters-only warm starts, a fresh optimizer
and sampler, and the same frozen-parent teacher. This makes both comparisons
honest and avoids calling a sampler restart a continuation. New checkpoints
serialize exact host sampler state for later resumes; the broader JAX
environment trajectory remains non-bit-exact on resume and is labelled as
such.

## 6. Decision

- Stop anchor-only training.
- Retain the family specialists as diagnostics, not the next capacity parents.
- Select `G-ADAPTIVE` update 2,000 as the common parent.
- Run the three-arm matched P5b star frozen in
  [`P5_FOLLOWUP_GOAL.md`](P5_FOLLOWUP_GOAL.md).
- Do not add a deep-uniform arm unless depth and uniform independently win.
- If neither wins, test action masking against the zero-completion/no-effect
  tail before changing rewards or mixing partial resets.
- Do not launch the 20,000-update P6 run on the 64-map training bank. A long run
  still requires the separately frozen 256-training-layouts-per-condition bank
  plus a passing P5b recipe gate.
