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

## 7. P5b terminal analysis (2026-08-03)

P5b jobs `9378174`, `9378175`, and `9378176` all completed 2,000 updates with
exit code zero and a final `PASSED` receipt. The immutable result bundle is
`/home/lorenzo/moleworks/.artifacts/terra_p5b_results_20260802_6c56610e`;
the standardized report is
`/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`
(input digest
`89d44919deef6240a9fcc71fcb1525766d1176584b514c877bb748dd1b8ddb42`).

### Selected checkpoints

| Arm | Selected update | Macro P / D | Foundation P / D | Trench P / D | Exact P / D |
|---|---:|---:|---:|---:|---:|
| Medium adaptive | 2,000 | 0.652 / 0.625 | 0.639 / 0.588 | 0.670 / 0.674 | 1/512 / 2/512 |
| Deep adaptive | 1,000 | 0.653 / 0.628 | 0.625 / 0.571 | 0.689 / 0.702 | 2/512 / 2/512 |
| Medium uniform | 1,000 | 0.647 / **0.664** | **0.625 / 0.630** | 0.675 / **0.708** | 2/512 / **6/512** |

The selected table is descriptive because its updates differ. In the matched
update-1,000 comparison, deep/adaptive beats medium/adaptive by
`+0.023/+0.013` promotion/development macro, while medium/uniform beats it by
`+0.017/+0.049`. Both treatments satisfy the predeclared same-update gates at
that checkpoint. Neither retains its advantage through update 2,000:
deep/adaptive is `-0.035/-0.018` and medium/uniform `-0.088/-0.089` relative to
medium/adaptive. This is a transient recipe result, not a capacity or sampler
promotion.

The selected factor-axis macros show where the policies fail:

| Axis | Medium adaptive P / D | Deep adaptive P / D | Medium uniform P / D |
|---|---:|---:|---:|
| Anchor | 0.718 / 0.706 | 0.731 / 0.713 | 0.740 / 0.765 |
| Capacity | 0.766 / 0.673 | 0.738 / 0.644 | 0.722 / 0.711 |
| Composed | 0.428 / 0.467 | 0.510 / 0.499 | 0.485 / 0.471 |
| Distance | 0.361 / 0.435 | 0.264 / 0.310 | 0.330 / 0.364 |
| Dump layout | 0.697 / 0.516 | 0.635 / 0.567 | 0.626 / 0.634 |
| Geometry | 0.695 / 0.696 | 0.703 / 0.714 | 0.677 / 0.721 |
| Site | 0.653 / 0.603 | 0.661 / 0.599 | 0.647 / 0.656 |

Remote distance is the weakest shared axis and composed conditions are the
next failure band. Foundations remain weaker than trenches. Uniform's selected
checkpoint has the strongest development macro, family floor, anchor, dump,
geometry, and site slices, but it is weaker on the constrained distance tail
than medium/adaptive. Exact completion remains nearly absent under every
recipe.

### Teacher and entropy diagnosis

The P5b deep run used the intended architecture-growth playbook. Its
`2,699,117`-parameter policy is an exact function-preserving growth of the
`2,441,223`-parameter P5 parent, resets optimizer state, and uses that frozen
parent for KL (`1.0 -> 0` over 1,500 updates) and value (`0.5 -> 0` over 500
updates) distillation. The mechanism is therefore not missing.

The historical E8 comparison had been described incorrectly: E8 and E3 use
the same `2,441,223`-parameter architecture. E8 was a multitask parameters-only
warm start from E3 with E3 as teacher; it was not a bigger student. The earlier
size jump was approximately `994,825 -> 2,441,223` parameters when growing the
smaller policy into E3.

A current-protocol replay also shows why E8's historical near-1 online `swhr`
is not a valid capability baseline. Its serialized 33-field `EnvConfig` is
incompatible with current Terra, so the 126 parameter leaves were extracted
with the frozen historical source, verified shape-for-shape, and inserted into
a current evaluation-only skeleton. On the all-free controls those parameters
score only `0.013/0.027` promotion/development macro and `0/32` exact. This is
a parameters-only compatibility replay, not a reproduction of E8's historical
maps, resets, observation semantics, termination, or aggregation.

All P5b arms drop together at update 1,500 as KL reaches zero, while P5b's
entropy schedule is still about `0.137` because it decays from `0.15` to
`0.005` over 7,600 updates. The synchronized timing makes excessive exploration
after teacher handoff a credible hypothesis. It does not prove causality. P5c
therefore changes the common entropy schedule to the historical
`0.02 -> 0.005` over 10,000 updates and holds maps, rewards, teacher, PPO, and
reset protocol fixed. P5b/P5c claims are valid only at matched checkpoints
through update 2,000; later P5c checkpoints are learning-curve evidence only.

### No-dump-constraint capability floor

The accepted bank did not contain a true no-dump-constraint baseline. The new
diagnostic panel uses `fnd-slab-allfree` and `trn-straight-allfree`, paired to
existing source identities and excavation geometries while expanding only the
visible dump mask to every legal non-dig cell. It is intentionally excluded
from the constrained macro.

| Generalist | Macro P / D | Foundation P / D | Trench P / D | Exact P / D |
|---|---:|---:|---:|---:|
| P5 parent | 0.385 / 0.465 | 0.423 / 0.493 | 0.347 / 0.436 | 0/32 / 0/32 |
| Medium/adaptive @2,000 | 0.629 / 0.613 | 0.655 / 0.635 | 0.604 / 0.591 | 0/32 / 0/32 |
| Deep/adaptive @1,000 | **0.718 / 0.736** | **0.770 / 0.757** | **0.666 / 0.715** | 0/32 / 0/32 |
| Medium/uniform @1,000 | 0.484 / 0.540 | 0.452 / 0.459 | 0.517 / 0.622 | 0/32 / 0/32 |

The P5 foundation specialist reaches `0.668/0.640` on its all-free foundation
control with `0/16` exact on both panels. The P5 trench specialist reaches
`0.556/0.606` on its all-free trench control, with `1/16` exact on promotion
and `0/16` on development. The missing exact capability floor is therefore
not explained by generalist interference alone.

These controls are mechanically easier but visually out of distribution: the
target mask is almost entirely accepted dump area, unlike the 32 constrained
training conditions. Their zero exact rate cannot be read as a pure mechanics
failure. P5c will report them at every checkpoint as a separate capability
floor. Adding them to training would be a separately named 34-condition-bank
treatment, never a silent mutation of P5c.

### Revised decision

- Run the five-arm, 4,000-update P5c matrix frozen in
  [`P5_FOLLOWUP_GOAL.md`](P5_FOLLOWUP_GOAL.md): medium adaptive, medium
  uniform, deep uniform, foundation medium-uniform, and trench medium-uniform.
- Use the two specialists only as family dose ceilings. The causal generalist
  comparisons are medium adaptive versus medium uniform, then medium uniform
  versus deep uniform.
- Evaluate every 500 updates on constrained promotion/development and both
  all-free diagnostic panels.
- Admit a long run only after positive multi-checkpoint evidence with no family,
  bottom-tail, or capability-floor regression. One favorable checkpoint does
  not justify a 120-hour queue.
