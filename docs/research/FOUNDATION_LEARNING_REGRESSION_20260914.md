# Foundation historical competence and learning regression — September 14, 2026

The old foundation policies still complete almost every map under today's
physical rules. Fresh frozen-weight replays finish **62/64 for the old zero-cost
control** and **63/64 for the old 2× policy**, compared with **11/64 for the new
u41000 global-normalization checkpoint**. Treat the older control as the primary
historical reference. Comparing 11/64 only with the recently regressed 3/64
understates the remaining deficit.

## Completed current-environment comparison

All five reports below use Terra `46738cde`, frozen baselines evaluator
`866e8e2`, the same 64 validation maps and reset identities, greedy inference,
evaluation seed 20260907, and a 450-action horizon. Both fresh replays finished
with their `EVAL_DONE` markers. All 320 report rows pass identity and recorded
integrity checks; actual checkpoint-file SHA-256 values match their reports.

| Policy | Exact completion | Mean excavated | Mean accepted disposal |
| --- | ---: | ---: | ---: |
| Old zero-cost control u15000, fresh replay | 62/64 (96.9%) | 99.56% | 99.55% |
| Old 2× u15000, fresh replay | 63/64 (98.4%) | 99.94% | 99.94% |
| Current scratch parent u33000 | 8/64 (12.5%) | 93.29% | 90.97% |
| Current local-normalization u41000 | 3/64 (4.7%) | 79.41% | 77.14% |
| Current global-normalization u41000 | 11/64 (17.2%) | 91.09% | 90.61% |

Both old policies complete all 11 maps solved by the new global-normalization
checkpoint, plus 51 and 52 additional maps respectively. Old control completes
20/21 L shapes and old 2× completes 21/21; the new global checkpoint solves 0/21.
The old 2× treatment has additional behavior costs and is not the zero-cost
causal control. Strong old-control performance does not depend on those costs.

On the original environment, the same old u15000 checkpoints solved 63/64
(control) and 64/64 (2×). Between the historical reports and today’s replay, each old policy loses one
net completion. That comparison spans environment and evaluator/runtime history,
so the loss is not attributed exclusively to physics. It also does not prove
unchanged difficulty of learning those skills from random initialization.

[Completed report comparison](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/current_environment_comparison.json) ·
[Fresh old-control evaluation](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/old_control_current_env.json) ·
[Fresh old-2× evaluation](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/old_double_current_env.json).

## Why trench rules do not explain this foundation deficit

All 256 training and 64 validation metadata files identify foundations, with no
trench axes. Actual saved resets have no trench-cell membership and disable
foundation-border alignment. The trench gate returns unrestricted pose
validity for these non-trench cells. No accidental trench-yaw constraint on this
foundation bank was found.

Several physical repairs were shared across tasks: all positive soil now blocks
chassis travel and base rotation; deposition and relaxation exclude active
chassis cells; and base translation checks its swept footprint. The eligible
loose-soil selection fix also affects foundations. Thus the changes were broader
than only dumping beneath the chassis, even though trench alignment is scoped.

The first movement repair had a real rounded-intermediate-path collision bug.
It dropped the frozen old 2× policy from 64/64 to 51/64. Correcting that bug
restored 63/64 while preserving the soil protections. Current Terra `46738cde`
has exactly the same `terra/` runtime tree as corrected `7fb30402`, and today's
fresh replay independently reproduces 63/64.

All 70 saved environment fields match between old control u15000 and current
u33000/u41000 except live consecutive-episode counters. Geometry, spawn flags,
reward configuration, observation flags and alignment gates agree. Old control
already enabled the executable-dig observation. All 1,280 local training arrays
(256 maps × five reset layers) were freshly hashed against their scenario
manifest and match. The bank registry and distance-sidecar identities agree
between old and current native checkpoints. This is not a fresh remote disk
inspection.

[Shared-rule audit](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/foundation_rule_matrix.md) ·
[Foundation metadata receipt](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/foundation_metadata_receipt.json) ·
[Runtime-source equality](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/runtime_source_equivalence.json) ·
[Native environment/configuration comparison](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/native_environment_comparison.json) ·
[Training-bank support receipt](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/training_bank_support_receipt.json).

## What changed in learning

The older easy-bank policies continued a generalist at u5000, inheriting model
weights, Adam state and the absolute schedule clock. That parent had 327.68
million transitions across mixed tasks. The new foundation lineage started
with random weights and fresh Adam on the easy bank.

| Checkpoint | Mixed-task pretraining | Additional easy-bank transitions | Total transitions | Original exact result |
| --- | ---: | ---: | ---: | ---: |
| Old control u7000 | 327.68M | 32.768M | 360.448M | 60/64 |
| Old control u15000 | 327.68M | 163.84M | 491.52M | 63/64 |
| Old control u25000 | 327.68M | 327.68M | 655.36M | 64/64 |
| New global-normalization u41000 | 0 | 671.744M | 671.744M | 11/64 |

The current checkpoint has more total transitions than these old near-perfect
checkpoints. Insufficient total sample count is not a sufficient explanation.
The original generalist parent itself scored 0/64 before adaptation; transfer
of useful representations or subskills is plausible, not directly measured.
Transferred Adam state, pretraining distribution, seed, exploration timing and
physical-rule history are also confounded.

The large deficit also preceded four-GPU migration: the scratch u33000 policy
was already 8/64 at 1×512 environments, the same local minibatch-normalization
layout used by the old foundation fine-tune. GPU-local normalization can be
investigated as a contributor to the later 8→3 drop, but cannot explain the
preceding 63→8 gap by itself. The global-normalization 11/64 screen has not
recovered historical foundation competence.

The nominal entropy schedule is cosine 0.15→0.02 over 20,000 absolute updates.
The older run reached 60/64 at u7000 while entropy was about 0.1145; the scratch
lineage reached the 0.02 floor before learning robust completion. This is a
candidate learning-strategy problem, not proof that entropy alone caused the
regression. The model, base reward and nominal PPO recipe otherwise match;
normalization intentionally changes in the latest screen.

[Historical lineage and configuration evidence](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/HISTORICAL_LINEAGE.md) ·
[Historical paired comparison receipt](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/historical_comparison.json).

## Decision and limits

Keep the corrected soil-free chassis physics. The old zero-cost foundation
policy is the strongest recovery-parent candidate, with its native optimizer
and observation contract preserved. Use it as the reference for future
foundation comparisons and investigate scratch learning separately. Additional
behavior penalties are not a supported remedy for poor completion.

The evidence establishes a learned-policy/training-history deficit and shows
that current rules permit near-perfect foundation completion. It does not
isolate one training cause, prove that shared rules leave scratch exploration
unchanged, or exclude every subtle implementation bug. A controlled training
comparison is needed to separate initialization, optimizer state and schedule
history. This audit changed no training jobs, source code or rewards.

[Independent seven-report review](../../../../../.artifacts/terra_foundation_lineage_audit_20260914/independent_review.md) validates
all 448 historical and current rows and the five actual retained checkpoints.
