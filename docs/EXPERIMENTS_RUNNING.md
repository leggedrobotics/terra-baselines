# Experiments — current state (updated 2026-08-13 CEST)

## v6.1 reward-v2 + stall age + Continuous Banded v3

One direct continuation is prepared but not yet submitted. It starts from the
selected v6.1 reward-v2 checkpoint at update 14,000 and targets absolute update
40,000. The declared practical bundle contains exactly:

1. one material-stall-age observation; and
2. the final family-free Continuous Banded v3 curriculum.

This is a capability run, not a causal one-variable ablation. Reward-v2 and its
timing, v6.1 spatial architecture, no-action-mask setting, PPO shape, learning
rate, entropy schedule, horizon, bank and seed remain fixed.

### Observation

```text
stall_age = min(consecutive transitions without material-state change, 32) / 32
```

The material signature is the raw soil/action map plus every active
excavator's load and carry-relocation credit. Soil, load or carry changes reset
the counter; pose and cabin motion do not. Separate zero-initialized 704-wide
actor and critic embeddings inject the scalar without changing update-14,000
outputs at age zero.

### Curriculum

Continuous Banded v3 has no foundation/trench assignment quota. It places 80%
of assignment mass globally on open conditions with depth weights `4:2:1`,
20% uniformly on mastered replay, and caps each condition at 15% by
water-filling. At the source checkpoint, 29 conditions are mastered and 18 are
open; v3 assigns exactly 0.80/0.20 open/replay mass, 84.83% foundation and
15.17% trench mass, with maximum cell mass 6.96%.

The source checkpoint stopped 50 updates into a sampler window. The offline
materializer preserves model/Adam state, optimizer and entropy clocks,
mastery, competence, closed-window history, refresh schedule and sampler RNG,
then clears only that unfinished window and writes native v3 state. Runtime has
no sampler-migration mode.

### Compute and admission

- requested shape: 8 RTX 4090, `gpuhe.24h`, 23:45;
- PPO shape: 8 x 256 environments x 32 steps, 32 minibatches, two epochs;
- global environments and transitions/update remain 2,048 and 65,536;
- the allocation runs CUDA convolution-backward and NCCL preflight before
  training;
- the first completed update checks finite loss, parameters and optimizer;
- checkpoints are written every 500 absolute updates, so a wall-time stop is
  continuable.

Pinned inputs:

- Terra: `c2d2a94a124759e9f21c2b37930f717e299f0c46`;
- source checkpoint SHA-256:
  `79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609`;
- prepared checkpoint `v8_v61_stall_age_v3_u14000_prepared.pkl`, 27,741,529
  bytes, SHA-256
  `68aea1a0f5dc3c05d11319fdf640ade05495125225533bc99ad92592475fcb75`;
- terra-baselines launch revision: pending final commit;
- Slurm job: pending submission.

The incorrectly prepared job `10616190` retained the old family-balanced
sampler. It was cancelled while pending and consumed zero runtime; it produced
no checkpoint, W&B run or training evidence. Earlier resume-smoke job
`10572344` was likewise cancelled without allocation.

### Readout

1. Verify the first completed update and first finite rolling checkpoint.
2. Evaluate fixed promotion checkpoints against the existing v6.1 curve.
3. Inspect stall-age mean/saturation and the frozen recurrence failure strata.
4. Continue while fixed-panel performance improves; stop on a measured
   plateau, not wall time.

If recurrence remains, the next separate experiment is an actor-only GRU-64
head over the unchanged v6.1 encoder. It requires contiguous 32-step PPO
sequences and a sequence-batched feed-forward control; it is not bundled here.

## Latest completed reference

The selected v6.1 update-14,000 policy scored 407/720 exact on the fixed
promotion panel versus 281/720 for the code-matched compact policy. It improved
foundations by 66 maps and trenches by 60, while all 313 remaining failures
timed out. The failure and recurrence analysis is in
[`research/V8_V61_FAILURE_AUDIT_20260813.md`](research/V8_V61_FAILURE_AUDIT_20260813.md),
and the architecture result is in
[`research/V8_V61_ABLATION_RESULT_20260813.md`](research/V8_V61_ABLATION_RESULT_20260813.md).

Completed historical runs remain in [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md)
and git history. Superseded launchers and sampler modes are intentionally not
kept as executable compatibility surfaces.
