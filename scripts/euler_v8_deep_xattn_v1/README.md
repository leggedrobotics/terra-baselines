# V8 deep+xattn curriculum campaign

This launcher compares one architecture change at a time on the accepted V8
distribution:

| Arm | Initialization | Trainable policy |
|---|---|---|
| `G-DEEP-V8-DENSE-WARM` | trained deep u4000 | deep SE |
| `G-DEEP-XATTN-V8-DENSE-WARM` | output-preserving graft from the same u4000 | deep SE + E4-prime cross-attention |

At Stage A, both arms use the same trained deep checkpoint as teacher, a fresh
optimizer, identical KL/value kickstart, PPO settings, seed, fixed V8 stage
weights, dense reward, 450-step horizon, full resets, exact visible dump mask,
and disabled absolute trench shaping. The xattn arm does not import E4/E7
attention weights; the appended attention contribution is exactly zero at
update zero.

The map curriculum is checkpoint-bounded:

- `capability`: 2 all-free controls, 50/50 foundation/trench;
- `nearby`: 50% capability replay + 50% V7 adjacent geometry core;
- `full`: 25% capability + 25% nearby core + 50% V6 constraints.

Every stage is family-balanced. The first screen is `capability` for at most
2,000 updates; it evaluates all numbered checkpoints on the frozen capability
promotion and development panels. A later stage is launched only after its
recorded gate passes. Reward progression is not part of the initial architecture
comparison. It may begin only from a qualified full-stage checkpoint under the
separate checkpoint-bounded reward protocol.

This launcher revision accepts `capability`, `nearby`, and `full`. Nearby/full
require one immutable passing prior-stage receipt per architecture and derive
each parent path and SHA-256 only from that receipt. Map-stage transitions are
parameter-only, use a fresh optimizer and fixed sampler, and disable teacher
kickstart. They must never restart from the original P5c parent. Only an
unchanged full-stage continuation uses true optimizer/schedule/sampler resume.

Frozen inputs:

- V8 release: `terra_v8_v6_constraints_v7_adjacent_train96_v5`;
- archive: `.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst`;
- archive SHA-256: `dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b`;
- `bank/dataset.json` SHA-256: `715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798`;
- parent/teacher SHA-256: `4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc`;
- Terra revision: `a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4`.
- capability sampler SHA-256:
  `a569e04eba1bc2ed7cff9d084ff75c7a09224df6d600a4ab647a7b28c15f8633`;
- nearby sampler SHA-256:
  `a681e5e92562a322db2627825e607df2d7b8ece708f9bcd87d5d0d710b3ae398`;
- full sampler SHA-256:
  `2a457be780e086c02e0474489b2060d6c577fac0ac429c48ad1a7e1e5e011357`.

Run the immutable dry-run first, then the update-1 admission jobs, then the
bounded capability screen:

```bash
SUBMIT=0 scripts/euler_v8_deep_xattn_v1/submit.sh smoke capability 20260730
SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit.sh smoke capability 20260730
SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit.sh screen capability 20260730

# After both per-arm Stage-A gate receipts exist locally:
SUBMIT=0 scripts/euler_v8_deep_xattn_v1/submit.sh smoke nearby 20260730 \
  deep_stage_a_gate.json xattn_stage_a_gate.json
SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit.sh smoke nearby 20260730 \
  deep_stage_a_gate.json xattn_stage_a_gate.json
SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit.sh screen nearby 20260730 \
  deep_stage_a_gate.json xattn_stage_a_gate.json

# After both per-arm Stage-B gate receipts exist locally. The screen command
# also schedules one afterany tail evaluator per arm.
SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit.sh smoke full 20260730 \
  deep_stage_b_gate.json xattn_stage_b_gate.json
SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit.sh screen full 20260730 \
  deep_stage_b_gate.json xattn_stage_b_gate.json

# After a full receipt qualifies for long compute. Use matched_architecture_pair
# for each arm when both qualify; omit PAIRING when only one qualifies.
PAIRING=matched_architecture_pair SUBMIT=1 \
  scripts/euler_v8_deep_xattn_v1/submit_continuation.sh \
  deep_full_stage_gate.json
PAIRING=matched_architecture_pair SUBMIT=1 \
  scripts/euler_v8_deep_xattn_v1/submit_continuation.sh \
  xattn_full_stage_gate.json
```

`SUBMIT=0` performs no SSH, upload, scratch, W&B, or Slurm mutation. Submission
fails closed on dirty source trees, source/checkpoint/archive hashes, the exact
four-RTX-4090 runtime, CUDA/NCCL admission, V8 loader validation, update-1
finite/integrity validation, and the preceding smoke receipt.

Capability and nearby evaluate in the bounded training allocation. Full-stage
training schedules a separate `gpuhe.4h` evaluator with `afterany`: it accepts a
completed or wall-time-limited parent, freezes the longest contiguous 500-step
checkpoint prefix, evaluates main/capability promotion and development panels,
and writes `tail_eval/stage_gate.json`. Gaps, duplicate checkpoints, other
Slurm failures, non-finite integrity, or checkpoint/sampler mutation fail
closed.

Capability mastery is `12/16` per control; nearby mastery is foundation
`78/96`, trench `91/112`, and `12/16` per core cell, with inherited retention
thresholds carried without ratcheting. Any two adjacent treatment-level
retention failures anywhere in the stage history trigger rollback, even if the
last two checkpoints later recover.

Long compute is deliberately permissive but not automatic. Relative to the
checkpoint 1,000 updates earlier, the latest full checkpoint needs either one
new exact V6-constraint solution or `0.001` V6 condition-macro gain. Foundation,
trench, micro-p10, and worst-condition guards allow at most five percentage
points of regression on promotion and development. A qualified continuation
uses `--resume_from`, an absolute update-80,000 target, four RTX 4090s on
`gpuhe.120h` for 119:45, and checkpoints every 500 updates. It keeps the source
treatment name so fixed evaluations remain comparable while using a new linked
W&B run ID. The continuation launcher always schedules a second `afterany`
evaluator on `gpuhe.24h`; it evaluates the source, every 2,000 continuation
updates, and the latest complete checkpoint on all four frozen panels. It writes
aggregate and per-condition histories and may issue a dense-to-reward
qualification receipt, but it never launches the reward experiment.
