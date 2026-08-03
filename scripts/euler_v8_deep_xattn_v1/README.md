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

This launcher revision accepts `capability` and `nearby`. Nearby requires one
immutable passing Stage-A receipt per architecture and derives each parent path
and SHA-256 only from that receipt. It uses a parameter-only stage transition
with a fresh optimizer and fixed sampler and disables teacher kickstart. `full`
remains closed until the equivalent Stage-B receipt path is implemented. Later
stages must never restart from the original P5c parent.

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
  `a681e5e92562a322db2627825e607df2d7b8ece708f9bcd87d5d0d710b3ae398`.

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
```

`SUBMIT=0` performs no SSH, upload, scratch, W&B, or Slurm mutation. Submission
fails closed on dirty source trees, source/checkpoint/archive hashes, the exact
four-RTX-4090 runtime, CUDA/NCCL admission, V8 loader validation, update-1
finite/integrity validation, and the preceding smoke receipt.

Every screen evaluates the latest checkpoint pair on the promotion panel and
writes `stage_gate.json`. Nearby additionally evaluates the capability panel at
the identical checkpoint paths and hashes. Capability mastery is `12/16` per
control; nearby mastery is foundation `78/96`, trench `91/112`, and `12/16`
per core cell, with inherited retention thresholds carried without ratcheting.
The gate also loads both qualifying checkpoints and verifies the exact ordered
condition list and frozen probability vector from each sampler state. Any two
adjacent treatment-level retention failures anywhere in the stage history
trigger rollback, even if the last two checkpoints later recover.
