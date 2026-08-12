# V8 continuation tail evaluation

This is the tail-safe evaluator for one qualified 120-hour dense continuation.
It never launches or changes a reward curriculum.

The evaluator runs with `afterany`, accepts only a Slurm `COMPLETED` or
`TIMEOUT` continuation, and rejects cancellation, node failure, OOM, and every
other state. It freezes and hashes the complete 500-update checkpoint sequence
strictly after the qualified resume checkpoint. It evaluates:

- the qualified source checkpoint;
- every 2,000 continuation updates relative to that source;
- the latest complete checkpoint as an extra diagnostic when it is off cadence.

All selected checkpoints are evaluated together on main promotion,
main development, capability promotion, and capability development. This makes
`eval_fixed_bank.py` enforce one path-independent treatment fingerprint across
the source and continuation. The reward gate uses only the fixed promotion
schedule; the off-cadence latest diagnostic does not count as a consecutive
scheduled evaluation. Before a receipt can be issued, every selected checkpoint
is also reloaded and checked for finite model and optimizer state, increasing
optimizer step, the exact full-bank architecture and treatment fingerprint,
the frozen sampler state, and the qualified resume source.

Standalone dry run and submission:

```bash
SUBMIT=0 scripts/euler_v8_deep_xattn_v1/submit_continuation_tail.sh \
  qualified_full_stage_gate.json CONTINUATION_JOB_ID \
  /cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_deep_xattn_v1/REVISION/continuation/full/s20260730/ARM-matched

SUBMIT=1 scripts/euler_v8_deep_xattn_v1/submit_continuation_tail.sh \
  qualified_full_stage_gate.json CONTINUATION_JOB_ID \
  /cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_deep_xattn_v1/REVISION/continuation/full/s20260730/ARM-matched
```

For automatic integration, submit `continuation_tail.sbatch` with
`afterany:CONTINUATION_JOB_ID` and pass the same exports already used by the
continuation launcher plus `CONTINUATION_JOB_ID` and
`CONTINUATION_RUN_DIR`. Keep the evaluator on `gpuhe.24h`; a long continuation
can yield dozens of selected checkpoints.

Outputs live in `CONTINUATION_RUN_DIR/continuation_tail/`:

- `checkpoint_inventory.json`: every continuation checkpoint and hash plus the
  selected evaluation schedule;
- `eval/*.json`: the four fixed panel histories;
- `benchmark/LEADERBOARD.md`: short human summary;
- `benchmark/history.csv`: aggregate history for all four panels;
- `benchmark/per_condition.csv`: every condition at every checkpoint;
- `benchmark/leaderboard.json`: structured history;
- `benchmark/dense_reward_gate_receipt.json`: immutable gate receipt.

The dense-to-reward receipt is qualified only when the latest three scheduled
promotion evaluations each meet `576/720` overall, `308/384` foundation,
`269/336` trench, `10/16` in every main condition, the frozen capability/core
retention thresholds, and integrity. The receipt names the final dense parent,
but records `reward_launched=false`; reward-v2 remains a separate experiment.
