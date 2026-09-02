# V8 GRU media/dashboard recovery

This evaluation-free recovery consumes the completed, hash-pinned fixed-panel
JSONs from failed job `11303967`, replays only the two u44 policies needed for
qualitative media, and builds the three benchmark dashboards. It never invokes
`eval_fixed_bank.py` and does not rewrite the failed parent directory.

The parent fixed evaluations completed before media setup failed:

- GRU u40/u44 JSON SHA-256:
  `83440b8f1b01f5d4d3b217da4e8c08a5bc7c60ab1b76483680f78cf6c5e576e2`;
- FF u44/u86 JSON SHA-256:
  `c0c53f54ee2d282c8cd5e4151e52ac3910c449cc909c5ee70c16a20965e800e5`;
- 40-slot review selection SHA-256:
  `675436d00ed6a156bfa1a00a325141c6fde98f52f09da85b02e21c7df9f93070`.

The renderer now mirrors canonical accepted-panel reset verification. Terra's
`reset_prepared` consumes the manifest input keys and advances the key stored in
the resulting state, so replay verifies the frozen *input-key receipt* from the
fixed JSON instead of incorrectly requiring the post-reset state key to equal
its input.

Dry-run without SSH:

```bash
scripts/euler_v8_gru_media_recovery_v1/submit.sh
```

Stage and validate immutable inputs without creating a Slurm job or result
directory:

```bash
SUBMIT=stage scripts/euler_v8_gru_media_recovery_v1/submit.sh
```

After review, submit one RTX 4090 recovery job:

```bash
SUBMIT=1 scripts/euler_v8_gru_media_recovery_v1/submit.sh
```

The recovery is qualitative replay plus presentation only. It does not change
or strengthen the already completed fixed-panel quantitative evidence.
