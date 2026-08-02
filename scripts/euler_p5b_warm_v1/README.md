# P5b warm-start architecture screen

This is one matched three-arm star:

- `G-MEDIUM-ADAPTIVE-WARM`: common medium/adaptive control;
- `G-DEEP-ADAPTIVE-WARM`: depth-only treatment with residual block counts
  `(2, 2, 3, 3)`;
- `G-MEDIUM-UNIFORM-WARM`: sampler-only uniform treatment.

All arms use the existing frozen P5 bank, parent parameters and teacher, seed,
PPO/reward/horizon settings, and fresh optimizer. The two comparisons change
one factor each against the common control. There is deliberately no
deep-uniform arm.

After the terra-baselines branch is committed and clean, inspect the resolved
paths and Slurm commands without touching Euler:

```bash
SUBMIT=0 scripts/euler_p5b_warm_v1/submit.sh smoke 20260730
```

The environment authority is the clean paired Terra worktree
`/home/lorenzo/moleworks/.worktrees/terra_simple_mapbank_reward_20260730` at
`a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4`. The live
`/home/lorenzo/moleworks/terra` checkout is not used because it has moved to a
different environment protocol.

Submit all three one-update smokes:

```bash
SUBMIT=1 scripts/euler_p5b_warm_v1/submit.sh smoke 20260730
```

After all smoke jobs produce a passing `smoke_validation.json`, submit the
2,000-update screens:

```bash
SUBMIT=1 scripts/euler_p5b_warm_v1/submit.sh screen 20260730
```

The screen evaluates the exact promotion and development panels at added
updates 500, 1000, 1500, and 2000. Outputs live under
`/cluster/scratch/lterenzi/codex_terra_edge_runs/p5b_warm_v1/`. The launcher
reuses the P5 bank archive, selected parent checkpoint, Euler venv, and runtime
check already used by the completed accepted-bank campaign.
