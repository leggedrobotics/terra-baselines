# Terra v6-main + Capability-Floor, Train-96 v1

This launcher is the separate 34-condition map-support treatment that follows
P5c. It does not mutate or rename any P5, P5b, P5c, constrained evaluation, or
diagnostic-control dataset.

Training support is exactly 34 conditions with 96 layouts per condition:

- the preserved 32-condition Terra v6-main constrained distribution;
- `fnd-slab-allfree`; and
- `trn-straight-allfree`.

The two all-free conditions are training support and capability-floor
diagnostics. They remain excluded from the constrained 32-condition macro.
Promotion and development results are computed only from the frozen constrained
panels. The separate, frozen two-condition diagnostic bank is evaluated at the
same checkpoints and is never passed to the constrained W&B logger.

## Five frozen roles

| Role | Training support | Sampler | Initial parameters | Teacher |
|---|---|---|---|---|
| `G-MEDIUM-ADAPTIVE-WARM` | 34 | adaptive | P5 generalist u2000 | P5 generalist u2000 |
| `G-MEDIUM-UNIFORM-WARM` | 34 | uniform | P5 generalist u2000 | P5 generalist u2000 |
| `G-DEEP-UNIFORM-WARM` | 34 | uniform | depth-grown P5 generalist u2000 | P5 generalist u2000 |
| `F-MEDIUM-UNIFORM-WARM` | 19 foundation | uniform | P5 generalist u2000 | P5 generalist u2000 |
| `T-MEDIUM-UNIFORM-WARM` | 15 trench | uniform | P5 generalist u2000 | P5 generalist u2000 |

Every role preserves the P5c reward, full reset, 450-step horizon, action and
observation contract, PPO batch and optimizer settings, KL/value kickstart
schedules, and entropy schedule (`0.02 -> 0.005 / 10000`). Medium roles use the
same medium spatial-ResNet; the one deep role changes only the frozen depth-grown
encoder stage layout. Smoke is one update. Screen is 4,000 updates with numbered
checkpoints every 500 updates.

## Fail-closed use

The immutable bank archive contains exactly one top-level `bank/` tree. Its
frozen local defaults are:

- archive: `/home/lorenzo/moleworks/.artifacts/terra_v6main_capfloor34_train96_v1_20260803_a14d8302.tar.zst`;
- archive SHA-256: `c19b27c0771eddb09b8c1f1f09655ec3bf9a84858b3f23b19cd6eda619db21cb`;
- `bank/dataset.json` SHA-256: `2a1d74eec0ff8115b0922c9f82f14ddb1589aecec2d63f26d8461339b2f66f45`; and
- payload manifest SHA-256: `9e3a811ea480fdb72013ea9086c2c39990e2ceae5d902805f75164a310bc01db`.

Dry-run validates those receipts and remains the first command:

```bash
SUBMIT=0 scripts/euler_train96_capfloor_v1/submit.sh smoke 20260730
```

The exact frozen archive can then be smoke-tested and screened:

```bash
SUBMIT=0 scripts/euler_train96_capfloor_v1/submit.sh smoke 20260730
SUBMIT=1 scripts/euler_train96_capfloor_v1/submit.sh smoke 20260730
SUBMIT=1 scripts/euler_train96_capfloor_v1/submit.sh screen 20260730
```

`SUBMIT=0` performs no SSH, scratch, W&B, or Slurm mutation. `SUBMIT=1` remains
blocked if the local archive or either hash is absent or inexact. Environment
overrides exist only to relocate the same immutable archive, not to change the
release identity.
