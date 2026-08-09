# V8 dense architecture control

This launcher compares the current compact deep+xattn policy with the original
small Atari policy. Both arms train from random initialization on all 47 V8
conditions under the same `continuous_banded_v1` sampler and constant dense
reward:

| Arm | Architecture | Parameters | Screen allocation |
|---|---|---:|---|
| `compact_xattn` | medium deep SE encoder plus cross-attention | 2,856,685 | 4x RTX 4090, `gpuhe.120h` |
| `atari_base` | original base Atari CNN and base heads | 480,137 | 4x RTX 4090, `gpuhe.24h` |

The Atari arm is an intentionally small system control, not an encoder-only
ablation: the base preset also uses smaller actor, critic, and local-map heads.
No teacher, warm start, reward annealing, or PPO batch change is included.

Everything outside the architecture block is matched: seed `20260807`, the
47x96 accepted V8 bank and hashes, full 450-step resets, dense reward,
`continuous_banded_v1`, 4 devices, 512 environments/device, 32 rollout steps,
32 minibatches, 2 PPO epochs, learning rate `3e-4`, entropy `0.15 -> 0.02`,
20,000 updates, checkpoints every 500 updates, and fixed promotion,
development, and capability evaluations every 1,000 updates after training.
Each arm therefore sees 1,310,720,000 transitions.

The 24-hour Atari allocation is conservative relative to the historical
foundation-only control `nnsksyva`: it completed the same number of transitions
in 6.50 hours and ended near 113k steps/s on 4x4090. That run used an older map
distribution and twice as many environments/device, so it is only a scheduling
estimate. A time-limit exit is not a scientific result; retain its checkpoints
and add a separately reviewed true-resume launcher to reach the absolute
20,000-update target before comparison. This v1 launcher intentionally exposes
only smoke and the first full screen.

## Launch

The runtime Terra revision must be a committed full SHA. The launcher also
requires the baselines and Terra worktrees to be committed and clean.

Dry-run, then submit both 4x3090 update-1 smokes:

```bash
SUBMIT=0 scripts/euler_v8_architecture_control_v1/submit.sh smoke <runtime-terra-sha>
SUBMIT=1 scripts/euler_v8_architecture_control_v1/submit.sh smoke <runtime-terra-sha>
```

Only after both smoke jobs are `COMPLETED` and their generic, sampler, and
architecture receipts pass, submit both screens:

```bash
SUBMIT=0 scripts/euler_v8_architecture_control_v1/submit.sh screen <runtime-terra-sha>
SUBMIT=1 scripts/euler_v8_architecture_control_v1/submit.sh screen <runtime-terra-sha>
```

Screen admission checks both smoke receipts before submitting either arm. The
launcher does not cancel, hold, or supersede any existing job.

## Interpretation

Use fixed-panel exact success, macro completion, per-family, per-depth, and
per-condition results. Online success and reward are diagnostics. The
historical Atari metrics use a different evaluation contract and must not be
compared numerically with V8 fixed-panel results.
