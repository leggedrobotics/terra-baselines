# P5c low-entropy follow-up

This five-arm matrix separates the two admitted P5b effects under the
historical low-entropy kickstart regime and retains explicit family ceilings:

- medium/adaptive is the common low-entropy control;
- medium/uniform isolates the sampler; and
- deep/uniform isolates added depth after uniform passed the P5b gate;
- foundation/medium/uniform is the concentrated foundation ceiling; and
- trench/medium/uniform is the concentrated trench ceiling.

All five reuse the same frozen P5 parent and teacher, reward, horizon, PPO
settings, seed, and full-reset distribution. The three generalists use the
same 32-condition bank; specialists select the existing 18-foundation or
14-trench subset. All use entropy `0.02 -> 0.005 / 10000`. The screen runs
4,000 updates and evaluates every 500-update checkpoint on promotion and
development.

Those five screens completed in 6.2--8.0 hours and their online training
success was still rising. They are therefore early learning-curve evidence,
not saturation results. Successor runs must use an intentionally oversized
absolute update target, checkpoint every 100--500 updates, receive at least a
healthy 24-hour behavioral screen, and continue promising checkpoints with
`--resume_from` on the 120-hour queue. A wall-time exit with a finite, loadable
checkpoint is `CONTINUABLE`; fixed evaluation may run as a separate job after
the timeout. The canonical contract is
[`P5_ACCEPTED_BANK_EXPERIMENTS.md`](../../docs/research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
section 12.

P5b-to-P5c entropy comparisons are causal only at matched checkpoints through
update 2,000. P5c checkpoints after update 2,000 have no high-entropy control
and are interpreted only as within-P5c learning curves. Specialist/generalist
differences include their deliberately different per-condition training dose;
the specialists are feasibility ceilings, not negative-transfer estimates.

Dry-run, smoke, and screen:

```bash
SUBMIT=0 scripts/euler_p5c_low_entropy_v1/submit.sh smoke 20260730
SUBMIT=1 scripts/euler_p5c_low_entropy_v1/submit.sh smoke 20260730
SUBMIT=1 scripts/euler_p5c_low_entropy_v1/submit.sh screen 20260730
```

Every numbered checkpoint is also evaluated on paired all-free foundation and
trench capability controls. Those results are written as diagnostic panels,
never enter the constrained 32-condition macro, and are not added to P5c
training support. A later 34-condition successor bank is a separate map-support
treatment, not a silent mutation of this recipe experiment.
