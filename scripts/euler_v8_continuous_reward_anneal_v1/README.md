# V8 continuous curriculum: constant dense versus dense-to-terminal

This launcher runs the minimal matched reward-curriculum experiment accepted in
`docs/research/V8_10M_SCALEUP.md`. Both compact deep+xattn policies start from
the same random seed with no teacher and see all 47 V8 conditions through
`continuous_banded_v1`. The only treatment difference is the reward schedule:

- `constant_dense`: `dense_skill` for all 20,000 updates.
- `dense_to_terminal`: starts dense; once both foundation and trench samplers
  have active depth at least 2, it irreversibly linearly mixes to the terminal
  objective over 5,000 updates.

The terminal objective rewards exact success, with small workspace- and
step-efficiency terms. Its success base is scale-matched to the normalized
dense terminal component. Failure is penalized. Reward returns are not comparable
across arms. The primary comparison is exact and macro completion on fixed
promotion and development panels; productive workspace cycles and steps are
compared only on maps both policies solve.

Everything else is paired: seed `20260807`, random initialization, 47
conditions x 96 training maps, permanent 10/75/15 continuous bands, full
450-step resets, compact 2,856,685-parameter architecture, PPO settings,
20,000-update target, checkpoint interval 500, and fixed-panel spacing 1,000.
The 20,000-update jobs request `gpuhe.120h`: the earlier compact 20,000-update
run took about 25h53, so a 23h45 allocation can truncate the comparison.

The runtime Terra reward implementation is deliberately separate from the
Terra revision embedded in the accepted V8 bank. Pass its exact committed SHA
to the launcher; both arms archive and run the same Terra and terra-baselines
revisions.

First dry-run and submit both update-1 smokes:

```bash
SUBMIT=0 scripts/euler_v8_continuous_reward_anneal_v1/submit.sh smoke <runtime-terra-sha>
SUBMIT=1 scripts/euler_v8_continuous_reward_anneal_v1/submit.sh smoke <runtime-terra-sha>
```

Only after both same-binary smoke receipts pass, submit the pair:

```bash
SUBMIT=1 scripts/euler_v8_continuous_reward_anneal_v1/submit.sh screen <runtime-terra-sha>
```

The launcher never cancels another job. Pending dense job `10015084` is held by
the user with zero runtime and
superseded only after both common-binary smokes pass; cancel it explicitly at
that point before submitting this pair. Historical nearby reward screens
`10009405` and `10009411` were already cancelled with zero runtime. Their
passing update-1 smoke artifacts remain historical diagnostics.

This is a one-seed screen. Replicate only if the fixed held-out trajectories
show a material reward-schedule effect without family, tail, or anchor
regression.
