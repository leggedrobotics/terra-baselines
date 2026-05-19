# Terra Early-Learning Diagnostic

Date: 2026-05-15

## Scope

Question: are the current Terra policies actually getting stuck early, or is learning just slow and sparse? Evidence used:

- W&B histories for `ti3k3tdp` masked current, `tkzxeaas` clean baseline, `faen4uin` R1/R2 synchronized-timeout run, and `d1dkjojl` base ResMap run.
- Latest saved checkpoints copied from Euler.
- Reset-state checkpoint probe in `docs/policy_diagnostics/checkpoint_reset_probe.json`.
- Local full-horizon checkpoint rollouts in `docs/policy_diagnostics/local_rollout_*_16x550_seed0.json`.

Historical early checkpoint files are not recoverable from W&B: those runs uploaded no `.pkl` files, and the trainer overwrote one rolling `checkpoints/{name}.pkl`. For the pending 120h staggered R1/R2 run, `train_mixed.py` now preserves step-indexed snapshots at updates `0`, `100`, `500`, `1000`, `2000`, `5000`, and `10000`.

## 2026-05-19 Superseding Masking Note

The May 15 W&B comparison made the masked lineage look promising, but later
forced-mask probes on the ResMap/R1/R2 checkpoints changed the conclusion for
the current PPO default. The coarse actor mask removes invalid selections, but
it also lowers `DO`, raises `DO_NOTHING`, and did not recover completions in
local rollouts. Treat the May 15 masked checkpoint as evidence that a policy can
overfit to a mask, not as evidence that the current coarse mask should be
enabled by default.

Current default: keep PPO training/eval unmasked. Revisit masking only as a
narrow ablation for exact physical-invalid actions that can be computed with
the same semantics on the real robot.

## Local Checkpoint Evidence

The following probes were run locally on the RTX 4090 using checkpoint `.pkl` files copied from Euler into `checkpoints/`. Clean/base/R1R2 local hashes match Euler exactly:

- clean baseline: `f4937f8fed33e1843529933a448aba1625891ba028c790b84b216ab58e981d6c`
- R1/R2 sync: `752bdc6f5468d9a73016fff8257100f31ab4a30a042ccb7ac956dc2af1efa4b5`
- base ResMap: `79e13dab3a8a059fb1a03e0051947ca5a827dc16d838ba5d63935877a9c9bf7f`

The masked run is still running, so its rolling checkpoint changed while this analysis was in progress. I refreshed the local file and reran its local rollout; the refreshed local hash matched the then-current Euler hash `b2c2e1d3975fea8b0a9d54a2a6d8a06a2848d4c426ce4163be9308a731b3d578`.

All rollouts below use `16` local envs, `550` steps, seed `0`, `DATASET_PATH=/home/lorenzo/moleworks/terra_data/train`, and `DATASET_SIZE=64`.

| checkpoint | local mode | success | terrain changed | avg return | entropy | invalid selected | DO selected | main reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| refreshed masked current | masked | `4/16` | `15/16` | `4.90` | `0.067` | `0.000` | `0.063` | Historical success, but later probes show this is not a safe default. |
| refreshed masked current | raw unmasked logits | `0/16` | `14/16` | `-2.95` | `0.255` | `0.484` | `0.247` | Without the mask, the same policy selects many illegal actions. |
| clean baseline | unmasked | `3/16` | `15/16` | `4.69` | `0.049` | `0.015` | `0.059` | Real local successes; deterministic mature policy. |
| R1/R2 sync | unmasked | `0/16` | `15/16` | `0.91` | `1.852` | `0.0005` | `0.021` | Not inert: changes terrain, but no completions and too much diffuse exploration. |
| base ResMap | unmasked | `0/16` | `15/16` | `-0.44` | `1.896` | `0.028` | `0.123` | Also not inert; more invalid/unproductive action use than R1/R2. |

This local evidence strengthens the conclusion: R1/R2 is not stuck in a no-op loop, and it is not failing because it cannot edit terrain. It is failing to convert terrain edits into completed episodes at this early checkpoint. Compared with base ResMap, R1/R2 has better local return and much lower invalid-action rate, but it selects `DO` much less often.

## W&B History

| run | updates inspected | first success | success near 10k | success near 20k | latest inspected | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ti3k3tdp` masked current | 38.5k | 700 | 0.0127 | 0.424 | 0.626 at 38.5k | Not stuck; slow discovery then strong takeoff. |
| `tkzxeaas` clean baseline | 41.1k | 1700 | 0.0073 | 0.216 | 0.419 at 40k | Not stuck, but slower than masked. |
| `faen4uin` R1/R2 sync | 2.0k | 800 | n/a | n/a | 0.00073 at 1.9k | Too early to judge learning; value spikes are timeout-phase locked. |
| `d1dkjojl` base ResMap | 1.5k | one small blip around 1.1k | n/a | n/a | 0 at 1.4k eval | Too early and unstable; not enough evidence for stuck policy. |

Important curve details:

- Masked and clean runs both show the same qualitative pattern: very rare successes for the first few thousand PPO updates, then a large jump between about `10k` and `20k`.
- Masked run reaches higher success than clean by the same update range: about `0.424` vs `0.216` near `20k`, and about `0.626` vs `0.419` near `40k`. This is historical W&B evidence; it is superseded for the current default by the later forced-mask diagnostics above.
- The R1/R2 synchronized-timeout run had good early reward/max-reward movement compared with base ResMap, but the critic was dominated by synchronized timeout shocks.
- For `faen4uin`, top value-loss rows had `train/timeout_rate ~= 0.031 = 1/32`, `task_done_rate` near zero, `train/value_loss_timeout_bucket_75_100` around `0.86-1.53`, and negative bucket explained variance. That supports the timeout-spike diagnosis, not a task-terminal problem.

## Checkpoint Probe

The reset-state probe looked at `128` fresh reset observations per checkpoint. It is secondary evidence; the local full-horizon rollouts above are the direct checkpoint execution evidence.

| checkpoint | reset entropy | mean max prob | raw invalid argmax | reset argmax summary |
| --- | ---: | ---: | ---: | --- |
| masked current | 0.121 | 0.953 | 0.0 | Mostly `CLOCK`/`ANTICLOCK`; no `DO` at reset. |
| clean baseline | 0.204 | 0.914 | 0.0 | Mostly `CABIN_ANTICLOCK`/`CLOCK`; no `DO` at reset. |
| R1/R2 sync | 1.686 | 0.391 | 0.0 | High-entropy early policy; mostly cabin/turn actions. |
| base ResMap | 1.748 | 0.372 | 0.0 | High-entropy early policy; one reset `DO` argmax. |

`mean_do_valid` at reset was only `0.141`, so "no DO at reset" is not by itself a bug. The policy usually needs to navigate/turn/align before productive `DO`.

## Diagnosis

The current evidence does not support "the policies are stuck" as the main explanation.

The stronger explanation is:

1. The task has a long sparse-success horizon, so early eval success is an unreliable signal until at least several thousand updates.
2. Masked/clean policies do eventually learn, with the first meaningful transition happening around `10k-20k` updates.
3. The cancelled R1/R2 run should not be judged on learning yet because it ran only about `2k` updates and had a known synchronized-timeout critic pathology.
4. The R1/R2 architecture is not obviously collapsed: its reset/full-rollout entropy is still high, local rollouts show terrain edits in `15/16` envs, and W&B max reward/reward improved early. The critic spike issue is the main blocker to interpreting it.

## Termination Guidance

Do not terminate `66660010` just because early success is near zero before `5k` updates.

For the staggered R1/R2 run, use these gates:

- Before update `2k`: only check runtime health, timeout desynchronization, and catastrophic collapse. Do not judge learning.
- At `5k`: continue if max reward is rising or any successes appear, even if success rate is still tiny.
- At `10k`: expect at least clear eval reward/max-reward progress; if success remains zero and entropy has collapsed, terminate and change exploration/curriculum.
- Immediately terminate/rework if timeout-rate waves still hit `~1/32` with value-loss spikes after the startup staggering patch.

## Sample-Efficiency Recommendations

Highest leverage, in order:

1. Keep the staggered-timeout fix and verify it on `66660010`. The previous R1/R2 value-loss spikes are a critic target problem, and training through that noise is wasting samples.
2. Preserve early checkpoints and benchmark them. This is now implemented for the pending run.
3. Add rollout diagnostics for `terrain_changed_rate`, first productive `DO` step, loaded fraction, dump success event, and per-action entropy. Current W&B action percentages are too coarse to tell whether a policy is spinning, aligning, or making productive terrain edits.
4. Improve early exploration/curriculum before changing the architecture again: shorter first curriculum horizon, start states closer to legal dig/dump opportunities, or a mixed curriculum with easier maps until success appears reliably.
5. Stabilize the critic if spikes remain after staggering: value normalization or reward normalization, value clipping/lower value-loss coefficient, and a critic-specific LR/extra critic capacity are more targeted than changing the actor.
6. Be careful with entropy changes. Masked/clean policies eventually become very deterministic, but R1/R2 early entropy is still high. Raising entropy globally may not fix R1/R2; it is more useful if the new run shows early entropy collapse before successes.

## 2026-05-16 R1/R2 Checkpoint Diagnosis

Run `xxf7eoap` / Slurm `66670662` reached about update `3500` with timeout desynchronization working:

- `train/timeout_rate ~= 0.0018`, close to `1/550`, not the old `1/32` synchronized wave.
- `explained_variance ~= 0.999`.
- `entropy ~= 1.9`, while max entropy for eight actions is `ln(8) = 2.08`; entropy did not collapse.
- `eval/success_rate = 0`.

Latest-checkpoint local probes showed the policy is not no-op, but also not finishing:

| checkpoint | mode | success | avg return | entropy | invalid selected | DO selected | terrain-change/action | terrain-changed envs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| update 0 | unmasked | `0/16` | `-3.81` | `2.08` | `0.938` | `0.938` | `0.000` | `0.000` |
| update 500 | unmasked | `0/16` | `-1.93` | `1.91` | `0.0001` | `0.045` | `0.010` | `0.625` |
| update 2000 | unmasked | `0/16` | `-2.12` | `1.90` | `0.055` | `0.121` | `0.011` | `0.562` |
| latest | unmasked | `0/16` | `-1.84` | `1.96` | `0.0003` | `0.068` | `0.010` | `0.562` |
| latest | forced mask | `0/16` | `-1.76` | `1.86` | `0.000` | `0.068` | `0.010` | `0.562` |

The forced-mask diagnostic did not recover success, so invalid actions are not the main blocker for this checkpoint. The policy learned to avoid the initial invalid-DO collapse by update `500`, but it mostly moves/turns and uses productive terrain-changing `DO` too rarely to complete the task.

The stronger bug was in reward semantics: W&B showed nonzero `eval/failure/rewards/terminal` even when `eval/success_rate=0`. Code inspection found that `terra/state.py` paid terminal reward on `done`, and `done` includes max-step timeout. This lets a partial-completion timeout receive the success-shaped terminal bonus, so PPO can optimize "be partially complete at timeout" instead of "reach `task_done`".

Fix applied on 2026-05-16:

- Gate `terminal_r` on `task_done`, not `done`.
- Add a validation fixture with `done=True`, `timeout_done=True`, `task_done=False`, `completion > 0.5`, and terminal reward `0`.
- Cancel `66670662`.
- Launch replacement `66756388` with the same architecture and `terminalfix` in the W&B/checkpoint name.

## Current Action

`66756388` is the correct next experiment: same R1/R2 architecture, startup timeout staggering, terminal reward only on true task completion, 120h queue, and preserved early checkpoints. Judge it after the first checkpoint sequence rather than the first few hundred updates.

First live evidence from `66756388` / W&B `04e8dada`:

- The run started on `eu-g6-072` with `4 x RTX 4090`; GPU guard, CUDA/cuDNN/NCCL preflight, and full-shape W&B-disabled smoke all passed.
- W&B step `0` and checkpoint `update_000000` agree: `entropy=2.0771589`, essentially the maximum `ln(8)=2.07944`, so entropy is not collapsed.
- The terminal reward fix is live: `eval/failure/rewards/terminal=0` and `eval/rewards/terminal=0` with `eval/success_rate=0`.
- Initial `value_loss=0.0018877` and `explained_variance=0.0001058` are expected at initialization and are not yet learning evidence.
- Continue to the update `100` eval and the preserved update `100` checkpoint before making the next learning call.

Update `100` evidence:

- W&B step `100`: `entropy=1.9799`, `value_loss=0.00231`, `explained_variance=0.954`, `train/timeout_rate=0.0017166`, `train/task_done_rate=0`.
- Eval: `eval/success_rate=0`, `eval/positive_terminations=0`, `eval/max_reward=0.86875`, `eval/rewards=-0.00295`, and terminal rewards on eval failures remain `0`.
- Local rollout JSON: `docs/policy_diagnostics/04e8dada_update100_rollout_32x550_seed0_1.json`.
- Probe implementation note: local rollout terrain scores now use `info["final_observation"]["action_map"]` captured at first done per env, so the final dig/dump coverage is not computed from a post-timeout reset map.
- Local probe result, `32` envs x `550` steps, seeds `0` and `1`, unmasked plus forced-mask:
  - seed `0`, unmasked: `0/32` success, avg return `-2.406`, mean entropy `1.985`, invalid selected `1.02%`, `DO=2.20%`, `DO_NOTHING=40.4%`, terrain changed `10/32`, final dig coverage `0.075`, final dump coverage `0.281`.
  - seed `0`, forced mask: `0/32` success, avg return `-2.386`, invalid selected `0`, `DO=1.19%`, `DO_NOTHING=41.4%`, terrain changed `10/32`, final dig coverage `0.075`, final dump coverage `0.281`.
  - seed `1`, unmasked: `0/32` success, avg return `-2.602`, mean entropy `1.959`, invalid selected `0`, `DO=1.18%`, `DO_NOTHING=31.7%`, terrain changed `8/32`, final dig coverage `0.071`, final dump coverage `0.219`.
  - seed `1`, forced mask: `0/32` success, same action counts as unmasked, mean entropy `1.883`, terrain changed `8/32`, final dig coverage `0.071`, final dump coverage `0.219`.

This does not change the diagnosis. Entropy is healthy and the reward fix is live, but the policy is still too early and mostly moves/no-ops with sparse productive `DO`. The next meaningful checkpoint is `500`, with `1000/2000/5000` needed before changing architecture or rewards unless a collapse signal appears.

Update `500` evidence from `04e8dada`:

- W&B around step `536`: `entropy=1.8695`, `value_loss=0.00244`, `explained_variance=0.9907`, `train/timeout_rate=0.001724`, `train/task_done_rate=0`, and terminal rewards on eval failures remain `0`.
- Eval still has no completions: `eval/success_rate=0`, `eval/positive_terminations=0`, `eval/max_reward=0.823`, `eval/rewards=-0.00046`.
- Behavior is improving: `eval/DO=0.0725`, `eval/DO_NOTHING %=0.156`, `behavior/terrain_changed_rate=0.0309`, `behavior/dig_success_rate=0.0155`, `behavior/dump_success_rate=0.0154`.
- Local rollout JSON: `docs/policy_diagnostics/04e8dada_update500_rollout_32x550_seed0_1.json`.
- Local probe result:
  - seed `0`, unmasked: `0/32` success, avg return `-1.412`, max return `1.312`, mean entropy `1.838`, invalid selected `4.09%`, `DO=5.56%`, `DO_NOTHING=5.13%`, terrain changed `30/32`, final dig coverage `0.505`, final dump coverage `0.938`.
  - seed `0`, forced mask: `0/32` success, avg return `-1.208`, max return `1.312`, invalid selected `0`, `DO=2.11%`, `DO_NOTHING=23.5%`, terrain changed `30/32`, final dig coverage `0.505`, final dump coverage `0.938`.
  - seed `1`, unmasked: `0/32` success, avg return `-1.046`, max return `2.464`, mean entropy `1.881`, invalid selected `5.02%`, `DO=4.55%`, `DO_NOTHING=17.5%`, terrain changed `30/32`, final dig coverage `0.505`, final dump coverage `0.933`.
  - seed `1`, forced mask: `0/32` success, avg return `-0.893`, max return `1.652`, invalid selected `0`, `DO=1.60%`, `DO_NOTHING=30.4%`, terrain changed `30/32`, final dig coverage `0.499`, final dump coverage `0.933`.

This is the first clear behavioral improvement in the terminal-fix run. It does not prove learning to completion, but it argues against killing or changing the architecture at update `500`. The policy has moved from sparse terrain interaction at update `100` to broad terrain interaction at update `500`; continue to `1000/2000` and watch whether `DO`, max reward, positive returns, dig/dump success, and completion metrics continue to rise.

Update `1000` evidence from `04e8dada`:

- W&B has tiny nonzero eval success rows at step `600` (`eval/success_rate=0.000244`), step `900` (`0.000244`, `eval/max_reward=4.59`), and step `1000` (`0.000977`, `eval/max_reward=5.22`). This is promising early signal, not persistent success yet.
- Step `1100` eval returned to `0` success with `eval/max_reward=0.847`, `eval/rewards=0.000135`, `eval/DO=0.0886`, and `eval/DO_NOTHING %=0.155`. Completion still trends upward, so this weakens any "persistent success" claim but does not by itself justify intervention before update `2000`.
- Step `1200` eval returned to tiny success: `eval/success_rate=0.000244`, `eval/max_reward=5.45`, `eval/rewards=0.000303`, `eval/DO=0.0829`, and `eval/DO_NOTHING %=0.153`.
- Step `1300` eval/latest summary at step `1320`: tiny success rose to `0.001709`, `eval/max_reward=5.54`, `eval/rewards=0.000518`, `eval/DO=0.0820`, `eval/DO_NOTHING %=0.159`, `entropy=1.8642`, `value_loss=0.00148`, `explained_variance=0.9956`, `train/timeout_rate=0.00182`, `train/task_done_rate=0`, `progress/completion=0.6711`, `core=0.8164`, and `edge=0.6129`.
- Value/termination history through W&B step `1053`: no return of the old synchronized timeout spike pattern. `value_loss` median is `0.00185`, p95 `0.00264`, max `0.00627`; `explained_variance` median is `0.992`; `train/timeout_rate` stays near `1/550` (`0.00153-0.00208`); `train/task_done_rate` is mostly `0` with rare `1.5e-5` rows.
- Local rollout JSON: `docs/policy_diagnostics/04e8dada_update1000_rollout_32x550_seed0_1.json`.
- Local probe result:
  - seed `0`, unmasked: `0/32` success, avg return `-1.290`, max return `0.981`, mean entropy `1.724`, invalid selected `2.55%`, `DO=7.40%`, `DO_NOTHING=10.8%`, terrain changed `27/32`, final dig coverage `0.587`, final dump coverage `0.790`.
  - seed `0`, forced mask: `0/32` success, avg return `-1.180`, max return `1.528`, invalid selected `0`, `DO=4.85%`, `DO_NOTHING=21.0%`, terrain changed `27/32`, final dig coverage `0.587`, final dump coverage `0.790`.
  - seed `1`, unmasked: `0/32` success, avg return `-0.986`, max return `1.562`, mean entropy `1.859`, invalid selected `1.90%`, `DO=7.45%`, `DO_NOTHING=5.02%`, terrain changed `29/32`, final dig coverage `0.627`, final dump coverage `0.897`.
  - seed `1`, forced mask: `0/32` success, avg return `-0.841`, max return `1.562`, invalid selected `0`, `DO=4.94%`, `DO_NOTHING=20.5%`, terrain changed `29/32`, final dig coverage `0.621`, final dump coverage `0.897`.
- Failure summary: final undug fraction is still high at about `0.37-0.41`. Positive moved dirt exists in `26/32` seed-0 failures and `29/32` seed-1 failures; within those positive-dirt failures, only about `1.0-2.7%` of moved dirt is off the dump zones, and all positive-dirt failures are below the simple `<=20%` off-dump threshold. Only `9-11/32` failures meet the "mostly dug" threshold (`<=20%` undug), and `8-10/32` envs are still loaded at timeout. The current local evidence points more to unfinished digging/sequence completion than to wrong-place dumping, but it does not prove every failed env completed dumping.

Interpretation: update `1000` is still not a learned policy, but it is not collapsed. Entropy remains high, terminal-reward leakage is still absent, `DO` increased to about `7.4%`, invalid raw selections fell to about `2%`, dig coverage improved to about `0.59-0.63`, and terrain-change/action improved to about `0.021-0.025`. W&B eval blips are now appearing in more than one eval row, so continue to update `2000` before making a new architecture or reward change unless these behavior metrics regress.

Latest rolling-checkpoint evidence at W&B step about `1258`:

- Local rollout JSON: `docs/policy_diagnostics/04e8dada_latest_step1258_rollout_32x550_seed0_1.json`.
- Result: still `0/128` successes, but final maps improved versus update `1000`: final dig coverage is about `0.65-0.70`, final dump coverage about `0.89-0.97`, final undug fraction about `0.30-0.35`, off-dump moved dirt about `0.4-1.4%`, and loaded-at-timeout is down to `0-1/32`.
- New concern: unmasked `DO` fell to `4.1-4.8%`, forced-mask `DO` fell to `2.2-2.3%`, and `DO_NOTHING` rose to `27-43%`. The checkpoint is closer in terrain state but may be drifting toward too much waiting.

Stochastic rolling-checkpoint diagnostic:

- W&B eval samples from the policy distribution, while the first local probes use deterministic argmax. Added `--stochastic` to `scripts/analysis/probe_checkpoint_rollouts.py` and saved `docs/policy_diagnostics/04e8dada_latest_step1258_stochastic_rollout_32x550_seed0_1.json`.
- Stochastic unmasked result, seeds `0` and `1`: `0/64` successes, `DO=8.0-8.1%`, `DO_NOTHING=15.2-15.7%`, mean entropy about `1.88`, invalid selections about `7.1-7.3%`, final dig coverage `0.74-0.77`, final dump coverage `0.995`, final undug fraction `0.23-0.26`, no loaded-at-timeout failures, and off-dump moved dirt about `0.4-0.5%`.
- Interpretation: the high deterministic no-op rate is partly an argmax artifact. The W&B-comparable bottleneck is now more specific: the policy moves dirt to the correct dump zones but still leaves about a quarter of required dig tiles unfinished by timeout.

Clean-baseline stochastic target:

- Local rollout JSON: `docs/policy_diagnostics/clean_baseline_stochastic_rollout_32x550_seed0_1.json`.
- Same local dataset size and stochastic unmasked probe on the mature clean checkpoint: seed `0` succeeds `14/32`, seed `1` succeeds `12/32`, max return is about `9.0`, average success step is about `111-133`, and average episode length is about `368-386`.
- Clean target final state: final dig coverage `0.89-0.90`, final undug fraction `0.095-0.112`, final dump coverage `0.966-0.968`, and off-dump dirt only `0.1-0.3%`.
- Clean target policy behavior: entropy is nearly deterministic (`0.032-0.041`), invalid selection is low (`0.3-0.8%`), `DO` is similar to current stochastic (`7.2-8.0%`), but productive terrain-change/action is about `0.058-0.064` versus current `0.029-0.031`.
- Interpretation: current `04e8dada` does not need more raw `DO`; it needs more productive DO and a sharper finishing strategy. The clean policy uses a similar amount of DO but turns roughly twice as many actions into terrain changes, leaves roughly half as much undug target, and becomes much more deterministic.
