# Experiments — completed log

The first table is the historical spatial-encoder line. Metrics:
eval/positive_terminations /
eval/rewards (final), swhr = eval/success_within_horizon_rate, ep_len =
eval/avg_positive_episode_length. Full incident trail:
`docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`.

| Date | Run | W&B | Config | Final | Verdict |
|---|---|---|---|---|---|
| 07-19/20 | spatial A/B | pqtmfmqy | resnet_spatial_8x8 base, 10k updates, mb32 | 2.810 / 0.131 | spatial beats atari +39% per-sample at 3.9× wall-clock; became the E3 teacher |
| 07-20 | atari control | nnsksyva | atari base, 10k, mb32 | 2.020 / 0.095 | control; 113k steps/s |
| 07-21 | E2 | nr032qs7 | se+bf16+critic512 FROM SCRATCH, 20k | 2.329 / 0.108, swhr 0.976 | below teacher — from-scratch drag of heavier bundles (bf16 confirmed +50% throughput) |
| 07-22 | E1 | 3buorfp3 | spatial_8x8 + algo fixes (no value clip, flat shuffle, 19k ent), 20k | 2.833 / 0.132, swhr 0.997, ep_len 59.2 | beats teacher on its own encoder → PPO fixes net-positive |
| 07-22 | E3 | j0bs2fkl | medium se, grown init + kickstart from pqtmfmqy, 20k | **3.054 / 0.142, swhr 0.997, ep_len 55.2** | **best policy to date (+8.7%); kickstart playbook validated — surpassed teacher at 30% budget** |
| 07-22 | E4 | k8vnwp5u | se_xattn FROM SCRATCH, 20k | 2.586 / 0.121, swhr 0.993, ep_len 63.0 | xattn beats SE from scratch +11% (2.59 vs 2.33) — real architecture win; still below warm-started runs |
| 07-22 | E5 | gud7cbwg | dumpzone transfer (E3 warm-start + teacher), CANCELLED @850/20k | 0.000 / −0.005 | naive cross-task kickstart does NOT bootstrap a new task family (no reward stream found); E5b = 2-stage curriculum + higher entropy when dumpzone becomes priority |
| 07-22 | E9 | — | 128×128 pilot, 5-stage medium SE, teacher_obs_downsample=2 | FAILED in smoke | teacher module was built with the student 128 env, causing 128-row position embeddings for a 64-row teacher checkpoint; fixed locally with regression + real-checkpoint CPU smoke before relaunch |
| 07-22 | E9b | — | fixed 128×128 pilot relaunch, same 512 env/GPU memory shape | FAILED in smoke | teacher-env fix worked, but 128×128 PPO update OOMed (`RESOURCE_EXHAUSTED`, temp 10.70 GiB plus 11.49 GB allocation attempt); next gate should try `num_minibatches=64` |
| 07-23 | E9c | 0ixsswn4 | fixed 128×128 pilot relaunch, same 512 env/GPU with `num_minibatches=64` | CANCELLED @~1.4k; all-NaN from smoke update 0 | memory-fit shape worked, but smoke gate missed NaN loss/params: smoke FINAL and production checkpoint both had all model params NaN; forward-only behavior was a symptom, not ordinary learning failure |
| 07-23 | E9d | — | E9c plus embedding clamp, local-map `IntMap`, local-map area scale 4, loaded/downsample fix, finite guard | SUBMITTED 8323457 | exact local full-shape 1-update gate passed finite before Slurm submit; job pending priority |

Cross-run findings:
- Warm-start (grow + kickstart) >> from-scratch for introducing architecture/capacity
  changes (E3 vs E2; E4' line continues this).
- bf16 encoder compute: +50% production throughput (43k vs 28.8k steps/s), numerics clean.
- Finite-loss/param checks are now mandatory for smoke gates: E9c completed update 0 and
  entered production despite all-NaN model params/loss scalars. The E9d script keeps
  `--fail_on_nonfinite` through smoke and production.
- Attention follow-up launched as E10 (pending at submission): `--attention_compute_dtype
  float32` isolates attention-softmax precision inside a bf16 trunk;
  `--token_mixer_residual_init_scale 0.001` wakes v5 mixer gradients without changing the
  default identity-at-init contract.
- pmap scaling ~95% (single-GPU probe 11.4k vs 10.8k/GPU ×4).
- Entropy-schedule stretch makes mid-run comparisons vs 9.5k-schedule runs invalid —
  compare finals or matched entropy phase only.
- Episode length: E3 55.2 vs E1 59.2 steps — bigger warm-started net is also faster per
  episode; 300-step horizon (E6) exerts no pressure since episodes are ~55 steps.

## Accepted-bank campaigns

The metrics below are deterministic fixed-bank terminal absolute completion,
not the legacy online spatial-run summaries above. Promotion and development
remain separate. Exact is solved maps / evaluated maps.

| Date | Campaign / arm | Slurm | Selected update | Macro P / D | Exact P / D | Verdict |
|---|---|---:|---:|---:|---:|---|
| 08-02 | P5 six-arm accepted-bank screen | — | 2,000 | generalists 0.574--0.588 / 0.574--0.577 | at most 1/512 | all six completed and passed; `G-ADAPTIVE` selected only by the predeclared retention gate, not as a general scheduler claim |
| 08-02 | P5b `G-MEDIUM-ADAPTIVE-WARM` | `9378174` | 2,000 | 0.652 / 0.625 | 1/512 / 2/512 | completed 2,000, `PASSED`; strongest selected constrained distance axis |
| 08-02 | P5b `G-DEEP-ADAPTIVE-WARM` | `9378175` | 1,000 | 0.653 / 0.628 | 2/512 / 2/512 | completed 2,000, `PASSED`; transient matched gain, not retained at 2,000 |
| 08-02 | P5b `G-MEDIUM-UNIFORM-WARM` | `9378176` | 1,000 | 0.647 / 0.664 | 2/512 / 6/512 | completed 2,000, `PASSED`; best selected development family floor, transient at 2,000 |
| 08-03 | all-free capability-floor evaluation | local fixed eval | selected P5/P5b checkpoints | 0.385--0.718 / 0.465--0.736 | generalists 0/32; trench specialist 1/32 promotion only | integrity-clean diagnostic; physically easier but target-mask OOD, excluded from constrained macro |
| 08-03 | P5c five-arm low-entropy screen | `9461489`, `9461500`, `9461504`, `9461507`, `9461512` | none | deep latest 0.624 / 0.586 | deep latest 168/512 / 143/512 | all fixed evaluations clean; no arm passed the long-run gate at two consecutive checkpoints |
| 08-10 | V8 Atari-base small-system control | `10128519` | 19,000 (descriptive promotion selection) | 0.457 / 0.428 | 16/752 / 20/752 | completed 20k; zero mastered conditions and depth 0/0; negative 480k-system result, not an encoder-only ablation |

P5b result root:
`/home/lorenzo/moleworks/.artifacts/terra_p5b_results_20260802_6c56610e`.
Standard leaderboard:
`/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`.
Capability-floor results:
`/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_eval_20260802`.
The parameters-only current-protocol E8 compatibility replay scores
`0.013/0.027` macro and `0/32` exact; its historical near-1 online `swhr` is a
different evaluation contract and is not numerically comparable.

P5b deep used function-preserving growth (`2,441,223 -> 2,699,117`), a fresh
optimizer, and the frozen parent as KL/value teacher. E8 was not a larger E3:
both had `2,441,223` parameters. The likely recipe mismatch is P5b entropy
`0.15 -> 0.005 / 7,600`, still about `0.137` at the synchronized update-1,500
decline when KL reached zero. This is the P5c hypothesis, not a post-hoc claim
that entropy caused the decline.

## Completed fixed-evaluation screen: P5c

P5c freezes five 4,000-update arms with evaluation every 500 updates: medium
adaptive, medium uniform, deep uniform, foundation medium-uniform, and trench
medium-uniform. Allocated update-1 smoke jobs `9458568`, `9458581`, `9458585`,
`9458616`, and `9458619` all completed `0:0` and passed. Screen jobs `9461489`,
`9461500`, `9461504`, `9461507`, and `9461512` were then submitted from
revision `3478af87950d3d35059344b078209d00785c8481` and crossed finite update
1 with transition-integrity checks enabled. All five subsequently completed
4,000 updates in 6.2--8.0 hours. All share entropy
`0.02 -> 0.005 / 10,000` and the common P5 parent/teacher. The specialists are
family dose ceilings; they do not enter the causal sampler/depth comparison.
All 40 numbered checkpoints were evaluated on constrained
promotion/development and diagnostic all-free promotion/development: 160
integrity-clean evaluations and 43,520 episodes with no integrity failure.
Deep uniform at update 4,000 is the strongest descriptive endpoint:
promotion/development macro `0.624/0.586`, foundation `0.556/0.533`, trench
`0.711/0.654`, and exact `168/512` / `143/512`. Its checkpoint SHA-256 is
`4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc`.

This endpoint is not a formal selection. Foundation specialist had one clean
interval at update 3,000, deep uniform one at 4,000, medium adaptive one at
2,500, trench specialist one at 3,500, and medium uniform none. No arm passed
the predeclared improvement/retention gate at two consecutive checkpoints.
No 120-hour continuation or P6 training was launched. Online training success
was still rising, so the result is also not saturation evidence; it says the
current learning curve is unstable on fixed held-out panels.

The condition-balanced leaderboard is frozen at
`/home/lorenzo/moleworks/.artifacts/terra_p5c_leaderboard_20260803_3478af8/LEADERBOARD.md`
with input digest
`ac665b7088942b66159a52f7170c1484dc6e36175f2ec7decbd8c4383094c5ac`.
The complete read-only campaign archive is
`/cluster/work/rsl/lterenzi/terra_p5c_campaign_20260803_3478af87950d3d35059344b078209d00785c8481/`;
its payload-manifest SHA-256 is
`605922d0965206f82e7fe54a10fac202e028b548de24454febcd2691709ff42f`.
Future behavioral screens receive at least 24 healthy hours with an oversized
absolute update target; an admitted checkpoint continues with true resume
state on the 120-hour queue. See
[`research/P5_ACCEPTED_BANK_EXPERIMENTS.md`](research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
section 12.
