# v6_3m_yolo_rv2 — rationale for every change (review document, 2026-08-11)

One combined "YOLO" screen carries every architecture/optimization improvement
at once because GPU budget does not allow the one-treatment-per-axis ladder.
Screen semantics, agreed in advance: a WIN must later be decomposed through
single-treatment runs before any paper claim; a LOSS indicts nothing
individually. This document gives the evidence behind each bundled change so
the design can be reviewed independently of the outcome.

## The three arms

| arm | architecture | D3 masking | runtime terra | status |
|---|---|---|---|---|
| `reward_v2_scratch` (baseline) | compact `se_xattn`, 2,856,701 params | no | `3051054b` | phase1 running |
| `v6_3m_yolo_rv2` | v6 bundle, 2,134,771 params | yes | `04c67bba` (= 3051054b + mask obs) | killed (day 2) |
| `v6_3m_yolo_rv2_nomask` | v6 bundle | no | `3051054b` | killed (day 2) |
| `v6_1_rv2` | 3-delta readout, no aux head, 2,303,421 params | no | `3051054b` | replaces both, 2026-08-11 |

All arms: reward-v2, `continuous_banded_v2`, carry-work obs, seed 20260807,
4×512×32/32 minibatches, 2 epochs, LR 3e-4, entropy 0.15→0.02/20k, phase1 =
14,000 updates on 4×4090 — matched updates AND transitions. The nomask arm
exists because masking has failed for us before (Lorenzo's prior attempts) and
because it isolates masking against the architecture bundle; it runs on the
baseline's exact terra because the mask terra computes the mask
unconditionally per step.

## Evidence base (measured on our own runs, 2026-08-10)

1. **Rank collapse at the readout.** Trained compact encoder-output effective
   rank: 6.0 of 160 dims (10M: 5.5). 62% of the 10M's final-grid channels
   near-dead. The interface is signal-starved, not width-starved.
2. **The flatten Dense is a frozen positional matrix.** 41% of compact params
   (6144→192), receives 2–3% of encoder-internal gradient, moves at 0.25× the
   median RMS(grad)/RMS(param). Impoola (arXiv 2503.05546; PPO at exactly our
   64×64→8×8 geometry) independently identifies this layer as the scaling
   bottleneck and wins with 35% fewer params after removing it.
3. **The 10M scale-up failed structurally**: 0% of its +7.4M params landed
   after the encoder exit; its competence was acquired only inside a
   distillation window (+2 maps over 8,000 further PPO updates); it never ran
   from scratch (OpenAI Five precedent: non-function-preserving surgery
   plateaus below scratch).
4. **PPO's own signal is thin**: one 9-way action + a value scalar per
   transition. Dense per-cell auxiliary supervision (UNREAL, GridNet, Lux
   winners) is the standard remedy.
5. **vf_coef=2.0 value-dominates the shared trunk** (PPG; Moalla et al. —
   both PPO evidence) while EV sits at 0.985–0.99, i.e. there is slack to
   trade value-fit for policy signal.
6. **Obstacle stalls are mechanics, not reward**: 7 dev episodes with 270–450
   consecutive no-effect actions; the env computes an exact effect-based
   action mask but returned it as an info-only zeros field (audit D3).

Full sources: `V8_ARCH_SCALING_DIAGNOSIS_20260810.md` (measurements),
`V8_ARCH_SCALING_LITREVIEW_20260810.md` (187-source review),
`architecture_opt_redesign_20260810_1645.md` (plan + run contract).

## Change-by-change rationale

| # | change | evidence | expected effect | risk / mitigation |
|---|---|---|---|---|
| 1 | Flatten shrink: 1×1 conv 96→32 before flatten (−783k params) | Evidence 2; staged-epoch layout-memorization churn | removes the memorization-prone dead block; frees budget | flatten carried ~2× the xattn contribution in the healthy checkpoint — shrunk, not deleted |
| 2 | Token mixer ON (`se_sa_xattn`, residual scale 0.1) | 8×8 tokens never interact except through convs; machinery existed identity-init since F14 | content-based global mixing before both readouts | live (0.1) not zero-init because from scratch there is no function to preserve |
| 3 | Latent queries 4→8 | 5-query readout was the only content-addressed path (480-d total) | more attention bandwidth at the readout | +62k params only; qkv kept 96 = token dim |
| 4 | Stage rebalance (2,2,3,3)→(3,3,2,2) | dig-cell semantics live at 64×64; stage 0 held 1.5% of params; Lux winners keep ~85% at full res; Procgen-HD input-res gains | spatial detail for trench/dig geometry | −25% steps/s (measured 18.8k→14.1k, dominated by this change; predicted +30–50% FLOPs in the plan doc) — priced in, fits 24h queue |
| 5 | Per-cell aux decoder (BCE 0.25, f32 head, 32×32: remaining-dig / dump-deficit / dumpability / obstacle) | Evidence 1+4 — THE rank-collapse fix; targets are obs-derived (no env change) | forces task-semantic spatial features; doubles as 8×8 bandwidth test | +25k params; head dropped at deploy; f32 after bf16 eager-init tripped cuDNN (job 10307312) |
| 6 | vf_coef 2.0→0.5 | Evidence 5 | more policy-shaped trunk gradient | EV may drop (observed 0.99→0.94 online — within tolerance); non-architecture delta, listed so the pair is not read as pure-architecture |
| 7 | D3 action-logit masking (`where(mask, logits, −1e9)` in rollout+loss+eval; mask appended as obs[22]; DO_NOTHING always valid) | Evidence 6; Huang & Ontañón invalid-action-masking for PPO | eliminates no-effect exploration/stalls | Lorenzo's prior masking attempts failed → the nomask arm isolates it; rollout/loss/eval provably share one distribution (unit-tested); no all-invalid rows possible; support-restricted entropy reads ~0.3 lower mechanically |
| 8 | Params 2,134,771 (−25% vs baseline) | deliberate: a win at fewer params is the stronger result (Impoola won at −35%); rank evidence says width is not binding | — | a LOSS is ambiguous (shape vs size) — accepted for a screen |

## v6.1 (`v6_1_rv2`, launched 2026-08-11)

Day-2 evidence invalidated the premises behind four of the eight bundled
changes, so both yolo arms were killed and replaced by one arm that keeps only
the readout redesign. Exactly three architecture deltas remained against the
frozen compact reward-v2 source contract:

| # | flag | v6.1 | was (yolo) |
|---|---|---|---|
| 1 | `--map_encoder` + `--token_mixer_residual_init_scale` | `se_sa_xattn`, 0.1 | same |
| 2 | `--flatten_reduce_channels` | 32 | same |
| 3 | `--attn_latent_queries` | 8 | same |

Reverted to the baseline: `--resnet_blocks_per_stage 2,2,3,3` (change 4, the
full-res stage rebalance — also returns the ~25% steps/s it cost), the per-cell
aux decoder (change 5 — `--aux_coef 0`, and `get_model_ready` builds the head
exactly when `aux_coef > 0`, so the decoder is absent from the tree, not merely
untrained), `vf_coef` 2.0 (change 6 — the flag is not passed at all, so the
trainer default applies), and D3 action-logit masking (change 7), which also
puts the arm back on the baseline's exact terra `3051054b`. Everything else —
seed 20260807, bank, sidecar, sampler, reward-v2 flags, LR, entropy schedule,
4×512×32/32 — is the baseline's, so updates and transitions still match.

Parameters: 2,303,421 under carry-work (2,303,405 without). The count rises vs
the yolo arms only because the stage rebalance is reverted (the dropped aux head
is 24,804 of it); it is still 19.4% below the compact baseline's 2,856,701.

The phase-1 implementation was frozen at commit `9abf88eb`. Its configurable
multi-arm launch wiring was removed after the immutable checkpoints and
receipts were recorded; the active launcher now supports only the selected
v6.1 continuation recipe.

## Deliberately NOT changed

LR 3e-4 (worked at this scale; LR-width sweep is reserved for the 10M ladder),
`max_grad_norm` 0.5 (binds ~100% everywhere but Adam largely cancels chronic
clipping; changing it here would add an axis), entropy schedule, the 160-d
encoder exit and heads at 3M scale (rank 6/160 says the interface is not the
binding constraint), the 8×8 exit (trenches — narrowest geometry — score 0.98;
the aux decoder doubles as the bandwidth test; 16×16 only if post-aux
predictions blur at cell boundaries), reward (belongs to R2), sampler, bank,
horizon.

## Readout plan

- Panels every 1k updates (in-job sweep at phase1 end + on-demand observer
  jobs on retained checkpoints; observer sbatch staged on Euler under
  `observer_evals/`).
- Primary: frozen promotion panel exact/720; development for generalization;
  per-condition tails for the foundation clusters that motivated the design.
- Decomposition logic: nomask-vs-baseline = architecture bundle;
  mask-vs-nomask = D3; any single-component claim requires its own later run.
- Early online reading (u8000, de-confounded per-condition EMAs): masked YOLO
  trails baseline (unweighted mean 0.078 vs 0.136) with both arms at ~0 on all
  constrained foundations — the deciding divergence is expected late, where
  foundations convert.

## Provenance

Branch `experiment/v8-v6-yolo-rv2-20260810` (worktree
`.worktrees/terra_baselines_v8_v6_yolo_rv2_20260810`), forked off Lorenzo's
`290d258` + merged `3979b0c` phase split. Key commits: `8716f9b` v6 readout,
`15d9e9f` aux loss + vf_coef, `825a391` D3 baselines wiring, `04c67bba`
(terra) D3 mask-in-obs, `3857518` phase mirror + terra pin, `4922c58` f32 aux
head, `7fca4eb`+`ae951d4` host-side init (cuDNN eager-init failures
10307312/10307751/10308172), `dedf850` nomask variant. Jobs: masked phase1
10311471; nomask smoke 10387993 (phase1 auto-chained); observers
10388070/10388071. Tests: 408 passed at launch revision; smoke receipts pin
2,134,771 params.

## External review response (2026-08-11)

Reviewer verdict accepted: finish the screen; current evidence favors compact
at matched updates; reward-v2 needs no broad retune. Adopted from the review:
paper-facing name = **"V6 spatial-auxiliary system"** (eight-change bundle,
never an encoder ablation); V6 is not the cheap model for reward pilots; the
nomask arm is essential (was already chained). Stale item: the failed
observers were fixed+resubmitted (10388599/10388600) before the review
landed. Pre-registered next reward experiment, gated on the compact u14k
panels remaining ambiguous: `shaping_weight ∈ {1.0, 0.5}`, two 1-GPU arms,
~1M spatial ResNet pilot, 4k updates (extend to 10k only if both anchors
≥12/16), confirm any winner on compact; one fixed manifest, one scalar, one
runner. Proposed 8-condition manifest (shaping-sensitive + anchors):
v7-fnd-slab-adjacent, v7-fnd-bearing-walls-adjacent (D4b overlap cells),
fnd-slab-apron-d12, fnd-slab-apron-d16 (illegal-dump planning cells),
fnd-slab-split (dump-choice canary), trn-straight-allfree,
v7-trn-tee-adjacent, fnd-slab-allfree. Note: halving w also halves the
relocation-progress guidance through Φ — this doubles as the first data on
the relocation-ablation question (see the β discussion of 2026-08-10).

### Review claim audit (2026-08-11, second pass)

- "Compact ~44% online exact at u10.5–11.5k" — **VERIFIED** (window mean
  0.437, p90 0.482); my earlier lower reading was a stale tail. Withdrawn.
- "d12/d16 failures ~75–85% illegal-dump" — provenance unresolved: no
  reward-v2 panel existed at review time, so this can only come from
  dense-run artifacts. Being re-measured natively from the observer evals
  (per_map terminal_illegal_dump_volume / dump_purity on the rv2
  checkpoints); the answer feeds the shaping_weight pilot trigger.
- "Success/timeout returns separated ≥7.46 in every rated condition" —
  treated as the analytic admission bound (design guarantee), not an
  empirical per-condition measurement, until sourced otherwise.
- Process rule going forward: quantitative claims in reviews carry a
  provenance line (metric key + window, or file + field).
- Eval-shim crash (missed by the review, found while chasing its stale
  observer note): fixed in 9925ebb; both phase1 in-job eval stages would
  have crashed; observers on the fixed tree are the recovery path.
