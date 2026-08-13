# V8 v6.1 readout bundle vs the compact baseline — matched-pair result (2026-08-13)

Paper ablation record. One matched pair, one seed, 14,000 updates: the compact
`se_xattn` reward-v2 baseline against the v6.1 three-delta readout bundle. The
treatment is **smaller** than the control by 19.4% of parameters and wins the
promotion panel by +126 exact completions (+17.5pp) at the terminal
checkpoint, with no sign of a plateau.

## 1. The matched pair

| | control | treatment |
|---|---|---|
| arm | `reward_v2_scratch` | `v6_1_rv2` |
| map encoder | `resnet_spatial_8x8_se_xattn` (compact) | `resnet_spatial_8x8_se_sa_xattn` |
| parameters | 2,856,701 | 2,303,421 (**−553,280, −19.4%**) |
| readout deltas | — | token-mixer residual init 0.1; 1×1 flatten-reduce 96→32; 8 latent queries (from 4) |
| aux decoder | none | none (`--aux_coef 0` drops the head entirely) |
| `vf_coef` | 2.0 | 2.0 |
| D3 action-logit masking | off | off |
| runtime terra | `3051054b` | `3051054b` |

Everything else is byte-identical between the arms: reward-v2 (protocol
`material_potential_v2`, distance protocol
`obstacle_geodesic_8_physical_global_v1`, sidecar
`f0c43065…`), sampler `continuous_banded_v2` on the 47-condition
`terra_v8_v6_constraints_v7_adjacent_train96_v5` bank (96 maps/condition),
carry-work observation channel, seed 20260807, batch shape 4×512×32 with 32
minibatches and 2 epochs (65,536 transitions/update), LR 3e-4, entropy
0.15→0.02 over 20,000 updates, horizon 450, `--no_value_clip`,
`--flat_minibatch_shuffle`, `bfloat16` encoder / `float32` attention. Matched
updates **and** matched transitions.

The three readout deltas are the surviving subset of the original eight-change
`v6_3m_yolo_rv2` "YOLO" bundle. v6.1 reverts the full-res stage rebalance
(keeps the baseline's `blocks_per_stage=2,2,3,3`), the per-cell aux decoder,
`vf_coef=0.5`, and D3 masking. Rationale for each original change:
`V8_V6_YOLO_RATIONALE_20260811.md`.

## 2. Result

Promotion panel, exact completions out of 720 episodes, deterministic policy,
horizon 450, seed 20260807. **Both arms** evaluated with the same
code-matched, chunk-120 eval path (see §4).

| update | control (exact/720) | v6.1 (exact/720) | Δ |
|---|---|---|---|
| 2,000 | 0 | 8 | +8 |
| 4,000 | 1 | 20 | +19 |
| 6,000 | 1 | 139 | +138 |
| 8,000 | 44 | 181 | +137 |
| 9,000 | 109 | 230 | +121 |
| 10,000 | 174 | 250 | +76 |
| 12,000 | 194 | 288 | +94 |
| 13,000 | 234 | 334 | +100 |
| 14,000 | 281 | 407 | **+126 (+17.5pp)** |

- v6.1 is ahead at **13 of 14** evaluated checkpoints.
- The treatment's terminal step 334 → 407 is its largest late gain, so 14,000
  updates is not a plateau for either arm; the pair is truncated, not
  converged. This is the direct motivation for continuing v6.1 to 40,000.
- The early separation is large (u6k: 139 vs 1). The control's first
  double-digit checkpoint is u8k; the treatment's is u2k.

## 3. Caveats — stated as constraints on what may be claimed

These are limits of this experiment, not hedges to be dropped in the paper.

1. **This is a bundled three-delta screen, not three ablations.** The
   pre-registered rule for the YOLO family applies unchanged: a WIN must be
   decomposed by single-treatment runs before any per-component claim. Nothing
   here attributes the +126 to the token mixer, to the flatten-reduce, or to
   the latent-query count individually. The only supported claim is about the
   bundle. Three decomposition runs (each reverting exactly one delta) are the
   prerequisite for a component-level sentence.
2. **The chunked eval is not bit-identical to the unchunked one.** The
   affected 1×1 conv returns a bit-identical output *sum* chunked at
   90/128/180 and unchunked, but full rollouts still differ: `bfloat16`
   near-ties in the policy logits flip the argmax on a small number of
   episodes, which changes those episodes' trajectories. Measured spread from
   the dedicated control job (`10530460`): **±7 episodes per checkpoint, ≈0.7%
   of the 720-episode panel, unbiased in sign**. That is an order of magnitude
   below the +126 margin and cannot account for it, but every number in §2 is
   a ±7 quantity, not an exact constant.
3. **The control column is a code-matched rerun.** The baseline was
   re-evaluated with the same chunked eval binary as the treatment so the two
   columns share one code path. Earlier baseline artifacts (unchunked, from
   the original phase-1 job) differ from the rerun by **≤6 episodes** per
   checkpoint — consistent with (2) — and the terminal value moves 281 either
   way. The table deliberately reports the rerun.
4. **One seed.** No variance estimate across seeds exists for either arm. The
   14-checkpoint trajectory (13/14 ahead, monotone-ish separation from u2k)
   is the only evidence that the terminal gap is not a single lucky
   checkpoint. A paper claim of effect size needs seed replication; a claim
   that the bundle is not harmful is adequately supported.

## 4. Eval-protocol dependency: the cuDNN defect (commit `546f7aa`)

The paper's eval numbers depend on a workaround, so it belongs in the record.

Every v6-family GPU eval died with `CUDNN_STATUS_EXECUTION_FAILED`
(`cuda_dnn.cc:7927`) on the flatten-reduce 1×1 conv, while the compact arm —
same ResNet trunk, same bf16 encoder — evaluated fine. Bisected on `rtx_4090`
with the real encoder module at the panel batch:

| configuration | result |
|---|---|
| v6.1, bf16, batch 720 | FAIL (reproduces production) |
| compact, bf16, batch 720 | PASS (control) |
| v6.1 with `flatten_reduce=None` | PASS |
| v6.1 with the token mixer off | PASS |
| v6.1 jitted | FAIL (not eager dispatch) |
| v6.1 with a float32 encoder | FAIL (not the bf16 trunk) |
| v6.1 at batch 360 / batch 180 | FAIL / PASS |

So the trigger needs **both** readout deltas present (mixer *and*
flatten-reduce) and a batch above the 180–360 boundary. Peak memory was 1.4
GiB of 17.6 GiB, so this is cuDNN algorithm selection for that conv shape, not
model capacity. Training never hit it: training never drives the encoder past
512/device through the mixer path, and the panel forward is the only
larger-batch consumer.

Fix: `EVAL_FORWARD_CHUNK = 120` in `eval_mcts.py` (`_apply_in_batch_chunks`),
chunking the policy forward over episodes. The model has no batch-mixing ops,
so each episode's output depends only on its own row; 720 % 120 == 0 keeps
every chunk the same shape. Any replication of §2 must use this eval path (or
a batch ≤180) — and must accept the ±7 spread of caveat (2).

## 5. Negative result recorded in the same epoch: rv2p1 (reward-v2.1 timing)

Pre-registered as a risk before launch, and confirmed. `rv2p1_scratch`
replaced reward-v2's implicit dwell rent — the
`w·(1−γ)·Φ` by-product of `β·D_bound` — with undiscounted shaping
(γ_shape = 1) plus an explicit step cost of 3.6 (0.0080/step), on the capped
`continuous_banded_v3` sampler.

- **Early:** conversions appear sooner, 6–16 exact at u1k–u5k, i.e. the timing
  change does buy earlier conversion as designed.
- **Terminal:** collapse to **26–32 / 720**, against **276** for the
  reference. The arm is not a slower winner; it loses outright.
- **Reading:** the discounted-shaping "rent" reward-v2 charged as a
  by-product was doing load-bearing work as anti-stall pressure. Replacing it
  with a flat per-step cost removes the pressure that scales with remaining
  potential, and the policy settles into profitable dwelling instead of
  finishing. The exact mechanism is inferred from the reward decomposition,
  not separately instrumented.
- **Disposition:** reward-v2 timing variant 0 remains the resume/replay
  contract. The checkpoint architecture guard treats
  `reward_v2_timing_variant` like `action_logit_masking`, so a resume cannot
  silently change which reward the returns were fitted against.

## 6. Provenance

- Arms: `scripts/euler_v8_v6_yolo_rv2/{submit.sh,run.sbatch}`,
  `scripts/run_v8_v6_yolo_rv2.sh` (treatment);
  `scripts/euler_v8_r2_reward_v2/`, `scripts/run_v8_r2_reward_v2.sh` (control).
- v6.1 run revision `9abf88eb`, runtime terra `3051054b`, seed 20260807,
  4×`rtx_4090`, `gpuhe.24h`.
- Eval: `eval_fixed_bank.py --accepted-panel promotion` (720 episodes) +
  `--capability-panel promotion`, selection by
  `scripts/euler_v8_r2_reward_v2/select_promotion.py`.
- Parameter counts are asserted in-job and in
  `tests/test_v8_v6_yolo_rv2_launcher.py`: 2,303,421 with the carry-work
  channel, 2,303,405 without.
- Continuation of this arm past 14,000 updates is a separate,
  non-matched run (larger batch): it is a capability run, not part of this
  matched pair, and must not be folded into the table above.
