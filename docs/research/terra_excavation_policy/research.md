# Terra Excavation Policy

Status: E1/E3 finals 2026-07-22 — E3 (kickstart medium) 3.05/0.142 is the best Terra policy to date (+8.7% over teacher), E1 confirms the PPO fixes; E4 from scratch gives a positive cross-attention signal; F16 attention hardening knobs are implemented but not launched. Previously: kickstart validated 2026-07-21 — medium student (E3) surpassed its
teacher's final performance at 30% of its training budget and holds a perfect
success-within-horizon rate; spatial-encoder family confirmed over Atari-CNN per-sample.

## Vision

A deployable solo-excavator policy in the Terra grid environment that completes full
excavation plans (foundations, dump zones, eventually relocations/trenches) reliably and in
few steps, with an architecture/training recipe that scales to bigger networks and harder
map distributions without retraining from scratch.

## Methodology

- **Encoder evolution, versioned for checkpoint compatibility** (canonical behavior-based
  names, aliases for history): `atari` → `resnet_global_pool` (PR #15) →
  `resnet_spatial_8x8` (flatten readout preserves spatial layout) → `resnet_spatial_8x8_se`
  (11 input channels: + remaining_dig, dump_deficit, coord x/y; squeeze-excitation) →
  `resnet_spatial_8x8_se_xattn` (agent-conditioned cross-attention readout: 64 tokens @8×8,
  agent query + 4 latents — content-based "find leftover cells / highlight piles" selection)
  → `resnet_spatial_8x8_se_sa_xattn` (v5: two pre-norm token self-attention blocks before
  the flatten+xattn readouts).
- **PPO fixes** (each behind a default-off flag): value clipping off, flat env×time
  minibatch shuffling, entropy schedule stretched to the full run, bf16 encoder compute
  (f32 params), critic-head width override.
- **Warm-start playbook — the central methodological result**: introduce architecture or
  task changes via `scripts/grow_checkpoint.py` (function-preserving depth growth via
  zero-init second conv; stage-aware block remap) + kickstart distillation
  (KL(teacher‖student) + value MSE, cosine-annealed, LR warmup, low entropy start).
  From-scratch training of heavier bundles measurably underperforms (E2 vs E3).
- **Evaluation contract**: primary `eval/success_within_horizon_rate` (bounded);
  `eval/positive_terminations` + `eval/rewards` as legacy comparators;
  successful-episode length for deployability (step efficiency).
- Multi-agent implementation pipeline: spec doc → Opus implementation agents → 8-angle
  review + adversarial verification → fix pass → Euler snapshot launches with sha-pinned
  sbatch, smoke gates, GPU-type guards.

## Results

| Run | Config | Final pos_term / reward | Note |
|---|---|---|---|
| nnsksyva | atari base, 10k updates | 2.02 / 0.095 | control; 113k steps/s |
| pqtmfmqy | spatial_8x8 base, 10k | 2.81 / 0.131 | +39% per-sample vs atari at 3.9× wall-clock; the E3 teacher |
| E2 nr032qs7 | se+bf16+critic512 from scratch, 20k | 2.33 / 0.108 | below teacher — from-scratch drag of the heavier bundle |
| E1 3buorfp3 | spatial_8x8 + algo fixes, 20k | 2.83 / 0.132 (final) | beats teacher on its own encoder — algo fixes confirmed net-positive |
| **E3 j0bs2fkl** | **medium se, grown init + kickstart from pqtmfmqy, 20k** | **3.05 / 0.142 (final), swhr 0.997, ep_len 55.2** | **best policy to date: +8.7% over teacher; also fastest episodes (55 vs E1 59 steps)** |
| E4 k8vnwp5u | se_xattn from scratch, 20k | 2.59 / 0.121 final | +11% over SE from scratch (E2) — cross-attention is a real architecture win, but still below warm-started runs |

Supporting measurements: bf16 encoder ≈ 2× fwd+bwd (43k vs 28.8k steps/s in production);
pmap data-parallel scaling ~95% (1-GPU probe 11.4k vs 10.8k/GPU on 4); update loop is ~86%
encoder backprop. Reward-side fix (timeout terminal 0.1×completion², graded) removed the
coast-at-80% attractor identified 2026-07-17.

## Open directions

0. **Multi-task foundations + trenches (E8)**: one policy digging both map families.
   Feasibility verified 2026-07-22: trench maps are 64×64 with identical value semantics
   and depth-1 targets — same reward stream, only target geometry differs (unlike the
   dumpzone family, where completion semantics change). E8 = E3 warm-start + light teacher
   anchor on a foundations→double→double_diagonal curriculum (mixed after graduation).
1. **E5 — harder maps via cross-task kickstart**: student on
   `foundations_rectangles_real_dumpzone` (2-level curriculum preset exists) initialized
   from and KL-anchored to the E3 checkpoint; no fresh teacher training needed. Then
   `relocations_harder`, `trenches`.
2. **Step efficiency / deployability**: success-within-horizon is saturated (1.000) at
   550-step horizons → tighten `max_steps_in_episode` (e.g. 300) as direct pressure on
   time-to-completion; track successful-episode length as a primary metric; optionally
   early-finish terminal bonus.
3. **Attention ablations (F16)**: E4′/E7 follow-ups with
   `--attention_compute_dtype float32`; v5 epsilon mixer screens with
   `--token_mixer_residual_init_scale 1e-3`/`1e-2`; checkpoint rollout probes via
   `scripts/analysis/ablate_attention_checkpoint.py --mode xattn|token_mixer`.
4. **E4′ — xattn kickstarted** from the E3 checkpoint (grow v3-medium → v4-medium).
5. Timeout truncation bootstrapping (mask-branch design port), exact invalid-action
   masking, Muon-on-trunk optimizer A/B (only sanctioned optimizer experiment — no RL
   evidence for Shampoo-class preconditioning at this scale).

## Artifacts

- Branch `agent/spatial-v3-improvements` (terra-baselines), spec
  `docs/IMPROVEMENTS_SPATIAL_V3_2026-07-20.md`, launch ledger
  `docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`.
- Euler snapshots `spatial-v3-3a21cd6`, `spatial-v3-d1765d7`, `spatial-v4-0cf7f4a`;
  W&B project `aless-weber-eth/mixed-agents`.
