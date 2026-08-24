# V8 GRU-64 slow learning: diagnosis (2026-08-17)

Question: why is `v8_relay_gru64_e5e1c3c50da9_s20260817` (job 10949597) so far
behind `v8_relay_partial_2778766683_s20260815` at matched updates — suspected
recurrent-PPO formulation or PPO-parameter bug?

## Answer

No recurrent-PPO bug found. The formulation is sequence-correct and its
telemetry is healthy. The run is not uniformly "slow": it matches the
feed-forward relay to ~u800 and then **plateaus structurally**. The mechanism
is the actor head, not PPO: all current-observation information is forced
through a GRU(64) whose gates never left their mushy initialization (update
gate z ~= 0.5 after 3,500 updates), so the logits see a tanh-bounded 64-d
blend of ~55% current input and ~45% history EMA. That attenuation caps
exactly the sharp state-conditioned discriminations the task needs next
(valid-action selection, dump placement), and the FF baseline's breakout
never happens.

## Formulation audit (clean)

- `done`/carry alignment is consistent everywhere: `transition.done = timestep.done`
  stored with `obs_t`; rollout zeroes the carry after a done step
  (train_mixed.py:3396); replay resets after producing step t's distribution
  (ResettableGRUCell, models.py); GAE uses the same convention. No off-by-one.
- `rollout_actor_h0` is captured pre-rollout with stop_gradient, permuted with
  the same permutation as the sequences, reshaped consistently; carry persists
  across rollouts and updates; epoch-1/minibatch-1 replay is bit-consistent
  with rollout (unit test `test_gru_step_matches_sequence_with_terminal_reset`).
- Loss shapes [B=16, T=32] all aligned; critic bootstrap via `method="value"`
  does not advance actor memory; `obs_to_model_input` is rank-agnostic.

## Evidence

W&B matched-update medians (GRU vs FF):

| u2901–3101 | GRU | FF |
|---|---|---|
| ppo/approx_kl | 0.0035 | 0.0036 |
| ppo/clip_fraction | 0.037 | 0.035 |
| ppo/entropy | 1.96 | 1.98 |
| ppo/explained_variance | 0.94 | 0.99 |
| absolute_completion | 0.296 | 0.532 |
| episode_success | 0.012 | 0.349 |

PPO numerics are indistinguishable — the policy moves the same KL distance per
update in both runs. It churns without improving.

Curve shape (absolute_completion, 400-update windows): GRU 0.235 → 0.304 (u800)
→ flat 0.27–0.31 through u4000. FF 0.329 → flat ~0.33 to u1200 → breaks out
exactly when exact successes appear (2% → 5% → 14% → 46%) and compounds. The
GRU never enters that success-bootstrap regime (98.8% timeouts throughout).

Where it stalls (u400-800 → u3600-4000):

- no_effect_action_rate stuck 0.457 → 0.432 (FF: 0.391 → 0.365) while
  do-fraction *rises* 0.139 → 0.161 — it digs more with less effect.
- dig_completion creeps 0.613 → 0.673, but dump_volume_completion is flat
  0.425 → 0.450 and dump_purity *degrades* 0.780 → 0.691; workspace cycles
  per episode inflate 18.3 → 26.0 (FF: drop to 15.5). The precision phase
  (placement, conversion) never sharpens.

Checkpoint probe (u500 vs u3500 weights, numpy replay of the trained cell,
scratchpad `gru_probe2.py`):

- Update gate at origin: z = 0.500 → 0.511; under synthetic drive z ≈ 0.38–0.49.
  After 3,500 updates the cell still mixes ~half old memory into every step —
  it neither learned pass-through (z→0) nor gating (z→1).
- Dynamics are contractive and stationary: |h| settles within ~50 steps
  (0.5–0.75 under drive), saturation ≤ 0.5 at strongest drive, input
  sensitivity dh/dx constant from t=5 to t=445. **No hidden-state drift or
  saturation over the 450-step episodes** — that failure mode is excluded.
- Kernels grow modestly (input 0.09→0.15 rms) — gradient reaches the cell;
  it just never escapes the z≈0.5 basin.

Secondary (not the driver, adds gradient noise): sequence minibatches contain
only 16 sequences/device (64 global independent units vs ~2048 for FF flat
shuffle) and advantages are normalized per device-minibatch over those
correlated samples.

Throughput correction: per-GPU the GRU stack is *faster* (16.1k/4 = 4.0k
steps/s/GPU vs 27.2k/8 = 3.4k) — the wall-clock gap is purely the 4-vs-8 GPU
allocation, not the scan.

## Confounds

Pre-registered in the contract: no stateless control, so recurrence,
sequence-minibatching, device layout, and seed are jointly confounded. The
checkpoint probe and the no_effect/purity signature point at the head, but
attribution between "64-d tanh bottleneck" and "16-seq gradient noise" is not
closed.

## Ranked fixes (v2 head)

1. Pass-through skip: `logits = actor_post_gru(concat(gru_output, actor_input))`
   so current-obs features reach the logits unattenuated and the GRU only adds
   memory. Directly targets the stuck no_effect/purity discrimination. This is
   the same lesson as the WP hybrid review (residual GRU).
2. Init update-gate bias toward pass-through (biz ≈ −1…−2, z≈0.2–0.3) so early
   training starts near the FF policy class instead of having to escape the
   50/50 blend.
3. More sequences per gradient step (num_minibatches 8 → 64 seqs/device/mb)
   to cut sequence-gradient noise ~2×; requires relaxing the
   num_steps == num_minibatches guard and accepting a 4× encoder replay batch.
4. If attribution matters for the paper: one stateless control (carry zeroed
   every step, everything else identical) separates optimization-machinery
   cost from recurrence cost.

Waiting to u10k without a head change is unlikely to change the verdict: the
plateau has been flat for 3,000+ updates and the gates show no trajectory out
of the init basin. The pre-registered fixed-panel readout at the next
checkpoint is still worth taking as the formal record.

Artifacts: probe scripts + W&B pulls in the session scratchpad
(`pull_wandb.py`, `analyze.py`, `curve_shape.py`, `gru_probe2.py`);
checkpoints u500/u3500 pulled from
`$SCRATCH/codex_terra_edge_runs/terra_v8_relay_gru_v1/runs/e5e1c3c…/s20260817/checkpoints/`.
