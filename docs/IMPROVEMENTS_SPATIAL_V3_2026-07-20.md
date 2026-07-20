# Terra spatial-encoder v3 + PPO improvements — implementation spec (2026-07-20)

Branch: `agent/spatial-v3-improvements` (worktree `terra-baselines_improvements`, based on main `cdab283`).

Context: matched 10k-update A/B on Euler 4×4090 showed `resnet_spatial_8x8` beats `atari`
per-sample (positive_terminations 2.81 vs 2.02, eval reward 0.131 vs 0.095) at 3.9× wall-clock
cost (28.8k vs 113k steps/s). Update-loop profiling: ~86% of time is PPO backprop through the
encoder, ~14% rollout. These features (a) improve the encoder, (b) fix known PPO issues,
(c) recover throughput, (d) enable warm-starting bigger networks by distillation.

Reference runs: spatial `pqtmfmqy`, atari control `nnsksyva`
(W&B `aless-weber-eth/mixed-agents`).

## Ground rules (all features)

- **Checkpoint compatibility is sacred.** Existing encoders (`atari`, `resnet_global_pool`,
  `resnet_spatial_8x8`) and default config values must produce byte-identical param trees and
  behavior. Every change is behind a new flag/encoder name that defaults to current behavior.
- Environments: CPU gates `/home/lorenzo/moleworks/.venv-terra-uv/bin/python` with
  `JAX_PLATFORMS=cpu` and
  `PYTHONPATH=/home/lorenzo/moleworks/terra:/home/lorenzo/moleworks/terra-baselines_improvements`.
  Local GPU smokes: `/home/lorenzo/moleworks/.venv-terra-gpu-uv/bin/python` (invoke the binary
  directly, do not trust `activate`; export the `LD_LIBRARY_PATH` nvidia lib dirs from
  site-packages as in existing scripts).
- Tests run as plain scripts: `python tests/test_models.py` etc. (unittest, no pytest installed).
- Terra env repo is used read-only. All derived observation channels are computed inside the
  model from existing obs — do not touch `/home/lorenzo/moleworks/terra`.
- Match the existing code style in each file. No drive-by refactors.

---

## F1 — `resnet_spatial_v3` encoder: derived channels + coordinates (utils/models.py)

New canonical encoder name `resnet_spatial_v3` (add to `MAP_ENCODER_ALIASES`, alias
`resnet_spatial_v3` → itself). Selected via `--map_encoder resnet_spatial_v3`.

Input assembly in `MapsNet.__call__` when the canonical encoder is `resnet_spatial_v3`
(compute from **raw** action/target maps BEFORE the normalize() call, then normalize
action/target exactly as the v2 path does):

- `remaining_dig = ((target_map < 0) & (action_map > target_map)).astype(float32)` —
  the set of cells that must still be dug. This is the quantity that must reach zero for
  task completion; feeding it explicitly removes the need for the net to compute a
  two-channel interaction.
- `dump_deficit = ((target_map > 0) & (action_map <= 0)).astype(float32)` — dump-zone
  cells not yet filled.
- `coord_x`, `coord_y`: meshgrid over the map grid, each normalized to [-1, 1]
  (linspace over height/width), broadcast to the batch. Gives the position-aware flatten
  readout position-dependent features in earlier layers too.

Channel order for v3 (11 channels): the existing 7 in current order, then
`remaining_dig, dump_deficit, coord_x, coord_y`.

The v3 topology reuses `Spatial8x8MapResNet` with the same stage/dense defaults and the same
`model_size` preset kwargs (`resnet_stage_channels`, `resnet_blocks_per_stage`,
`resnet_dense_layers`), plus F2 and F3 below.

## F2 — Squeeze-excitation in ResidualMapBlock (v3 only)

Add `use_se: bool = False` (and `se_reduction: int = 4`) to `ResidualMapBlock`. When enabled:
after the second LayerNorm of the conv path and BEFORE the residual add, apply a standard SE
gate: global average pool over H,W → Dense(features // se_reduction) → relu →
Dense(features) → sigmoid → multiply the conv-path tensor channel-wise.
`Spatial8x8MapResNet` gets `use_se: bool = False` and passes it to its blocks;
v3 constructs it with `use_se=True`. The v2 (`resnet_spatial_8x8`) path must keep
`use_se=False` so its param tree is unchanged.

## F3 — Encoder mixed precision (bf16 compute, f32 params)

New config/CLI: `--encoder_compute_dtype {float32,bfloat16}`, default `float32`
(config field `encoder_compute_dtype: str = "float32"`).

Plumb a `compute_dtype` through `MapsNet` to the selected encoder module. Inside
`Spatial8x8MapResNet` (both v2 and v3 usage) and its blocks, pass
`dtype=compute_dtype, param_dtype=jnp.float32` to every `nn.Conv`, `nn.Dense`, and
`nn.LayerNorm`. Cast the encoder input to `compute_dtype` at entry and cast the final output
back to `float32` before it is returned from `MapsNet`. All loss/logit math stays f32.
With default `float32` the param tree and numerics are identical to today (guard with a test).
`atari` and `resnet_global_pool` may ignore the flag (raise ValueError if a non-f32 dtype is
requested for them, to avoid silently un-accelerated runs).

Rationale: the full-res stages are memory-bandwidth-bound (TF32 measured as a no-op);
bf16 halves bytes moved. Expected ~1.7–2× on the dominant update cost.

## F4 — Critic-head width override

New CLI `--critic_hidden_dims "512,256"` (config field
`critic_hidden_dims: tuple | None = None`, parsed from comma-separated string). When set,
`get_model_ready` overrides `hidden_dim_v = tuple(parsed) + (1,)`. Default `None` keeps the
model_size preset exactly. Motivation: `vf_coef=2.0` means value gradients dominate the shared
trunk while the value head is tiny; widening the head is capacity where the May research doc
says it pays, at negligible FLOPs.

## F5 — Value-clipping toggle (train.py)

Config field `use_value_clip: bool = True` + CLI `--no_value_clip` (store_false). In
`ppo_update_networks`, when False use plain `value_loss = 0.5 * jnp.square(value - targets).mean()`
(no clipped-max). Default True preserves current behavior. Motivation: the on-policy ablation
literature (see docs in mask worktree) found PPO value clipping harmful; EV is 0.994 so the
critic does not need the "stabilizer".

## F6 — Flat env×time minibatch shuffling (train_mixed.py + train.py)

Config field `flat_minibatch_shuffle: bool = False` + CLI `--flat_minibatch_shuffle`.

Current behavior: permute envs only; each minibatch is [mb_envs, seq_len, ...] so samples from
the same 32-step fragment always share a minibatch (correlated gradients AND correlated
per-minibatch advantage normalization). The policy is feedforward — nothing needs sequence
structure after GAE.

When enabled, in `_update_epoch`: flatten `[seq, env, ...] → [seq*env, ...]` for
transitions/advantages/targets, permute over all `seq*env` samples, reshape to
`[num_minibatches, mb_size, ...]`. `ppo_update_networks` must handle the flat layout: gate its
internal reshapes on the config flag (flat arrays have no seq axis; the obs reshape
`(shape[0]*shape[1], ...)` must be skipped, and value/log_prob keep shape `[mb_size]`).
GAE is computed before flattening and is unaffected.

Test: with `flat_minibatch_shuffle=True`, one update step on random data produces finite losses
and the same *total sample set* per epoch (permutation invariance of the union). Also assert the
blocked path is untouched by default.

## F7 — Kickstart distillation (warm-starting bigger networks)

New CLI/config:

- `--teacher_checkpoint PATH` (default None = feature off; everything below inert when None)
- `--kickstart_kl_coef` (default 1.0) — initial λ
- `--kickstart_kl_anneal_updates` (default 1500) — cosine anneal λ → 0
- `--kickstart_value_coef` (default 0.5) — initial β
- `--kickstart_value_anneal_updates` (default 500) — cosine anneal β → 0
- `--kickstart_lr_warmup_updates` (default 100) — linear LR warmup from lr/3 to lr,
  applied ONLY when a teacher is set (optax schedule joined with the constant lr)

Semantics (the design is fixed; do not re-derive):

- Load the teacher pkl (`{"model": params, "train_config": ..., "env_config": ...}`),
  rebuild the teacher model with `get_model_ready(teacher_train_config, env)` — the teacher
  architecture comes from ITS OWN train_config, the student's from the current run config.
  Validate `num_prev_actions` match (same env interface); hard-fail otherwise.
- Teacher params are frozen (`jax.lax.stop_gradient` / no optimizer state), replicated to
  devices (closure capture in the pmapped update is acceptable at ~1M params).
- Student trains with PPO from update 0 on its OWN on-policy rollouts (kickstarting — no
  separate BC phase). Extra loss terms per minibatch, computed on the same obs the PPO loss
  uses:
  - `kl = mean( sum_a p_T(a|s) * (log p_T(a|s) - log p_S(a|s)) )`  (KL(teacher ‖ student),
    teacher logits stop-gradient)
  - `v_distill = mean( (V_S(s) - V_T(s))^2 )`
  - `total_loss += λ_t * kl + β_t * v_distill`
- λ_t, β_t: cosine from initial value to 0 over their anneal windows (clamp 0 after), computed
  on the host per update from the update index and passed into the pmapped update like
  `ent_coef_current` (broadcast scalar). Log both coefficients and both loss terms to W&B
  (`kickstart/kl`, `kickstart/value_mse`, `kickstart/kl_coef`, `kickstart/value_coef`).
- Entropy: when a teacher is set, the run should be launched with a LOW entropy schedule
  (e.g. 0.02 → 0.005); this is run configuration, not code — but add a printed warning if
  `teacher_checkpoint` is set and `ent_schedule_start > 0.05`.
- The teacher forward can run in f32 regardless of student encoder dtype.

## F8 — `scripts/grow_checkpoint.py` (function-preserving growth)

Standalone script: `python scripts/grow_checkpoint.py --src ckpt.pkl --map_encoder
resnet_spatial_v3 --model_size medium [--critic_hidden_dims ...] --out grown.pkl`.

- Builds the target param tree via `get_model_ready` with the target config (derived from the
  source checkpoint's train_config with the requested overrides applied).
- Tree-walk both param trees by path:
  - identical shape → copy;
  - larger target (conv kernels/dense/LN along channel dims) → copy the source into the
    leading slices, init the rest with the fresh init scaled by 0.1;
  - **new `ResidualMapBlock`s that constitute added depth: zero-init the SECOND conv kernel**
    of the block → the block computes `relu(0 + residual) = residual` exactly (residual is
    post-relu ≥ 0), i.e. added depth is function-preserving at step 0;
  - SE params and new derived-channel input slices: fresh init (scale 0.1 for the first-conv
    new input channels);
  - anything unmatched: fresh init, and print it.
- Print a per-leaf report (copied / sliced / zero-init / fresh) and the param counts.
- Output pkl mirrors the source checkpoint structure with `model` replaced and `train_config`
  updated with the overrides, so `--resume_from`/`--teacher_checkpoint` can consume it.
- Include a test (tests/test_grow_checkpoint.py) that grows a tiny base→medium v2→v3 param
  tree and asserts: exact copies where shapes match, zero second-conv kernels on added blocks,
  no shape errors, and (function preservation) the grown v2→v2-deepened model reproduces the
  source model's output on a random obs batch to ~1e-5 when only depth was added.

## Explicitly deferred (do NOT implement in this batch)

- Timeout truncation bootstrapping (final-obs preservation through reset + GAE reset-boundary
  stop). Design exists on the mask branch; it touches the terra env and deserves its own pass.
  The 0.1×completion² graded timeout terminal + staggered timeouts already mitigate.
- Exact invalid-action masking in PPO.
- Separate actor/critic trunks.

---

## Experiment plan (after review)

All on Euler, 4× RTX 4090 (`eu-g6` nodelist + GPU-type hard guard per terra-rl skill),
`solo_excavator`, 1024 envs/device, num_steps 32, update_epochs 2, num_minibatches 32,
lr 3e-4, gamma 0.9984. 20,000 updates (~2.6B env steps). Entropy schedule
0.15 → 0.005 over 19,000 updates for E1/E2 (the 10k runs hit the floor while still improving).
W&B project `aless-weber-eth/mixed-agents`, tag `spatial-v3-batch-2026-07-20`.

- **E1 (algo-fixes control):** `--map_encoder resnet_spatial_8x8 --no_value_clip
  --flat_minibatch_shuffle`. Isolates the PPO fixes on the proven v2 encoder.
- **E2 (v3 encoder):** `--map_encoder resnet_spatial_v3 --encoder_compute_dtype bfloat16
  --no_value_clip --flat_minibatch_shuffle --critic_hidden_dims 512,256`.
- **E3 (kickstart medium):** E2 flags + `--model_size medium --teacher_checkpoint
  <pqtmfmqy final ckpt on Euler> --kickstart_kl_coef 1.0 --kickstart_kl_anneal_updates 1500
  --kickstart_value_coef 0.5` + low entropy schedule (`--ent_schedule_start 0.02
  --ent_schedule_end 0.005 --ent_schedule_steps 10000`). Optionally initialize the student
  from `grow_checkpoint.py` output via `--resume_from` if param loading accepts it; otherwise
  fresh init + kickstart is the supported path.

### Euler logistics (verified 2026-07-20)

- Workspace: `TERRA_WORK=/cluster/home/lterenzi/codex_terra_edge_validation`; venv
  `/cluster/scratch/lterenzi/codex_terra_edge_venv` (verified present). Code runs from
  snapshot dirs: `$TERRA_WORK/snapshots/<tag>/{terra,terra-baselines}` — create a new
  snapshot for this branch (e.g. `main-cdab283+spatial-v3`), rsync the worktree + terra main.
- Template sbatch (job name pattern, GPU guard, smoke-then-train structure):
  `$TERRA_WORK/scripts/terra_foundation_spatial8x8_base_4gpu_20260717.sbatch`.
- **E3 teacher checkpoint (pqtmfmqy final):**
  `$TERRA_WORK/snapshots/main-34b5d39d-095b261/terra-baselines/checkpoints/terra-foundation-spatial8x8-mb32-10k-euler-2026-07-19-07-19-27_FINAL.pkl`
- The local 4090 is occupied by a Newton workload (96% util) — skip local GPU smokes; use the
  Euler W&B-disabled smoke stage in the sbatch template instead.

Gate order per terra-rl skill: py_compile → unit tests (CPU) → Euler W&B-disabled full-shape smoke (update 1 completes)
→ production submit → verify allocation GPU types via sacct + nvidia-smi → record Slurm job
IDs, W&B ids, and ledger entries. Success metrics per skill: `eval/positive_terminations`,
`eval/rewards`, `eval/max_reward`, DO/DO_NOTHING rates, explained variance. Compare E1/E2/E3
against pqtmfmqy (2.81 / 0.131) and nnsksyva (2.02 / 0.095) at matched update counts.

## Acceptance gates (implementation)

1. `python -m py_compile` on every touched file.
2. All existing tests still pass unchanged: `tests/test_models.py`,
   `tests/test_training_utils.py`, `tests/test_eval_mcts.py` (CPU venv).
3. New tests cover: v3 channel assembly (11 channels, derived channels correct on a crafted
   map), SE param presence only in v3, bf16 output dtype f32 + finite grads, default-config
   param-tree identity vs main for all three existing encoders, value-clip-off loss path,
   flat-shuffle update step, kickstart loss terms + anneal schedule values, grow-checkpoint
   function preservation.
4. A 2-env, 3-update CPU training smoke of `train_mixed.py` runs for: (a) default flags,
   (b) `--map_encoder resnet_spatial_v3 --no_value_clip --flat_minibatch_shuffle`,
   (c) same + `--teacher_checkpoint` pointing at a checkpoint saved from (a).
   (Use DATASET_PATH/DATASET_SIZE as in tests; tiny env counts; JAX_PLATFORMS=cpu.)
