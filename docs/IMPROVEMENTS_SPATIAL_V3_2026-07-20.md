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

Canonical encoder name `resnet_spatial_8x8_se` (behavior-based per AGENTS.md), with
`resnet_spatial_v3` kept as a compatibility alias mapping to it in `MAP_ENCODER_ALIASES`.
Selected via `--map_encoder resnet_spatial_8x8_se` (the `resnet_spatial_v3` spelling is
still accepted).

Input assembly in `MapsNet.__call__` when the canonical encoder is `resnet_spatial_8x8_se`
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
IDs, W&B ids, and ledger entries. **Primary success metric: `eval/success_within_horizon_rate`**
(the bounded evaluation metric per the AGENTS.md Training Metric Contract — fraction of initial
reset episodes that succeed within the fixed eval step budget). Secondary/diagnostic:
`eval/rewards`, `eval/max_reward`, DO/DO_NOTHING rates, explained variance.
`eval/positive_terminations` is retained as a **legacy** secondary metric only for
comparability with the reference runs (per-env episodes, can exceed 1). Compare E1/E2/E3
against pqtmfmqy (positive_terminations 2.81 / reward 0.131) and nnsksyva (2.02 / 0.095) at
matched update counts.

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

---

# Addendum 2026-07-20b — F13: cross-attention readout (encoder v4)

Motivation: the flatten readout has position-rigid weights — no mechanism to *dynamically*
select task-relevant cells (leftover dig cells, piles near dump zones) independent of where
they are. A small agent-conditioned cross-attention branch adds content-based selection
("attend where remaining_dig=1", "attend to piles when loaded"), the pattern that carries
AlphaStar-style agents. Conv trunk stays (Lux evidence: CNN trunks win; attention only as
recalibration/readout).

## F13 — `resnet_spatial_8x8_se_xattn` (canonical, behavior-based; alias `resnet_spatial_v4`)

Identical to `resnet_spatial_8x8_se` (11-channel input, SE trunk, same stage presets) plus a
cross-attention readout branch. All decisions below are final:

- **Tokens**: the final 8×8×C feature map (pre-flatten) → 64 tokens of dim C. Add a learned
  positional embedding table (64×C, zeros-init) to the tokens (coord channels already carry
  position in content; the table is cheap insurance). Pre-LayerNorm on tokens.
- **Queries (5 total)**: one projected from the ACTIVE agent's embedding (the
  `AgentStateNet` output for agent index 0 — obs are ordered acting-agent-first), plus 4
  learned latent query vectors. LayerNorm on queries.
- **Attention**: `nn.MultiHeadDotProductAttention`, num_heads=4, qkv_features=`attn_qkv`,
  out_features=`attn_qkv` per query; flatten the 5 query outputs and project with a Dense to
  `attn_out` features, relu.
- **Fusion (encoder output dim unchanged)**: `out = Dense(dense_layers[-1])( concat(
  relu(Dense(dense_layers[0])(flatten)), attn_branch ) )` — the flatten path stays exactly as
  in v3 (pure-addition change; low regression risk).
- **Sizes**: base `attn_qkv=64, attn_out=128`; medium `96/160`; large `128/192` (add to the
  model_size preset dicts as `resnet_attn_qkv`, `resnet_attn_out`).
- **Interface**: `MapsNet.__call__(obs, agent_embedding=None)` — the parent net passes the
  active agent's `AgentStateNet` embedding (`x_agents[:, 0, :]`, computed before the maps
  branch) ONLY for the xattn encoder; default None keeps every existing encoder path
  byte-identical (param trees and numerics of atari / global_pool / spatial_8x8 / _se must
  not change). Hard-fail (ValueError) if the xattn encoder is selected and agent_embedding
  is None.
- **bf16**: attention runs in `encoder_compute_dtype` like the trunk (dtype=compute,
  param_dtype=f32); branch output cast back to f32 with the rest.
- **Grow interplay**: growing v3→v4 copies the trunk via the stage-aware remap and
  fresh-inits the attention leaves (documented, not function-preserving; kickstart covers).
  No grow-script changes required — new leaves fall into the existing "fresh" category.

Tests (tests/test_models.py additions): (1) v3 param tree unchanged by the new code paths;
(2) v4 init + forward shapes at base and medium (output dim = dense_layers[-1]);
(3) content-based-selection probe: identical maps except one remaining-dig cell at two
different positions → attended branch outputs differ (run with the flatten-path weights
zeroed or compare full outputs; assert max-abs diff > tolerance); (4) agent-conditioning
probe: same maps, loaded=0 vs loaded=max in the agent obs → v4 outputs differ;
(5) ValueError when xattn selected without agent_embedding; (6) bf16 v4: f32 output dtype,
finite grads; (7) model_size medium changes attn param shapes.

## E4 (run after review; own snapshot)

E2's exact flags with the v4 encoder: `--map_encoder resnet_spatial_8x8_se_xattn
--encoder_compute_dtype bfloat16 --critic_hidden_dims 512,256 --no_value_clip
--flat_minibatch_shuffle`, ent 0.15→0.005 over 19000, 20k updates, 4×4090. E4 vs E2
isolates the readout. Snapshot tag `spatial-v4-<sha>`; E1/E2/E3 keep reading the v3 snapshot.

---

# Addendum 2026-07-22 — F14: token self-attention mixer (encoder v5)

Motivation: the cross-attention readout SELECTS task-relevant cells but tokens never
interact — it cannot compute relations BETWEEN map regions (pile↔dump-zone matching,
routing across work sites). One or two self-attention blocks over the 64 tokens add that,
and an identity initialization makes the upgrade exactly function-preserving, so it
composes with the warm-start playbook (grow from E3/E4-line checkpoints + kickstart).

## F14 — `resnet_spatial_8x8_se_sa_xattn` (canonical; alias `resnet_spatial_v5`)

Identical to v4 (`resnet_spatial_8x8_se_xattn`) plus a token-mixing stage. Decisions final:

- After the conv trunk produces the 8×8×C grid: view as 64 tokens, add the learned
  positional table ONCE here (the cross-attention readout then reuses the mixed tokens
  WITHOUT re-adding position; refactor the v4 pos-table location accordingly — v4's own
  param tree must remain unchanged when v5 is not selected).
- **2 pre-norm residual self-attention blocks** over the tokens:
  `x = x + Attn(LN(x))` then `x = x + MLP(LN(x))`; Attn = MultiHeadDotProductAttention,
  4 heads, qkv_features = resnet_attn_qkv, out_features = C with the output projection
  kernel **zeros-initialized**; MLP = Dense(2C) → gelu → Dense(C) with the second Dense
  kernel **zeros-initialized**. Both residual contributions are exactly zero at init →
  the whole stage is the identity function at init (function-preserving growth from
  v3/v4 checkpoints; fresh init in grow_checkpoint automatically inherits the zero init).
- Mixed tokens reshape back to 8×8×C and feed BOTH readouts (flatten and cross-attention),
  which are byte-identical to v4's.
- bf16: blocks run in encoder compute dtype, f32 params, as everywhere.
- Presets: 2 blocks at every model_size; qkv follows the existing resnet_attn_qkv values.

Tests (extend tests/test_models.py): (1) identity at init — a freshly initialized v5
forward equals the v4 forward computed from the SAME shared parameter values (construct by
copying the overlapping subtree), max abs diff 0; (2) after perturbing one block's output
projection, moving a remaining-dig cell at one map corner changes the attended output for
queries whose attention lands elsewhere (token mixing actually propagates); (3) param
counts base/medium; (4) bf16 finite fwd+bwd, f32 output; (5) v3/v4 param trees unchanged.

## E7 (run after review): grow E3 final → v5 medium (`grow_checkpoint.py --map_encoder
resnet_spatial_8x8_se_sa_xattn --model_size medium`), kickstart from E3 raw final,
E4′ flags otherwise. E7 vs E4′ isolates the self-attention increment at the ceiling.

---

# Addendum 2026-07-22b — F15: 128×128 resolution scaling (E8 pilot)

Goal: same physical game at half the tile size (equivalence, not difficulty). Deployability
motivation: finer grid ≈ real elevation-map resolution. Design rules are FINAL; the
implementing agent enumerates the concrete fields and reports the resulting table.

## Equivalence rules (env side — reachable via env_config._replace from terra-baselines;
terra repo stays read-only unless a field is provably hardcoded, in which case STOP and report)

- ×2 every tile-denominated agent geometry: agent.width, agent.height, move_tiles, dig
  workspace/cone radii — one action covers the same METERS.
- ×4 every tile-count volume/capacity quantity: truck_capacity, skidsteer_capacity,
  loaded_max (config), workspace capacity constants if any.
- dig_depth stays 1 (depth resolution unchanged; only lateral resolution doubles).
- Rewards: per-tile dig/dump rewards now fire on ~4× tiles per action → scale the reward
  normalizer ×4 (70 → 280) as the single-knob compensation; add config override
  `--reward_normalizer` to make this settable without terra edits (env_config.rewards is a
  NamedTuple — _replace(normalizer=...) from the baselines side).
- max_steps_in_episode stays 450 (equivalence test: episode step counts should match 64² runs).
- angles_cabin/base, local-map layout: untouched (resolution-independent).
- New preset `solo_excavator_128` (foundations_128 maps, max_steps 450) + config plumbing
  for the agent-geometry overrides (fields on MixedAgentTrainConfig applied via _replace,
  validated: print the full scaled table at startup).

## Dataset (separate agent)

- Find the generator that produced the `foundations` family (terra/terra/env_generation/),
  produce `foundations_128`: 600 train maps at 128×128 with all PHYSICAL parameters scaled
  ×2 in tiles (foundation extents, margins, borders) so the map content is the same sites
  at finer resolution; include every sidecar the loader needs (images/occupancy/dumpability/
  metadata/distance as in the 64 dataset). Verify: img_*.npy shapes (128,128); visual
  spot-check saved as PNGs; loader smoke (init_maps_buffer DATASET_PATH=... DATASET_SIZE=5).
- Local output: /home/lorenzo/moleworks/terra_data/train_128/foundations_128. Upload to
  Euler at /cluster/scratch/lterenzi/codex_terra_edge_runs/datasets/train_128/ (scratch =
  regenerable artifact per storage contract; note ~15-day purge).

## Model + trainer (code agent)

- CLI overrides `--resnet_stage_channels "16,32,48,64,64"` and `--resnet_blocks_per_stage
  "1,1,2,2,2"` (comma lists → existing config fields; model_size presets remain defaults
  when unset). The 5th stride-2 stage keeps the readout at 8×8 for 128 inputs; SAME channel
  count as stage 4 → flatten/readout/head shapes unchanged → those params copy exactly on
  grow. Add both fields to _validate_checkpoint_architecture (None-aware).
- grow_checkpoint: (a) support the added STAGE (stage-aware remap already maps by (stage,
  block); a target stage beyond the source's stage count = all-added → fresh init; its
  stride-2 entry cannot be identity — document); (b) positional-table interpolation: when
  token counts differ (64→256), bilinearly interpolate the source table on its 2D grid
  (8×8→16×16... NOTE: with the 5th stage the grid stays 8×8/64 tokens — interpolation only
  needed if stages are NOT extended; implement it anyway, gated on shape mismatch).
- Teacher obs-downsampling for cross-resolution kickstart: `--teacher_obs_downsample N`
  (default 1 = off). When 2: in the teacher forward path ONLY, transform the obs — global
  map channels [12..15,18..20] stride-2 nearest subsample (in-distribution discrete values,
  no fractional mask values), agent pos_x/pos_y integer-divided by 2 in agent_states,
  loaded dirt divided by 4, agent_width/height halved. Local maps are resolution-independent
  1D workspace summaries, but must be model-side scaled by `--local_map_area_scale 4` for
  128² equivalence. Teacher model is built from ITS train_config (64² world) as today.
  Hard-fail if downsample != 1 while teacher maps already match student size.
- Tests: stage-override parsing + validation; grow 4→5-stage exact-copy of flatten/readout/
  heads (max abs diff 0 on those subtrees); pos-table interpolation shape + corner-value
  sanity; teacher-downsample transform unit test (crafted 4×4→2×2 maps, pos halving);
  full-policy init at 128 input with 5 stages (obs built at 128 edge length).

## E8 pilot (after review + dataset ready)

Grow E3 final → 5-stage medium (encoder per best available arm: v3-se now; re-grow from
E4'/E7 winner later), kickstart with `--teacher_obs_downsample 2`, teacher = E3 raw final.
4×4090, **512 envs/device**, num_steps 32, num_minibatches 32, bf16, critic 512,256,
algo fixes, ent 0.02→0.005 over 10k, KL anneal 1000, **10k updates**, `--config
solo_excavator_128`, `--reward_normalizer 280`. MANDATORY pre-launch memory smoke (1 update,
W&B off) before production; expect ~11-13k steps/s and validate ep_len ≈ 55 (equivalence).

## Addendum 2026-07-23 — E9 128² numerical hardening

E9c showed why the local gate must be the exact per-GPU production shape: a weak smoke
allowed an all-NaN first update into production. Required hardening now implemented:

- Local-map workspace sums use `IntMap` instead of `IntLowDim`; 128-equivalent local sums
  can exceed `int8` range.
- `AgentStateNet` clips discrete indices before `nn.Embed`; Flax returns NaNs for high
  out-of-range indices, and exact-edge positions appear in 128²/teacher-downsample states.
- Smoke/production use `--fail_on_nonfinite` with named rollout, GAE, logits, entropy,
  teacher, gradient, parameter, and optimizer-state diagnostics.
- E9d was only submitted after an exact 1-GPU, 512-env/GPU, mb64, bf16, one-update local
  gate saved a finite checkpoint.

---

# Addendum 2026-07-22c — F16: attention hardening + probe utilities

Motivation: E4 from scratch eventually beat the SE-only control, but the first attention
arms ran with the same `encoder_compute_dtype=bfloat16` bundle that hurt early learning in
E2. E7 also tested an exactly identity-initialized token mixer: good for function-preserving
growth, but at step 0 it blocks gradients into the MHA/MLP inner params until the residual
output projections move away from zero. F16 adds gated knobs to test those hypotheses without
changing existing E4'/E7 semantics.

## F16a — f32 attention inside bf16 encoders

New config/CLI:

- `--attention_compute_dtype {encoder,float32,bfloat16}`, default `encoder`.

Semantics:

- `encoder` preserves the current behavior exactly: v4/v5 attention submodules use
  `encoder_compute_dtype`.
- `float32` runs the v4/v5 cross-attention readout and v5 token mixer in f32 while the
  spatial-ResNet conv trunk can still use `--encoder_compute_dtype bfloat16`.
- Params remain float32; only submodule compute dtype changes. The param tree and shapes are
  unchanged, so checkpoint loading/resume compatibility is preserved.
- Non-default values hard-fail unless `map_encoder` is v4/v5 (`resnet_spatial_8x8_se_xattn`
  or `resnet_spatial_8x8_se_sa_xattn`) to avoid silent no-op experiment flags.

Target ablation:

- Run the E4' recipe with `--encoder_compute_dtype bfloat16 --attention_compute_dtype float32`.
  If this closes the gap to E6/E3 while retaining bf16 trunk throughput, attention-softmax
  precision was the likely drag. If not, the issue is not just attention numerics.

## F16b — epsilon-init v5 mixer residual projections

New config/CLI:

- `--token_mixer_residual_init_scale FLOAT`, default `0.0`.

Semantics:

- `0.0` preserves the F14 contract: the v5 mixer is exactly identity at init, existing param
  counts and leaves stay unchanged, and grown checkpoints remain function-preserving.
- A small positive value (candidate `1e-3` to `1e-2`) replaces the zero initializers on the
  token-mixer attention output projection and MLP second projection with
  `scale * lecun_normal`. The mixer is no longer exactly identity, but the perturbation is
  small and gradients reach the inner MHA/MLP params immediately.
- Nonzero values hard-fail unless the encoder is v5, again to avoid silent no-ops.

Target ablation:

- E7b: v5 kickstart from E3/E4' checkpoint, `--attention_compute_dtype float32`, and
  `--token_mixer_residual_init_scale 0.001` (or a 1-GPU screen at `0.001` vs `0.01`). Compare
  against E4' and E7 at matched updates and final eval.

## F16c — checkpoint attention ablation helper

New script:

```bash
python scripts/analysis/ablate_attention_checkpoint.py ckpt.pkl xattn_off.pkl --mode xattn
python scripts/analysis/ablate_attention_checkpoint.py ckpt.pkl mixer_off.pkl --mode token_mixer
python scripts/analysis/ablate_attention_checkpoint.py ckpt.pkl dry.pkl --mode all_attention --dry_run
```

Modes:

- `xattn`: zeros the direct cross-attention branch leaves (`attn_pos_embed`,
  `attn_latent_queries`, direct xattn MHA, agent-query Dense, branch-projection Dense) while
  leaving the final fusion Dense intact.
- `token_mixer`: zeros v5 token-mixer block leaves, which makes the trained mixer blocks act
  as identity while preserving the shared xattn readout and positional table.
- `all_attention`: both modes.
- `--pattern SUBSTRING` can target exact `jax.tree_util.keystr` paths after a `--dry_run`.

Use this only for evaluation/rollout probes; it is not a training initialization path.

## F16 launch discipline

Any new attention variant must run a quiet local first-update smoke before Slurm submission:
W&B disabled, tiny env count, same model/checkpoint/teacher flags, and evidence that update 1
completed and checkpoint save/reload works. E9's teacher-env failure is the concrete reason:
shape-compatible model init is not a sufficient gate for trainer changes.
