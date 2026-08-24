# Trench fixed-bank benchmark and trajectory gallery

This workflow benchmarks a strict-alignment trench checkpoint and turns a
bounded, deterministic subset of the same evaluation into a portable browser
gallery. The fixed-bank JSON is the quantitative source of truth; GIFs are for
failure analysis, not for scoring or checkpoint selection.

## Evaluation contract

- Bank: the finite-metadata-enriched V8 R2 accepted bank.
- Panel: `evaluation/gate_main/development`.
- Enumeration: all 608 manifest slots in manifest order.
- Policy: deterministic argmax.
- Seed: `20260724`.
- Horizon: 450 steps.
- Primary trench endpoint: the 176 trench episodes outside `trn-net4-*`.
- `trn-net4-*`: visible in the report, but labeled as a structural diagnostic
  and excluded from the primary endpoint.
- Integrity: use `eval_fixed_bank.py` and require its mutation, non-finite, and
  termination checks to pass.

Do not render a selected map in a smaller batch. Terra step RNG depends on the
batch contract. `scripts/trench_benchmark_gallery.py` therefore replays the
full 608-slot panel and extracts only the selected trajectories. Before writing
each GIF, it requires the selected slot's success flag, episode length, and
final dig fraction to reproduce the fixed-bank receipt exactly.

## 1. Produce the fixed-bank receipt

Run from the baselines checkout that contains the checkpoint-compatible
evaluator. Replace the uppercase paths and revisions with immutable values.

```bash
PYTHONPATH=PATH_TO_TERRA:PATH_TO_BASELINES \
JAX_PLATFORMS=cuda,cpu \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHON -u eval_fixed_bank.py \
  --checkpoint CHECKPOINT.pkl \
  --bank-root BANK_ROOT \
  --panel-family gate_main \
  --accepted-panel development \
  --terra-revision BANK_PROTOCOL_TERRA_REVISION \
  --horizon 450 \
  --seed 20260724 \
  --output OUTPUT_DIR/fixed_gate_main_development.json
```

The Terra revision passed here is the bank's protocol pin, not necessarily the
runtime checkout HEAD or the Terra revision used during training.

## 2. Build the gallery

Use a new output directory for every checkpoint, for example
`preview_u68500` or `final_u100000`. Three representatives per condition makes
42 GIFs across the 14 trench conditions.

```bash
PYTHONPATH=PATH_TO_TERRA:PATH_TO_BASELINES \
JAX_PLATFORMS=cuda,cpu \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
SDL_VIDEODRIVER=dummy \
PYGAME_HIDE_SUPPORT_PROMPT=1 \
PYTHON -u scripts/trench_benchmark_gallery.py \
  --checkpoint CHECKPOINT.pkl \
  --benchmark-json OUTPUT_DIR/fixed_gate_main_development.json \
  --bank-root BANK_ROOT \
  --panel-family gate_main \
  --accepted-panel development \
  --terra-revision BANK_PROTOCOL_TERRA_REVISION \
  --horizon 450 \
  --seed 20260724 \
  --per-condition 3 \
  --frame-every 4 \
  --output-dir OUTPUT_DIR \
  --training-baselines-revision TRAINING_BASELINES_SHA \
  --training-terra-revision TRAINING_TERRA_SHA \
  --evaluation-baselines-revision EVALUATION_BASELINES_SHA \
  --evaluation-terra-revision EVALUATION_TERRA_SHA
```

The default representative rule is deterministic and outcome-balanced, in this
order where available:

1. median-duration success;
2. median-progress stall;
3. highest-progress stall;
4. fastest success;
5. lowest-progress stall;
6. a second success or deterministic coverage fill.

This is a visualization sample, not a statistical estimator. Use the full
per-map receipt and condition summary for quantitative conclusions.

## Outputs

- `index.html`: browser gallery with geometry, condition, and outcome filters.
- `gifs/*.gif`: animated trajectories.
- `posters/*.png`: terminal-state previews used before GIF playback.
- `summary.json`: benchmark summary, provenance, selected rows, and artifact
  hashes.
- `condition_summary.csv`: compact per-condition metrics.
- `selection.json`: the deterministic representative selection.
- `gallery_manifest.json`: path, size, and SHA-256 for every gallery file.

Open `index.html` directly in a browser. If a browser restricts local media,
serve only the completed output directory:

```bash
python3 -m http.server --directory OUTPUT_DIR 8765
```

Then open `http://127.0.0.1:8765/`. Stop the server when inspection is done.

## Terminal rerun

A running training allocation is not a finished checkpoint. Keep interim
results under `preview_uNNNNNN`. When the final continuation has a clean
completion receipt, download that exact checkpoint, verify its remote and local
SHA-256, rerun both commands into `final_uNNNNNN`, and compare the fixed-bank
receipts before promoting it.
