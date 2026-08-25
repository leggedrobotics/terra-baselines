# GRU recurrence probe v1

This diagnostic evaluates the historical twelve V8 failure-audit promotion
slots twice in one process. Each target retains its canonical position modulo
the 120-row policy-forward chunk, and both arms use the same 108 deterministic
padding slots. The first arm
carries GRU memory normally; the second zeros only the actor carry before every
decision. Both arms retain their ordinary current observation and five-action
history.

Run it in the same pinned Terra/JAX runtime used for fixed-panel evaluation:

```bash
python scripts/gru_recurrence_probe_v1/run_probe.py \
  --checkpoint /path/to/gru_checkpoint.pkl \
  --fixed-eval /path/to/canonical_gru_fixed_panel.json \
  --bank-root /path/to/accepted_v8_bank \
  --terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 \
  --output-dir /path/to/new_probe_output
```

The normal-carry target rows must reproduce the canonical full-panel result for
success, length, terminal material, and no-effect actions. The output is one
fail-closed `receipt.json` plus compact target-only trace arrays. The receipt
includes the 108 padding/control outcomes but does not use them as mechanism
evidence. This probe isolates evaluation-time memory use; it does not separate
recurrent training, parameter count, or sequence batching.
