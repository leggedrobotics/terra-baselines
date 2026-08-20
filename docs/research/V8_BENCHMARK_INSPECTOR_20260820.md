# Terra V8 benchmark inspector

Date: 2026-08-20

## Question

The concat-skip GRU64 has reached near-ceiling online full-start performance.
That establishes broad task learning, but it does not by itself establish that
recurrent state fixes Terra's historical no-op and limit-cycle failures. The
benchmark system therefore separates three claims:

1. **online learning health** remains in W&B;
2. **fixed-panel capability** comes from the deterministic 720-map evaluator;
3. **memory use** requires a same-process normal-carry versus zero-carry
   recurrence intervention on the historical failure cases.

The feed-forward and GRU training runs are a capability pilot. They differ in
more than recurrence, including seed, device count, parameter count,
sequence-minibatch training, and effective training runtime. Their comparison
cannot be called a recurrence-only ablation.

## V1 readout

The fixed promotion-panel evaluation is complete. All four checkpoints were
evaluated on one RTX 4090 under the same deterministic 720-map contract:

| checkpoint | exact | macro terminal completion | worst-condition completion | successful mean steps |
| --- | ---: | ---: | ---: | ---: |
| GRU u40k | 677/720 | 0.9591 | 0.4656 | 89.7 |
| GRU u44k | 674/720 | 0.9584 | 0.6190 | 86.4 |
| feed-forward u44k | 652/720 | 0.9424 | 0.4351 | 87.9 |
| feed-forward u86k | 670/720 | 0.9541 | 0.4237 | 83.6 |

At matched u44k, the recurrent-agent treatment gains 22 exact maps over the
feed-forward policy (674 versus 652). It also remains four maps ahead of the
feed-forward u86k frontier. This is strong sample-efficiency evidence for the
complete recurrent treatment, not a recurrence-only causal estimate.

The online curve hid meaningful held-out churn. GRU u40k to u44k converts 24
maps and regresses on 27, for a net loss of three exact maps, despite shorter
successful episodes and a much higher worst-condition completion score. The
latest online success rate must therefore not be used as checkpoint promotion
evidence by itself.

The same-process carry intervention establishes that the trained GRU uses its
memory: with identical current observations and five-action histories, normal
and zeroed carry produce different actions on 17 decisions, spanning all 12
target maps. Removing carry increases target no-effect actions from 24.3% to
50.9%. It does not change target exact completion, however: both arms solve
9/12. Slots 250, 247, and 177 remain failed observable input/action cycles.
Thus memory improves control robustness but is not yet shown to cause the
historical target conversions or to remove the last three traps.

The durable local evidence bundle is
`/home/lorenzo/moleworks/.artifacts/terra_v8_gru_benchmark_20260820`.

## V1 workflow

```text
four pinned checkpoints
        |
        v
canonical fixed 720-map evaluation
        |
        +--> aligned condition/map outcomes and CSV
        |
        +--> deterministic review-slot receipt
                  |
                  v
          full-panel parity replay
                  |
                  v
        compact traces + selected GIFs
                  |
                  v
          self-contained index.html
```

The first benchmark compares:

- GRU u40k and latest complete GRU u44k;
- feed-forward u44k at matched updates; and
- feed-forward u86k as the mature frontier.

Every evaluation uses the same RTX 4090 process, common Terra runtime, frozen
promotion manifest, greedy unmasked policy, horizon 450, seed 20260807, and
forward chunk 120. Architectures are evaluated in separate calls because the
fixed evaluator correctly requires a constant treatment fingerprint within a
checkpoint sequence.

The promotion panel contains 45 conditions x 16 maps = 720 episodes. It omits
two conditions in the current 47-condition training distribution:
`fnd-slab-allfree` and `trn-straight-allfree`. The dashboard states this
coverage explicitly rather than calling the panel complete V8 coverage.

## Inspector surface

`scripts/build_v8_benchmark_dashboard.py` consumes two canonical fixed-bank
records and produces:

- `index.html`, which opens directly with `file://`;
- `dashboard_data.json`, the aligned machine-readable comparison;
- `episodes.csv`, one flat row for each of the 720 maps; and
- `review_selection.json`, the deterministic media-selection receipt.

The HTML provides:

- exact totals, net change, conversions, and regressions;
- a clickable 45 x 16 outcome matrix;
- condition filters and per-condition exact deltas;
- a searchable and sortable 720-map table;
- material, no-effect, stall, and step diagnostics for each map; and
- side-by-side reference/candidate GIFs when rendered media exists.

Outcome colors are paired facts: both exact, conversion, regression, or both
fail. Labels such as `loaded endpoint`, `staged-soil residue`, `high
no-effect`, and `near-finish cleanup` are overlapping descriptive signals, not
root-cause claims. Off-zone positive material is always displayed as staged
soil, never as an illegal action.

## Media contract

`scripts/render_v8_fixed_panel_gifs.py` does not rerun a selected map in a
smaller inference batch. It runs the complete 720-map cohort with the canonical
chunking and recurrent carry, retaining frames only for the selected slots.
Before accepting media it must reproduce all 720 fixed-record outcomes,
episode lengths, terminal material partitions, and aggregate authoritative
no-effect action counts. The fixed record does not contain per-step action and
effect sequences, so this is not a per-step trace identity claim.

For each selected episode it records:

- action and physical effect;
- dig, terminal, staged, and loaded material fractions;
- whether material changed;
- exact processed policy-input hash;
- GRU hidden-state hash, L2 norm, and stepwise change;
- top-action probability and margin;
- longest no-effect and repeated-action streaks; and
- an exact terminal input-plus-action cycle, when present.

The review set is deterministic and prioritizes regressions, persistent
failures, conversions, and high-carry successes. High carry is only a coverage
signal, not proof that the policy performed an off-zone relay. The selection
receipt makes the qualitative subset reproducible and auditable; high-carry
anchors are condition-diverse before repeated conditions are admitted. Every
GIF uses the same uniform simulator-step cadence and freezes completed episodes
through the shared horizon, so reference and candidate clips have equal
duration. The dashboard still exposes all 720 scalar rows even when only a
bounded subset receives GIFs.

## Memory mechanism panel

The historical twelve promotion slots remain the mechanism-enriched panel.
The GRU probe must duplicate them within one rollout:

- twelve rows use normal recurrent carry;
- twelve identical resets zero the actor carry before every decision; and
- both arms keep the same five-action observation history.

The probe records instantaneous input hashes, hidden hashes/norms, logits,
actions, action effects, material traces, recurrence periods, and no-effect
streaks. A credible memory-use result requires all of the following:

1. normal carry converts historical cycles or no-op traps;
2. identical instantaneous inputs can yield different hidden/logit/action
   states; and
3. zeroing carry reintroduces failures or materially worsens escape behavior.

This isolates evaluation-time use of recurrent state in a shared process. It
does not remove the training-time capacity and sequence-minibatching confounds;
the paper must either describe the whole recurrent-agent treatment or add a
matched stateless/sequence-trained control.

## V2 after the first readout

Only after the V1 artifact is inspected:

- add a checkpoint-history matrix for acquisition and regression over time;
- render all maps whose exact outcome changes between milestones;
- add synchronized paired GIFs or a step scrubber;
- add source-disjoint 47-condition confirmatory panels; and
- optionally upload the immutable bundle as a W&B artifact while keeping the
  local static HTML as the primary qualitative review surface.

Do not build a database or web service for the current scale. One canonical
evaluator, one compact trace renderer, and one static dashboard keep the code
and the scientific authority clear.
