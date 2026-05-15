# Terra Baselines Scripts

Keep short-lived research helpers grouped by what they do:

- `validation/`: deterministic gates and checkpoint parity checks that should pass before launching or merging.
- `analysis/`: local checkpoint rollout benchmarks and invalid-action inspection tools.
- `euler/`: current Slurm launch recipes for repeatable Euler experiments.

Prefer adding a flag to an existing script over creating another near-duplicate script. New Slurm
files should be kept only when they document a repeatable experiment shape that differs materially
from the existing recipes. Retire one-off launchers after their job ids are captured in
`docs/EXPERIMENTS_LOG.md`.

Run analysis scripts directly from the repo root, for example:

```bash
python scripts/analysis/trace_invalid_actions.py <checkpoint.pkl> --config solo_excavator
```
