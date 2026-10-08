# GRU adaptation to cutting space and precision edges

The parent is `gru_rules_ft_c057_s20260930_update_110000.pkl`. Its frozen
road-network result is 32/32 with the original rules and 0/32 with the new
2.5 m cutting-space rule. This experiment adapts the policy to the new rules;
the parent result does not establish that its actions are suitable teacher
targets on new bulk episodes.

The student loads the parent's network weights with a fresh optimizer and
training clock. A binary global precision-required boundary band is added to
the map encoder. Its input weights start at zero. All original architecture,
reward coefficients, machine dimensions, and the 450-step horizon are retained.
One seed, 20261006, is used.

Half of each device's lanes use precision boundaries for their entire
episodes. They sample a screened subset of broad training foundations. The
remaining lanes sample the full 20,480-slot training bank with bulk boundaries.
Both groups use the new 2.5 m continuous cutting-space requirement. Precision
requires a pull within 25 degrees of an admissible boundary tangent in a 0.6 m
band. Ramps are implicit; physical depth profiles and bucket volume are absent.

The frozen recurrent parent uses a separate recurrent carry and its saved
observation preprocessing on the current environment's observations. Episode
termination resets that carry. Its pre-action logits are cached with the
student rollout. Policy KL is allowed only on bulk episodes whose training map
slots pass the teacher qualification. It fades from 1 to 0 over the first
3,000 updates. Value imitation is zero. If no selected map qualifies, teacher
guidance is disabled; an empty qualification is never expanded to all bulk
maps. This path does not use offline demonstrations.

The initial qualification completed 0/32 episodes across eight training
conditions (two distinct source geometries). Consequently the initial
production recipe has an empty teacher whitelist and effective KL coefficient
zero. This is limited qualification evidence, not a claim that the frozen
teacher fails on every bulk geometry. The CUDA numerical smoke separately
forces the teacher branch on bulk lanes to test it; those weights are discarded.

The qualification selects training maps without looking at policy results and
requires exact completion from four fixed starts under the same new bulk
physics and observations. Passing these starts is limited evidence of teacher
competence, not a guarantee on all states visited by the adapting student.
Held-out geometries are not used to select teacher targets or precision slots.

The first CSCS allocation uses one node with four GH200 GPUs, 512 environments
per device, 32 rollout steps, two PPO epochs, and 32 minibatches. A 24-hour
segment has an absolute 50-billion-transition ceiling and saves every 100
updates. A subsequent segment must use native optimizer/clock resume. Bulk and
precision completion counts and success rates are reported separately.

Before production, focused CPU checks cover observation parity, recurrent
teacher resets, masked loss normalization, and mixed-mode resets. The CUDA
smoke must complete one update and reload finite model, optimizer, rollout,
and teacher diagnostics. Compilation alone is not a passed smoke. Native
precision completion witnesses and per-cell static coverage are recorded
separately: static coverage is only a necessary geometric condition.

Campaign scripts are in `scripts/pull_direction_training/`. Local evidence is
under `.artifacts/terra_pull_direction_20261006/cscs_training_20261006/` in the
Moleworks workspace. Scheduler IDs and observed startup state belong in the
experiment running/log ledgers once submitted.
