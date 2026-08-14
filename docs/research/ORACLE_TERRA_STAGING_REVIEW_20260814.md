## Overall assessment

The corrected abstraction is coherent. Off-terminal staging is an intended intermediate state, not a failed or illegal action. The strongest evidence is the successful `d16` replay: 5 off-zone dumps, 9 rehandles, exact completion, and carrier-ledger conservation of (H) to numerical tolerance (`V8_REWARD_TERMINATION_AUDIT.md:138–158`). That directly rejects the prior blanket diagnosis that relay handling is an environment defect.

The residual failures instead separate into:

1. **Directly observed policy-input attractors and missing transition-outcome information.**
2. **A rare, confirmed one-tile mechanics obstruction.**
3. **Several source-proven relay discontinuities whose actual failure incidence has not yet been measured.**

## 1. What reward-v2 actually penalizes

For the stated gamma-matched baseline, reward-v2 contains only exact success, generic horizon failure, uniform step cost, and potential shaping (`terra/state.py:4239–4279`). The legacy dense terms are replaced rather than added (`terra/state.py:3425–3487`).

### Not a direct staging penalty

Off-zone positive soil enters

[
H=
\text{remaining excavation work}
+\text{off-zone haul work}
+\text{carried haul credit},
]

using the static terminal-distance field (`terra/state.py:4134–4156`). This says that staged soil is **unfinished terminal work**. It does not classify the dump action as invalid or assign a fixed “wrong dump” penalty.

Consequently:

* A relift that correctly transfers staged-soil work into carrier credit should approximately conserve (H).
* An off-zone dump closer to the terminal decreases (H) and receives positive shaping.
* A same-distance transfer is approximately (H)-neutral.
* A farther staging move increases (H) and receives negative shaping.

The successful `d16` replay confirms that this accounting supports a real multi-stage relay rather than suppressing it (`V8_REWARD_TERMINATION_AUDIT.md:140–153`).

The completion code calls off-zone material `illegal_dump_volume`, but this is a misleading diagnostic name, not an action veto. It reduces dump purity and prevents exact completion while the material remains there, as it should; the transition itself is allowed (`terra/state.py:3563–3655`). Solo dumping explicitly selects off-zone cells when no accepted cells are reachable (`terra/state.py:2426–2444`).

### Terms that do charge necessary relay work

Necessary staging is still penalized in three general ways:

1. **Uniform action cost.** Every stage, move, relift, and redump pays (-1/450) in the baseline (`terra/config.py:30–38`; `terra/state.py:4264–4279`).

2. **Discounted-potential dwell cost.** When (\Phi_{t+1}=\Phi_t), shaping is
   [
   (\gamma-1)\Phi_t<0.
   ]
   Therefore flat-(H) empty navigation and (H)-conserving handling steps pay an additional implicit time cost because the potential is shifted positive (`terra/state.py:4170–4174,4259–4279`; `V8_REWARD_TERMINATION_AUDIT.md:557–586`).

3. **Non-monotone relay legs.** A necessary temporary move farther from the terminal, or removal of soil from an accepted region to restore access, increases (H) and is negatively shaped. This follows from the static point-distance definition, not from an off-zone label.

The generic horizon penalty and withheld exact-success bonus also penalize any relay that does not finish by step 450, but they correctly enforce the fixed task contract.

The old explicit penalty for digging soil from an accepted dump region exists only in the legacy dense reward (`terra/state.py:2837–2863`) and is inactive under reward-v2.

## 2–3. Ranked issues by evidence strength

### 1. Missing action-outcome observability — **verified and directly implicated**

Terra computes `action_had_effect` from terrain and physical-agent changes, but exposes it only through transition diagnostics, not the policy observation (`terra/env.py:188–198,201–263,583–618`). The actor sees five prior action IDs but not whether those actions moved the machine, changed material, or did nothing.

Every targeted greedy failure enters an exact full-policy-input fixed point or short cycle. The two obstacle cases repeat one identical input and choose no-effect `backward` hundreds of times; carry and near-finish failures contain fixed points or 2–18-step cycles (`V8_V61_FAILURE_AUDIT_20260813.md:312–368`). Three loaded endpoints had valid complete unload headings, and `DO` immediately improved completion, so those cases were policy-choice failures rather than dump deadlocks (`V8_V61_FAILURE_AUDIT_20260813.md:193–218`).

This is the highest-confidence current issue. It does not prove that one particular observation addition will train successfully, but it proves that the current feed-forward controller cannot react differently once its complete input repeats.

### 2. The `<=1` eligible-tile dig gate — **verified causal instance, low measured prevalence**

Every dig mask affecting zero or one eligible tile is erased (`terra/state.py:2147–2170`). Because this happens before distinguishing fresh excavation from positive-soil relift, it applies equally to staged soil.

One targeted near-finish state is directly blocked by this rule, and exactly 2/313 frozen failures end with one undug target cell (`V8_V61_FAILURE_AUDIT_20260813.md:185–192,225–234`). Thus it is real but cannot explain the main residual gap. Positive-soil singleton prevalence is not available from the frozen summary, so its staging-specific importance remains unknown.

### 3. Atomic positive relift and capacity rejection — **verified mechanic, plausible relay trap**

For positive soil, Terra removes the entire selected positive footprint (`terra/state.py:1509–1528,2173–2186`). The transition proceeds only if the complete selected volume fits the low-dimensional carrier capacity; otherwise the action is a complete no-op (`terra/state.py:2261–2271`).

Combined with the one-tile gate, this creates two discontinuities:

* a one-cell staged pile cannot be relifted even when its volume is small;
* a multi-cell selected pile cannot be partially relifted when its total volume exceeds capacity.

These are directly counterproductive to robust stage–relift semantics. However, no attached trace establishes that an over-capacity positive relift caused one of the selected failures. This is therefore a source-proven risk, not a measured aggregate cause.

### 4. Accepted-first dumping without fallback — **verified control-flow risk**

If any accepted cell is physically reachable, the code selects only the accepted subset before testing whether that subset can represent the complete load (`terra/state.py:2395–2449`). If complete-load containment or storage validation then fails, the dump becomes a no-op; it does not retry the reachable off-zone subset (`terra/state.py:2451–2515`).

Thus a legitimate staging dump can be suppressed merely because a small or unusable terminal fragment is also visible. The audit already identifies this exact limitation but states that occurrence in a real failed trajectory is unverified (`V8_REWARD_TERMINATION_AUDIT.md:397–404`). The selected loaded endpoints do not establish it: they had valid complete unload actions.

### 5. Traversability observation/physics mismatch — **verified mismatch, causal incidence unknown**

The observation wrapper marks **every nonzero soil cell** as blocked (`terra/wrappers.py:91–102`). Actual movement physics is less conservative: isolated height-one positive soil can be traversable, while holes, high piles, and sufficiently dense patches block movement (`terra/state.py:509–574,623–646`). The optional reachability channel is disabled (`terra/config.py:254–256`).

After staging, the actor can therefore see a route as blocked even when the transition physics permits it. This is especially relevant to repeated empty relocation between staged workspaces. The raw action map gives the network some opportunity to learn around the inconsistency, so it is misleading rather than complete information loss. No trace yet proves that this mismatch selected a particular failed route (`V8_V61_FAILURE_AUDIT_20260813.md:136–146`).

### 6. Hidden `last_dig_mask` — **verified hidden transition state, narrower risk**

`last_dig_mask` changes both dig eligibility and dump workspace exclusion (`terra/state.py:2009–2030,2104–2119`). It is updated after excavation or relift and cleared only on a successful dump (`terra/state.py:2208–2250,2480–2497`). It is absent from the observation dictionary (`terra/env.py:583–618`).

Therefore visually equivalent map/pose/load states can have different dig or dump outcomes depending on prior handling history. That is genuine hidden transition state. Its relay impact is narrower than the previous issues because the normal successful lift–dump cycle clears it. It is most suspect after a failed dump or when a policy revisits a workspace through an unusual sequence. No attached failure trace isolates it.

### 7. Static point-distance potential — **verified limitation, not a demonstrated defect**

The distance field values material cell locations but omits the excavator footprint, five-tile movement, heading, workspace geometry, and future serviceability (`terra/state.py:4134–4155`; `V8_V61_FAILURE_AUDIT_20260813.md:164–171`).

It can therefore:

* reward a pointwise-closer stage that blocks future access;
* penalize a necessary temporary move away from the terminal;
* provide no positive material signal during an empty navigation leg;
* assign zero remaining haul cost to accepted-zone soil even when its placement is operationally poor.

This is a plausible reason that correct relay plans require policy learning beyond immediate potential ascent. It is not evidence for broad reward replacement: the successful `d16` relay shows that the same potential can support repeated staging and rehandling.

### 8. Clipped global pile heights — **verified information degradation, specifically not the observed alias source**

The environment supplies the raw `action_map` (`terra/env.py:604–607`), but the v6.1 policy preprocessing clips the global map, so global pile height and capacity information are compressed (`V8_V61_FAILURE_AUDIT_20260813.md:70–77,130–134`). Local positive/negative workspace summaries retain signed aggregate quantities (`terra/wrappers.py:419–430,478–499`), so volume information is degraded rather than completely absent.

Most importantly, the exact recurrence replay found zero repeated policy-input hashes paired with different raw-map hashes. Clipping therefore did **not** cause the measured exact aliases in the targeted cycles (`V8_V61_FAILURE_AUDIT_20260813.md:314–356`). It remains a possible long-range capacity-planning weakness, but it ranks below the observed outcome/cycle issue.

## 4. Smallest diagnostic and correction

### Smallest diagnostic

Add a read-only reason code to exact replay; do not alter policy, reward, terminal semantics, or dynamics. Run it on the successful `d16` relay as a positive control and the existing loaded/staging failures.

For each attempted `DIG` or `DO`, record:

* positive eligible-tile count and selected positive volume;
* rejection by `<=1`, capacity, obstacle, or `last_dig_mask`;
* accepted and off-zone reachable counts;
* whether the accepted candidate can represent the complete load;
* whether the counterfactual off-zone candidate could represent it;
* observed traversability versus actual movement validity;
* physical effect, material effect, (\Delta H), and `stall_age`.

This single trace makes each candidate mechanism falsifiable. In particular, it distinguishes “policy ignored a valid relay action” from “the action silently no-oped because of a specific transition filter.”

### Smallest evidence-backed correction now

Expose the previous transition outcome using two booleans:

* `previous_action_had_physical_effect`;
* `previous_action_changed_material_or_load`.

These values already have close equivalents in `_transition_diagnostics` and `_next_stall_age_steps` (`terra/env.py:201–263`; `terra/state.py:300–332`). They distinguish:

* a blocked/no-op action;
* legitimate empty movement or rotation;
* a material-handling transition.

This is smaller and less semantically confounded than a large recurrent model. It directly targets the strongest observed issue, preserves unmasked action selection, and leaves the macro-action abstraction and exact terminal contract unchanged. It should not be claimed to solve effective pose cycles by itself.

The smallest **conditional mechanics patch**, only if the reason-coded replay fires it, is accepted-first two-pass dumping: validate a complete accepted-zone deposit first; only if that candidate fails, validate the reachable off-zone candidate. Commit at most one complete, mass-conserving dump. That preserves terminal preference while restoring legal staging (`terra/state.py:2426–2515`; `V8_REWARD_TERMINATION_AUDIT.md:397–404`).

Likewise, exemption of positive relifts from the `<=1` gate is justified only after positive-soil singleton incidence is measured. Partial-capacity relift is a larger semantic change and is not yet evidence-backed.

## 5. `stall_age` is material age, not unambiguously stall

The counter resets only when the action map, load, or carry credit changes. Base movement and cabin rotation do not reset it (`terra/state.py:300–332`). It is capped at 32 and exposed as a normalized scalar (`terra/state.py:59`; `terra/env.py:613–617`).

It can help because it changes the input during the measured 2–18-step material-neutral cycles, giving a feed-forward policy a bounded opportunity to leave them.

It can also mislabel the intended relay:

1. dump at an intermediate stage;
2. move empty through several workspaces;
3. rotate and align;
4. relift staged soil.

All correct actions between steps 1 and 4 increase `stall_age`, even when every movement is physically effective and necessary. It therefore conflates **no material change** with **no useful progress**. At 32 it saturates, after which a long cycle can again become observation-stationary. Because it is only an input, it does not directly penalize navigation, but training can learn an undesirable correlation such as “interact with soil before the counter grows,” which would work against distant relays.

The appropriate negative control is the successful `d16` staging trace: measure material-neutral run lengths and reject the feature if successful relays commonly approach or hit 32. The attached note already states this acceptance condition (`V8_V61_FAILURE_AUDIT_20260813.md:436–451`). There is currently no training evidence that `stall_age` helps: the earlier continuation jobs produced none, and the prepared replacement combines it with a sampler change, preventing component attribution (`V8_V61_FAILURE_AUDIT_20260813.md:550–608`).

Therefore `stall_age` is a plausible bounded cycle-breaking feature, not an established remedy. A direct action-outcome observation is better aligned with the confirmed no-effect failures and does not call legitimate empty relay movement a stall.

Unsupported interpretations remain: off-zone staging is intrinsically illegal; reward-v2 directly punishes every rehandle; horizon 450 explains the selected traps; clipping caused the measured exact aliases; the selected loaded cases are mechanics deadlocks; or action masking, weaker completion, broad reward retuning, or a large recurrent model is presently required. The 0/10 rescue at 900 steps and the exact branch/effect evidence directly argue against several of those claims (`V8_V61_FAILURE_AUDIT_20260813.md:250–310,620–660`).

TAGING_REVIEW_COMPLETE

