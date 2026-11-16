# Revision 8: reading notes (step 0)

Branch `revision-8`, from `origin/revision-7` at a6dea21. Written before any revision-8 training or evaluation run,
and before the addendum. **No addendum exists yet, and none will be written until the conflicts in section 8 are resolved
by the author.** The only thing run so far is the design probe in section 8, which records decision contexts on
training-pool seeds and reads no outcome metric.

Interpreter here (cloud, Linux): `/root/r8env` (Python 3.12, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.3, torch 2.2.2 CPU,
cma 4.6.0). The system Python is 3.13, and NumPy 1.26.4 has no wheel for it. 4 vCPUs.

## 1. Seed sets

| Set | Seeds | Used for | Source |
|---|---|---|---|
| Training pool | 100–131 (32) | episode draws for every learned policy; coverage-target ceilings | `configs/agent.yaml:363`, `agents/train.py:123-147` (refuses overlap), `results/coverage_targets.json` (`provenance.n_seeds` 32, seeds 100–131) |
| Normaliser hold-out | 900–904 | feature normalisation only | `configs/agent.yaml:50` |
| CAL | 300–309 | q*, m* only | `results/revision5/PREREGISTRATION.md:11`, `protocol.json` `seed_sets.CAL` |
| SEL | 400–419 | operating-point selection for every method | `PREREGISTRATION.md:12,16-20` (replaced 100–119, which overlapped the training pool) |
| EVAL | 200–229 | final scoring only | `PREREGISTRATION.md:13` |
| Veins Tier A | 200–209 | revision-6/7 Veins replays (EVAL-hindsight) | `results/revision6/PREREGISTRATION_ADDENDUM.md:61` |
| Veins Tier B | 210–229 | optional in revision 6; never replayed for rules or policy | same line; `REVISION6_REPORT.md:27-28` |

Disjointness of CAL, SEL, EVAL, LEGACY (0–9), the training pool and 900–904 was verified in revision 5
(`protocol.json` `seed_sets.disjointness`).

**How clean Tier B is.** Tier B is held out from every revision-6/7 Veins replay of rules and policy. It is *not*
unseen in general:
* custom simulator: 210–229 were EVAL seeds in revision 5, and every revision-5 EVAL result on them has been read
  (classical, rule and AI-HARP rows, `results/revision5/runs/*EVAL*`);
* Veins: classical replays of 200–229 exist (`results/revision5/veins/veins_rows.jsonl`, 6,600 rows = 55 settings ×
  4 cells × 30 seeds), seeds 210–229 entered the revision-5 cross-fitted audit (`DEVIATIONS.md:100-102`), and
  210–219 were the calibration seeds of the D6 sub-epoch mode (`DEVIATIONS.md:128-129`). That mode is not used here.
* The provenance of R1 and R5 (section 3 of the instruction) is Tier A: R1 comes from one replay on seed 200
  (`REVISION6_REPORT.md:78`), and R5 from the H3 PDR-vs-distance gaps on seeds 200–209 (`DEVIATIONS.md:72-88`).
  Tier A is therefore not independent of the randomisation ranges. That is one more reason to make Tier B primary.

## 2. Frozen q*, m* per headline cell (custom simulator, CAL)

From `results/revision5/thresholds.json` (rule at line 29: q* = 0.95 × max mean oracle RWCR over the 55 settings;
m* = min mean actionable miss + 0.05):

| Cell | q* | m* | Lines |
|---|---|---|---|
| rural d=2 | 0.59323 | 0.58200 | 31, 40, 47 |
| rural d=20 | 0.87101 | 0.35749 | 50, 59, 66 |
| rural d=80 | 0.87082 | 0.36948 | 69, 78, 85 |
| urban NLOS d=20 | 0.92316 | 0.12673 | 88, 97, 104 |

Absolute variant: m* = 0.40 (`PREREGISTRATION.md:31`). Veins has no CAL. Its thresholds are recomputed per cell
inside Veins on the analysed replay seeds (`PREREGISTRATION_ADDENDUM.md:72-74`). On Tier A they are rural d=20
0.879 / 0.404 and urban NLOS d=20 0.498 / 0.602 (`REVISION6_REPORT.md:31-36`).

## 3. Algorithm 1 (matched cost)

* q* = 0.95 × max over the 55 classical settings of mean RWCR; the guard is mean actionable miss ≤ m*, with m* = min mean miss + 0.05
  (`DEVIATIONS.md:96-99`, D3; `ANALYSIS_SPEC.md:52-56`; `PREREGISTRATION_ADDENDUM.md:72-77`).
* The cost at q* is `PolicyCurve.overhead_at` (`analysis/pareto.py:100-140`). Take the cheapest setting that reaches q* and
  passes the guard. Interpolate linearly against the next-cheaper setting that falls short, only if that setting also
  passes the guard (`pareto.py:129-140`). If no setting reaches q*, the cost is `inf`.
* In the custom-simulator H2 the primary cost is **not** Algorithm 1. It is the realised EVAL cost of the SEL-selected
  setting (`PREREGISTRATION.md:33-48`; selection code `experiments/revision5/analysis/common.py:76-88`). Algorithm 1
  is the matched cost in H3 and in the Veins analyses of revisions 6 and 7.

## 4. Qualification rules

* Custom simulator, single-setting method: its SEL pick qualifies if EVAL mean RWCR ≥ q* and EVAL mean miss ≤ m*
  (`ANALYSIS_SPEC.md:17`; `h2.py:55-62`).
* Custom simulator, learned policy: each training seed picks its own knob value (suppression bias) on SEL. The cell
  qualifies if at least half of the seeds pick (`len(picks)*2 >= n`, so 3 of 5 and 5 of 10), and the means over the
  picking seeds of EVAL RWCR and EVAL miss meet q* and m*. The policy's cost is the mean over the picking seeds
  (`ANALYSIS_SPEC.md:37-39`; `h2.py:65-84`).
* Veins (revisions 6 and 7): a training seed qualifies in a cell if its Algorithm-1 matched cost over its 6 biases is
  finite. The aggregate is defined when at least half of the seeds qualify (`PREREGISTRATION_ADDENDUM.md:79-82`), which
  revision 7 states as 3 of 5 (`results/revision7/ADDENDUM.md:58-60`).
* H2 test (to be reused for H4). The statistic is the pooled mean, over cells where both methods qualify, of
  (policy / adaptive − 1). It uses a hierarchical bootstrap over training seeds × SEL × EVAL with every selection
  recomputed, B = 10,000, RNG 20261031, a percentile 95 % CI, and a two-sided p = 2·min(P(T ≤ 0), P(T ≥ 0)).
  Superiority requires the CI upper bound < 0. Equivalence uses TOST ±5 % (90 % CI inside (−0.05, 0.05)). A
  non-significant result is never called "no difference". Sources: `PREREGISTRATION.md:68-81`, `ANALYSIS_SPEC.md:32-47`,
  `h2.py:87-150`. The adaptive classical comparator is the cheapest qualifying setting of all 55 on SEL, per cell
  (`ANALYSIS_SPEC.md:36`).

## 5. Deviation numbering so far

D1–D7 in `results/revision5/DEVIATIONS.md` (D1 :5, D2 :33, D4 :42, D5 :56, D3 :70, D6 :107, D7 :165). D8–D11 are in
`results/revision6/PREREGISTRATION_ADDENDUM.md` (D8 :105, D9 :113, D10 :135, D11 :171). Revision 7 logged none
(`results/revision7/ADDENDUM.md`, `REVISION7_REPORT.md`). **The last one is D11, as the instruction says.** Revision 8 starts at D12.

## 6. The Veins bridge interface

* App (revision-7 copy, `experiments/revision7/veins/aiharp/`). Policy kinds are listed at `AiharpApp.h:27-28`; the kinds that
  receive duplicate callbacks at `AiharpApp.h:111`. `policyDecide` sends AI_HARP to `bridgeDecide`
  (`AiharpApp.cc:467`). `bridgeDecide` (`AiharpApp.cc:582-630`) serialises the holder's DecisionContext as one JSON
  line. Fields: own state from Veins mobility, `relevance`/`eta_s` from the app's port of `hazard/risk_field.py`
  (`AiharpApp.cc:599`), `sender_distance_m` (:596), `comm_range_m`, and the neighbour arrays `nb`, `nb_dist`, … from the
  beacon table (:620). The call is synchronous (`bridgeCall`). The reply is `act delay cancel relay1 relay2`, and the app
  applies it with the engine's `_apply` semantics (`AiharpApp.cc:632-660`, `decide`).
* Neighbour table in Veins: the latest decoded beacon per sender, a 1.05 s lifetime (`AiharpApp.h:32`), extrapolated to
  now and restricted to the nominal range (`AiharpApp.cc:222-235`). CAM velocity goes into beacons for the policy kinds
  (`AiharpApp.cc:263`, D9). In the custom engine the table is trace geometry within the nominal range, plus an NLOS
  link-budget test in urban NLOS (`sim/engine.py:786-811`).
* Decision server (`experiments/revision6/veins/decision_server.py`): `ctx_from_request` (:40-65) rebuilds the
  DecisionContext, `Session.decide` (:98-114) draws from streams named (replay seed, vehicle, epoch, k) (:36-37,
  :99-106), and the server logs every request and reply for the trace check (:139). Port validation: unit check over
  1,000 recorded contexts, in-process vs through the server, plus a trace check (all logged requests re-sent, 0
  mismatches) (`validate_policy.py:1-14`; revision 7 `ADDENDUM.md:96-97`).
* **STRUCT inputs exist in the bridge context.** `relevance` = relevance_causal (:599); n_neighbours = `len(nb)`, which is
  beacon-derived (:620, :222-235); `sender_distance_m` (:596); `comm_range_m`. The distance-slot term is a function of
  sender_distance_m and comm_range_m. No new field is needed.

## 7. Every place the engine calls a policy, and duplicate receptions

`sim/engine.py`:
* ORIGINATE: :234 (`_decide_and_apply`, trigger ORIGINATE).
* First reception: :466-481. Per epoch, a receiver that decoded at least one frame gets **one** delivery, whatever
  the number of frames it decoded (:457, the sender chosen at :459-461). A fresh holder runs the policy with RECEIVE
  (:476).
* Duplicate reception: :482-510. `dup_count[r] += 1` happens **once per epoch**, however many duplicate frames were
  decoded (:483). If a cancel threshold is set and reached and a transmission is pending, the transmission is cancelled
  (:486-492). Otherwise the policy is called again, but only if `wants_duplicate_callbacks` (:214) and no transmission is
  pending (:493). Revision-7 cap: :499-504.
* TIMER (carry / re-decision): :268-279, skipped while a transmission is queued (:272).
* The policy RNG stream `rng_pol` (:181) is passed only to the policy (:188, contexts). Nothing else in the engine draws
  from it.

Veins (`AiharpApp.cc:731-770`): every decoded hazard frame is its own `onWSM` call. A fresh one decides (:745-750). A
duplicate increments `dupCount` **per frame** (:752), runs the same cancel test (:753-758), and makes a duplicate
callback for the kinds at `AiharpApp.h:111` (:759-768). So the duplicate counter grows faster in Veins than in the
custom engine. For a policy without duplicate callbacks (slotted, RAR, STRUCT as specified), that is exactly what R1
must randomise.

## 8. Conflicts found before any design was fixed (blocking)

The instruction asks for STRUCT heads p = sigmoid(w_p·x) and delay = 1 + round(50·sigmoid(w_d·x')), and requires exact
equivalence (0 mismatches, same RNG consumption) to slotted_1p (n_slots 5 and 50) and RAR (default abc). If that fails,
it says to stop and report rather than change the design. A probe was run to check (not training, not evaluation):
`python -m experiments.revision8.probe_heads` → `results/revision8/probe_heads.json`. It ran RAR (1, 0.5, 0) on
**training-pool seeds 100–103** only, in rural d=20, urban NLOS d=20 and rural d=2 (fallback mobility), and recorded
1,698 first-reception contexts with RAR's own uniform draws. No outcome metric was read or stored. The sigmoid weights
were fitted on the same contexts, which is the most favourable case for the sigmoid.

| Head | Target | As specified (sigmoid), best fit | Clipped-linear alternative |
|---|---|---|---|
| p | RAR (1, 0.5, 0) | **72 decision mismatches of 1,698** (max abs p error 0.094) | 0 mismatches, p identical (max abs error 0.0) |
| delay, slot term s50 | slotted_1p n=5 | **565** mismatches | 0 (`1 + clip(round(0.1·s50 − 0.45), 0, 50)`) |
| delay, slot term s50 | slotted_1p n=50 | **844** | 0 (`1 + clip(round(s50), 0, 50)`) |
| delay, slot term logit((s50+0.25)/50) | slotted_1p n=5 | **484** | – |
| delay, same | slotted_1p n=50 | 0 | – |

1. **p head.** No sigmoid of a linear function of (1, ρ_c, n/n_max, d/R) can equal RAR's clipped-linear
   p = clip(a·ρ_c + b·(1 − min(n,30)/30) + c, 0, 1). A clipped-linear head p = clip(w_p·x, 0, 1) contains RAR exactly,
   and contains slotted_1p (p ≡ 1) with w_p = (1, 0, 0, 0).
2. **Delay head.** With one distance-slot term, no 1 + round(50·sigmoid(·)) reproduces both n_slots = 5 and 50. A
   clipped-linear head 1 + clip(round(w_d·x'), 0, 50), with x' = (x, s50) and s50 the 50-slot index of
   `agents/persistence.py:43-58`, reproduces both. RAR's 5-slot deferral is also covered, because
   floor(5(1−d)) = floor(s50/10) on the clamped range.
3. **RNG consumption cannot match both targets under any single rule.** slotted_1p (p = 1) draws no uniform
   (`persistence.py:163`). RAR always draws one, even when its clipped p is 0 or 1 (`rule_baselines.py:60`), and in
   51.8 % of the recorded RAR contexts p is clipped to 0 or 1. "Draw only when 0 < p < 1" breaks RAR; "always draw"
   breaks slotted_1p's per-decision count. Because `rng_pol` is read by nothing but the policy (section 7), "always
   draw" still gives slotted_1p-identical **whole runs**, and that can be tested byte-for-byte.

**Not blocking, recorded for the addendum:** counter_based is not representable: its delay is a random draw of
1 + U{0..4} (`counter.py:66`) where STRUCT's delay is deterministic, and its RNG use differs. ETSI CBF at the 100 ms
epoch reduces to "defer 1 epoch, cancel on 1" inside its area (`cbf.py:60-67`), which is slotted_1p with n_slots = 1.
STRUCT can represent that, but not CBF's x-coordinate area test.

## 9. Other points the addendum must settle (proposals, not decisions)

**Resolved by the author (2026-11-16), recorded in `ADDENDUM.md` §1.** Both heads are clipped-linear:
p = clip(w·x − 0.15·δ, 0, 1) and delay = 1 + clip(round(w·x'), 0, 50), with the 50-slot index as the slot feature.
The sigmoid design of the instruction was dropped because it cannot contain RAR or slotted_1p, which is the point of
STRUCT. In `probe_heads.json` the best-fit sigmoid p head mismatches RAR in 72 of 1,698 decisions. The sigmoid delay
head with one slot term mismatches slotted_1p n=5 in at least 484 decisions, or n=50 in 844. Clipped-linear heads
mismatch in 0. STRUCT always draws one uniform (RAR's convention), and slotted_1p equivalence is also tested as
byte-identical whole runs. The proposals below were accepted as written, except: 10 training seeds per arm and a
+10 % non-inferiority margin for H4; Veins H5 on training seeds 0–4; R2 checked against the raw outputs first (no
neighbour counts recorded, so it is [ASSUMED]); R4 kept (SUMO 1.19.0 runs here).

* **H5 thresholds in Veins.** The custom CAL thresholds cannot be used in Veins: revision 6 recomputed them inside Veins
  because H3 failed. "No threshold may be fitted on Tier B" then implies: fit Veins q*, m* per cell on the Tier A
  classical replays (`veins_rows.jsonl`, 200–209) and apply them to STRUCT's Tier B replays. The Tier A
  EVAL-hindsight version would be reported as well, for comparability with revisions 6 and 7.
* **H4 comparator.** Revision 5 has no "best classical family" in the custom H2. Its comparator is the adaptive classical
  pick (cheapest qualifying of the 55 on SEL, per cell), which is the per-cell best family. Proposed: use exactly that.
* **"Pooled matched cost not higher".** Revision 5 defines superiority (95 % CI upper < 0) and TOST equivalence, but no
  non-inferiority. A rule must be chosen before any run. Proposed: one-sided non-inferiority at the revision-5 margin
  of +5 % (90 % CI upper < +0.05), with the Holm p = bootstrap P*(T ≥ 0.05). The revision-5 superiority and two-sided
  p would be reported too. **Precision, stated in advance:** with 5 training seeds the revision-5 precision table gives a
  95 % half-width of about ±11 % (`results/revision5/PREREGISTRATION.md:141-148`, n_train = 5). A 5 % non-inferiority
  margin is therefore unlikely to be demonstrable unless STRUCT is clearly cheaper.
* **STRUCT's knob.** Selection and Algorithm 1 need an operating curve per training seed, like AI-HARP's six suppression
  biases (`protocol.json` `selection_rule.settings.learned_policy`). Proposed: δ ∈ {−2, …, 3}, entering the p head as
  clip(w_p·x − 0.15·δ, 0, 1). The scale 0.15 is a design choice to fix before training.
* **Warm start.** The SEL-best classical or rule setting differs per cell, but STRUCT is one policy for all cells.
  Proposed rule (SEL only): among the settings STRUCT can represent exactly (slotted_1p's 9, RAR's 27), take the one
  that qualifies on SEL in the most headline cells, with ties broken by mean SEL cost over those cells.
* **R2 neighbour-table error.** Only one summary statistic is committed: 7 of 269 decisions where the beacon count equals
  the geometric count (`results/revision6/veins/port_validation_rules.json` `rar.decisions_with_same_neighbour_count`).
  No distribution of the difference is committed, so the R2 range is [ASSUMED], anchored to that statistic.
* **R4 SUMO.** No SUMO trace is committed (`cache/traces/` is git-ignored, `ENVIRONMENT.md` "Storage"). SUMO would have to
  be installed here (`pip install eclipse-sumo`), or R4 dropped and logged.
* **R5.** The H3 PDR-vs-distance bins (Veins and custom, 50 m bins, headline cells) are in `results/revision5/veins/`
  (`veins_rows.jsonl` `raw.bins_*`, `custom_h3.jsonl`) and in `analysis/h3.json`. They are Tier A, see section 1.
* **Experiments C and D.** C (geometric table) can be done in the decision server: rebuild the neighbour arrays from the
  SUMO trace at the request's step with the engine's own `_context` geometry, which includes the NLOS test, with the app
  sending its full overheard set. D (lag removed) can be an app parameter that shifts the app's step ↔ time mapping by
  one epoch (`stepAt`/`timeOf`, `AiharpApp.cc:188-190`), so a decision at step k sees Veins positions equal to the
  trace at k. Both need an off-means-byte-identical gate, as in revision 7 (`ADDENDUM.md:88-100`).

## 10. Timing seen so far

The probe's 12 RAR runs (4 seeds × rural d=20, urban NLOS d=20, rural d=2), including first-time trace generation, took
34.7 s on one core. This is a first indication only. The estimate required before training will come from a timed smoke
run of the trainer.
