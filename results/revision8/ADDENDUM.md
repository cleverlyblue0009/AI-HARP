# Revision 8 — pre-registration addendum: one strong structured learned policy, and audit-informed randomisation

Written 2026-11-16 on branch `revision-8` (from `origin/revision-7` at a6dea21), **before any revision-8 training or
evaluation run**. Push commit of this file: `PENDING` (filled in by the next commit, which changes nothing else).

Registered analyses of revisions 5–7 stay as they are. No revision-5/6/7 result, threshold, checkpoint or analysis
file is changed. Interpreters: cloud `/root/r8env` (Python 3.12, NumPy 1.26.4), laptop `D:\aiharp-gpu\Scripts\python.exe`
(NumPy 1.26.4). Every job logs `sys.executable` and the NumPy version.

## 0. What had been seen before this was written

* Every revision-5/6/7 result, including all revision-5 custom-simulator EVAL rows on seeds 200–229 (classical, rule,
  AI-HARP) and every revision-6/7 Veins result (Tier A only).
* `results/revision8/READING_NOTES.md` and `results/revision8/probe_heads.json` (commit af9e05d): RAR decision
  contexts on training-pool seeds 100–103; no outcome metric.
* Platform checks, none on a STRUCT policy: (a) one committed classical SUMO row (rural d=20, seed 200, slotted_1p n=8)
  was reproduced exactly on this machine with `eclipse-sumo==1.19.0` (RWCR 0.86805, cost 0.6, 102 transmissions, as
  committed); (b) slotted_1p n=5 timing runs on training-pool seed 100 in the four headline cells, both backends
  (per episode with cached traces: 0.1–0.6 s, rural d=80 2.5 s; first SUMO trace generation 11–59 s). No metric of these
  runs was recorded.
* **No STRUCT code, no STRUCT run, no trained parameter and no randomised engine run exists.**

## 1. Author decisions recorded (2026-11-16, after READING_NOTES §8–9)

1. Both STRUCT heads are clipped-linear (§2). The sigmoid design of the instruction was dropped because it cannot
   contain RAR or slotted_1p: best-fit sigmoid p head 72 / 1,698 decision mismatches against RAR; sigmoid delay head with
   one slot term ≥ 484 mismatches against slotted_1p n=5 or n=50; clipped-linear 0 in every case (`probe_heads.json`).
2. Random draws: STRUCT always draws one uniform per first-reception decision (RAR's convention). Equivalence to
   slotted_1p is additionally shown as byte-identical whole engine runs, and a test proves that no engine component
   other than the policy reads the policy RNG stream. If either test fails, work stops and is reported.
3. Veins thresholds for H5: fitted on the committed Tier A classical replays (200–209), frozen, applied to Tier B (§6).
4. H4 comparator: the revision-5 H2 adaptive-classical comparator, selected on SEL only, scored on Tier B (§5).
5. 10 training seeds per STRUCT arm; H4 cost test is one-sided non-inferiority at +10 % (§5); Veins H5 uses training
   seeds 0–4 of each arm, fixed now; Holm across H4 and H5.
6. Cost knob: p shifted by 0.15·δ, δ ∈ {−2, −1, 0, 1, 2, 3}.
7. Tier B's partial exposure goes into the limitations (§10).
8. R2: the committed Veins raw outputs were checked for neighbour counts at decision times and contain none (§3).
9. R4 is kept; SUMO 1.19.0 runs here (§3).

## 2. Arms and the STRUCT policy

**Arms.** (a) AI-HARP reference: existing checkpoints and existing rows only, nothing retrained. (b) **STRUCT-plain**:
trained with every randomisation axis off. (c) **STRUCT-rand**: identical, but each training episode samples the axes of
§3. Training seeds 0–9 per STRUCT arm (20 runs).

**STRUCT** (`agents/structured.py`, registry name `struct`). Inputs, all causal and all present in the Veins bridge
request (READING_NOTES §6): x = (1, ρ_c, min(n, 30)/30, min(max(d, 0), R)/R), with ρ_c = `ctx.relevance`,
n = `ctx.n_neighbours`, d = `ctx.sender_distance_m`, R = `ctx.comm_range_m`; x' = (x, s50), with
s50 = `distance_slot(d, R, 50)` (`agents/persistence.py:43-58`).

* ORIGINATE → BROADCAST_NOW. own_tx_count > 0 → SUPPRESS.
* Otherwise (first reception; `wants_duplicate_callbacks = False`, so duplicates act only through the engine's
  cancel counter, exactly as for the slotted baselines): draw u = `ctx.rng.random()` (always exactly one draw);
  p = clip(w_p·x − 0.15·δ, 0, 1); if u ≥ p → SUPPRESS; else DEFER with
  delay = 1 + clip(round(w_d·x'), 0, 50) epochs (`numpy.round`, half to even) and cancel_on_duplicates = k,
  k = 1 + argmax(W_k·x) ∈ {1, 2, 3} (ties → the smaller k).
* Parameters: w_p 4 + w_d 5 + W_k 12 = **21** (≤ 40). δ is the evaluation knob, 0 in training.
* Equivalence parameters (tests must show 0 mismatches and the same number of draws as RAR; for slotted_1p, identical
  decisions and byte-identical whole runs):
  * slotted_1p n_slots=50: w_p = (1,0,0,0), w_d = (0,0,0,0,1), W_k row 1 = (1,0,0,0), rows 2–3 = 0;
  * slotted_1p n_slots=5: same with w_d = (−0.45,0,0,0,0.1);
  * RAR (a, b, c): w_p = (b + c, a, −b, 0), w_d as for n_slots=5, k = 1 (RAR's 5-slot deferral equals
    floor(s50/10) on the clamped range).
  * Exactly representable classical/rule settings: slotted_1p n ∈ {1, 2, 5, 50} (n divides 50), all 27 RAR settings.
    Not exactly representable: slotted_1p n ∈ {3, 8, 12, 20, 30}; counter_based (random delay 1 + U{0..4},
    `agents/counter.py:66`); weighted_p and p_persistence (BROADCAST, not DEFER, and no cancel); greedy/DV-CAST/SCF
    (neighbour geometry, carry); ETSI CBF (x-coordinate area test; inside its area at the 100 ms epoch it equals
    slotted_1p n=1, which is representable).
* Gate: `tests/test_structured.py` (≥ 1,000 recorded contexts from real engine runs on training-pool seeds; slotted_1p
  n=5 and n=50, RAR (1, 0.5, 0); whole-run identity on ≥ 3 training-pool seeds × the four headline cells for both slot
  counts; RNG-stream isolation) and `tests/test_oracle_isolation.py` with `agents/structured.py` and
  `agents/train_structured.py` checked, including transitive imports, must pass before any training run.

## 3. Audit-informed randomisation (custom simulator; every axis a flag, off by default)

Each flag has a test that the flag-off engine reproduces committed revision-5/7 rows exactly and that the default
engine output is byte-identical to explicit flag-off. Each randomised quantity draws from its own named RNG stream, so
the flags draw nothing when off. Per training episode of STRUCT-rand, each axis is sampled independently as below.

| Axis | Flag (`SimSettings` unless noted) | Per-episode sampling | Provenance |
|---|---|---|---|
| R1 duplicate multiplicity | `dup_count_per_frame` (bool) | off / on, p = 0.5. On: every decoded duplicate frame increments the duplicate counter (Veins, `AiharpApp.cc:752`); a fresh receiver that decoded m frames in its epoch decides first and then counts m − 1 duplicates, with the same cancel test (`AiharpApp.cc:745-758`). Off: one increment per epoch (`sim/engine.py:483`). | MEASURED: `results/revision6/REVISION6_REPORT.md:78` (2,223 Veins vs 635 custom decisions, seed 200); `results/revision7/ADDENDUM.md:22-27` (2,175.8 / 2,537.9 duplicate callbacks per Veins replay). These are Tier A seeds. |
| R2 neighbour-table error | `nb_table_error` = (p_miss, p_stale, ring) | p_miss ~ U[0, 0.5]: each geometric neighbour absent from the table; p_stale ~ U[0, 0.5]: each active vehicle at geometric distance in (R, 1.15 R] present (stale entry); ring fixed at 1.15 | **[ASSUMED]**. The only evidence: beacon count equals geometric count in 7 of 269 Veins decisions (`results/revision6/veins/port_validation_rules.json`, `rar.decisions_with_same_neighbour_count`). Checked and absent: neighbour counts at decision times in the committed raw outputs (`results/revision7/veins/raw/*/*.events.csv.gz`: no `F` lines in any of 1,055 logs; `*.sca.gz`: per-vehicle totals only, e.g. `receivedBSMs`). No revision-5/6 raw outputs are committed. |
| R3 coupling lag | `obs_lag_steps` ∈ {0, 1} | 0 / 1, p = 0.5. Decision inputs (own state, ρ_c, neighbour table, sender distance) are taken at step k − 1; radio and reception unchanged | MEASURED: `port_validation_rules.json` `rar.veins_position_equals_trace_step_minus_lag` = 269 of 269 |
| R4 mobility backend | trainer `--r4` (existing `backend` option) | fallback / SUMO, p = 0.5 | MEASURED: revision-5 SUMO-vs-fallback results (`results/revision5/analysis/secondary.json` `sumo`); SUMO 1.19.0 (`eclipse-sumo==1.19.0`), training-pool traces only, generated here |
| R5 PDR vs distance | `pdr_perturb` = (u, per-bin gaps) | u ~ U[0, 1]. For an in-nominal-range link in 50 m bin b with signed gap g_b = PDR_Veins − PDR_custom: s = u·g_b; if s > 0 a failed decode succeeds with probability s/(1 − PDR_custom,b); if s < 0 a success fails with probability −s/PDR_custom,b (capped at 1). Bins with fewer than 10 links in either simulator, or a NaN, have g = 0. Links beyond the nominal range and half-duplex receivers are not touched. | MEASURED: `results/revision5/analysis/h3.json` `network_layer.<cell>.pdr_vs_distance` (epoch mode, the pre-registered primary of D6), per headline cell. Tier A seeds 200–209. |

STRUCT-plain trains with every axis off. Evaluation of both arms in the custom simulator uses the default engine (every
axis off).

## 4. Trainer (`agents/train_structured.py`)

* CMA-ES (`cma` 4.6.0), population = the library default for 21 parameters (4 + ⌊3 ln 21⌋ = 13), seed of the CMA
  sampler = 1000 + training seed.
* Initial scale (per coordinate, `CMA_stds`, with σ0 = 1): w_p 0.2 each; w_d bias 2.0, ρ/n/d weights 2.0, s50 weight
  0.2; W_k 0.5 each.
* **Training cells:** the four headline (scenario, density) pairs, clear weather, fog_bank hazard, on training-pool seeds
  100–131 only. Per generation, 3 seeds per cell are drawn without replacement from the pool by
  `default_rng([training seed, generation, 0x5752])`; all 13 candidates of a generation are evaluated on the same 12
  episodes (and, for STRUCT-rand, the same sampled axes, from `default_rng([training seed, generation, cell, episode, 0x52])`).
* **Fitness (Deb's feasibility rule, no Lagrangian, no oracle quantity).** Per cell group, the pooled causal shortfall is
  s_g = (Σ target − Σ coverage) / Σ target over that cell's episodes. Causal coverage comes from
  `agents/constrained_reward.py` (`warned_at_risk`, `causal_coverage`), and target = 0.95 × the 32-seed ceiling in
  `results/coverage_targets.json` for `<scenario>|<density>|clear|fog_bank`. A candidate is feasible iff s_g ≤ 0 in
  all four groups. Feasible candidates are ranked by pooled cost Σ transmissions / Σ causal at-risk vehicles.
  Infeasible candidates are ranked by Σ_g max(s_g, 0). Every feasible candidate beats every infeasible one. This is
  implemented as the scalar f = cost if feasible, else 10⁶ + shortfall. An episode with no causal at-risk vehicle
  contributes nothing to s_g and its transmissions count in the cost numerator.
* **Warm start (SEL only).** Among the settings STRUCT represents exactly (slotted_1p n ∈ {1, 2, 5, 50}; RAR's 27), take
  the one that qualifies on SEL (selection-rule test: SEL mean RWCR ≥ q*, mean miss ≤ m*, D = ∞) in the most headline
  cells, ties broken by the mean SEL cost over those cells, then by the lower setting index. Source rows:
  `results/revision5/runs/classical__SEL__grid24__default__fallback.csv` and `rule__SEL__grid24__default__fallback.csv`.
  The CMA mean starts at its equivalence parameters, for every training seed.
* **Output:** the CMA mean after the last generation (no selection of a "best" candidate on any data).
* **Budget, fixed:** 80 generations per run. No extension after any evaluation result is seen. If the cloud session
  ends, the run resumes exactly from its last checkpoint (every generation is checkpointed: CMA state, RNG states,
  generation counter). A resume on a different machine is reported as such.
* Logging: `history.jsonl` per generation: best/median f, feasibility rate, best and median cost, per-group shortfall of
  the best candidate, the mean vector, σ. Bit-reproducibility is checked by re-running generation 1 of one run and comparing.

## 5. H4 — STRUCT-plain vs the retuned classical comparator, custom simulator

* Rows: STRUCT-plain seeds 0–9 × δ ∈ {−2..3} × the four headline cells, on SEL (400–419) and EVAL Tier A (200–209)
  and Tier B (210–229); fallback mobility, default channel, epoch mode, `experiments/revision1/runner.py` rows.
* Thresholds: frozen revision-5 q*, m* (`results/revision5/thresholds.json`). D = ∞.
* STRUCT per cell: each training seed picks its δ on SEL by the revision-5 selection rule; the cell qualifies if ≥ 5
  of 10 seeds pick and the means over picking seeds of EVAL RWCR ≥ q* and EVAL miss ≤ m*; its cost is the mean EVAL cost
  over picking seeds (`experiments/revision5/analysis/h2.py` `policy_eval`, unchanged).
* Comparator: adaptive classical = cheapest qualifying of the 55 classical settings on SEL per cell (`h2.py`
  `method_eval`), scored on the same EVAL tier. Never selected on Tier A or B.
* **Statistic T** = pooled mean over headline cells where both qualify of (STRUCT cost / comparator cost − 1).
* **Uncertainty:** hierarchical bootstrap, B = 10,000, resampling training seeds (10), SEL seeds (20), EVAL seeds of the
  tier (20 for Tier B); every selection recomputed; q*, m* frozen; RNG 20261116; percentile intervals; undefined
  replicates (no cell where both qualify) counted and excluded, infinite ones kept.
* **Decision rule, Tier B (primary): H4 is supported iff both hold:**
  (i) the number of headline cells where STRUCT-plain qualifies on Tier B ≥ the number where the comparator qualifies
  on Tier B (point counts);
  (ii) non-inferiority: the one-sided 95 % upper bound of T (the 0.95 percentile of the bootstrap distribution) is
  < +0.10. If T is undefined (no cell where both qualify), (ii) fails.
* **p-value for Holm:** p_H4 = share of defined bootstrap replicates with T ≥ +0.10 (p_H4 = 1 if T is undefined).
  H4 is supported only if (i) and (ii) hold and the Holm-adjusted p_H4 < 0.05.
* **Margin, justified in advance:** the revision-5 precision table gives a 95 % half-width of ±7.8 % for 10 training
  seeds × 30 EVAL seeds (`results/revision5/PREREGISTRATION.md:141-150`), and training-seed variance dominates. A +5 %
  margin is therefore not detectable at this precision. +10 % is the smallest margin that a true difference of 0 could
  plausibly clear. Tier B has 20 EVAL seeds, which changes the half-width little (the table: n_eval 20 vs 30 both
  0.078 at 10 training seeds).
* Also reported (descriptive, outside Holm): the revision-5 two-sided p and superiority/TOST verdicts for T; Tier A
  versions of everything; STRUCT-rand in the custom simulator; STRUCT vs the best rule (revision-5 rule comparator);
  the AI-HARP reference on Tier B from the committed revision-5 rows (seeds ref_s0–ref_s9).

## 6. H5 — does randomisation let STRUCT survive Veins?

* Replays (laptop): STRUCT-plain and STRUCT-rand, **training seeds 0–4 of each arm (fixed now)**, δ ∈ {−2..3}, rural
  d=20 and urban NLOS d=20, Veins Tier B seeds 210–229, SUMO mobility, the revision-8 app (§8) with the decision server.
  2 arms × 5 × 6 × 20 × 2 = 2,400 replays. Then Tier A 200–209 (1,200 replays) for comparability.
* **Thresholds:** per cell, q* = 0.95 × max mean RWCR over the 55 classical settings and m* = min mean actionable miss +
  0.05, computed on the committed Tier A classical replays (`results/revision5/veins/veins_rows.jsonl`, seeds 200–209,
  revision-5 rows as registered in revision 6), frozen in `results/revision8/veins_thresholds.json` before any STRUCT
  replay, and applied to Tier B. (Same definition and data as revision 6, so they will equal the revision-6 Tier A
  values; that is expected.)
* **Tier B Veins classical replays exist** for 210–229 (revision-5 audit), but **no Tier B classical matched cost will
  be used**: H5 is a qualification test only, not a cost comparison against classical in Veins. A Tier B classical cost
  comparison, if wanted later, is a separate deviation.
* Qualification: a training seed qualifies in a cell if its Algorithm-1 matched cost (`PolicyCurve.overhead_at` over
  its six δ, Tier B seed means, guard mean miss ≤ m*) is finite.
* **Per-cell rule:** H5 holds in cell c iff STRUCT-rand qualifies for ≥ 3 of 5 training seeds and STRUCT-plain for < 3
  of 5 in c.
* **Aggregate rule: H5 is supported iff the per-cell rule holds in at least one of the two cells and the Holm-adjusted
  p_H5 < 0.05.** "Rule met" and "statistically supported" are reported separately.
* **p-value:** per cell, one-sided Fisher exact test of the 2 × 2 table (arm × qualifies, 5 seeds each; alternative:
  rand qualifies more often); p_H5 = min(1, 2 · min(p_rural, p_urban)). Stated in advance: the smallest attainable
  per-cell p is 1/252 (5/5 vs 0/5), so p_H5 ≥ 0.0079, and only near-extreme splits can be significant; e.g. 3/5 vs 2/5
  meets the rule with p = 0.5.
* Tier A (descriptive, labelled EVAL-hindsight): the revision-6 procedure (thresholds on the same seeds).

## 7. Multiplicity

Holm across p_H4 and p_H5, α = 0.05. Everything else is exploratory.

## 8. Experiments C and D (descriptive; existing AI-HARP checkpoints)

* AI-HARP ref_s0–ref_s4, biases −2..3, τ = 0.5, sampled actions; rural d=20 and urban NLOS d=20; Tier A 200–209 (the
  seeds of the revision-6/7 0/5 result); SUMO mobility; thresholds and Algorithm 1 exactly as revision 7 Experiment A
  (Veins q*, m* on the same seeds from `veins_rows.jsonl`). 600 replays each.
* **C — geometric neighbour table:** the decision server replaces the beacon-derived neighbour arrays by the custom
  engine's geometric table at the request's step (`sim/engine.py:786-811`, including the NLOS test), using the SUMO trace
  and the app's full overheard set for `neighbour_informed`. Everything else unchanged.
* **D — one-epoch coupling lag removed:** an app parameter (default 0) shifts the app's step ↔ time mapping by one
  epoch, so a decision at step k sees Veins mobility equal to the trace at step k.
* **Decision rule (each):** "the difference explains the failure" iff qualification at rural d=20 in Veins rises from
  0/5 to ≥ 3/5 training seeds. Otherwise "does not explain it", which does not show that it plays no part.
* Gates before any reported replay (as revision 7): every new option off → classical slotted_1p n=8, rural d=20, seed
  200 event log byte-identical to the revision-6 gate (`707d6533…`), and ref_s0 bias +1 byte-identical to the revision-6
  replay (`220dd1fd…`); STRUCT bridge unit check (1,000 recorded contexts, in-process vs through the server: identical
  actions, delays and cancel thresholds) and trace check (every logged request re-sent, 0 mismatches).

## 9. Deviations

Any change, after this file's push commit, to anything fixed here (design, parameters count, axes and ranges,
trainer, budget, seeds, cells, thresholds, selection, tests, decision rules, scope) is a deviation. Each is logged in
`results/revision8/DEVIATIONS.md` as D12, D13, … with a timestamp, what changed, why, and exactly which results (if
any) had been seen. A bug fix that changes no definition is still logged. A change made after any Tier A or Tier B
STRUCT evaluation number has been seen says so explicitly.

## 10. Limitations stated in advance

* **Tier B is only partly held out.** Custom simulator: seeds 210–229 were revision-5 EVAL seeds; every revision-5
  result on them has been read (classical, rule, AI-HARP), although no STRUCT result exists. Veins: classical replays of
  210–229 exist and entered the revision-5 cross-fitted audit; 210–219 calibrated the D6 sub-epoch mode (not used here).
  Tier A (200–209) is less independent still: the R1 and R5 ranges were measured on Tier A seeds.
* The R2 range is assumed (one summary statistic as evidence).
* Training is on the four headline (scenario, density) pairs with clear weather and fog_bank only; no claim is made
  for other weathers, hazards or the held-out urban grid.
* STRUCT is deliberately small; reimplementing a published DRL dissemination scheme is out of scope.
* H5 uses 5 training seeds per arm; the power of the Fisher test is low (§6).
