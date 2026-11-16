# Revision 9 — pre-registration addendum: a causal deadline-miss constraint in STRUCT training

Written 2026-11-16 on branch `revision-9` (from `origin/revision-8` at b67f18c), **before any revision-9 training or
evaluation run**. Push commit of this file: `PENDING` (recorded by the next commit, which changes nothing else).
Revision-5/6/7/8 results, thresholds, checkpoints, analysis files and deviations are unchanged and never edited.
Deviations continue the revision-8 numbering (next: D16).

## 0. Why revision 9 exists (post hoc, stated plainly)

Revision 8's H5 was **not supported** (`results/revision8/analysis/h5.json`). A **post-hoc** observation, read after that
registered result (`results/revision8/REVISION8_STATUS.md` §6; `h5.json` `per_run`): in Veins rural d=20, STRUCT-rand
reached mean RWCR 0.897–0.933 (> q* 0.879) at cost 0.28–0.46, but its actionable deadline-miss rate was 0.54–0.85
(> m* 0.404), so no matched cost was finite. The revision-8 training fitness (`agents/train_structured.py` at b67f18c,
:164-183) scored only causal coverage and cost; it had no deadline term, while every evaluation enforces a miss guard.
**That omission was a design error in the revision-8 specification.** Revision 9 tests one pre-specified fix: a causal
deadline-miss constraint in training. **Revision 9 is motivated by a post-hoc observation**; this is stated here and in
every revision-9 report. Revision 8's results stand unchanged.

## 1. What had been seen before this was written

* Every revision-5–8 result, including revision-8 H4, H5, Experiments C/D and every descriptive table.
* `results/revision9/READING_NOTES.md`; the causal-miss identity test (`tests/test_causal_miss.py`); the flag-off identity
  test (`tests/test_train_flag_identity.py`).
* `results/revision9/miss_targets.json`, `gates.json` and their episodes (`gates_runs*.csv`): **training pool only**
  (seeds 100–131), fixed policies only (§2). No STRUCT9 parameter, training run or evaluation row exists.

## 2. The causal deadline miss, its bounds and the gates (all training pool, all committed before this file)

* **Causal actionable miss** (`agents/causal_miss.py`): the `analysis/metrics.py` actionable-miss formula (:246-307)
  with its two oracle-only inputs replaced by their causal counterparts (causal at-risk set; causal relevance > 0 at the
  informing step). Bit-identical to the metric when the oracle inputs are injected. Never imports the oracle (tested
  transitively).
* **Bounds** b_g per training group = min(pool-mean causal miss of flooding default, of DV-CAST default) + 0.05, over all
  32 training-pool seeds, fallback mobility, the four training groups (`results/revision9/miss_targets.json`; episodes
  `gates_runs_slice*of20.csv` from 20 runners; 3,510 of them also computed in the cloud session agree to ≤ 1 ulp,
  `gates_runs_local_partial_crosscheck.csv`):

  | Group | Flooding default | DV-CAST default | **b_g** | Coverage target |
  |---|---|---|---|---|
  | rural d=2 | 0.4442 | 0.4421 | **0.4921** | 0.7201 |
  | rural d=20 | 0.1209 | 0.1206 | **0.1706** | 0.9256 |
  | rural d=80 | 0.0036 | 0.0014 | **0.0514** | 0.9484 |
  | urban NLOS d=20 | 0.7357 | 0.7413 | **0.7857** | 0.2656 |
* **Gate A (feasibility): passed.** Classical settings (of 55) meeting both the coverage target and b_g on the 32-seed pool means: rural d=2 10 (coverage alone 10); rural d=20 40 (coverage alone 40); rural d=80 46 (coverage alone 47); urban NLOS d=20 35 (coverage alone 35) (`gates.json` `gate_A`).
* **Gate B (fitness sanity): passed.** Revision-9 fitness over all 4 × 32 pool episodes (`gates.json` `gate_B`): always_suppress f = 1000005.6951 (infeasible); flooding f = 2.3738 (feasible); slotted_1p_5 f = 0.6599 (feasible); slotted_1p_50 f = 1000000.2850 (infeasible); dvcast_default f = 0.6588 (feasible); rar_warm_start f = 1000000.0252 (infeasible). Always-suppress is infeasible; slotted_1p 5 and DV-CAST default are feasible and cheaper than flooding. **The revision-8 warm start (RAR (1, 1, 0.25)) is infeasible under the new constraint**; it stays the warm start (not changed: the instruction fixes it), so CMA-ES starts from an infeasible point.
* **Gate C (proxy quality, descriptive):** Spearman ρ between causal and oracle actionable miss per episode: reference policies pooled 0.298 (n = 768); per group rural d=2 0.965, rural d=20 0.521, rural d=80 0.492, urban NLOS d=20 0.372; all 59 settings (key `pooled_all_57_settings`) pooled 0.394 (`gates.json` `gate_C_descriptive`). The proxy tracks the oracle well only in rural d=2; elsewhere moderately. Nothing is tuned on it.

## 3. The only change to training

Revision-8 Deb's rule, with feasibility extended (`agents/train_structured.fitness` with `miss_bounds`, enabled by
`--miss-targets results/revision9/miss_targets.json`):
* **feasible** iff every group's pooled causal coverage shortfall ≤ 0 **and** every group's mean causal actionable miss
  (mean over that group's episodes in the generation's batch with at least one causally actionable vehicle) ≤ b_g; a
  group with no such episode in the batch is not miss-constrained in that generation;
* feasible candidates ranked by pooled cost Σ transmissions / Σ causal at-risk vehicles (unchanged);
* **infeasible** candidates scored 10⁶ + Σ_g max(shortfall_g, 0) + Σ_g max(miss_g − b_g, 0) — unscaled, both terms being
  fractions in [0, 1].

**Everything else is identical to revision 8:** STRUCT (21 parameters, clipped-linear heads, `agents/structured.py`),
warm start = revision-8 `warm_start.json` (RAR (1, 1, 0.25)), CMA-ES (cma 4.6.0, popsize 13, `CMA_stds` as revision 8,
seed 1000 + training seed), 80 generations, 4 training groups × 3 pool seeds per generation, R1–R5 ranges and sampling
(revision-8 ADDENDUM §3; **R2 still [ASSUMED]**), δ knob {−2..3}, output = CMA mean after the last generation. With the
flag off, the trainer is revision 8's (`tests/test_train_flag_identity.py`).

## 4. Arms and seeds (fixed now)

**STRUCT9-plain** (all randomisation axes off) and **STRUCT9-rand** (R1–R5 sampled per episode), training seeds **0–4**
each: 10 runs, outputs `results/revision9/train/<arm>_s<k>/`.

## 5. PRIMARY H6 (Veins) — the only test in its family (no Holm partner)

* Replays: STRUCT9-plain and STRUCT9-rand s0–4, δ −2..3, rural d=20 and urban NLOS d=20, **Veins Tier B seeds 220–229**
  (never used by any STRUCT replay), SUMO mobility, the revision-8 app and decision server: 1,200 replays.
* Thresholds: `results/revision8/veins_thresholds.json`, unchanged (rural d=20 q* 0.879, m* 0.404; urban NLOS d=20 q*
  0.498, m* 0.602).
* A training seed qualifies in a cell if its Algorithm-1 matched cost over its six δ (seed means over 220–229, guard mean
  actionable miss ≤ m*) is finite.
* **Per-cell rule:** STRUCT9-rand qualifies for ≥ 3/5 seeds **and** STRUCT9-plain for < 3/5.
* **Aggregate:** rule met in ≥ 1 cell. One-sided Fisher exact per cell (rand qualifies more often);
  p_H6 = min(1, 2 · min(p_rural, p_urban)). **H6 is supported iff the rule is met and p_H6 < 0.05.** The smallest per-cell
  p is 1/252, so only near-extreme splits can be significant (e.g. 3/5 vs 2/5 gives p = 0.5).
* Analysis code: `experiments/revision9/analysis/veins9.py` (committed before any data).

## 6. Secondary and completion analyses (descriptive, pre-specified)

* **S1:** STRUCT9-rand vs revision-8 STRUCT-rand on the same Veins seeds 220–229, per cell: qualification count, best
  mean RWCR per seed, actionable miss and cost at that setting, minimum miss over δ (`veins9.py s1`). Question: does the
  deadline term lower miss, and at what reach and cost?
* **S2:** STRUCT9-plain and STRUCT9-rand in the custom simulator, Tier B 210–229 (and Tier A 200–209), four headline
  cells, revision-8 H4 comparator and analysis code (`experiments/revision8/analysis/h4.py analyse_tier`, unchanged),
  5 seeds (a cell qualifies if ≥ 3 of 5 pick on SEL); labelled descriptive (H4 used 10 seeds) (`analysis/s2.py`).
* **C1:** revision-8 STRUCT-plain and STRUCT-rand s0–4 on Veins Tier B 220–229 — the scope revision 8 registered and D12
  cut. **The registered revision-8 H5 verdict (210–219) stays the result of record**; the 220–229 table and the combined
  210–229 table are reported beside it as descriptive and never replace it (`veins9.py c1`).

## 7. Compute (as revision-8 D14)

GitHub-hosted runners; outputs on branch `r9-ci`, merged into `revision-9`. Training: one run per runner; each
`run_meta.json` records CPU, Python, NumPy 1.26.4, cma version. Veins: every runner restores the committed AI-HARP
checkpoints (hash-checked) and passes **gate check 1** (classical slotted_1p n=8, rural d=20, seed 200, byte-identical to
`707d6533…`) **and gate check 2** (AI-HARP ref_s0 bias +1, every new option off, byte-identical to `220dd1fd…`) before any
replay, or contributes nothing; every STRUCT decision log is re-sent offline (0 mismatches required); replay acceptance
as revision 8. Every runner verifies that its outputs are on `r9-ci` before the job succeeds. Decision-server logs are
never committed to `revision-9` (`.gitignore`; removed from the index after every checkout from `r9-ci`). Order:
training (10 runs) → H6 (1,200 replays) → S2 → C1 (1,200 replays).

## 8. Deviations

Any change after this file's push commit to anything fixed here is a deviation, logged in
`results/revision9/DEVIATIONS.md` as D16, D17, … with a timestamp and exactly what had been seen.

## 9. Limitations stated in advance

* **Post-hoc motivation** (§0).
* 5 training seeds per arm; the Fisher test has low power (smallest per-cell p 1/252).
* Proxy quality: training sees the causal miss, evaluation scores the oracle miss (gate C).
* R2 remains [ASSUMED]. CPU heterogeneity across runners (runs are deterministic per CPU type).
* Reimplementing a published DRL scheme stays out of scope.
