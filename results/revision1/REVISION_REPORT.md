# Revision 1 — numbers and figures

Branch `revision-1`, forked from `main` at `461c754`. Everything new lives under
`results/revision1/`. No paper text was edited. Each section gives the
numbers, the file each number came from, and the command that regenerates it.
Where a number in the paper could not be reproduced, that is stated.

<!-- SUMMARY-TABLE -->

## Environment note (affects every new run)

The interpreters `ENVIRONMENT.md` documents (`D:\aiharp-env`, `D:\aiharp-gpu`)
and SUMO (`D:\sumo-1.19.0`) **no longer exist on this machine**. `D:\aiharp-gpu`
was rebuilt exactly as `ENVIRONMENT.md` specifies (Python 3.12, torch 2.2.2+cu121,
torch-geometric 2.8.0, NumPy 1.26.4, SciPy 1.13.1). Stage A needed no torch and
ran under the base Miniconda Python 3.12 (`numpy 2.0.1`, reading files only).
SUMO was not reinstalled, so nothing in this revision re-runs a SUMO cell; all
new simulation is fallback-backend, like every committed headline number.

---

## Stage A — fact extraction (no new runs)

Regenerate all of Stage A: `python -m experiments.revision1.stage_a`
→ `results/revision1/A/stage_a_facts.json` (every number below is in that file
unless another file is named). A6 is in `results/revision1/A6_facts.md`.

### A1. Coverage target q* — three definitions, not one

| # | used by | definition | code |
|---|---|---|---|
| E | Algorithm 1, Fig. 2 (`fig02_headroom`), Table 4/reference table, Fig. 3, every regret/margin | q* = 0.95 × max mean **oracle RWCR** over every baseline knob setting in the cell (55 settings), eval seeds 0–9, agent excluded; **plus** a deadline guard: actionable-miss ≤ cell best + 0.05 | `analysis/comparator.py` `resolve_target`, `resolve_miss_bound` |
| T | training q*_g (λ updates, Fig. 6/12 shortfall) | q*_cell = 0.95 × max(flooding default, DV-CAST default) mean **causal coverage**, 32 training seeds (100–131), per (scenario, density, weather, hazard); the group (scenario, density) multiplier steps on pooled shortfall | `agents/constrained_reward.py`, `experiments/coverage_targets.py`, `results/coverage_targets.json` |
| F | §VII-F feasibility sweep, Table 5 arms | training side: definition T **but computed from the 2-seed ceilings** in force at `a378bcf`; evaluation side: 0.95 × best of **63** settings (55 knob + 8 registry defaults) | `experiments/sparse_feasibility.py`, `results/feasibility/sparse_feasibility.json` |

**E — per headline cell** (`results/pareto_cells.json`):

| cell | ceiling (setting) | q* | miss bound |
|---|---|---|---|
| rural d=2 | 0.67469 (dvcast n_slots=50) | 0.64095 | 0.57918 |
| rural d=20 | 0.92498 (p_persistence_03 p=0.9) | 0.87874 | 0.36353 |
| rural d=80 | 0.91529 (slotted_1p n_slots=3) | 0.86952 | 0.36801 |
| urban_nlos d=20 | 0.97489 (counter_based threshold=4) | 0.92615 | 0.11653 |

**T — per training group** (mean of the 6 training weather × hazard cells; the
headline cell's own clear/fog_bank target in the last column):

| group | mean q*_g | min–max over cells | clear/fog_bank cell |
|---|---|---|---|
| rural d=1 / 2 / 3 / 5 | 0.240 / **0.748** / 0.832 / 0.917 | 0.213–0.273 / 0.710–0.777 / 0.789–0.869 / 0.893–0.944 | — / 0.720 / — / — |
| rural d=10 / 20 / 40 / 80 | 0.930 / 0.937 / 0.944 / 0.949 | 0.913–0.949 / 0.923–0.950 / 0.940–0.947 / 0.948–0.950 | — / 0.926 / — / 0.948 |
| urban_nlos d=1 / 2 / 3 / 5 | 0.201 / 0.293 / 0.298 / 0.293 | 0.196–0.206 / 0.275–0.309 / 0.274–0.322 / 0.268–0.322 | |
| urban_nlos d=10 / 20 / 40 / 80 | 0.283 / 0.283 / 0.298 / 0.339 | 0.261–0.308 / 0.255–0.313 / 0.264–0.335 / 0.294–0.383 | — / 0.266 / — / — |

**Mismatches flagged**

1. **§VII-F / Table 5 used stale training targets.** `sparse_feasibility.json`
   records `target_seeds_used_for_ceiling: 2`; its rural d=2 mean target is
   **0.891** and urban d=2 **0.329**. The targets every campaign run (and the
   reference agent) trained against are the 32-seed ones: **0.748** and
   **0.293**. "0 of 63 fixed settings meet the rural d=2 target" is true only
   for the stale 0.891 target (best fixed coverage 0.781 > 0.748). Rescored
   against current targets in B3.
2. **E and T measure different things.** E is oracle RWCR on eval seeds; T is
   causal coverage on training seeds. In urban_nlos the two differ by ~3×
   (T ≈ 0.27–0.29 vs E = 0.926 at d=20), because the causal grid estimator
   correlates only 0.19 with the oracle (A6 §5). A policy can satisfy T and
   miss E, and vice versa.
3. **E's ceiling set differs from F's**: 55 settings (no registry defaults) vs
   63. At rural d=2 clear fog_bank both give 0.67469 (the best setting is a
   knob setting), so the eval targets agree in the headline cell.
4. **Stale README table.** The "Measured headroom" table at the top of
   `README.md` (ceiling 0.689 at rural d=2, dvcast 0.70 / 0.90 at d=20, etc.)
   predates the fixed engine and the deadline guard; the committed values are
   those above and in `results/tables/table_reference.tex`.
5. "95% of each cell's ceiling" (Fig. 2) omits the deadline guard, which is
   part of Algorithm 1's qualification rule in the code.

### A2. Where 0.717 comes from

* 0.71681 = matched cost of the reference agent at rural d=20 clear fog_bank,
  τ = 0.5, sampled policy, from `results/agent_ref/sampled/agent_evaluation.json`.
* **What was swept:** the agent's `suppression_bias` ∈ {−2, −1, 0, 1, 2, 3}
  (`experiments/evaluate_agent.py: DEFAULT_BIASES`), 10 eval seeds each, gate on.
* **Computation** (`PolicyCurve.overhead_at`): q* = 0.878736, miss bound 0.363525.
  Settings reaching q* within the bound: bias −2 (RWCR 0.904851, cost 0.736182),
  −1 (0.8939, 0.7674), 0 (0.8909, 0.7690). Cheapest = **bias −2** (upper anchor).
  Most expensive setting that falls short and is cheaper: **bias +1**
  (RWCR 0.878182, cost 0.716404) (lower anchor).
  0.716404 + (0.878736 − 0.878182)/(0.904851 − 0.878182) × (0.736182 − 0.716404)
  = 0.716404 + 0.020768 × 0.019777 = **0.716815**.
  The anchors are chosen by cost, not knob adjacency: the agent's RWCR is not
  monotone in bias, so the interpolation spans biases +1 and −2. Bias +1 misses
  q* by 0.00055, so 0.717 is essentially the cost of bias +1.
* **Table 2's 0.85 / 0.79** are different quantities: single setting (bias 0),
  **pooled over all 4 weathers × 5 hazards × 10 seeds (200 runs)** at rural
  d=20 (`analysis/tables.py: main_comparison_table` filters only scenario and
  density). Unrounded: τ=0 0.85182, τ=0.5 0.78660. At clear/fog_bank alone,
  bias 0 gives 0.83007 (τ=0) and 0.76904 (τ=0.5) (`results/runs.csv`).
  Table 2's caption "10 seeds, mean ± std" is therefore misleading: the ± is
  over 200 runs spanning 20 conditions.

### A3. Table 2 RWCR spread

The code's "nine baselines" (`configs/experiment.yaml: sweep.policies`):
flooding, p_persistence_03, p_persistence_05, p_persistence_07, slotted_1p,
weighted_p, counter_based, greedy_farthest, dvcast. "p-persistence" in Table 2
is `p_persistence_03` (p = 0.3) and **is one of the nine**.

| view | all nine | without p_persistence_03 |
|---|---|---|
| Table 2 (pooled, 200 runs/scheme) | 0.87226 – 0.91680 (spread 0.04455) | 0.91505 – 0.91680 (0.00175) |
| clear/fog_bank only (10 runs/scheme) | 0.88928 – 0.92498 (0.03570) | 0.92139 – 0.92498 (0.00360) |

"Nine baselines span 0.915–0.917" is true only for **eight** (excluding the
default p-persistence).

### A4. Miss rates

* Table 2's "Miss rate" column is `actionable_deadline_miss_rate`, pooled over
  the 200 rural d=20 runs: 0.3142–0.3160 for eight baselines, 0.3505 for
  p_persistence_03, AI-HARP 0.3517 (τ=0) / 0.3404 (τ=0.5).
* **§IV-A's 0.240 (unrestricted) vs 0.102 (actionable) cannot be reproduced
  from any committed artefact.** No per-run row, no cell mean, and no pooling
  level (by density, weather, hazard, policy, or combinations) in
  `results/runs.csv` gives that pair, and it is not in git history. Under the
  current metric (metrics_version 3) the two rates are nearly identical on the
  rural corridor: mean per-run gap 0.0074; largest per-run gap 0.270. Rural
  baselines by density, unrestricted / actionable: d=2 0.673/0.672, d=20
  0.327/0.319, d=80 0.363/0.329. The §IV-A pair most likely predates the
  oracle at-risk set (metrics_version < 3); it should be re-derived or removed.

### A5. λ multiplier: peak and plateau per group

Source: `checkpoints/{run8,feas_A_cap500,feas_B_init500,campaign/ref}/history.jsonl`
(git-ignored; local only), `lambda_next_by_group`. Plateau = mean over the
last 100 updates.

| group | run8 (cap 50) peak / plateau | arm A (cap 500, climbs from 2) peak / plateau | arm B (init 500) peak / plateau | campaign ref (cap 500, 32-seed targets) peak / plateau |
|---|---|---|---|---|
| rural d=2 | 50.0 / 49.99 | 49.70 (at its last update) / 45.04 | 500 / 499.95 | **33.18 (at update 1000)** / 31.84 |
| rural d=3 | 49.69 / 48.34 | 33.20 (last update) / 30.20 | 500 / 499.86 | 22.93 / 21.36 |
| urban d=2 | 50.0 / 49.97 | 48.66 / 41.79 | 500 / 499.89 | 29.52 (update 999) / 27.50 |
| urban d=1 | 2.59 / 0.42 | 3.41 / 0.25 | 499.62 / 327.97 | 31.89 / 28.21 |
| rural d=5 | 9.83 / 5.50 | 7.59 / 5.85 | 500 / 480.38 | 12.66 / 10.69 |

Updates with any group saturated: run8 357 / 1000, arm A 0 / 500, arm B 522 / 800,
campaign ref 0 / 1000. Full per-group table in `stage_a_facts.json`.

**The truth:** "plateaus near 50" is run8 (pinned at its cap of 50) — and arm A,
which reached 49.7 but was *still climbing* when it stopped at 300 finetune
updates. "Peak 33.1, never reached [the cap]" is the campaign reference run
(cap 500, 32-seed targets), which is what Fig. 12 plots. Both are correct for
different runs; the text and figure must name which run. Note that the
reference's rural d=2 λ peaks at its **final** update and urban d=2 at update
999: it is still rising, so "interior equilibrium" (README) is not supported —
"not cap-limited, still rising at the 1,000-update budget" is.

### A6. Facts from code

See `results/revision1/A6_facts.md`. Headlines: the ensemble gate is **never
used** (all results use the entropy gate; `ensemble_size: 5` is read by no
code); heads are **averaged** in the final layer to form the relay ranking;
610 m vs 480 m comes from path-loss exponents (1.9/3.8 vs 2.0/4.2), and the
"562 m" in `phy.yaml`/README is stale; Table 3's correlation is a **mean over
all 8 densities**; a code "cell" is scenario × density × weather × hazard
(480 cells; "24 cells" = scenario × density); there are **4** weathers (2
training, 2 held out); "26" is the family-size threshold above which Holm
cannot reject at 10 seeds (0.00195 × 26 > 0.05).

### A7. Validation units and the Amador citation

* **Citation** (verified on arXiv 2403.05994): O. Amador, M. Urueña,
  M. Calderon, I. Soto, "Evaluation and improvement of ETSI ITS
  Contention-Based Forwarding (CBF) of warning messages in highway scenarios",
  *Vehicular Communications* 34 (2022) 100454,
  DOI **10.1016/j.vehcom.2022.100454** (arXiv DOI 10.48550/arXiv.2403.05994).
  `experiments/validate_amador.py` carries authors/title/venue but no DOI.
* **Their scenario:** 5 km straight highway, 8 lanes (4 per direction), Artery
  (Veins/OMNeT++, Vanetza, SUMO), DENM warnings disseminated with **ETSI
  GeoNetworking Contention-Based Forwarding (CBF)** (they also propose DPD,
  GPC, FoT variants; the "ETSI" column is the reference). Table 3 header:
  "Density (veh/km lane)"; densities 10–50 **vehicles/km per lane**.
* **Units:** `validate_amador.py` uses the paper's per-lane values directly.
  `analysis/report.py:470` labels Fig. 13's x-axis "Vehicle density (veh/km)"
  — **wrong; it is veh/km/lane**. Table 6 should say the same.
* **Mapping to our corridor.** Ours is two lanes (one per direction), so
  20 veh/km/lane = **40 veh/km** of road; Amador's 10 veh/km/lane = 80 veh/km.

  | basis | policy's qualifying cell (rural d=20) maps to | deviation there |
  |---|---|---|
  | same per-lane density | Amador 20 veh/km/lane | **+0.0019** (0.9980 vs 0.9961) |
  | same vehicles per km of road (40 veh/km) | Amador 5 veh/km/lane | **outside the validated range** (lowest point 10 veh/km/lane: −0.0021) |
  | same mean neighbours in range | ours 40 veh/km × 2 × 610 m ≈ 49; Amador at 10/lane: 80 × 2 × 778 m ≈ 1,245 | outside the validated range by ~25× |

  Per-lane matching is what the current figure implicitly does; on either
  physical basis (road density or neighbour count) the qualifying cell lies
  below the densest-connected regime Amador validated, so the validation
  supports the simulator's dense-regime PDR level, not the policy's cell.

### A8. Regret and margin at rural d=20, unrounded

From `results/agent_ref/sampled/agent_evaluation.json`: agent a = 0.7168148,
per-cell oracle o = 0.7038312 (greedy_farthest), best fixed f = 0.7304127
(dvcast n_slots=30).

| quantity | formula (code) | value |
|---|---|---|
| regret | a/o − 1 | **1.8447%** |
| margin | f/a − 1 | **1.8970%** |
| alt. regret | (a − o)/a | 1.8113% |
| alt. margin | (f − a)/f | 1.8617% |
| reviewer's 0.013/0.730 | (a − o)/f, rounded | 1.78% |

The paper's 1.8% / 1.9% match the code's definitions (a/o − 1, f/a − 1).
The reviewer's 1.78% divides a − o = 0.01298 by the *fixed* cost 0.730, which
mixes the two references. The paper should state the formulas.


---

## Plan change (user, 2026-10-16 evening)

C1 = **2 extra full-model seeds** (3 in total with committed seed 0); C2 = **1 extra seed
per ablation** (GCN, MLP, star; 2 in total each); **C5 skipped**; agent evaluations of
the **final checkpoint only**; Stage B headline cells first, 24-cell versions after;
E1 = headline-number smoke test only; deterministic kernels kept.

---

## Stage B — headline cells (rural d=2/20/80, urban_nlos d=20; clear, fog_bank)

All new per-seed data comes from `experiments/revision1/runner.py` (one `run_single`
call per row — the committed paired path; every row carries config hash,
metrics_version, git commit and seed). **Reproduction checks:** the per-seed re-run of
the reference agent matches every committed agent point
(`results/agent_ref/{sampled,argmax}/pareto_cells_agent.json`) to ≤ 4.4e-16 (48 points,
B5); B1 recomputed from per-seed rows gives the committed 0.7168 / +1.84% / +1.90% exactly.

### B1. Latency-bounded matched coverage (headline cells)

Regenerate: `python -m experiments.revision1.b1_latency` (committed curves) →
`results/revision1/B1/b1_headline.{json,md}`; both gate arms from per-seed rows →
`results/revision1/B1/b1_headline_perseed_tau0_tau05.json`. Figure (D4):
`results/revision1/D/fig_d4_latency_bounded_headline.{pdf,png}`.

Rule: Algorithm 1 plus "seed-mean of per-run median TIR ≤ D"; the bound also applies
to the interpolation's lower anchor. The primary variant holds q* and the miss bound at
their unbounded values ("fixed"); "rebased" (q* recomputed among latency-feasible
settings) is in the JSON and changes nothing below except the best fixed setting at
D = 1.5 s (`slotted_1p(n_slots=12)`).

**Derived deadline** D_r = (R − v·t_r − v²/2a)/v: a vehicle one radio hop from a
holder, approaching at the scenario's maximum car speed v, can still react and stop if
the next hop arrives within D_r. t_r = 2.5 s (configs/hazard.yaml), a = 3.4 m/s² (AASHTO
comfortable deceleration; **assumed**), R = nominal range, or for urban_nlos the
blocked-corner range:

| scenario | v (m/s) | R (m) | stopping distance (m) | D_r |
|---|---|---|---|---|
| rural_highway | 27.78 | 610.4 (LOS) | 182.9 | **15.39 s** |
| urban_grid | 15.28 | 479.6 (LOS) | 72.5 | 26.65 s |
| urban_nlos | 15.28 | 101.4 (NLOS corner) | 72.5 | **1.89 s** |

Reference points per bound (q* fixed; cost = transmissions per at-risk vehicle informed):

| D | best single fixed | rural d=2 oracle / fixed | rural d=20 oracle / fixed | rural d=80 oracle / fixed | urban d=20 oracle / fixed |
|---|---|---|---|---|---|
| none | dvcast(n_slots=30) | dvcast 1.560 / 1.560 | greedy 0.704 / 0.730 | weighted_p 1.512 / 1.703 | counter 0.533 / 0.567 |
| 2.0 s | slotted_1p(n_slots=20) | slotted 1.654 / 1.688 | greedy 0.704 / 0.768 | weighted_p 1.512 / 1.837 | counter 0.533 / 0.578 |
| 1.5 s | slotted_1p(n_slots=8) | slotted 1.654 / 1.654 | greedy 0.704 / 0.898 | weighted_p 1.512 / 2.282 | counter 0.533 / 0.640 |
| 1.0 s | slotted_1p(n_slots=8) | slotted 1.654 / 1.654 | greedy 0.704 / 0.898 | weighted_p 1.512 / 2.282 | counter 0.533 / 0.640 |
| 0.5 s | slotted_1p(n_slots=3) | slotted 1.755 / 1.755 | **dvcast 0.730** / 1.341 | weighted_p 1.512 / 3.979 | p_persist 0.550 / 0.832 |
| derived | slotted_1p(n_slots=20) | dvcast 1.560 / 1.688 | greedy 0.704 / 0.768 | weighted_p 1.512 / 1.837 | counter 0.533 / 0.578 |

Agent (reference checkpoint, sampled, eval seeds 0–9). Rural d=2 and d=80 are never
reached by either arm under any bound:

| D | τ=0.5 rural d=20: cost / regret / margin | τ=0 rural d=20 | τ=0 urban d=20 | cells matched τ=0.5 / τ=0 |
|---|---|---|---|---|
| none | 0.717 / +1.8% / +1.9% | 0.706 / +0.3% / +3.5% | 0.864 / +62.1% / −34.4% | 1 / 2 |
| 2.0 s | 0.717 / +1.8% / +7.1% | 0.706 / +0.3% / +8.8% | 0.864 / +62.1% / −33.1% | 1 / 2 |
| 1.5 s, 1.0 s | 0.717 / +1.8% / +25.3% | 0.706 / +0.3% / +27.3% | 0.864 / +62.1% / −26.0% | 1 / 2 |
| 0.5 s | 0.717 / **−1.9%** / **+87.1%** | 0.706 / −3.4% / +90.1% | 0.864 / +57.1% / −3.7% | 1 / 2 |
| derived | 0.717 / +1.8% / +7.1% | 0.706 / +0.3% / +8.8% | 0.864 / +62.1% / −33.1% | 1 / 2 |

**Do the slot-scheme reference points survive?** DV-CAST's 0.730 (the best fixed
setting `dvcast(n_slots=30)` at rural d=20, median TIR 0.46 s) survives every bound and
becomes the per-cell oracle at D = 0.5 s. Greedy-farthest's 0.704 (`fallback_n_slots=32`,
median TIR 0.915 s) survives D ≥ 1 s and is **eliminated at D = 0.5 s** (no greedy setting
reaches q* with TIR ≤ 0.5 s). The bound mainly changes the **best single fixed** scheme:
DV-CAST(30) waits 2–3 s at rural d=2 and urban d=20, so for D ≤ 2 s it stops being
deployable everywhere and the fixed reference moves to ever-shorter `slotted_1p`,
which costs much more elsewhere. The agent's median TIR (0.31–0.32 s at rural d=20)
meets every bound, so its cost never changes and its margin grows from +1.9% (none) to
+87% (D = 0.5 s); regret against per-cell tuning stays ≤ +1.8%. It still reaches the
target in only 1 (τ=0.5) or 2 (τ=0) of 4 cells under every bound. **New:** the
network-only arm (τ=0), never scored with Algorithm 1 before, matches 2 cells, and at
rural d=20 is cheaper than τ=0.5 (0.706 vs 0.717). The gate's fallback is what loses
urban d=20 at τ=0.5. The 24-cell version is running.

### B2. Slot grid beyond 50

Regenerate: `python -m experiments.revision1.b2_slots` → `results/revision1/B2/b2_slots.{csv,json}`,
`fig_b2_slots.{pdf,png}`. Extended runs: `results/revision1/B_sweeps/slots_runs.csv`
(n_slots 75/100/150/200 for slotted_1p and DV-CAST; fallback_n_slots 64/128 for greedy;
10 seeds). Cost / median TIR:

| cell | scheme | 50 (greedy 32) | 75 (64) | 100 (128) | 150 | 200 |
|---|---|---|---|---|---|---|
| rural d=20 | dvcast | 0.716 / 0.56 s | 0.712 / 0.67 s | **0.705** / 0.99 s | 0.710 / 0.85 s | 0.722 / 1.02 s |
| rural d=20 | slotted_1p | 0.726 / 0.51 s | **0.703** / 0.67 s | 0.730 / 0.77 s | 0.713 / 0.92 s | 0.713 / 1.09 s |
| rural d=20 | greedy | 0.704 / 0.92 s | 0.712 / 1.03 s | **0.702** / 1.06 s | – | – |
| rural d=80 | dvcast | 1.658 / 0.28 s | 1.677 | 1.634 | 1.663 | 1.625 / 0.29 s |
| urban d=20 | dvcast | 0.551 / 3.07 s | 0.568 / 4.17 s | 0.571 / 5.75 s | 0.603 / 9.58 s | 0.617 / 12.8 s |
| rural d=2 | dvcast | 1.446 / 3.37 s | 1.481 / 3.86 s | 1.697 / 5.19 s | 1.391 / 9.33 s (RWCR 0.575) | 1.312 / 5.96 s (RWCR 0.514) |

**Cost stops falling at about 50–100 slots while latency keeps rising.** At rural d=20 the
best extended point (0.702–0.705) is 0.3% below the committed oracle 0.704 at roughly
twice the median TIR. At rural d=80 cost drifts by < 2%. At urban d=20 cost **rises** past
50 slots (0.551 → 0.617) while RWCR and the miss rate degrade. At rural d=2 RWCR
collapses past 100 slots (0.68 → 0.51). Every extended urban and rural-d=2 point fails the
deadline guard, so **Algorithm 1's guarded oracle is unchanged in 3 of 4 cells**. At rural
d=20 (loose guard) the extension would lower the oracle from 0.704 to 0.702 (greedy, 128
slots, TIR 1.06 s), raising the agent's regret from +1.8% to +2.1%.

### B3. Feasibility under every target definition

Regenerate: `python -m experiments.revision1.sweeps headline` and `... sweeps train`, then
`python -m experiments.revision1.b3_feasibility` → `results/revision1/B3/b3_feasibility.json`.
Inputs: 63 settings (55 knob + 8 registry defaults) per headline cell. Eval side:
`results/revision1/B_sweeps/headline_runs.csv` (oracle RWCR, seeds 0–9). Training side:
`results/revision1/B_sweeps/train_runs.csv` (causal coverage, seeds 100–131; rural d=2
from the committed `results/feasibility/sparse_feasibility_runs.csv`). Definitions
(A1): E = Algorithm 1 (with and without deadline guard); F63 = 0.95 × best of 63;
T32 = current 32-seed training target; T2 = stale 2-seed target the §VII-F sweep used
(`git show a378bcf:results/coverage_targets.json`). Envelope = best setting chosen
separately for every seed; shortfall = (Σ target − Σ envelope)/Σ target (negative = reachable).

| cell | def. | q* | fixed settings meeting q* | best fixed (mean) | hindsight envelope (shortfall) |
|---|---|---|---|---|---|
| rural d=2 | E | 0.6410 | 14 / 63 (11 with guard) | dvcast(50) 0.6747 | 0.8054 (−0.257) |
| | F63 | 0.6410 | 14 / 63 | dvcast(50) 0.6747 | 0.8054 (−0.257) |
| | T32 | 0.7201 | **12 / 63** | dvcast(5) 0.7580 | 0.8970 (−0.246) |
| | T2 (stale) | 0.8640 | **0 / 63** | dvcast(5) 0.7580 | 0.8970 (−0.038) |
| rural d=20 | E / F63 | 0.8787 | 49 / 63 | p_persist(0.9) 0.9250 | 0.9267 (−0.055) |
| | T32 / T2 | 0.9256 / 0.9280 | 48 / 63 | counter(6) 0.9747 | 0.9766 (−0.055 / −0.052) |
| rural d=80 | E / F63 | 0.8695 | 53 / 63 | slotted(3) 0.9153 | 0.9156 (−0.053) |
| | T32 / T2 | 0.9484 / 0.9481 | 55 / 63 | counter(6) 0.9999 | 0.9999 (−0.054) |
| urban d=20 | E | 0.9261 | 47 / 63 (45 with guard) | counter(4) 0.9749 | 0.9803 (−0.059) |
| | T32 | 0.2656 | 41 / 63 | slotted(50) 0.3944 | 0.4020 (−0.514) |
| | T2 (stale) | 0.2918 | 5 / 63 | slotted(50) 0.3944 | 0.4020 (−0.378) |

**Classification of every agent miss** (reference agent, τ=0.5, sampled):

| cell | eval side (E, F63): agent best RWCR vs q* | class | training side (T32): agent group shortfall, last 200 updates | class |
|---|---|---|---|---|
| rural d=2 | 0.507 vs 0.641 | **policy failure** (14/63 fixed settings reach it) | +0.030 (short) | **policy failure** (12/63 reach the clear/fog_bank cell's target) |
| rural d=20 | 0.905 vs 0.879 | reaches | +0.026 (short) | policy failure (48/63) |
| rural d=80 | 0.846 vs 0.870 | **policy failure** (53/63) | −0.0002 (met) | — |
| urban d=20 | 0.921 vs 0.926 | **policy failure** (47/63) | −0.008 (met) | — |

**No agent miss is "target unreachable" under any definition now in force.** The only
definition under which no fixed setting qualifies is the stale 2-seed rural d=2 target
(0.864): that is the §VII-F "0 of 63" result, and it does not describe the target the
agent was trained or evaluated against. Even there the per-seed hindsight envelope
reaches the target (−0.038). Under the current target, 12 of 63 fixed settings meet it.
Caveats: the training-side agent number is a group mean over 6 weather × hazard cells,
and feasibility is counted on the clear/fog_bank cell. A bug in a first version of this
script (urban rows overwrote rural ones at rural d=2) was caught by cross-checking
flooding's coverage against `coverage_targets.json` (0.7517 in both) and fixed before
these numbers.

### B4. Statistics Section VI promises (headline cells)

Regenerate: `python -m experiments.revision1.b4_stats wilcoxon` →
`results/revision1/B4/b4_wilcoxon_sampled.{json,csv}`; `... b4_stats bootstrap` →
`results/revision1/B4/b4_bootstrap.{json,csv}`.

Method: Wilcoxon signed-rank (`analysis.stats.paired_test`, as in `analysis/grid_stats.py`),
10 paired eval seeds, Holm across the 4 metrics within each (cell, comparison),
matched-pairs rank-biserial r. The agent is taken at its cheapest qualifying bias
(Algorithm 1's upper anchor) or, where it never qualifies, its max-RWCR bias.
References are at their qualifying knob values. Each cell shows median paired
difference (agent − reference), r, Holm p; * = p_Holm < 0.05.

τ = 0.5:

| cell (agent bias) | vs | RWCR | cost | median TIR | miss |
|---|---|---|---|---|---|
| rural d=2 (+1, misses) | dvcast(30) (= fixed = oracle) | −0.146, −0.60, 0.32 | +0.286, +0.31, 0.46 | −2.15 s, −1.00, 0.031* | +0.043, +0.45, 0.46 |
| rural d=20 (−2, reaches) | fixed dvcast(30) | −0.017, −0.96, 0.039* | +0.004, +0.09, 0.85 | −0.10 s, −1.00, 0.039* | +0.014, +0.93, 0.039* |
| rural d=20 | oracle greedy(32) | −0.014, −0.75, 0.074 | +0.016, +0.45, 0.23 | −0.60 s, −1.00, 0.0078* | +0.015, +0.87, 0.063 |
| rural d=80 (−1, misses) | fixed dvcast(30) | −0.063, −1.00, 0.0078* | −0.978, −1.00, 0.0078* | −0.075 s, −1.00, 0.020* | +0.048, +1.00, 0.0078* |
| rural d=80 | oracle weighted_p(0.15) | −0.048, −1.00, 0.0078* | −0.741, −1.00, 0.0078* | −0.10 s, −1.00, 0.014* | +0.041, +1.00, 0.0078* |
| urban d=20 (−1, misses) | fixed dvcast(30) | −0.042, −0.96, 0.0078* | +0.101, +1.00, 0.0078* | −1.75 s, −1.00, 0.0078* | +0.040, +1.00, 0.0078* |
| urban d=20 | oracle counter(1) | −0.044, −1.00, 0.0078* | +0.143, +1.00, 0.0078* | −0.30 s, −1.00, 0.0078* | +0.041, +1.00, 0.0078* |

τ = 0 is in the CSV with the same pattern: the agent is significantly faster in every
cell except rural d=80 (TIR tie), has significantly lower RWCR and a higher miss rate in
every cell, and is cheaper only at rural d=80, where it misses the target. **In the one
cell it qualifies at τ=0.5 (rural d=20), its cost does not differ significantly from
either reference (p_Holm 0.85 / 0.23): the 1.8% / 1.9% are not statistically resolved at
10 seeds.**

**Bootstrap between baselines at rural d=20** (replaces the "0.002 vs per-seed SD 0.030"
argument; 10,000 resamples, percentile 95% CIs). Per-scheme SD across seeds is 0.022–0.026
(clear/fog_bank) and 0.029–0.030 (pooled), but **the SD of paired differences is only
0.003–0.007** (0.10–0.11 for pairs with p_persistence_03), because seeds are paired.

| view | pairs whose CI excludes 0 |
|---|---|
| clear/fog_bank, n = 10 seeds | 1 / 36 (dvcast − weighted_p = −0.0036, CI [−0.0064, −0.0009]) |
| Table 2 pooled, n = 200 (weather × hazard × seed), i.i.d. bootstrap | 19 / 36 |
| Table 2 pooled, **seed-cluster** bootstrap (the 200 runs share 10 mobility seeds) | **12 / 36**: all 8 vs p_persistence_03 (+0.043 … +0.045), and counter_based over dvcast (+0.0018, CI [0.0003, 0.0034]), slotted_1p (+0.0015), greedy (+0.0013), weighted_p (+0.0012) |

So at the headline cell the eight non-default baselines are statistically
indistinguishable, but in Table 2's pooled view counter_based is resolvably, if
negligibly, ahead of four others. "Differences are below seed noise" is the wrong
argument; the right one is that the paired differences are ≤ 0.002 in absolute size.

### B5. Sampled vs argmax, all 4 headline cells (reference agent, τ = 0.5)

Regenerate: `python -m experiments.revision1.b5_modes` → `results/revision1/B5/b5_modes.json`.

| cell | q* | best RWCR sampled (bias) | best RWCR argmax (bias) | gap | sampled: matched cost / regret / margin | argmax |
|---|---|---|---|---|---|---|
| rural d=2 | 0.641 | 0.507 (+1) | 0.477 (+3) | **+0.030** | never | never |
| rural d=20 | 0.879 | 0.905 (−2) | 0.904 (−1) | **+0.001** | 0.717 / +1.8% / +1.9% | 0.686 / −2.5% / +6.4% |
| rural d=80 | 0.870 | 0.846 (−1) | 0.825 (−2) | **+0.021** | never | never |
| urban d=20 | 0.926 | 0.921 (−1) | 0.927 (−2) | **−0.006** | never | 0.702 / +31.8% / −19.3% |

Per-bias paired RWCR differences with Wilcoxon p are in the JSON. **The paper's "0.172
RWCR gap in the dense rural cell" is run8's number** (README: run8 argmax 0.642 vs
sampled 0.814 at rural d=80). For the reference agent the paper otherwise reports, the
rural d=80 gap is 0.021. Mode decides whether urban d=20 qualifies (argmax yes, sampled no).

---

## Stage D (partial)

Regenerate: `python -m experiments.revision1.stage_d` → `results/revision1/D/`. Paper
figures are mapped to repo figures by content, since the numbering differs: **Fig. 8 =
`fig03_regret_margin`** (the only bar chart with 3 of 4 cells empty and slanted x
labels), Fig. 12 = `fig06_training`, Fig. 13 = `fig10_validation`, Fig. 14 =
`fig09_latency_cost_inversion`, Fig. 15 = `figS3_weather_*`. If Fig. 8 is the ablation
chart instead, its table is `results/tables/table_ablation.tex` (regenerated with C2).

* **D1** `table_fig8_regret_margin.{csv,tex}`: Fig. 8's content as a table (cost and TIR
  axes; sampled and argmax in the CSV).
* **D2** `fig08_paper_regret_margin_fixed`: horizontal two-line x labels, nothing
  clipped. `fig14_paper_latency_cost_fixed`: labels placed by a collision check, so 4 and
  5 no longer overlap. A bug found while fixing it: **the old Fig. 14 averaged the τ=0
  and τ=0.5 agent rows into a single "AI-HARP" point** (cost 0.82). The two arms are now
  separate points (0.79 and 0.85, matching Table 2). The fixes are in `analysis/figures.py`,
  so `reproduce.sh` regenerates the corrected figures.
* **D3** `fig12_paper_training_fixed`: the annotation now reads "cap 500 never reached:
  peak λ = 33.1 (6.6% of cap; rural highway d=2, update 994)". The figure plots the λ used
  at each update; A5's 33.18 at update 1000 is the post-update value. Neither run came near
  the cap. "Plateaus near 50" belongs to run8 / arm A and must not be attached to this figure.
* **D4** `fig_d4_latency_bounded_headline`: oracle, best fixed and agent cost against D.


---

## C3. Learning vs fallback (reference checkpoint, headline cells)

Regenerate: `python -m experiments.revision1.agent_eval --checkpoint checkpoints/campaign/ref/ckpt_final.pt --name ref_s0_randomgate --random-gate-from results/revision1/agent_runs/ref_s0.csv`
then `python -m experiments.revision1.c3_fallback` → `results/revision1/C3/c3_fallback.json`.
Rows: `results/revision1/agent_runs/ref_s0{,_randomgate}.csv`. Each variant is scored
alone against the 55 baseline settings (Algorithm 1, deadline guard, eval seeds 0–9).
Code: `agents/ai_harp.py` `random_gate_rate` replaces the confidence test with a
coin (private RNG) whose fallback probability is the learned gate's measured rate in
that (cell, bias). Measured rates are reproduced to ±0.04: rural d=2 9–56%, rural d=20
4–22%, rural d=80 6–16%, urban d=20 1–35% (by bias).

Matched cost / regret vs per-cell oracle / margin vs best fixed (`dvcast(n_slots=30)`):

| variant | rural d=2 | rural d=20 | rural d=80 | urban d=20 | cells matched |
|---|---|---|---|---|---|
| learned gate (τ=0.5) | never (best RWCR 0.507) | **0.717 / +1.8% / +1.9%** | never (0.846) | never (0.921) | **1** |
| network only (τ=0) | never (0.551) | 0.706 / +0.3% / +3.5% | never (0.851) | 0.864 / +62.1% / −34.4% | **2** |
| random gate, same rate | never (0.480) | 0.782 / +11.1% / −6.6% | **1.200 / −20.6% / +41.9%** | 0.807 / +51.4% / −29.8% | **3** |
| pure weighted_p, tuned curve | never | 0.926 / +31.5% | 1.512 (it *is* the oracle) | 0.628 / +17.8% | 2 of 3 non-trivial |
| pure weighted_p, default | RWCR 0.420 (misses) | 1.436 (meets) | 3.578 (meets) | 0.653 (meets) | 3 |

What this says:

1. **The learned confidence gate is not what earns the result.** A coin that defers
   the same *fraction* of decisions matches more cells (3 vs 1) and is cheaper than
   every baseline at rural d=80, where the learned gate never reaches the target.
   The entropy gate defers the decisions the network is unsure about, and those are
   evidently not the ones whose deferral helps coverage. Only at rural d=20 does the
   learned gate choose better than chance (0.717 vs 0.782).
2. **The network alone (τ=0) beats the gated agent** in the paper's qualifying cell
   (0.706 vs 0.717) and qualifies in one more cell (urban d=20, though at +62% regret).
   At τ=0.5 the fallback to weighted_p is what loses urban d=20 (README's weighted_p
   die-out mechanism).
3. **The fallback alone is far worse than the agent at rural d=20** (tuned weighted_p
   0.926, +31.5% regret), so the 0.717 there is learned behaviour, not fallback.
4. Caveat: one training seed, 10 eval seeds; C1 adds seeds 1–2 for the learned-gate
   and network-only arms (random gate is not repeated per seed).


---

## E1. Reproduction from a fresh clone (headline numbers)

Fresh `git clone --branch revision-1` from GitHub into an empty directory: no
`cache/` (mobility traces regenerated from scratch) and no `checkpoints/`.
Environment: the rebuilt `aiharp-gpu` (torch 2.2.2+cu121, NumPy 1.26.4).
Artefacts: `results/revision1/E1/`.

| check | result |
|---|---|
| `analysis.comparator --reuse` reference table | identical: ceilings 0.675 / 0.925 / 0.915 / 0.975; oracle dvcast 1.56, greedy 0.70, weighted_p 1.51, counter 0.53; fixed dvcast(30) 1.56 / 0.73 / 1.70 / 0.57 |
| re-simulate every headline reference setting (oracle k* + fixed, 4 cells × 10 seeds, empty trace cache): `python -m experiments.revision1.e1_headline` | **max abs diff 1.1e-16** vs `results/pareto_cells.json` (`e1_headline.json`) |
| re-evaluate the two agent anchors behind 0.717 (bias −2 and +1, τ=0.5, sampled; checkpoint copied in) | rwcr 0.904851 / 0.878182, cost 0.736182 / 0.716404: **identical** |
| all 220 headline baseline points (55 settings × 4 cells), re-simulated per seed | max abs diff **3.6e-15** (RWCR, cost, median TIR, miss) |
| all 48 committed agent points (τ=0.5, sampled + argmax); GCN and MLP seed-0 curves (24 points each) | max abs diff ≤ 4.4e-16 |
| test suite | 566 passed |
| `reproduce.sh --smoke --no-train`, as committed | **failed** at step 5b: `checkpoint not found: checkpoints/reproduce_smoke/ckpt_final.pt` (exit 1); and step 4 had **overwritten the committed `results/pareto_cells.json` with 2-seed curves** (`reproduce_smoke_original.txt`) |
| same, after fixing `reproduce.sh` | **exit 0**; `results/pareto_cells.json` untouched (smoke curves go to `results/smoke/`); agent steps skipped with a message (`reproduce_smoke_patched.txt`) |
| figures rebuilt by step 6 | pixel-identical to the committed PNGs except `fig03` and `fig09`, which carry the D2 fixes; `fig06` (paper Fig. 12) is **not rebuilt on a clone** because `checkpoints/campaign/ref/history.jsonl` is git-ignored |

What does not regenerate from a clone as committed: **every agent number and Fig. 12**,
because `checkpoints/` (weights and training histories) is git-ignored. Fixed for this
revision by committing copies of the four seed-0 final checkpoints (0.3–0.5 MB each)
and their training histories (1.5 MB each) under `results/revision1/checkpoints/`, with
SHAs in `SHA.json` (`ref_s0` = `b589be512827fcbb`, the SHA the committed evaluation
records). Not run in E1 (user scope: headline numbers only): the full 52,800-run grid,
training, the SUMO/OSM re-runs (SUMO is not installed on this machine any more), and
the Amador validation.

## E2. Public readiness

* **Secrets:** none found (`git grep` for key/token/password/private-key patterns:
  only code comments about tokens).
* **Absolute local paths in tracked files:**
  * usage docstrings and `reproduce.sh` defaults name `D:/aiharp-env/python.exe` /
    `D:/aiharp-gpu/...` (documented local interpreters; `reproduce.sh` takes
    `AIHARP_PYTHON` / `AIHARP_TRAIN_PYTHON` and fails with a clear message if the
    default is missing). ENVIRONMENT.md describes environments that **no longer exist on
    this machine**; the build recipe in it was followed verbatim to rebuild `aiharp-gpu`,
    with two extra pins it lacks: `contourpy<1.3` and `pandas<3` (both otherwise resolve to
    NumPy-2-only / breaking versions);
  * committed results that embed `C:\Users\UPASANA\...`: `results/profile_*.txt` (25
    lines each, cProfile paths), `results/agent_{ref,gcn,mlp,star}/*/agent_evaluation.json`
    (checkpoint path), `results/attention_event.json`. Left as they are (rule: never
    overwrite existing results); they leak only a Windows user name;
  * revision-1 outputs: none (provenance records the environment name, not a path).
* **README:** documents every figure's source; it now also lists every table's source,
  states that checkpoints are not committed and what `--no-train` does on a clone, and
  points to `--revision1`. The previous line "`--no-train` reuse the committed
  checkpoint" was wrong (fixed in README and `reproduce.sh`).
* **Large files:** largest revision-1 file is `results/revision1/B_sweeps/grid24_runs.csv`
  (~5 MB when complete); nothing over 50 MB is committed.


---

## C1/C2. Training seeds — INTERIM (training seed 0 only; seeds 1–2 training)

Regenerate: per-seed evals `python -m experiments.revision1.agent_eval --checkpoint <ckpt_final.pt> --name <variant>_s<seed> --taus 0.5 0`, then
`python -m experiments.revision1.c_seeds` → `results/revision1/C1C2/c_seeds.json`. Rows:
`results/revision1/agent_runs/{ref,gcn,mlp,star}_s0.csv`; the τ=0.5 curves reproduce the committed
`results/agent_<variant>/sampled/pareto_cells_agent.json` to ≤ 4.4e-16 for all four variants.
τ=0 had never been evaluated for the ablations before.

Matched cost per cell (Algorithm 1, unbounded; "–" = never reaches q*), seed 0:

| variant | τ | rural d=2 | rural d=20 | rural d=80 | urban d=20 | cells matched |
|---|---|---|---|---|---|---|
| ref (GATv2) | 0.5 | – | 0.717 | – | – | 1 |
| GCN | 0.5 | – | 0.681 | 0.991 | 0.782 | 3 |
| MLP | 0.5 | – | 1.149 | 0.708 | 0.902 | 3 |
| star | 0.5 | – | 0.699 | – | 0.722 | 2 |
| ref (GATv2) | 0 | – | 0.706 | – | 0.864 | 2 |
| GCN | 0 | – | 0.759 | 0.833 | 0.731 | 3 |
| MLP | 0 | – | 1.283 | 0.832 | 1.000 | 3 |
| star | 0 | – | 0.991 | – | 0.869 | 2 |

**Same cells for every variant** (the reviewer's request). At τ=0.5 the only cell every
variant matches is rural d=20; at τ=0 it is rural d=20 and urban d=20. Mean regret / margin
over those common cells:

| variant | τ=0.5, common = {rural d=20} | τ=0, common = {rural d=20, urban d=20} | committed table (each variant's own matched cells, τ=0.5): regret / margin |
|---|---|---|---|
| ref | +1.8% / +1.9% | +31.2% / −15.4% | +1.8% / +1.9% (1 cell) |
| GCN | **−3.2% / +7.2%** | +22.5% / −13.1% | +3.1% / +17.2% (3 cells) |
| MLP | **+63.3% / −36.4%** | +85.0% / −43.2% | +26.5% / **+22.3%** (3 cells) |
| star | −0.7% / +4.5% | +51.9% / −30.5% | +17.4% / −8.5% (2 cells) |

**The committed ablation table's ranking is an artefact of averaging over different
cells.** MLP has "the best margin" (+22.3%) only because its average includes rural d=80,
where it is cheap. On the one cell every variant reaches, MLP is the most expensive by
far (1.149, 63% above per-cell tuning, 36% dearer than the best fixed baseline). On that
common cell GCN is the cheapest and the only variant below per-cell tuning. With default
settings (bias 0) averaged over all 4 cells (defined for every variant): RWCR / cost / TIR
at τ=0.5 are ref 0.752 / 0.976 / 0.274 s, GCN 0.804 / 1.018 / 0.300 s, MLP 0.817 / 1.274 / 0.309 s,
star 0.764 / 1.071 / 0.281 s. One training seed per variant: this table will be replaced by
mean ± SD over 3 seeds (ref) and 2 seeds (ablations) when training finishes.

---

## Run log: overnight interruption (2026-10-16 22:59 → 2026-10-17 09:35)

Every job was killed at 22:59 when the interactive session that had launched them
ended (both training runs logged `BrokenProcessPool`, `crash.json` at updates 251 / 278).
About 10.5 h were lost. Recovery:
* ref_s1 / ref_s2 resumed exactly from `ckpt_latest.pt` (updates 250 / 277; the repo's
  resume restores model, optimiser, multipliers, both RNG streams and stage position).
  Histories are contiguous. Each of these two runs therefore has one resume point, recorded here.
* Long jobs now run under Windows Task Scheduler (`experiments/revision1/launch_revision1.cmd`,
  task `AIHARP_revision1`), outside any interactive session.
* **An error of mine during recovery:** a truncation check that counted commas
  misread the quoted-JSON column and rewrote seven per-seed CSVs with headers only. Six
  were committed and were restored byte-for-byte from git (re-verified: every row parses,
  row counts as committed). The seventh, `agent_runs/ref_s0_gen.csv` (C4 agent runs, not
  yet committed), was lost and is being re-run from scratch. No reported number was
  affected: nothing had been computed from that file yet.


---

## C4. Generalization: held-out topology, hazards, weathers (reference agent, τ=0.5)

Regenerate: `python -m experiments.revision1.sweeps gen` (55 settings × 11 cells × 10 seeds),
`python -m experiments.revision1.agent_eval --checkpoint checkpoints/campaign/ref/ckpt_final.pt --name ref_s0_gen --cells generalization --taus 0.5`,
then `python -m experiments.revision1.c4_generalization` → `results/revision1/C4/c4_generalization.json`.
Algorithm 1 per cell (q* = 0.95 × ceiling, deadline guard), eval seeds 0–9, sampled policy.
"Shipped" = `dvcast(n_slots=30)`, the best fixed setting chosen on the headline cells,
applied unchanged. "Refit" = best fixed re-chosen over these 11 cells
(`greedy_farthest(fallback_n_slots=16)`, hindsight on held-out data).

| held out | cell | q* | per-cell oracle | shipped | agent cost (best RWCR) | regret | margin vs shipped |
|---|---|---|---|---|---|---|---|
| topology | urban_grid d=2 | 0.940 | p_persist 0.544 | 0.667 | 1.266 (0.989) | +132.6% | −47.3% |
| topology | urban_grid d=20 | 0.942 | weighted_p 0.234 | 0.323 | never (0.938) | – | – |
| topology | urban_grid d=80 | 0.900 | p_persist 0.086 | 0.377 | 0.130 (0.918) | +50.5% | **+189.8%** |
| hazard | rural d=20 waterlogging | 0.873 | greedy 0.704 | 0.730 | 0.718 (0.899) | +2.1% | +1.7% |
| hazard | rural d=20 black_ice | 0.872 | greedy 0.704 | 0.730 | 0.719 (0.898) | +2.1% | +1.7% |
| hazard | urban_nlos d=20 waterlogging | 0.925 | counter 0.533 | 0.567 | never (0.920) | – | – |
| hazard | urban_nlos d=20 black_ice | 0.925 | counter 0.533 | 0.567 | never (0.920) | – | – |
| weather | rural d=20 moderate_rain | 0.877 | greedy 0.744 | 0.766 | 0.725 (0.905) | **−2.5%** | +5.6% |
| weather | rural d=20 dense_fog | 0.881 | greedy 0.795 | 0.850 | 0.746 (0.919) | **−6.1%** | **+13.9%** |
| weather | urban_nlos d=20 moderate_rain | 0.920 | counter 0.552 | 0.597 | never (0.913) | – | – |
| weather | urban_nlos d=20 dense_fog | 0.897 | dvcast 0.684 | 0.684 | never (0.869) | – | – |

**Agent matches 6 of 11 cells; the shipped fixed baseline matches 11 of 11.** The pattern
repeats the headline cells: the agent qualifies on the rural corridor at d=20 under every
held-out hazard and weather, and beats per-cell tuning in both held-out weathers
(−2.5%, −6.1%). It fails every held-out urban_nlos condition (best RWCR 0.869–0.920 vs
q* 0.897–0.925, the same near-miss as the headline urban cell) and urban_grid d=20. On
the held-out topology it is far dearer than tuning at d=2 (+133%) and at d=80 it beats the
shipped scheme 2.9× but not per-cell tuning (+50%).

**Caveat: the held-out hazards are a weak test.** For every baseline, waterlogging and
black ice produce the same transmissions and cost as fog_bank in all 550 paired
(setting, seed) runs per cell; only the oracle at-risk set, hence RWCR, shifts (by
≤ 0.018). Baselines never read the hazard type, and the hazard is placed at the same
location, so these cells differ from the training cells only through the agent's own
hazard-dependent inputs (relevance features). The held-out weathers do change the
network (traffic-mediated: speeds and headways), and the per-cell oracle cost moves
(0.704 → 0.744 → 0.795 on the corridor).

**Second interruption (2026-10-17 10:27 → 15:20).** All jobs died together again:
`BrokenProcessPool` in both training runs (crash at updates 495 / 518), with Intel
runtime "control-C event" aborts in the logs. Cause: the v1 launcher started every job
with `start /b`, so they shared one console, and one console control event killed all of
them. About 5 h lost; nothing completed was lost. Both runs resumed exactly (histories
contiguous, verified), and every CSV parsed cleanly. Fix: `launch_revision1.ps1` gives
each job its own hidden console (`Start-Process`). A watchdog task
(`watchdog_revision1.ps1`, every 20 min) relaunches only what is missing; train_queue
adopts live runs, so a relaunch cannot duplicate one. ref_s1 / ref_s2 therefore each have
two resume points (250 and 494 / 277 and 517).


---

## C1. Full model, 3 training seeds (0 = committed campaign/ref; 1, 2 new)

Regenerate: training `python -m experiments.revision1.train_queue --only ref_s1` / `ref_s2`
(= `agents.train --seed 1|2`, staged curriculum, 1,000 updates, deterministic GPU PPO); per-seed
evaluation of each FINAL checkpoint `python -m experiments.revision1.agent_eval --checkpoint
checkpoints/revision1/ref_s<k>/ckpt_final.pt --name ref_s<k> --taus 0.5 0`; aggregation
`python -m experiments.revision1.c_seeds` → `results/revision1/C1C2/c_seeds.json`.
Run records (seed, command, commit, config hash, checkpoint SHA, run summary):
`results/revision1/C_train/ref_s{1,2}.json`. Both runs: 1,000 updates, stopped at
`max_updates` (constraints never held for 30 consecutive updates; same as seed 0),
`bit_reproducible: true`, checkpoint SHAs 4867ffabbdf23aac (s1) / 0b5fa61c06396e28 (s2).
Each new run was resumed twice after the interruptions logged above.
Evaluation: 4 headline cells × 10 paired eval seeds × biases −2…+3, sampled.

**Per seed, matched cost / regret / margin (Algorithm 1, unbounded; "–" = never reaches q*)**

| τ | seed | rural d=2 | rural d=20 | rural d=80 | urban d=20 | cells matched |
|---|---|---|---|---|---|---|
| 0.5 | 0 | – | 0.717 / +1.8% / +1.9% | – | – | 1 |
| 0.5 | 1 | – | 0.725 / +3.1% / +0.7% | – | – | 1 |
| 0.5 | 2 | – | **0.554 / −21.3% / +31.9%** | 1.065 / −29.6% / +60.0% | 0.649 / +21.7% / −12.6% | 3 |
| 0 | 0 | – | 0.706 / +0.3% / +3.5% | – | 0.864 / +62.1% / −34.4% | 2 |
| 0 | 1 | – | 0.856 / +21.7% / −14.7% | – | 0.842 / +58.1% / −32.7% | 2 |
| 0 | 2 | – | 0.606 / −13.9% / +20.5% | 1.108 / −26.7% / +53.7% | 0.878 / +64.8% / −35.5% | 3 |

**Mean ± SD across the 3 training seeds**

| quantity | τ = 0.5 | τ = 0 |
|---|---|---|
| cells matched (of 4) | 1.67 ± 1.15 | 2.33 ± 0.58 |
| regret, common cells* | −5.5% ± 13.7 | +32.2% ± 7.2 |
| margin, common cells* | +11.5% ± 17.7 | −15.6% ± 8.1 |
| rural d=20 matched cost | 0.665 ± 0.097 | 0.723 ± 0.126 |
| RWCR at default bias, mean of 4 cells | 0.784 ± 0.034 | 0.805 ± 0.029 |
| cost at default bias, mean of 4 cells | 1.036 ± 0.074 | 1.127 ± 0.054 |
| median TIR at default bias, mean of 4 cells | 0.293 ± 0.017 s | 0.304 ± 0.015 s |
| actionable miss at default bias, mean of 4 cells | 0.377 ± 0.025 | — (in JSON) |

\* Common cells = matched by every seed: τ=0.5 {rural d=20}; τ=0 {rural d=20, urban d=20}.

**Latency-bounded (B1 bounds), τ = 0.5, rural d=20 margin per seed:** none +1.9 / +0.7 / +31.9%;
D=2 s +7.1 / +5.9 / +38.7%; D=1 s +25.3 / +23.9 / +62.2%; D=0.5 s +87.1 / +84.9 / +142.2%
(regret at D=0.5 s −1.9 / −0.7 / −24.2%). Cells matched do not change with the bound
(the agent's median TIR, 0.27–0.31 s, meets every bound). Full per-bound tables in the JSON.

**What this says.** Training-seed variance is large and dominates the effects the paper
reports. The paper's 1.8% / 1.9% at rural d=20 come from seed 0; across seeds the same
cell ranges from +3.1% to −21.3% regret, and the number of cells matched from 1 to 3. Seed 2
is the only seed that reaches rural d=80 and urban d=20 at τ=0.5. No seed reaches rural d=2
(the sparse cell the paper targets) at either τ. The τ=0 arm's urban d=20 result (+58–65%
regret) is stable across seeds; its rural d=20 result is not. With 3 seeds, a ± SD is
descriptive only.
