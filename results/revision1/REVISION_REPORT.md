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
