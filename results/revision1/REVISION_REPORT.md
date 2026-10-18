# Revision 1 — numbers and figures

Branch `revision-1`, forked from `main` at `461c754`. Everything new lives under
`results/revision1/`. No paper text was edited. Each section gives the
numbers, the file each number came from, and the command that regenerates it.
Where a number in the paper could not be reproduced, that is stated.

## Summary

Status after the revised plan (C1 = 3 full-model training seeds, C2 = 2 per ablation,
C5 skipped). Every number below is detailed, with its source file and regeneration
command, in the section of the same id. Agent = reference checkpoint unless stated;
q* = 0.95 × cell ceiling; "matched" = reaches q* under Algorithm 1's deadline guard.

| task | status | key numbers | files |
|---|---|---|---|
| A1 targets | done | three definitions (eval: oracle RWCR, 55 settings; training: causal, 32 seeds; §VII-F used **stale 2-seed** targets: rural d=2 0.891 vs 0.748 in force) | `A/stage_a_facts.json` |
| A2 0.717 | done | interpolation between bias +1 (0.716) and −2 (0.736), clear/fog_bank only; Table 2's 0.85/0.79 = bias 0 pooled over 200 runs (20 conditions) | `A/stage_a_facts.json` |
| A3 spread | done | 9 baselines 0.872–0.917 (Table 2 view); "0.915–0.917" holds for 8, i.e. without default p-persistence | `A/stage_a_facts.json` |
| A4 miss rates | done, **not reproducible** | §IV-A's 0.240/0.102 matches no committed artefact; under metrics v3 the two rates differ by ~0.007 on average | `A/stage_a_facts.json` |
| A5 λ | done | "≈50" = run8 (cap 50, pinned) / arm A; "peak 33.1" = campaign ref (cap 500), peak at the last updates (still rising) | `A/stage_a_facts.json` |
| A6 code facts | done | ensemble gate never used (entropy gate); heads averaged; 610 m vs 480 m from path-loss exponents ("562 m" stale); Table 3 = mean over 8 densities; 4 weathers (2 train); "26" = Holm threshold | `A6_facts.md` |
| A7 validation | done | Amador et al. 2022, DOI 10.1016/j.vehcom.2022.100454; Fig. 13 axis should read veh/km/lane; qualifying cell (40 veh/km) is below the validated range on physical bases | `REVISION_REPORT.md §A7` |
| A8 regret/margin | done | 1.8447% / 1.8970% (a/o−1, f/a−1) | `A/stage_a_facts.json` |
| B1 latency bound (4 cells) | done | greedy's 0.704 eliminated at D=0.5 s, DV-CAST 0.730 survives; agent margin +1.9% → +87% at 0.5 s; derived D: rural 15.4 s, NLOS 1.9 s | `B1/b1_headline*.json`, `D/fig_d4_*` |
| B1 latency bound (24 cells) | done | agent matched 12/24 (unbounded), median regret +84%; beats best fixed (slotted_1p(20)) in 4/12, 7/10 at D=0.5 s | `B1/b1_grid24.json` |
| B2 slots > 50 | done | cost flat beyond 50–100 slots, TIR keeps rising; guarded oracle unchanged in 3/4 cells | `B2/` |
| B3 feasibility | done | **no agent miss is "target unreachable"** under any target in force; the "0/63" holds only for the stale 2-seed target (12/63 under the current one) | `B3/b3_feasibility.json` |
| B4 statistics | done | rural d=20 cost difference not significant vs fixed/oracle (p_Holm 0.85/0.23); agent faster, lower RWCR, higher miss nearly everywhere; paired-difference SD 0.003–0.007, 12/36 baseline pairs differ (seed-cluster CI) | `B4/` |
| B5 sampled vs argmax | done | gaps +0.030/+0.001/+0.021/−0.006; the paper's 0.172 is run8's | `B5/b5_modes.json` |
| C1 3 seeds | done | cells matched 1.67 ± 1.15 (τ=0.5); rural d=20 regret +1.8/+3.1/**−21.3%**, margin +1.9/+0.7/+31.9%; no seed reaches rural d=2 | `C1C2/c_seeds.json`, `C_train/` |
| C2 ablations ×2 seeds | done | same cell (rural d=20): GCN −3.2% ± 0.1, star −0.5% ± 0.2, **MLP +55.9% ± 10.4** (was "best margin" in the committed table) | `C1C2/table_ablation_seeds.tex` |
| C3 fallback | done | **random gate at the learned rate matches 3/4 cells vs 1/4**; network-only (τ=0) 2/4, cheaper at rural d=20 (0.706 vs 0.717) | `C3/c3_fallback.json` |
| C4 generalization | done | agent 6/11 cells, shipped fixed 11/11; beats tuning in held-out weathers on the corridor (−2.5%, −6.1%); held-out hazards leave every baseline run unchanged | `C4/c4_generalization.json` |
| C5 2,000 updates | **skipped** (user) | – | – |
| D1 Fig. 8 as table | done | | `D/table_fig8_regret_margin.{csv,tex}` |
| D2 label fixes | done | Fig. 14 also averaged both τ arms into one point (fixed) | `D/fig08_*`, `D/fig14_*`, `analysis/figures.py` |
| D3 Fig. 12 annotation | done | "peak λ = 33.1, 6.6% of cap, rural d=2, update 994" | `D/fig12_*` |
| D4 latency figure | done | | `D/fig_d4_latency_bounded_headline.*` |
| D5 reproduction | done | `./reproduce.sh --revision1` | `experiments/revision1/reproduce_revision1.sh` |
| E1 fresh clone | done | headline numbers bit-exact (≤ 3.6e-15); `--smoke --no-train` failed and overwrote `pareto_cells.json` → fixed | `E1/` |
| E2 public-ready | done | no secrets; Windows user name in some committed results (left untouched); README updated; checkpoints committed under `results/revision1/checkpoints/` | §E2 |

Caveats that apply throughout: fallback mobility backend only (SUMO not installed on this
machine any more); 10 paired eval seeds; 2–3 training seeds; ref_s1/ref_s2 each resumed
twice after interruptions (exact resume; see run log).

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

## C1/C2. Training seeds — INTERIM, SUPERSEDED by the final C1 and C2 sections below (kept as a record)

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


---

## B1 (24 cells). Latency-bounded matched coverage over every scenario × density

Regenerate: `python -m experiments.revision1.sweeps grid24` (55 settings × 24 cells × 10 seeds,
clear/fog_bank; urban_grid d=2/20/80 rows reused from `gen_runs.csv`), `python -m
experiments.revision1.agent_eval --checkpoint checkpoints/campaign/ref/ckpt_final.pt --name ref_s0_grid24
--cells grid24 --taus 0.5 --also results/revision1/agent_runs/ref_s0.csv results/revision1/agent_runs/ref_s0_gen.csv`,
then `python -m experiments.revision1.b1_latency --rows results/revision1/B_sweeps/grid24_runs.csv
--extra-rows results/revision1/B_sweeps/gen_runs.csv --agent-rows results/revision1/agent_runs/ref_s0.csv
results/revision1/agent_runs/ref_s0_gen.csv results/revision1/agent_runs/ref_s0_grid24.csv --tag grid24`
→ `results/revision1/B1/b1_grid24.{json,md}` (every per-scheme k*, oracle, fixed and agent
value per cell and per bound; both target variants). Agent = reference checkpoint (seed 0),
τ=0.5, sampled. The best single fixed setting is re-chosen over the 24 cells: it is
**`slotted_1p(n_slots=20)`**, not the 4-cell `dvcast(n_slots=30)`.

| bound D | best single fixed | cells where fixed misses q* | cells with no qualifying scheme | agent matched | agent median regret | agent beats fixed |
|---|---|---|---|---|---|---|
| none | slotted_1p(20) | 2 | 0 | **12 / 24** | +84% | 4 of 12 |
| 2.0 s | slotted_1p(20) | 3 | 2 | 11 / 24 | +84% | 4 of 11 |
| 1.0 s | slotted_1p(8) | 4 | 3 | 10 / 24 | +84% | 5 of 10 |
| 0.5 s | slotted_1p(3) | 4 | 3 | 10 / 24 | +84% | **7 of 10** |
| derived (rural 15.4 s, grid 26.7 s, NLOS 1.9 s) | slotted_1p(8) | 2 | 1 | 12 / 24 | +84% | 5 of 12 |

Per cell, unbounded (cost; regret / margin for the agent where it qualifies):

| cell | q* | per-cell oracle | fixed slotted_1p(20) | agent | regret / margin |
|---|---|---|---|---|---|
| rural d=1 | 0.085 | dvcast 2.867 | 2.933 | – | |
| rural d=2 | 0.641 | dvcast 1.560 | 1.688 | – | |
| rural d=3 | 0.774 | dvcast 1.711 | – | – | |
| rural d=5 | 0.858 | greedy 0.902 | 0.966 | – | |
| rural d=10 | 0.859 | greedy 0.704 | 0.803 | 0.832 | +18% / −3% |
| rural d=20 | 0.879 | greedy 0.704 | 0.768 | 0.717 | +2% / +7% |
| rural d=40 | 0.860 | weighted_p 0.701 | 0.826 | – | |
| rural d=80 | 0.870 | weighted_p 1.512 | 1.837 | – | |
| urban_nlos d=1 | 0.672 | greedy 2.020 | – | – | |
| urban_nlos d=2 | 0.897 | dvcast 1.266 | 1.300 | – | |
| urban_nlos d=3 | 0.933 | counter 0.923 | 1.001 | 1.134 | +23% / −12% |
| urban_nlos d=5 | 0.929 | dvcast 0.905 | 0.940 | 0.917 | +1% / +2% |
| urban_nlos d=10 | 0.919 | slotted 0.826 | 0.826 | – | |
| urban_nlos d=20 | 0.926 | counter 0.533 | 0.578 | – | |
| urban_nlos d=40 | 0.922 | p_persist 0.322 | 0.427 | 0.592 | +84% / −28% |
| urban_nlos d=80 | 0.876 | weighted_p 0.463 | 0.544 | – | |
| urban_grid d=1 | 0.941 | slotted 0.822 | 0.933 | 1.285 | +56% / −27% |
| urban_grid d=2 | 0.940 | p_persist 0.544 | 0.647 | 1.266 | +133% / −49% |
| urban_grid d=3 | 0.937 | weighted_p 0.364 | 0.503 | 1.212 | +233% / −58% |
| urban_grid d=5 | 0.941 | weighted_p 0.326 | 0.430 | 1.074 | +229% / −60% |
| urban_grid d=10 | 0.942 | p_persist 0.271 | 0.459 | 0.968 | +258% / −53% |
| urban_grid d=20 | 0.942 | weighted_p 0.234 | 0.359 | – | |
| urban_grid d=40 | 0.935 | p_persist 0.095 | 0.303 | 0.270 | +184% / +12% |
| urban_grid d=80 | 0.900 | p_persist 0.086 | 0.403 | 0.130 | +50% / +210% |

**What changes with the bound.** As in the headline cells, the per-cell oracle is mostly
stable down to D = 1 s, and the bound mainly pushes the best fixed scheme to ever-shorter
slotted settings, which are expensive elsewhere (at D = 0.5 s `slotted_1p(3)` costs 3.98 at
rural d=80 against the oracle's 1.51). The agent's margin therefore improves with the bound
(beats the fixed scheme in 4 → 7 cells). But it also loses cells: at D ≤ 1 s it no longer
qualifies at urban_grid d=1 and d=2, where its median TIR exceeds the bound and no scheme
reaches q* within it. Greedy-farthest's 0.704 at rural d=20 is eliminated at D = 0.5 s;
DV-CAST's 0.730 survives (it is the oracle there), as in the 4-cell analysis.

**Overall.** Over all 24 cells, the reference agent reaches matched coverage in half
of them. It is near per-cell tuning in two (rural d=20 +2%, urban_nlos d=5 +1%), and on the
held-out urban_grid topology it is 50–258% dearer than tuning in every cell, though cheaper
than the shipped fixed scheme at d=40 and d=80. It never reaches the target in any rural
cell below d=10, i.e. the sparse regime the paper targets.


---

## C2. Ablations over training seeds (FINAL; supersedes the interim seed-0 section)

Regenerate as C1 (runs `gcn_s1`, `mlp_s1`, `star_s1`; seed 0 = committed `checkpoints/campaign/<v>`),
then `python -m experiments.revision1.c_seeds` → `results/revision1/C1C2/c_seeds.json` and
`table_ablation_seeds.tex`. Run records: `results/revision1/C_train/{gcn,mlp,star}_s1.json`. All three:
1,000 updates, `max_updates`, `bit_reproducible: true`; SHAs cb855009fb1ddc3a (GCN), 39b35e7d49c2c4b2 (MLP),
a9e4d85b834ff15c (star). Per the revised plan: 3 training seeds for the full model, 2 per ablation.

**Same cells for every variant (τ = 0.5).** Rural d=20 is the only cell every variant and
every seed matches, so regret and margin are compared there. Mean ± SD across training seeds:

| variant | training seeds | cells matched (of 4) | regret, rural d=20 | margin, rural d=20 | cost at bias 0, mean of 4 cells | RWCR at bias 0 | median TIR at bias 0 |
|---|---|---|---|---|---|---|---|
| full (GATv2) | 3 | 1.67 ± 1.15 | −5.5% ± 13.7 | +11.5% ± 17.7 | 1.04 ± 0.07 | 0.784 ± 0.034 | 0.293 ± 0.017 s |
| GCN encoder | 2 | **3.00 ± 0.00** | **−3.2% ± 0.1** | +7.2% ± 0.1 | 1.16 ± 0.20 | 0.806 ± 0.003 | 0.298 ± 0.003 s |
| MLP encoder | 2 | 3.00 ± 0.00 | **+55.9% ± 10.4** | **−33.3% ± 4.5** | 1.23 ± 0.06 | 0.813 ± 0.006 | 0.305 ± 0.005 s |
| star graph | 2 | 1.50 ± 0.71 | −0.5% ± 0.2 | +4.3% ± 0.2 | 1.07 ± 0.01 | 0.765 ± 0.002 | 0.292 ± 0.015 s |

**Every matched cell, per training seed, τ = 0.5** (cost; regret / margin):

| variant | rural d=20 | rural d=80 | urban d=20 |
|---|---|---|---|
| full | s0 0.717 (+1.8/+1.9), s1 0.725 (+3.1/+0.7), s2 0.554 (−21.3/+31.9) | s2 1.065 (−29.6/+60.0) | s2 0.649 (+21.7/−12.6) |
| GCN | s0 0.681 (−3.2/+7.2), s1 0.682 (−3.1/+7.1) | s0 0.991 (−34.5/+71.9), s1 0.955 (−36.8/+78.4) | s0 0.782 (+46.8/−27.6), s1 0.859 (+61.2/−34.0) |
| MLP | s0 1.149 (+63.3/−36.4), s1 1.045 (+48.5/−30.1) | s0 0.708 (−53.2/+140.6), s1 0.833 (−44.9/+104.5) | s0 0.902 (+69.3/−37.2), s1 0.626 (+17.4/−9.4) |
| star | s0 0.699 (−0.7/+4.5), s1 0.701 (−0.3/+4.1) | – | s0 0.722 (+35.5/−21.6) |

No variant or seed reaches rural d=2 at τ=0.5.

**Network-only arm (τ = 0), every matched cell:** full: rural d=20 s0/s1/s2 0.706/0.856/0.606,
rural d=80 s2 1.108, urban d=20 s0/s1/s2 0.864/0.842/0.878 (+58–65% regret); GCN: rural d=20
0.759/0.941, rural d=80 0.833/0.807, urban d=20 0.731/0.887; MLP: rural d=20 1.283/1.694, rural d=80
0.832/0.873, urban d=20 1.000/1.054; star: rural d=20 s0 0.991, rural d=80 s1 0.764, urban d=20 s0 0.869,
and **rural d=2 s1 1.937 (+24.2% regret, −19.5% margin) — the only agent of any variant, seed
or gate arm to reach the sparse cell the paper targets.** No cell is matched by every
variant and seed at τ=0, so no common-cell mean is defined there.

**What this says.**
1. **The committed ablation table's ranking does not survive the same-cells comparison.**
   It averaged each variant over its own matched cells: MLP had the best margin (+22.3%).
   On the common cell MLP is the worst variant on both seeds (+48.5% / +63.3% regret).
   Its averages were carried by rural d=80, where every ablation is cheap.
2. **GCN is the most consistent variant:** 3 of 4 cells on both seeds, with near-identical
   costs (0.681 / 0.682 at rural d=20, −3% vs per-cell tuning). The full GATv2 model matches
   1, 1 and 3 cells across its three seeds; its best seed (s2) beats GCN at rural d=20 and
   matches its cell count, and its other two do not.
3. **The architecture claim cannot be resolved at this seed count.** The full model's seed
   spread (−21% to +3% regret, 1–3 cells) is wider than every between-variant difference
   except MLP's. With 2–3 training seeds, the defensible statement is "removing attention
   (GCN) or neighbour edges (star) does not make the policy worse; removing the graph (MLP)
   makes it markedly more expensive where all variants qualify."
4. Training-seed variance should be reported for every agent number in the paper.

---

# REVISION 2

Same rules as revision 1; outputs under `results/revision2/<id>/` (per-run CSVs shared
with revision 1 stay in `results/revision1/B_sweeps/` and `results/revision1/agent_runs/`).

## Run log (revision 2)

* 2026-10-17 23:03: overnight jobs launched via Task Scheduler (`AIHARP_revision2` →
  `launch_revision1.ps1 -Revision2`, which runs `watchdog_revision2.ps1`; the same script runs
  every 20 min as `AIHARP_revision2_watchdog` and relaunches any unfinished job with no live
  process, each in its own hidden console). Jobs: R6 `ref_s3`, `ref_s4` and R7 `norel_s1`
  (`--drop-node-feature relevance_causal`) training in parallel, 4 rollout workers each; R7
  seed 0 = committed `checkpoints/campaign/no_relevance` (1,000 updates, seed 0, feature zeroed),
  evaluated only; R4 training-seed sweep (`sweeps trainsel`, 13,200 runs). Laptop: sleep,
  screen-off and hibernate on AC set to never for the night (screen-off is what triggers
  Modern Standby on this machine).


## R1. Tail-latency bound (seed-mean p95 TIR ≤ D)

Regenerate: `python -m experiments.revision2.r1_tail_latency` → `results/revision2/R1/r1_tail_latency.json`
(inputs: `results/revision1/B_sweeps/{headline,grid24,gen}_runs.csv`, `results/revision1/agent_runs/ref_s{0,1,2}.csv`,
`ref_s0_gen.csv`, `ref_s0_grid24.csv`; no new runs). Rule as B1 (bound in the eligibility
test, incl. the interpolation anchor; q* and miss bound held at unbounded values) on `tir_p95_s`.
D ∈ {1, 2, 5 s, derived: rural 15.39 s, urban_grid 26.65 s, urban_nlos 1.89 s}. Agent: full
model, τ=0.5, sampled; training seeds 0–2 in the headline cells (3–4 will be added after R6),
seed 0 on the 24-cell grid. Cost = transmissions per at-risk vehicle informed.

**Headline cells** (oracle scheme and cost; best fixed cost; agent cost with regret / margin per seed):

| D | best fixed | rural d=2 oracle / fixed | rural d=20 oracle / fixed | agent rural d=20 s0 / s1 / s2 | rural d=80 oracle / fixed | urban d=20 oracle / fixed | agent cells matched s0/s1/s2 |
|---|---|---|---|---|---|---|---|
| none | dvcast(30) | dvcast 1.560 / 1.560 | greedy 0.704 / 0.730 | 0.717 (+1.8/+1.9) / 0.725 (+3.1/+0.7) / 0.554 (−21.3/+31.9) | weighted_p 1.512 / 1.703 | counter 0.533 / 0.567 | 1 / 1 / 3 |
| 5 s | dvcast(30) | as none | as none | as none | as none | as none | 1 / 1 / 3 |
| 2 s | slotted_1p(8) | slotted 1.654 / 1.654 | greedy 0.704 / 0.898 | margins +25.3 / +23.9 / +62.2 | weighted_p 1.512 / 2.282 | counter 0.533 / 0.640 | 1 / 1 / 3 |
| 1 s | slotted_1p(3) | slotted 1.755 / 1.755 | **dvcast 0.730** / 1.341 | regret −1.9 / −0.7 / −24.2; margin +87 / +85 / +142 | weighted_p 1.512 / 3.979 | counter 0.533 / 0.832 | 1 / 1 / 3 |
| derived | slotted_1p(8) | dvcast 1.560 / 1.654 | greedy 0.704 / 0.898 | margins +25.3 / +23.9 / +62.2 | weighted_p 1.512 / 2.282 | counter 0.533 / 0.640 | 1 / 1 / 3 |

Seed 2's other cells: rural d=80 1.065 (regret −29.6%, margin +60% → +274% at D = 1 s) and urban d=20 0.649
(+21.7%; margin −12.6% none, +28.2% at D = 1 s). Greedy-farthest's 0.704 (p95 1.905 s) survives D ≥ 2 s and
is eliminated at D = 1 s, where DV-CAST's 0.730 (p95 0.959 s) becomes the oracle. Under the p95 bound, as under
the median bound, the agent's matched cells never change: its p95 TIR at its qualifying setting is 0.42–0.54 s.

**24 cells, agent seed 0:**

| D | best fixed | cells where fixed misses | cells with no qualifying scheme | agent matched | median regret | agent beats fixed |
|---|---|---|---|---|---|---|
| none | slotted_1p(20) | 2 | 0 | 12 | +84% | 4 |
| 5 s | slotted_1p(20) | 3 | 2 | 11 | +84% | 4 |
| 2 s | slotted_1p(8) | 4 | 2 | 11 | +51% | 5 |
| 1 s | slotted_1p(3) | 4 | 3 | 10 | +84% | 7 |
| derived | slotted_1p(8) | 2 | 1 | 12 | +84% | 5 |

**Per-vehicle deadline (actionable miss rate) at each reference point**, unbounded, headline
cells (fraction of at-risk vehicles that still had time to react at origination but were not
warned before their own reaction deadline; Algorithm 1 admits a point only if this is ≤ the
cell's best + 0.05):

| cell | best any setting | admitted up to | per-cell oracle point | best fixed point | agent point s0 / s1 / s2 |
|---|---|---|---|---|---|
| rural d=2 | 0.529 | 0.579 | 0.565 | 0.565 | never qualifies |
| rural d=20 | 0.314 | 0.364 | 0.317 | 0.317 | 0.332 / 0.351 / 0.344 |
| rural d=80 | 0.318 | 0.368 | 0.327 | 0.319 | – / – / 0.353 |
| urban d=20 | 0.067 | 0.117 | 0.078 | 0.083 | – / – / 0.099 |

So the miss guard is the per-vehicle constraint. At every reference point in the corridor
cells, about a third of the vehicles that could have been saved are not warned in time
(over half at rural d=2), and the agent's qualifying points miss 1–4 points more than the
references. Over the 24 cells the oracle points' actionable miss ranges from 0.057 to 0.926
(the 0.926 is rural d=1, where the target is 0.085).


**R1 update with R6 (5 training seeds, regenerated by the same command).** Agent cells matched
s0–s4: **1 / 1 / 3 / 3 / 3** under every p95 bound (unchanged by the bound). Seeds 3 and 4, unbounded:
rural d=20 0.668 (−5.2% / +9.4%) and 0.534 (−24.1% / +36.8%); rural d=80 0.934 (−38.2% / +82.3%) and
0.980 (−35.2% / +73.8%); urban d=20 0.686 (+28.7% / −17.4%) and 0.731 (+37.1% / −22.5%). At D = 1 s their
rural d=20 margins rise to +101% / +151%. Their qualifying points' actionable miss: rural d=20 0.340 / 0.358,
rural d=80 0.355 / 0.350, urban d=20 0.110 / 0.113 (bounds 0.364 / 0.368 / 0.117).

## R2. Sensitivity of Algorithm 1 to target, miss tolerance and interpolation

Regenerate: `python -m experiments.revision2.r2_target_sensitivity` →
`results/revision2/R2/r2_target_sensitivity.{json,csv}` (the CSV has, for each of the 18
configurations × 4 cells: q*, oracle scheme and cost, best fixed and its cost, and every seed's cost,
regret and margin; the JSON lists every flip). Agent: full model, τ=0.5, training seeds 0–4.
No-interpolation = the cost of the cheapest *realised* qualifying point.

| q* | miss tol. | best fixed | rural d=2 oracle | rural d=80 oracle | cells matched s0–s4 | rural d=20 regret s0 / s1 / s2 / s3 / s4 (%) | flips vs paper (interp / no-interp) |
|---|---|---|---|---|---|---|---|
| 0.90 | 0.02 | slotted_1p(20) | dvcast 1.640 | weighted_p 1.512 | 1/1/2/1/1 | +4.6 / +39.6 / +14.6 / +40.1 / +59.6 | 13 / 13 |
| 0.90 | 0.05 | dvcast(30) | dvcast 1.560 | **p_persist 0.647** | 1/1/3/3/3 | −6.8 / +3.1 / −21.3 / −5.2 / −24.1 | 2 / 2 |
| 0.90 | 0.10 | dvcast(30) | dvcast 1.560 | p_persist 0.647 | **3/3/3/3/3** | −17.4 / −6.4 / −41.6 / −21.6 / −24.1 | 7 / 7 |
| 0.95 | 0.02 | slotted_1p(20) | slotted 1.654 | weighted_p 1.512 | 1/1/2/1/1 | +4.6 / +39.6 / +14.6 / +40.1 / +59.6 | 14 / 14 |
| **0.95** | **0.05** | **dvcast(30)** | **dvcast 1.560** | **weighted_p 1.512** | **1/1/3/3/3** | **+1.8 / +3.1 / −21.3 / −5.2 / −24.1** | **0 / 1** |
| 0.95 | 0.10 | dvcast(30) | dvcast 1.560 | weighted_p 1.273 | 1/1/3/3/3 | +1.8 / +1.2 / −27.4 / −15.9 / −24.1 | 0 / 1 |
| 0.98 | 0.02 | dvcast(30) | **none qualifies** | weighted_p 1.512 | 0/1/1/1/1 | – / +39.6 / +14.6 / +40.1 / +59.6 | 15 / 15 |
| 0.98 | 0.05 | greedy(32) | none qualifies | weighted_p 1.512 | 0/1/1/1/1 | – / +36.9 / −3.0 / +17.5 / +50.9 | 14 / 16 |
| 0.98 | 0.10 | slotted_1p(50) | none qualifies | weighted_p 1.477 | 0/1/1/1/1 | – / +36.9 / −3.0 / +17.5 / +50.9 | 14 / 16 |

The rural d=20 oracle is greedy-farthest 0.704 and the urban d=20 oracle counter-based 0.533 in every
configuration.

**Conclusions that flip:**
1. **The paper's seed-0 margin at rural d=20 exists only because of interpolation:** without it,
   +1.9% becomes −0.8% (the only flip at the paper's q* and tolerance).
2. **The miss tolerance decides the agent's verdict.** At 0.02, every seed's rural d=20 regret is
   positive (+4.6% to +59.6%); seeds 2–4 lose rural d=80 and urban d=20; the best fixed scheme becomes
   `slotted_1p(20)`. At 0.10 (with q* 0.90), every seed matches 3 cells and every regret is negative.
   The agent's qualifying points sit close to the guard (actionable miss 0.33–0.36 against a 0.364
   bound at rural d=20; R1), which is why the tolerance matters so much.
3. **q* = 0.98:** no scheme qualifies at rural d=2; seed 0 no longer qualifies anywhere; the best fixed
   scheme changes with the tolerance.
4. **Robust:** the per-cell oracle at rural d=20 and urban d=20, the agent never reaching rural d=2,
   and the 0.95 → 0.90 target change at tolerance 0.05 (only rural d=80's oracle and the sign of seed
   0's regret move).


## R3. Hierarchical bootstrap (training seeds × eval seeds)

Regenerate: `python -m experiments.revision2.r3_hier_bootstrap --seeds 0 1 2 3 4` (and `--seeds 0 1 2`)
→ `results/revision2/R3/r3_hier_bootstrap_seeds01234.json`, `..._seeds012.json`. 10,000 resamples, RNG seed
20261018. Each resample draws training seeds and the 10 eval seeds jointly with replacement (one eval
draw for baselines and agent, so pairing is kept) and re-runs Algorithm 1 in full, so the per-cell
oracle and the best fixed setting are re-selected per resample. Point estimates reproduce the
comparator exactly (seed 0: 0.018447 / 0.018970). Agent: full model, τ=0.5, sampled.

| | 5 training seeds (0–4) | 3 training seeds (0–2) |
|---|---|---|
| rural d=20 regret: point (mean over seeds) | −9.1% | −5.5% |
| rural d=20 regret: 95% CI | **[−19.2%, +10.5%]** | [−20.7%, +7.3%] |
| share of resamples with regret < 0 | 0.80 | 0.78 |
| rural d=20 margin: point | +16.1% | +11.5% |
| rural d=20 margin: 95% CI | [−1.1%, +129.9%] | [−0.7%, +141.7%] |
| share of resamples with margin > 0 | 0.97 | 0.97 |
| cells matched (of 4): point | 2.20 | 1.67 |
| cells matched: 95% CI | [1.4, 3.0] | [1.0, 3.0] |
| agent fails rural d=20 (share of draws) | 0 | 0 |

**Reading.** Regret against per-cell tuning is not distinguishable from zero (the CI spans
−19% to +10%). The margin over the best fixed scheme is positive in 97% of resamples but its
upper tail is very long. The cause is checked, not assumed: **the identity of the best single
fixed setting is unstable at 10 eval seeds.** Over 400 resamples `dvcast(n_slots=30)` wins only
30%, and 17 different settings win at least once (greedy_farthest(16) 14%, slotted_1p(12) 11%,
slotted_1p(20) 9%, …), some of which cost 1.2–1.8 at rural d=20 (e.g. counter_based(3), dvcast(3)).
The fixed reference is itself a noisy estimate, and that noise belongs in any margin claim.


## R4. Adaptive classical baseline (per-cell tuning on training seeds, no hindsight)

Regenerate: `python -m experiments.revision1.sweeps trainsel` (55 settings × training seeds 100–109 ×
24 cells = 13,200 runs → `results/revision1/B_sweeps/trainsel_runs.csv`), then
`python -m experiments.revision2.r4_adaptive_baseline` → `results/revision2/R4/r4_adaptive_baseline.json`.
Rule: per cell, on training seeds only, q*_train = 0.95 × best training-seed mean RWCR, miss bound =
best training-seed miss + 0.05; pick the cheapest **realised** setting meeting both. Deploy that setting
on eval seeds 0–9 and score it with Algorithm 1's eval-side target and guard. Derived-deadline version:
median TIR ≤ D_r applied in selection and evaluation. Agent: full model, τ=0.5, seeds 0–4 (headline),
seed 0 (24 cells).

**Headline cells** (unbounded; the derived deadline changes nothing here):

| cell | adaptive pick (training seeds) | eval cost | meets eval q* + guard | regret vs eval oracle | eval oracle | agent cost s0 / s1 / s2 / s3 / s4 |
|---|---|---|---|---|---|---|
| rural d=2 | dvcast(n_slots=8) | 1.640 | no | – | 1.560 | never (all seeds) |
| rural d=20 | slotted_1p(n_slots=30) | 0.749 | yes | +6.5% | 0.704 | **0.717 / 0.725 / 0.554 / 0.668 / 0.534** |
| rural d=80 | p_persistence(p=0.08) | 0.769 | **no** (RWCR short on eval seeds) | – | 1.512 | – / – / 1.065 / 0.934 / 0.980 |
| urban d=20 | counter_based(1) | **0.533** | yes | **0.0%** | 0.533 | – / – / 0.649 / 0.686 / 0.731 |

**24 cells:**

| | unbounded | derived deadline |
|---|---|---|
| adaptive meets the eval target | **13 / 24** (median regret **+6.5%**) | 14 / 24 (+6.0%); 1 cell has no qualifying training-seed setting |
| agent (seed 0) meets the eval target | 12 / 24 (median regret +84%, B1) | 12 / 24 |
| both qualify / agent cheaper there | 8 / **1** (rural d=20) | 8 / 1 |
| agent only / adaptive only / neither | 4 / 5 / 7 | 4 / 6 / 6 |

Per-cell picks, costs and regrets are in the JSON (examples: urban_grid d=20 weighted_p(0.15) =
oracle 0.234 where the agent never qualifies; urban_grid d=5 weighted_p(0.2) 0.326 vs agent 1.074).

**Reading.** A classical scheme retuned per cell on training traffic (no access to evaluation
seeds) is close to hindsight tuning (median +6.5%) and qualifies about as often as the agent
(13 vs 12 of 24). Where both qualify it is cheaper in 7 of 8 cells. **The agent's advantage is
confined to the rural corridor at d=20–80:** it is cheaper at rural d=20 on all five training
seeds, and at rural d=80 three of five seeds qualify where the adaptive pick does not
generalise from training to eval seeds. This is the strongest available comparison for the
"learned vs retuned" question: the per-cell oracle is not deployable, but this baseline is.


## R6. Full model, 5 training seeds (C1 updated)

Training: `python -m experiments.revision1.train_queue --only ref_s3` / `ref_s4` (= `agents.train --seed 3|4`,
same config, 1,000 updates, deterministic GPU PPO), run overnight; both ran uninterrupted (no resume;
histories contiguous; 4.23 h / 4.20 h), stopped at `max_updates`, `bit_reproducible: true`. Records:
`results/revision1/C_train/ref_s{3,4}.json` (seed, command, commit, config hash, checkpoint SHA).
Final-checkpoint evaluation (4 headline cells × 10 eval seeds, τ 0 and 0.5, sampled):
`results/revision1/agent_runs/ref_s{3,4}.csv`. Aggregation: `python -m experiments.revision1.c_seeds` →
`results/revision1/C1C2/c_seeds.json`, `table_ablation_seeds.tex`.

**Per seed, τ = 0.5** (matched cost; regret):

| seed | rural d=2 | rural d=20 | rural d=80 | urban d=20 | cells matched |
|---|---|---|---|---|---|
| 0 | – | 0.717 (+1.8%) | – | – | 1 |
| 1 | – | 0.725 (+3.1%) | – | – | 1 |
| 2 | – | 0.554 (−21.3%) | 1.065 (−29.6%) | 0.649 (+21.7%) | 3 |
| 3 | – | 0.668 (−5.2%) | 0.934 (−38.2%) | 0.686 (+28.7%) | 3 |
| 4 | – | 0.534 (−24.1%) | 0.980 (−35.2%) | 0.731 (+37.1%) | 3 |

**Mean ± SD across 5 seeds:**

| quantity | τ = 0.5 | τ = 0 |
|---|---|---|
| cells matched (of 4) | **2.20 ± 1.10** | 2.40 ± 0.55 |
| rural d=20 regret (matched by all seeds) | **−9.1% ± 12.8** | +9.3% ± 15.6 (0.3 / 21.7 / −13.9 / 18.4 / 20.0) |
| rural d=20 margin | **+16.1% ± 17.0** | −3.3% ± 15.2 |
| RWCR at bias 0, mean of 4 cells | 0.789 ± 0.025 | 0.813 ± 0.024 |
| cost at bias 0, mean of 4 cells | 1.087 ± 0.097 | 1.178 ± 0.093 |
| median TIR at bias 0, mean of 4 cells | 0.299 ± 0.015 s | 0.307 ± 0.011 s |

Hierarchical-bootstrap CIs for these 5 seeds are in R3 (regret [−19.2%, +10.5%], cells matched [1.4, 3.0]).
Latency-bounded versions are in R1. With 5 seeds the picture is: **3 of 5 seeds reach 3 of 4 cells**,
the paper's seed 0 is one of the two weakest, and no seed reaches rural d=2 at either τ. At τ=0 the
network-only arm qualifies at urban d=20 on all 5 seeds but at +38–65% regret.

## R7. Relevance-removal ablation (causal relevance zeroed at train and eval)

Seed 0 = committed `checkpoints/campaign/no_relevance` (1,000 updates, `drop_node_features:
[relevance_causal]`; never evaluated before this revision). Seed 1 trained overnight
(`train_queue --only norel_s1` = `agents.train --seed 1 --drop-node-feature relevance_causal`; uninterrupted;
4.17 h; `max_updates`; bit-reproducible; `results/revision1/C_train/norel_s1.json`). The switch travels in the
checkpoint config, so evaluation zeroes the feature too (verified on norel_s1's checkpoint). Evaluation
rows: `results/revision1/agent_runs/norel_s{0,1}.csv`; aggregation as R6.

| variant | seeds | cells matched (τ=0.5) | rural d=20 regret (τ=0.5) | other cells (τ=0.5) | cells matched (τ=0) |
|---|---|---|---|---|---|
| full model | 5 | 2.20 ± 1.10 (1/1/3/3/3) | −9.1% ± 12.8 | see R6 | 2.40 ± 0.55 |
| **no causal relevance** | 2 | **2.50 ± 0.71** (2 / 3) | **−0.9% ± 14.1** (s0 0.767 = +9.0%, s1 0.627 = −10.9%) | rural d=80 s1 0.656 (−56.6%); urban d=20 s0 0.740 (+38.9%), s1 0.794 (+49.0%) | 2.50 ± 0.71 |

Margin at rural d=20: +5.8% ± 15.1 (no relevance) vs +16.1% ± 17.0 (full). Default-bias means over 4 cells
(τ=0.5): RWCR 0.777 ± 0.007, cost 1.023 ± 0.075, TIR 0.287 ± 0.001 s.

**Reading.** Removing the causal-relevance input does not reduce the number of cells the policy
qualifies in (2 and 3, within the full model's 1–3 range), and its rural d=20 regret lies inside
the full model's seed spread. Its seed 1 has the lowest cost of any agent at rural d=80 (0.656).
At 2 seeds this ablation cannot show that the causal-relevance feature contributes; the
premise that the policy depends on it is not supported by this evidence.


## R5. Reproducibility table

Regenerate: `python -m experiments.revision2.r5_repro_table` → `results/revision2/R5/r5_repro_table.{csv,tex}`
(288 rows). One row per row of every table drawn from `results/`: Table 2 and the 23 other per-cell
main tables (`results/tables/table_main*.tex`, 11 rows each), the reference table, the ablation table,
Table 3 (risk estimation), Table 5 (λ arms), Table 6 (validation), and the revision ablation-over-seeds
table. Columns: source file, the last commit that changed it, config hash, content hash, checkpoint SHA,
metrics_version.

* **Config hash.** `results/runs.csv` records one config hash per run, so a Table 2 row gets
  `set:<hash of its 200 sorted per-run config hashes> (200 runs)`. Training-derived rows (ablation,
  Table 5, seeds table) get the hash of the training configuration stored in the checkpoint.
  `pareto_cells.json`, `risk_estimation.csv` and `amador2022.json` record **no** config hash; those rows
  say "not recorded" and give `content_sha256` of the exact source rows instead. This is a gap in the
  committed evidence, reported rather than filled.
* **Checkpoint SHAs.** Agent rows of Table 2: `b589be512827fcbb` (campaign/ref = the `ref_s0` of
  this revision). Ablation table: ref `b589be512827fcbb`, GCN `4090fd9374db4d4c`, MLP `20b6c8942ebfab28`,
  star `b3536e1a471535a4`. Each matches the SHA its evaluation recorded. Table 5: run8 `ab0e1e94f9d4c4b9`,
  arm A `f74dc381d6697208`, arm B `727352ad75621d8d`. Seeds table: all 13 checkpoints (also in
  `results/revision1/checkpoints/SHA.json`, where copies of the final checkpoints are committed).
* **Commits.** runs.csv / Table 2: `754f34e`; reference table: last change to `pareto_cells.json`;
  ablation table: `c18bb90` (ref) and `6eb790e` (GCN/MLP/star); Table 5: `a378bcf`.

## REVISION 2 summary

| task | status | key numbers | files |
|---|---|---|---|
| R1 tail-latency bound | done | p95 bound: greedy's 0.704 survives D ≥ 2 s, eliminated at 1 s (DV-CAST 0.730 survives); agent cells matched unchanged by the bound (1/1/3/3/3 over seeds 0–4); 24 cells 10–12 matched; per-vehicle actionable miss at every reference point ~0.32 (corridor), agent points 0.33–0.36 against a 0.364 guard | `revision2/R1/` |
| R2 target sensitivity | done | **miss tolerance decides the verdict** (0.02: every seed's rural d=20 regret > 0; 0.10 with q* 0.90: every seed matches 3 cells); **seed 0's +1.9% margin is an interpolation artefact** (−0.8% on realised points); q*=0.98: nothing qualifies at rural d=2 | `revision2/R2/` |
| R3 hierarchical bootstrap | done | 5 seeds: rural d=20 regret −9.1%, **95% CI [−19.2%, +10.5%]**; margin +16.1% [−1.1%, +129.9%]; cells matched 2.2 [1.4, 3.0]; best-fixed identity unstable (dvcast(30) only 30% of resamples, 17 settings win) | `revision2/R3/` |
| R4 adaptive classical baseline | done | tuned on training seeds: meets eval target in **13/24 cells, median regret +6.5%** (agent seed 0: 12/24, +84%); where both qualify it is cheaper in 7/8; agent cheaper at rural d=20 on all 5 seeds and qualifies at rural d=80 where the adaptive pick fails | `revision2/R4/` |
| R5 reproducibility table | done | 288 rows; config hash missing in 3 committed sources (content hashes given) | `revision2/R5/` |
| R6 full model ×5 seeds | done | cells matched **2.20 ± 1.10** (1/1/3/3/3); rural d=20 regret −9.1% ± 12.8, margin +16.1% ± 17.0; no seed reaches rural d=2 | `revision1/C1C2/`, `C_train/ref_s{3,4}.json` |
| R7 relevance removal | done | 2 seeds: 2.50 ± 0.71 cells matched, rural d=20 regret −0.9% ± 14.1; within the full model's spread, so no evidence the feature helps | `revision1/C1C2/`, `C_train/norel_s1.json` |

Overnight: R6/R7 (3 runs) trained in parallel 23:03 → ~03:25 with no interruption (no resume),
evaluations finished 03:48, the R4 sweep (13,200 runs) finished overnight; the watchdog logged no
failure.


---

# REVISION 3

Outputs under `results/revision3/<id>/`. Per-run CSVs shared with earlier revisions stay in
`results/revision1/B_sweeps/` and `results/revision1/agent_runs/`.

## R10. Deadline assumptions (derived deadline, p95-TIR bound)

Regenerate: `python -m experiments.revision3.r10_deadline_assumptions` → `results/revision3/R10/r10_deadline_assumptions.json`.
D_r = (R − v·t_r − v²/2a)/v (as B1) for t_r ∈ {1.5, 2.5, 3.5} s × a ∈ {2.5, 3.4, 6.0} m/s², applied to seed-mean
p95 TIR in the 4 headline cells (q* and guard fixed); agent seeds 0–4, τ=0.5. D_r < 0 is treated as 0.

| t_r (s) | a (m/s²) | D rural / urban_nlos (s) | best fixed | agent cells matched s0–s4 | changes vs paper (2.5 s, 3.4 m/s²) |
|---|---|---|---|---|---|
| 1.5 | 2.5 | 14.92 / 2.08 | slotted_1p(8) | 1/1/3/3/3 | none |
| 1.5 | 3.4 | 16.39 / 2.89 | slotted_1p(20) | 1/1/3/3/3 | best fixed |
| 1.5 | 6.0 | 18.16 / 3.86 | slotted_1p(20) | 1/1/3/3/3 | best fixed |
| 2.5 | 2.5 | 13.92 / 1.08 | counter_based(3) | 1/1/3/3/3 | best fixed |
| **2.5** | **3.4** | **15.39 / 1.89** | **slotted_1p(8)** | **1/1/3/3/3** | (paper) |
| 2.5 | 6.0 | 17.16 / 2.86 | slotted_1p(20) | 1/1/3/3/3 | best fixed |
| 3.5 | 2.5 | 12.92 / **0.08** | dvcast(30) | 1/1/**2/2/2** | best fixed; urban d=20: **no scheme qualifies**, seeds 2–4 lose the cell |
| 3.5 | 3.4 | 14.39 / 0.89 | counter_based(3) | 1/1/3/3/3 | best fixed; urban d=20 oracle counter 0.533 → p_persist 0.550 |
| 3.5 | 6.0 | 16.16 / 1.86 | slotted_1p(8) | 1/1/3/3/3 | none |

**What changes.** The rural deadline (12.9–18.2 s) never binds: no corridor oracle, fixed cost or agent
verdict moves, and rural d=20 regret is unchanged in all nine configurations. Everything that moves comes from
urban_nlos, whose one-NLOS-hop deadline is 0.08–3.86 s. The best single fixed setting changes in 6 of the 8
alternatives (it is chosen across cells and sensitive to the urban bound). Agent verdicts change only at the
most pessimistic corner (t_r = 3.5 s, a = 2.5 m/s²), where no scheme can meet q* within 0.08 s in urban d=20.

## R11. Absolute miss level the guard admits

Regenerate: `python -m experiments.revision3.r11_absolute_guard` → `results/revision3/R11/r11_absolute_guard.{csv,json}`.
Admitted level = best actionable miss of any baseline setting in the cell + tolerance. Points are those
Algorithm 1 selects at q* = 0.95 × ceiling (agent: its qualifying point, seeds 0–4, τ=0.5; "–" = no qualifying point).

| cell | best any setting | admitted (tol 0.02 / **0.05** / 0.10) | per-cell oracle point | best fixed point | agent points s0–s4 (tol 0.05) |
|---|---|---|---|---|---|
| rural d=2 | 0.529 | 0.549 / **0.579** / 0.629 | 0.565 (0.534 at tol 0.02) | 0.565 | – (no seed qualifies) |
| rural d=20 | 0.314 | 0.334 / **0.364** / 0.414 | 0.317 | 0.317 | 0.332 / 0.351 / 0.344 / 0.340 / 0.358 |
| rural d=80 | 0.318 | 0.338 / **0.368** / 0.418 | 0.327 | 0.319 | – / – / 0.353 / 0.355 / 0.350 |
| urban d=20 | 0.067 | 0.087 / **0.117** / 0.167 | 0.078 | 0.083 | – / – / 0.099 / 0.110 / 0.113 |

At tolerance 0.02 the agent points that still qualify at rural d=20 are 0.319–0.332, and seed 2's rural d=80 point is 0.333.
The reference points sit within 0.003–0.036 of the best achievable miss rate; the agent's
qualifying points use 0.015–0.046 of the 0.05 tolerance. That is why R2 found the tolerance decisive.
The guard is a per-vehicle constraint stated in absolute terms: e.g. at rural d=20 up to 36.4% of vehicles that
could have been saved may miss their reaction deadline at the paper's tolerance.


## R12. Random-gate control, full-model seeds 0–4

Regenerate: per seed, `python -m experiments.revision1.agent_eval --checkpoint <ref_s{k} ckpt_final.pt> --name ref_s{k}_randomgate
--random-gate-from results/revision1/agent_runs/ref_s{k}.csv` (rate-matched to that seed's own learned-gate fallback
rate per (cell, bias), exactly as C3), then `python -m experiments.revision3.r12_random_gate` →
`results/revision3/R12/r12_random_gate.json`. Each arm is scored alone against the 55 baseline settings (Algorithm 1).

| seed | learned gate (τ=0.5): cells / rural d=20 cost (regret) | network only (τ=0) | random gate, same rate |
|---|---|---|---|
| 0 | 1 / 0.717 (+1.8%) | 2 / 0.706 (+0.3%) | **3** / 0.782 (+11.1%) |
| 1 | 1 / 0.725 (+3.1%) | 2 / 0.856 (+21.7%) | **3** / 0.748 (+6.3%) |
| 2 | 3 / **0.554 (−21.3%)** | 3 / 0.606 (−13.9%) | 3 / 0.713 (+1.3%) |
| 3 | 3 / **0.668 (−5.2%)** | 2 / 0.834 (+18.4%) | 3 / 0.953 (+35.4%) |
| 4 | 3 / **0.534 (−24.1%)** | 3 / 0.845 (+20.0%) | 3 / 0.762 (+8.3%) |
| mean ± SD | 2.20 ± 1.10 cells; −9.1% ± 12.8 | 2.40 ± 0.55; +9.3% ± 15.6 | **3.00 ± 0.00**; +12.5% ± 13.3 |

Cells matched by the random gate on every seed: rural d=20, rural d=80, urban d=20 (never rural d=2).

**Reading (refines C3).** The learned confidence gate is not useless, but it is not what makes
the policy qualify. A coin that defers the same fraction of decisions qualifies in 3 of 4 cells on
all five seeds, more reliably than the learned gate (1–3 cells). Where every arm qualifies (rural
d=20), the learned gate is cheaper than the random gate on all five seeds (by 4.2–29.8 points of
regret) and cheaper than the network alone on four of five. So the entropy gate picks *cheaper*
decisions to defer, and the random gate picks *safer* ones. On 5 seeds this is a trade-off
between cost and robust qualification, not an across-the-board gain from learning which decisions
to hand to the fallback.

## Run log (revision 3)

* 2026-10-18 11:13: R8 (rungs B, C × seeds 0, 1) and R12 launched via Task Scheduler
  (`launch_revision1.ps1 -Revision3` → `watchdog_revision3.ps1`, every 20 min), one hidden console per job.
* 2026-10-18 12:38: **R9 first pass invalidated by my own bug, caught before reporting.** The calibrated-loss
  knob drew its random numbers from the engine's `mac` stream. Enabling it therefore shifted every later MAC
  draw, so the rural d=2 and d=20 cells changed although their extra-loss probability is exactly zero (receivers
  there have < 250 neighbours in carrier-sense range); the agent's cells matched read 1/1/3/3/3 → 1/1/1/1/1.
  Fix: the knob draws from its own named stream (`common/seeding.py` "extra_loss"; stream keys are name hashes,
  so no other stream moves). Verified: with the knob on, rural d=2 and d=20 runs are now bit-identical to default,
  while urban d=20 and rural d=80 change. The fit and the rerun were redone; the invalid outputs are kept,
  labelled, in `results/revision3/R9/invalid_rng_drift/` (README there). R8 training is unaffected (knob off).


## R9. Calibrated-channel stress test

Regenerate: `python -m experiments.revision3.r9_calibrated_channel fit` → `results/revision3/R9/r9_fit.json`;
`... rerun` → `r9_baseline_runs.csv` (55 settings × 4 headline cells × 10 eval seeds) and `r9_agent_ref_s{0..4}.csv`
(full model, τ=0.5, sampled, biases −2…+3); `... analyse` → `r9_analysis.json`. Code: `sim/mac.py`
(`extra_loss_*`, off by default), `sim/engine.py` (applied after the background-beacon check, own RNG stream
`extra_loss`), `common/seeding.py`. With the knob off every run is bit-identical to before; with it on, cells where
the loss is zero are bit-identical too (verified).

**Fitted function.** Extra per-reception loss
**p(n) = 0.9004 · clip((n − 250)/(300 − 250), 0, 1)**, n = the receiver's neighbours within carrier-sense range.
Ramp start from the validation geometry: n ≈ 12.4 × density in Amador's 8-lane, 778 m scenario, so the published
step between 20 and 30 veh/km/lane lies between n ≈ 249 and 373. Ramp end and level fitted jointly (n1 ∈ {300, 330, 373},
p_max by bisection, 10 seeds; the pair with the smallest maximum deviation over 30–50 kept; 30-seed verification):

| veh/km/lane | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| Amador ETSI CBF | 0.9998 | 0.9961 | 0.9280 | 0.9371 | 0.9372 |
| ours, calibrated (30 seeds) | 0.9977 | 0.9977 | 0.9172 | 0.9386 | 0.9428 |
| deviation | −0.0021 | +0.0016 | **−0.0108** | +0.0015 | +0.0056 |

**The 0.01 criterion is met at 4 of 5 densities and missed at d = 30 by 0.0008.** No member of this functional
family does better: the one-parameter fit (n1 = 373, p_max 0.910) missed at both 30 (+0.011) and 40 (−0.011); its
outputs are kept in `R9/superseded_1param_fit/`, and every finding below is identical under it. The published curve is
non-monotone (0.928 at 30, 0.937 at 40), which a monotone loss in n can only approximate. Physically the fit is
harsh: about 90% of receptions are lost once a receiver has ≥ 300 neighbours in carrier-sense range. It is a
stress test, not a validated channel model.

**Where the knob bites.** Receivers in rural d=2 and d=20 never exceed 250 neighbours, so those cells are
bit-identical to the default channel. Rural d=80 and urban_nlos d=20 are affected.

| | default channel | calibrated channel | same / changed |
|---|---|---|---|
| best single fixed setting | dvcast(n_slots=30) | slotted_1p(n_slots=8) | **changed** |
| rural d=2: ceiling / oracle | 0.675 / dvcast 1.560 | 0.675 / dvcast 1.560 | same (cell unaffected) |
| rural d=20: ceiling / oracle | 0.925 / greedy 0.704 | 0.925 / greedy 0.704 | same (cell unaffected) |
| rural d=80: ceiling / oracle | 0.915 / weighted_p 1.512 | 0.876 / **dvcast 5.030** | **changed** |
| urban d=20: ceiling / oracle | 0.975 / counter 0.533 | **0.484** / **slotted_1p 1.398** | **changed** |
| coverage saturation at rural d=20 (best RWCR spread across 6 families excl. p-persistence) | 0.0025 | 0.0025 | same (< 0.01) |
| rural d=80 family spread | 0.008 | 0.136 | saturation **breaks** in the dense corridor |
| urban d=20 family spread | 0.023 | 0.413 | saturation **breaks** |
| R4 adaptive pick meets eval target (rural d=2 / 20 / 80 / urban 20) | no / yes / no / yes | no / yes / no / **no** | **changed** in urban d=20 (counter_based(1): 0.533 → 0.979, below q*) |
| agent cells matched s0–s4 | 1 / 1 / 3 / 3 / 3 | **1 / 1 / 1 / 1 / 1** | **changed**: seeds 2–4 lose rural d=80 and urban d=20 |
| agent rural d=20 cost (regret) s0–s4 | 0.717 (+1.8), 0.725 (+3.1), 0.554 (−21.3), 0.668 (−5.2), 0.534 (−24.1) | identical | same (cell unaffected) |
| agent rural d=20 margin s0–s4 | +1.9 / +0.7 / +31.9 / +9.4 / +36.8% | +25.3 / +23.9 / +62.2 / +34.6 / +68.3% | larger (the best fixed setting changed to slotted_1p(8), costing 0.898 there) |

**Reading.** If the simulator is made as lossy in dense traffic as Amador's ETSI-CBF reference implies, every
dense-cell conclusion moves. Coverage no longer saturates in urban_nlos d=20 or rural d=80; their oracle schemes
change; the per-cell oracle cost at rural d=80 more than triples; and the agent's success outside rural d=20
(seeds 2–4) does not survive. The sparse and medium corridor (rural d=2, d=20), including every rural d=20 agent
number, is untouched, because the calibrated loss is zero there. This is the precise sense in which the
headline corridor result is robust to the validation gap and the dense-cell results are not.
