# Revision 5 — analysis specification (operational choices)

Committed **before any H1, H2 or H3 statistic was computed**. `PREREGISTRATION.md` and `protocol.json` define the
hypotheses, seed sets, thresholds and decision rules; this file only fixes the operational details they leave open.
Code: `experiments/revision5/analysis/`. Every output carries a provenance stamp (`experiments/revision1/provenance.py`).

## Common

* **Data:** committed per-run rows in `results/revision5/runs/` only. Seed sets CAL 300–309, SEL 400–419, EVAL 200–229.
* **Metrics:** RWCR = `rwcr` (oracle); cost = `tx_per_at_risk_informed`; miss = `actionable_deadline_miss_rate`;
  p95 TIR = `tir_p95_s`. Means are over seeds; an infinite cost (nobody at risk informed) stays infinite.
* **Thresholds:** q*, m* per cell from `thresholds.json` (frozen CAL), except where a hypothesis says they are
  recomputed in each bootstrap replicate (H1).
* **Selection rule (all methods):** on SEL, the cheapest realised setting with mean RWCR ≥ q*, mean miss ≤ m* and
  seed-mean p95 TIR ≤ D; otherwise "no pick". **D = ∞ is primary**; D = D_r (rural 15.39 s, urban_grid 26.65 s,
  urban_nlos 1.89 s) is reported alongside with identical machinery. Ties in cost go to the lower setting index.
* **Qualification on EVAL:** EVAL mean RWCR ≥ q* and EVAL mean miss ≤ m*. Absolute variant: miss ≤ 0.40.
* **Families (H1):** flooding, p_persistence_03, slotted_1p, weighted_p, counter_based, greedy_farthest, dvcast
  (the 55 settings of `analysis.pareto.POLICY_SWEEPS`). A family's pick in a cell = the selection rule applied to its
  own settings.
* **Bootstrap:** 10,000 replicates, seed resampling with replacement, the same resampled seed indices used for all
  cells of a replicate; percentile 95 % intervals; fixed RNG seed 20261031.

## H1

Per replicate: resample CAL, SEL and EVAL seeds; recompute q*, m* per cell from the resampled CAL rows; each family's
SEL pick; its EVAL qualification and mean cost. The cell's winner is the family with the lowest EVAL cost among
families whose pick qualifies (none if none qualifies). Statistic: share of replicates in which a single family wins
all four headline cells. Decision and p-value exactly as pre-registered (Clopper–Pearson 95 % upper bound < 0.05;
exact one-sided binomial test against 0.05, lower tail).

## H2

* **Methods:** policy = full-model training seeds 0–9 (rows: `agent__{SEL,EVAL}__grid24__default__fallback.csv` for
  seeds 0–4, `…__headline__…__s5_9.csv` for 5–9); each seed picks its own bias on SEL. Adaptive classical = the
  cheapest qualifying classical setting on SEL (over all 55). Rule comparator = the same with the 39 rule settings.
* **Policy cost in a cell** = mean over training seeds with a pick of their pick's EVAL mean cost. The policy
  qualifies in a cell if ≥ half of the seeds have a pick and the mean over those seeds of EVAL RWCR ≥ q* and of EVAL
  miss ≤ m*.
* **Primary statistic:** pooled mean over headline cells where both qualify of (policy cost / adaptive cost − 1).
* **Bootstrap:** resample training seeds (10), SEL seeds (20) and EVAL seeds (30); recompute every selection;
  q*, m* stay frozen (H2 scores against the frozen CAL thresholds). CI = percentile 95 %; p = 2·min(P(T ≤ 0),
  P(T ≥ 0)); superiority iff CI upper < 0; TOST: equivalence iff the 90 % interval lies in (−0.05, 0.05).
* **P(policy cheaper)** = share of replicates with T < 0. Per-cell values are reported too.
* **Qualification rates:** policy = qualifying (cell, training seed) picks / 40; adaptive = qualifying cells / 4;
  Wilson intervals; difference with the Newcombe hybrid-score interval.
* **Secondary:** the same against the best rule baseline (cheapest qualifying of the 39 rule settings), and the
  24-cell version with seeds 0–4 (no Holm).

## H3 (definitions fixed in DEVIATIONS D3)

PDR vs distance, single-hop delay, busy ratio and loss rates as in D3, per headline cell, Veins vs custom-on-SUMO on
the paired seeds 200–209. Matched cost: Algorithm 1 per (cell, family) inside each simulator — q* = 0.95 × max mean
RWCR over the 55 settings; cost at q* from the family's mean curve by `PolicyCurve.overhead_at` (interpolated
crossing). Spearman ρ over the (cell, family) pairs finite in both; bootstrap over the 10 seeds for the CI and
p = P(ρ* ≤ 0.6).

**Cross-fitted H1 in each simulator (D1):** folds 200–209 / 210–219 / 220–229. For fold f, q*, m* and every family's
pick come from the other two folds (20 seeds, CAL and SEL roles); the pick is scored on fold f. A family qualifies in
a cell if its pick qualifies on the held-out fold in all three folds; its cross-fitted cost is the mean held-out cost
over the 30 seeds. Winner and statistic as in H1; the bootstrap resamples seeds within each fold. The H3 criterion is
"the same verdict (supported / not supported) in both simulators".

## Holm

Over the three pre-registered p-values: H1 (binomial), H2 (bootstrap two-sided), H3 (bootstrap p for ρ ≤ 0.6).
