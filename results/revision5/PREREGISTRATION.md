# Revision 5 — pre-registration

Committed on branch `revision-5` **before any revision-5 evaluation run**. The machine-readable twin is
`protocol.json`. The thresholds q* and m* are computed from CAL runs only and frozen in `thresholds.json`, which
is committed before any SEL or EVAL run exists. Any later change goes in `DEVIATIONS.md` with a timestamp and a reason.

## Seed sets (pairwise disjoint; verified against the training pool and the normalisation seeds)

| set | seeds | used for |
|---|---|---|
| CAL | 300–309 (10) | target and guard calibration only |
| SEL | **400–419** (20) | operating-point selection for every method |
| EVAL | 200–229 (30) | final scoring only; never used for any choice |
| LEGACY | 0–9 | comparison with the paper only |

The prompt specified SEL = 100–119, which lies inside the policy training seed pool (100–131), so selecting the
policy's operating point on SEL would select on training traffic. Per the prompt's own abort rule, the user chose
fresh SEL seeds 400–419 before any run (2026-10-18). Consequence: the existing classical records on seeds 100–109
are not reused for SEL. The verified disjointness check (`experiments/revision5` session log): CAL, SEL, EVAL and
LEGACY are pairwise disjoint and disjoint from the training pool 100–131 and the normalisation seeds 900–904.

## Cells

The headline cells are rural_highway d = 2, 20, 80 and urban_nlos d = 20. The 24-cell grid is rural_highway,
urban_nlos and urban_grid × d ∈ {1, 2, 3, 5, 10, 20, 40, 80}. All cells use clear weather and the fog_bank hazard.

## Target and guard (pre-specified, from CAL only)

* q* = 0.95 × the maximum, over the 55 swept baseline settings, of mean oracle RWCR on CAL.
* m* = the minimum, over the same settings, of mean actionable deadline miss rate on CAL, + 0.05.
* Absolute safety variant: the same q*, with actionable miss ≤ 0.40 absolute.

## Selection rule (identical for every method)

On SEL, take the cheapest **realised** setting (no interpolation) with mean RWCR ≥ q*, mean actionable miss ≤ m*,
and seed-mean p95 TIR ≤ D. If nothing qualifies, the method is "no pick" in that cell.

* D variants: D = ∞, and D = the derived deadline D_r (rural 15.39 s, urban_grid 26.65 s, urban_nlos 1.89 s).
* Settings per method: classical = (scheme, parameter) over the 55 settings; learned policy = suppression bias
  ∈ {−2, …, 3} per training seed; new rule-based baselines = their grids (below).
* Adaptive classical = the cheapest qualifying classical setting per cell. The contextual per-density lookup
  coincides with it here (one weather and one hazard per cell) and is reported as such.

## Scoring on EVAL

* A chosen setting qualifies if its EVAL mean RWCR ≥ q* and its EVAL mean actionable miss ≤ m*, using the frozen
  CAL thresholds. The primary cost is the realised EVAL cost.
* Secondary, clearly labelled: the EVAL-hindsight oracle ("optimistic benchmark"), and interpolated costs
  ("estimate, not an implementable configuration"), with a with/without-interpolation table.
* Primary safety columns:
  * absolute actionable miss rate;
  * actionable coverage = 1 − miss;
  * share of at-risk vehicles warned before their own deadline D_r (per vehicle, end to end);
  * p95 TIR (median TIR is reported separately and labelled).

## Primary hypotheses (Holm across H1, H2, H3; α = 0.05)

### H1 — classical rankings

No single classical scheme family is the cheapest qualifying family in all four headline cells.

* Test: 10,000 hierarchical bootstrap replicates, resampling CAL, SEL and EVAL seeds. In each replicate, recompute
  q* and m*, each family's SEL selection, and its EVAL qualification and cost.
* Statistic: the share of replicates in which one family wins all four cells.
* Decision: supported if the Clopper–Pearson 95% upper bound of that share is < 0.05.
* p-value: exact one-sided binomial test against 0.05.

### H2 — policy vs retuned classical, deployable on both sides

* Primary statistic: the pooled mean, over headline cells where both qualify, of (mean policy cost / mean adaptive
  cost − 1). Per-cell values are also reported.
* Uncertainty: a hierarchical bootstrap over the full model's 10 training seeds × EVAL seeds, recomputing the
  selections in every replicate (10,000 replicates, percentile 95% CI).
* Also reported: P(policy cost < adaptive cost), and the qualification-rate difference with a Newcombe/Wilson
  interval.
* Decision rules:
  * two-sided test;
  * superiority only if the 95% CI upper bound is < 0;
  * equivalence only by TOST with a ±5% margin;
  * a non-significant result is never called "no difference".
* p-value: bootstrap two-sided.

### H3 — packet-level agreement with Veins

SUMO mobility, identical traces and seeds, 4 headline cells, 10 seeds per cell. All of the following are required:

* Spearman ρ of classical matched costs across cells and schemes ≥ 0.8, with a lower 95% bootstrap bound ≥ 0.6;
* per-hop PDR-vs-distance within 0.05 absolute in every 50 m bin up to the nominal range;
* channel busy ratio within 20% relative;
* median single-hop delay within one epoch (100 ms);
* identical H1 verdict in both simulators.

The Holm-tested component is the bootstrap p for ρ ≤ 0.6.

Decision rule:
* **Pass:** the custom simulator is validated for these metrics in these cells.
* **Fail:** add a sub-epoch timing mode (13 µs slots, AIFS/backoff; default unchanged), calibrate it only on the
  validation metrics, freeze it, and re-run.
* **Still failing:** the classical audit computed in Veins becomes primary, and custom-simulator claims are
  restricted accordingly.

Everything else is **exploratory**.

## New non-learning baselines (same selection protocol)

* **RAR** (relevance-aware rule):
  * rule: p = clip(a·ρ_c + b·(1 − n/n_max) + c, 0, 1), with n_max = 30 and slotted deferral by distance (5 slots);
  * inputs: exactly the policy's causal inputs (relevance, neighbour count, distance to sender, message age);
  * grid: a ∈ {0.5, 1, 2} × b ∈ {0, 0.5, 1} × c ∈ {−0.25, 0, 0.25} = **27 settings**, cut from ~60 by the user
    before any run.
* **SCF-slotted:**
  * rule: slotted 1-persistence, plus carry and rebroadcast on a new neighbour until a carry timeout;
  * grid: n_slots ∈ {3, 8, 20, 50} × timeout ∈ {2, 5, 10} s = **12 settings**.

## Training (same config and 1,000-update budget; only the seed changes)

Priority order:

1. full model seeds 5–9;
2. rung B seeds 2–4;
3. rung C seeds 2–4;
4. GCN, MLP and no-relevance seeds 2–4;
5. budget extension: full model seeds 0–2 to 2,000 updates (checkpoints at 1,250, 1,500, 1,750, 2,000).

The star variant is dropped unless time remains.

## Pre-specified cuts (user, before any run; the full protocol projected 5–6 days)

* 24-cell policy evaluation: full-model seeds 0–4 (not 0–9);
* 24-cell stress-test policy evaluation: seeds 0–2;
* RAR 27 settings and SCF 12 settings.

Never cut: the H1, H2 and H3 runs, and every 4-cell experiment.

## Precision justification (from existing records, before any new run)

Source: `results/revision5/precision/precision.json` (`python -m experiments.revision5.precision`). Rural d=20,
the policy (training seeds 0–4, each at its legacy qualifying bias) vs the adaptive pick slotted_1p(30), on
legacy seeds 0–9. Paired relative cost difference: mean −0.140. Variance components: σ_train = 0.125, σ_eval = 0.069.

| n_eval | n_train | SE | 95% CI half-width | MDE (80% power) |
|---|---|---|---|---|
| 10 | 5 | 0.057 | 0.111 | 0.159 |
| 10 | 10 | 0.040 | 0.078 | 0.112 |
| 20 | 5 | 0.056 | 0.110 | 0.157 |
| 20 | 10 | 0.040 | 0.078 | 0.111 |
| 30 | 5 | 0.056 | 0.110 | 0.157 |
| 30 | 10 | 0.040 | **0.078** | 0.111 |

**30 EVAL × 10 training seeds does not reach the ≤ 5% half-width target. The achieved precision is ±7.8%.**
Training-seed variation dominates, so more EVAL seeds barely help. Reaching ±5% would take about 24 training seeds.

Caveats:
* these legacy biases were chosen on the same seeds;
* σ_train comes from only 5 seeds;
* rural d=20 only.

The ±5% TOST equivalence margin in H2 is therefore unlikely to be demonstrable, and this is stated in advance.
