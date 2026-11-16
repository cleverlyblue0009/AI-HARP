# Revision 8 — status (final)

Branch `revision-8` (from `origin/revision-7` at a6dea21). **Addendum push commit: `53eefc3`**
(`results/revision8/ADDENDUM.md`; hash recorded in its header by a0d94fb, which changes nothing else). Deviations:
`results/revision8/DEVIATIONS.md` (D12–D15). Every number below is read from the committed file named next to it.

## Answers

| Hypothesis | Verdict | Source |
|---|---|---|
| **H4** STRUCT-plain vs the revision-5 adaptive classical comparator (custom simulator, Tier B 210–229) | **not supported**: (i) qualifying cells 3 vs 4 fails; (ii) one-sided 95 % upper bound of T +0.104 ≥ +0.10 fails; raw p 0.053, Holm 0.106 | `analysis/h4.json`, `analysis/holm.json` |
| **H5** STRUCT-rand survives Veins where STRUCT-plain does not (Tier B 210–219, D12) | **not supported**: rule met in neither cell (rural d=20: rand 0/5, plain 0/5; urban NLOS d=20: rand 1/5, plain 0/5); p_H5 1.0, Holm 1.0 | `analysis/h5.json`, `analysis/holm.json` |
| Experiment C: geometric neighbour table (AI-HARP, Veins rural d=20, Tier A) | **does not explain the failure**: qualification 0/5 (rule: ≥ 3/5) | `analysis/exp_cd.json` |
| Experiment D: one-epoch coupling lag removed (same) | **does not explain the failure**: 0/5 | `analysis/exp_cd.json` |

## 1. What was implemented

| File | What |
|---|---|
| `agents/structured.py` | STRUCT, registry `struct`: 21 parameters, clipped-linear p and delay heads (author decision 1; the sigmoid heads cannot contain RAR or slotted_1p, `probe_heads.json`), k ∈ {1,2,3}, cost knob δ (p − 0.15 δ) |
| `agents/train_structured.py` | CMA-ES trainer (cma 4.6.0, popsize 13, 80 generations), Deb's rule on causal coverage targets, R1–R5 sampling, per-generation checkpoint with exact resume, CPU model in `run_meta.json` |
| `sim/engine.py`, `common/seeding.py` | R1 `dup_count_per_frame`, R2 `nb_table_error`, R3 `obs_lag_steps`, R5 `pdr_perturb`, all off by default; R4 is the existing `backend` option |
| `tests/test_structured.py`, `tests/test_randomisation_flags.py`, `tests/test_oracle_isolation.py` (+ structured modules, transitive), `tests/test_revision8_analysis.py` | gates (below) |
| `experiments/revision8/` | probe, warm start, Veins thresholds, traces, `sweep_struct` (+ slices/merge), `train_summary`, `analysis/h4.py`, `analysis/h5.py` (H5, C/D, Holm), `veins/` (app with `struct`, `stepOffset`, `sendHeard`; `decision_server8`, `run_veins8`, `queue8`, `validate8`, `score8`, `ci_veins`, `assemble_ci`), manifest |
| `.github/workflows/r8-{train,eval,veins}.yml`, `ci/r8-*.json` | D14: training, evaluation and Veins shards on GitHub-hosted runners, outputs on branch `r8-ci` |
| `reproduce_revision8.{sh,ps1}`, `results/revision8/LAPTOP_RUNBOOK.md` | reproduction; laptop runbook (superseded for Veins by D14) |

All analysis code (H4, H5, Holm, shard assembly) was committed before the data it reads existed.

## 2. Equivalence tests (`tests/test_structured.py`, all pass; full suite 665 passed, 1 skipped)

* ≥ 1,000 first-reception contexts recorded from RAR (1, 0.5, 0) runs on training-pool seeds 100–102 × the four headline
  cells: STRUCT vs slotted_1p n_slots 5 and 50 (and 1, 2): **0 mismatches** (action, delay, cancel threshold).
* STRUCT vs RAR (1, 0.5, 0): **0 decision mismatches, 0 contexts with different policy-stream consumption**; all 27 RAR
  grid settings: 0 mismatches.
* Whole runs, slotted_1p n_slots 5 and 50 vs STRUCT at the matching parameters: **byte-identical engine output** on
  seeds 100–102 in each of the four headline cells (24 pairs).
* No engine component other than the policy reads the policy RNG stream: runtime guard (raises on any access) in all four
  cells, plus a static check of `sim/`.
* Not exactly representable (docstring, READING_NOTES §8): counter_based (random delay), slotted_1p n ∈ {3, 8, 12, 20, 30},
  weighted_p/p_persistence, greedy/DV-CAST/SCF; ETSI CBF only inside its area (= slotted_1p n=1).
* Oracle isolation: `agents.structured`, `agents.train_structured` import neither `hazard.oracle`, `privileged` nor
  `analysis.metrics`, transitively.

## 3. Randomisation ranges and provenance (unchanged from ADDENDUM §3)

| Axis | Range | Provenance |
|---|---|---|
| R1 | per-frame duplicate counting on/off, p = 0.5 | MEASURED (Tier A): `results/revision6/REVISION6_REPORT.md:78`; `results/revision7/ADDENDUM.md:22-27` |
| R2 | p_miss, p_stale ~ U[0, 0.5], ring 1.15 R | **[ASSUMED]**; only evidence 7/269 equal counts (`results/revision6/veins/port_validation_rules.json`); no neighbour counts in the committed raw outputs |
| R3 | lag 0 / 1 epoch | MEASURED: 269/269 (`port_validation_rules.json`) |
| R4 | fallback / SUMO 1.19.0 | SUMO 1.19.0 runs in the cloud and on runners; a committed classical SUMO row reproduced exactly; training-pool traces only |
| R5 | u ~ U[0, 1] × per-bin PDR gap | MEASURED (Tier A): `results/revision5/analysis/h3.json` |

Flag tests: flag-off output byte-identical to the revision-7 engine source (a6dea21) for slotted_1p, RAR and
counter_based in all four headline cells; zero-effect settings byte-identical to off; committed revision-5 rows
reproduced (counts and costs exactly; `rwcr` ≤ 1 ulp and `risk_peak_corr` ~1e-15 differ from the Windows-written rows,
identically with and without the revision-8 changes).

## 4. Training (`results/revision8/train_summary.json`, `results/revision8/train/<run>/`)

All 20 runs: 80 generations, NumPy 1.26.4, cma 4.6.0. Warm start (SEL only): RAR (1, 1, 0.25), qualifying on SEL in 4/4
headline cells (`warm_start.json`). Feasibility = share of candidates meeting all four causal coverage targets,
mean over the last 10 generations (training pool only).

| Run | Machine | CPU | Feasibility, last 10 | Wall (min) |
|---|---|---|---|---|
| plain_s0 | cloud session | Intel Xeon @ 2.10 GHz (not in its `run_meta.json`; recorded here) | 0.55 | 58 |
| plain_s1 | cloud session | same | 0.37 | 40 |
| plain_s2 | GitHub runner | Intel Xeon 6975P-C | 0.38 | 53 |
| plain_s3 | GitHub runner | AMD EPYC 9V45 | 0.46 | 27 |
| plain_s4 | GitHub runner | Intel Xeon 6975P-C | 0.45 | 43 |
| plain_s5 | GitHub runner | AMD EPYC 7763 | 0.24 | 74 |
| plain_s6 | GitHub runner | Intel Xeon 6975P-C | 0.40 | 36 |
| plain_s7 | GitHub runner (rerun, see below) | Intel Xeon 6975P-C | 0.60 | 25 |
| plain_s8 | GitHub runner | Intel Xeon 6975P-C | 0.38 | 43 |
| plain_s9 | GitHub runner | Intel Xeon 6975P-C | 0.45 | 45 |
| rand_s0 | GitHub runner | Intel Xeon 6975P-C | 0.07 | 58 |
| rand_s1 | GitHub runner | AMD EPYC 7763 | 0.10 | 77 |
| rand_s2 | GitHub runner | Intel Xeon 6975P-C | 0.23 | 48 |
| rand_s3 | GitHub runner | Intel Xeon 6975P-C | 0.08 | 39 |
| rand_s4 | GitHub runner | Intel Xeon 6975P-C | 0.05 | 42 |
| rand_s5 | GitHub runner | Intel Xeon 6975P-C | 0.10 | 51 |
| rand_s6 | GitHub runner | AMD EPYC 7763 | 0.06 | 75 |
| rand_s7 | GitHub runner | Intel Xeon 6975P-C | 0.01 | 54 |
| rand_s8 | GitHub runner | Intel Xeon 6975P-C | 0.16 | 38 |
| rand_s9 | GitHub runner | Intel Xeon 6975P-C | 0.18 | 44 |

* Exact resume verified (interrupted-then-resumed = uninterrupted, history and θ); two invocations give identical
  generations; `spawn` = `fork` (one generation, plain_s3). The cloud container was reclaimed several times; plain_s0/s1
  resumed exactly from per-generation checkpoints.
* **plain_s7 was trained twice.** Its first run (job 133172223491) finished, but its outputs were lost: the push to
  `r8-ci` was refused (a runner may not create a branch containing workflow files) and the workflow masked the failure.
  Only that job's log (training-pool fitness lines) was seen. The committed plain_s7 is the rerun (same code, seed and
  budget; possibly a different CPU type than the lost run). The workflow bug is fixed.
* STRUCT-rand meets the causal targets on far fewer candidates (0.01–0.23) than STRUCT-plain (0.24–0.60) under the
  randomised episodes.

## 5. Custom simulator: H4 (`results/revision8/analysis/h4.json`, B = 10,000, RNG 20261116)

Rows: `results/revision8/runs/struct_plain__{SEL,EVAL}__headline__default__fallback.csv` (4,800 + 7,200; 20 runner
slices merged, every task exactly once). Frozen revision-5 q*, m*; comparator = revision-5 adaptive classical, SEL-selected.

**Tier B (210–229), primary:**

| Cell | STRUCT-plain (seeds picking on SEL) | EVAL cost | Comparator (SEL pick) | EVAL cost | Relative |
|---|---|---|---|---|---|
| Rural d=2 | qualifies (5/10) | 2.127 | DV-CAST n=20, qualifies | 1.763 | +20.6 % |
| Rural d=20 | qualifies (7/10) | 0.550 | greedy_farthest 32, qualifies | 0.651 | −15.5 % |
| Rural d=80 | qualifies (10/10) | 0.735 | weighted_p 0.1, qualifies | 0.994 | −26.1 % |
| Urban NLOS d=20 | **does not qualify** (4/10 pick) | – | counter_based 1, qualifies | 0.535 | – |

T = −0.070 (3 cells); one-sided 95 % upper bound **+0.104** (margin +0.10); percentile 95 % CI [−0.456, +0.158];
p_noninferiority 0.053; revision-5 two-sided p 0.319 (no superiority); 188 of 10,000 replicates undefined.
Qualification (seed, cell) 24/40, Wilson [0.446, 0.737]. **Decision: (i) 3 < 4 fails, (ii) fails → H4 not supported.**

**Tier A (200–209), descriptive:** T −0.219 over 2 cells (STRUCT 2 vs comparator 3 qualifying), upper bound +0.105,
p_noninferiority 0.052. **Vs the best rule (descriptive):** Tier B +1.119 (rural d=80 only, the one cell both
qualify), p two-sided 0.015; Tier A +0.736 (2 cells). **AI-HARP reference on the same tiers (revision-5 rows,
descriptive):** Tier B qualifies in rural d=20 (0.695) and urban NLOS d=20 (0.840), not in rural d=2 or d=80.

**STRUCT-rand, custom simulator (descriptive, `analysis/h4_rand_descriptive.json`):** Tier B qualifies in 2 cells
(rural d=20 0.678, rural d=80 1.306) vs the comparator's 4; T +0.177 (rural d=20 +4.1 %, d=80 +31.4 %); Tier A +0.161.

## 6. Veins: H5 (`results/revision8/analysis/h5.json`)

Runner shards (D14): 20 / 20 accepted (`veins/h5B_rows_merge.json`): every runner's gate replay byte-identical to the
revision-6 gate log (`707d6533…`, `veins/ci/*/gate.json`); **1,200 decision logs, 441,252 STRUCT requests re-sent
offline, 0 mismatches** (`veins/ci/*/trace_check.json`); 1,200 / 1,200 replays accepted (exit 0, identical SUMO
trace, gated binary sha256 `6b032de0…`). Raw OMNeT++ outputs (357 MB) in `veins/ci/*/raw/`.

Frozen thresholds from Tier A classical replays (`veins_thresholds.json`): rural d=20 q* 0.879, m* 0.404; urban NLOS d=20
q* 0.498, m* 0.602.

| Cell | STRUCT-rand qualifying | STRUCT-plain qualifying | Rule met | Fisher p (one-sided) |
|---|---|---|---|---|
| Rural d=20 | 0 / 5 | 0 / 5 | no | 1.0 |
| Urban NLOS d=20 | 1 / 5 (rand_s2, matched cost 0.826) | 0 / 5 | no | 0.5 |

p_H5 = min(1, 2 × 0.5) = 1.0. **H5 not supported.**

**Holm over H4 and H5 (`analysis/holm.json`):** raw H4 0.053, H5 1.0; adjusted H4 0.106, H5 1.0; neither supported.

**Descriptive, post hoc (read after the registered result; no change made):** in rural d=20 every STRUCT-rand seed reaches
mean RWCR above q* in Veins (best 0.897–0.933 per seed; STRUCT-plain 0.632–0.815), at low cost (0.28–0.46), but its
actionable deadline-miss rate is 0.54–0.85 in every setting, above m* = 0.404, so no matched cost is finite: randomised
training restored reach in Veins, not timeliness. Per-setting RWCR, cost and miss: `h5.json` `per_run`.

## 6b. Experiments C and D (`results/revision8/analysis/exp_cd.json`; D12 scope, D15)

Runner shards (10, seeds 200–209, rural d=20): all accepted (`veins/exp{C,D}_rows_merge.json`): gate check 1
(classical, `707d6533…`) and **gate check 2 (AI-HARP ref_s0, bias +1, every new option off, byte-identical to the
revision-6 replay `220dd1fd…`) passed on every runner**; checkpoints restored from `results/revision1/checkpoints/` with
hashes equal to `checkpoint_shas.json`; 300 + 300 replays accepted; every logged decision re-sent offline (C 773,544,
D 759,634 requests; 0 mismatches). Raw OMNeT++ outputs (168 MB) in `veins/ci_cd/*/raw/`. Thresholds: the frozen Veins
q* 0.879, m* 0.404 (Tier A, `veins_thresholds.json`).

| Experiment | Rural d=20 qualifying (ref_s0–s4) | Best mean RWCR per seed | Verdict |
|---|---|---|---|
| C, geometric neighbour table | 0 / 5 | 0.817, 0.862, 0.864, 0.864, 0.864 | does not explain the failure |
| D, coupling lag removed | 0 / 5 | 0.809, 0.844, 0.864, 0.864, 0.864 | does not explain the failure |

In both, AI-HARP's reach stays at or below 0.864 (flooding's value in this cell, as in revisions 6 and 7) and below
q* = 0.879, so no bias setting can qualify. As pre-registered, "does not explain it" does not show that the neighbour table
or the lag plays no part.

## 7. Deviations (`results/revision8/DEVIATIONS.md`)

* **D12** (2026-11-16T05:31Z, author): Veins scope cut to fit one day — H5 Tier B seeds 210–219 (not 210–229), no H5
  Tier A, Experiments C/D rural d=20 only. Before any result.
* **D13** (author): AI-HARP checkpoints believed lost → Experiments C and D and gate check 2 not run. Before any result.
  **Wrong premise** — see D15.
* **D15** (author): D13 reversed; the checkpoints are committed in `results/revision1/checkpoints/` (hashes match).
  Experiments C and D and gate check 2 run on runners. Written after H4 and H5 had been seen (stated in D15); C and D are
  descriptive and outside Holm.
* **D14** (author): compute on GitHub Actions — training (one run per runner), custom evaluation (20 slices), Veins H5
  (20 shards, each gated). Before any evaluation result.
* **Not deviations, disclosed:** the training order; the worker trace-cache bound (memory only); plain_s7 trained twice
  (§4); the decision-server logs of the shards were committed by mistake in 020edeb and removed from the tree in 6bedf1d
  (they remain in that commit's history and on `r8-ci`; history not rewritten).

## 8. What remains (none of it needed for H4/H5)

1. H5 Tier A (200–209; 1,200 replays, descriptive) and H5 Tier B seeds 220–229: dropped by D12; `ci/r8-veins.json`
   with those seeds would run them on runners in about 1 h each.
2. Clean-clone reproduction check on the laptop. Done here (cloud, working tree): `AIHARP_PY=/root/r8env/bin/python ./reproduce_revision8.sh`
   exit 0; `h4.json`, `h4_rand_descriptive.json`, `h5.json`, `holm.json` regenerated identical apart from their provenance
   stamps; `exp_cd.json` records C and D as not run.

## 9. Open questions

* Whether the STRUCT-rand timeliness failure in Veins comes from STRUCT's own deferral (delays up to 51 epochs) or from
  the Veins channel; the registered analysis does not separate them.
* plain_s0/s1 (cloud) and the runner runs used different CPU types; runs are deterministic per machine but a rerun on a
  different CPU type may not be bit-identical (D14).
* R2 remains [ASSUMED].
* Reimplementing a published DRL dissemination scheme was out of scope (limitation only).
