# Revision 8 — status

Branch `revision-8` (from `origin/revision-7` at a6dea21). **Addendum push commit: `53eefc3`**
(`results/revision8/ADDENDUM.md`; the hash was recorded in its header by a0d94fb, which changes nothing else). Every
number below is read from the committed file named next to it. This file is updated as work proceeds; the section
"State at this commit" says what exists.

## State at this commit

TRAINING_STATE_PLACEHOLDER

## 1. What was implemented

| File | What |
|---|---|
| `agents/structured.py` | STRUCT, registry `struct`: 21 parameters, clipped-linear p and delay heads, k ∈ {1,2,3}, cost knob δ; equivalence parameters `theta_slotted`, `theta_rar` |
| `agents/train_structured.py` | CMA-ES trainer (cma 4.6.0, popsize 13), Deb's rule on causal coverage targets, R1–R5 sampling for STRUCT-rand, per-generation checkpoint with exact resume |
| `sim/engine.py`, `common/seeding.py` | R1 `dup_count_per_frame`, R2 `nb_table_error`, R3 `obs_lag_steps`, R5 `pdr_perturb` (all off by default; R2/R5 on their own streams `r8_nbtable`, `r8_pdr`); R4 is the existing `backend` option |
| `agents/registry.py` | registers `struct` |
| `tests/test_structured.py` | equivalence gate (contexts, draws, whole runs, policy-stream isolation) |
| `tests/test_randomisation_flags.py` | flags off: byte-identical to the revision-7 engine source; committed rows reproduced; each axis does what it says |
| `tests/test_oracle_isolation.py` | + `agents.structured`, `agents.train_structured` checked, including transitive imports |
| `tests/test_revision8_analysis.py` | H4 and H5 analysis machinery on stand-in data |
| `experiments/revision8/probe_heads.py` | step-0 probe (sigmoid vs clipped-linear heads) |
| `experiments/revision8/warm_start.py` | warm start on SEL only |
| `experiments/revision8/veins_thresholds.py` | frozen Veins q*, m* from Tier A classical replays |
| `experiments/revision8/traces.py`, `train_all.sh` | training-pool traces (both backends); the 20 training runs |
| `experiments/revision8/sweep_struct.py` | custom-simulator rows (SEL, EVAL 200–229) for both arms |
| `experiments/revision8/analysis/h4.py`, `h5.py` | H4; H5, Experiments C/D, Holm (written before any evaluation data) |
| `experiments/revision8/veins/` | app (`aiharp/`: kind `struct`, `stepOffset`, `sendHeard`), `build_app.sh`, `run_veins8.py`, `decision_server8.py`, `queue8.py`, `validate8.py`, `score8.py` |
| `experiments/revision8/manifest.py`, `reproduce_revision8.{sh,ps1}` | manifest; reproduction (analysis only by default) |
| `results/revision8/` | `READING_NOTES.md`, `ADDENDUM.md`, `DEVIATIONS.md`, `TIMING_ESTIMATE.md`, `LAPTOP_RUNBOOK.md`, this file |

## 2. Equivalence tests (`tests/test_structured.py`, all pass)

* Recorded first-reception contexts from RAR (1, 0.5, 0) runs on training-pool seeds 100–102 × the four headline cells
  (≥ 1,000; the probe on 100–103 recorded 1,698, `probe_heads.json`).
* slotted_1p n_slots = 5 and 50 (and 1, 2): **0 mismatches** (action, delay, cancel threshold).
* RAR (1, 0.5, 0): **0 decision mismatches and 0 contexts with different policy-stream consumption**; all 27 RAR grid
  settings: 0 mismatches on every 7th context.
* Whole runs, slotted_1p n_slots 5 and 50 vs STRUCT: **byte-identical engine output** (arrays, counters, transmissions
  log, action counts, metadata, metrics) on seeds 100–102 in each of the four headline cells (24 pairs).
* Policy-stream isolation: with the policy stream replaced by a guard that raises on any access, the engine runs to
  completion in all four headline cells (slotted_1p) and gives identical output; RAR under the guard raises, as it
  must; no file in `sim/` calls a method on `rng_pol`.
* Not exactly representable (docstring of `agents/structured.py`, READING_NOTES §8): counter_based (random delay
  1 + U{0..4}, `agents/counter.py:66`), slotted_1p n ∈ {3, 8, 12, 20, 30}, weighted_p/p_persistence (BROADCAST, no
  cancel), greedy/DV-CAST/SCF; ETSI CBF only inside its area, where it equals slotted_1p n=1.

## 3. Randomisation ranges and provenance

As registered (ADDENDUM §3), unchanged:

| Axis | Range | Provenance |
|---|---|---|
| R1 | per-frame duplicate counting on/off, p = 0.5 | MEASURED, Tier A: `REVISION6_REPORT.md:78`, `results/revision7/ADDENDUM.md:22-27` |
| R2 | p_miss ~ U[0, 0.5], p_stale ~ U[0, 0.5], ring 1.15 R | **[ASSUMED]**; only evidence 7/269 equal counts (`results/revision6/veins/port_validation_rules.json`); no neighbour counts in the committed raw outputs (`results/revision7/veins/raw/`: 0 `F` lines in 1,055 event logs; `.sca` per-vehicle totals only) |
| R3 | lag 0 / 1 epoch, p = 0.5 | MEASURED: 269/269 (`port_validation_rules.json`) |
| R4 | fallback / SUMO 1.19.0, p = 0.5 | SUMO runs here (`eclipse-sumo==1.19.0`); a committed classical SUMO row reproduced exactly; training-pool traces only (`results/revision8/traces_sumo.json`, `traces_fallback.json`) |
| R5 | u ~ U[0, 1] × per-bin PDR gap | MEASURED, Tier A: `results/revision5/analysis/h3.json` `network_layer.*.pdr_vs_distance` |

Flag tests (`tests/test_randomisation_flags.py`, all pass): flag-off engine output byte-identical to the revision-7
engine source (a6dea21) for slotted_1p, RAR and counter_based in all four headline cells; zero-effect settings (R2 with
0/0, R5 with u = 0) byte-identical to off; committed revision-5 rows (CAL fallback, EVAL SUMO) reproduced. **Platform
finding:** counts and costs match exactly, but `rwcr` (≤ 1 ulp) and `risk_peak_corr` (~1e-15) differ in the last bit from
the rows written on the Windows laptop, identically with and without the revision-8 changes; the test compares those two
to 1e-12 relative and relies on the revision-7-source comparison for byte identity.

## 4. Training

TRAINING_PLACEHOLDER

## 5. Custom-simulator H4

H4_PLACEHOLDER

## 6. Deviations

DEVIATIONS_PLACEHOLDER

## 7. What remains for the laptop, in order

`results/revision8/LAPTOP_RUNBOOK.md`: (0) checkout + gate tests, 5 min; (1) `r6.txt` for seeds 210–229, 10 min;
(2) build the revision-8 app, 5 min; (3) bridge unit checks, 15 min; (4) Veins gate, about 1 h; (5) H5 Tier B, 2,400
replays, about 26 h; (6) H5 Tier A, 1,200 replays, about 13 h; (7) Experiments C and D, 600 replays each, about 7 h each;
(8) Holm, manifest, commit, 15 min. About 55 wall-hours in total.

## 8. Open questions

OPEN_PLACEHOLDER
