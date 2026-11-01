# Revision 5 — report

Branch `revision-5`. Pre-registration: `results/revision5/PREREGISTRATION.md` (seed sets CAL 300–309, SEL 400–419,
EVAL 200–229; thresholds frozen in `thresholds.json` before any SEL/EVAL run). Analysis definitions:
`results/revision5/analysis/ANALYSIS_SPEC.md`. Deviations: `results/revision5/DEVIATIONS.md` (D1–D7). Every number below
is read from a committed output under `results/revision5/analysis/`; the file is named with each result.

## Summary: H1, H2, H3

Holm correction over H1, H2 and H3, α = 0.05 (`holm.json`).

| | Raw p | Holm-adjusted p | Verdict |
|---|---|---|---|
| H1 | 1.7 × 10⁻²²³ | 5.2 × 10⁻²²³ | **supported** |
| H2 | 0.0636 | 0.127 | **not supported** |
| H3, epoch mode (pre-registered primary) | 0.776 | 0.776 | **fails** |
| H3, frozen sub-epoch mode (D6 re-run) | 0.169 | 0.169 (post-D6 family; H1 and H2 unchanged) | **fails** |

**H1 (no single classical family is the cheapest qualifying family in all four headline cells): supported.**
`h1.json`. In 0 of 10,000 hierarchical bootstrap replicates (CAL × SEL × EVAL seeds resampled; q*, m* and every
selection recomputed) did one family win all four cells; Clopper–Pearson 95 % [0, 0.00037]; one-sided binomial test
against 0.05, p = 1.7 × 10⁻²²³. Point winners: slotted_1p (rural d=2), greedy_farthest (rural d=20), weighted_p
(rural d=80), counter_based (urban NLOS d=20). With the bounded-delay variant D_r (`h1_Dr.json`) the result and the
winners are identical. In Veins (3-fold cross-fitted audit, D1) H1 is also supported (0 / 10,000, same interval), but
there the point estimate has a qualifying family in only one cell (rural d=80: DV-CAST); in the other three cells no
family's pick qualifies on all three held-out folds (`h3.json`, `crossfit_H1`).

**H2 (the policy is cheaper than the SEL-selected adaptive classical scheme): not supported.** `h2.json`.
Pooled relative cost difference (policy / adaptive − 1, over headline cells where both qualify) = **+31.1 %**: the
policy is *more* expensive. 95 % CI [−0.8 %, +57.9 %]; two-sided p = 0.0636; Holm-adjusted p = 0.127;
P(policy cheaper) = 0.032. Neither superiority nor inferiority is shown (the 95 % CI contains 0), and equivalence by
TOST ±5 % fails (90 % CI [+1.9 %, +52.9 %]). Both qualify in only two cells: rural d=20 (+5.6 %) and urban NLOS d=20
(+56.5 %). 528 of 10,000 replicates are undefined (no cell where both qualify) and excluded; 1 is infinite.
Qualification: the policy's (seed, cell) picks qualify 20 / 40 times (Wilson 95 % [0.35, 0.65]); adaptive classical
qualifies in 3 / 4 cells ([0.30, 0.95]); Newcombe difference [−0.50, +0.22]. Against the SEL-selected best rule
baseline (exploratory, outside the Holm family) the policy is significantly costlier: **+78.5 %**, 95 % CI
[+29.8 %, +178.9 %], p = 0.00026 (rural d=20 +70.3 %, urban NLOS d=20 +86.7 %). The D_r variant (`h2_Dr.json`) gives
the same point estimates (+31.1 % vs adaptive, +78.5 % vs rule). The pre-registered precision note applies: 30 EVAL × 10
training seeds give about ±7.8 % half-width, so a ±5 % equivalence result was not expected to be reachable.

**H3 (packet-level agreement between the custom simulator on SUMO mobility and Veins): fails, in both modes.**
`h3.json` (epoch mode, the pre-registered test) and `h3_subepoch.json` (the frozen sub-epoch mode of D6).

| Criterion | Epoch mode | Sub-epoch mode (D6) |
|---|---|---|
| Spearman ρ of matched costs ≥ 0.8 with lower bound ≥ 0.6 | ρ 0.564 [−0.117, 0.791], p(ρ ≤ 0.6) 0.776 — **fail** | ρ 0.842 [0.373, 0.933], p 0.169 — **fail** (lower bound) |
| PDR vs distance within 0.05 in every 50 m bin | largest gap 0.460 / 0.340 / 0.316 / 0.457 — **fail** | 0.331 / 0.301 / 0.261 / 0.187 — **fail** |
| Channel busy ratio within 20 % | 33 / 45 / 45 / 89 % — **fail** | 1.4 / 10 / 44 / 49 % — **fail** (passes in rural d=2 and d=20) |
| Median single-hop delay within 100 ms | 0.100 vs 0.100 s in every cell — pass | pass |
| Same H1 verdict in both simulators | supported in both — pass | pass |

(Cells in the order rural d=2 / rural d=20 / rural d=80 / urban NLOS d=20; 10 matched-cost pairs.) The D6 mode was
calibrated only on the validation metrics and only on the audit seeds 210–219; the minimum loss (0.652) fell on a corner
of the pre-declared grid (carrier sense −85 dBm, capture SINR 10 dB); Veins' own CCA threshold (−65 dBm) lies outside that
grid, and the grid was not widened (D6). **Pre-registered consequence, applied (D6):** the classical audit computed in
Veins (cross-fitted H1) is primary; custom-simulator claims are restricted to relative comparisons inside the custom
simulator (scheme and policy rankings, selection, qualification); no absolute packet-level number from the custom
simulator (PDR, busy ratio, loss rates) is presented as validated. The policy exists only in the custom simulator, so
H2 and every secondary policy result carry that restriction.

## 1. Traceability: reviewer concerns C1–C8

**Not completed.** The step-6 instruction asks for a table mapping reviewer concerns C1–C8 to evidence, but the list of
C1–C8 is not in the instructions or anywhere in the repository (the only lettered list is revision 1's work items
C1–C5 — training seeds, ablations, fallback baselines, generalisation, 2,000-update budget — in
`results/revision1/REVISION_REPORT.md`, which are not reviewer concerns and number five). No mapping has been invented.
Evidence index to make the mapping quick once the list is supplied:

| Topic | Evidence | Status |
|---|---|---|
| Classical rankings / per-cell tuning (H1) | `h1.json`, `h1_Dr.json`, ledger "classical schemes" | answered (supported) |
| Policy vs retuned classical, held-out selection (H2) | `h2.json`, `h2_Dr.json`, `tables/T-heldout.tex` | answered (not supported) |
| Policy vs simple rule baselines | `h2.json` `vs_best_rule`, `T-heldout` | answered (policy costlier) |
| Simulator validity vs Veins (H3) | `h3.json`, `h3_subepoch.json`, `d6_calibration.json`, `T-veins`, figures `fig_veins_*` | answered (fails; claims restricted) |
| Training seeds / variance | `h2.json` (10 seeds), `T-gate`, `training_diagnostics.json`, `fig_training` | answered |
| Architecture ablations (GCN, MLP, no relevance) | `secondary.json` `variants`, `T-ablation` | answered |
| Privileged-information ladder | `secondary.json` `ladder`, `T-ladder` | answered |
| Learned gate vs fallback | `secondary.json` `gate`, `T-gate` | answered |
| Training budget | `secondary.json` `budget`, `compute_curves.json`, `fig_compute` | answered (D4 checkpoints) |
| Generalisation: channel stress test, SUMO mobility | `secondary.json` `stress`, `sumo`, ledger | answered |
| Interpolated vs realised operating points | `secondary.json` `interp`, `T-interp` | answered |
| Sensitivity to q*, miss tolerance, interpolation | `secondary.json` `sensitivity`, `T-sens` | answered |
| Safety outcomes (miss, warned share, p95 TIR, miss cap 0.40) | `secondary.json` (every method entry), `T-heldout` | answered |
| Compute cost vs classical tuning | `compute_curves.json`, `fig_compute` | answered |
| Robustness ledger | `ledger.json` | answered |

## 2. Secondary results (exploratory; no Holm)

All from `secondary.json` (hierarchical bootstrap, 10,000 replicates); "qualifies" = the cell aggregate qualifies on
EVAL (at least half of the training seeds pick a setting on SEL, and their mean EVAL RWCR ≥ q* and mean miss ≤ m*).
Full numbers with intervals: `tables/T-*.tex`. Custom-simulator results, restricted as stated above.

**Held-out table (`T-heldout`).** Rural d=2: the SEL-selected adaptive classical pick (dvcast n_slots=20) does not
qualify on EVAL; the best rule (scf_slotted) qualifies; 4 of 10 policy seeds pick and 2 qualify, so the policy does not
qualify. Rural d=20: all three qualify (adaptive 0.671, rule 0.416, policy 0.708). Rural d=80: adaptive (1.017) and rule
(0.355) qualify; only 3 of 10 policy seeds pick, so the policy does not. Urban NLOS d=20: all qualify (adaptive 0.537,
rule 0.450, policy 0.841). Under the absolute miss cap 0.40 the rural d=2 cell is unreachable for every method (best
rule miss 0.580; policy picks 0.568 on average).

**Ladder (`T-ladder`, seeds 0–4).** Cells reached: A (full model) 2, B (oracle reward) 2, C (oracle input,
non-deployable) 2. Rural d=20 margin vs adaptive classical: A +2.8 %, B −1.4 %, C +15.9 %. Privileged reward or input
does not raise the number of cells reached.

**Encoder variants (`T-ablation`, seeds 0–4).** Cells reached: full model (GATv2) 2, GCN 3, MLP 4, no relevance feature
1. Rural d=20 margin: full model +2.8 %, GCN −3.6 %, MLP +68.4 %, no relevance +8.0 %. The MLP qualifies in more cells but
is much costlier where it does; the full model is not better than the GCN on either count.

**Gate (`T-gate`, seeds 0–9).** Cells reached: learned gate (τ = 0.5) 2, ungated (τ = 0) 4, random gate at the learned
gate's SEL rate 3. Rural d=20 margin: +5.6 %, +17.6 %, +7.2 %. The ungated network reaches all four cells but at a
higher cost; a rate-matched random gate reaches more cells than the learned gate (as in revision 3, R12).

**Budget (D4; seeds 0–2 at 1,000 / 1,200 / 1,500 / 1,700 / 2,000 updates).** Cells reached: 1 / 2 / 2 / 2 / 3. Rural
d=20 margin: +6.9 % / −4.4 % / −3.3 % / +2.4 % / −7.4 %. At 2,000 updates rural d=80 is reached; rural d=2 at none.
Training CPU at the budget points: `compute_curves.json` (5.63 CPU-h at 1,000 updates is an estimate because seeds
0–2's first 1,000 updates predate CPU logging; 8.43 CPU-h at 2,000).

**SUMO mobility (headline cells; frozen default-channel thresholds, no SUMO CAL).** The policy (seeds 0–9) qualifies in
**0 of 4** cells (2 seeds qualify at rural d=80, below the half-of-seeds rule). Adaptive classical qualifies in none of
the four; the best rule qualifies in rural d=20 and d=80.

**Stress test (24 cells, R9 calibrated channel, own thresholds, D5).** Policy seeds 0–4 reach **9 of 24** cells (rural
d=20 and d=40; urban NLOS d=2, 5, 10; urban grid d=1, 2, 3, 5); adaptive classical qualifies in 14, the best rule in 12.
In urban NLOS d=2 the policy qualifies but its mean cost is infinite: one EVAL run (seed 210, bias 0, one checkpoint)
transmitted without informing any at-risk vehicle, and the pre-registered cost keeps that infinite. Where both qualify,
the policy is costlier than adaptive classical in every comparable cell (`ledger.json`). The pre-registered cut was
seeds 0–2; seeds 0–4 were run. Restricted to seeds 0–2 the policy also reaches 9 cells, with urban grid d=10 in place of
rural d=40. On the default channel, the 24-cell H2 version (`h2_grid24.json`, seeds 0–4) has an infinite pooled
statistic because of urban NLOS d=2.

**Interpolation (`T-interp`).** Realised (SEL-selected, scored on EVAL) and interpolated (Algorithm 1 crossing on EVAL
means, an estimate) costs agree for most families; where they differ, the realised point is primary.

**Sensitivity grid (`T-sens`, 18 settings: q* factor 0.90/0.95/0.98 × miss tolerance 0.02/0.05/0.10 × interpolation
on/off).** In no setting does one family win all four cells (EVAL-hindsight families). The rural d=20 margin of the
policy vs adaptive classical (positive = policy costlier) is positive in 16 of 18 settings (baseline +5.6 %; range up to
+31.0 % at q* factor 0.98) and negative only at q* factor 0.90 with miss tolerance 0.10 (−7.7 % realised, −6.5 %
interpolated).

**Training diagnostics (`training_diagnostics.json`, `fig_training`).** The rural d=2 multiplier λ ends at 40.2 ± 4.3
(mean ± SD over seeds 0–9) after 1,000 updates and is still rising in 7 of 10 seeds (slope over the last 100 updates,
95 % CI above 0): the rural d=2 constraint is not satisfied within the training budget.

**Ledger (`ledger.json`; verdict rules from the ANALYSIS_SPEC addendum, D7 for undefined columns).**

| Group | Claim | Verdict |
|---|---|---|
| Classical | H1: no universal family | **stable** (Set., calibrated channel, SUMO, Veins, sensitivity grid) |
| Classical | Identity of the per-cell winners | **sensitive** (changes on the calibrated channel, on SUMO and in Veins) |
| Policy | H2: cheaper than adaptive classical | **not supported** |
| Policy | Costlier than the best rule | **sensitive** (holds on the default and calibrated channels; undefined on SUMO; not in the sensitivity grid) |
| Policy | Qualifies in rural d=20 and urban d=20 only | **sensitive** (urban d=20 lost on the calibrated channel; no cell on SUMO) |

## 3. Deviations

From `DEVIATIONS.md`, each written with a timestamp and a statement of what had been seen:

* **D1** (2026-10-19) — H3 seed sets: network layer on EVAL 200–209 paired; Veins H1 by 3-fold cross-fitting over
  200–229 (the full CAL/SEL/EVAL protocol in Veins, ~13,000 replays, was infeasible). No H3 statistic seen.
* **D2** (2026-10-19) — incident, no protocol change: the first SEL sweeps ran on the wrong mobility backend; rows
  quarantined in `runs/invalid_sumo_backend/`, sweeps restarted (~9.5 h lost). Found before any SEL result was viewed.
* **D3** (2026-10-31) — H3 completion rule and operational definitions (PDR, delay, busy ratio, loss, matched cost,
  cross-fitted H1), fixed before any Veins statistic.
* **D4** (2026-10-31) — budget checkpoints at 1,200 and 1,700 updates instead of 1,250 and 1,750 (no such checkpoints
  exist); no budget result seen.
* **D5** (2026-10-31) — the stress test uses its own CAL-derived thresholds (`thresholds_stress.json`); no calibrated
  row existed.
* **D6** (2026-10-31, outcome 2026-11-01) — H3 failed; the pre-registered fail branch was executed (sub-epoch timing
  mode, calibrated on audit seeds only, frozen, re-run); H3 still fails; consequence applied as stated above.
* **D7** (2026-11-01) — the ledger scores a column where the claimed comparison is undefined as "not holding" (decided
  after the Step 1–2 results were seen; no verdict changes under the alternative reading); the secondary bootstrap was
  re-run with the pre-registered 10,000 replicates (first run used the script default 2,000); the byte-identity check
  method (below).
* **Not a deviation, disclosed:** the stress-test policy evaluation used seeds 0–4 where the pre-specified cut allowed
  0–2 (a superset; both results are given above).

## 4. Blockers and incomplete items

* **C1–C8 traceability:** the concern list is not available (section 1).
* **Azure spend:** not determinable here (no Azure CLI or billing access on this machine; no cost record in the repo).
  Cloud compute used: three VMs (`aiharp-ci`, `aiharp-ea`, `aiharp-ks`) ran 4,840 of the 6,600 Veins replays,
  828.4 replay wall-hours in total (200.7 / 307.7 / 320.0 h), synced between 2026-10-19 and 2026-10-24.
* **Veins replays cannot be rerun by the reproduce script** (they need the OMNeT++ 6.0.3 / Veins 5.3 toolchain set up by
  `experiments/revision5/cloud/setup_cloud_vm.sh`); the script reuses the committed replay outputs.
* **One recovered Veins scalar file** (rural d=80 s=221 counter_based 1): regenerated on the laptop with a
  byte-identical event log (`results/revision5/veins/recovery.md`).
* **Cost undefined in urban NLOS d=2 (stress test):** one infinite-cost EVAL run (above).
* **Training CPU for seeds 0–2's first 1,000 updates** was not logged; the compute curve uses an estimate there.

## 5. Reproduction

```
powershell -ExecutionPolicy Bypass -File reproduce_revision5.ps1                 # every analysis output from committed rows
powershell -ExecutionPolicy Bypass -File reproduce_revision5.ps1 --from-scratch  # rerun the simulations first
./reproduce_revision5.sh [--from-scratch]                                        # same, bash
```

The scripts refuse to run unless the interpreter reports NumPy 1.26.4. `--from-scratch` moves the existing run files
aside (`*.bak-<timestamp>`; nothing is deleted) and reruns every sweep in the pre-registered order, using the committed
checkpoints and Veins replays. Output map (rows → script → command → commit → config hash → SHA-256):
`results/MANIFEST.md`, `results/manifest.json`.

**Clean-clone check:** passed (section 7).

## 6. CPU-hours per track

Per-run `cpu_s` as recorded by the runner (process CPU of each simulation), summed per track:

| Track | Rows | CPU-hours |
|---|---|---|
| Default channel H1/H2 protocol (CAL/SEL/EVAL; classical, rule, policy) | 9 files | 78.3 |
| Step 1: ladder, variants, gate (τ = 0, random gate), budget | 12 files | 86.4 |
| Step 2: stress test, calibrated channel (CAL/SEL/EVAL; classical, rule, policy) | 7 files | 38.9 |
| Step 2: SUMO mobility (classical, rule, policy) | 6 files | 20.1 |
| **Simulation sweeps, total** | 34 files | **223.6** |
| Training, 23 revision-5 runs (logged main + rollout CPU) | | 115.3 |
| Veins replays, laptop (1,760 replays; wall-hours) | | 235.3 |
| Veins replays, Azure VMs (4,840 replays; wall-hours) | | 828.4 |

Not recorded: the custom-simulator H3 runs (2,200 + 6,600 D6 re-run + 3,200 D6 calibration runs; their records carry no
CPU field) and the analysis scripts.

## 7. Clean-clone reproduction check

Fresh `git clone --branch revision-5` into `D:\aiharp-clean` (commit d391f30), then `reproduce_revision5.ps1` (one
command, analysis from committed rows; interpreter check passed: NumPy 1.26.4; exit code 0; 2026-11-01 23:15 →
2026-11-02 00:14). Each regenerated output was compared with the committed version in that clone
(`analysis/clean_clone_check.json`):

* **h1.json, h1_Dr.json, h2.json, h2_Dr.json, h2_grid24.json, h3.json, h3_subepoch.json** and the six other JSON outputs:
  **identical once the `provenance` block is removed** (parsed and re-serialised in file order). The whole files are
  not byte-identical, and cannot be: the provenance stamp records the generation time and the commit. In h3.json the
  stamp also differs in `config` (the committed file was written before `h3.py` gained `--mode`, so its stamp has no
  `mode` key and a different config hash); its content is identical.
* **All 7 tables and all 5 figures: byte-identical** as whole files.

The check found two defects, both fixed before this final run and neither changing a number:
1. The analysis needed the trained checkpoints (not committed) to identify which training run each policy row belongs
   to; the first clean-clone run failed at h2. Fix: committed map `results/revision5/checkpoint_shas.json` (47 runs; every
   hash verified against the committed rows), used when a checkpoint file is absent and checked against it when present.
2. The per-seed entries of h2/h2_Dr/h2_grid24 were written in Python-set order, which changes between processes; all
   values were identical. Fix: run order.
