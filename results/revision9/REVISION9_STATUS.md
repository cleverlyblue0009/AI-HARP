# Revision 9 — status

Branch `revision-9` (from `origin/revision-8` at b67f18c). Runner outputs on branch `r9-ci`, merged here.
**Revision 9 is motivated by a post-hoc observation** (ADDENDUM §0): after revision-8 H5 was not supported, STRUCT-rand
was seen to reach RWCR above q* in Veins rural d=20 while missing deadlines above m*, and the revision-8 training
fitness had no deadline term (a design error in the revision-8 specification). Revision 9 tests one fix. Revision-8
results stand unchanged; no revision 5–8 file was modified.

Every number below is read from a committed file named beside it.

## 1. Pre-registration

| Item | Commit |
|---|---|
| Analysis code (H6, C1, S1, S2), before any revision-9 data | `9e2ab6f` |
| Miss targets and gates A–C (training pool only), before the addendum | `4cbe675` |
| **Addendum pushed** (`results/revision9/ADDENDUM.md`), before any training or evaluation run | **`b9feb0c`** |
| Addendum header records its push commit (no other change) | `7d85aac` |
| First training run requested | `200f10e` |

## 2. Gates (training pool 100–131; `results/revision9/gates.json`, `miss_targets.json`)

Causal actionable miss: `agents/causal_miss.py`, the `analysis/metrics.py` formula with causal inputs; bit-identical to the
metric when the oracle inputs are injected (`tests/test_causal_miss.py`); never reaches the oracle, `privileged/` or
`analysis.metrics` transitively (`tests/test_oracle_isolation.py`).

| Group | Flooding default miss | DV-CAST default miss | **b_g** | Coverage target |
|---|---|---|---|---|
| rural d=2 | 0.4442 | 0.4421 | **0.4921** | 0.7201 |
| rural d=20 | 0.1209 | 0.1206 | **0.1706** | 0.9256 |
| rural d=80 | 0.0036 | 0.0014 | **0.0514** | 0.9484 |
| urban NLOS d=20 | 0.7357 | 0.7413 | **0.7857** | 0.2656 |

* **Gate A — passed.** Classical settings (of 55) meeting coverage and b_g: 10 / 40 / 46 / 35 (coverage alone 10 / 40 / 47 / 35).
* **Gate B — passed.** always_suppress f = 1000005.6951 (infeasible); flooding 2.3738 (feasible); slotted_1p 5 0.6599
  (feasible); slotted_1p 50 1000000.2850 (infeasible); DV-CAST default 0.6588 (feasible); RAR warm start 1000000.0252
  (**infeasible** — CMA-ES starts from an infeasible point, as stated in the addendum).
* **Gate C (descriptive).** Spearman ρ causal vs oracle miss: reference policies pooled 0.298 (n = 768); per group
  0.965 / 0.521 / 0.492 / 0.372; all 59 settings pooled 0.394.
* Flag-off identity (`tests/test_train_flag_identity.py`): with the flag off, the fitness function matches the
  revision-8 trainer (loaded from b67f18c), and a 1-generation run matches the committed revision-8 `plain_s0` history
  line; with the flag on, always-suppress is infeasible. All pass.

## 3. Training (`results/revision9/train_summary.json`; `results/revision9/train/<run>/`)

80 generations each, `--miss-targets results/revision9/miss_targets.json`; every `run_meta.json` records CPU, Python
3.12, NumPy 1.26.4, cma 4.6.0 and the miss bounds. "Feasible last 10" = mean feasible fraction of the population over
generations 71–80. Best f ≥ 10⁶ means the best candidate of the last generation was infeasible.

| Run | CPU | Feasible last 10 | Best f (last gen) | Wall |
|---|---|---|---|---|
| plain_s0 | Intel Xeon 6975P-C | 0.40 | 0.350 | 33 min |
| plain_s1 | Intel Xeon Platinum 8370 | 0.20 | 0.373 | 53 min |
| plain_s2 | Intel Xeon 6975P-C | 0.14 | 1000000.155 | 30 min |
| plain_s3 | AMD EPYC 7763 | 0.48 | 0.377 | 62 min |
| plain_s4 | Intel Xeon 6975P-C | 0.28 | 0.389 | 30 min |
| rand_s0 | AMD EPYC 9V74 | 0.00 | 1000000.482 | 44 min |
| rand_s1 | AMD EPYC 7763 | 0.02 | 1000000.522 | 61 min |
| rand_s2 | Intel Xeon Platinum 8573C | 0.00 | 1000000.380 | 45 min |
| rand_s3 | Intel Xeon 6975P-C | 0.00 | 1000000.067 | 30 min |
| rand_s4 | AMD EPYC 9V74 | 0.00 | 1000000.193 | 48 min |

STRUCT9-rand almost never satisfied the training constraints under randomisation; the output is the CMA mean after the
last generation regardless (as in revision 8).

## 4. H6 — primary (Veins Tier B 220–229; `results/revision9/analysis/h6.json`)

1,200 replays on 20 runners; **20/20 shards accepted** (gate check 1 `707d6533…` and gate check 2 `220dd1fd…` passed on
every runner; 60/60 decision logs re-sent offline with 0 mismatches per shard; 60/60 replays merged per shard;
`results/revision9/veins/h6_rows_merge.json`). Thresholds `results/revision8/veins_thresholds.json`.

| Cell | q* | m* | STRUCT9-rand qualifying | STRUCT9-plain qualifying | Rule met | One-sided Fisher p |
|---|---|---|---|---|---|---|
| rural d=20 | 0.879 | 0.404 | 0/5 | 0/5 | no | 1.000 |
| urban NLOS d=20 | 0.498 | 0.602 | **4/5** | 0/5 | **yes** | 0.0238 |

**p_H6 = min(1, 2 × 0.0238) = 0.0476 < 0.05; rule met in 1 cell → H6 supported (as pre-registered).**

Read with: post-hoc motivation; 5 seeds per arm; the margin is thin (p_H6 0.048; the qualifying STRUCT9-rand seeds have
urban miss 0.536–0.581 against m* 0.602, at cost 0.71–0.95); and the plain arm is weak on these seeds — revision-8
STRUCT-plain qualified 2/5 in the same cell and seeds (C1, §6). In rural d=20, where the post-hoc observation was made,
no seed of either arm qualifies.

## 5. S1 — STRUCT9-rand vs revision-8 STRUCT-rand, Veins 220–229 (`results/revision9/analysis/s1.json`, descriptive)

Means over the 5 training seeds (per seed: the δ with the highest mean RWCR; "min miss" = lowest mean miss over δ).

| Cell | Arm | Qualifying | Best RWCR | Miss at best | Cost at best | Min miss |
|---|---|---|---|---|---|---|
| rural d=20 (q* 0.879, m* 0.404) | STRUCT9-rand | 0/5 | 0.866 | **0.417** | 0.572 | **0.417** |
| | revision-8 STRUCT-rand | 0/5 | 0.920 | 0.618 | 0.397 | 0.612 |
| urban NLOS d=20 (q* 0.498, m* 0.602) | STRUCT9-rand | 4/5 | 0.520 | **0.578** | 0.799 | **0.575** |
| | revision-8 STRUCT-rand | 2/5 | 0.539 | 0.691 | 0.736 | 0.686 |

Per seed, rural d=20: STRUCT9-rand miss at best 0.555 / 0.414 / 0.389 / 0.354 / 0.375 with RWCR 0.835 / 0.888 / 0.877 /
0.871 / 0.859 (cost 0.31–0.81); revision-8 miss 0.736 / 0.759 / 0.580 / 0.458 / 0.560 with RWCR 0.900–0.937.
**The deadline term lowered Veins miss by ~0.20 in rural d=20 and ~0.11 in urban, at a cost of ~0.05 RWCR in rural
(below q* for 4 of 5 seeds) and +0.06–0.18 transmissions per at-risk vehicle.** In rural d=20 three seeds now reach
miss < m* but not RWCR ≥ q* at the same δ, so the cell still has no finite matched cost.

## 6. C1 — revision-8 STRUCT on Veins 220–229 (`results/revision9/analysis/c1.json`, descriptive)

The registered revision-8 H5 verdict (seeds 210–219, `results/revision8/analysis/h5.json`: not supported, p_H5 = 1.0)
**stays the result of record.**

| Seeds | Cell | Rand qualifying | Plain qualifying | Rule met | Fisher p |
|---|---|---|---|---|---|
| 210–219 (revision-8 H5, of record) | rural d=20 | 0/5 | 0/5 | no | 1.0 |
| | urban NLOS d=20 | 1/5 | 0/5 | no | 0.5 |
| 220–229 (C1) | rural d=20 | 0/5 | 0/5 | no | 1.0 |
| | urban NLOS d=20 | 2/5 | 2/5 | no | 0.738 |
| 210–229 combined (descriptive) | rural d=20 | 0/5 | 0/5 | no | 1.0 |
| | urban NLOS d=20 | 1/5 | 0/5 | no | 0.5 |

1,200 replays, 20/20 shards accepted under the same checks as H6 (`results/revision9/veins/c1_rows_merge.json`).

## 7. S2 — custom simulator (`results/revision9/analysis/s2.json`, descriptive; 5 seeds, H4 used 10)

Revision-8 H4 comparator and code (`experiments/revision8/analysis/h4.py analyse_tier`, unchanged). T = pooled relative
matched-cost difference STRUCT9 vs comparator (negative = cheaper); non-inferiority margin +10%.

| Arm | Tier | Qualifying cells (STRUCT9 / comparator) | T | One-sided 95% upper | p (non-inferiority) | Rule met |
|---|---|---|---|---|---|---|
| STRUCT9-plain | A (200–209) | 3 / 3 | −0.204 | 0.042 | 0.025 | yes |
| STRUCT9-plain | B (210–229) | 4 / 4 | −0.087 | 0.053 | 0.017 | yes |
| STRUCT9-rand | A | 3 / 3 | +1.101 | 1.774 | 1.0 | no |
| STRUCT9-rand | B | 4 / 4 | +0.954 | 1.545 | 1.0 | no |

In the custom simulator the deadline-constrained plain policy is non-inferior to the classical comparator; the
randomised one costs roughly twice as much. Descriptive only (revision-8 H4 remains the registered custom-simulator test).

## 8. Deviations (`results/revision9/DEVIATIONS.md`)

* **D16** — H6 and S2 launched in one push instead of in sequence; S2 finished first. No shared inputs or outputs; no
  result existed at launch. Written after `h6.json` and `s2.json` existed (stated in the entry).
* **D17** — S2 plain slice 8 hung in `actions/checkout` (> 50 min, cancel refused with 403); the workflow gained an
  optional slice subset and that slice alone was re-run. What a slice computes did not change.

No change to any hypothesis, threshold, seed set, arm, training setting or analysis.

## 9. Reproduction

* `./reproduce_revision9.sh` (default: analysis only; NumPy 1.26.4 interpreter via `AIHARP_PY`) regenerates
  `miss_targets.json`, `gates.json`, `analysis/h6.json`, `c1.json`, `s1.json`, `s2.json` from committed rows.
  **Checked 2026-11-16: exit 0; all six outputs identical to the committed files apart from `provenance`.**
* `reproduce_revision9.ps1` is the same steps for Windows; not executed here (no PowerShell in the cloud container).
* `--from-scratch` prints how to rerun the runner work (request files `ci/r9-gates.json`, `ci/r9-train.json`,
  `ci/r9-veins.json`, `ci/r9-eval.json`; outputs on `r9-ci`; never commit `bridge_logs/`).
* Manifest: `python -m experiments.revision9.manifest` → `results/MANIFEST.md` (Revision 9 section), `results/manifest.json`.
* Test suite: 681 passed, 2 skipped before each code push.

## 10. Open questions

1. **Is H6 robust?** p_H6 = 0.048 rests on 4/5 vs 0/5 in one cell, with qualifying misses 0.02–0.07 under m*. A second
   set of Veins seeds (e.g. Tier A 200–209, which Veins thresholds were derived from, or new seeds) would show whether
   the split holds. Not pre-registered here.
2. **Why is STRUCT9-plain weak in Veins urban (0/5) when revision-8 plain was 2/5 on the same seeds?** The deadline term
   lowered custom-simulator miss for plain (S2, rural d=20 Tier A miss 0.33–0.37) but did not carry to Veins urban.
3. **Rural d=20 remains unsolved** by every STRUCT variant: lower miss came with RWCR below q*. Gate C showed the causal
   miss tracks the oracle miss only moderately outside rural d=2 (ρ 0.37–0.52), which may limit what a causal training
   constraint can do.
4. **STRUCT9-rand never became feasible in training** (feasible fraction ≤ 0.02), so the evaluated policy is a CMA mean of
   infeasible candidates; whether a longer run or another warm start would change that is untested (and would be a new
   pre-registration, not a deviation).
5. R2 remains [ASSUMED] (revision-8 ADDENDUM §3).
