# Revision 10 — pre-registration addendum: replication of revision-9 H6 on fresh Veins seeds

Written 2026-11-18 on branch `revision-10` (from `origin/revision-9` at f1c8949), **before any policy or classical
replay on Veins seeds 230–249**. Push commit of this file: `27aea29`.
Revision 5–9 results, thresholds, checkpoints, analysis files and deviations are unchanged and never edited.
Nothing is retrained, retuned or redesigned. Deviations are numbered from **D18**.

## 0. Why revision 10 exists

Revision-9 H6 was supported at p_H6 = 0.0476 (`results/revision9/analysis/h6.json`): in Veins urban NLOS d=20, seeds
220–229, STRUCT9-rand qualified for 4 of 5 training seeds and STRUCT9-plain for 0 of 5 (one-sided Fisher p = 0.0238;
rural d=20 0/5 vs 0/5). That hypothesis was motivated post hoc (revision-9 ADDENDUM §0), the result rests on one cell, and
its margins are thin (the qualifying STRUCT9-rand seeds' actionable miss was 0.536–0.581 against m* 0.602).
**Revision 10 asks only whether that result replicates on Veins seeds that no STRUCT policy has ever seen.**

## 1. What had been seen before this was written

* Every revision 5–9 result, including revision-9 H6, C1, S1, S2 and their per-seed tables.
* Revision-10 step 0–1 files only: `results/revision10/READING_NOTES.md`, `seed_inventory.json`, `seeds.json` and
  `gates.json` (commit `10c0966`). On seeds 230–249, `gates.json` contains bundle content hashes, trace hashes and the
  mobility sanity numbers (vehicle counts, achieved density) — **no replay of any policy or classical scheme was run on
  them.** Seed 220 was replayed once (the checkpoint gate: rand_s0, δ = 0, urban NLOS d=20), reproducing the committed H6
  replay byte for byte.

## 2. Arms, checkpoints and settings (fixed)

* **STRUCT9-rand** (`rand_s0`–`rand_s4`) and **STRUCT9-plain** (`plain_s0`–`plain_s4`): the committed revision-9
  parameters `results/revision9/train/<run>/theta.json`, unchanged (sha256 as in `gates.json` `checkpoint_gate`:
  plain_s0 `0e59c4d3…`, plain_s1 `8dc4d5f9…`, plain_s2 `88bf3d5d…`, plain_s3 `b1ba433a…`, plain_s4 `e4fc6c0b…`,
  rand_s0 `4f957993…`, rand_s1 `ec82ea75…`, rand_s2 `a13108c2…`, rand_s3 `25b1b8f4…`, rand_s4 `3626a2bf…`). **No retraining.**
* δ ∈ {−2, −1, 0, 1, 2, 3} (`experiments/revision8/veins/ci_veins.py:43`); 10 runs × 6 δ = 60 replays per (cell, seed).
* Every app and decision-server setting exactly as in H6: the revision-8 Veins app (binary sha256 `6b032de0…`, the H6
  binary), `experiments/revision9/veins/decision_server9.py` with `train_dir = results/revision9/train`, the revision-5
  bundle exporter and `config.xml` (BreakpointPathloss + NakagamiFading + SimpleObstacleShadowing), OMNeT++ 6.0.3,
  Veins 5.3, source-built SUMO 1.19.0, SUMO mobility. The shard runner `experiments/revision10/veins/ci10.py` is
  `experiments/revision9/veins/ci9.py` with only the output directory, the seed lists and one added check (§7) changed.

## 3. Thresholds (frozen)

`results/revision8/veins_thresholds.json`, unchanged: urban NLOS d=20 q* = 0.498350, m* = 0.601940; rural d=20
q* = 0.878958, m* = 0.403870.

## 4. Seeds (fixed in `results/revision10/seeds.json`, commit `6702c83`)

The first 20 integers ≥ 230 that occur nowhere in the seed inventory: **R-A = 230–239**, **R-B = 240–249** (none skipped).

## 5. PRIMARY H7 — exact replication of H6's urban cell

* Data: Veins urban NLOS d=20, seeds **R-A (230–239)**, 10 seeds as H6 used: 600 replays.
* Per training seed: the Algorithm-1 matched cost over its six δ, from the R-A **seed means** of RWCR, cost and actionable
  miss, with the miss guard (eligible iff mean actionable miss ≤ m*) — `experiments/revision6/analysis/veins_methods.py`
  `matched`, unchanged, as in H6 (READING_NOTES §1). The training seed **qualifies iff that matched cost is finite**.
* **Rule:** STRUCT9-rand qualifies for ≥ 3/5 training seeds **and** STRUCT9-plain for < 3/5.
* **Test:** one-sided Fisher exact test on the 2×2 table (qualifying / not × rand / plain), alternative: rand qualifies
  more often (`scipy.stats.fisher_exact(..., alternative="greater")`, the revision-9 function
  `experiments/revision9/analysis/veins9.fisher_greater`). **Single hypothesis, no multiplicity correction.**
* **H7 is supported iff the rule is met AND p_H7 < 0.05.**
* Values fixed in advance (computed by code: `tests/test_revision10_gate.py::test_preregistered_fisher_values`):

  | STRUCT9-rand qualifying | STRUCT9-plain qualifying | Rule met | One-sided Fisher p | H7 supported |
  |---|---|---|---|---|
  | 3/5 | 0/5 | yes | 0.0833 | no |
  | 4/5 | 0/5 | yes | 0.0238 | yes |
  | 5/5 | 0/5 | yes | 0.0040 | yes |
  | 5/5 | 1/5 | yes | 0.0238 | yes |
  | 4/5 | 1/5 | yes | 0.1032 | no |
  | 3/5 | 1/5 | yes | 0.2619 | no |
  | 3/5, 4/5 or 5/5 | 2/5 | yes | ≥ 0.0833 | no |

  Only 4/5 vs 0/5, 5/5 vs 0/5 and 5/5 vs 1/5 support H7.
* **If H7 is not supported, the paper reports H6 as not replicated, and H6 is presented as a preliminary result only.**
* Analysis code: `experiments/revision10/analysis/veins10.py h7`, committed before any R-A replay, with a test that the
  same code reproduces `results/revision9/analysis/h6.json`'s urban numbers from the committed 220–229 rows.

## 6. Secondary analyses (descriptive, pre-specified)

* **S3:** the same per-seed qualification, rule and Fisher p in urban NLOS d=20 on **R-B (240–249)**, and on
  **R-A ∪ R-B pooled** (each training seed's means over all 20 seeds). Descriptive; neither replaces H7.
* **S4:** per training seed and δ: mean RWCR, actionable miss and cost on R-A beside the H6 values on 220–229, and each
  training seed's qualification status in both — does each seed's status repeat?
* **S5:** rural d=20 on R-A, both arms, the same per-seed analysis — does the "reach without timeliness" pattern
  (revision-9 S1: RWCR near q* while miss stays above m*) repeat?

## 7. Compute and acceptance (as revision 9 §7, plus one check)

GitHub-hosted runners; outputs on branch `r10-ci`, merged into `revision-10`. **Order: H7 (urban R-A, 600 replays) →
S3 (urban R-B, 600) → S5 (rural R-A, 600).** Every shard (one cell × one seed, 60 replays):
1. restores the committed AI-HARP checkpoints (hash-checked) and passes **gate check 1** (`707d6533…`) **and gate check 2**
   (`220dd1fd…`) before any replay, or contributes nothing;
2. **added:** regenerates its bundle and requires its content digest to equal the one recorded in `gates.json` for that
   (cell, seed), or contributes nothing;
3. re-sends every decision log offline to a fresh decision server: **0 mismatches** required;
4. scores only exit-0 replays with an identical SUMO trace and the gated binary; accepted iff 60/60 merged and the trace
   check passes over n = 60 (revision-9 acceptance, `experiments/revision9/analysis/veins9.py:41-68`).
Every runner verifies its outputs on `r10-ci` before the job succeeds. Bridge and decision-server logs are never committed
to `revision-10`. A shard that fails for an infrastructure reason may be re-run once on a fresh runner (logged as a
deviation); a gate failure on every runner, or a second failure of the same shard, means STOP and report. H7 is computed
only when all 10 R-A shards are accepted (the per-seed table requires every (setting, seed)).

## 8. Deviations

Any change after this file's push commit to anything fixed here is a deviation, logged in
`results/revision10/DEVIATIONS.md` as D18, D19, … with a timestamp and exactly what had been seen.

## 9. Limitations stated in advance

*(The revision-10 instruction was cut off after its first limitation; the items after the first were completed by the
analyst from `READING_NOTES.md` and the revision-9 limitations, before any replay on 230–249.)*

* **Same training checkpoints** (replication of the evaluation, not of training): the five training seeds per arm and
  their parameters are the very objects H6 evaluated. H7 tests whether their Veins qualification pattern holds on new
  mobility seeds; it does not test whether retraining would produce policies that behave the same way.
* **Selected cell:** urban NLOS d=20 is replicated because it is the cell where H6 was significant (rural d=20 was 0/5 vs
  0/5); a successful replication speaks to that cell only.
* **Low power:** 5 training seeds per arm; only three splits support H7 (§5), and 4/5 vs 1/5 meets the rule without
  significance.
* **Same frozen thresholds** (derived from Veins Tier A 200–209 classical rows); thin margins near m* can flip a seed's
  status on new seeds.
* **One simulator stack:** Veins 5.3 has no log-normal shadowing model (`gates.json` `veins_source_check_report_only`);
  channel variation in Veins comes from Nakagami fading (m = 1) and obstacle shadowing only.
* **H6's post-hoc motivation** (revision-9 ADDENDUM §0) is not removed by a replication; R2 remains [ASSUMED].
* Reimplementing a published DRL scheme stays out of scope.
