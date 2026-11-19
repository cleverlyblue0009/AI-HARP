# Revision 10 — status

Branch `revision-10` (from `origin/revision-9` at f1c8949). Runner outputs on branch `r10-ci`, merged here.
Revision 10 is a pre-registered **replication** of revision-9 H6 on Veins seeds that no STRUCT policy had seen. Nothing was
retrained, retuned or redesigned; revision 5–9 files are unchanged. Every number below is read from the committed file
named beside it.

**Headline: H7 is not supported (STRUCT9-rand 0/5 vs STRUCT9-plain 0/5, p = 1.0). H6 did not replicate.** As fixed in
the addendum (§5): the paper reports H6 as not replicated, and H6 is presented as a preliminary result only.

## 1. Pre-registration

| Item | Commit |
|---|---|
| Reading notes, seed inventory, replication seeds (`seeds.json`) | `6702c83` |
| Step-1 gates (`gates.json`), before the addendum | `10c0966` |
| **Addendum pushed** (`results/revision10/ADDENDUM.md`), before any replay on 230–249 | **`27aea29`** |
| Addendum header records its push commit (no other change) | `738dd3e` |
| Analysis code (`experiments/revision10/analysis/veins10.py`) + reproduction test, shard runner, reproduce scripts — before any data | `f9e4388` |
| First replay on 230–249 requested (H7) | `48f7883` |

The addendum's limitation list was cut off in the revision-10 instruction after its first item; the remaining items
were completed by the analyst before any replay and are marked as such in the addendum (§9).

## 2. Seeds and gates (`results/revision10/seeds.json`, `seed_inventory.json`, `gates.json`)

* **Seeds:** R-A = 230–239, R-B = 240–249 — the first 20 integers ≥ 230 found nowhere in the inventory (data seeds in
  every results file, Veins bundle paths on `revision-10`, `r8-ci`, `r9-ci`, every seed line of code; none skipped).
* **Pipeline gate — pass.** The seed-220 urban NLOS d=20 bundle, regenerated on two independent runners (Intel Xeon
  Platinum 8573C; Intel Xeon 6975P-C), gave the same content digest `5f07c276…` and matched every identity H6 committed:
  trace sha256 `e38b8fa0…`, t0 155.3 s, minEnd 175.3 s, playground 1200 × 1200 m, maxInterfDist 1888.5 m. (Bundles are
  compared by content — XML comment headers and the provenance stamp excluded — because H6 never committed its bundle
  bytes and SUMO writes a date and work path into every regeneration; READING_NOTES §3.)
* **Checkpoint gate — pass.** All 10 STRUCT9 parameter files loaded (80 generations each; sha256 recorded); gate check 1
  (`707d6533…`) and gate check 2 (`220dd1fd…`) byte-identical; the H6 replay rand_s0, δ = 0, urban NLOS d=20, seed 220
  re-run with events byte-identical to the committed one (`116f850b…`), identical SUMO trace (98,293 FCD rows),
  436 decisions re-sent offline with 0 mismatches.
* **Bundles for 230–249:** 40 (urban and rural × 20 seeds), all traces distinct, content digests recorded; no replay.
  Mobility sanity (report only): urban 855–882 vehicles, achieved 19.58–20.45 veh/km/lane (commanded 20); rural 524
  vehicles, 21.22–21.25 veh/km/lane (the rural corridor holds its count fixed) — all within the 25 % tolerance.
* **Veins analogue models (report only; settles a reviewer comment).** The built Veins 5.3 tree
  (`~/veins-r5/veins-veins-5.3/src/veins/modules/analogueModel/`) contains BreakpointPathlossModel, NakagamiFading,
  PERModel, SimpleObstacleShadowing, SimplePathlossModel, TwoRayInterferenceModel and VehicleObstacleShadowing; **there is
  no LogNormalShadowing model**, and "lognormal" in any spelling occurs nowhere in the Veins source
  (`gates.json` `veins_source_check_report_only`). The project's per-replay `config.xml` enables
  **BreakpointPathlossModel + NakagamiFading (constM, m = 1) + SimpleObstacleShadowing** (9 dB per cut, 0.4 dB/m, when the
  bundle has buildings) — `experiments/revision5/veins/run_veins.py:47-58`, reused by
  `experiments/revision8/veins/run_veins8.py:34`, read via `experiments/revision8/veins/aiharp/omnetpp.ini:12,29-30`.
  `experiments/revision5/veins/export.py:142-147` already recorded log-normal shadowing as not representable in Veins 5.3.

## 3. Replays and acceptance

| Run | Cell | Seeds | Shards accepted | Replays | Rows file |
|---|---|---|---|---|---|
| H7 | urban NLOS d=20 | R-A 230–239 | 10/10 | 600 | `veins/h7_rows.jsonl` |
| S3 | urban NLOS d=20 | R-B 240–249 | 10/10 | 600 | `veins/s3_rows.jsonl` |
| S5 | rural d=20 | R-A 230–239 | 10/10 | 600 | `veins/s5_rows.jsonl` |

Every accepted shard: gate checks 1 and 2 passed, regenerated bundle digest equal to `gates.json`, 60/60 replays merged,
60/60 decision logs re-sent offline with 0 mismatches (`veins/<mode>_rows_merge.json`, per-shard `gate.json`,
`trace_check.json`, `merge.json`). No shard was re-run. Decision-server logs are on `r10-ci` only.

## 4. H7 — primary (`results/revision10/analysis/h7.json`)

Urban NLOS d=20, R-A, thresholds `results/revision8/veins_thresholds.json` (q* 0.498, m* 0.602).

| | STRUCT9-rand qualifying | STRUCT9-plain qualifying | Rule met | One-sided Fisher p | Supported |
|---|---|---|---|---|---|
| H6 (revision 9, seeds 220–229, of record) | 4/5 | 0/5 | yes | 0.0238 | yes (p_H6 0.0476) |
| **H7 (R-A 230–239)** | **0/5** | **0/5** | **no** | **1.000** | **no** |

**H7 not supported → H6 not replicated; H6 is a preliminary result only** (ADDENDUM §5).

## 5. S3 — urban R-B and pooled (`results/revision10/analysis/s3.json`, descriptive)

| Seeds | STRUCT9-rand qualifying | STRUCT9-plain qualifying | Rule met | Fisher p |
|---|---|---|---|---|
| R-B 240–249 | 2/5 (s1, s4) | 1/5 (s4) | no | 0.500 |
| R-A ∪ R-B, 20 seeds pooled | 2/5 (s1, s4) | 0/5 | no | 0.222 |

## 6. S4 — per training seed, R-A vs H6 (`results/revision10/analysis/s4.json`, descriptive)

Qualification status repeats for **6 of 10** training seeds (all five plain seeds and rand_s0 fail on both); the four
seeds that qualified in H6 (rand_s1–s4, matched cost 0.808 / 0.710 / 0.915 / 0.947) all fail on R-A. Best mean RWCR and
the actionable miss at that δ, per seed (✓ = qualifies):

| Training seed | H6 220–229 | R-A 230–239 | R-B 240–249 |
|---|---|---|---|
| rand_s0 | 0.517 / 0.679 | 0.501 / 0.644 | 0.502 / 0.628 |
| rand_s1 | ✓ 0.525 / 0.581 | 0.497 / 0.605 | ✓ 0.521 / 0.599 |
| rand_s2 | ✓ 0.518 / 0.538 | 0.482 / 0.572 | 0.497 / 0.563 |
| rand_s3 | ✓ 0.513 / 0.536 | 0.480 / 0.563 | 0.497 / 0.554 |
| rand_s4 | ✓ 0.525 / 0.555 | 0.490 / 0.571 | ✓ 0.510 / 0.554 |
| plain_s0 | 0.447 / 0.773 | 0.442 / 0.783 | 0.461 / 0.775 |
| plain_s1 | 0.481 / 0.583 | 0.457 / 0.607 | 0.465 / 0.610 |
| plain_s2 | 0.462 / 0.629 | 0.461 / 0.656 | 0.479 / 0.650 |
| plain_s3 | 0.462 / 0.689 | 0.481 / 0.674 | 0.495 / 0.696 |
| plain_s4 | 0.511 / 0.612 | 0.483 / 0.684 | ✓ 0.510 / 0.705 (qualifies at another δ: miss 0.593) |

Reading (descriptive): STRUCT9-rand's misses stay below m* on R-A for s2–s4 (0.563–0.572), but its best reach fell from
0.513–0.525 to 0.480–0.501, just under q* = 0.498. Qualification in this cell is decided within a few hundredths of RWCR
around q*, and which 10 mobility seeds are drawn moves the outcome.

## 7. S5 — rural d=20 on R-A (`results/revision10/analysis/s5.json`, descriptive)

Rural d=20, R-A, q* 0.879, m* 0.404. **STRUCT9-rand 0/5, STRUCT9-plain 0/5** (rule not met, Fisher p = 1.0) — the same
as H6's rural cell on 220–229 (0/5 vs 0/5). Best mean RWCR and the actionable miss at that δ:

| Training seed | H6 220–229 | R-A 230–239 | Reach ≥ q* with miss > m* on R-A |
|---|---|---|---|
| rand_s0 | 0.835 / 0.555 | 0.868 / 0.592 | no (reach below q*) |
| rand_s1 | 0.888 / 0.414 | 0.896 / 0.437 | **yes** |
| rand_s2 | 0.877 / 0.389 | 0.875 / 0.350 | no (miss < m*, reach 0.004 below q*) |
| rand_s3 | 0.871 / 0.354 | 0.872 / 0.354 | no (miss < m*, reach 0.007 below q*) |
| rand_s4 | 0.859 / 0.375 | 0.875 / 0.352 | no (miss < m*, reach 0.004 below q*) |
| plain_s0 | 0.836 / 0.523 | 0.738 / 0.594 | no |
| plain_s1 | 0.826 / 0.395 | 0.832 / 0.383 | no |
| plain_s2 | 0.711 / 0.511 | 0.668 / 0.554 | no |
| plain_s3 | 0.776 / 0.519 | 0.813 / 0.472 | no |
| plain_s4 | 0.871 / 0.495 | 0.860 / 0.496 | no |

Reading (descriptive): the revision-8 "reach without timeliness" pattern (RWCR above q* while miss stays above m*)
repeats for one STRUCT9 seed (rand_s1). The other STRUCT9-rand seeds show the opposite near-miss, as on 220–229: miss
under m* (0.350–0.354) with best reach 0.004–0.007 below q* (cost at that δ 0.47–0.81). The rural cell remains unsolved
by every STRUCT variant on every seed set examined.

## 8. Deviations (`results/revision10/DEVIATIONS.md`)

None. Compute ran in the pre-registered order H7 → S3 → S5; no shard failed or was re-run; nothing fixed in the addendum
was changed.

## 9. Reproduction

* `./reproduce_revision10.sh` (default: analysis only; NumPy 1.26.4 via `AIHARP_PY`) regenerates `gates.json` and
  `analysis/h7.json`, `s3.json`, `s4.json`, `s5.json` (and the assembled `veins/<mode>_rows.jsonl`) from committed files.
  **Checked 2026-11-19: exit 0; all 11 outputs identical to the committed files apart from `provenance`** (the three
  assembled row files byte-identical).
* `reproduce_revision10.ps1`: the same steps for Windows; not executed here (no PowerShell in the cloud container).
* `--from-scratch` prints how to rerun the runner work (`ci/r10-gates.json`, `ci/r10-veins.json` on `revision-10`;
  outputs on `r10-ci`; never commit `bridge_logs/` or `gate2_logs/`).
* `experiments.revision10.seed_inventory` is not part of the script (it fetches `r8-ci` / `r9-ci`); run it by hand to
  re-derive `seeds.json`.
* Tests: 692 passed, 2 skipped (full suite, before every code push), including `tests/test_revision10_analysis.py`
  (the H7 code reproduces `h6.json`'s urban cell from the committed 220–229 rows) and `tests/test_revision10_gate.py`
  (bundle digest invariances; the pre-registered Fisher table).

## 10. Open questions

1. **How variable is per-seed qualification in this cell?** STRUCT9-rand's best RWCR moved by ~0.02–0.04 between the
   three 10-seed sets (220–229, 230–239, 240–249) and q* sits inside that range; a 10-seed matched-cost verdict per
   training seed is evidently unstable here. Quantifying that (e.g. a bootstrap over seeds) would be descriptive and is
   not pre-registered.
2. **Is the urban cell's threshold derivation (Tier A 200–209 classical rows) itself seed-sensitive?** The thresholds
   are frozen and were not re-examined; doing so would be a new pre-registration.
3. Every limitation of revisions 8–9 stands: post-hoc origin of H6, 5 training seeds per arm, causal-vs-oracle miss
   proxy, R2 [ASSUMED], no log-normal shadowing in Veins 5.3.
