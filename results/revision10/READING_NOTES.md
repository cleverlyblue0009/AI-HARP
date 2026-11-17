# Revision 10 — reading notes (written before any revision-10 replay)

Branch `revision-10` from `origin/revision-9` at `f1c8949`. Nothing below changes a revision 5–9 file. Citations are
`file:line` at that commit.

## 1. The H6 procedure, exactly as run in revision 9

**Checkpoints.** STRUCT9 parameters `results/revision9/train/{plain,rand}_s{0..4}/theta.json` (21 parameters, CMA mean after
generation 80). The decision server reads them from the replay config's `train_dir`
(`experiments/revision9/veins/decision_server9.py:28-34`; H6 passes `train_dir = results/revision9/train`,
`experiments/revision9/veins/ci9.py:33-34,124-126`). Policy RNG per replay: `np.random.default_rng([seed, STREAM_SALT, 1])`
(`decision_server9.py:33`). Every runner also restores the revision-1 AI-HARP checkpoints, hash-checked against
`results/revision5/checkpoint_shas.json` (`experiments/revision8/veins/ci_veins.py:188-200`), for gate check 2.

**δ set.** `DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)` (`ci_veins.py:43`); 10 runs × 6 δ = 60 replays per (cell, seed)
shard (`ci9.py:124-126`).

**Gates before any replay** (`ci9.py:99-122`): gate check 1, classical `slotted_1p` n=8, rural d=20, seed 200, events
byte-identical to `707d6533…` (`ci_veins.py:41`); gate check 2, AI-HARP `ref_s0` bias +1, byte-identical to
`results/revision7/veins/raw/r7_gate/rural_highway_d20_s200__ai_harp_ref_s0_b1__s200.events.csv.gz` (`220dd1fd…`,
`ci_veins.py:184-185`). After the replays: every decision log re-sent offline to a fresh `Session9`, 0 mismatches
(`ci9.py:128-135`, `decision_server9.py:78-89`); scoring only of exit-0 replays with an identical SUMO trace and the gated
binary (`ci9.py:136-145`, `ci_veins.py:160-170`). Shard acceptance for the analysis: both gates passed, trace check all
pass over n = 60, merged = 60 (`experiments/revision9/analysis/veins9.py:41-68`).

**Algorithm 1 / qualification.** Rows → per-(setting, seed) table, every (setting, seed) present exactly once
(`experiments/revision5/analysis/common.py:49-73`, called with `key = (run, δ)`, `veins9.py:87`). Per training seed and
cell: `matched(t, run, q*, m*)` (`experiments/revision6/analysis/veins_methods.py:103-107`) builds one operating point per
δ from the **seed means** of RWCR, cost (transmissions per at-risk informed vehicle) and actionable miss, and returns
`PolicyCurve.overhead_at(q*, eligible = miss ≤ m*)` (`analysis/pareto.py:100-140`: minimum cost reaching q* among
eligible settings, interpolated only against an eligible next-cheaper anchor; ∞ if none). **A training seed qualifies iff
that matched cost is finite** (`veins9.py:91-96`).

**Rule and test.** Per cell: number of qualifying seeds per arm; rule met iff rand ≥ 3/5 and plain < 3/5; one-sided Fisher
exact, `fisher_exact([[aq, an−aq], [bq, bn−bq]], alternative="greater")` (`veins9.py:103-117`). H6 (two cells):
p_H6 = min(1, 2·min p) (`veins9.py:116`); supported iff rule met in ≥ 1 cell and p_H6 < 0.05 (`veins9.py:120-126`).

**H6 result of record** (`results/revision9/analysis/h6.json`): urban NLOS d=20 rand 4/5, plain 0/5, p = 0.0238; rural
d=20 0/5 vs 0/5; p_H6 = 0.0476; supported. Post-hoc motivated (`results/revision9/ADDENDUM.md` §0).

## 2. Frozen thresholds

`results/revision8/veins_thresholds.json` (derived from Veins Tier A 200–209 classical rows,
`experiments/revision8/veins_thresholds.py:19`), read by `veins9.frozen()` (`veins9.py:36-38`):
rural d=20 q* = 0.878958, m* = 0.403870; urban NLOS d=20 q* = 0.498350, m* = 0.601940 (as echoed in `h6.json`
`thresholds`).

## 3. How Veins seed bundles and SUMO traces are generated

* Per shard the runner calls `bundles()` (`ci_veins.py:56-65`): `python -m experiments.revision5.veins.export <scenario>
  <density> <seed>` with the source-built SUMO 1.19.0 (`SUMO_HOME=~/veins-r5/sumo-1.19.0`), then
  `experiments.revision6.veins.r6_bundle --seeds s-s` (risk-field / hazard lines from the custom simulator's own objects,
  `experiments/revision6/veins/r6_bundle.py:28-47`; refuses a mismatching origin or vehicle count, `:64-69`).
* `export()` (`experiments/revision5/veins/export.py:163-247`): canonical trace `get_trace(..., backend="sumo")`
  (`:166`); regenerates the SUMO run and **refuses to export unless it equals the canonical trace exactly**
  (`:173-179`); copies net/routes/cfg files (`:192-196`); reference FCD `fcd_ref.npz` (`:197-200`); building polygons
  (`:201`); runs SUMO once more through `veins.sumocfg` and **refuses unless the FCD is identical** (`:202-205`); writes
  `bundle.json` with `trace_sha256` (`:48-56,231`), the Veins radio parameters (`:142-160,215`), vehicle activity
  intervals, origin and a provenance stamp (`:229-246`).
* SUMO seed = run seed + scenario `seed_offset` (`mobility/sumo_runner.py:541`; rural 1000, urban 2000, urban NLOS 3000:
  `configs/scenario_rural.yaml:79`, `scenario_urban.yaml:66`, `scenario_urban_nlos.yaml:75`); grid route RNG
  `default_rng(seed + 7919)` (`mobility/sumo_runner.py:390`). Within one scenario these derived seeds coincide only when
  the run seeds do, so a fresh run seed gives fresh SUMO and route streams.
* Density sanity in the generator: `_check_density` warns if achieved density misses the command by > 25 %
  (`mobility/sumo_runner.py:60,173-196`).
* **Nothing about a bundle is committed except what each replay records**: `bundle_trace_sha256` and the Veins command
  line (t0, minEnd, playground, maxInterfDist) in every `raw/*.run.json` (`experiments/revision8/veins/run_veins8.py:88-93,
  122`), plus the replay's own FCD identity check. H6 regenerated every bundle on its runner (`.github/workflows/
  r9-veins.yml:76-80`; no bundle cache step); `cache/veins/` is not committed. Searched: the whole tracked tree and branches
  `r8-ci`, `r9-ci` — no bundle files.
* **Consequence for the revision-10 pipeline gate (the repository wins):** the H6 bundle bytes cannot be compared because
  they were never kept, and two regenerations are never byte-identical as files: SUMO / randomTrips write a comment header
  with the date and the absolute work path into `net.net.xml`, `trips.trips.xml` and `demand.rou.xml`, and `bundle.json`
  embeds those files' hashes and a provenance stamp. Measured here (two local exports of urban NLOS d=20 seed 220,
  pip SUMO 1.19.0): every other file is byte-identical, and the differences are confined to those comment lines and
  `bundle.json`. The gate therefore uses a **content digest** (XML comments removed, `fcd_ref.npz` by array contents,
  `bundle.json` without provenance and with its file map recomputed; `experiments/revision10/veins/gate10.py`) that must
  be identical on two independent runners, **plus** every identity H6 committed for that bundle (trace_sha256
  `e38b8fa0…`, t0 155.3 s, minEnd 175.3 s, playground 1200 × 1200 m, maxInterfDist 1888.5 m), **plus** the
  checkpoint-gate replay, whose events must be byte-identical to the committed H6 replay (that replay ran on the
  regenerated bundle, so it tests the bundle end to end). The local export already reproduced trace_sha256 `e38b8fa0…`,
  t0, minEnd, playground and maxInterfDist.

## 4. Complete seed inventory

Produced by `experiments/revision10/seed_inventory.py` → `results/revision10/seed_inventory.json` (data seeds per
results directory; Veins path seeds on `revision-10`, `r8-ci`, `r9-ci`; every seed line of code with its integers).
The data scan reads every key containing "seed", so it also picks up a few counts stored under such keys (e.g.
`qualification_per_seed_cell` 13, 15, 16, 17, 19, 22, 24, 40 in `results/revision8/analysis/h4*.json` and
`results/revision9/analysis/s2.json`) — conservative: such values can only exclude candidates.

| Seeds | Use | Where defined |
|---|---|---|
| 0–9 | custom-simulator evaluation seeds (Phase 4/7 tables, `results/runs*.csv`, risk/slot studies) | `configs/experiment.yaml:83,123`; smoke `:108` (0–1) |
| 0–10 | AI-HARP agent evaluation (`results/agent_*`, revision 1) | revision-1 evaluation scripts |
| 0–29 | validation and revision-3 studies (`results/validation`, `results/revision3`) | revision-3 scripts |
| 0–4 | revision-2 studies; STRUCT / STRUCT9 **training** seeds (`plain_s0-4`, `rand_s0-4`) | `agents/train_structured.py` CLI |
| 100–131 | training pool (32 seeds): agents, coverage targets, revision-8/9 training, revision-9 gates | `configs/agent.yaml:363` |
| 200–209 | Tier A / NET / Veins Tier A (thresholds) | `experiments/revision5/analysis/common.py:19`, `revision8/analysis/h4.py:29`, `revision8/veins_thresholds.py:19` |
| 210–219 | Tier B part 1; revision-8 H5 Veins (D12); revision-5 D6 calibration | `revision8/analysis/h5.py:32`, `revision5/veins/d6_calibrate.py:37` |
| 220–229 | Tier B part 2; revision-9 H6 and C1 Veins | `revision9/analysis/veins9.py:31` |
| 200–229 | revision-5 EVAL (custom) | `revision5/analysis/common.py:19` |
| 300–309 | revision-5 CAL | `revision5/analysis/common.py:19` |
| 400–419 | revision-5 SEL (also used by revision-8/9 S2/H4 selection) | `revision5/analysis/common.py:19` |
| 900–904 | AI-HARP hold-out seeds (configured) | `configs/agent.yaml:50` |
| 1000 + k | CMA-ES seed for training seed k (1000–1004) | `agents/train_structured.py:245` |
| 20261016, 20261018, 20261031, 20261102, 20261116 | bootstrap RNG seeds | `revision1/b4_stats.py:45`, `revision2/r3_hier_bootstrap.py:34`, `revision5/analysis/common.py:25`, `revision6/analysis/veins_methods.py:10,39`, `revision8/analysis/h4.py:11` |

Veins path seeds: tracked 200–229; `r8-ci` 200–219; `r9-ci` 200–229. The largest data seed below the bootstrap seeds is
419; **no integer in 230–249 occurs in any results file, any Veins path on any of the three branches, or on any seed line
of code** (the only candidate, 230 at `veins9.py:31`, is the exclusive end of `range(220, 230)`; the scanner was corrected
to treat it as such). No markdown file mentions a seed in 230–249 (`git grep`). Hence **R-A = 230–239, R-B = 240–249**
(`results/revision10/seeds.json`), with nothing skipped.

## 5. Veins analogue models (for the source check)

The project writes `config.xml` per replay with `BreakpointPathlossModel`, `NakagamiFading` and, when the bundle has
buildings, `SimpleObstacleShadowing` (`experiments/revision5/veins/run_veins.py:47-58`, reused by
`run_veins8.py:34`); the app's `omnetpp.ini` reads them via `xmldoc("config.xml")`
(`experiments/revision8/veins/aiharp/omnetpp.ini:12,29-30`). `export.py:142-147` already states that log-normal
shadowing is "not representable in Veins 5.3 and therefore absent". The runner check lists the built tree's
`src/veins/modules/analogueModel/` to settle this from the source rather than from that comment.
