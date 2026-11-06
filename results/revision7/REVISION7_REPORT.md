# Revision 7 — report

Branch `revision-7`, from `origin/revision-6` at c766e7a; not merged into `main`. Addendum `results/revision7/ADDENDUM.md`,
pushed at **2d1a1fc** before any replay or run. Rural d=80 was not replayed. Pinned interpreter
`D:\aiharp-gpu\Scripts\python.exe` (NumPy 1.26.4); the path `D:\aiharp-gpu\python.exe` in the instruction does not exist.
No revision-5 or revision-6 result was changed.

## Summary

**Experiment A: the decision cap is not sufficient to restore qualification.** With `max_decisions_per_epoch = 1`, the learned
policy qualifies in Veins at rural d=20 for **0 of 5** training seeds (3 needed), the same as without the cap. The cap
cut Veins decisions per replay from 2,542.9 to 978.2 (−61.5 %; 1,586.2 capped callbacks per replay) but transmissions only from
226.1 to 219.5 (−2.9 %), against 116.6 in the custom simulator. The policy's best mean RWCR per seed in Veins rural d=20 is
0.807–0.864 capped and 0.828–0.864 uncapped, below q* = 0.879 in every case, so no bias setting can qualify at any cost. Urban
NLOS d=20 qualifies 0 of 5 in both arms and both simulators. In the custom simulator the cap never fired (0 capped callbacks),
as the addendum expected, and all 600 capped runs equal the committed rows; custom rural d=20 stays at 3 of 5 seeds, aggregate
0.790. **Experiment B (descriptive):** with the D9 beacon-velocity fix, DV-CAST's matched cost goes from 1.308 to 1.353 at rural
d=2, and from 0.226 to undefined at rural d=20 (its qualifying setting, 30, drops from RWCR 0.8805 to 0.8764, below q* = 0.879);
greedy_farthest stays undefined in all three cells. The best classical family changes only at rural d=20 (DV-CAST 0.226 →
slotted_1p 0.235). V1 stays at 2 cells (Holm p = 0.487, registered 0.483); V2 and V3 stay undefined. Scoring: 600 + 450
replays merged, 0 rejected. Validation: all six checks passed.

## Validation (before any reported run; `results/revision7/veins/`)

| Check | Result |
|---|---|
| 1. `slotted_1p` n=8, rural d=20, seed 200, new binary with both options off | event log byte-identical to the revision-6 gate (`707d6533…`) |
| 2. `greedy_farthest` 1 and `dvcast` 1, same cell and seed, options off | byte-identical to the stored revision-5 replays |
| 3. Policy ref_s0, bias +1, options off | byte-identical to the revision-6 replay (`220dd1fd…`) |
| 4. Bridge unit check | 1,000 contexts, max difference 0.0, 0 gate mismatches |
| 5. Trace check, capped replay | 719 requests, 0 mismatches (1,189 capped callbacks; uncapped validation replay: 2,223 decisions) |
| 6. Custom simulator | flag-off default unchanged (tests: 22 passed); flag on equals flag off for all 600 runs |

Revision-7 binary `~/veins-r7/aiharp/aiharp`, sha256 `a1279a02…ca14e463`. The revision-5 and revision-6 binaries were not
rebuilt and their hashes are unchanged.

## Experiment A — decision-frequency control (`analysis/exp_a.json`)

EVAL-hindsight comparison on seeds 200–209, no new test. A seed qualifies when its Algorithm-1 matched cost is finite.

| Cell | Simulator | q* / m* | Arm | Seeds qualifying | Aggregate | Decisions / replay | Capped callbacks / replay | Transmissions / replay |
|---|---|---|---|---|---|---|---|---|
| Rural d=20 | Veins | 0.879 / 0.404 | uncapped | 0/5 | undefined | 2,542.9 | – | 226.1 |
| | | | **capped** | **0/5** | undefined | 978.2 | 1,586.2 | 219.5 |
| | Custom | 0.849 / 0.400 | uncapped | 3/5 | 0.790 | 766.2 | – | 116.6 |
| | | | capped | 3/5 | 0.790 | 766.2 | 0.0 | 116.6 |
| Urban NLOS d=20 | Veins | 0.498 / 0.602 | uncapped | 0/5 | undefined | 3,002.5 | – | 305.0 |
| | | | capped | 0/5 | undefined | 1,207.0 | 1,813.9 | 295.9 |
| | Custom | 0.528 / 0.603 | uncapped | 0/5 | undefined | 740.9 | – | 122.3 |
| | | | capped | 0/5 | undefined | 740.9 | 0.0 | 122.3 |

Custom rural d=20 per-seed matched costs (both arms): ref_s2 0.723, ref_s3 0.954, ref_s4 0.694; ref_s0 and ref_s1 undefined.
Best classical at rural d=20: DV-CAST 0.226 in Veins, 0.493 in custom (best rule: none in Veins, RAR 0.446 in custom). Custom
decisions for the uncapped arm are taken from the capped runs, which is allowed by the addendum because the cap fired 0 times.

Best mean RWCR per training seed over the six biases:

| Cell, simulator | Arm | ref_s0 | ref_s1 | ref_s2 | ref_s3 | ref_s4 | q* |
|---|---|---|---|---|---|---|---|
| Rural d=20, Veins | uncapped | 0.828 | 0.859 | 0.864 | 0.864 | 0.862 | 0.879 |
| | capped | 0.807 | 0.857 | 0.864 | 0.864 | 0.864 | 0.879 |
| Urban NLOS d=20, Veins | uncapped | 0.4971 | 0.4972 | 0.4971 | 0.4974 | 0.4976 | 0.4983 |
| | capped | 0.4974 | 0.4972 | 0.4971 | 0.4971 | 0.4972 | 0.4983 |

0.8636, the value several seeds reach, equals flooding's RWCR in that cell; the addendum recorded it (0.864) as the uncapped
policy's best before any capped replay. RWCR, cost and miss per seed and bias are in
`exp_a.json`. At high bias the cap lowers cost (e.g. ref_s0 bias 3: 1.135 → 0.848) without bringing reach to q*.

**Decision rule.** 0 of 5 < 3: **"not sufficient"**. As the addendum states, this does not show that call frequency plays no
part. The addendum also recorded, before any capped replay, that a large fall in decisions was expected either way: only about
34 (rural) and 19 (urban) same-epoch calls per uncapped replay had a non-suppress reply.

## Experiment B — D9-affected classical ports (`analysis/exp_b.json`, descriptive)

Fixed rows replace `greedy_farthest` and `dvcast` in rural d=2, rural d=20 and urban NLOS d=20; rural d=80 keeps its
revision-5 rows (**unfixed**). Thresholds recomputed; every other row as in revision 6.

| Cell | q* / m* | DV-CAST, revision 5 | DV-CAST, fixed | greedy_farthest, rev. 5 | greedy_farthest, fixed | Best classical, rev. 5 | Best classical, fixed |
|---|---|---|---|---|---|---|---|
| Rural d=2 | 0.095 / 1.042 | 1.308 | 1.353 | undefined | undefined | DV-CAST 1.308 | DV-CAST 1.353 |
| Rural d=20 | 0.879 / 0.404 | 0.226 | **undefined** | undefined | undefined | DV-CAST 0.226 | **slotted_1p 0.235** |
| Urban NLOS d=20 | 0.498 / 0.602 | undefined | undefined | undefined | undefined | none | none |
| Rural d=80 (unfixed) | 0.870 / 0.371 | 0.329 | – | 0.382 | – | slotted_1p 0.322 | slotted_1p 0.322 |

Why DV-CAST drops out at rural d=20: setting 30 had RWCR 0.8805, miss 0.4004 (both inside q*, m*); fixed, 0.8764 and 0.4151.
Setting 50 reaches 0.890 but misses 0.541 in both. greedy_farthest's closest setting (32) reaches 0.881 but misses 0.445
(revision 5) and 0.457 (fixed), above m* = 0.404. In urban NLOS, settings that reach q* miss more than m* in both versions.
Per-setting RWCR, cost and miss are in `exp_b.json`. The best rule is unchanged in every cell (RAR 1.000 at d=2, RAR 0.165 at d=80).

| Test | Registered (revision 6) | With fix |
|---|---|---|
| V1, cells where the best rule is cheaper than the best classical | 2 (rural d=2, d=80); P(fewer than 2) 0.356, Holm 0.483 | 2 (same cells); P 0.355, Holm 0.487 |
| V2, pooled relative difference | undefined; 95 % CI [−0.177, 3.764], p 0.065, Holm 0.194; 3,236 of 10,000 replicates undefined | undefined; CI [−0.239, 3.717], p 0.075, Holm 0.226; 3,239 undefined |
| V3 | undefined; CI [−0.293, 3.761], p 0.242, Holm 0.483; 4,882 undefined | undefined; CI [−0.293, 3.761], p 0.243, Holm 0.487; 4,882 undefined |

So V1, V2 and V3 would not change in substance. The registered analysis stays as it was.

## Execution record

* **Wall time.** Experiment A in Veins: 6.51 wall-hours on 8 workers (16:33–23:05, 5 Nov), plus a second queue of 3 workers
  for 0.68 h that ran 18 replays. Custom runs: 16:33–16:56. Experiment B: 23:14–about 23:46, then 2.00 wall-hours after the
  restart (finished 01:50, 6 Nov). All on one laptop (Ryzen 7 7445HS, 6 cores, 12 threads).
* **Second queue (19:45–20:28).** Added to use cores freed by the finished custom runs (commit c9d3bdc: `--reverse`,
  `--first-worker`; defaults unchanged). It did not raise throughput (92 replays/h against 96–97 before), so it was drained:
  its unstarted replays were marked as held, and it exited after its current replays. Same binary, seeds and acceptance rule.
* **Interruptions.** (1) The laptop slept 16:28:55–16:31:30 during the validation gate, before any reported run; the gate
  finished and passed. Sleep was then disabled on AC and battery. (2) The Claude Code session that kept the watchdog and the
  Experiment B queue running was closed by mistake between 23:45 and 23:47; WSL then failed to start
  (`Wsl/Service/CreateInstance/HCS_E_CONNECTION_TIMEOUT`). Fix 1: `wsl --shutdown`, which worked; the binary hash was rechecked
  and the queue was relaunched at 23:48:59. 202 of 450 replays were already good; the 11 that were in progress were rerun from
  the start. No replay was rejected.
* **Power setting.** A switch to the vendor "Turbo" mode at 19:07 did not change throughput (96/h against 97/h).
* **Gate error, fixed before any run.** The first gate run failed with `<!> Error: command-line option -` because a new runner
  argument was shadowed by a local variable; fixed by renaming it (`app_args`). No revision-6 result was affected.

## Files

* Rows: `veins/capped_rows.jsonl` (600), `veins/d9_rows.jsonl` (450), `runs/custom_capped_rows.jsonl` (600), with merge records.
* Analysis: `analysis/exp_a.json`, `analysis/exp_b.json`.
* Raw OMNeT++ outputs (`.events.csv.gz`, `.sca.gz`, `.run.json`): `veins/raw/` (r7_capped, r7_d9, r7_gate, r7_gate_capped),
  271.2 MB in total (268.0 MB of `.gz`), under 1 GB, so **committed**. The decision-server logs (`D:\aiharp-r7-bridge-logs`,
  901.7 MB) are not OMNeT++ outputs and are not committed.
* Manifest: `results/MANIFEST.md`, `results/manifest.json` (`python -m experiments.revision7.manifest`).
* Reproduce: `reproduce_revision7.ps1` / `.sh` (analysis only by default; `--from-scratch` reruns validation, replays and runs).
  Clean-clone check: a fresh clone of d80ff39 ran `reproduce_revision7.ps1` with exit 0, and `exp_a.json` and `exp_b.json`
  equal the committed files apart from their provenance stamps (the manifest changes only through those stamps' hashes).
* Code: `sim/engine.py` (`max_decisions_per_epoch`, off by default), `tests/test_decision_cap.py`, `experiments/revision7/`.
