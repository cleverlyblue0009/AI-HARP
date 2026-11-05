# Revision 7 — addendum (written before any revision-7 replay or run)

Branch `revision-7`, from `origin/revision-6` at c766e7a. Registered analyses of revisions 5 and 6 stay as they are, and
no revision-5 or revision-6 result is changed. Rural d=80 is not replayed in this revision.

**Interpreter:** the instruction names `D:\aiharp-gpu\python.exe`; that path does not exist. The pinned interpreter is
`D:\aiharp-gpu\Scripts\python.exe` (NumPy 1.26.4, torch 2.2.2), the same one used in revision 6. Every job logs
`sys.executable` and the NumPy version.

## What had been seen before this was written

* All of revision 6: `REVISION6_REPORT.md`, `veins_methods.json`, `veins_methods_diagnostics.json`. In particular the
  uncapped policy's best mean RWCR at rural d=20 in Veins is 0.864 (flooding's value) against q* = 0.879.
* **No capped result and no fixed-beacon result of any kind exists.**
* Code reading, recorded because it bears directly on Experiment A:
  * `sim/engine.py` delivers at most one reception callback per vehicle per epoch, however many frames it decoded, and
    timers run after receptions. No earlier decision can exist in the epoch when a duplicate callback runs, so **the cap
    is expected never to fire in the custom simulator**. The custom runs are still made, as instructed, and the number of
    firings is reported. This also means the 2,223 vs 635 comparison is partly an effect of this collapse.
  * The policy already answers a settled vehicle's duplicate with SUPPRESS without querying the network
    (`agents/ai_harp.py`, `_pre_network`).
* A count over the 900 revision-6 decision logs (requests and replies only, `D:\aiharp-r6-bridge-logs`), per replay:

  | Cell | Policy calls | Duplicate callbacks | Repeat in the same epoch | Of those, reply was not suppress |
  |---|---|---|---|---|
  | Rural d=20 | 2,542.9 | 2,175.8 | 1,581.7 | 33.8 |
  | Urban NLOS d=20 | 3,002.5 | 2,537.9 | 1,820.4 | 18.9 |

  So the cap should remove about 60 % of calls, but only about 34 and 19 calls per replay could change behaviour directly.
  **A large fall in decisions is therefore expected and is not by itself evidence that call frequency caused the failure.**
  These counts come from uncapped rollouts; capped rollouts diverge and may differ.

None of this changes the design or the decision rules below, which are those instructed.

## Experiment A — decision-frequency control

**Question.** Is the policy's failure in Veins a property of the policy, or of how often it is called?

**Option `max_decisions_per_epoch`**, default off (0), value 1 in this experiment, applied only to the learned policy:
* A vehicle's last decision epoch is recorded whenever the policy is called, for any trigger.
* On a duplicate reception that would reach the policy (the existing conditions: not cancelled by the duplicate count, no
  pending transmission), if the vehicle already has a decision in this 100 ms epoch, the policy is **not called** and the
  callback is answered with SUPPRESS.
* The answer has SUPPRESS's existing effect: it clears a pending transmission (none can exist here) **and a pending carry
  timer**. It does not mark the vehicle as settled and does not change the policy's state.
* The duplicate count, largest sender distance and the overheard set are updated as before. First receptions, origination
  and timer decisions are never capped.
* Sampling: calls that are made use their existing streams. In Veins that is (replay seed, vehicle, epoch, k), with k
  counting calls made in that epoch; in the custom simulator the policy generator is simply not drawn from by capped callbacks.
* Implemented in `sim/engine.py` (a `SimSettings` flag, off by default) and in a copy of the revision-6 app under
  `experiments/revision7/veins/aiharp` (an OMNeT++ parameter, off by default). Capped callbacks are counted: log line `C` in
  Veins, `capped_callbacks` in custom. They are not decisions.

**Runs.** Rural d=20 and urban NLOS d=20; training seeds ref_s0–ref_s4; biases −2, −1, 0, 1, 2, 3; τ = 0.5; sampled actions;
seeds 200–209; SUMO mobility; epoch mode in the custom simulator. 600 Veins replays and 600 custom runs.

**Analysis (`veins_methods` definitions).** Within each simulator, on seeds 200–209: q* and m* from the existing classical
rows (Veins: `results/revision5/veins/veins_rows.jsonl`; custom: `classical__EVAL__headline__default__sumo.csv`); matched cost
by Algorithm 1 per training seed; a seed "qualifies" in a cell if its matched cost is finite; the aggregate is defined when at
least 3 of 5 seeds qualify. Uncapped figures: Veins from `results/revision6/veins/veins6_rows.jsonl`, custom from
`agent__EVAL__headline__default__sumo.csv`. Reported per simulator and cell, with and without the cap: per-seed matched cost,
qualification, aggregate, decisions per replay, transmissions per replay, plus RWCR, cost and miss per seed and bias. Decisions
in Veins are `D` lines; the custom uncapped rows have no decision count, so it is taken from the capped runs only if those
show zero firings, and otherwise reported as unavailable. This is an EVAL-hindsight comparison on 10 seeds, with no new test.

**Decision rule.**
* If the capped policy qualifies in **Veins at rural d=20 for at least 3 of 5 training seeds**: the cap is called
  **"sufficient to restore qualification"**.
* Otherwise: **"not sufficient"**.
* Both outcomes are reported. "Not sufficient" does not show that call frequency plays no part. Urban NLOS d=20 and the custom
  simulator are reported but do not enter the rule.

## Experiment B — D9-affected classical ports

**Setup.** A new app build gets a parameter `beaconCamVelocity`, default off, which sends the CAM velocity in beacons exactly as
the revision-6 policy kinds do. The revision-5 and revision-6 binaries are untouched. On: `greedy_farthest` (all 6 settings)
and `dvcast` (all 9), rural d=2, rural d=20, urban NLOS d=20, seeds 200–209: 450 replays. Scored with the existing scorer.

**Report (descriptive; the registered analysis stays as it was).**
* Matched cost of both families with the fix, next to the revision-5 values, with RWCR, cost and miss per setting.
* Whether the best classical family per cell changes.
* Whether V1, V2 and V3 would change: `veins_methods` point statistics and its bootstrap (B = 10,000, RNG 20261102, Holm), with
  the fixed rows replacing those two families' rows in the three cells, thresholds recomputed, and every other row as in
  revision 6. **Rural d=80 keeps its revision-5 rows for both families** and is flagged as unfixed.

There is no decision rule beyond these comparisons.

## Validation, before any reported run

The build goes to `~/veins-r7/aiharp`. Each check must pass; a failure stops that experiment and is reported.
1. **Off means unchanged, classical:** `slotted_1p` n=8, rural d=20, seed 200, event log byte-identical to the revision-6 gate
   (`707d6533…`).
2. **Off means unchanged, affected ports:** `greedy_farthest` and `dvcast`, one setting each, same cell and seed, byte-identical
   to the stored revision-5 replays.
3. **Off means unchanged, policy:** ref_s0, bias +1, same cell and seed, byte-identical to the revision-6 replay.
4. **Bridge unit check** repeated: 1,000 contexts, max difference ≤ 1e-6, 0 gate mismatches.
5. **Trace check** repeated on a capped replay from the new binary: 0 mismatches.
6. **Custom simulator:** the flag's default leaves behaviour unchanged (test), and a run with the flag on matches the flag off.

Outputs of 4 and 5 go to `results/revision7/veins/`; the revision-6 validation files are not rewritten.

## Acceptance, rerun and output rules

* A replay is accepted only with exit code 0, an identical SUMO trace and the gated revision-7 binary. Rejected replays are
  counted and listed, never scored.
* Every job is resumable and stamped. Replay records are never overwritten once good.
* Raw OMNeT++ outputs (`.events.csv.gz`, `.sca.gz`) are committed if they total under 1 GB; otherwise they are zipped for Zenodo
  and the size is reported.
* If a step blocks: record the error, try at most two fixes, then report.
