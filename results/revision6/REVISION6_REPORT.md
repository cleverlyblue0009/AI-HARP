# Revision 6 — report

Branch `revision-6`. Pre-registration: `results/revision6/PREREGISTRATION_ADDENDUM.md` (deviations D9, D10, D11). Numbers
come from `results/revision6/analysis/veins_methods.json` (pre-registered analysis) and
`veins_methods_diagnostics.json` (descriptive and post hoc, written after the first had been read). Table:
`analysis/tables/T-veins-methods.tex`. Figure: `figures/fig_veins_vs_custom.pdf`. No revision-5 result, classical replay
or Veins radio or MAC setting was changed.

## Answer to the reviewer

**1. "Run the learned policy (AI-HARP) and the rule baselines in Veins." Done, with a reduced scope for the policy.**
The rules ran in all four headline cells. The policy ran in three cells with five training seeds; rural d=80 was not
run for it (D10, decided on cost and time).

* **The policy does not qualify in Veins.** Its aggregate is undefined in all 3 cells run, and 1 of 15 (training seed,
  cell) pairs has a matched cost (Wilson 95 % [0.012, 0.298]). V2 and V3 therefore have no point statistic. This is not
  evidence of "no difference": where the comparison can be read descriptively, the policy costs far more (below).
* **The rules are cheaper than the best classical family in 2 cells, without statistical support.** The V1 count is
  met, but 35.6 % of bootstrap replicates fall below 2 cells (Holm p = 0.483).

**2. "Use more training seeds for the diagnostic variants." Not done.** Step 5 was stopped by D10 on cost and time.
`T-ablation10`, `T-ladder10` and `variants10.json` do not exist, and no variant has more than the revision-5 seeds. Partial
checkpoints are in `checkpoints_unused/`, labelled and never evaluated. **This request remains unmet.**

## Results

EVAL-hindsight comparison on Tier A seeds 200–209: thresholds, crossings and comparison use the same seeds. Tier B
(210–229) and the cross-fitted version were not run. Matched cost is transmissions per informed at-risk vehicle at q*,
over settings with mean actionable miss ≤ m*.

| Cell | q* | m* | Best classical | Best rule | Policy picks | Policy cost |
|---|---|---|---|---|---|---|
| Rural d=2 | 0.095 | 1.042 | DV-CAST 1.308 | RAR 1.000 | 1 / 5 | undefined |
| Rural d=20 | 0.879 | 0.404 | DV-CAST 0.226 | none | 0 / 5 | undefined |
| Rural d=80 | 0.870 | 0.371 | slotted_1p 0.322 | RAR 0.165 | not run | not run |
| Urban NLOS d=20 | 0.498 | 0.602 | none | none | 0 / 5 | undefined |

| Test | Statistic | 95 % CI | p | Holm p | Reading |
|---|---|---|---|---|---|
| V1, best rule cheaper than best classical in ≥ 2 cells | 2 cells | – | 0.356 | 0.483 | count met, not significant |
| V2, policy vs best classical | undefined | [−0.177, +3.764], see below | 0.065 | 0.194 | cannot be evaluated |
| V3, policy vs best rule | undefined | [−0.293, +3.761], see below | 0.242 | 0.483 | cannot be evaluated |
| V4, Veins vs custom on SUMO (descriptive) | ρ = 0.614, 12 pairs | [0.105, 0.826] | – | – | no policy pair |

**The V2 and V3 intervals are not estimates of the point result.** They are computed only on the replicates where the
statistic happens to be defined (6,764 and 5,118 of 10,000; the rest are undefined and excluded, as pre-registered),
while the statistic on the actual data is undefined. They should not be quoted as the policy's effect.

V1 bootstrap distribution, cells where the rule is cheaper: 1 cell 3,555, 2 cells 5,627, 3 cells 818 replicates.
Qualification: best classical 3 / 4 cells, best rule 2 / 4, policy aggregate 0 / 3.

V4 compares the Veins side with a custom simulator that failed H3 in revision 5 (D6), so it describes agreement only.
All 12 pairs (10 classical, 2 rule) are cheaper in Veins than in the custom simulator.

## Why the policy does not qualify (descriptive, post hoc)

| Cell | Policy best mean RWCR | q* | Policy lowest mean cost | Lowest mean cost, classical settings / rule settings |
|---|---|---|---|---|
| Rural d=2 | 0.100 | 0.095 | 0.700 | 0.650 / 1.000 |
| Rural d=20 | 0.864 | 0.879 | 1.107 | 0.210 / 0.204 |
| Urban NLOS d=20 | 0.4976 | 0.4983 | 1.244 | 0.246 / 0.203 |

The lowest-cost columns ignore q* and m*; they show the range of cost, not matched cost.

* **Rural d=20:** the policy's best reach, 0.864, equals flooding's and stays below q*; only slotted_1p and DV-CAST reach
  q* within the guard. Near that reach it costs 1.47–2.11, against 2.51 for flooding. On the custom simulator the policy
  aggregate qualifies here at 0.790, against 0.493 (DV-CAST) and 0.446 (RAR): it is costlier there too, and does not
  qualify at all in Veins.
* **Urban NLOS d=20:** **no method qualifies.** q* is set by a slotted_1p setting (0.525) that fails the miss guard.
  Policy reach matches flooding's (0.497) at costs of 1.24–3.34.
* **Rural d=2:** reach is about 0.09–0.10 for every method, and the policy's settings take only the values 0.094 and
  0.100. Flooding itself falls below q* (0.094 vs 0.095). The single pick (seed 2, 0.865) and the qualification of rules
  and classical families in this cell therefore rest on very small differences, and the seed 2 pick should not be read
  as a win. The RAR result here, which is one of V1's two cells, carries the same caution.
* **Rural d=80:** no policy result. This is the densest cell, so nothing can be claimed about the policy under high
  density in Veins.

Cause not established. The validation replay (rural d=20, seed 200) showed 2,223 decisions in Veins against 635 in the
custom simulator, with more duplicate receptions, each a decision callback. That is a candidate, not a tested explanation.

## Other findings a reader needs

* **D9, revision-5 ports (found in this revision).** The revision-5 greedy_farthest and DV-CAST Veins ports ran with zero
  beacon velocities (`setHostSpeed = false`). The revision-6 app fixes this for RAR, SCF and the policy only; rule 6
  forbade changing the classical replays, and revision-5 results are unchanged. **DV-CAST is the best classical family
  in two cells, so its numbers here come from that affected port.** Without both families (post hoc): best classical
  becomes slotted_1p at rural d=2 (1.536) and d=20 (0.235); V1 stays at 2 cells; V2 and V3 stay undefined.
* **One-epoch lag.** In the Veins coupling as configured for the classical replays, mobility at a decision in epoch k
  equals the custom trace at epoch k − 1. It applies to all methods and was not changed.
* **The SCF timeout has no effect.** In every cell, and in both simulators, the timeouts 2, 5 and 10 s give identical
  RWCR and cost at each slot count. SCF therefore has 4 distinct settings rather than 12. Its matched cost equals
  slotted_1p's at rural d=2 (1.536) and d=80 (0.322).
* **RAR reaches much less in Veins at low and medium density.** Best mean RWCR, Veins vs custom: rural d=2 0.100 vs
  0.210; rural d=20 0.593 vs 0.860; rural d=80 0.913 vs 0.912; urban 0.433 vs 0.477. The port's decisions match the
  reference code given its inputs (0 mismatches in 269). Beacon-derived neighbour counts equal the geometric count in only
  7 of those 269 decisions, a candidate cause that was not tested.
* **Analysis bug found before scoring (D11).** Resampled run labels would have given V2 and V3 no intervals. It was fixed
  before any Tier A score existed, and no definition changed.

## Port validation

| Check | Result |
|---|---|
| Binary identity, classical replay, revision-6 vs revision-5 binary vs stored | event logs byte-identical |
| RAR: ρ_c and decisions recomputed by the reference code | max difference 0.0; 0 mismatches in 269 |
| SCF: decisions recomputed by the reference code | 0 mismatches in 424 |
| Policy unit check: 1,000 contexts, in-process vs through the bridge | max difference 0.0; 0 gate mismatches |
| Policy trace check: 2,223 logged requests re-sent offline | 0 mismatches |

These checks show the ports decide as the reference code does on the given inputs, on one cell and seed for the replays.
They do not show that the inputs match the custom simulator's; they do not. Details: `veins/port_validation_rules.md`,
`veins/port_validation_policy.md`.

## Replays, scope and compute

Scoring: **merged 2,460, rejected 0** (`veins/veins6_rows_merge.json`). Acceptance: exit code 0, identical SUMO trace,
gated app binary. Neither pre-registered cut applied (72 h for rules, 96 h for the policy).

| Track | Replays | Elapsed, first start to last finish | App core-hours (sum of replay wall time) |
|---|---|---|---|
| Rules | 1,560 | 57.7 h (2026-11-02 12:43 to 11-04 22:23) | 522.1, of which 54.3 h overlap standby (22 replays) |
| Policy | 900 | 16.5 h (11-04 22:29 to 11-05 15:01), about 7 h running | 55.3, decision servers not included |

Rural d=80 took 450 of the rule core-hours (median 3,496 s per replay). The stopped variant training is not totalled here.

**Reduced by D10, on cost and time:** policy rural d=80 (300 replays); training seeds 5–9 (Tier A-P2, 1,200 replays); Step 5.
When D10 was decided, no Tier A result had been scored, but the validation sanity replays had been seen; the addendum
states which.

**Interruptions.** Every replay resumes from its record, and no finished record was lost.

| When | What | Rework |
|---|---|---|
| 11-02 16:25–18:41 | Sign-out and sleep, cause not identified | none recorded |
| 11-03 00:19 | Accidental shutdown; restart 00:21 | up to 11 replays |
| 11-03 06:36–08:26 | Power cut, standby | none |
| 11-03 15:50–18:56 | Power change, standby until the power button | none |
| 11-04 16:08–19:08 | VS Code restarted about 16:07, WSL shut down 16:08; link between them likely, not proven | up to 11 replays |
| 11-05 01:55–11:21 | Logoff and software-requested sleep ("Application API"), cause not identified | up to 8 replays |

**"FAILED" log lines are not replay failures** (418 in total, each checked by time):
* 178 in `tierA_rules.log` and 231 in `tierA_policy_A.log`, written within seconds as the system went down at 00:19 on
  11-03 and 01:54 on 11-05. The queue ran through its list while WSL was unavailable; no records were written, and the
  replays were retried.
* 9 in `tierA_rules_rev.log`, on 11-04 from 21:40: the second queue found the last three replays claimed by the first.

**Training incident.** A watchdog bug exported `SUMO_HOME`, so gcn_s7 resumed on SUMO mobility from update 492 on 11-02 at
19:00, and the assistant running the jobs first told the author, wrongly, that there were no bad results. The bug was fixed in f64380c and the run quarantined
as `gcn_s7_mixed_mobility_invalid`. It was never evaluated and is in no result.

## Limits on these results

* Hindsight comparison on 10 replay seeds: thresholds are fitted on the seeds they are compared on.
* Five training seeds, three cells for the policy; no policy result at the highest density.
* One binary, one configuration and SUMO mobility throughout. The classical side is the revision-5 replays, from a
  different machine; the identity gate covers the application binary on one replay, and wall times are not comparable.
* Bootstrap p-values use the pre-registered definitions, on few cells; V1's statistic can only take a handful of values.

## Reproduce

`reproduce_revision6.ps1` or `.sh` (NumPy 1.26.4 guard; about 5 minutes) regenerates the analysis, table, figure and manifest
from committed rows; `--from-scratch` reruns the replays first. **Clean-clone check, passed:** a fresh clone at c0d78b2
gave both JSON outputs identical apart from provenance, and the table and figure identical once line endings are normalised.
The replay records and logs are in `cache/` and `D:\aiharp-r6-bridge-logs`, which are not committed, so the scoring step
itself was not part of that check. `results/MANIFEST.md` and `results/manifest.json` list every output.
