# Revision 5 — deviations from pre-registration

Every entry: timestamp, what changed, why, and whether any result had been seen.

## D1 — H3 seed sets and the Veins H1 verdict (2026-10-19T09:56+05:30, user decision)

**Pre-registered:** "SUMO mobility, identical traces and seeds, 4 headline cells, 10 seeds per cell", with
"identical H1 verdict in both simulators" as one of the H3 criteria. Which seeds, and how the Veins H1
verdict obtains its CAL/SEL/EVAL split, were not specified.

**Changed to:**

1. **Network-layer criteria** (per-hop PDR vs distance, collision and loss rates, channel busy ratio,
   single-hop delay and TIR distributions): EVAL seeds 200–209, each replay paired with the custom
   simulator on the identical trace. Bootstrap CIs over packets and seeds.
2. **Veins H1 / matched-coverage audit:** EVAL seeds 200–229 with 3-fold cross-fitting. Three folds of
   10 seeds; target (q*) and guard (m*) are computed, and each scheme's setting selected, on the other
   two folds; the held-out fold is scored; the folds rotate so every seed is scored exactly once. The
   bootstrap resamples the whole procedure, recomputing target, guard and selection in every replicate.
   The custom simulator is run on SUMO with exactly the same folds, so the two verdicts are directly
   comparable.

**Why:** the CAL/SEL/EVAL protocol (60 seeds × 55 settings × 4 cells) in Veins would take about 13,000
replays, which is infeasible here.

**Results seen before this change:** no H3 statistic or comparison had been computed. Pipeline
development used two flooding replays on seed 200 (rural_highway d=20 and urban_nlos d=20) to debug the
Veins application; they are development runs, not H3 data, and are not reused.

**Also decided with this entry:** greedy_farthest and DV-CAST are ported to Veins before scaling up (all
55 classical settings); if either cannot be ported, that is reported explicitly.
