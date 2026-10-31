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

## D2 — incident, no protocol change: SEL sweeps ran on the wrong mobility backend (2026-10-19T10:05+05:30)

The first SEL classical and SEL agent sweeps (00:31–09:56) ran on SUMO mobility instead of the fallback
backend, because SUMO_HOME had been set as a user variable and the runner used backend "auto". Found by the
bit-exact gate before any SEL result was computed or inspected. The rows are quarantined in
`results/revision5/runs/invalid_sumo_backend/` (README there), the backend is now explicit everywhere in the
protocol runner, and both SEL sweeps restart from zero. Training was not affected (verified). Cost: ~9.5 h of
SEL compute.

## D4 — budget checkpoints at 1,200 and 1,700 instead of 1,250 and 1,750 (2026-10-31T11:08+05:30)

**Pre-registered:** budget extension of full-model seeds 0–2 to 2,000 updates, evaluated at 1,250, 1,500, 1,750 and
2,000 updates.

**Changed to:** 1,200, 1,500, 1,700 and 2,000 (plus the existing 1,000-update evaluation of the same seeds).

**Why:** the extension runs saved checkpoints every 100 updates (`checkpoints/revision1/ref_s{0,1,2}_u2000/ckpt_0011..0020`),
so no 1,250- or 1,750-update checkpoint exists. Rule applied: the last checkpoint saved before each missing
pre-registered point. Retraining to save those two points was not chosen (it would add ~9 h of training for two
interpolating points).

**Results seen before this change:** none — no budget-checkpoint evaluation row existed when this entry was written.
