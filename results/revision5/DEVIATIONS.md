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

## D5 — stress-test channel: own CAL thresholds (pre-specified extension) (2026-10-31T11:15+05:30)

**What:** the 24-cell stress test on the R9 calibrated extra-loss channel (`--channel calibrated`) runs the full
protocol separately: CAL classical (seeds 300–309) -> q*, m* for this channel frozen in
`results/revision5/thresholds_stress.json` -> SEL and EVAL for classical, rule (RAR + SCF) and full-model seeds 0–4.
The default-channel thresholds are not reused, because q* and m* are properties of the channel.

**Why this is an extension, not a change:** the pre-registration lists the stress test but does not say which
thresholds it uses; the default channel's protocol is unchanged.

**Results seen:** none — no calibrated-channel row of any kind exists when this entry is written
(`results/revision5/runs/` has no `*calibrated*` file). Rule baselines get no CAL run: CAL only sets the thresholds,
which the protocol derives from the classical schemes.

## D3 — H3 completion rule and operational definitions (2026-10-31T11:19+05:30)

**Completion rule (user, 2026-10-31):** compute the network-layer criteria from the paired EVAL seeds 200–209; compute
the cross-fitted Veins H1 audit only if all 30 seeds x 55 settings x 4 cells are complete; if the audit cannot finish
within 48 h, report the network-layer criteria and the audit completion fraction, and no partial-seed Veins H1 verdict.

**Completion at the time of writing:** Veins 6,600 / 6,600 replays (network layer 2,200 / 2,200; audit 6,600 / 6,600),
every one with exit code 0 and an identical SUMO trace; 4,840 of them from the three gated Azure VMs. One replay
(rural_highway d=80 s=221 counter_based 1) lacks its OMNeT++ scalar file (busy-ratio input only); it is being regenerated
and accepted only if its event log is byte-identical, otherwise its busy ratio is reported missing. Custom simulator on
SUMO: 6,600 / 6,600 metric rows. **Both H3 parts are therefore computed.**

**Operational definitions (fixed here, before any Veins statistic is computed):**
* Custom-simulator network-layer data: the 2,200 paired runs (seeds 200–209) are re-run on SUMO with
  `SimSettings.record_rx_pairs` (records every in-nominal-range (tx, rx) outcome and distance; bit-neutral, verified by
  the rule-7 gate) because the existing rows hold aggregates only.
* PDR vs distance: per transmission, active receivers within the nominal range (excluding concurrent transmitters),
  binned by distance in 50 m bins; pooled over all hazard transmissions of the 55 settings; criterion |Veins − custom|
  <= 0.05 in every bin up to the nominal range; CIs by bootstrap over seeds.
* Single-hop delay: reception time of a first copy minus the time the sender itself became informed (Veins: event times;
  custom: (informed_step(r) − informed_step(sender)) x 0.1 s); criterion |median_Veins − median_custom| <= 0.1 s.
* Channel busy ratio: Veins = mean over vehicles of busy time while active / active time (Mac1609_4 busy signal) per
  replay; custom = the channel load its MAC model applies, `beacon_cbr` + `dissemination_cbr` from analysis.metrics;
  criterion relative difference <= 20 %, per cell, on the medians over the paired replays.
* Loss rates: in-range loss = 1 − PDR in both; the custom loss split (sensitivity / SINR / beacon) is reported, and for
  Veins the MAC's SNIR- and TX/RX-lost counts (all frames, descriptive).
* Matched cost (Spearman criterion): Algorithm 1 per (cell, scheme family) within each simulator on seeds 200–209 —
  q* = 0.95 x max over the 55 settings of mean RWCR, guard = min mean actionable miss + 0.05, cost at q* by the
  interpolated crossing (PolicyCurve.overhead_at); Spearman rho over the cell x family pairs where both simulators
  have a finite matched cost; 95 % CI and the bootstrap p for rho <= 0.6 by resampling seeds (10,000 replicates).
* Veins H1 (cross-fitted, D1): three folds 200–209 / 210–219 / 220–229; q*, m* and each family's selection on the
  other two folds; scored on the held-out fold; the bootstrap redoes the whole procedure; identical procedure for
  the custom simulator on SUMO.

**Results seen before this entry:** no H3 statistic; only the two port-validation replays of 2026-10-19
(results/revision5/veins/port_validation.md) and the two development flooding replays already disclosed in D1.

## D6 — H3 failed: executing the pre-registered fail branch (sub-epoch timing mode) (2026-10-31T18:59+05:30)

**Seen before this entry:** the full H3 result (results/revision5/analysis/h3.json, commit 0908711): Spearman rho 0.564
[-0.117, 0.791], p(rho <= 0.6) = 0.776; PDR-vs-distance and busy-ratio criteria fail (Veins delivers more at short range,
the custom busy ratio is 33–89 % higher); median single-hop delay and the cross-fitted H1 verdict agree. Pre-registered
rule: "Fail: add a sub-epoch timing mode (13 µs slots, AIFS/backoff; default unchanged), calibrate it only on the
validation metrics, freeze it, and re-run." The user chose to follow it (2026-10-31).

**The mode** (`mac.timing: sub_epoch`; default `epoch`, which stays bit-identical — verified by the rule-7 gate):
within an epoch with hazard transmissions, every active vehicle has a background beacon with probability
rate_DCC x 0.1 s at a uniform start on the 13 µs grid; each hazard frame starts at AIFS + backoff x 13 µs and defers
past any beacon its transmitter senses (mean received power >= the carrier-sense threshold); a beacon collides with a
hazard frame only if their airtimes overlap and its sender could not sense the hazard frame (hidden, or same slot);
reception is the existing SINR/capture test with all overlapping hazard frames and beacons as interference (replacing
the distance-blind Bernoulli survival exp(-lambda T_vuln)); the reported busy ratio is each vehicle's expected busy
fraction from frames it senses above the carrier-sense threshold (beacons at their DCC rates + hazard airtime),
averaged over active vehicles. Hazard decisions remain on the 100 ms epoch.

**Calibration (fixed now, before any calibration number exists):**
* free parameters, mode only: carrier-sense threshold {−95, −92, −88, −85} dBm x capture SINR threshold {4, 6, 8, 10} dB
  (16 points);
* calibration data: Veins replays of the **audit seeds 210–219** (not the H3 validation seeds 200–209), 5 settings
  (flooding; p_persistence_03 p=0.3; slotted_1p n=8; weighted_p max_p=0.5; counter_based C=3) x 4 headline cells
  x 10 seeds = 200 paired runs per point;
* loss = max over cells of the largest |ΔPDR| over 50 m bins up to the nominal range + mean over cells of the relative
  |Δ busy ratio|; ties -> the point closest to the current phy.yaml values (SINR 8 dB, carrier sense −92 dBm);
* the minimum is frozen in configs (mode on only for the H3 re-run), recorded with its loss, and never revisited.

**Re-run:** custom network-layer runs (seeds 200–209, all 55 settings) and the custom metric rows for the cross-fitted
audit (seeds 200–229) in the frozen mode; H3 recomputed with the same code and definitions (D3). If H3 still fails,
the pre-registered consequence applies: the classical audit computed in Veins becomes primary and custom-simulator
claims are restricted accordingly.

**D6 implementation note (2026-10-31T19:40+05:30, written before any calibration run or number exists).** (a) Busy ratio of the
mode, as implemented: beacon part = mean over active vehicles of min(1, sum over the vehicles they sense of
rate_DCC x T_beacon), sampled every 10th epoch (trace-only; memoised per trace); hazard part = per epoch, mean over
active vehicles of min(1, sensed hazard frames (own excluded) x T_frame / 0.1 s), averaged over all epochs with active
vehicles (0 where none is sent); reported busy ratio = their sum, capped at 1 (metrics beacon_cbr / dissemination_cbr
carry the two parts, so the D3 sum is unchanged in form). (b) "relative |Δ busy ratio|" uses the D3 denominator
(the custom median). (c) "closest to the current values" = Euclidean distance in (dBm, dB). (d) PDR bins empty in one
simulator are skipped in the loss and listed. (e) Veins' own CCA threshold is −65 dBm (ccaThreshold default; the ini
sets minPowerLevel −85 dBm) — outside the pre-declared grid; the grid is not changed. Driver:
experiments/revision5/veins/d6_calibrate.py (run / score / rerun); frozen point -> configs/phy_patch_d6_frozen.json.

**D6 outcome (2026-11-01T11:40+05:30).** Calibration (results/revision5/analysis/d6_calibration.json): minimum loss 0.652 at
carrier sense −85 dBm, capture SINR 10 dB (a corner of the pre-declared grid; loss at the phy.yaml values −92 dBm / 8 dB
was 0.698); frozen in configs/phy_patch_d6_frozen.json. Frozen-mode re-run: 6,600 runs, 0 failures. H3 recomputed with
the same code and definitions (h3_subepoch.json): Spearman ρ 0.842, 95 % CI [0.373, 0.933], p(ρ ≤ 0.6) = 0.169 (was
0.564 [−0.117, 0.791], p 0.776); PDR-vs-distance fails in every cell (largest bin gap 0.19–0.33, criterion 0.05); busy
ratio passes in rural d=2 (1.4 %) and d=20 (10 %) but fails in rural d=80 (44 %) and urban NLOS d=20 (49 %); median
single-hop delay and the cross-fitted H1 verdict (supported in both) agree. **H3 fails.** Holm with the post-D6 H3 p:
H1 5.2e-223 (reject), H2 0.127, H3 0.169. Pre-registered consequence, applied: the classical audit computed in Veins
(cross-fitted H1) is primary; custom-simulator claims are restricted to relative comparisons inside the custom
simulator (scheme and policy rankings, selection and qualification), and no absolute packet-level number from the
custom simulator (PDR, busy ratio, loss rates) is presented as validated. The policy (H2 and all secondary policy
results) exists only in the custom simulator and is reported with that restriction. Both H3 results (epoch and
sub-epoch) are reported; the epoch-mode one is the pre-registered primary test.

## D7 — ledger: columns where the claimed comparison is undefined (2026-11-01T22:07+05:30)

**Not specified before:** the ledger addendum (ANALYSIS_SPEC, 2026-10-31T12:05+05:30) says "a column whose rows are
incomplete reads 'not tested', never 'holds'", but not how to score a column whose rows are complete while the claimed
comparison is undefined there (the policy qualifies in no SUMO headline cell, so "policy vs adaptive classical / best
rule" has no cell to compare).

**Decided:** such a column counts as **not holding** (a "stable" verdict needs positive evidence in every applicable
column). The H1 Cal. and SUMO columns, and the per-cell-winner columns, are point estimates recomputed from the committed
rows with the H1 selection code (no bootstrap), and are labelled as such.

**Seen before this entry:** all Step 1–2 secondary results (secondary.json, B = 2,000), including the SUMO policy
result (qualifies in 0 of 4 headline cells) and the stress-test results; the first ledger draft. **Effect on verdicts:
none.** With the alternative reading (undefined = not applicable): H2 is "not supported" from its Set. column alone;
"policy costlier than the best rule" stays "sensitive" because its sensitivity-grid column is not tested; the
qualification claim fails on SUMO (the policy reaches no cell, so the comparison is defined and false).

**Also recorded:** the secondary bootstrap first ran with the script default B = 2,000; it was re-run with the
pre-registered 10,000 replicates (ANALYSIS_SPEC "Bootstrap: 10,000 replicates") and all downstream outputs regenerated.
Point estimates do not depend on B. For the Step 5 clean-clone check, the provenance stamp of every output carries the
generation time and commit, so a regenerated file cannot be byte-identical as a whole; the check compares (a) the full
files and (b) the outputs with the `provenance` block removed, and reports both.
