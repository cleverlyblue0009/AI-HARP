# Revision 6 — pre-registration addendum: rules and the learned policy in Veins; more diagnostic seeds

Written 2026-11-02T12:05+05:30 on branch `revision-6` (from `origin/revision-5`, 59baa42), **before any revision-6 replay,
port-validation replay or training run**. Push commit of this file: *recorded in the follow-up commit*.

**Seen before writing:** every revision-5 result (`results/revision5/`), including the custom-simulator rule and policy
results on SUMO mobility (EVAL seeds 200–229), which form the custom side of V4. **Not seen (does not exist yet):** any
Veins replay of a rule or of the policy, any decision-server output, any run of the new training seeds.

Interpreter note: the pinned environment is the venv at `D:\aiharp-gpu` (NumPy 1.26.4, torch 2.2.2); its executable is
`D:\aiharp-gpu\Scripts\python.exe` (the `D:\aiharp-gpu\python.exe` path named in the task does not exist). Every job
logs `sys.executable` and the NumPy version.

## 1. What is run in Veins, and how

* **Unchanged:** the Veins radio, MAC, analogue models, `omnetpp.ini`, beacon/DCC application traffic, scenario bundles and
  the revision-5 app binary and its sources (`experiments/revision5/veins/aiharp/`). No revision-5 result is touched.
* **Revision-6 app:** a copy of the revision-5 app (`experiments/revision6/veins/aiharp/`), built into a separate
  directory, adding three policy kinds (`rar`, `scf_slotted`, `ai_harp`). It runs with the identical `omnetpp.ini`,
  `config.xml` and bundles. **Gate:** before any rule or policy replay, one classical replay (rural d=20, seed 200,
  slotted_1p n=8) with the revision-6 binary must produce an event log byte-identical to the same replay with the
  revision-5 binary. If it does not, nothing is run.
* **Causal relevance ρ_c and ETA** (`hazard/risk_field.py`, both geometries: highway for rural, grid for urban NLOS):
  computed in the app from the holder's own position and velocity (Veins mobility, converted to trace coordinates),
  its carriageway sign (the trace's static per-vehicle sign, exported in the bundle), the hazard's parameters (span,
  affected direction, severity law, onset; they ride in the hazard message, so a holder knows them) and the epoch time.
  If any of these cannot be obtained, the rule track stops and is reported.
* **Beacon-derived neighbour table** (as for the classical ports): latest decoded beacon per sender, 1.05 s lifetime,
  extrapolated to now, restricted to the nominal range. For the policy, each beacon additionally carries its sender's
  carriageway sign (an application field; the beacon's configured length on air is unchanged); a holder computes a
  neighbour's relevance and ETA from the neighbour's beacon state and the hazard it holds, with the same functions.
  `neighbour_informed` = the holder has decoded a hazard frame from that neighbour. These are the only definitions
  used; they are not tuned.
* **Policy bridge:** the app sends each decision request (the holder's DecisionContext: own state + the beacon-derived
  neighbour arrays) over TCP to a decision server running the unchanged agent code in the pinned interpreter, one
  server per worker. The server builds the decision graph with the existing graph code, the frozen normalisation
  statistics and the scenario's PHY object exactly as the custom evaluation does, applies the confidence gate
  (τ = 0.5) and the suppression bias, samples the action from an RNG stream named by (replay seed, vehicle index,
  epoch), and returns action, delay and relay target. The call is synchronous (simulation time does not advance); the
  action is applied at the same epoch boundary as in the custom engine.
* **Precision note (stated in advance):** the custom trace stores positions as printed by SUMO's FCD output and
  velocities as float32, while Veins sees TraCI doubles. The port checks therefore require agreement to 1e-9 **on
  identical inputs** (the app logs its inputs; the Python reference recomputes ρ_c and the decisions from them), and
  report, as measured, the difference against the custom simulator's trace-based values.

## 2. Port validation (all must pass before any reported replay)

* **Rules:** one RAR setting and one SCF setting on seed 200 (rural d=20), identical SUMO trace in both simulators:
  ρ_c within 1e-9 on identical inputs; transmit decisions identical under identical inputs (same ρ_c, neighbour
  count, sender distance and random draw); written to `results/revision6/veins/port_validation_rules.md`.
* **Policy:** (1) unit check — 1,000 decision contexts recorded in the custom simulator, sent through the server:
  action distributions within 1e-6 of the in-process policy; (2) trace check — `ref_s0` at its revision-5 SEL bias in
  rural d=20 (bias +1), seed 200, every request logged and re-sent offline, every action reproduced; (3) sanity —
  RWCR, cost and action mix of that replay next to the custom simulator's (no tuning in response); written to
  `results/revision6/veins/port_validation_policy.md`. If the bridge is not validated within two working days of the
  first bridge replay, the policy track stops, the reason is written, and the rules continue alone.

## 3. Design

* **Cells:** rural highway d=2, d=20, d=80; urban NLOS d=20.
* **Seeds:** Tier A (required) 200–209 (paired with the classical Veins replays); Tier B (optional) 210–229.
* **Rules:** all 39 settings (RAR 27, SCF 12) as in `results/revision5/protocol.json`. Tier A = 1,560 replays.
* **Policy:** full-model training seeds 0–4 (Tier A-P), then 5–9 (Tier A-P2); every suppression bias in {−2, −1, 0, 1,
  2, 3}; τ = 0.5; sampled actions. Tier A-P = 1,200 replays, Tier A-P2 = 1,200.
* **Order:** gate and validations → Tier A rules → Tier A-P → Tier A-P2 → Tier B (if time allows).
* **Cut rules:** if Tier A rules cannot finish within 72 wall-hours of their start, the RAR settings with β1 = 0.5 are
  dropped; if Tier A-P cannot finish within 96 wall-hours of its start, its biases are cut to {−1, 0, 1}. Whether a cut
  applies is decided from elapsed time and throughput only, never from results.

## 4. Analysis (Veins)

* **Thresholds:** per cell, inside Veins, on the replay seeds analysed: q* = 0.95 × max mean RWCR over the 55 classical
  settings; m* = min mean actionable miss over the 55 classical settings + 0.05 (D3 definitions).
* **Matched cost (Algorithm 1, D3):** per (cell, family) the cost at q* by the interpolated crossing
  (`PolicyCurve.overhead_at`) over the family's settings with mean miss ≤ m*. Families: the 7 classical families; RAR
  (27 settings); SCF (12 settings); each policy training seed (its 6 biases). Best classical = the classical family with
  the lowest finite matched cost; best rule = the lower of RAR and SCF. Per rule setting, its mean cost, RWCR, miss and
  whether it meets q* and m* are also reported.
* **Policy per cell:** a training seed "has a pick" if its matched cost is finite; the cell aggregate is defined when at
  least half of the training seeds have a pick and is the mean of their matched costs (revision-5 rule, applied to
  matched costs; the mean RWCR of the picks meets q* and the mean miss meets m* by construction of Algorithm 1, and
  this is checked).
* **Tier A comparison:** every method's matched cost on the same 10 seeds; this is an **EVAL-hindsight comparison**
  (thresholds, crossings and comparison on the same seeds) and is labelled as such everywhere.
* **Tier B (only if 200–229 complete for every method):** 3-fold cross-fitted version (folds 200–209 / 210–219 /
  220–229) exactly as `crossfit_H1` (D1): thresholds and each method's setting selected on two folds, scored on the
  third; for every method.
* **Hypotheses:**
  * **V1:** the best rule's matched cost is lower than the best classical family's in at least 2 of the headline cells
    where both qualify. Statistic: the number of such cells; p = bootstrap share of replicates with fewer than 2.
  * **V2:** the policy's matched cost is lower than the best classical family's. Statistic: pooled relative difference,
    mean over cells where both qualify of (policy / best classical − 1); two-sided p = 2·min(P(T ≤ 0), P(T ≥ 0)) as in
    revision-5 H2; superiority only if the 95 % CI lies entirely below 0.
  * **V3:** the policy's matched cost is lower than the best rule's; same test as V2.
  * **V4 (descriptive):** Spearman ρ with a bootstrap 95 % CI between the Veins and the custom-on-SUMO (epoch mode,
    the pre-registered primary of D6) matched costs over every (method, cell) pair finite in both, on seeds 200–209;
    custom side from the committed revision-5 rows (`classical__EVAL__headline__default__sumo.csv`,
    `rule__EVAL__headline__default__sumo.csv`, `agent__EVAL__headline__default__sumo.csv`).
* **Bootstrap:** B = 10,000, RNG seed 20261102; hierarchical: replay seeds (and, for the policy, training seeds)
  resampled; q*, m*, every crossing and every "best" recomputed in each replicate; undefined replicates counted and
  excluded. Holm over V1–V3 (α = 0.05). Qualification counts with Wilson 95 % intervals.
* **Restriction carried over (D6):** the custom simulator failed H3; the custom side of V4 is reported as such, and
  the Veins results are the primary evidence for the rules and the policy.

## 5. D8 — variant seed extension (custom simulator)

Training seeds 5–9 for GCN, MLP, no-relevance, rung B and rung C (priority in that order), unchanged configuration,
1,000 updates; evaluated exactly like the revision-5 variants (biases −2..3, SEL 400–419, EVAL 200–229, headline
cells, sampled actions, τ = 0.5; rung C on its privileged path). `T-ablation` and `T-ladder` are recomputed with 10
seeds per variant (mean ± SD, bootstrap CIs) and reported next to the 5-seed versions. **No result of these runs has
been seen; none exists when this is written.** Runs only on cores not needed by the Veins replays.
