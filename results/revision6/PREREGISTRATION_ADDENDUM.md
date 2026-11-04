# Revision 6 — pre-registration addendum: rules and the learned policy in Veins; more diagnostic seeds

Written 2026-11-02T12:05+05:30 on branch `revision-6` (from `origin/revision-5`, 59baa42), **before any revision-6 replay,
port-validation replay or training run**. Push commit of this file: `b695924`.

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

## D9 — beacon velocity in the revision-6 app (2026-11-02T12:34+05:30, before any reported replay)

**Found:** checking the policy request's neighbour arrays against the custom trace (the validation replay ref_s0, bias
+1, rural d=20, seed 200) showed every neighbour velocity equal to zero (median |Δvx| 19.4 m/s = the vehicles' speed;
own state, positions and headings agreed to the expected one-epoch lag). Cause: `omnetpp.ini` sets
`veinsmobility.setHostSpeed = false`, so Veins never sets the mobility speed, `getCurrentSpeed()` is (0, 0), and
DemoBaseApplLayer copies that into every beacon's `senderSpeed`. Consequences for the policy: neighbour velocity,
relevance and ETA features wrong; for every port using the beacon table, entries are not extrapolated to the
decision time. **The revision-5 classical ports that use the beacon table (greedy_farthest, DV-CAST) ran with the
same zero beacon velocities** (DV-CAST's opposing-traffic test saw no moving neighbour); this is reported in the
revision-6 report; no revision-5 result is changed.

**Fix (app content only; the configuration is unchanged, so the classical replays' configuration is untouched):**
for the policy kinds rar, scf_slotted and ai_harp, each beacon's `senderSpeed` is set to the sender's velocity from
its TraCI speed and heading (`heading.toCoord() × speed`), as a CAM/BSM carries it. Classical kinds keep the
revision-5 behaviour exactly (the identity gate is re-run on the rebuilt binary). The rule and policy validations are
re-run on the rebuilt binary; only those runs count.

**Seen before this entry:** the first validation replays (RAR, SCF, policy on seed 200, rural d=20) and the sanity
metrics of the policy replay (Veins RWCR 0.461, cost 1.47; custom 0.836, 0.693). The fix was identified from the
input comparison (neighbour arrays vs trace geometry), not from those outcomes, and nothing else is changed.

## D10 — scope cuts on cost and time (2026-11-02T21:50+05:30, decided by the author)

**Numbering:** the author's instruction called this D9; D9 was already taken by the beacon-velocity fix above, so it is
recorded as D10.

**Decided:**
1. **Rules, Tier A:** unchanged, all four cells. The section 3 cut (RAR β1 = 0.5) applies only if the 72-wall-hour limit
   is actually missed, judged on elapsed time and throughput. Clock start: first replay, 2026-11-02T12:43+05:30; limit
   2026-11-05T12:43+05:30.
2. **Policy:** training seeds 0–4 only (Tier A-P), in rural d=2, rural d=20 and urban NLOS d=20. **Rural d=80 is not run
   for the policy** (900 replays instead of 1,200). **Tier A-P2 (seeds 5–9, 1,200 replays) is cancelled.**
3. **Step 5 / D8 (variant seeds 5–9) is stopped.** No partially trained run is evaluated. `T-ablation10`, `T-ladder10`
   and `variants10.json` are therefore not produced as planned. What exists is committed in
   `results/revision6/checkpoints_unused/`, labelled and unused: gcn_s5 (finished, 1,000 updates, never evaluated),
   gcn_s6 (86 updates), gcn_s7 (restart, stopped at 334 updates), and the quarantined gcn_s7 whose updates from 492 on
   used SUMO mobility by mistake (watchdog bug, fixed in f64380c).
4. **Freed cores go to the Veins replays.** No VM.

**Reason:** cost and time only. Projected Veins wall time was about 9 days on a single laptop (6 cores, 12 threads), with
rural d=80 replays at about 1 h each. That figure was a projection: the revision-5 d=80 median of 1,712 s, scaled by the
2.2× ratio of this laptop's d=20 median to revision 5's. No d=80 replay had finished when the projection was made.

**What had and had not been seen when this was decided:**
* **Not seen:** any Tier A rule or policy result. No Tier A replay has been scored; `veins6_score` has not been run on
  `cache/veins_runs/r6/`, and no policy Tier A replay exists. Only replay status and wall times were read.
* **Seen (the instruction asked for a statement that no Veins result had been viewed; that is not exact, so this is
  stated instead):** the validation sanity replays, which were reported to the author earlier the same day. Seed 200,
  rural d=20, final binary: policy (ref_s0, bias +1) RWCR 0.780, cost 1.262; RAR (1.0, 0.5, 0.0) RWCR 0.587, cost 0.181;
  SCF-slotted (8, 5) RWCR 0.868, cost 0.224. Also the pre-D9 policy replay (D9 above). The assistant described the
  policy sanity replay to the author as "a warning". No rural d=80 result of any rule or of the policy has been seen.

**Consequences for the analysis:** the policy has no rural d=80 Veins result, so any comparison involving the policy
covers three cells and five training seeds; V1–V4 and the Veins tables say so wherever they apply. Classical and rule
methods keep all four cells. Rural d=80 is the densest cell, so omitting it for the policy limits what can be claimed
about the policy under high density in Veins; this is stated in the report.

## D11 — analysis code before scoring (2026-11-05T00:00+05:30)

**Changed in `experiments/revision6/analysis/veins_methods.py`, before any Tier A replay was scored:**
1. **D10 scope:** the default runs are ref_s0–ref_s4, and the policy's qualification counts are out of three cells. V2,
   V3 and the policy's V4 pairs already leave out cells without a policy result, so they cover three cells; V1, the
   classical methods and the rules keep four.
2. **Bug fix:** in the bootstrap, resampled training runs carry labels such as `ref_s0#3`, and the matched-cost lookup
   compared those labels with the plain run names. Every replicate would have had no policy cost, so V2 and V3 would have
   had no intervals or p-values. The lookup now strips the suffix. No definition, threshold, test or bootstrap setting
   changes. Found by reading the code and confirmed on made-up numbers.

**What had and had not been seen:**
* **Not seen:** any Tier A score. `veins6_score` has not been run on `cache/veins_runs/r6/`; `veins6_rows.jsonl` and
  `veins_methods.json` do not exist.
* **Seen:** replay status and wall times; the simulated end time (`covered_until_s`) of the rural d=80 SCF-slotted
  replays, read to explain why three were running long; and the last two lines of one unfinished event log (seed 205,
  SCF-slotted 50 slots, timeout 2 s), which are not a metric.
