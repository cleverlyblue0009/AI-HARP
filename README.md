# AI-HARP

Simulation framework for hazard-message dissemination in sparse, NLOS-degraded
vehicular networks.

| Phase | Scope | State |
|---|---|---|
| 1 | Mobility: SUMO scenarios, FCD parsing, cached `.npz` traces, pure-Python fallback | done (SUMO path never executed — see below) |
| 2 | Network simulator: 802.11p PHY, CSMA/CA MAC, dissemination engine | done |
| 3 | Hazard model + causal risk field + oracle risk field | done |
| 4 | Seven baseline policies (9 registered variants) | done |
| 5 | GATv2 agent, PPO / Dueling DQN, confidence gate, curriculum training | code done; **not yet trained to convergence** |
| 6 | Metrics (PDR, RWCR, TIR, overhead, deadlines, channel load) | done |
| 7 | Paired sweeps, Wilcoxon + Holm + effect sizes | runner + stats done; **ablations not run** |
| 7b | Simulator validation | SUMO installed and running; backend comparison done; **published-curve reproduction not started** |
| 8 | IEEE figures, LaTeX tables, `reproduce.sh` | pipeline done; agent-dependent figures pending |

**Read `ENVIRONMENT.md` first** — the Python environment lives on `D:`, not `C:`.

```bash
D:/aiharp-env/python.exe -m pytest tests/ -q          # 325 tests, ~17 s
D:/aiharp-env/python.exe -m experiments.run_sim --smoke
./reproduce.sh --smoke                                 # whole pipeline
```

---

## What the paper claims, and where each claim is earned

### The headline is overhead at matched coverage, not RWCR

RWCR **saturates**. Measured against the oracle at-risk set, every baseline
lands within a few points of every other at most densities. A result of the
form "0.98 versus 0.976" is not a contribution.

What does not saturate is *cost at matched coverage*. Each scheme's suppression
knob is swept to trace an operating curve in (RWCR, transmissions per at-risk
vehicle informed), and the question becomes who reaches a given coverage most
cheaply. `analysis/pareto.py` computes those curves;
`analysis/comparator.py` turns them into the two reference points.

### "Versus flooding" is dead

Flooding needs 2.11 transmissions per at-risk vehicle informed where the
deployable cluster needs 0.41. Beating it 5× restates that nobody deploys
flooding. Four baselines already sit within 6% of each other, and **that
cluster is the bar**.

The comparator reports two things instead:

| | question it answers |
|---|---|
| **(a) per-cell oracle-best** | the best baseline in each cell *with its knob tuned for that cell*. Not deployable — that is why it is the right upper bound. The agent's **regret** against it measures how much of hindsight tuning a single learned policy recovers. |
| **(b) best single fixed baseline** | one setting held constant across every cell — what an engineer would ship. The agent's **margin** over it measures whether the learned policy earns its complexity. |

Measured headroom (10 seeds, matched at 95% of each cell's own ceiling):

| cell | ceiling | oracle-best | fixed-best | penalty |
|---|---|---|---|---|
| rural d=2 | 0.689 | slotted_1p 1.78 | 1.79 | 1.00× |
| rural d=20 | 0.925 | dvcast 0.70 | 0.90 | 1.29× |
| rural d=80 | 0.915 | weighted_p 0.91 | 1.88 | 2.06× |
| urban_nlos d=20 | 0.980 | counter_based 0.47 | 0.62 | 1.33× |

**The best baseline is a different policy in every cell.** The best fixed
choice — `slotted_1p(n_slots=5)`, the only one that reaches the target
everywhere — costs 42% more than per-cell hindsight tuning on average and 106%
more at d=80. That gap is exactly what a learned policy has to earn.

On the **latency** axis the picture inverts: flooding is oracle-best in 3 of 4
cells. Reporting overhead alone would hide that, so both axes are primary.

### Matched quality is resolved per cell, not absolutely

Ceilings differ enormously by cell (0.689 sparse rural, 0.980 urban_nlos), so a
fixed absolute target is unreachable in three cells out of four and reports
`n/a` everywhere. The target is a fraction of each cell's own achievable
ceiling, computed excluding the policy being scored so a stronger agent cannot
raise its own bar. `--mode absolute` remains for comparability with prior work.

---

## The oracle/causal split (the most important design decision here)

Two relevance fields, with a hard boundary between them:

| | module | used by | sees |
|---|---|---|---|
| **causal** | `hazard/risk_field.py` | agent features, reward, all policies | only information available at time *t* |
| **oracle** | `hazard/oracle.py` | `analysis/metrics.py` only | realised trajectories — ground truth |

Why: on `urban_nlos` the causal field reported RWCR 0.291 while **100% of
vehicles were informed**, because it marks a vehicle at-risk only if its
*instantaneous* heading points at the hazard, and grid vehicles turn. RWCR was
measuring vehicle heading, not warning effectiveness.

The oracle's lookahead is derived, not chosen:
`H = eta_full + tau·ln(severity/threshold)`, capped by hazard lifetime — the
ETA at which the risk field's own kernel falls below its own at-risk threshold,
so both definitions ask the same question over the same window.

**`tests/test_oracle_isolation.py` enforces the boundary by parsing the import
graph.** `agents/`, `sim/` and `mobility/` may not import the oracle or mention
its symbols; `hazard/__init__` may not re-export it; `sim/engine.py` may not
mention ground truth. A comment cannot satisfy those tests.

### Risk estimation difficulty is itself a result

`estimation_agreement()` reports how well the causal field recovers the oracle:

| scenario | precision | recall | peak-relevance correlation |
|---|---|---|---|
| rural_highway | 0.78 | 1.00 | **0.81** |
| urban_nlos | 0.57 | 1.00 | **0.13** |

On a corridor heading determines destiny. In a grid it does not. That gap
quantifies the difficulty the learned policy is being asked to overcome, and
belongs in the paper *before* any policy result.

---

## Constants and their provenance

```bash
D:/aiharp-env/python.exe -m analysis.constants_table          # console
D:/aiharp-env/python.exe -m analysis.constants_table --latex  # paper table
```

Tags: `STD-COMPUTED` (computed from a standard's own equations and validated
against its published table) → `STD` → `DERIVED` → `MEAS` → `STD-UNVERIFIED` /
`MEAS-UNVERIFIED` (attributable but we could not open the primary document) →
`ASSUMED` → `INERT` (recorded but read by no computation).

Three things worth knowing:

**Rain and fog are computed, not recalled.** ITU-R P.838-3 and P.840-8 are
implemented from their governing equations and evaluated at 5.9 GHz. The
reconstructed P.838-3 closed form reproduces the Recommendation's own Table 5
to within 0.11% across 1–10 GHz, which is what confirms the coefficient signs
(the published PDF's text layer drops minus signs). Computing K_l rather than
recalling it **changed the fog coefficient by 14×**.

**There is no weather-channel claim.** Heavy rain attenuates a 562 m DSRC hop
by 0.04 dB; thick fog by 0.007 dB. The empirical excess-loss term that earlier
revisions carried was invented, no citable V2V measurement campaign exists at
this frequency, so it was deleted (`enable_empirical_excess_loss: false`).
**Weather in this project acts through traffic, not the radio** — drivers slow
and increase headway, which changes topology. Every weather figure caption must
say "traffic-mediated"; worst-case channel effect is a 0.24% range change.

**The path-loss breakpoint is derived**, `d_bp = 4·h_t·h_r/λ` = 177 m at 1.5 m
antennas, giving a 610 m nominal range. A single-slope model would predict
~2.1 km and delete the sparse regime this paper is about.

`cw_min` remains `[STD-UNVERIFIED]` and is **load-bearing** — it sets the
contention window that drives the whole collision model. IEEE 802.11-2020 is
paywalled and ETSI blocks automated download. Report it as a sensitivity across
access categories until verified.

---

## Scenarios

| name | what it is | role |
|---|---|---|
| `rural_highway` | 10 km two-lane corridor | primary; sparse regime below ~3 veh/km/lane |
| `urban_nlos` | 6×6 grid with building-blocked links | realistic fragmentation (NLOS range 101 m vs 480 m LOS) |
| `urban_grid` | same grid, no NLOS | well-connected control, and **held out** from training |

The brief's density sweep started at 5 veh/km/lane, but instrumenting the
policies' action counts showed DV-CAST's store-carry-forward branch firing
**0.0 times per run** at every density from 5 to 120: with a 610 m range on a
two-way corridor those are all *connected* networks. Densities 1, 2 and 3 are
prepended. At 120 veh/km/lane the corridor is **jammed** (mean speed ~2.5 m/s),
so every run carries a `regime` column (`free_flow`/`congested`/`jammed`)
derived from measured speed.

---

## The agent (Phase 5)

- `agents/graph.py` — per-holder decision graph, NumPy-only so it is testable
  without torch. Neighbour-to-neighbour edges are included deliberately: a star
  graph would make GATv2 degenerate to attention pooling and render the
  GAT-vs-MLP ablation vacuous. Normalisation statistics are **frozen** and
  schema-checked on load — per-batch normalisation would make a vehicle's
  features depend on its batch-mates, which a real OBU cannot reproduce.
  **The statistics must also be representative, and this was found broken.**
  The first fit kept the first 120 graphs of one rural run, so `rel_vy` and
  `heading_sin` had std exactly 0 (an east-west corridor), floored at 1e-6.
  Measured on an urban_nlos d=40 episode:

  | | max normalised input | value_loss, untrained head |
  |---|---|---|
  | before | 2.06×10⁷ (`rel_vy`) | 8.2×10⁸ |
  | after  | 6.04 | 1,060 |

  The fit now spans both training scenarios, densities 2–80 and two hold-out
  seeds, striding through each run; degenerate features are centred rather
  than scaled and listed; and stats carry provenance, so a smoke fit can no
  longer be silently reused by a full run. Residual: `message_age_s` still has
  a narrow fitted std (0.34 s) because decisions cluster early in a message's
  life, so late decisions normalise to several units.
- **Three action-semantics bugs made silence unlearnable, and were found only by
  measuring normalised cost.** run2 (40 updates, stopped at 13) looked healthy
  on entropy and confidence, but transmissions per informed vehicle sat at
  0.963–0.972 on every update — flooding-level redundancy — and the network had
  cut suppress from ~15% of decisions to 0.35%. Raw per-episode transmission
  counts hid this, because each update samples a random mix of densities;
  **never read learning from raw transmission counts**. Replaying an untrained
  network (the policy PPO starts from) located the cause:
  1. every one of a vehicle's decisions received its full reward, so GAE
     over-credited its earliest decisions;
  2. a suppress was not final — 90% were followed by a re-ask on a later
     duplicate and a transmission, so they carried the transmitter's reward;
  3. the agent's `defer` never cancelled on duplicates, unlike every slotted
     baseline's, so it was "broadcast later" and strictly dominated.

  | advantage of suppress minus transmit | |
  |---|---|
  | as built | −8.02 (PPO pushed suppress down) |
  | reward-placement fix only | +1.66 |
  | all three fixes | **+8.45** (ideal with no re-ask: +11.58) |

  With all three fixes, even an untrained network's transmissions per informed
  vehicle fell from 0.971 to 0.782. **Training runs 1 and 2 predate these
  fixes and are invalid**; run1 additionally had the confidence gate active in
  training (fallback rate 1.0) and run2 dropout 0.1.
- **The reward is broken: it prefers silence. The claimed calibration is
  retracted.** With the semantics fixed, training run3 learned quickly —
  transmissions per informed vehicle fell 0.76 → 0.14 in 10 updates, below
  `slotted_1p`. On evaluation seeds it had learned near-total silence: RWCR
  **0.034**, 23 of ~800 vehicles informed, one transmission, actionable miss
  rate 0.977. Scoring fixed policies with the reward shows why:

  | policy (rural d=40, eval seed 0) | RWCR | tx | total reward |
  |---|---|---|---|
  | always-suppress | 0.020 | 1 | **+1.5** |
  | slotted_1p | 0.954 | 175 | −829.3 |
  | weighted_p | 0.949 | 309 | −2,613.7 |
  | flooding | 0.954 | 755 | −9,301.2 |

  The coverage and miss terms are divided by the at-risk count and added to
  every vehicle, so total failure versus near-perfect coverage shifts the
  reward by 0.016 per vehicle, against ≥ 1.33 per transmission. The
  "calibration" balanced one transmission against the relevance it directly
  informs, and never checked that the reward ranks whole policies correctly.
  Lesson recorded for every future reward: **score a set of fixed reference
  policies under it before training**, and require the ranking to be sane.
  No training run to date has a valid objective.
- **The replacement is a constrained objective, and it passed that check before
  any training.** Minimise transmissions per at-risk vehicle subject to causal
  coverage ≥ 95% of each training cell's measured ceiling (the better of
  flooding and DV-CAST), via a Lagrange multiplier λ. Per-vehicle credit is a
  Shapley split of each warned vehicle's relevance along its ancestor path in
  the `informed_by` tree, so relays have a stake in coverage they enabled
  downstream. Code: `agents/constrained_reward.py`; gate:
  `python -m analysis.reward_check`.

  | regime (eval seed 0) | λ at which silence stops being optimal | verdict |
  |---|---|---|
  | rural d=40 | 0.679 (0.535 on the pre-`5f02df7` engine) | SANE |
  | rural d=2 | 0.731 (0.915) | SANE |
  | urban_nlos d=20 | 1.207 (1.217) | SANE |

  Re-run after the busy-medium fix; in every regime an efficient scheme
  (`slotted_1p`, DV-CAST or `weighted_p`) outranks flooding just above
  break-even.

  Beyond break-even, `slotted_1p` and DV-CAST outrank flooding, so it does not
  trade silence for flooding. Two caveats:
  1. **In grid cells the constraint is enforced on a weak proxy.** Every
     urban_nlos policy scores causal coverage 0.28–0.29 against ~0.98 oracle
     RWCR, because the causal grid estimator correlates only 0.13 with ground
     truth. Per-cell targets keep the constraint feasible, but there the agent
     learns which vehicles the proxy flags, not which are truly at risk.
     Training may not use the oracle, so this is the honest limit of the
     causal-only rule; evaluation measures the gap.
  2. **λ may oscillate.** Dense cells exceed their targets easily, so dual
     ascent pulls λ toward the ~0.5 break-even where silence becomes
     attractive again. λ is logged every update so this is visible.
- `agents/gat_drl.py` — 3×GATv2 with edge features. **The final layer's
  attention on `neighbour → holder` edges *is* the relay ranking**;
  `relay_top_k` designates the k-th most attended neighbour, so the heatmap
  figure shows the quantity that drove the decision.
- `agents/confidence.py` — the gate. Ensemble disagreement uses the **maximum**
  per-action spread, not the mean: with nine actions and an ensemble split
  between two of them, the mean reports confidence 0.78 for maximal
  disagreement. `tau = 0` is exactly the gate-off ablation.
- `agents/train.py` — PPO, density curriculum, TensorBoard, checkpoints. GAE
  runs along **each vehicle's own decision sequence**; the flat transition list
  is not one trajectory.

### Superseded: the weighted-sum reward "calibration"

> **Retracted — kept only as a record of what was wrong.** The derivation below
> balances one transmission against the relevance it directly informs. It never
> checked how the reward ranks whole policies, and under it always-suppress
> outranks every scheme that informs anyone (see "The reward is broken" above).
> Training uses the constrained objective; `configs/agent.yaml` marks the old
> `reward:` block as known broken. The collision-attribution fix at the end of
> this section is still in force.

A transmission is reward-neutral when its total price equals
`mean_relevance / target_cost`. The measured front fixes that: 0.41 for the
deployable cluster, 2.11 for flooding, mean at-risk relevance ~0.75, so the
total price is 1.83 and anything below 0.36 makes **flooding reward-optimal**.

The collision term is part of that price and must enter the calibration:

```
w2 = mean_relevance/target_cost − w3·E[collisions caused]
   = 1.83 − 0.25×2.0 = 1.33
```

Excluding it (w3 = 0.5, collisions un-attributed) made the effective price
10.03 — 5.5× too high — putting the break-even at 0.075 against a 0.41 target.
The agent would have learned near-silence and it would have looked like a
finding. Collisions are now attributed **per transmitter** by the engine
(`RunResult.collisions_caused`), summing exactly to `n_fail_sinr`.

---

## Reproducing

```bash
./reproduce.sh --smoke      # minutes
./reproduce.sh              # hours
./reproduce.sh --no-train   # reuse the committed checkpoint
```

Every run is seed-controlled: one master seed derives independent named RNG
streams, so seed *k* gives every policy identical mobility, hazard placement,
shadowing and fading. That is what makes the Wilcoxon signed-rank test
legitimate.

Every results row carries a `config_hash` **and** a `metrics_version`.
`metrics_version` is part of the de-duplication key, not a label: the same
config under a changed metric definition is a different result. 360 pre-oracle
rows are quarantined in `results/archive/` with a README explaining why they
must not be plotted alongside current ones.

`analysis/report.py` builds figures and tables from committed results only —
it never re-runs the simulator, and it **states what it could not build**
rather than omitting it silently.

### Figures

Generated by `analysis/figures.py`; 300 dpi, serif, vector PDF + PNG,
single-column (3.5 in) and double-column (7.16 in).

The categorical palette is Okabe-Ito, validated with a checker rather than by
eye. **Panels are capped at four series**: seven simultaneous series cannot be
coloured legally (a seventh hue put indigo against blue at ΔE 11.7 for normal
vision, below the 15 floor), so figures facet by policy family instead of
cycling hues. Cycling is what silently gave `flooding` and `p_persistence_03`
the same orange square in the first draft. `style_for()` now raises rather than
cycles. Flooding is drawn as a neutral reference mark, not a categorical
series — it has no knob, so its curve is a single point.

Every series carries a distinct marker *and* line style. That is not
decoration: it is what makes the palette legal at ΔE 7.6 all-pairs, and what
keeps the figures readable in greyscale, since Okabe-Ito sits in a narrow
lightness band.

---

## Known limitations

These are the paper's limitations section, pre-written. They are the most
valuable thing in this repository.

- **Backend comparison (10 seeds, fixed engine): orderings hold; absolute
  numbers are backend-dependent.** Eclipse SUMO 1.19.0 (see ENVIRONMENT.md),
  rural d=20, seeds 0–9, run at `df62b5d` after the busy-medium fix
  (`results/backend_validation_rural_d20.txt`):

  | metric | fallback → SUMO (all four policies) | ordering |
  |---|---|---|
  | RWCR | 0.921–0.924 → 0.862–0.866 (−0.06) | reshuffled within noise (movers span 0.002 vs seed-std 0.023) |
  | median TIR | +0.07 to +0.18 s under SUMO | within noise (only slotted/DV-CAST swap, 0.005 s apart; greedy last in both) |
  | tx per at-risk informed | 30–40% lower under SUMO | **preserved**, Spearman +1.0 |
  | PDR | −0.002 to −0.020 | **preserved**, Spearman +1.0 (spread 0.226 vs seed-std 0.017) |

  The PDR question left open by the 3-seed run is resolved: preserved. RWCR
  and TIR cannot rank these four schemes at this cell under *either* backend,
  so no ranking claim may rest on them here. Absolute levels move materially
  (SUMO's lane-changing traffic is less platooned than the fallback's), so
  headline absolute numbers must state their backend. An earlier 3-seed
  version of this entry, on the pre-fix engine, reported RWCR −0.16 and an
  unresolved PDR verdict; it is superseded. That version had also overstated
  flips by judging separability from all policies' spread (flooding's outlier
  cost made reshuffles among near-identical schemes look real); the rule
  counts only the policies that moved, and calls a flip real only if their
  gap exceeds seed noise under both backends.
  A second bug surfaced here: `tx_per_at_risk_informed` was missing from
  `METRIC_DIRECTION`, so cost was ranked higher-is-better and flooding came out
  "best on cost". Ordering *comparisons* were unaffected (both backends were
  reversed equally), but `analysis.stats.strongest_baseline` would have chosen
  flooding as the reference baseline for cost. Fixed and pinned by tests.
- **The fallback backend is still what most committed results used.** It is a
  real Krauss microscopic model, but it has no lane changing (hence no
  overtaking), no OSM geometry and no junction gap acceptance. Every trace is
  stamped `backend=fallback`, and the headline numbers should be regenerated
  under SUMO before submission.
- **No published curve has been reproduced.** The network layer is custom
  Python rather than NS-3 or Veins. Installing SUMO addresses the mobility half
  of the reviewer risk; the PHY/MAC half is still unvalidated against an
  external implementation.
- **The SUMO network is synthetic, not an OSM extract.** `netconvert` builds a
  straight corridor / `netgenerate` grid. Point
  `configs/scenario_rural.yaml -> sumo.osm_extract` at a real extract before
  submission.
- **The agent is not trained to convergence.** The pipeline runs end to end;
  no performance claim is supported yet. Training run4 (first run on the
  constrained objective) is a 40-update pilot against 2,000 configured
  updates. Its first attempt crashed at update 14 on a trace-cache file
  truncated when run3 was stopped mid-save; cache writes are now atomic and
  unreadable entries are regenerated. The restart reproduced the crashed
  attempt's per-update history exactly through update 13, so training is
  deterministic under fixed seeds on this machine (CPU, torch 2.2.2).
  **At 40 updates the constraint is not yet satisfied on training cells.**
  From `checkpoints/run4/history.jsonl`:

  | curriculum phase | updates | λ | coverage / target (mean) | updates meeting target | cost per at-risk vehicle |
  |---|---|---|---|---|---|
  | dense (40, 80) | 1–10 | 2.00 → 2.85 | 0.548 / 0.579 | 5 / 10 | 1.022 |
  | mid (10–40) | 11–20 | 2.85 → 4.25 | 0.570 / 0.622 | 3 / 10 | 0.423 |
  | sparse mix (1–80) | 21–40 | 4.25 → 5.77 | 0.512 / 0.568 | 4 / 20 | 0.508 |

  λ rose every phase and never settled (max 5.81, cap 50), so the feared
  oscillation toward the dense-cell break-even did not occur. But
  transmissions fell faster than coverage followed the rising price: the
  policy under-covers by ~5 points. Mean wall-clock was 88 s per update, so
  the configured 2,000 updates is ~49 h on this CPU.
  **The multiplier step can mask misses.** λ is driven by the mean of
  per-episode shortfalls, each normalised by its own target. Low-ceiling cells
  (d=1 targets down to ~0.12) that overshoot pull the mean negative while the
  pooled coverage is still short — update 35 had coverage 0.451 against target
  0.520 yet a mean shortfall of −0.048. **Changed after run4** (the user chose
  this before any further training): one multiplier per (scenario, density)
  group — 16, weather and hazard pooled — each stepped on its group's pooled
  shortfall `(Σ target − Σ coverage) / Σ target`. Pooling alone would let
  dense-cell surplus hide sparse-cell misses, the paper's regime; per-cell
  (96 multipliers) is too sparsely sampled for a ~200-update run. History
  now logs `lambda_by_group`, `shortfall_by_group` and the old per-episode
  mean for comparison. run4's numbers above were produced by the old rule.
- **run4 pilot evaluation (40 updates): no general cost advantage, and the
  sparse cell fails.** `experiments.evaluate_agent` on seeds 0–9, the four
  committed cells, suppression bias swept −2…+3, with the gate on (τ = 0.5,
  fallback `weighted_p`) and off (τ = 0, network only). Cost is transmissions
  per at-risk vehicle informed, at the cheapest agent setting reaching 95% of
  the cell's baseline RWCR ceiling, against the cheapest baseline that does.
  Outputs live in `checkpoints/run4/eval*` (git-ignored); nothing entered
  `results/`.

  | cell | cheapest baseline at target | gated τ=0.5 | network only τ=0 |
  |---|---|---|---|
  | rural d=2 | slotted_1p 1.79 | never (best RWCR 0.401 vs 0.655) | never (best 0.613) |
  | rural d=20 | DV-CAST 0.70 | 0.61 (−13%) | 0.77 (+10%) |
  | rural d=80 | weighted_p 0.91 | 0.47 (−48%) | 0.49 (−46%) — see below |
  | urban d=20 | counter_based 0.47 | 0.73 (+56%) | 0.56 (+20%) |

  - The sparse cell — the paper's target regime — is not reached at any
    setting, gated or not.
  - The network alone is cheaper than the best baseline in one cell of four,
    and the committed −46% overstated it. The agent's qualifying point sits
    at RWCR 0.876 against target 0.869 (seed-std 0.019), where the committed
    baseline knob grids had no points (nothing between RWCR 0.649 and 0.891).
    A finer cheap-end sweep at rural d=80 (seeds 0–9; not merged into
    `results/pareto_cells.json`):

    | setting | RWCR | cost | median TIR |
    |---|---|---|---|
    | `weighted_p` max_p=0.07 | 0.852 ± 0.083 | 0.574 | 0.61 s |
    | `p_persistence` p=0.08 | 0.869 ± 0.083 | 0.787 | 0.59 s |
    | agent, network only, bias +2 | 0.876 ± 0.020 | 0.490 | 0.32 s |
    | agent, network only, bias +1 | 0.901 ± 0.016 | 0.859 | 0.34 s |

    `p = 0.08` clears the target (0.869275) by 0.00002, so the refined
    cheapest qualifying baseline is 0.787 and the agent's margin is **−38%**;
    interpolating `weighted_p` to the agent's RWCR gives 0.78 (−37%). Near
    RWCR 0.90 the margin shrinks to −14% (0.86 vs interpolated 1.00). The
    agent is also ~4× less variable across seeds and reaches at-risk vehicles
    in about half the median time. This is one dense cell, just above its
    target, from a 40-update pilot — not yet a finding.
    **The committed baseline grids are too coarse at the cheap end** for
    matched-quality comparisons in dense cells; the Phase 7 sweep must extend
    them (e.g. `p`, `max_p` below 0.1) before regret or margin is reported.
  - The gate at τ = 0.5 helps only by mixing in `weighted_p`; it cuts urban
    coverage below target and imports `weighted_p`'s die-out (below).
  - TIR: network-only median TIR is within 3% of flooding in the three cells
    where it reaches target, the one axis where the pilot looks competitive.
- **FIXED in `5f02df7`: the engine silently discarded frames after five
  busy-channel deferrals, and in dense cells this could kill a whole episode.**
  Every number below and above this entry that was simulated before
  `5f02df7` — the run4 pilot evaluation, the baseline-reproduction check, the
  rural d=80 fine sweep, the `weighted_p` die-out statistics, run3–run5 — used
  the old engine and is superseded; `results/pareto_cells.json` and
  `results/coverage_targets.json` are being regenerated, and
  `METRICS_VERSION` is now 3. Found while diagnosing run5,
  whose urban d=80 multiplier kept receiving pooled coverage of exactly 0.
  `DisseminationEngine._channel_access` (since Phase 2, `ba85971`) draws a
  clear-channel check once per 100 ms epoch with `p_busy` = the background
  beacon channel load (≈ 0.50 at urban d=80 under DCC, capped near its 0.62
  target), defers a busy frame one epoch, and after
  `max_busy_deferrals = 5` sets it to unscheduled. The give-up is not counted
  as a drop (only `n_busy_deferrals` rises), no test pins it, and the
  `[ASSUMED]` config comment was the only record. A frame therefore vanishes
  with probability ≈ `p_busy^6`, and local `p_busy` runs well above the
  network-average CBR: the DCC beacon-rate floor (1 Hz) lets it reach 0.81
  around an urban d=80 originator, so `0.81^6` ≈ 28% of episodes never
  transmit at all. When the frame is the originator's, nothing is ever sent:
  urban_nlos d=80 seed 2 gives 0 transmissions, 0 reception attempts and 6
  busy deferrals for flooding, `slotted_1p` and the agent alike. Measured with
  `_channel_access` instrumented (flooding / `slotted_1p`, seed 0):

  | cell | local p_busy (mean / max) | relay frames given up |
  |---|---|---|
  | urban d=80 | 0.66 / 0.82 | 8.8% / 2.1% |
  | rural d=80 | 0.62 / 0.62 | 5.9% / 1.0% |
  | urban d=20 | 0.62 / 0.62 | 5.8% / 0.6% |
  | rural d=20 | 0.31 / 0.36 | 0.0% / 0.0% |

  On the training pool, flooding at urban d=80 transmitted nothing on 2 of the
  first 8 seeds (103, 104). The coverage targets were measured on seeds 100–101,
  which both disseminated, so roughly a quarter of urban d=80 training episodes
  cannot reach their target under any policy — run5's urban d=80 multiplier
  rose to 7.3 by update 24 chasing them. Real 802.11p defers in 13 µs
  backoff slots and would send this frame within milliseconds, so this is a
  modelling artefact, not channel physics. It affects every dense cell —
  committed baseline curves, coverage targets and training alike — and
  penalises low-redundancy schemes (slotted, counter, the agent) more than
  flooding, which has spare relays.
  **The fix** (user decision: stop run5, fix, rerun): a frame that finds the
  medium busy is delayed within its 100 ms epoch and still transmitted, as
  802.11p's 13 µs backoff allows; the busy draw only counts
  `n_busy_deferrals`, and `max_busy_deferrals` is removed and refused if
  configured. Verified on the formerly dead episodes: urban d=80 seed 2 now
  reaches RWCR 0.921 under flooding (was 0.000) and training seed 103 0.914
  (was 0.000); flooding sends exactly one frame per informed vehicle
  (1,915 / 1,915). Tests pin that the originator transmits, in its scheduled
  epoch, on a medium forced 99% busy.
- **Cost at matched RWCR is unbounded in latency, so slot-based baselines
  win by waiting.** Regenerated on the fixed engine with the extended grids
  (`results/pareto_cells.json`, seeds 0–9), the cheapest qualifying point of
  `slotted_1p`, DV-CAST and greedy sits on the **largest slot count in every
  cell**, and cost is still falling there while median TIR climbs:

  | cell | `slotted_1p` n_slots = 12 → 50 | DV-CAST 12 → 50 |
  |---|---|---|
  | rural d=2 | cost 1.79 → 1.56, TIR 0.91 → 3.09 s | 1.70 → 1.45, 0.81 → 3.37 s |
  | rural d=20 | 0.80 → 0.73, 0.41 → 0.51 s | 0.79 → 0.72, 0.41 → 0.56 s |
  | rural d=80 | 2.06 → 1.65, 0.22 → 0.28 s | 2.04 → 1.66, 0.22 → 0.28 s |
  | urban d=20 | 0.61 → 0.54, 1.02 → 3.03 s | 0.60 → 0.55, 1.01 → 3.07 s |

  RWCR credits a warning whenever it arrives while the vehicle is still at
  risk, so a scheme that waits seconds still "matches quality", and the 100 ms
  slot granularity turns 50 slots into up to 5 s of wait (a real slotted
  scheme uses millisecond slots). Extending the grids further would only move
  the edge. Consequently the regenerated reference table — best fixed baseline
  `dvcast(n_slots=50)` within 1.00–1.04× of per-cell hindsight tuning in three
  of four cells, 1.30× at rural d=80 — describes baselines that trade latency
  for cost.
  **Resolved (user decision): matched quality now carries a deadline guard.**
  A setting qualifies only if RWCR ≥ 95% of the cell ceiling **and** its
  actionable-deadline miss rate is at most the cell's best (excluding the
  policy being scored) + `miss_margin`, default 0.05 absolute
  (`analysis/comparator.py`; `--miss-margin`, `--no-miss-guard` on both the
  comparator and `experiments.evaluate_agent`). It disqualifies exactly the
  waiting points: at rural d=2 the 50-slot `slotted_1p` / DV-CAST points miss
  0.65 / 0.64 against a best of 0.53, at urban d=20 0.13 against 0.067; at
  rural d=20 and d=80 every scheme is within ~0.01 of the best and 50 slots
  cost only ~0.5 s, so nothing there is removed.
  A first version dropped guard-failing points before interpolating, and the
  crossing jumped across unmeasured grid: rural d=80 `p_persistence_03` read
  0.90 at margin 0.05 but 1.34 at 0.10. The lower interpolation anchor is now
  always the next-cheaper short setting of the full curve, and no
  interpolation is done if it fails the guard; a test pins that loosening the
  margin never raises matched cost.

  Reference table on the regenerated curves (cost; oracle-best / best fixed /
  headroom):

  | cell | no guard | margin 0.02 | **margin 0.05** | margin 0.10 |
  |---|---|---|---|---|
  | rural d=2 | 1.45 / 1.45 / 1.00× | 1.65 / 1.69 / 1.02× | **1.56 / 1.56 / 1.00×** | 1.56 / 1.56 / 1.00× |
  | rural d=20 | 0.70 / 0.72 / 1.02× | 0.70 / 0.77 / 1.09× | **0.70 / 0.73 / 1.04×** | 0.70 / 0.73 / 1.04× |
  | rural d=80 | 1.27 / 1.66 / 1.30× | 1.51 / 1.84 / 1.22× | **1.51 / 1.70 / 1.13×** | 1.27 / 1.70 / 1.34× |
  | urban d=20 | 0.53 / 0.55 / 1.04× | 0.53 / 0.58 / 1.09× | **0.53 / 0.57 / 1.06×** | 0.53 / 0.57 / 1.06× |
  | best fixed | `dvcast(n_slots=50)` | `slotted_1p(n_slots=20)` | **`dvcast(n_slots=30)`** | `dvcast(n_slots=30)` |

  **What this means for the paper's claim:** under a deadline-guarded
  comparison a single fixed scheme is within 0–13% of per-cell hindsight
  tuning in every committed cell (0–34% across margins). A learned policy can
  show a large margin over the best fixed baseline only by beating per-cell
  hindsight tuning itself. These are four clear-weather fog-bank cells; the
  headroom elsewhere in the factorial is unmeasured.
- **The committed baseline curves still reproduce.** `results/pareto_cells.json`
  was generated at `788d5fe`, before six later commits touched simulation,
  metrics or policy code. Eight committed points (all four cells; flooding,
  p-persistence, slotted, weighted-p, counter and DV-CAST) were re-run through
  `analysis.pareto.sweep_policy` on seeds 0–9 at `a22301c`: RWCR and cost
  match to every printed digit (worst absolute difference 0). Agent regret and
  margin are therefore scored against current-code baselines. This is a
  sample, not a full regeneration.
- **The confidence gate's fallback (`weighted_p`) can kill a message, and the
  agent inherits that.** `weighted_p` decides once per vehicle with
  probability `sender_distance / R` and never retries. On urban_nlos d=20
  seed 2 the originator's broadcast reached 14 vehicles, all 13 relays drew
  suppress, and dissemination died one hop out (RWCR 0.014); `slotted_1p` on
  the same seed informs 477 of 480. It is the only baseline with this failure:
  every `weighted_p` setting at useful coverage has seed-std RWCR ≈ 0.28 there
  (a mean of ~0.87 is nine seeds at ~0.96 and one at ~0), against ≤ 0.02 for
  every timer, counter and distance scheme. When the gate hands seed 2 to the
  fallback, the agent dies the same way (bias 0: RWCR 0.014, fallback rate
  1.0); at bias −2 a 0.2% share of network decisions kept it alive (0.964).
  Consequences: (1) urban "reaches matched quality" for any agent setting can
  turn on this single seed rather than typical behaviour; (2) the fallback
  choice is itself a design decision the gate ablation must report — a
  fallback that cannot die (e.g. `slotted_1p`) is the obvious candidate, but
  it has not been changed or tested.
- **Grid risk estimation is approximate.** The causal field in a grid is
  route-unaware (Manhattan distance + bearing gate), which is why its
  correlation with ground truth is 0.13 there. The oracle fixes *evaluation*;
  the agent still has to work from the weak causal estimate, which is the
  honest problem statement.
- **Deferring schemes pay a 100 ms slot granularity.** A real slotted scheme
  uses millisecond slots. This inflates absolute TIR for every deferring scheme
  equally, so internal comparisons hold, but absolute latencies are pessimistic.
  `slot_epochs` is exposed for a sensitivity check.
- **`cw_min` is unverified and load-bearing** (see Constants).
- **Weather is traffic-mediated only.** No channel-degradation claim anywhere.
- **One originator per hazard.** Redundant sensing would flatter every policy.
- **Background CAM/BSM load is a Poisson arrival process** at the receiver with
  ETSI-style DCC rate adaptation, not individually simulated beacon frames.
- **Ensemble confidence costs N forward passes** per 100 ms decision; only the
  entropy gate is realistic at that budget.
