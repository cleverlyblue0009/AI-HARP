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

`estimation_agreement()` reports how well the causal field recovers the oracle.
Over the full grid (`experiments/risk_estimation.py` →
`results/risk_estimation.csv`: 8 densities × 4 weathers × 5 hazards × 10 seeds
= 1,600 cells per scenario; mean ± std across cells; the estimate does not
depend on the dissemination policy):

| scenario | precision | recall | peak-relevance correlation |
|---|---|---|---|
| rural_highway | 0.78 ± 0.12 | 0.91 ± 0.19 | **0.74 ± 0.14** |
| urban_grid | 0.57 ± 0.09 | 1.00 | **0.17 ± 0.14** |
| urban_nlos | 0.53 ± 0.08 | 1.00 | **0.19 ± 0.12** |

(Earlier single-cell figures were 0.81 for the corridor and 0.13 for
urban_nlos.) By density the corridor correlation holds at 0.75–0.79 up to
d = 40 and falls to 0.54 at d = 80; the grids improve with density, from
0.09–0.14 at d = 1 to 0.33 at d = 80.

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

`cw_min` is **load-bearing** — it sets the contention window that drives the
whole collision model — and is now **half verified**. The per-access-category
values were checked by the user against IEEE 802.11-2020's default EDCA
parameters for `dot11OCBActivated = true` (2026-09-14): with aCWmin = 15 and
aCWmax = 1023 for the OFDM PHY,

| AC | CWmin | CWmax | AIFSN |
|---|---|---|---|
| AC_BK | 15 | 1023 | 9 |
| AC_BE | 15 | 1023 | 6 |
| **AC_VI** (configured) | **7** = (aCWmin+1)/2 − 1 | 15 | 3 |
| AC_VO | 3 = (aCWmin+1)/4 − 1 | 7 | 2 |

so the configured `cw_min = 7`, `cw_max = 15`, `aifsn = 3` are the correct
AC_VI values. The edition's table number was not recorded. **Still
unverified: that a hazard DENM is sent as AC_VI.** That mapping comes from
ETSI (EN 302 663 / EN 302 636-4-1 traffic-class mapping), not 802.11; if
DENMs use AC_VO, `cw_min` is 3 and results must be regenerated. Until that is
checked, report `cw_min` as a sensitivity across AC_VO / AC_VI / AC_BE.

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
     RWCR, because the causal grid estimator correlates only 0.19 with ground
     truth (urban_nlos, mean over the full grid in
     `results/risk_estimation.csv`; an earlier single cell gave 0.13). Per-cell targets keep the constraint feasible, but there the agent
     learns which vehicles the proxy flags, not which are truly at risk.
     Training may not use the oracle, so this is the honest limit of the
     causal-only rule; evaluation measures the gap.
  2. **λ may oscillate.** Dense cells exceed their targets easily, so dual
     ascent pulls λ toward the ~0.5 break-even where silence becomes
     attractive again. λ is logged every update so this is visible.
- **Carrying was a free postponement, and run6 collapsed onto it.** On the
  fixed engine with per-group λ, run6 trained normally for 40 updates, then
  at update 41 entropy fell 0.73 → 0.27 and the batch held 34,932 transitions
  (typically 11–19k). Replaying that checkpoint:

  | episode | policy | decisions / informed vehicle | carry share | informed | RWCR |
  |---|---|---|---|---|---|
  | rural d=40, seed 105 | run6 u41 | 19.5 | 94% | 586 | 0.821 |
  | | `slotted_1p` | 1.0 | 0% | 792 | 0.900 |
  | urban d=40, seed 124 | run6 u41 | 16.1 | 92% | 908 | 0.912 |
  | | `slotted_1p` | 1.0 | 0% | 953 | 0.966 |

  `carry_and_forward` costs no transmission, never settles, and the reward
  lands only on a vehicle's last decision, so holding and being re-asked
  every second was never penalised — the same shape as the earlier suppress
  and defer bugs. **Fix (user decision): at most
  `action_space.max_carries_per_vehicle = 3` carries per vehicle per
  message**, after which carry is masked out. The mask is applied inside the
  network's sampling (so the recorded log-prob and entropy describe the
  distribution actually sampled), stored on the transition, and re-applied
  when PPO recomputes log-probs; tests pin that a real network never samples
  a masked action and that `evaluate_actions` reproduces the masked log-prob
  exactly. History now logs `carry_rate` and `decisions_per_informed`. run6
  (42 updates) is superseded.
- **run7 (200 updates, carry cap, fixed engine, per-group λ): stable, but
  the sparse constraints are not met.** 3.25 h at `2b239a1`; no crash, no
  update with a zero-coverage group, no λ at its cap (max 14.95), carry
  bounded (25–39% of decisions, peak 0.50; 1.3–1.6 decisions per informed
  vehicle), entropy steady at ~0.9. Mean pooled shortfall over updates
  161–200 was +0.068 (target met in 8 of 40). Per group over those updates:

  | group | mean shortfall (+ = short) | met | final λ |
  |---|---|---|---|
  | rural d=1 | −0.153 | 7/15 | 1.09 |
  | **rural d=2** | **+0.342** | 0/18 | 9.37 |
  | **rural d=3** | **+0.171** | 1/12 | 14.95 |
  | rural d=5 | +0.056 | 5/17 | 8.41 |
  | rural d=10 / 20 / 40 / 80 | −0.020 / −0.020 / +0.025 / −0.015 | 11/12, 13/15, 5/20, 12/17 | 6.34 / 2.09 / 5.78 / 1.50 |
  | urban d=1 | −0.281 | 13/16 | **0.00** |
  | **urban d=2** | **+0.295** | 2/14 | 7.66 |
  | urban d=3 / 5 | −0.031 / +0.060 | 10/16, 7/17 | 3.94 / 6.49 |
  | urban d=10 / 20 / 40 / 80 | +0.100 / +0.096 / +0.062 / −0.015 | 2/17, 2/14, 4/16, 8/15 | 13.46 / 11.69 / 10.76 / 2.62 |

  The dense rural groups sit on their targets. The sparsest groups — rural
  d=2–3 and urban d=2, the paper's regime — are still 0.17–0.34 short with λ
  climbing, i.e. 200 updates (with those densities appearing only in the last
  ~80) is not enough to meet them. Urban d=1's target (~0.12–0.19) is so low
  that its λ fell to 0: satisfying that constraint does not show the policy
  is useful there (the low-ceiling caveat above).
- **run7 evaluation, network only (τ = 0): the claim does not hold at 200
  updates.** `ckpt_000200.pt` (sha `f23eabbec9f10413`), seeds 0–9, the four
  committed cells, suppression bias −2…+3, deadline guard (margin 0.05),
  against the regenerated fixed-engine baselines. Outputs in
  `checkpoints/run7/eval_tau0` (git-ignored).

  | cell | target RWCR / miss bound | agent best RWCR | at target? | agent cost | per-cell oracle / best fixed |
  |---|---|---|---|---|---|
  | rural d=2 | 0.641 / 0.579 | 0.593 (bias −2) | **no** | — | DV-CAST 1.56 / 1.56 |
  | rural d=20 | 0.879 / 0.364 | 0.916 | yes | **0.95** | greedy 0.70 / 0.73 → regret **+35%**, margin **−23%** |
  | rural d=80 | 0.870 / 0.368 | 0.848 (bias −1) | **no** | — | weighted_p 1.51 / 1.70 |
  | urban d=20 | 0.9261 / 0.117 | 0.9258 (bias −2; seed-std 0.020) | **no** — by 0.00034 | 0.89 at bias −2 | counter_based 0.53 / 0.57 |

  It fails the matched-quality test in three of four cells. The urban miss
  is statistically a tie with the target, but only at the most
  transmit-happy setting and at ~60–70% more cost than the baselines. Where
  it qualifies (rural d=20) it is 35% dearer than per-cell hindsight tuning
  and 30% dearer than the best fixed scheme. **Latency is the one strength**:
  median TIR 0.30–0.39 s in every cell, and at rural d=20 0.32 s against
  flooding's 0.35 s (regret −7%). Confound: 200 of the 2,000 configured
  updates, with the sparse groups still short of target when training
  stopped — this says "not yet", not "cannot". Per the standing rule the
  configuration is not being tuned toward the claim.
- **run8 (staged curriculum, 1,000 updates, GPU PPO deterministic): the
  sparse constraints are not met, and the claim does not hold.** Pretrain 200
  dense updates, then finetune to the 800-update cap (early stopping never
  triggered); 3.17 h at `566f2ad`, `bit_reproducible: true`.

  Training, mean shortfall over the last 100 finetune updates (run7 = its last
  40 updates):

  | group | run7 | run8 | run8 final λ |
  |---|---|---|---|
  | rural d=2 | +0.342 | **+0.153** (met 13/50) | **50 (cap)** |
  | rural d=3 | +0.171 | +0.046 (20/52) | 49.6 |
  | urban d=2 | +0.295 | **+0.144** (9/52) | **50 (cap)** |
  | rural d=5, urban d=1/3/5/10/80, dense rural | −0.25 … +0.06 | −0.25 … +0.02 | 0 … 13 |
  | urban d=20 / 40 | +0.096 / +0.062 | +0.038 / +0.025 | 28.2 / 21.8 |
  | pooled | +0.068 (met 8/40) | +0.026 (met 35/100) | |

  The staged curriculum roughly halved the sparse shortfall, but rural d=2 and
  urban d=2 plateaued ~0.15 short with λ at its cap from update ~600 — the
  configured finding that these targets are not met at maximum price. Cost
  per at-risk vehicle rose 0.46 → 0.73 as coverage was bought.

  Evaluation of `ckpt_final.pt` (seeds 0–9, four committed cells, deadline
  guard 0.05). **Headline = sampled policy with the confidence gate (τ = 0.5)**
  (user decision; see the next entry):

  | cell | target RWCR | gated sampled | gated argmax | network-only sampled | network-only argmax | cheapest baseline / best fixed |
  |---|---|---|---|---|---|---|
  | rural d=2 | 0.641 | never (best 0.603) | never | never (0.599) | never (0.604) | 1.56 / 1.56 |
  | rural d=20 | 0.879 | **0.71** (regret +0.4%, margin **+3.4%**) | 0.71 (+1.0%, +2.7%) | 0.95 (+35%) | 0.84 (+20%) | 0.70 / 0.73 |
  | rural d=80 | 0.870 | never (best 0.857) | never | never (**0.868**) | never (0.716) | 1.51 / 1.70 |
  | urban d=20 | 0.926 | 0.70 (+31%, −19%) | 0.69 (+29%, −18%) | 0.83 (+56%) | 1.05 (+96%) | 0.53 / 0.57 |
  | mean regret / margin | | **+15.7% / −7.7%** | +15.0% / −7.4% | +45.5% / −27.5% | +58.0% / −29.5% | |

  - The agent reaches matched quality in two of four cells and is competitive
    in one (rural d=20: within 0.4% of per-cell hindsight tuning, 3.4% cheaper
    than the best fixed scheme). It is 31% dearer in urban d=20 and never
    reaches rural d=2 or d=80.
  - The gate is doing much of the work: at τ = 0.5 it hands 13–55% of
    decisions to `weighted_p` and cuts network-only cost from 0.95 to 0.71
    (rural d=20) and 0.83 to 0.70 (urban d=20).
  - Latency remains the strength: median TIR 0.20–0.39 s at the best points.
  - Outputs: `checkpoints/run8/eval{,_tau0}{,_sampled}` (git-ignored).
- **Evaluation was scoring a different policy from the one trained.**
  `evaluate_agent` used argmax actions, but the coverage constraint was
  trained on the stochastic policy. On rural d=80 (seeds 0–4, τ = 0) argmax
  reached oracle RWCR 0.642 against 0.814 sampled at the same cost; rural d=2
  was identical either way. The run7 and run8 argmax evaluations understate
  the dense cell. `--policy-mode sampled` is now the default; sampled
  evaluation draws its uniforms from a generator seeded per run from a hash of
  the run's policy RNG (without advancing it), so results depend only on the
  run seed (`tests/test_eval_sampling.py`).
- **Two further training-side limitations surfaced in run8.** (1) Early
  stopping could not fire: it needs 30 consecutive updates with every sampled
  group at target, but with 8 episodes per update each group gets one or two
  episodes, so almost every update has a noisy short group (the counter never
  left 0, even while most groups met target on average). (2) Groups whose
  target is tiny (rural and urban d=1) drove λ to 0, and at zero price the
  policy occasionally went near-silent (coverage 0.6–0.7 short in single
  episodes) before λ recovered. Neither was changed mid-run.
- **Was run8's sparse plateau a cap artefact or infeasibility? Both, split by
  group.** Two checks, before any further experiment.

  *(1) λ cap.* Sparse-only finetunes (densities {1, 2, 3, 5}) from run8's
  `ckpt_pretrain.pt`, cap raised 50 → 500
  (`results/feasibility/lambda_cap_arms.txt`, `experiments/lambda_cap_summary.py`;
  deterministic CUDA, `bit_reproducible: true`). **A**, as specified (λ climbs
  from its initial value, 300 updates), cannot answer the question: at
  λ_lr = 1 its λ reached 49.7 / 48.6, never exceeding the old cap. **B** prices
  every sparse group at 500 from the first update, 600 updates. Pooled
  shortfall, mean ± s.e. over the last 200 finetune updates:

  | group | run8 (cap 50) | B (λ = 500) |
  |---|---|---|
  | rural d=2 | +0.139 ± 0.020 | **+0.131 ± 0.019** — unchanged |
  | urban d=2 | +0.155 ± 0.019 | **+0.051 ± 0.016** — cut by two thirds, still short |
  | rural d=3 | +0.077 ± 0.016 | +0.037 ± 0.012 |

  B's shortfall is flat from update ~200 to 600, so neither group closes with
  more updates at this price. At λ = 500 transmissions per update rose
  ~30 → ~54 and the carry rate fell 0.24 → 0.02. The cap was binding for
  urban d=2 (partly) and not for rural d=2.

  *(2) Is the target reachable by anyone?* `experiments/sparse_feasibility.py`
  runs every baseline at every knob setting (63 settings: the Pareto sweeps plus
  each registry default) in all six training weather × hazard cells at d = 2,
  on all 32 training seeds (causal coverage, what training optimises) and on
  evaluation seeds 0–9 (oracle RWCR). 31,752 runs;
  `results/feasibility/sparse_feasibility{.txt,.json,_runs.csv}`. Its runner
  reproduces the committed ceilings and Pareto points exactly.

  | group | training target (mean) | best fixed baseline, pooled shortfall | settings meeting it | per-episode hindsight envelope | agent: run8 / B |
  |---|---|---|---|---|---|
  | rural d=2 | 0.891 | +0.123 (`dvcast` n_slots=2) | **0 of 63** | −0.050 | +0.139 / +0.131 |
  | urban d=2 | 0.329 | −0.197 (`slotted_1p` n_slots=50) | 7 of 63 | −0.375 | +0.155 / +0.051 |

  - **rural d=2: the training target is infeasible for every fixed baseline at
    any cost.** It is 0.95 × a ceiling measured on two seeds (100–101), where
    flooding reached causal coverage 0.909; over the 32-seed pool flooding
    averages 0.778. The agent (+0.131) sits at the best fixed baseline
    (+0.123) within one standard error. Only a policy choosing the best setting
    per episode in hindsight could meet it. The agent is not being singled out
    here: the target was a measurement artefact of `objective.target_seeds: 2`.
  - **urban d=2: the target is feasible, and the agent misses it.** Seven fixed
    settings clear it, the best by 20%, and all of them wait long before
    rebroadcasting (`slotted_1p` / `dvcast` n_slots 20–50, `greedy_farthest`
    fallback 32). The agent
    stays 0.05 short even at ten times the price. Its longest wait is
    `defer_3` plus at most three carries, which cannot express those waits.
    That is a hypothesis about the action space, not yet tested.
  - **Evaluation-side ceilings are optimistic in rural d=2.** Each comparator
    target is 0.95 × the best of 63 means, and rural d=2 seeds vary by
    ±0.16–0.27 RWCR, so the ceiling (e.g. 0.675 at clear/fog_bank) sits
    ~0.12 above the median setting (0.557) and is 4–14 settings deep. Urban
    d=2 ceilings are tight (±0.02–0.06).
  - Consequence: every training target in `results/coverage_targets.json` was
    measured on the same two seeds, so dense-cell targets may be biased too;
    this has not been checked. Nothing has been re-measured or retuned.
- **Trace cache race (fixed).** Two processes caching the same trace could
  crash on Windows (`os.replace` refused while the other copy was open); the
  feasibility sweep died on it after 25,856 runs and resumed from its CSV.
  Since traces are deterministic in their key, the save now retries and keeps
  the existing copy (`tests/test_mobility.py`).
- **Decisions after the feasibility checks (user, before any campaign run).**
  (1) Coverage targets are re-measured on all 32 training seeds
  (`objective.target_seeds: 32`; same definition: 0.95 × the better of flooding
  and DV-CAST defaults, causal coverage). (2) `objective.lambda_max: 500`. (3)
  The action space of the reference agent is unchanged; longer waits are one
  reported ablation, not a change to the agent. run8 stays in this README as
  the historical run under the old targets and cap; every campaign number is
  against the new objective.

  Re-measured ceilings (`results/coverage_targets.json`, 96 cells × 32 seeds;
  mean over the six weather × hazard cells of each group):

  | group | old (2 seeds) | new (32 seeds) | change |
  |---|---|---|---|
  | rural d=1 | 0.298 | 0.253 | −0.046 (one cell −0.276) |
  | **rural d=2** | 0.938 | **0.787** | **−0.151** |
  | rural d=3 | 0.956 | 0.876 | −0.080 |
  | rural d ≥ 5 | 0.962–0.999 | 0.965–0.999 | ≤ 0.009 |
  | urban d=1 | 0.157 | 0.212 | +0.055 |
  | urban d=2 | 0.346 | 0.308 | −0.038 |
  | urban d=3–80 | 0.297–0.362 | 0.298–0.357 | ≤ 0.022 |

  The 2-seed ceilings were high exactly in the sparse rural cells where
  seed-to-seed variance is largest. The new rural d=2 target (0.95 × 0.787 =
  0.748) is met by the best fixed baseline's 32-seed coverage (0.781), so it is
  no longer infeasible for everyone. Dense-cell targets barely moved. Gate
  before training: `analysis.reward_check` SANE (silence break-even λ 0.679,
  cap 500); 480 tests pass.
- **Reference agent under the new objective (`checkpoints/campaign/ref`, 1,000
  updates, 3.45 h, bit-reproducible): closer, still not met.** Pooled shortfall
  over the last 200 finetune updates +0.022 (run8: +0.026 against the old,
  partly infeasible targets); early stopping never fired. Per group, mean ± s.e.
  (fraction of updates meeting target): rural d=2 +0.030 ± 0.022 (54%), urban
  d=2 +0.069 ± 0.020 (39%), rural d=3 +0.056 ± 0.017, rural d=5 +0.045 ± 0.008,
  rural d=40 +0.048 ± 0.007 (8%), rural d=10 +0.055 ± 0.016; met on average:
  rural d=1 and d=80, urban d=3/5/10/20/40/80. λ peaked at 33 (rural d=2) and
  31 (urban d=1), far below the 500 cap.

  **Headline evaluation** (`results/agent_ref/sampled/`; seeds 0–9, sampled
  policy, gate τ = 0.5, deadline guard 0.05, the four committed cells): the
  agent reaches matched quality in **one of four** cells, down from two for
  run8.

  | cell | target RWCR | agent best RWCR (cost) | result |
  |---|---|---|---|
  | rural d=2 | 0.641 | 0.507 (1.68) | never — short by 0.134 |
  | rural d=20 | 0.879 | 0.905 (0.74) | **0.72, regret +1.8%, margin +1.9%** |
  | rural d=80 | 0.870 | 0.846 (0.71) | never — short by 0.023 |
  | urban d=20 | 0.926 | 0.921 (0.66) | never — short by 0.005 (run8 reached it at 0.70) |

  It is not better than run8 anywhere: even in the one cell it reaches, rural
  d=20, it is slightly dearer (0.72 vs run8's 0.71; regret +1.8% vs +0.4%,
  margin +1.9% vs +3.4%), and two further cells fall just below the bar;
  urban d=20 misses by less than seed noise, and it is reported as a miss
  because that is the comparator's rule. Latency is again the strength: at
  rural d=20 its median TIR (0.32 s) beats the per-cell latency-best baseline
  (flooding, 0.35 s). Gate fallback 14–34% of decisions at the best points.

  **Argmax vs sampled, same checkpoint** (`results/agent_ref/{sampled,argmax}`),
  best RWCR over the suppression-bias sweep in each cell:

  | cell | target | sampled | argmax | sampled − argmax |
  |---|---|---|---|---|
  | rural d=2 | 0.641 | 0.507 | 0.477 | +0.030 |
  | rural d=20 | 0.879 | 0.905 ✓ | 0.904 ✓ | +0.001 |
  | rural d=80 | 0.870 | 0.846 | 0.825 | +0.021 |
  | urban d=20 | 0.926 | 0.921 | **0.927 ✓** | −0.006 |

  Sampling is still the better mode in three cells, but the gap has collapsed:
  run8's argmax lost 0.17 RWCR at rural d=80 (0.642 vs 0.814), this agent loses
  0.021. The retrained policy is sharp enough that the action-selection mode
  barely matters. It also makes the headline count mode-dependent: **urban d=20
  clears the bar under argmax (by 0.001) and misses it under sampling (by
  0.005)**, so "reaches matched quality in 1 of 4 cells" would read 2 of 4 had
  the argmax run been the headline. Sampled remains the headline because it is
  the policy the constraint trained (user decision); the swing is reported
  rather than used to pick the flattering mode.
- **Simulator validation against a published curve: the low-density plateau
  is reproduced, the density-driven drop is not.** `experiments/validate_amador.py`
  reproduces Table 3 (ETSI CBF) of Amador et al., *Vehicular Communications* 34
  (2022) 100454 (Artery/Veins/OMNeT++): 5 km, 4 lanes per direction, 20 mW,
  α = 2.0 free-space path loss, 6 Mbit/s, 10 MHz, DENM 301 B, hop limit 10,
  lifetime 10 s, destination area 4 km behind the source + 100 m ahead, ETSI CBF
  timers 1–100 ms over DIST_MAX 1000 m (`agents/cbf.py`). Run at a 10 ms step so
  the timers are resolved (user decision). 30 seeds per density
  (`results/validation/amador2022.{txt,json}`):

  | veh/km/lane | ours (PDR ± s.e.) | paper | ours − paper |
  |---|---|---|---|
  | 10 | 0.9977 ± 0.0008 | 0.9998 | −0.002 |
  | 20 | 0.9980 ± 0.0003 | 0.9961 | +0.002 |
  | 30 | 0.9987 ± 0.0003 | 0.9280 | **+0.071** |
  | 40 | 0.9990 ± 0.0002 | 0.9371 | **+0.062** |
  | 50 | 0.9993 ± 0.0002 | 0.9372 | **+0.062** |

  Mean |difference| 0.040. At 10–20 veh/km/lane we agree within 0.002; at
  30–50 the paper loses ~6–7% of the area and we lose almost none. Not a
  counting error (vehicles in the area match the commanded density: 326 vs 328
  expected at d = 10), and not a loss-free channel: at d = 50 one run logged
  76,435 SINR failures and 32,291 background-beacon losses, but with ~1,630
  vehicles in range and 201 CBF rebroadcasts nearly every vehicle still hears
  some copy. The paper attributes its drop to ETSI CBF's own behaviour; our
  engine abstracts 802.11p contention (every frame of a 10 ms epoch is
  concurrent; backoff is a tie probability, not per-13 µs slots) and implements
  CBF duplicate handling as cancel-on-one-duplicate, and the paper does not give
  enough detail to tell which mechanism is missing. **Implication for this
  paper:** coverage in dense cells is likely optimistic, which flatters every
  policy but most of all the high-redundancy ones (flooding, counter-based).
  Unstated in the reference and therefore [ASSUMED] here: receiver sensitivity
  (derived as −92.67 dBm so the range equals the stated 778 m; verified 778.0 m),
  noise figure (3.33 dB, same derivation), vehicle speeds (100–130 km/h), CAM
  rate (10 Hz with DCC), source placement. Nothing was tuned toward the paper's
  numbers.
- **Real OSM extracts (user decision: straight real road + projection).**
  Rural: US-50, central Nevada (`data/osm/rural_us50_nevada.osm.xml`, ©
  OpenStreetMap contributors, ODbL). Two lanes, undivided; the straightest
  10 km window runs 10,394 m (39.3130 N 117.9425 W → 39.3819 N 117.8607 W) and
  never deviates more than 1.3 m from its chord, so projecting positions onto
  distance along the chord changes no distance measurably. Urban grid: Midtown
  Manhattan; 98.5% of street length lies within 3° of two perpendicular axes
  (61° / 151° from east), so a −61° rotation aligns it with the grid model.
  Integration: `mobility/osm_geometry.py` (straightest window, chord
  projection, grid rotation, axis-classified edges), `mobility/sumo_osm.py`
  (carriageway routes for the real highway, `randomTrips` demand for grids,
  post-processing), scenarios `rural_highway_osm` / `urban_grid_osm`
  (`extends` their synthetic base and reuse its channel via `phy_profile`; the
  pure-Python fallback is refused). **US-50 on SUMO, d = 20, seed 0:** achieved
  21.3 veh/km/lane (commanded 20), projection window 10,000 m with 1.42 m
  maximum polyline deviation, 99th-percentile lateral offset 4.6 m, 213 / 213
  vehicles per carriageway at every sampled step; a flooding run completed
  end to end (RWCR 0.867).
- **Two SUMO-backend defects found on the way (fixed; neither affected any
  committed result, which all used the fallback backend).** (1) SUMO grid
  traces carried no `edges` list, so a grid hazard could not be placed at all,
  and no `grid_rows` / `grid_cols` / `block_length_m`, so the NLOS building
  model silently assumed 5 × 5 blocks for the 6 × 6 `urban_nlos` grid. (2) The
  synthetic SUMO grid's only demand was one flow over two edges: `urban_nlos`
  reached 0.25 veh/km/lane against 20 commanded, no vehicle reached the hazard,
  RWCR 0. Every SUMO grid now uses `randomTrips`, whose insertion rate is an
  estimate, so a grid trace is calibrated from its own measured density: up to
  three SUMO runs, a proportional step and then interpolation, because density
  is not proportional to demand once the grid congests. `urban_nlos` d = 20
  seed 0: 33.5 → 16.4 → **19.8 veh/km/lane** (commanded 20), recorded in
  `trace.meta["demand_calibration"]`; flooding then reaches RWCR 0.514. SUMO
  trace cache keys carry a pipeline version, so no pre-fix SUMO trace is
  served (fallback keys are unchanged). Also: FCD is recorded only
  after the warm-up (`--device.fcd.begin`); the US-50 export had been 933 MB,
  ~90% of it warm-up.
- **First ablation result (training side): GCN satisfies the constraint better
  than GATv2.** `checkpoints/campaign/gcn`, 1,000 updates, bit-reproducible,
  identical to the reference except `--encoder gcn`. Pooled shortfall over the
  last 200 finetune updates **+0.006 against the reference's +0.022**; per
  group (mean ± s.e., fraction of updates meeting target): rural d=3
  +0.021 ± 0.015 (53%) vs +0.056 ± 0.017 (40%), rural d=40 +0.017 ± 0.009
  (51%) vs +0.048 ± 0.007 (**8%**), urban d=2 +0.060 vs +0.069, rural d=2
  +0.037 vs +0.030. So on the quantity training optimises, attention is not
  earning its place. This is the training side only: cost at matched quality
  decides it, and the ablation evaluations are pending. Reported as measured,
  per the standing rule.
- **The headline conclusions are backend-dependent, and that is a result about
  the evidence base.** The four committed cells re-run with SUMO mobility
  (`results/pareto_cells_sumo.json`; synthetic networks, same PHY, same seeds,
  every cell within its commanded density — 40/40 runs OK, rural d=80 held at
  ~76 veh/km/lane by the congestion plan):

  | cell | ceiling, fallback → SUMO | oracle-best cost | best-fixed cost |
  |---|---|---|---|
  | rural d=2 | 0.675 → **0.221** | dvcast 1.56 → slotted_1p 1.15 | 1.56 → 1.15 |
  | rural d=20 | 0.925 → 0.875 | greedy_farthest 0.70 → dvcast 0.51 | 0.73 → 0.52 |
  | rural d=80 | 0.915 → 0.915 | weighted_p 1.51 → p_persistence 0.96 | 1.70 → 1.59 |
  | urban d=20 | 0.975 → **0.540** | counter_based 0.53 → slotted_1p 1.20 | 0.57 → **inf** |

  The single best fixed baseline changes identity (`dvcast(n_slots=30)` →
  `slotted_1p(n_slots=50)`), and in urban d=20 no fixed setting reaches 95% of
  the SUMO ceiling at all. The dense corridor agrees (identical ceiling), so
  this is not a blanket offset: the sparse corridor and the grid are where the
  mobility model decides the answer. **Every row in `results/runs.csv`, every
  target in `coverage_targets.json` and every agent result so far is
  fallback-backend**, so they describe one traffic model, not two.

  **Cause of the rural d=2 collapse: platooning, not density.** Both backends
  put ~40 vehicles on the corridor (fallback 40.0, SUMO 39.7 per step) and both
  look well connected on average (99% / 95% of vehicles have a neighbour within
  the 562 m nominal range). The spacing distribution is what differs: median
  gap 188 m (fallback) against 34 m (SUMO), with a longer tail (p90 596 m vs
  808 m; 12% vs 19% of gaps beyond range). SUMO's vehicles bunch into platoons
  separated by out-of-range gaps. At the hazard's onset step the fallback
  corridor is 2 connected clusters, the originator's holding 25 vehicles across
  5.4 km, so **62%** of the oracle at-risk set is reachable without crossing a
  gap; SUMO's is 11 clusters, the originator's holding 4 vehicles across 103 m,
  leaving **11%** reachable. A ceiling of 0.221 is what that allows.
  `mobility/fallback.py` regulates vehicle *count* per lane and spawns at the
  tail's speed, which spreads traffic far more evenly than car-following from a
  boundary inflow does. The sparse regime this paper is about is therefore the
  regime where the mobility model decides the answer. Nothing has been
  re-tuned; the choice of what to re-run is open (see the campaign note below).

  **The real maps side with SUMO, not with the fallback**
  (`results/pareto_cells_osm.json`: US-50 for the corridor, Midtown Manhattan
  for the grid, 40/40 runs within their commanded density). Achievable ceiling
  per cell:

  | cell | fallback | SUMO (synthetic net) | real OSM map |
  |---|---|---|---|
  | rural d=2 | **0.675** | 0.221 | 0.209 |
  | rural d=20 | 0.925 | 0.875 | 0.877 |
  | rural d=80 | 0.915 | 0.915 | 0.914 |
  | urban d=20 | **0.975** | 0.540 (urban_nlos) | 0.831 (Midtown) |

  Two independent car-following backends land within 0.012 of each other at
  d=2 and both sit ~0.46 below the fallback; at d=80 all three agree to 0.001.
  The fallback is the outlier, and only where the corridor is sparse. The best
  fixed baseline differs in every backend (`dvcast(30)` / `slotted_1p(50)` /
  `greedy_farthest(32)`), and so does the oracle-best cost at rural d=2 (1.56 /
  1.15 / 0.32). The urban comparison is between *different* scenarios
  (`urban_nlos` vs Midtown), so only the corridor rows are strictly
  backend-to-backend. This is now a question about the evidence base rather
  than about one figure, and it is with the user.
- **Every architecture ablation so far beats the reference on the training
  constraint.** Pooled shortfall over the last 200 finetune updates, each run
  identical to the reference but for one switch:

  | run | switch | pooled | rural d=2 | urban d=2 | rural d=3 | rural d=40 |
  |---|---|---|---|---|---|---|
  | ref | — (GATv2) | +0.022 | +0.030 | +0.069 | +0.056 | +0.048 |
  | gcn | `--encoder gcn` | **+0.006** | +0.037 | +0.060 | +0.021 | +0.017 |
  | star | `--star-graph` | **+0.007** | +0.024 | +0.060 | +0.016 | +0.058 |
  | mlp | `--encoder mlp` | **+0.013** | +0.041 | +0.072 | +0.052 | +0.063 |

  Attention (gcn), neighbour-to-neighbour edges (star) and the graph itself
  (mlp) can each be removed and the constraint is satisfied *better*. The star
  result is the sharpest: this README claims a star graph makes GATv2
  degenerate to attention pooling, and degenerating it helps. On the quantity
  training optimises, the architecture is not earning its place. Cost at
  matched quality still decides it and the ablation evaluations are pending —
  but if they agree, the paper's architecture section is a negative result.
- **The ablation evaluations agree: the architecture section is a negative
  result.** Each checkpoint scored exactly as the reference was (same cells,
  seeds, biases, τ = 0.5, deadline guard, sampled policy;
  `experiments/evaluate_ablations.py` → `results/agent_<run>/sampled/`):

  | run | cells failed (of 4) | regret vs per-cell oracle | margin vs best fixed | margin, median TIR |
  |---|---|---|---|---|
  | ref (GATv2) | **3** | +1.8% | +1.9% | +6.2% |
  | gcn | **1** | +3.1% | **+17.2%** | +16.6% |
  | mlp | **1** | +26.5% | **+22.3%** | +13.2% |
  | star | 2 | +17.4% | −8.5% | **+27.3%** |

  GCN reaches matched quality in 3 of 4 cells against the reference's 1, and is
  17.2% cheaper than the best fixed baseline where the reference manages 1.9%.
  MLP — no graph at all — fails the same single cell and has the best margin
  (+22.3%), though the worst hindsight regret (+26.5%), i.e. it is far from
  what per-cell tuning could achieve while still beating anything shippable.
  The reference's low regret (+1.8%) is measured over the one cell it reaches,
  so it is not comparable to a number averaged over three. **On the paper's own
  headline metric, removing the attention mechanism improves the agent.**
  Remaining ablations (heads 1/8, k 4/20, no causal relevance, long waits) are
  training or awaiting evaluation.
- **Slot granularity inflates latency only where slots are used**
  (`experiments/slot_granularity.py` → `results/slot_granularity.{csv,txt}`;
  `slot_epochs` × 100 ms per slot, 10 seeds, the four committed cells). Median
  TIR against slot length, `slotted_1p` / DV-CAST:

  | cell | 1× | 2× | 5× | 10× |
  |---|---|---|---|---|
  | rural d=2 | 0.55 s | 1.63× | 2.39× | **3.72×** (dvcast 4.12×) |
  | rural d=20 | 0.37 s | 0.97× | 1.05× | 1.19× |
  | rural d=80 | 0.21 s | 1.00× | 1.00× | 1.00× |
  | urban d=20 | 0.54 s | 1.30× | 1.82× | **2.54×** |

  Cost and RWCR barely move (rural d=2: cost 1.74 → 1.70, RWCR 0.638 → 0.674),
  so the slot length buys nothing and costs latency. In the dense corridor TIR
  is flat: a relay is always close enough that the first slot fires, so the
  granularity is invisible there. **The reported absolute latency of every
  deferring scheme is therefore a property of the 100 ms epoch in the sparse
  and urban cells, and not in the dense one** — which is where the agent's
  latency advantage was claimed, so that advantage is not a granularity
  artefact.
- **Paired statistics over the full grid** (`analysis/grid_stats.py` →
  `results/stats/`; Wilcoxon signed-rank, 10 paired seeds, Holm across the four
  metrics within each cell, effect size = matched-pairs rank-biserial).
  3,828 of 7,680 tests are significant; **the agent wins 615 and loses 3,213**.
  By metric, counting only significant tests:

  | metric | vs best fixed baseline | vs per-cell oracle-best |
  |---|---|---|
  | RWCR | 0 wins / 579 | 0 wins / 601 |
  | actionable miss rate | 0 wins / 577 | 0 wins / 604 |
  | transmissions per informed | **407 wins** / 636 | **187 wins** / 703 |
  | median TIR | **21 wins** / 21 | 0 wins / 107 |

  The pattern is consistent with the headline cells and sharper: the agent is
  cheaper than the shipped baseline in most cells where cost differs
  significantly, and never better on coverage or on the actionable-deadline
  miss rate — it loses those wherever they separate at all. A policy that buys
  cost by informing fewer at-risk vehicles is exactly what the constrained
  objective was meant to prevent, and at 10 seeds this is the strongest
  statement the evidence supports (no correction across cells is possible; see
  the module docstring).
- **Campaign training queue** (`experiments/campaign_train.py`, sequential,
  skip-if-done, exact resume; `checkpoints/campaign/<name>/`): the reference
  agent, then one retrain per architectural ablation, each differing from the
  reference in exactly one switch — `--encoder gcn`, `--encoder mlp`,
  `--star-graph` (`graph.include_neighbour_edges: false`), `--heads 1` / `8`,
  `--neighbour-cap 4` / `20`, `--drop-node-feature relevance_causal`
  (`graph.drop_node_features`: the column is zeroed, so shapes and the
  normaliser are unchanged), and `--defer-epochs 5 20 50`
  (`action_space.defer_epochs`, previously a dead key: `defer_k` now waits the
  configured epochs; a slotted scheme waits 1 + slot epochs, so the seven
  settings meeting the urban d=2 target span 1–50 epochs). Graph and
  action-space switches travel in the checkpoint config, so evaluation uses
  what a run was trained with (`tests/test_graph_ablations.py`,
  `tests/test_agent_semantics.py`). "10 seeds" per ablation means the paired
  evaluation seeds 0–9: every run uses one training seed, and
  training-seed variance is a limitation of every ablation result.
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

## Training throughput

Wall-clock per PPO update, measured by `experiments/profile_training.py` on
the same 3 updates (mixed densities, run7 weights, 8 episodes each) on the
development laptop (AMD Ryzen 7 7445HS, 6 cores / 12 threads; NVIDIA RTX 4050
Laptop, 6 GB). Nothing below changes the constrained objective, the reward,
the action semantics or the oracle/causal boundary.

| step | change | serial | 8 workers | 2,000 updates (8 workers) | profile |
|---|---|---|---|---|---|
| 0 | baseline | 45.2 s | 27.6 s | 15.3 h | `results/profile_baseline.txt` |
| 2 | batched decision inference | 24.4 s (1.85x) | 19.4 s (1.42x) | 10.8 h | `results/profile_step2_batching.txt` |
| 6 | risk features cached per epoch | 23.2 s | 18.3 s | 10.1 h | `results/profile_step6_cheap.txt` |
| 3 | PPO step on the GPU, deterministic | 18.4 s | **12.0 s** | **6.7 h** | `results/profile_optimised.txt` |

**2.3x overall at 8 workers.** The staged curriculum caps a run at 1,000
updates (~3.3 h), and early stopping on the constraints can end it sooner.

What each step found:

- **Baseline profile.** Per-decision inference was 44% of a serial update
  (15,247 single-graph forward passes, 4.12 ms each, 12.9-node graphs);
  decisions per engine epoch averaged 52. Amdahl bound for inference-only
  work: 1.85x. With 8 workers the serial PPO step was 43% of an update.
- **Batching** (`training.batch_inference`): all of an engine loop's decisions
  go through one forward pass (369 calls instead of 15,247; 0.93 ms per
  decision). It cannot be bit-identical to per-decision inference on CPU -- a
  bare `nn.Linear` over 1,300 rows differs from 13-row chunks by 3.6e-7 --
  so the correctness gate (user decision) is identical actions, masks,
  rewards, graphs, order and outcomes with log-prob/value/entropy within 1e-4
  (`tests/test_batched_inference.py`); 0 action flips in 1,794 decisions.
  Batched runs are byte-identical to themselves and across worker counts.
- **Cheap wins.** Kept: computing risk-field features once per epoch rather
  than per decision (simulator time 20.2 -> 13.5 s instrumented; exact,
  tested). Reverted: vectorising graph-edge construction (6.3 -> 6.0 s, noise).
  Reverted: 12 torch threads for PPO (16.1 s vs 14.0 s at 6).
- **GPU** (`training.device: auto`, `D:\aiharp-gpu`): PPO step on 6,222
  transitions 13.87 s CPU vs 3.13 s GPU -- but only 7.51 s with deterministic
  CUDA kernels, which are required so that no number in `results/` comes from a
  non-reproducible run (two identical non-deterministic GPU runs differed by
  2.6e-6 after 16 steps). `--nondeterministic-gpu` gives the 3.13 s for
  exploratory runs; `run_summary.json` records `bit_reproducible`. Rollouts
  stay on CPU workers.
- **Workers** (`results/bench_rollouts_laptop.txt`): 3.39x at 6, 3.89x at 8,
  3.75x at 11. Scaling stops at `rollout_episodes_per_update` (8) because one
  episode runs in one worker; `rollout_workers: auto` is capped there. A
  many-core cloud machine does not help unless episodes per update is raised,
  which changes the PPO batch and is an optimisation decision, not a free
  speed-up.

**The 5 h target for 2,000 updates is not reached on this laptop (6.7 h).**
What remains per update is ~6 s of rollouts, bounded by the slowest dense
urban episode, and ~6 s of deterministic GPU PPO. Remaining levers, none
applied: non-deterministic GPU kernels (~9 s/update, not reproducible); fewer
PPO epochs per update (4 -> 2 would cut PPO roughly in half but changes how the
policy learns); more episodes per update on a many-core machine (changes the
PPO batch). The staged plan's 1,000-update cap already fits in ~3.3 h.

**Launching the full two-stage run** (pretrain, then sparse-weighted finetune
with early stopping; resumable):

```bash
D:/aiharp-gpu/Scripts/python.exe -m agents.train --out checkpoints/run8 --keep-awake
# after an interruption, continue exactly where it stopped:
D:/aiharp-gpu/Scripts/python.exe -m agents.train --resume checkpoints/run8/ckpt_latest.pt --keep-awake
```

`run_summary.json` in the run directory records whether finetuning stopped on
the constraints or at its cap, the PPO device, and whether the run is
bit-reproducible.

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
  correlation with ground truth is 0.17 (urban_grid) / 0.19 (urban_nlos) over
  the full grid, against 0.74 on the corridor (`results/risk_estimation.csv`).
  The corridor estimate itself degrades in dense traffic (0.54 at d = 80). The oracle fixes *evaluation*;
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
