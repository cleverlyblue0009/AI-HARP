# AI-HARP

Simulation framework for hazard-message dissemination in sparse, weather-degraded
vehicular networks.

**Build status: Phases 1, 2, 3, 4 and 6 are implemented and tested. Phases 5, 7
and 8 are not started.**

| Phase | Scope | State |
|---|---|---|
| 1 | Mobility: SUMO scenarios, FCD parsing, cached `.npz` traces, pure-Python fallback | done (SUMO path untested — see below) |
| 2 | Network simulator: 802.11p PHY, CSMA/CA MAC, dissemination engine | done |
| 3 | Hazard model + risk field | done |
| 4 | Seven baseline policies | done (9 registered variants) |
| 5 | GAT-DRL agent (PPO / Dueling DQN) | not started |
| 6 | Metrics (PDR, RWCR, TIR, overhead, deadlines, lifetime) | done |
| 7 | Full factorial sweep, Wilcoxon tests, ablations | partial — paired runner + `results/runs.csv` exist |
| 8 | IEEE figures, LaTeX tables, `reproduce.sh` | not started |

---

## Quick start

```bash
pip install -r requirements.txt          # numpy, pandas, pyyaml, scipy, matplotlib
python -m pytest tests/ -q               # 201 tests, ~4 s
python -m experiments.run_sim --smoke    # whole pipeline, 256 runs, ~60 s
python -m experiments.compare --quiet    # Phase 4 baseline table, 10 seeds
```

The headline single run:

```bash
python -m experiments.run_sim --scenario rural_highway --density 20 \
    --weather clear --policy flooding --seed 0
```

Options: `--scenario {rural_highway,urban_grid}`, `--density <veh/km/lane>`,
`--weather {clear,moderate_rain,heavy_rain,dense_fog}`, `--policy`, `--seed`,
`--hazard <type>`, `--sample-hazard`, `--backend {auto,sumo,fallback}`, `--json`.

---

## Repository layout

```
configs/     phy.yaml (PHY+MAC), scenario_rural.yaml, scenario_urban.yaml,
             hazard.yaml, experiment.yaml     <- every constant lives here
common/      config loading, config hashing, seeding, logging
mobility/    SUMO generation + FCD parsing + pure-Python fallback + trace cache
sim/         phy.py (link budget), mac.py (CSMA/CA), engine.py (dissemination)
hazard/      model.py (the hazard object), risk_field.py (relevance)
agents/      base.py (policy interface), flooding.py, registry.py
analysis/    metrics.py
experiments/ run_sim.py (single run + --smoke)
tests/       111 unit + integration tests
cache/       generated traces (git-ignored)
results/     CSV, figures, LaTeX (git-ignored)
```

`common/` is not in the original layout sketch; it exists so `mobility`, `sim`,
`hazard` and `agents` can share config/seeding without importing each other.

---

## Which mobility backend is running

**No SUMO is installed in the current environment, so every number produced so
far comes from the pure-Python fallback generator.** The backend is printed as a
banner on every run and stamped into `trace.backend` and every metrics row.

The fallback is a real microscopic model — Krauss (1998) car-following with
SUMO-style dawdling, per-driver desired speeds, a mixed car/truck fleet, and
density regulated to the commanded value. It is **not** SUMO. It has:

- no lane changing and therefore **no overtaking** — on a single-lane
  carriageway every car behind a slow truck is stuck there, which depresses
  mean speed and creates platoons that *help* connectivity;
- no OSM geometry, no junction gap acceptance, no calibrated demand.

`mobility/sumo_runner.py` implements the SUMO path (network build via
`netconvert`/`netgenerate` or an OSM extract, demand sized from
`q = k·v·lanes`, FCD export, parse). **It has never been executed against a live
SUMO install.** Treat the first real SUMO run as an integration test.

To switch: install SUMO, set `SUMO_HOME`, and point
`configs/scenario_rural.yaml → sumo.osm_extract` at a real OSM extract of the
target corridor. `--backend sumo` then fails loudly rather than silently
falling back.

---

## Reproducing a number

Everything is seed-controlled. One master seed per run derives independent
named RNG streams (`mobility`, `hazard`, `shadowing`, `fading`, `mac`,
`policy`), so seed *k* gives **every** policy identical mobility, identical
hazard placement and identical fading — which is what makes the Phase 7
Wilcoxon signed-rank test operate on genuinely paired samples.

Traces are cached to `cache/traces/*.npz` under a hash of everything that
affects them; SUMO is never re-invoked. Delete `cache/` to force regeneration.
Every metrics row carries `config_hash` over the run spec, the PHY config, the
hazard config and the engine settings.

---

## Where the constants come from

Every numeric constant is in a YAML file with a `source:` field tagged
`[STD]` (a published standard), `[MEAS]` (a measurement campaign), `[DERIVED]`,
`[ASSUMED]` (our modelling choice, must be defended in the text), or
`[VERIFY]` (recalled from a standard — **must be checked against the primary
document before submission**). Grep for `[VERIFY]` before writing the paper:

```bash
grep -rn "VERIFY" configs/
```

### Two modelling points that affect what the paper may claim

**1. Rain does not meaningfully attenuate 5.9 GHz.** ITU-R P.838 gives
~0.12 dB/km at 25 mm/h, i.e. ~0.04 dB over a 300 m DSRC hop — negligible.
`configs/phy.yaml` therefore keeps two *separate* weather terms: `itu_rain`/
`itu_fog` (real hydrometeor physics, tiny) and `excess_loss_db_per_km` +
`exponent_delta` (empirical road-environment loss — wet-surface scattering,
spray, antenna wetting — which is what actually shrinks measured V2V range).
The empirical terms are currently **placeholders marked `[ASSUMED]`** and need
a citation. Reporting them as ITU-R rain attenuation would be wrong.

Weather also changes driver behaviour (`speed_factor`, `headway_factor`), which
changes network topology independently of the channel.

**2. Path loss is dual-slope, not single-slope.** A single *n* = 1.9 highway
exponent extrapolated over kilometres predicts a ~2.1 km DSRC range, ~4× the
measured value — which would make a 10 km corridor fully connected and delete
the sparse regime this paper is about. The model uses a ground-reflection
breakpoint (*n*₁ = 1.9 below 150 m, *n*₂ = 3.8 above), giving a nominal range of
**562 m**. `tests/test_phy.py::test_single_slope_would_overestimate_range`
guards this.

---

## The baselines (Phase 4)

Nine registered policies covering the brief's seven schemes (p-persistence
contributes three). Parameters and references are in `configs/policies.yaml`;
suppression logic is pinned by `tests/test_policies.py`.

| name | scheme |
|---|---|
| `flooding` | blind flooding |
| `p_persistence_03/05/07` | probabilistic p-persistence |
| `slotted_1p` | slotted 1-persistence (distance-ranked slots) |
| `weighted_p` | weighted p-persistence, `p = D/R` |
| `counter_based` | distance-based counter scheme |
| `greedy_farthest` | sender-side greedy relay designation |
| `dvcast` | DV-CAST-style store-carry-forward |

Two were deliberately implemented at full strength rather than as straw men:
`greedy_farthest` seeds both propagation directions and has an implicit-ACK
fallback (a designation over unacknowledged broadcast can simply be lost, and
without recovery the baseline would fail for a reason unrelated to relay
selection); `dvcast` uses oncoming traffic as carriers.

`python -m experiments.compare` runs them **paired** — seed *k* gives every
policy byte-identical mobility, hazard placement, shadowing and fading — and
appends to `results/runs.csv`. Paired samples are what Phase 7's Wilcoxon
signed-rank test requires.

### The density sweep in the brief misses the paper's own regime

Measured, not assumed. The engine records which action each policy chose
(`n_action_carry`, `n_action_defer`, …). DV-CAST's store-carry-forward branch
fires **0.0 times per run at 5, 10, 20, 40, 80 and 120 veh/km/lane**: with a
562 m nominal range on a two-way corridor, every one of the brief's densities
is a *connected* network. All schemes degenerate to slotted persistence and
RWCR saturates at 0.92–1.00 across the board, so the headline metric stops
discriminating.

The disconnected regime starts below ~3 veh/km/lane. At 2 veh/km/lane the carry
branch finally fires and DV-CAST reaches RWCR 0.666 ± 0.156 against flooding's
0.607 ± 0.190 and slotted 1-persistence's 0.589 ± 0.144 — with fewer
transmissions than either. At 1 veh/km/lane everything collapses (RWCR < 0.23):
the corridor is fragmented beyond what a 120 s window can bridge.

`configs/experiment.yaml` therefore prepends densities 1, 2 and 3 to the sweep
and keeps the brief's six. **If the paper claims a sparse-network contribution,
the evidence has to come from ≤3 veh/km/lane**, or from a configuration with a
shorter effective range.

---

## The risk field (Phase 3)

`relevance(v, h, t) = g_dir · g_geom · w_η(η) · severity(t)^γ`, all parameters in
`configs/hazard.yaml`. A vehicle driving away from a landslide scores 0; a truck
40 s upstream of a fog bank scores 1. Full equation in the
`hazard/risk_field.py` module docstring; the behavioural claims are pinned by
`tests/test_risk_field.py`.

Grid scenarios use an **approximate** route-unaware geometry (Manhattan distance
+ bearing gate), because a vehicle's future turns are unknown. It errs toward
counting vehicles as at-risk, so grid RWCR is under-stated. Labelled wherever
grid results appear.

---

## Metrics (Phase 6)

`pdr`, **`rwcr`**, **`tir_median_s` / `tir_p95_s` / `tir_uninformed_frac`**,
`latency_mean_s`/`p95`, `redundancy_ratio`, `collisions_per_delivered`,
`deadline_miss_rate`, `max_hops`, `spatial_reach_m`, plus channel diagnostics.

RWCR and TIR are defined with their equations in `analysis/metrics.py`
docstrings. Two properties of RWCR worth knowing: informing a vehicle
*contributes nothing* if the message arrives after it has passed the hazard, and
reaching irrelevant vehicles does not raise the score at all.

---

## Known limitations

- All current results use the fallback mobility backend (see above).
- `[VERIFY]` constants are recalled, not looked up: 802.11p EDCA/timing values,
  the ITU-R P.838/P.840 coefficients, the dual-slope fit, the DCC target CBR.
- Empirical weather excess-loss values are placeholders needing citations.
- At 120 veh/km/lane the corridor is **jammed**, not free-flowing (mean speed
  ~2.5 m/s): that density is at/above jam density for a fleet with 22% trucks.
  It is a legitimate congested regime but must be labelled as one, not
  presented as high-density free flow.
- Background CAM/BSM load is modelled as a Poisson arrival process at the
  receiver with ETSI-style DCC rate adaptation, not as individually simulated
  beacon frames.
- One originator per hazard (`max_originators: 1`); redundant sensing would
  flatter every policy.
- **Deferring schemes pay a 100 ms slot granularity.** The engine's decision
  epoch is 100 ms, but a real slotted scheme uses an estimated one-hop delay of
  a few milliseconds. This inflates absolute TIR for every scheme that defers
  (`slotted_1p`, `counter_based`, `greedy_farthest`, `dvcast`) and gives them
  more time to overhear duplicates, so better suppression. It applies equally
  to the baselines and to the agent's defer action, so internal comparisons
  stay fair — but absolute latencies for deferring schemes are pessimistic and
  the paper must say so. `slot_epochs` in `configs/policies.yaml` is exposed
  for a sensitivity check.
- **The deadline-miss rate barely discriminates** (0.27–0.39 across all
  policies at most densities). It is dominated by vehicles already inside or
  within reaction time of the hazard when it is first detected, which no
  dissemination policy can affect. Either report it with that caveat or
  restrict it to vehicles that were still outside the reaction-time envelope at
  origination.
- **The grid risk field makes urban RWCR uninterpretable. This currently blocks
  `urban_nlos` as a primary scenario.** Measured on `urban_nlos` at
  20 veh/km/lane: *100% of vehicles are informed*, yet RWCR is 0.291 and
  at-risk coverage is 0.277. The cause is the route-unaware grid geometry in
  `hazard/risk_field.py`: it marks a vehicle `APPROACHING` only if its
  *instantaneous* heading reduces Manhattan distance to the hazard. Grid
  vehicles turn constantly, so ~70% happen to be heading away at the moment
  they receive the message, score relevance exactly 0, and are counted as "not
  informed in time" despite being informed and later driving into the hazard.
  The at-risk set also swells to 97.7% of the network. RWCR in the grid
  therefore measures "was this vehicle pointing at the hazard when the packet
  arrived", not "was an at-risk vehicle warned".
  This is invariant to the radio model — sweeping NLOS corner loss over
  8/12/20 dB (NLOS range 219/176/101 m) leaves RWCR pinned at 0.28-0.30.
  **Fix before any urban result is reported:** define the grid at-risk set from
  each vehicle's *realised* trajectory (does it actually enter the hazard span
  during the run). That is legitimate for an evaluation-time ground-truth
  quantity, and must stay unavailable to the policies, which see only local
  observations. `Trace.routes` already records what is needed.
- **Weather is a traffic-mediated effect only, and is a SECONDARY result.**
  There is no channel-degradation claim anywhere in this project: the ITU-R
  hydrometeor terms are real but negligible at 5.9 GHz (worst case 0.24% change
  in nominal range), and the empirical excess-loss term was deleted for want of
  a citation. Weather changes dissemination by changing how people drive
  (`speed_factor`, `headway_factor`), which changes spacing and topology.
  Every weather figure caption must say "traffic-mediated"; no figure, table or
  caption may imply the radio channel degrades. The strongest honest framing is
  that fog destroys *optical* sensing while leaving 5.9 GHz untouched, which is
  exactly why V2X warning matters most in fog.
- `reproduce.sh` is Phase 8 and does not exist yet.

## Reproducing each figure

Pending Phase 8. Each figure will be listed here with the exact command that
regenerates it from `results/runs.csv`.
