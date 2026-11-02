# Revision 6: validation of the learned-policy bridge (2026-11-02)

**Verdict: pass (all three parts).** The OMNeT++ app sends each decision request to a decision server running the
unchanged agent code in the pinned interpreter (`experiments/revision6/veins/decision_server.py`); since WSL cannot
open connections to Windows on this laptop (Windows firewall), each worker's server dials a byte-copying broker inside
WSL (`bridge_broker.py`). App binary `5d92be2a…` (final, after D9).

## 1. Unit check — pass

`port_validation_policy_unit.json` (`python -m experiments.revision6.veins.validate_policy unit`). 1,000 decision
contexts recorded in the custom simulator (ref_s0, bias +1, τ = 0.5, rural d=20 on SUMO mobility; seeds 200 / 201 /
202 contributed 419 / 376 / 205), each serialised exactly as the app serialises its request and rebuilt by the
server's `ctx_from_request`. Action distribution (network probabilities after the suppression bias) and gate verdict
from the unchanged policy code, in-process vs through the server:

* max abs difference over all actions and contexts: **0.0** (criterion 1e-6); gate-verdict mismatches: **0**.

History: the first run recorded only 419 contexts (one replay; more seeds were then used to reach 1,000); the second
gave 1.03 × 10⁻⁶ (> 1e-6). Diagnosis: the graphs were not bit-identical because the custom engine passes neighbour
distances, dx, dy, vx, vy as float32 (the trace's dtype) and `agents/graph.py` then computes partly in float32, while
the server had rebuilt them as float64. The server now rebuilds those five arrays with the engine's dtypes; the
network itself is deterministic (re-running an identical graph differs by 0.0).

## 2. Trace check — pass

`port_validation_policy_trace.json` (`... validate_policy trace <log>`). Veins replay of ref_s0 at its revision-5 SEL
bias in rural d=20 (+1), seed 200; exit code 0, exact trace, 317 s wall. All **2,223** requests logged by the server
were re-sent offline, in order, to a fresh session with the logged configuration: **0** mismatches (every action,
delay, cancel count and relay target reproduced). Randomness is per request, from streams named by (replay seed,
vehicle, epoch, k), k = the vehicle's decision count within the epoch.

## 3. Sanity check (no tuning in response)

Same replay, scored with the revision-5 Veins scorer's functions (`veins6_score.py`), next to the custom simulator's
committed row for the same checkpoint, bias and seed (`results/revision5/runs/agent__EVAL__headline__default__sumo.csv`):

| | RWCR | cost (tx / at-risk informed) | actionable miss | transmissions | action mix (decisions) |
|---|---|---|---|---|---|
| Veins | 0.780 | 1.262 | 0.448 | 183 | suppress 1,934, defer 129, broadcast 90, relay 53, carry 17 |
| custom | 0.836 | 0.693 | 0.378 | 113 | suppress 474, defer 70, carry 45, broadcast 30, relay 16 |

The Veins replay makes many more decisions (2,223 vs 635: more duplicate receptions, each a callback for this
policy, mostly settled suppresses), transmits more and reaches fewer at-risk vehicles; neighbour tables, reception and
timing differ between the simulators.

**Inputs checked against the trace** (requests vs the custom trace one epoch earlier, the measured coupling lag):
own position median |Δ| 2.6 mm, own velocity 2.4 mm/s, heading 9 × 10⁻⁸ rad; neighbour dx median 0.86 m (beacon
age × speed), neighbour velocity 0.04 m/s, neighbour relevance p95 0.003.

## D9 — found by this validation

The first validation replay's neighbour velocities were all zero (median |Δvx| 19.4 m/s): `omnetpp.ini` sets
`veinsmobility.setHostSpeed = false`, so Veins never sets the mobility speed and every beacon carries (0, 0). The
revision-6 app now sends the CAM velocity (TraCI speed and heading) in beacons for the new policy kinds only; the
configuration is unchanged. Recorded as D9 in `results/revision6/PREREGISTRATION_ADDENDUM.md` before any reported run,
with what had been seen (that replay's sanity numbers: RWCR 0.461, cost 1.47). Pre-D9 runs are kept in
`cache/veins_runs/r6_validation_pre_D9/`. The revision-5 greedy_farthest and DV-CAST Veins ports ran with the same zero
beacon velocities; reported in the revision-6 report, revision-5 results unchanged.
