# H3: validation of the greedy_farthest and DV-CAST Veins ports (2026-10-19)

One replay each against the custom simulator on the identical SUMO trace (seed 200; Veins RNG seed 200).
Scored with analysis.metrics via experiments/revision5/veins/veins_metrics.py. Trace identity: exact in both.

| policy (knob) | cell | simulator | RWCR | tx / at-risk informed | actionable miss | TIR median / p95 (s) | PDR | transmissions |
|---|---|---|---|---|---|---|---|---|
| dvcast (n_slots 5) | rural d=2 | Veins | 0.1059 | 1.000 | 0.9615 | 0.1 / 0.1 | 1.000 | 2 |
| dvcast (n_slots 5) | rural d=2 | custom | 0.1059 | 1.000 | 0.9615 | 0.1 / 0.1 | 1.000 | 2 |
| greedy_farthest (fallback_n_slots 4) | rural d=20 | Veins | 0.8326 | 0.319 | 0.3784 | 1.3 / 2.0 | 0.591 | 52 |
| greedy_farthest (fallback_n_slots 4) | rural d=20 | custom | 0.8628 | 0.568 | 0.3552 | 0.9 / 1.5 | 0.416 | 96 |

DV-CAST: same originator, same 4 informed vehicles, same 2 transmissions; decision mix differs slightly
(Veins 1 carry / 2 defer, custom 2 carry / 1 defer), consistent with beacon- vs geometry-based neighbour tables.

Greedy: the port applies the rule as specified -- the originator designated the farthest neighbour on each side
(vehicles 395 and 55; 618.9 m and 592.8 m in the trace, 395 within 610 m by its beacon-extrapolated position).
Neither decoded the origin's frame in Veins (45 vehicles did, out to 583 m), so the chain fell back to the
distance-ranked fallback timers for the rest of the run (407 defers, 356 cancelled on a duplicate). The
difference is edge-of-range reception, a physical-layer difference between the simulators -- the kind of
difference H3 measures -- not a porting error. Nothing was tuned in response.

Port history: the first greedy replay used every beacon-table entry; the DecisionContext contract restricts
neighbours to the nominal range, so the filter was added and the replay re-run (metrics unchanged: the
designated relays were inside the nominal range by their table positions in both versions). A third replay
added relay ids to the transmit log line to confirm the picks; metrics again unchanged.
