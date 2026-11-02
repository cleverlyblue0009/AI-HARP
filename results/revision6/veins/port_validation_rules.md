# Revision 6: validation of the RAR and SCF-slotted Veins ports (2026-11-02)

**Verdict: pass.** Machine-readable: `port_validation_rules.json` (regenerate:
`python -m experiments.revision6.veins.validate_rules`). Settings fixed before any replay: RAR abc = (1.0, 0.5, 0.0),
SCF-slotted (8 slots, 5 s); rural highway d=20, seed 200, identical SUMO trace in both simulators (Veins trace check:
exact, 84,956 FCD rows). App binary `5d92be2a…` (the final one, after D9), which passed the identity gate
(`app_gate.json`).

| Check | Result |
|---|---|
| Binary-identity gate (classical slotted_1p n=8 replay, revision-6 vs revision-5 binary vs stored H3 replay) | event logs byte-identical (sha256 of the uncompressed log `707d6533…`, 2,468 lines) |
| RAR: ρ_c recomputed by `hazard/risk_field.py` from the inputs the app used | max abs difference **0.0** over 269 decisions (criterion 1e-9) |
| RAR: decision recomputed by `RelevanceAwareRule` from the logged ρ_c, neighbour count, sender distance and uniform draw | **0** mismatches in 269 (action and delay) |
| SCF: decision recomputed by `StoreCarryForwardSlotted` from the logged trigger, neighbour count, time and carry start | **0** mismatches in 424 |

**Measured against the custom simulator's own values (not a pass criterion, reported as found):**
* ρ_c vs the custom trace at the same (vehicle, epoch): median |Δ| 0.0, max 0.0052. Cause, measured: in all 269 RAR
  decisions the Veins mobility state at a decision in epoch k equals the custom trace at epoch k − 1 (to SUMO's
  printed precision) — a one-epoch lag of the Veins–SUMO coupling as configured for the classical replays (not
  changed, rule 6). Vehicles on the relevance plateau or outside the hazard's reach have identical ρ_c.
* Neighbour counts: the beacon-derived count equals the custom simulator's geometric count in 7 of 269 RAR
  decisions; where they are equal, the rebroadcast probability from the trace-based ρ_c differs by at most 0.0022.
  The neighbour tables differ, as expected (beacon reception vs trace geometry).

**History (all runs kept):** a first identity-gate run was invalid (both replays shared a working directory; fixed);
the app was rebuilt to log decision inputs at 17 significant digits; after D9 (beacon velocity, see the addendum) the
gate and both rule replays were re-run on the final binary — the numbers above are from those runs. The pre-D9 runs
are kept in `cache/veins_runs/r6_validation_pre_D9/` and gave the same rule results (ρ_c 0.0, 0 mismatches).
