# Quarantined: SEL sweeps that ran on SUMO mobility (invalid)

`classical__SEL__grid24__default__fallback.csv` (17,105 rows) and `agent__SEL__grid24__default__fallback.csv`
(9,072 rows) were produced 2026-10-19 00:31–09:56 IST by the "fallback" track, but every row has
`backend = sumo`: on 2026-10-18 ~23:10 SUMO_HOME was set as a persistent user variable (for the Veins/SUMO
work), the runner's backend was "auto", and "auto" picks SUMO whenever SUMO_HOME is set. Processes started
after that inherited it (the SEL sweeps, launched by the watchdog at 00:31); the training lanes (started
21:31, never relaunched) did not — verified from their live environments.

CAL (13,200 rows, all `backend = fallback`) and thresholds.json are unaffected.

Detected by the rule-7 bit-exact gate (it also ran "auto"). Fix: the protocol runner and the gate pass the
backend explicitly ("fallback" / "sumo"), the watchdog strips SUMO_HOME from its environment, and the user
variable was removed. The SEL sweeps restart from zero on fallback. These rows are kept, not deleted, and
are not used anywhere; no result was computed from them.
