# INVALID — superseded R9 outputs (kept as a record)

These were produced while the calibrated-loss knob drew its random numbers from the
engine's `mac` stream. Enabling the knob therefore shifted every later MAC draw, so cells
where the extra loss is exactly zero (rural d=2, d=20: < 250 neighbours in carrier-sense
range) changed through RNG drift alone (e.g. the agent's cells matched went 1/1/3/3/3 ->
1/1/1/1/1). Fixed by giving the knob its own named stream (`common/seeding.py` "extra_loss");
after the fix those cells are bit-identical with the knob on. Do not use these numbers.
