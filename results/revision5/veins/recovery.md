# Recovered scalar file (2026-10-31)

`rural_highway_d80_s221__counter_based_1__s221` (replayed on the Azure VM `aiharp-ks`, merged with its run record and
event log) arrived without its OMNeT++ scalar file (`.sca.gz`, the input of the channel-busy-ratio measurement). The VM
was no longer reachable. The replay was regenerated on the laptop (WSL, same Veins build): the bundle was re-exported
(canonical-trace sha256 and originator identical to the VM's bundle), the replay re-run (exit code 0, identical SUMO
trace), and its event log compared with the merged VM log: **byte-identical (492,976 bytes uncompressed)**. Only the
regenerated scalar file was then placed beside the merged replay; the merged run record and event log are unchanged.
Regenerated copy kept in `cache/veins_runs/recover/` (wall 5,424 s, host LAPTOP-0QCG7GBR).
