# Revision 8: time estimate, written before any training run (2026-11-16)

Machine: cloud container, 4 vCPU (`nproc` = 4), `/root/r8env` (Python 3.12, NumPy 1.26.4).

Timed smoke (`python -m agents.train_structured --arm plain --seed 0 --generations 2 --workers 2 --out <scratch>`),
while SUMO trace generation was using all 4 cores at the same time: 26.9–31.8 s per generation (13 candidates × 12
episodes = 156 episodes). Plain engine episodes with cached traces (one core, no contention): rural d=2 0.1 s, d=20 0.2 s,
urban NLOS d=20 0.3 s, rural d=80 2.5 s, i.e. about 9.3 s of CPU per candidate-batch and ≈ 121 CPU-s per generation.

Estimate with 4 workers and no contention: ≈ 121 / 4 ≈ 30 s → **15–35 s per generation**, so **20–47 min per
80-generation run**. STRUCT-rand adds SUMO episodes (same per-episode cost with cached traces) and R5/R2 draws: assume
up to 1.5×. **Total for 20 runs: about 8–20 wall-hours.** Custom evaluation (sweep_struct): 2 arms × 10 runs × 6 δ ×
(20 SEL + 30 EVAL) × 4 cells = 24,000 runs at ≈ 0.8 s mean CPU → ≈ 5.3 CPU-h ≈ 1.5 wall-hours on 4 cores.

If the session ends first, every run stops at its last per-generation checkpoint and resumes exactly
(`python -m agents.train_structured --arm <arm> --seed <k>`, verified on the smoke: an interrupted-then-resumed run
gives the identical history and θ).
