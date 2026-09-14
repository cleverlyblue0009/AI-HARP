# Environment

## The Python environment lives on D:, not C:

```
D:\aiharp-env\python.exe
```

Run everything through it:

```bash
D:/aiharp-env/python.exe -m pytest tests/ -q
D:/aiharp-env/python.exe -m experiments.run_sim --smoke
D:/aiharp-env/python.exe -m analysis.comparator --reuse
```

### Why

`C:` was at 100% (438 MB free of 308 GB). `conda install pytorch` **reported
exit 0 and installed nothing** — the real error was buried in its output:

```
InvalidArchiveError ... [Errno 28] No space left on device
```

It downloaded 150 MB, failed to extract, and exited clean. Anything that
appears to install successfully on a full disk should be verified by importing
it, not by trusting the exit code.

Roughly 800 MB was reclaimed from conda's own caches
(`conda clean --tarballs --index-cache --packages`), and that clean also had to
delete a corrupted `colorama` package left behind by the failed extraction
before a new environment could be created. `C:` now has ~9 GB free, which is
still tight for a 308 GB disk.

### Contents

| package | version | notes |
|---|---|---|
| python | 3.12 | |
| numpy | **1.26.4** | pinned `<2` deliberately — see below |
| torch | 2.2.2 | CPU build |
| torch-geometric | 2.8.0 | `GATv2Conv` |
| tensorboard | latest | |
| pandas, scipy, matplotlib, pyyaml, pytest | | |

**NumPy is pinned below 2.0.** torch 2.2.2 is compiled against the NumPy 1.x C
API; with NumPy 2.x installed it imports but emits
`Failed to initialize NumPy: _ARRAY_API not found` and the tensor/array bridge
is silently degraded. That is a quiet-wrong failure mode, so the pin is load
bearing. If torch is upgraded past 2.3, the pin can be lifted.

`pip` works inside this environment. The `pip` in the base `C:` miniconda is
broken (its vendored `cachecontrol/caches/` directory is empty), which is why
earlier installs had to go through conda.

### The base C: environment still works

`C:\Users\UPASANA\miniconda3\python.exe` still runs Phases 1–4 and 6 — numpy,
pandas, scipy, matplotlib, pyyaml, pytest are all present. It has no torch, so
Phase 5 (`agents/gat_drl.py`, `agents/train.py`) will not import there. Every
Phase 1–4 test is torch-free by design, so the suite passes under either
interpreter; only the Phase 5 tests require D:.

## Storage

- `cache/traces/*.npz` — generated mobility traces, regenerable, git-ignored.
  Currently ~430 MB. Delete freely if `C:` gets tight again.
- `results/` — CSVs and figures; `runs.csv` and `pareto_cells.json` are the
  evidence base and **are** committed.
- `checkpoints/`, `tb_logs/` — Phase 5 training artefacts, git-ignored. These
  will grow; they are written under the project on `C:` and should be moved to
  `D:` if space becomes a problem again.

## The GPU environment (D:\aiharp-gpu)

A second environment with a CUDA build of PyTorch, used only to run the PPO
step on the laptop's NVIDIA GeForce RTX 4050 (6 GB). `D:\aiharp-env` stays the
CPU-only reference environment for tests and CPU runs.

```bash
D:/aiharp-env/python.exe -m venv D:/aiharp-gpu
D:/aiharp-gpu/Scripts/python.exe -m pip install "numpy<2" torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
D:/aiharp-gpu/Scripts/python.exe -m pip install torch-geometric==2.8.0 pyyaml pandas "scipy==1.13.1"
D:/aiharp-gpu/Scripts/python.exe -m pip install "numpy<2"     # torch-geometric's deps pull NumPy 2
```

Two pins are load-bearing, and both were hit: installing torch-geometric pulled
NumPy 2.5.3 (torch 2.2.2 then prints `Failed to initialize NumPy`, the silent
degradation described below), and SciPy 1.18 requires NumPy >= 2, so SciPy is
held at 1.13.1.

Train with it exactly as with the CPU environment; `training.device: auto`
picks the GPU for the PPO step, and rollouts stay on CPU workers:

```bash
D:/aiharp-gpu/Scripts/python.exe -m agents.train --out checkpoints/run8 --keep-awake
```

**Determinism costs speed on the GPU.** Measured PPO step on 6,222 transitions
(`experiments/bench_ppo_device.py`): CPU 13.87 s, GPU deterministic 7.51 s,
GPU non-deterministic 3.13 s. `training.cuda_deterministic: true` (default)
keeps runs bit-reproducible, as the rule that no number in `results/` comes from
a non-reproducible run requires. `--nondeterministic-gpu` is for exploratory
runs only; `run_summary.json` records `bit_reproducible: false`. A GPU run is
never bit-identical to a CPU run (weights differ by ~1.6e-6 after 16 steps);
exact resume is tested on CPU.

## Training on a many-core cloud machine

Training throughput is CPU-bound (see `results/profile_baseline.txt`), so the
cheapest real speed-up is more cores for a few hours. On a fresh Linux box:

```bash
git clone <repo> AI-HARP && cd AI-HARP
python3.12 -m venv .venv && . .venv/bin/activate
pip install "numpy<2" torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.8.0 pandas scipy matplotlib pyyaml pytest tensorboard

# 1. Warm the trace cache ONCE, before training. Otherwise every rollout
#    worker regenerates the same traces during the first updates.
python -m experiments.warm_traces --jobs "$(nproc)"

# 2. Correctness gate before any long run (includes batched-vs-unbatched and
#    worker-count identity tests).
python -m pytest tests/ -q

# 3. Measure where scaling flattens on THIS machine.
python -m experiments.bench_rollouts --workers 1 8 16 32 --episodes 32

# 4. The full two-stage run (pretrain 200 dense updates, then sparse-weighted
#    finetune up to 800 more, with early stopping on the constraints).
python -m agents.train --out checkpoints/run8 --workers auto
```

`rollout_workers: auto` means `os.cpu_count() - 1`. **Workers beyond
`training.rollout_episodes_per_update` (8) are idle**: one episode runs in one
worker, and raising episodes per update changes each PPO batch, which is an
optimisation change rather than a free speed-up. A 64-vCPU machine therefore
does not train 8x faster than a 12-thread laptop unless that setting is
deliberately changed and justified.

**Splitting a run across sessions.** Every update rewrites
`ckpt_latest.pt` with the model, optimiser, per-group multipliers, both RNG
streams, the curriculum stage and the early-stop state, so a resumed run is
identical to an uninterrupted one (`tests/test_curriculum_resume.py`):

```bash
python -m agents.train --resume checkpoints/run8/ckpt_latest.pt
```

Copy `checkpoints/run8/` (and, to skip regeneration, `cache/traces/`) between
machines. A resumed run keeps the configuration stored in its checkpoint.

**Re-running stage 2 only**, e.g. with a different sparse weighting, from the
stage boundary:

```bash
python -m agents.train --stage finetune --sparse-weight 5 \
    --init-from checkpoints/run8/ckpt_pretrain.pt --out checkpoints/run8_ft_w5
```

The run writes `run_summary.json` saying whether finetuning stopped early
(constraints held for 30 consecutive updates) or reached its update cap -- the
latter is reported as a finding, not silently extended.

## SUMO

**Eclipse SUMO 1.19.0 is installed at `D:\sumo-1.19.0`.** It is not on
conda-forge under that name (`conda install sumo` pulls an unrelated
materials-science package), so it was taken from the official Windows zip.

Set the environment variable before any run that needs it:

```bash
export SUMO_HOME="D:/sumo-1.19.0"
export PATH="$SUMO_HOME/bin:$PATH"
```

`mobility/sumo_runner.py` detects it via `SUMO_HOME` and logs a banner naming
the active backend. With `--backend sumo` a missing installation is a hard
error rather than a silent fallback.

Two bugs surfaced on first contact, both of which had been sitting in code that
had never executed:

1. `--fcd-output.period` does not exist; the FCD sampling period is
   `--device.fcd.period`. SUMO exited 1.
2. **The corridor was never filling.** SUMO injects vehicles at the boundary,
   whereas the fallback pre-places them, so recording cannot start until one
   vehicle has traversed the whole corridor. At the configured 20 s warm-up the
   10 km corridor needs ~457 s, and a 2 km test recorded 18.6 concurrent
   vehicles against an expected 76 -- silently, since nothing re-checked
   density. Warm-up is now `max(configured, transit_time x 1.5)` and
   `_check_density()` warns if the achieved density still misses by >25%.
   After the fix: commanded 20, achieved 20.8 veh/km/lane.
