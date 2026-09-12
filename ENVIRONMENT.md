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

## SUMO

Still not installed. Every result to date uses the pure-Python fallback
mobility backend, which is logged as a banner on every run and stamped into
`trace.backend`. See README "Which mobility backend is running".
