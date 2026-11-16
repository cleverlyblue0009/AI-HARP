# Revision 8 — laptop runbook (Veins replays, Experiments C and D)

**Scope after D12 (`results/revision8/DEVIATIONS.md`, decided before any result):** H5 Tier B on seeds 210–219 (1,200
replays), no H5 Tier A, Experiments C and D in rural d=20 only (300 replays each). **Order for one day:** steps 0–4 and
then step 7 (Experiments C and D) can start **now**: they need no STRUCT training. Step 5 (H5 Tier B) starts when the cloud
has pushed `results/revision8/train/{plain,rand}_s{0..4}/theta.json` (expected 8–12 h from 05:30 UTC); step 3's two
`unit` checks also wait for those files (`geometric` does not). About 21 laptop wall-hours in total.

Everything Veins-side runs only on the laptop (Windows, `D:\aiharp-gpu`, WSL Ubuntu with Veins 5.3 / OMNeT++ 6.0.3 /
SUMO 1.19.0, the revision-5 bundles in `cache\veins`). **No Veins result is produced or approximated in the cloud.**
Pre-registration: `results/revision8/ADDENDUM.md` (push commit 53eefc3). Times below are estimates from revision-7
throughput (`results/revision7/REVISION7_REPORT.md`, "Execution record": 600 policy replays in 6.51 wall-hours on 8
workers, i.e. about 92 replays per hour for rural d=20 + urban NLOS d=20). STRUCT makes no duplicate callbacks, so
its bridge traffic is lower; the simulation itself dominates, so the same rate is assumed.

Interpreter everywhere: `D:\aiharp-gpu\Scripts\python.exe` (NumPy 1.26.4). Every job prints `[env]` with it.
`SUMO_HOME` is set **only** where a step says so (revision 5 D2: backend "auto" picks SUMO whenever it is set).

## 0. Preconditions (5 min)

```powershell
cd C:\Users\UPASANA\AI-HARP
git fetch origin; git checkout revision-8; git pull --ff-only
D:\aiharp-gpu\Scripts\python.exe -c "import sys, numpy, cma; print(sys.executable, numpy.__version__, cma.__version__)"
D:\aiharp-gpu\Scripts\python.exe -m pip install cma==4.6.0          # only if the line above fails on cma
```

Required in the checkout before step 1 (all committed from the cloud):
`results/revision8/train/{plain,rand}_s{0..4}/theta.json` (trained STRUCT parameters; H5 uses seeds 0–4 of each arm, fixed
in the addendum), `results/revision8/veins_thresholds.json` (frozen Veins q*, m*).

**Gate 0:** `D:\aiharp-gpu\Scripts\python.exe -m pytest tests/test_structured.py tests/test_randomisation_flags.py tests/test_oracle_isolation.py tests/test_revision8_analysis.py -q`
must pass (about 3 min). Output: none (console).

## 1. Bundles (about 30–45 min; regenerate if `cache\veins` is gone)

`cache\` is git-ignored, so a fresh clone has no bundles. Regenerate the ones revision 8 needs: rural d=20 and urban NLOS
d=20, seeds 200–219 (seed 200 is also the gate replay). `export` regenerates the SUMO trace with SUMO 1.19.0 and refuses
the bundle unless it matches the canonical trace; `r6_bundle` adds `r6.txt` and refuses if the custom run does not match
the bundle. Two windows in parallel:

```powershell
$env:SUMO_HOME = "D:\sumo\sumo-1.19.0"          # your SUMO 1.19.0 folder (the one revision 5-7 used)
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision5.veins.export rural_highway 20 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219
```
```powershell
$env:SUMO_HOME = "D:\sumo\sumo-1.19.0"
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision5.veins.export urban_nlos 20 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219
```
then, once both finish:
```powershell
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision6.veins.r6_bundle --seeds 200-219
Remove-Item Env:SUMO_HOME
```

Output: `cache\veins\{rural_highway_d20,urban_nlos_d20}_s{200..219}\` with `bundle.json` and `r6.txt` ("no bundle" lines
for rural d=2 / d=80 are expected and harmless). `queue8` refuses to start if any of them is missing. The gate (step 4)
then shows whether the regenerated bundles reproduce the revision-6/7 replays byte for byte (its references are the
committed logs in `results/revision7/veins/raw/r7_gate/`, sha256 `707d6533…` and `220dd1fd…`).

## 2. Build the revision-8 app (about 5 min)

```powershell
wsl -d Ubuntu -- bash /mnt/c/Users/UPASANA/AI-HARP/experiments/revision8/veins/build_app.sh
```

Output: `~/veins-r8/aiharp/aiharp` and its sha256 on the console. The revision-5/6/7 binaries are not touched.

## 3. Bridge unit checks, custom side (about 15 min)

```powershell
$env:SUMO_HOME = "D:\sumo\sumo-1.19.0"
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.validate8 unit --run plain_s0
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.validate8 unit --run rand_s0
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.validate8 geometric
Remove-Item Env:SUMO_HOME
```

Outputs: `results/revision8/veins/port_validation_unit_plain_s0.json`, `..._rand_s0.json`, `port_validation_geometric.json`.
**Gate 3:** each prints `"pass": true` (≥ 1,000 contexts; 0 decision and draw mismatches; for `geometric`, the rewritten
neighbour arrays equal the custom engine's in every context). The cloud already ran the same checks (committed files;
see `REVISION8_STATUS.md`); rerunning here checks the laptop interpreter.

## 4. Veins validation gate (about 1 h)

```powershell
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 gate            # checks 1, 2, 4, 5 (no STRUCT needed)
```

Replays at rural d=20, seed 200 (`cache\veins_runs\r8_gate\`), decision logs in `D:\aiharp-r8-bridge-logs\gate\`. Checks:
1. `slotted_1p` n=8, every new option off: event log byte-identical to the revision-6 gate (`707d6533…`);
2. AI-HARP ref_s0, bias +1, every option off: byte-identical to the revision-6 replay (`220dd1fd…` in `cache\veins_runs\r6`);
3. STRUCT plain_s0, δ 0: exit 0, identical trace, and every logged request re-sent offline gives the identical reply;
4. Exp C replay (`--send-heard`, geometric table): exit 0, identical trace, offline re-send identical;
5. Exp D replay (`--step-offset 1`): exit 0, identical trace (the FCD check uses the trace's own end).

Output: `results/revision8/veins/app_gate.json` **only if all pass** (with the binary's sha256); otherwise
`app_gate_FAILED.json`. Under D12, check 3 (STRUCT) runs separately as `queue8 gate_struct` once plain_s0 is
trained (`app_gate_struct.json`, same binary required); Experiments C/D need only `app_gate.json`, H5 needs both. **Gate 4: no replay mode starts without `app_gate.json`.** If it fails: record the error, at most
two fixes, then stop and report (revision-7 rule); any fix to the app or server is a deviation (D13+) logged before rerunning.

## 5. H5, Tier B — primary (about 13 wall-hours, 1,200 replays, 8 workers; seeds 210–219, D12)

```powershell
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 gate_struct      # check 3 of step 4, needs plain_s0 (~15 min)
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 h5B --jobs 8
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 status
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.score8 h5B
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.analysis.h5 h5
```

Resumable: rerun the same `queue8` line after any interruption; finished replays are skipped, a replay is accepted only
with exit code 0, an identical trace and the gated binary. Outputs: `cache\veins_runs\r8_h5B\*.{run.json,events.csv.gz,sca.gz}`,
`results/revision8/veins/h5B_rows.jsonl` (+ `_merge.json` with merged/rejected counts),
`results/revision8/analysis/h5.json`. **Do not change anything after reading h5.json**; a change is a deviation that says
the result had been seen.

## 6. H5, Tier A — NOT RUN under D12 (kept for a later revision: about 13 wall-hours, 1,200 replays)

```powershell
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 h5A --jobs 8
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.score8 h5A
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.analysis.h5 h5
```

Output: `h5A_rows.jsonl`; `h5.json` gains `tier_A_descriptive_hindsight`.

## 7. Experiments C and D (about 3.3 + 3.3 wall-hours, 300 replays each, rural d=20 only, D12) — can run first

```powershell
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 expC --jobs 8
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.queue8 expD --jobs 8
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.score8 expC
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.veins.score8 expD
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.analysis.h5 expcd
```

Needs the revision-5 checkpoints of ref_s0–ref_s4 (`checkpoints\revision1\…`, as in revision 7) and, for Exp C, the SUMO
trace cache of seeds 200–209 (the decision server loads it; it exists from revision 6/7). Output:
`results/revision8/analysis/exp_cd.json` with the verdict per experiment (rural d=20 ≥ 3/5 → "explains the failure").
Exp C's server rebuilds the neighbour table per request; if throughput falls far below 92/h, note it, do not cut.

## 8. Holm, manifest, commit (15 min)

```powershell
D:\aiharp-gpu\Scripts\python.exe -m experiments.revision8.analysis.h5 holm      # needs h4.json (cloud) and h5.json
powershell -ExecutionPolicy Bypass -File reproduce_revision8.ps1                  # analysis only, must exit 0
```

Commit: `results/revision8/veins/*_rows.jsonl`, `*_merge.json`, `app_gate.json`, `port_validation_*.json`,
`results/revision8/analysis/{h5,exp_cd,holm}.json`, and the raw OMNeT++ outputs (`.events.csv.gz`, `.sca.gz`, `.run.json`)
under `results/revision8/veins/raw/` if they total under 1 GB (revision-7 rule), else zip them for Zenodo and report the
size. The decision-server logs (`D:\aiharp-r8-bridge-logs`) are not committed.

## Total

Under D12: about 21 wall-hours (gate and checks about 1.5 h, Experiments C and D about 6.5 h, H5 Tier B about 13 h).
Recommended order: 0 → 1 → 2 → 3 (`geometric` only) → 4 → 7 → (when the ten STRUCT runs are pushed) 3 (`unit`) → 5 → 8.
The pre-registered full plan was about 55 wall-hours. If time is cut,
Tier B H5 (step 5) is the only Veins part the hypotheses need; Tier A and C/D are descriptive. Any cut is decided from
elapsed time only and logged as a deviation before the affected results are seen.
