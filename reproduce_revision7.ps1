# Revision 7: regenerate results/revision7/analysis/ (Experiments A and B) from committed rows.
#
#   powershell -ExecutionPolicy Bypass -File reproduce_revision7.ps1                 analysis only (about 10 minutes)
#   powershell -ExecutionPolicy Bypass -File reproduce_revision7.ps1 --from-scratch  rerun validation, replays and runs first
#
# Environment: the pinned interpreter only (NumPy 1.26.4, torch 2.2.2). Set AIHARP_PY to override its location.
# --from-scratch needs WSL Ubuntu with Veins 5.3, OMNeT++ 6.0.3, SUMO 1.19.0, the revision-7 app (~/veins-r7/aiharp, built by
# experiments/revision7/veins/build_app.sh; sha256 in results/revision7/veins/app_gate.json), the revision-5 bundles in
# cache/veins and the revision-6 r6.txt files (experiments.revision6.veins.r6_bundle). It MOVES the scored rows and
# cache/veins_runs/r7_* to *.bak-<timestamp>; nothing is deleted. The original took about 9 elapsed hours on one laptop
# (Experiment A in Veins 6.5 h on 8 workers, Experiment B about 2.5 h on 11). The committed raw OMNeT++ outputs are in
# results/revision7/veins/raw/.
param([switch]$FromScratch)
if ($args -contains "--from-scratch") { $FromScratch = $true }
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = if ($env:AIHARP_PY) { $env:AIHARP_PY } else { "D:\aiharp-gpu\Scripts\python.exe" }
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue   # backend "auto" must not pick SUMO by accident

$envline = & $py -c "import sys, numpy; print(sys.executable, numpy.__version__)"
Write-Host "[env] $envline"
if ($envline -notmatch " 1\.26\.4$") { throw "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" }

function Run([string]$a) {
    Write-Host ">> python -m $a"
    & $py -u -m $a.Split(" "); if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $a" }
}

if ($FromScratch) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($p in "results\revision7\veins\capped_rows.jsonl", "results\revision7\veins\capped_rows_merge.json",
                   "results\revision7\veins\d9_rows.jsonl", "results\revision7\veins\d9_rows_merge.json",
                   "results\revision7\runs\custom_capped_rows.jsonl", "results\revision7\runs\custom_capped_meta.json",
                   "cache\veins_runs\r7_gate", "cache\veins_runs\r7_gate_capped", "cache\veins_runs\r7_capped", "cache\veins_runs\r7_d9") {
        if (Test-Path $p) { Move-Item $p "$p.bak-$ts"; Write-Host "moved $p -> $p.bak-$ts" }
    }
    Run "experiments.revision7.veins.validate7 unit"
    Run "experiments.revision7.veins.validate7 gate"
    Run "experiments.revision7.veins.queue7 capped --jobs 8"
    Run "experiments.revision7.custom_cap --jobs 3"
    Run "experiments.revision7.veins.queue7 d9 --jobs 11"
    Run "experiments.revision7.veins.score7 capped"
    Run "experiments.revision7.veins.score7 d9"
}

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
Run "experiments.revision7.analysis.exp_a"
Run "experiments.revision7.analysis.exp_b --B 10000"
Run "experiments.revision7.manifest"
Write-Host "done: results/revision7/analysis, results/MANIFEST.md, results/manifest.json"
