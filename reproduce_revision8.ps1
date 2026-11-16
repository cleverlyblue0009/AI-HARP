# Revision 8: regenerate results/revision8/analysis/ from committed rows (results/revision8/ADDENDUM.md).
#
#   powershell -ExecutionPolicy Bypass -File reproduce_revision8.ps1                 analysis only
#   powershell -ExecutionPolicy Bypass -File reproduce_revision8.ps1 --from-scratch  retrain, re-evaluate, Veins gate/replays/scoring first
#
# Interpreter: NumPy 1.26.4 only (AIHARP_PY, default D:\aiharp-gpu\Scripts\python.exe). SUMO 1.19.0 for training and the
# STRUCT bridge checks: AIHARP_SUMO_HOME (default D:\sumo\sumo-1.19.0), set as SUMO_HOME only for those steps.
# --from-scratch needs WSL Ubuntu with Veins 5.3, OMNeT++ 6.0.3, the revision-8 app (~/veins-r8/aiharp, built by
# experiments/revision8/veins/build_app.sh), the revision-5 bundles in cache\veins and the r6.txt files for seeds 200-229.
# It MOVES existing outputs to *.bak-<timestamp>; nothing is deleted. Times: results/revision8/LAPTOP_RUNBOOK.md.
param([switch]$FromScratch)
if ($args -contains "--from-scratch") { $FromScratch = $true }
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = if ($env:AIHARP_PY) { $env:AIHARP_PY } else { "D:\aiharp-gpu\Scripts\python.exe" }
$sumo = if ($env:AIHARP_SUMO_HOME) { $env:AIHARP_SUMO_HOME } else { "D:\sumo\sumo-1.19.0" }
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
$envline = & $py -c "import sys, numpy; print(sys.executable, numpy.__version__)"
Write-Host "[env] $envline"
if ($envline -notmatch " 1\.26\.4$") { throw "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" }
function Run([string]$a, [switch]$Sumo) {
    Write-Host ">> python -m $a"
    if ($Sumo) { $env:SUMO_HOME = $sumo }
    & $py -u -m $a.Split(" "); $rc = $LASTEXITCODE
    Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
    if ($rc -ne 0) { throw "failed ($rc): $a" }
}
if ($FromScratch) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($p in "results\revision8\train", "results\revision8\runs", "results\revision8\veins\h5A_rows.jsonl",
                   "results\revision8\veins\h5B_rows.jsonl", "results\revision8\veins\expC_rows.jsonl", "results\revision8\veins\expD_rows.jsonl",
                   "cache\veins_runs\r8_gate", "cache\veins_runs\r8_h5A", "cache\veins_runs\r8_h5B", "cache\veins_runs\r8_expC", "cache\veins_runs\r8_expD") {
        if (Test-Path $p) { Move-Item $p "$p.bak-$ts"; Write-Host "moved $p -> $p.bak-$ts" }
    }
    Run "experiments.revision8.warm_start"
    Run "experiments.revision8.veins_thresholds"
    Run "experiments.revision8.traces --backend sumo" -Sumo
    Run "experiments.revision8.traces --backend fallback"
    foreach ($arm in "plain", "rand") { foreach ($k in 0..9) {
        if (-not (Test-Path "results\revision8\train\${arm}_s$k\theta.json")) { Run "agents.train_structured --arm $arm --seed $k" -Sumo } } }
    foreach ($arm in "plain", "rand") { foreach ($ss in "SEL", "EVAL") { Run "experiments.revision8.sweep_struct --arm $arm --seedset $ss --jobs 4" } }
    Run "experiments.revision8.veins.validate8 unit --run plain_s0" -Sumo
    Run "experiments.revision8.veins.validate8 unit --run rand_s0" -Sumo
    Run "experiments.revision8.veins.validate8 geometric" -Sumo
    Run "experiments.revision8.veins.queue8 gate" -Sumo
    foreach ($m in "h5B", "h5A", "expC", "expD") { Run "experiments.revision8.veins.queue8 $m --jobs 8" -Sumo; Run "experiments.revision8.veins.score8 $m" -Sumo }
}
# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
if (Test-Path "results\revision8\runs\struct_plain__EVAL__headline__default__fallback.csv") { Run "experiments.revision8.analysis.h4 --B 10000" }
if (Test-Path "results\revision8\runs\struct_rand__EVAL__headline__default__fallback.csv") { Run "experiments.revision8.analysis.h4 --B 10000 --arm rand" }
if (Test-Path "results\revision8\veins\h5B_rows.jsonl") { Run "experiments.revision8.analysis.h5 h5" }
Run "experiments.revision8.analysis.h5 expcd"
if ((Test-Path "results\revision8\analysis\h4.json") -and (Test-Path "results\revision8\analysis\h5.json")) { Run "experiments.revision8.analysis.h5 holm" }
Write-Host "done: results/revision8/analysis"
