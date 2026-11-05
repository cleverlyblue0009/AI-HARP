# Revision 6: regenerate results/revision6/analysis/ (plus the table and figure) from committed rows.
#
#   powershell -ExecutionPolicy Bypass -File reproduce_revision6.ps1                 analysis only (about 5 minutes)
#   powershell -ExecutionPolicy Bypass -File reproduce_revision6.ps1 --from-scratch  rerun the Veins replays first
#
# Environment: the pinned interpreter only (NumPy 1.26.4, torch 2.2.2). Set AIHARP_PY to override its location.
# --from-scratch needs WSL Ubuntu with Veins 5.3, OMNeT++ 6.0.3, SUMO 1.19.0 and the gated app (~/veins-r6/aiharp, built by
# experiments/revision6/veins/build_app.sh; sha256 in results/revision6/veins/app_gate.json), plus the revision-5 bundles
# in cache/veins. It MOVES the scored rows and cache/veins_runs/r6 to *.bak-<timestamp>; nothing is deleted. The original
# took about 75 elapsed hours on one laptop. The classical replays are the committed revision-5 rows and are not rerun.
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
    foreach ($p in "results\revision6\veins\veins6_rows.jsonl", "results\revision6\veins\veins6_rows_merge.json", "cache\veins_runs\r6") {
        if (Test-Path $p) { Move-Item $p "$p.bak-$ts"; Write-Host "moved $p -> $p.bak-$ts" }
    }
    Run "experiments.revision6.veins.r6_bundle"
    Run "experiments.revision6.veins.queue6 rules --seeds 200-209 --jobs 11"
    Run "experiments.revision6.veins.queue6 policy --runs ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 --seeds 200-209 --jobs 8"
    Run "experiments.revision6.veins.veins6_score"
}

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
Run "experiments.revision6.analysis.veins_methods --B 10000"
Run "experiments.revision6.analysis.diagnostics"
Run "experiments.revision6.analysis.outputs"
Run "experiments.revision6.manifest"
Write-Host "done: results/revision6/analysis, results/revision6/figures, results/MANIFEST.md, results/manifest.json"
