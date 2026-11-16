# Revision 9: regenerate results/revision9 analysis outputs from committed rows (results/revision9/ADDENDUM.md).
#   powershell -ExecutionPolicy Bypass -File reproduce_revision9.ps1                 analysis only
#   powershell -ExecutionPolicy Bypass -File reproduce_revision9.ps1 --from-scratch  how to rerun the runner work
param([switch]$FromScratch)
if ($args -contains "--from-scratch") { $FromScratch = $true }
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = if ($env:AIHARP_PY) { $env:AIHARP_PY } else { "D:\aiharp-gpu\Scripts\python.exe" }
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
$envline = & $py -c "import sys, numpy; print(sys.executable, numpy.__version__)"
Write-Host "[env] $envline"
if ($envline -notmatch " 1\.26\.4$") { throw "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" }
function Run([string]$a) { Write-Host ">> python -m $a"; & $py -u -m $a.Split(" "); if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $a" } }
if ($FromScratch) {
  Write-Host "The heavy work runs on GitHub-hosted runners: bump `"request`" in ci/r9-gates.json, ci/r9-train.json,"
  Write-Host "ci/r9-veins.json (mode h6 | c1) or ci/r9-eval.json and push to revision-9; outputs land on branch r9-ci."
  exit 0
}
if (Get-ChildItem results\revision9\gates_runs_slice*of20.csv -ErrorAction SilentlyContinue) { Run "experiments.revision9.gates analyse" }
if (Test-Path results\revision9\veins\h6_rows.jsonl) { Run "experiments.revision9.analysis.veins9 h6" }
if (Test-Path results\revision9\veins\c1_rows.jsonl) { Run "experiments.revision9.analysis.veins9 c1" }
if ((Test-Path results\revision9\veins\h6_rows.jsonl) -and (Test-Path results\revision9\veins\c1_rows.jsonl)) { Run "experiments.revision9.analysis.veins9 s1" }
if (Test-Path results\revision9\runs\struct9_rand__EVAL__headline__default__fallback.csv) { Run "experiments.revision9.analysis.s2 --B 10000" }
Write-Host "done: results/revision9"
