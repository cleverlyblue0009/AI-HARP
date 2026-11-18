# Revision 10: regenerate results/revision10 analysis outputs from committed files (results/revision10/ADDENDUM.md).
#   powershell -ExecutionPolicy Bypass -File reproduce_revision10.ps1                 analysis only
#   powershell -ExecutionPolicy Bypass -File reproduce_revision10.ps1 --from-scratch  how to rerun the runner work
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
  Write-Host "The heavy work runs on GitHub-hosted runners: bump `"request`" in ci/r10-gates.json or ci/r10-veins.json"
  Write-Host "(mode h7 | s3 | s5) and push to revision-10; outputs land on branch r10-ci."
  exit 0
}
if (Get-ChildItem results\revision10\gate\pipeline_*.json -ErrorAction SilentlyContinue) { Run "experiments.revision10.veins.gate10 assemble" }
foreach ($m in "h7", "s3", "s5") { if (Test-Path "results\revision10\veins\$m") { Run "experiments.revision10.analysis.veins10 assemble $m" } }
if (Test-Path results\revision10\veins\h7_rows.jsonl) { Run "experiments.revision10.analysis.veins10 h7"; Run "experiments.revision10.analysis.veins10 s4" }
if ((Test-Path results\revision10\veins\h7_rows.jsonl) -and (Test-Path results\revision10\veins\s3_rows.jsonl)) { Run "experiments.revision10.analysis.veins10 s3" }
if (Test-Path results\revision10\veins\s5_rows.jsonl) { Run "experiments.revision10.analysis.veins10 s5" }
Write-Host "done: results/revision10"
