# Revision 8: train every STRUCT run on the laptop (Windows), same order and budget as train_all.sh (ADDENDUM section 4).
# Resumable: a finished run (theta.json) is skipped; an interrupted one resumes exactly from ckpt.pkl (also one started
# in the cloud: that resume is on a different machine and is reported as such in run_meta / REVISION8_STATUS.md).
#   powershell -ExecutionPolicy Bypass -File experiments\revision8\train_all.ps1 [-Workers 10]
param([int]$Workers = 10)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$py = if ($env:AIHARP_PY) { $env:AIHARP_PY } else { "D:\aiharp-gpu\Scripts\python.exe" }
if (-not $env:SUMO_HOME) { $env:SUMO_HOME = "D:\sumo\sumo-1.19.0" }       # R4 (STRUCT-rand) needs SUMO 1.19.0
$jobs = "plain:0","plain:1","plain:2","plain:3","plain:4","rand:0","rand:1","rand:2","rand:3","rand:4",
        "plain:5","plain:6","plain:7","plain:8","plain:9","rand:5","rand:6","rand:7","rand:8","rand:9"
foreach ($j in $jobs) {
    $arm, $k = $j.Split(":")
    if (Test-Path "results\revision8\train\${arm}_s$k\theta.json") { continue }
    & $py -u -m agents.train_structured --arm $arm --seed $k --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "training failed: ${arm}_s$k" }
}
Write-Host "train_all: done"
