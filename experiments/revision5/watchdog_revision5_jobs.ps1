# Revision-5 simulation tracks for watchdog_revision5.ps1 (dot-sourced; adds to $jobs).
# Each job resumes from its CSV; "done" = the expected number of rows exists.
function Rows($p) { if (Test-Path $p) { (Get-Content $p | Measure-Object -Line).Lines - 1 } else { 0 } }
$runs = "results\revision5\runs"

# B1. CAL classical, 24 cells: q*, m* are frozen from these (thresholds.json) before any SEL/EVAL run.
$jobs += @{ n = "cal_classical"; alive = "sweep classical --seedset CAL"; ready = { $true };
            done = { (Rows "$runs\classical__CAL__grid24__default__fallback.csv") -ge 13200 };
            cmd = "$gpu -u -m experiments.revision5.sweep classical --seedset CAL --cells grid24 --jobs 3 >> logs\revision5\cal_classical.log 2>&1" }
