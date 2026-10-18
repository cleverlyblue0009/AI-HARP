# Revision-5 simulation tracks for watchdog_revision5.ps1 (dot-sourced; adds to $jobs).
# Each job resumes from its CSV; "done" = the expected number of rows exists.
function Rows($p) { if (Test-Path $p) { (Get-Content $p | Measure-Object -Line).Lines - 1 } else { 0 } }
$runs = "results\revision5\runs"

# B1. CAL classical, 24 cells: q*, m* are frozen from these (thresholds.json) before any SEL/EVAL run.
$jobs += @{ n = "cal_classical"; alive = "sweep classical --seedset CAL"; ready = { $true };
            done = { (Rows "$runs\classical__CAL__grid24__default__fallback.csv") -ge 13200 };
            cmd = "$gpu -u -m experiments.revision5.sweep classical --seedset CAL --cells grid24 --jobs 3 >> logs\revision5\cal_classical.log 2>&1" }

# B2. Freeze q*, m* (thresholds.json) and commit it -- only that path -- before any SEL/EVAL job may start.
$thr = "results\revision5\thresholds.json"
function Frozen { if (Test-Path $thr) { git ls-files --error-unmatch $thr 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }; return $false }
$jobs += @{ n = "thresholds"; alive = "experiments.revision5.thresholds"; ready = { (Rows "$runs\classical__CAL__grid24__default__fallback.csv") -ge 13200 };
            done = { Frozen };
            cmd = "$gpu -u -m experiments.revision5.thresholds >> logs\revision5\thresholds.log 2>&1 & if exist $thr git add -f $thr && git commit -m ""revision-5: freeze q*/m* from CAL (thresholds.json), before any SEL/EVAL run"" -m ""Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"" -- $thr >> logs\revision5\thresholds.log 2>&1" }

# B3. SEL classical, then SEL rule (RAR 27 + SCF 12), 24 cells; one at a time beside training.
$jobs += @{ n = "sel_classical"; alive = "sweep classical --seedset SEL"; ready = { Frozen };
            done = { (Rows "$runs\classical__SEL__grid24__default__fallback.csv") -ge 26400 };
            cmd = "$gpu -u -m experiments.revision5.sweep classical --seedset SEL --cells grid24 --jobs 3 >> logs\revision5\sel_classical.log 2>&1" }
$jobs += @{ n = "sel_rule"; alive = "sweep rule --seedset SEL"; ready = { (Rows "$runs\classical__SEL__grid24__default__fallback.csv") -ge 26400 };
            done = { (Rows "$runs\rule__SEL__grid24__default__fallback.csv") -ge 18720 };
            cmd = "$gpu -u -m experiments.revision5.sweep rule --seedset SEL --cells grid24 --jobs 3 >> logs\revision5\sel_rule.log 2>&1" }
