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

# B4. SEL agent, full model seeds 0-4, 24 cells (cell-major, so the 4 headline cells finish first);
#     started early on existing checkpoints beside SEL classical (user, 2026-10-18).
$jobs += @{ n = "sel_agent_s0_4"; alive = "sweep agent --seedset SEL --cells grid24 --runs ref_s0"; ready = { Frozen };
            done = { (Rows "$runs\agent__SEL__grid24__default__fallback.csv") -ge 14400 };
            cmd = "$gpu -u -m experiments.revision5.sweep agent --seedset SEL --cells grid24 --runs ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 --jobs 2 >> logs\revision5\sel_agent_s0_4.log 2>&1" }

# B5. H3 Veins scenario bundles for the 4 headline cells, EVAL seeds 200-209 (inputs only; no replays
#     are launched until the H3 seed/H1-split question is settled -- see REVISION5 report).
$jobs += @{ n = "veins_export"; alive = "veins.queue export"; ready = { $true };
            done = { @(Get-ChildItem cache\veins\*_s2[0-9][0-9]\bundle.json -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -match '_s20[0-9]$' }).Count -ge 40 };
            cmd = "set SUMO_HOME=D:\sumo\sumo-1.19.0&& $gpu -u -m experiments.revision5.veins.queue export --seeds 200 201 202 203 204 205 206 207 208 209 >> logs\revision5\veins_export.log 2>&1" }

# B6. Custom simulator on SUMO, 4 headline cells x EVAL seeds 200-229 x 55 classical settings: the paired
#     custom side of H3's network layer (200-209) and of the 3-fold cross-fitted audit (DEVIATIONS D1).
$jobs += @{ n = "eval_classical_sumo"; alive = "sweep classical --seedset EVAL --cells headline --channel default --backend sumo"; ready = { Frozen };
            done = { (Rows "$runs\classical__EVAL__headline__default__sumo.csv") -ge 6600 };
            cmd = "set SUMO_HOME=D:\sumo\sumo-1.19.0&& $gpu -u -m experiments.revision5.sweep classical --seedset EVAL --cells headline --channel default --backend sumo --jobs 2 >> logs\revision5\eval_classical_sumo.log 2>&1" }

# B7. H3 Veins replays, network layer first: EVAL seeds 200-209, all 55 settings, 4 headline cells (2200).
#     Gated on the greedy/DV-CAST validation (flag written by hand after both validation replays pass).
$jobs += @{ n = "veins_net"; alive = "veins.queue run --seeds 200 201"; ready = { Test-Path "cache\veins_runs\VALIDATED" };
            done = { @(Get-ChildItem cache\veins_runs\h3\*.run.json -ErrorAction SilentlyContinue).Count -ge 2200 -and ((& $gpu -m experiments.revision5.veins.queue status --seeds 200 201 202 203 204 205 206 207 208 209) -match " 0 to do") };
            cmd = "$gpu -u -m experiments.revision5.veins.queue run --seeds 200 201 202 203 204 205 206 207 208 209 --jobs 10 >> logs\revision5\veins_net.log 2>&1" }
