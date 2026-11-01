# Revision 5: regenerate every output in results/revision5/analysis/ (plus tables and figures) from committed rows.
#
#   powershell -ExecutionPolicy Bypass -File reproduce_revision5.ps1                 analysis only (about 1.5 h)
#   powershell -ExecutionPolicy Bypass -File reproduce_revision5.ps1 --from-scratch  rerun the simulations first
#
# Environment: the pinned interpreter only (NumPy 1.26.4, torch 2.2.2). Set AIHARP_PY to override its location.
# --from-scratch MOVES results/revision5/runs (and the custom-simulator H3 files) to *.bak-<timestamp>, nothing is
# deleted, then reruns every sweep in the pre-registered order. It uses the committed trained checkpoints (retraining:
# experiments.revision1.train_queue, see experiments/revision5/watchdog_revision5.ps1) and the Veins replays already
# collected in cache/veins_runs and results/revision5/veins (the replays themselves ran on Azure VMs,
# experiments/revision5/cloud; they cannot be rerun by this script). Expect several days of CPU time.
param([switch]$FromScratch)
if ($args -contains "--from-scratch") { $FromScratch = $true }
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = if ($env:AIHARP_PY) { $env:AIHARP_PY } else { "D:\aiharp-gpu\Scripts\python.exe" }
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue   # backend "auto" must not pick SUMO by accident
$sumo = if ($env:AIHARP_SUMO_HOME) { $env:AIHARP_SUMO_HOME } else { "D:\sumo\sumo-1.19.0" }

$envline = & $py -c "import sys, numpy; print(sys.executable, numpy.__version__)"
Write-Host "[env] $envline"
if ($envline -notmatch " 1\.26\.4$") { throw "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" }

function Run([string]$a, [switch]$Sumo) {
    Write-Host ">> python -m $a"
    if ($Sumo) { $env:SUMO_HOME = $sumo }
    try { & $py -u -m $a.Split(" ") ; if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $a" } }
    finally { if ($Sumo) { Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue } }
}

if ($FromScratch) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($p in "results\revision5\runs", "results\revision5\veins\custom_h3.jsonl", "results\revision5\veins\custom_h3_subepoch.jsonl",
                   "results\revision5\veins\d6_calibration") {
        if (Test-Path $p) { Move-Item $p "$p.bak-$ts"; Write-Host "moved $p -> $p.bak-$ts" }
    }
    New-Item -ItemType Directory -Force results\revision5\runs | Out-Null
    $R5 = "ref_s0 ref_s1 ref_s2 ref_s3 ref_s4"; $R10 = "$R5 ref_s5 ref_s6 ref_s7 ref_s8 ref_s9"
    # default channel: CAL -> thresholds (already frozen and committed; recomputed here only as a check) -> SEL -> EVAL
    Run "experiments.revision5.sweep classical --seedset CAL --cells grid24 --jobs 3"
    Run "experiments.revision5.thresholds"
    Run "experiments.revision5.sweep classical --seedset SEL --cells grid24 --jobs 3"
    Run "experiments.revision5.sweep rule --seedset SEL --cells grid24 --jobs 3"
    Run "experiments.revision5.sweep agent --seedset SEL --cells grid24 --runs $R5 --jobs 2"
    Run "experiments.revision5.sweep classical --seedset EVAL --cells grid24 --jobs 3"
    Run "experiments.revision5.sweep rule --seedset EVAL --cells grid24 --jobs 3"
    Run "experiments.revision5.sweep agent --seedset SEL --cells headline --runs ref_s5 ref_s6 ref_s7 ref_s8 ref_s9 --tag s5_9 --jobs 2"
    Run "experiments.revision5.sweep agent --seedset EVAL --cells headline --runs ref_s5 ref_s6 ref_s7 ref_s8 ref_s9 --tag s5_9 --jobs 4"
    Run "experiments.revision5.sweep agent --seedset EVAL --cells grid24 --runs $R5 --jobs 4"
    # Step 1: ladder, variants, gate, budget (SEL then EVAL each)
    $step1 = @(@("experiments.revision5.sweep", "rungB_s0 rungB_s1 rungB_s2 rungB_s3 rungB_s4", "ladderB", ""),
               @("privileged.sweep_privileged", "rungC_s0 rungC_s1 rungC_s2 rungC_s3 rungC_s4", "ladderC", ""),
               @("experiments.revision5.sweep", "gcn_s0 gcn_s1 gcn_s2 gcn_s3 gcn_s4 mlp_s0 mlp_s1 mlp_s2 mlp_s3 mlp_s4 norel_s0 norel_s1 norel_s2 norel_s3 norel_s4", "variants", ""),
               @("experiments.revision5.sweep", $R10, "tau0", "--taus 0"),
               @("experiments.revision5.sweep", $R10, "rgate", "--random-gate-from SEL"),
               @("experiments.revision5.sweep", "ref_s0_u2000@1200 ref_s0_u2000@1500 ref_s0_u2000@1700 ref_s0_u2000@2000 ref_s1_u2000@1200 ref_s1_u2000@1500 ref_s1_u2000@1700 ref_s1_u2000@2000 ref_s2_u2000@1200 ref_s2_u2000@1500 ref_s2_u2000@1700 ref_s2_u2000@2000", "budget", ""))
    foreach ($s in $step1) {
        foreach ($set in "SEL", "EVAL") {
            Run ("$($s[0]) agent --seedset $set --cells headline --runs $($s[1]) $($s[3]) --tag $($s[2]) --jobs 5" -replace "\s+", " ")
        }
    }
    # Step 2: stress test (calibrated channel, own thresholds, D5) and SUMO mobility
    Run "experiments.revision5.sweep classical --seedset CAL --cells grid24 --channel calibrated --jobs 3"
    Run "experiments.revision5.thresholds --channel calibrated"
    foreach ($set in "SEL", "EVAL") {
        Run "experiments.revision5.sweep classical --seedset $set --cells grid24 --channel calibrated --jobs 3"
        Run "experiments.revision5.sweep rule --seedset $set --cells grid24 --channel calibrated --jobs 3"
    }
    Run "experiments.revision5.sweep classical --seedset SEL --cells headline --channel default --backend sumo --jobs 3" -Sumo
    Run "experiments.revision5.sweep classical --seedset EVAL --cells headline --channel default --backend sumo --jobs 2" -Sumo
    Run "experiments.revision5.sweep rule --seedset SEL --cells headline --channel default --backend sumo --jobs 3" -Sumo
    Run "experiments.revision5.sweep rule --seedset EVAL --cells headline --channel default --backend sumo --jobs 3" -Sumo
    Run "experiments.revision5.sweep agent --seedset SEL --cells headline --channel default --backend sumo --runs $R10 --jobs 5" -Sumo
    Run "experiments.revision5.sweep agent --seedset EVAL --cells headline --channel default --backend sumo --runs $R10 --jobs 5" -Sumo
    Run "experiments.revision5.sweep agent --seedset SEL --cells grid24 --channel calibrated --runs $R5 --jobs 5"
    Run "experiments.revision5.sweep agent --seedset EVAL --cells grid24 --channel calibrated --runs $R5 --jobs 5"
    # Step 3: custom-simulator side of H3 (Veins side: committed replays), D6 calibration and frozen re-run
    Run "experiments.revision5.veins.custom_h3 --jobs 2"
    Run "experiments.revision5.veins.d6_calibrate run --jobs 4"
    Run "experiments.revision5.veins.d6_calibrate score"
    Run "experiments.revision5.veins.d6_calibrate rerun --jobs 4"
}

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
Run "experiments.revision5.veins.d6_calibrate score"
Run "experiments.revision5.analysis.h1 --B 10000"
Run "experiments.revision5.analysis.h1 --B 10000 --D Dr"
Run "experiments.revision5.analysis.h2 --B 10000"
Run "experiments.revision5.analysis.h2 --B 10000 --D Dr"
Run "experiments.revision5.analysis.h2 --B 10000 --grid24"
Run "experiments.revision5.analysis.h3 --B 10000"
Run "experiments.revision5.analysis.h3 --mode sub_epoch"
Run "experiments.revision5.analysis.holm"
Run "experiments.revision5.analysis.secondary --B 10000"
Run "experiments.revision5.analysis.training_diag"
Run "experiments.revision5.analysis.compute_curves"
Run "experiments.revision5.analysis.ledger"
Run "experiments.revision5.analysis.tables"
Run "experiments.revision5.analysis.figures"
Run "experiments.revision5.manifest"
Write-Host "done: results/revision5/analysis, results/revision5/figures, results/MANIFEST.md, results/manifest.json"
