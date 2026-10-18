# Detached launcher for the long revision-1 jobs (v2).
#
# v1 (launch_revision1.cmd) started every job with `start /b`, so all of them
# shared one console; a single console control event (Ctrl+C / close) then
# killed every job at once (2026-10-17 10:27, BrokenProcessPool + forrtl
# control-C aborts). Here each job gets its OWN hidden console via
# Start-Process, so no console event can reach more than one job.
# Run via Task Scheduler (task AIHARP_revision1). Every job is resumable.
param([switch]$NoTrain, [switch]$NoSweeps, [switch]$Revision2, [switch]$Revision3)
if ($Revision3) {
    # Revision-3: R8 diagnostic ladder (rungs B, C x seeds 0, 1) + R12 random-gate evals.
    # The job list and each job's done/alive test live in watchdog_revision3.ps1.
    & (Join-Path $PSScriptRoot "watchdog_revision3.ps1")
    return
}
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$cpu = Join-Path $env:USERPROFILE "miniconda3\python.exe"
New-Item -ItemType Directory -Force logs\revision1 | Out-Null

function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden `
        -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) launched $name" | Add-Content logs\revision1\launcher.log
}

if ($Revision2) {
    # Revision-2 overnight: R6 (ref seeds 3, 4) and R7 (no-relevance seed 1) in parallel,
    # one lane each; R7 seed 0 (committed campaign/no_relevance) is only evaluated; the
    # R4 training-seed sweep runs beside them. Each job: own hidden console, resumable.
    # The job list (and each job's done/alive test) lives in watchdog_revision2.ps1,
    # so the first launch and every later relaunch use exactly the same commands.
    & (Join-Path $PSScriptRoot "watchdog_revision2.ps1")
    return
}
if (-not $NoTrain) {
    Launch "laneA" "$gpu -u -m experiments.revision1.train_queue --only ref_s1 gcn_s1 --workers 5 >> logs\revision1\train_laneA.log 2>&1"
    Launch "laneB" "$gpu -u -m experiments.revision1.train_queue --only ref_s2 mlp_s1 --workers 5 >> logs\revision1\train_laneB.log 2>&1"
    Launch "laneC" "$gpu -u -m experiments.revision1.train_queue --only star_s1 --wait-for ref_s1 ref_s2 --workers 4 >> logs\revision1\train_laneC.log 2>&1"
}
if (-not $NoSweeps) {
    Launch "grid24" "$cpu -u -m experiments.revision1.sweeps grid24 --jobs 3 >> logs\revision1\sweeps5.log 2>&1"
}
