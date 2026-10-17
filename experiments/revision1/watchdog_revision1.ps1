# Watchdog for the revision-1 jobs (scheduled every 20 min as AIHARP_revision1_watchdog).
# Relaunches only what is missing; every job resumes from its own state, and
# train_queue adopts a run another process is still training, so a relaunch can
# never start a second copy of the same run.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$log = "logs\revision1\watchdog.log"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine }
$runs = "ref_s1", "ref_s2", "gcn_s1", "mlp_s1", "star_s1"
$pending = $runs | Where-Object { -not (Test-Path "checkpoints\revision1\$_\run_summary.json") }
$queueAlive = [bool]($procs | Where-Object { $_ -match "train_queue" })
$trainAlive = [bool]($procs | Where-Object { $_ -match "agents.train" })
$gridAlive = [bool]($procs | Where-Object { $_ -match "sweeps grid24" })
$gridRows = (Get-Content results\revision1\B_sweeps\grid24_runs.csv | Measure-Object -Line).Lines - 1
$launcher = Join-Path $PSScriptRoot "launch_revision1.ps1"
if ($pending -and -not $queueAlive -and -not $trainAlive) {
    "$(Get-Date -Format s) training not running, pending: $($pending -join ','); relaunching" | Add-Content $log
    & $launcher -NoSweeps
}
if ($gridRows -lt 13200 -and -not $gridAlive) {
    "$(Get-Date -Format s) grid24 not running ($gridRows rows); relaunching" | Add-Content $log
    & $launcher -NoTrain
}
"$(Get-Date -Format s) ok: pending=$($pending.Count) queue=$queueAlive train=$trainAlive grid=$gridAlive rows=$gridRows" | Add-Content $log
