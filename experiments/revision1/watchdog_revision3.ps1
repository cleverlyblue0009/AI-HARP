# Watchdog + launcher for the revision-3 long jobs (task AIHARP_revision3_watchdog, every 20 min;
# launch_revision1.ps1 -Revision3 runs it once). Per job: if unfinished and no live process
# matches, relaunch it alone in its own hidden console. Training resumes exactly from
# ckpt_latest.pt (train_queue adopts live runs); evaluations resume from their CSVs.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$log = "logs\revision1\watchdog3.log"
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine })

function Lines($p) { if (Test-Path $p) { (Get-Content $p | Measure-Object -Line).Lines } else { 0 } }
function Alive($pattern) { [bool]($procs | Where-Object { $_ -match $pattern }) }
function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) launched $name" | Add-Content $log
}

$jobs = @()
foreach ($r in "rungB_s0", "rungB_s1", "rungC_s0", "rungC_s1") {
    $evalmod = if ($r -like "rungC*") { "privileged.eval_privileged" } else { "experiments.revision1.agent_eval" }
    $jobs += @{ n = "train_$r"; alive = "train_queue --only $r\b|revision1\\$r\b"; ready = { $true };
                done = [scriptblock]::Create("Test-Path 'checkpoints\revision1\$r\run_summary.json'");
                cmd = "$gpu -u -m experiments.revision1.train_queue --only $r --workers 3 --eval-jobs 0 >> logs\revision1\train_$r.log 2>&1" }
    $jobs += @{ n = "eval_$r"; alive = "--name $r\b";
                ready = [scriptblock]::Create("Test-Path 'checkpoints\revision1\$r\run_summary.json'");
                done = [scriptblock]::Create("(Lines 'results\revision1\agent_runs\$r.csv') -ge 481");
                cmd = "$gpu -u -m $evalmod --checkpoint checkpoints\revision1\$r\ckpt_final.pt --name $r --taus 0.5 0 --modes sampled --jobs 2 >> logs\revision1\eval_$r.log 2>&1" }
}
# R12: random-gate control for full-model seeds 1-4 (rate-matched to each seed's own learned gate).
foreach ($s in 1, 2, 3, 4) {
    $ck = "checkpoints\revision1\ref_s$s\ckpt_final.pt"
    $jobs += @{ n = "r12_ref_s$s"; alive = "--name ref_s${s}_randomgate\b"; ready = { $true };
                done = [scriptblock]::Create("(Lines 'results\revision1\agent_runs\ref_s${s}_randomgate.csv') -ge 241");
                cmd = "$gpu -u -m experiments.revision1.agent_eval --checkpoint $ck --name ref_s${s}_randomgate --random-gate-from results\revision1\agent_runs\ref_s$s.csv --modes sampled --jobs 1 >> logs\revision1\eval_r12_s$s.log 2>&1" }
}
# R9: rerun under the calibrated channel (after the fit has written r9_fit.json).
$jobs += @{ n = "r9_rerun"; alive = "r9_calibrated_channel rerun";
            ready = { Test-Path "results\revision3\R9\r9_fit.json" };
            done = { ((Lines "results\revision3\R9\r9_baseline_runs.csv") -ge 2201) -and ((Lines "results\revision3\R9\r9_agent_ref_s4.csv") -ge 241) };
            cmd = "$gpu -u -m experiments.revision3.r9_calibrated_channel rerun --jobs 3 >> logs\revision1\r9_rerun.log 2>&1" }
$status = @()
foreach ($j in $jobs) {
    $isDone = & $j.done; $isAlive = Alive $j.alive; $isReady = & $j.ready
    if ($isReady -and -not $isDone -and -not $isAlive) { Launch $j.n $j.cmd }
    $status += "$($j.n)=$(if ($isDone) {'done'} elseif ($isAlive) {'alive'} elseif (-not $isReady) {'waiting'} else {'LAUNCHED'})"
}
"$(Get-Date -Format s) $($status -join ' ')" | Add-Content $log
