# Watchdog for the revision-2 overnight jobs (task AIHARP_revision2_watchdog, every 20 min).
# Per job: if it is not finished and no live process matches it, relaunch that job
# alone in its own hidden console. Training resumes exactly from ckpt_latest.pt
# (train_queue adopts a run that is still live, so it never starts a second copy);
# sweeps and evaluations resume from their CSVs.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$cpu = Join-Path $env:USERPROFILE "miniconda3\python.exe"
$log = "logs\revision1\watchdog2.log"
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine })

function Lines($p) { if (Test-Path $p) { (Get-Content $p | Measure-Object -Line).Lines } else { 0 } }
function Alive($pattern) { [bool]($procs | Where-Object { $_ -match $pattern }) }
function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) relaunched $name" | Add-Content $log
}

$jobs = @()
foreach ($r in "ref_s3", "ref_s4", "norel_s1") {
    $jobs += @{ n = "train_$r"; alive = "train_queue --only $r\b|revision1\\$r\b";
                done = [scriptblock]::Create("Test-Path 'checkpoints\revision1\$r\run_summary.json'"); ready = { $true };
                cmd = "$gpu -u -m experiments.revision1.train_queue --only $r --workers 4 --eval-jobs 0 >> logs\revision1\train_$r.log 2>&1" }
    $jobs += @{ n = "eval_$r"; alive = "--name $r\b";
                done = [scriptblock]::Create("(Lines 'results\revision1\agent_runs\$r.csv') -ge 481");
                ready = [scriptblock]::Create("Test-Path 'checkpoints\revision1\$r\run_summary.json'");
                cmd = "$gpu -u -m experiments.revision1.agent_eval --checkpoint checkpoints\revision1\$r\ckpt_final.pt --name $r --taus 0.5 0 --modes sampled --jobs 2 >> logs\revision1\eval_$r.log 2>&1" }
}
$jobs += @{ n = "eval_norel_s0"; alive = "--name norel_s0\b"; ready = { $true };
            done = { (Lines "results\revision1\agent_runs\norel_s0.csv") -ge 481 };
            cmd = "$gpu -u -m experiments.revision1.agent_eval --checkpoint checkpoints\campaign\no_relevance\ckpt_final.pt --name norel_s0 --taus 0.5 0 --modes sampled --jobs 2 >> logs\revision1\eval_norel_s0.log 2>&1" }
$jobs += @{ n = "r4_trainsel"; alive = "sweeps trainsel"; ready = { $true };
            done = { (Lines "results\revision1\B_sweeps\trainsel_runs.csv") -ge 13201 };
            cmd = "$cpu -u -m experiments.revision1.sweeps trainsel --jobs 3 >> logs\revision1\sweeps_trainsel.log 2>&1" }
$status = @()
foreach ($j in $jobs) {
    $isDone = & $j.done
    $isAlive = Alive $j.alive
    $isReady = & $j.ready
    if ($isReady -and -not $isDone -and -not $isAlive) { Launch $j.n $j.cmd }
    $status += "$($j.n)=$(if ($isDone) {'done'} elseif ($isAlive) {'alive'} elseif (-not $isReady) {'waiting'} else {'RELAUNCHED'})"
}
"$(Get-Date -Format s) $($status -join ' ')" | Add-Content $log
