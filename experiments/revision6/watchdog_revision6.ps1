# Watchdog for the revision-6 long jobs (task AIHARP_revision6_watchdog, every 20 min). Per job: if not done, ready
# and not alive, relaunch it in a hidden console. Everything resumes: Veins replays from their run records
# (queue6.py skips accepted replays), training from ckpt_latest.pt (train_queue adopts live runs).
# Order (addendum section 3): Tier A rules -> Tier A-P (ref_s0-4) -> Tier A-P2 (ref_s5-9); D8 training lane 0 runs
# throughout, lane 1 only once the Veins Tier A replays are done (Veins first; training on spare cores).
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$log = "logs\revision6\watchdog.log"
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
$env:SUMO_HOME = "D:\sumo\sumo-1.19.0"           # the decision servers read the cached SUMO traces
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine })
function Alive($pattern) { [bool]($procs | Where-Object { $_ -match $pattern }) }
function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) launched $name" | Add-Content $log
}
$r6 = "cache\veins_runs\r6"
$names = @(); foreach ($v in 'gcn','mlp','norel','rungB','rungC') { foreach ($s in 5..9) { $names += "${v}_s$s" } }
function TrainDone($lane) {
    for ($i = $lane; $i -lt $names.Count; $i += 2) { if (-not (Test-Path "checkpoints\revision1\$($names[$i])\run_summary.json")) { return $false } }
    return $true
}
$A2 = "ref_s5 ref_s6 ref_s7 ref_s8 ref_s9"; $A1 = "ref_s0 ref_s1 ref_s2 ref_s3 ref_s4"
$veinsDone = { Test-Path "$r6\_DONE_policy_$($A2 -replace ' ','_')_200-209" }
$jobs = @(
    @{ n = "veins_rules"; alive = "queue6 rules"; ready = { $true }; done = { Test-Path "$r6\_DONE_rules_200-209" };
       cmd = "$gpu -u -m experiments.revision6.veins.queue6 rules --seeds 200-209 --jobs 6 >> logs\revision6\tierA_rules.log 2>&1" },
    @{ n = "veins_policy_A"; alive = "queue6 policy --runs ref_s0"; ready = { Test-Path "$r6\_DONE_rules_200-209" };
       done = { Test-Path "$r6\_DONE_policy_$($A1 -replace ' ','_')_200-209" };
       cmd = "$gpu -u -m experiments.revision6.veins.queue6 policy --runs $A1 --seeds 200-209 --jobs 5 >> logs\revision6\tierA_policy_A.log 2>&1" },
    @{ n = "veins_policy_A2"; alive = "queue6 policy --runs ref_s5"; ready = { Test-Path "$r6\_DONE_policy_$($A1 -replace ' ','_')_200-209" };
       done = $veinsDone;
       cmd = "$gpu -u -m experiments.revision6.veins.queue6 policy --runs $A2 --seeds 200-209 --jobs 5 >> logs\revision6\tierA_policy_A2.log 2>&1" },
    @{ n = "train_lane0"; alive = "train_queue .*--lane 0"; ready = { $true }; done = { TrainDone 0 };
       cmd = "$gpu -u -m experiments.revision1.train_queue --only $($names -join ' ') --lane 0 --lanes 2 --workers 5 --eval-jobs 0 >> logs\revision6\train_lane0.log 2>&1" },
    @{ n = "train_lane1"; alive = "train_queue .*--lane 1"; ready = $veinsDone; done = { TrainDone 1 };
       cmd = "$gpu -u -m experiments.revision1.train_queue --only $($names -join ' ') --lane 1 --lanes 2 --workers 5 --eval-jobs 0 >> logs\revision6\train_lane1.log 2>&1" }
)
$status = @()
foreach ($j in $jobs) {
    $isDone = & $j.done; $isAlive = Alive $j.alive; $isReady = & $j.ready
    if ($isReady -and -not $isDone -and -not $isAlive) { Launch $j.n $j.cmd }
    $status += "$($j.n)=$(if ($isDone) {'done'} elseif ($isAlive) {'alive'} elseif (-not $isReady) {'waiting'} else {'LAUNCHED'})"
}
"$(Get-Date -Format s) $($status -join ' ')" | Add-Content $log
