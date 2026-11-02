# Watchdog for the revision-6 long jobs (task AIHARP_revision6_watchdog, every 20 min). Per job: if not done, ready
# and not alive, relaunch it in a hidden console. Everything resumes: Veins replays from their run records
# (queue6.py skips accepted replays), training from ckpt_latest.pt (train_queue adopts live runs).
# Order after D10 (addendum): Tier A rules (two queues sharing the list, 11 workers) -> Tier A-P (ref_s0-4, no rural
# d=80). Tier A-P2 and the D8 training are cancelled; every freed core goes to the Veins replays.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$log = "logs\revision6\watchdog.log"
# Never inherit SUMO_HOME: backend "auto" would then pick SUMO and training would leave its trace set (incident
# 2026-11-02: gcn_s7 resumed on SUMO from update 492, quarantined). queue6.py sets it for the decision servers only.
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine })
function Alive($pattern) { [bool]($procs | Where-Object { $_ -match $pattern }) }
function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) launched $name" | Add-Content $log
}
$r6 = "cache\veins_runs\r6"
$A1 = "ref_s0 ref_s1 ref_s2 ref_s3 ref_s4"
$jobs = @(
    @{ n = "veins_rules"; alive = "queue6 rules --seeds 200-209 --jobs 6"; ready = { $true }; done = { Test-Path "$r6\_DONE_rules_200-209" };
       cmd = "$gpu -u -m experiments.revision6.veins.queue6 rules --seeds 200-209 --jobs 6 >> logs\revision6\tierA_rules.log 2>&1" },
    @{ n = "veins_rules_rev"; alive = "queue6 rules --seeds 200-209 --jobs 5 --reverse"; ready = { $true }; done = { Test-Path "$r6\_DONE_rules_200-209" };
       cmd = "$gpu -u -m experiments.revision6.veins.queue6 rules --seeds 200-209 --jobs 5 --reverse >> logs\revision6\tierA_rules_rev.log 2>&1" },
    @{ n = "veins_policy_A"; alive = "queue6 policy --runs ref_s0"; ready = { Test-Path "$r6\_DONE_rules_200-209" };
       done = { Test-Path "$r6\_DONE_policy_$($A1 -replace ' ','_')_200-209" };
       cmd = "$gpu -u -m experiments.revision6.veins.queue6 policy --runs $A1 --seeds 200-209 --jobs 8 >> logs\revision6\tierA_policy_A.log 2>&1" }
)
$status = @()
foreach ($j in $jobs) {
    $isDone = & $j.done; $isAlive = Alive $j.alive; $isReady = & $j.ready
    if ($isReady -and -not $isDone -and -not $isAlive) { Launch $j.n $j.cmd }
    $status += "$($j.n)=$(if ($isDone) {'done'} elseif ($isAlive) {'alive'} elseif (-not $isReady) {'waiting'} else {'LAUNCHED'})"
}
"$(Get-Date -Format s) $($status -join ' ')" | Add-Content $log
