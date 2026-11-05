# Watchdog for the revision-7 long jobs. Per job: if not done, ready and not alive, relaunch it in a hidden console.
# Everything resumes: Veins replays from their run records, custom runs from their rows.
# Order (results/revision7/ADDENDUM.md): Experiment A in Veins and in the custom simulator, then Experiment B.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$log = "logs\revision7\watchdog.log"
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue   # backend "auto" must not pick SUMO; the jobs set it where they need it
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine })
function Alive($pattern) { [bool]($procs | Where-Object { $_ -match $pattern }) }
function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) launched $name" | Add-Content $log
}
$r = "cache\veins_runs"
$rows = "results\revision7\runs\custom_capped_rows.jsonl"
$jobs = @(
    @{ n = "veins_capped"; alive = "queue7 capped"; ready = { $true }; done = { Test-Path "$r\r7_capped\_DONE" };
       cmd = "$gpu -u -m experiments.revision7.veins.queue7 capped --jobs 8 >> logs\revision7\veins_capped.log 2>&1" },
    @{ n = "custom_capped"; alive = "revision7.custom_cap"; ready = { $true };
       done = { (Test-Path $rows) -and ((Get-Content $rows | Measure-Object -Line).Lines -ge 600) };
       cmd = "$gpu -u -m experiments.revision7.custom_cap --jobs 3 >> logs\revision7\custom_capped.log 2>&1" },
    @{ n = "veins_d9"; alive = "queue7 d9"; ready = { Test-Path "$r\r7_capped\_DONE" }; done = { Test-Path "$r\r7_d9\_DONE" };
       cmd = "$gpu -u -m experiments.revision7.veins.queue7 d9 --jobs 11 >> logs\revision7\veins_d9.log 2>&1" }
)
$status = @()
foreach ($j in $jobs) {
    $isDone = & $j.done; $isAlive = Alive $j.alive; $isReady = & $j.ready
    if ($isReady -and -not $isDone -and -not $isAlive) { Launch $j.n $j.cmd }
    $status += "$($j.n)=$(if ($isDone) {'done'} elseif ($isAlive) {'alive'} elseif (-not $isReady) {'waiting'} else {'LAUNCHED'})"
}
"$(Get-Date -Format s) $($status -join ' ')" | Add-Content $log
