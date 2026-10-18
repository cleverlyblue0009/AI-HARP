# Watchdog + launcher for the revision-5 long jobs (task AIHARP_revision5_watchdog, every 20 min).
# Per job: if unfinished and no live process matches, relaunch it alone in its own hidden console.
# Training resumes exactly from ckpt_latest.pt (train_queue adopts live runs); simulation jobs resume
# from their CSVs (experiments/revision1/runner.py skips finished task hashes). Jobs and their order
# are pre-registered in results/revision5/protocol.json.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$gpu = "D:\aiharp-gpu\Scripts\python.exe"
$log = "logs\revision5\watchdog.log"
New-Item -ItemType Directory -Force logs\revision5 | Out-Null
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.CommandLine })
function Alive($pattern) { [bool]($procs | Where-Object { $_ -match $pattern }) }
function Launch($name, $cmdline) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmdline" -WindowStyle Hidden -WorkingDirectory $root | Out-Null
    "$(Get-Date -Format s) launched $name" | Add-Content $log
}
function Done($name) {
    $s = "checkpoints\revision1\$name\run_summary.json"
    if (-not (Test-Path $s)) { return $false }
    if ($name -like "*_u2000") { return ((Get-Content $s -Raw | ConvertFrom-Json).total_updates -gt 1000) }
    return $true
}

# --- Slot A: training, two lanes (pre-registered priority order) ------------------------------
$train = @("ref_s5","ref_s6","ref_s7","ref_s8","ref_s9","rungB_s2","rungB_s3","rungB_s4","rungC_s2","rungC_s3","rungC_s4",
           "gcn_s2","gcn_s3","gcn_s4","mlp_s2","mlp_s3","mlp_s4","norel_s2","norel_s3","norel_s4",
           "ref_s0_u2000","ref_s1_u2000","ref_s2_u2000")
$jobs = @()
foreach ($lane in 0, 1) {
    $mine = @(); for ($i = $lane; $i -lt $train.Count; $i += 2) { $mine += $train[$i] }
    $jobs += @{ n = "train_lane$lane"; alive = "train_queue --only .* --lane $lane --lanes 2"; ready = { $true };
                done = [scriptblock]::Create("@('" + ($mine -join "','") + "') | ForEach-Object { Done `$_ } | Where-Object { -not `$_ } | Measure-Object | ForEach-Object { `$_.Count -eq 0 }");
                cmd = "$gpu -u -m experiments.revision1.train_queue --only $($train -join ' ') --lane $lane --lanes 2 --workers 5 --eval-jobs 0 >> logs\revision5\train_lane$lane.log 2>&1" }
}

# --- Slot B: simulation tracks are appended here as they are written (see revision-5 report) --
$extra = Join-Path $PSScriptRoot "watchdog_revision5_jobs.ps1"
if (Test-Path $extra) { . $extra }

$status = @()
foreach ($j in $jobs) {
    $isDone = & $j.done; $isAlive = Alive $j.alive; $isReady = & $j.ready
    if ($isReady -and -not $isDone -and -not $isAlive) { Launch $j.n $j.cmd }
    $status += "$($j.n)=$(if ($isDone) {'done'} elseif ($isAlive) {'alive'} elseif (-not $isReady) {'waiting'} else {'LAUNCHED'})"
}
"$(Get-Date -Format s) $($status -join ' ')" | Add-Content $log
