# Every 30 min (task AIHARP_cloud_sync): pull from the offload VMs (WSL rsync, pull-only), then merge with
# provenance checks. Logs: logs\revision5\cloud_sync.log
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $root
$log = "logs\revision5\cloud_sync.log"
"$(Get-Date -Format s) sync start" | Add-Content $log
wsl -d Ubuntu -- bash -c "tr -d '\r' < /mnt/c/Users/UPASANA/AI-HARP/experiments/revision5/cloud/pull.sh | bash" 2>&1 | Add-Content $log
& "D:\aiharp-gpu\Scripts\python.exe" -m experiments.revision5.cloud.merge 2>&1 | Where-Object { $_ -notmatch " INFO " } | Add-Content $log
"$(Get-Date -Format s) sync done" | Add-Content $log
