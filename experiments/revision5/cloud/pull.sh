#!/usr/bin/env bash
# Pull-only sync of the offload VMs into cache/cloud/<vm>/ (run in WSL). The VMs never push. Merging into the
# canonical locations is done separately by experiments.revision5.cloud.merge (provenance checks).
set -u
DEST=/mnt/c/Users/UPASANA/AI-HARP/cache/cloud
SSH="ssh -F $HOME/.ssh/aiharp_config"
for h in aiharp-ci aiharp-ea aiharp-ks; do
  mkdir -p "$DEST/$h/veins_runs" "$DEST/$h/bundles" "$DEST/$h/runs"
  rsync -az -e "$SSH" "$h:GATE.json" "$DEST/$h/" 2>/dev/null
  rsync -az -e "$SSH" "$h:AI-HARP/cache/veins_runs/h3/" "$DEST/$h/veins_runs/" 2>/dev/null
  rsync -az -e "$SSH" --include='*/' --include='bundle.json' --exclude='*' "$h:AI-HARP/cache/veins/" "$DEST/$h/bundles/" 2>/dev/null
  rsync -az -e "$SSH" --include='*slice*' --exclude='*' "$h:AI-HARP/results/revision5/runs/" "$DEST/$h/runs/" 2>/dev/null
  rsync -az -e "$SSH" "$h:*.log" "$DEST/$h/" 2>/dev/null
  echo "$(date -Is) pulled $h"
done
