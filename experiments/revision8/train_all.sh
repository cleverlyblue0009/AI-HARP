#!/usr/bin/env bash
# Revision 8: train every STRUCT run (plain s0-9, then rand s0-9), 80 generations each (ADDENDUM section 4).
# Resumable: a finished run (theta.json) is skipped, an interrupted one resumes exactly from ckpt.pkl.
#   SUMO_HOME=<sumo 1.19.0> bash experiments/revision8/train_all.sh [workers]
set -uo pipefail
cd "$(dirname "$0")/../.."
W=${1:-$(nproc)}
for arm in plain rand; do
  if [ "$arm" = rand ]; then
    while [ ! -f results/revision8/traces_sumo.json ]; do echo "waiting for SUMO training-pool traces"; sleep 60; done
  fi
  for k in 0 1 2 3 4 5 6 7 8 9; do
    [ -f "results/revision8/train/${arm}_s${k}/theta.json" ] && continue
    python -m agents.train_structured --arm "$arm" --seed "$k" --workers "$W" || exit 1
  done
done
echo "train_all: done"
