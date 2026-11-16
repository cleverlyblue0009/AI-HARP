#!/usr/bin/env bash
# Revision 8: train every STRUCT run, 80 generations each (ADDENDUM section 4). Order: the ten runs Veins H5 needs
# first (plain s0-4, rand s0-4), then plain s5-9 (H4), then rand s5-9 (descriptive). The order is not part of the
# pre-registration; each run is independent and deterministic.
# Resumable: a finished run (theta.json) is skipped, an interrupted one resumes exactly from ckpt.pkl.
#   SUMO_HOME=<sumo 1.19.0> bash experiments/revision8/train_all.sh [workers]
set -uo pipefail
cd "$(dirname "$0")/../.."
W=${1:-$(nproc)}
for job in plain:0 plain:1 plain:2 plain:3 plain:4 rand:0 rand:1 rand:2 rand:3 rand:4 \
           plain:5 plain:6 plain:7 plain:8 plain:9 rand:5 rand:6 rand:7 rand:8 rand:9; do
  arm=${job%%:*}; k=${job##*:}
  [ -f "results/revision8/train/${arm}_s${k}/theta.json" ] && continue
  if [ "$arm" = rand ]; then
    while [ ! -f results/revision8/traces_sumo.json ]; do echo "waiting for SUMO training-pool traces"; sleep 60; done
  fi
  python -m agents.train_structured --arm "$arm" --seed "$k" --workers "$W" || exit 1
done
echo "train_all: done"
