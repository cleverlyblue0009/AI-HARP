#!/usr/bin/env bash
# Revision 7: regenerate results/revision7/analysis/ (Experiments A and B) from committed rows.
#
#   ./reproduce_revision7.sh                  analysis only (about 10 minutes)
#   ./reproduce_revision7.sh --from-scratch   rerun validation, replays and runs first
#
# Environment: the pinned interpreter only (NumPy 1.26.4, torch 2.2.2). Set AIHARP_PY to its location (default: the
# Windows venv seen from Git Bash). --from-scratch needs WSL Ubuntu with Veins 5.3, OMNeT++ 6.0.3, SUMO 1.19.0, the
# revision-7 app, the revision-5 bundles and the revision-6 r6.txt files; it MOVES the scored rows and cache/veins_runs/r7_*
# to *.bak-<timestamp>, nothing is deleted. Same steps as reproduce_revision7.ps1, which has the details.
set -euo pipefail
cd "$(dirname "$0")"
PY="${AIHARP_PY:-/d/aiharp-gpu/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
unset SUMO_HOME

envline="$("$PY" -c 'import sys, numpy; print(sys.executable, numpy.__version__)' | tr -d '\r')"
echo "[env] $envline"
[[ "$envline" == *" 1.26.4" ]] || { echo "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" >&2; exit 1; }

run() { echo ">> python -m $*"; "$PY" -u -m "$@"; }

if [[ "${1:-}" == "--from-scratch" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  for p in results/revision7/veins/capped_rows.jsonl results/revision7/veins/capped_rows_merge.json \
           results/revision7/veins/d9_rows.jsonl results/revision7/veins/d9_rows_merge.json \
           results/revision7/runs/custom_capped_rows.jsonl results/revision7/runs/custom_capped_meta.json \
           cache/veins_runs/r7_gate cache/veins_runs/r7_gate_capped cache/veins_runs/r7_capped cache/veins_runs/r7_d9; do
    if [[ -e "$p" ]]; then mv "$p" "$p.bak-$ts"; echo "moved $p -> $p.bak-$ts"; fi
  done
  run experiments.revision7.veins.validate7 unit
  run experiments.revision7.veins.validate7 gate
  run experiments.revision7.veins.queue7 capped --jobs 8
  run experiments.revision7.custom_cap --jobs 3
  run experiments.revision7.veins.queue7 d9 --jobs 11
  run experiments.revision7.veins.score7 capped
  run experiments.revision7.veins.score7 d9
fi

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
run experiments.revision7.analysis.exp_a
run experiments.revision7.analysis.exp_b --B 10000
run experiments.revision7.manifest
echo "done: results/revision7/analysis, results/MANIFEST.md, results/manifest.json"
