#!/usr/bin/env bash
# Revision 6: regenerate results/revision6/analysis/ (plus the table and figure) from committed rows.
#
#   ./reproduce_revision6.sh                  analysis only (about 5 minutes)
#   ./reproduce_revision6.sh --from-scratch   rerun the Veins replays first
#
# Environment: the pinned interpreter only (NumPy 1.26.4, torch 2.2.2). Set AIHARP_PY to its location (default: the
# Windows venv seen from Git Bash). --from-scratch needs WSL Ubuntu with Veins 5.3, OMNeT++ 6.0.3, SUMO 1.19.0, the gated
# app and the revision-5 bundles; it MOVES the scored rows and cache/veins_runs/r6 to *.bak-<timestamp>, nothing is
# deleted. Same steps as reproduce_revision6.ps1, which has the details.
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
  for p in results/revision6/veins/veins6_rows.jsonl results/revision6/veins/veins6_rows_merge.json cache/veins_runs/r6; do
    if [[ -e "$p" ]]; then mv "$p" "$p.bak-$ts"; echo "moved $p -> $p.bak-$ts"; fi
  done
  run experiments.revision6.veins.r6_bundle
  run experiments.revision6.veins.queue6 rules --seeds 200-209 --jobs 11
  run experiments.revision6.veins.queue6 policy --runs ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 --seeds 200-209 --jobs 8
  run experiments.revision6.veins.veins6_score
fi

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
run experiments.revision6.analysis.veins_methods --B 10000
run experiments.revision6.analysis.diagnostics
run experiments.revision6.analysis.outputs
run experiments.revision6.manifest
echo "done: results/revision6/analysis, results/revision6/figures, results/MANIFEST.md, results/manifest.json"
