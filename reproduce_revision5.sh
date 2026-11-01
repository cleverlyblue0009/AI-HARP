#!/usr/bin/env bash
# Revision 5: regenerate every output in results/revision5/analysis/ (plus tables and figures) from committed rows.
#
#   ./reproduce_revision5.sh                  analysis only (about 1.5 h)
#   ./reproduce_revision5.sh --from-scratch   rerun the simulations first
#
# Environment: the pinned interpreter only (NumPy 1.26.4, torch 2.2.2). Set AIHARP_PY to its location (default: the
# Windows venv seen from Git Bash). --from-scratch MOVES results/revision5/runs (and the custom-simulator H3 files) to
# *.bak-<timestamp>, nothing is deleted, then reruns every sweep in the pre-registered order. It uses the committed
# trained checkpoints and the Veins replays already collected (those ran on Azure VMs, experiments/revision5/cloud).
# Same steps as reproduce_revision5.ps1.
set -euo pipefail
cd "$(dirname "$0")"
PY="${AIHARP_PY:-/d/aiharp-gpu/Scripts/python.exe}"
SUMO="${AIHARP_SUMO_HOME:-D:\\sumo\\sumo-1.19.0}"
export PYTHONIOENCODING=utf-8
unset SUMO_HOME

envline="$("$PY" -c 'import sys, numpy; print(sys.executable, numpy.__version__)' | tr -d '\r')"
echo "[env] $envline"
[[ "$envline" == *" 1.26.4" ]] || { echo "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" >&2; exit 1; }

run() { echo ">> python -m $*"; "$PY" -u -m "$@"; }
run_sumo() { echo ">> (SUMO) python -m $*"; SUMO_HOME="$SUMO" "$PY" -u -m "$@"; }

if [[ "${1:-}" == "--from-scratch" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  for p in results/revision5/runs results/revision5/veins/custom_h3.jsonl results/revision5/veins/custom_h3_subepoch.jsonl \
           results/revision5/veins/d6_calibration; do
    [[ -e "$p" ]] && mv "$p" "$p.bak-$ts" && echo "moved $p -> $p.bak-$ts"
  done
  mkdir -p results/revision5/runs
  R5="ref_s0 ref_s1 ref_s2 ref_s3 ref_s4"; R10="$R5 ref_s5 ref_s6 ref_s7 ref_s8 ref_s9"
  S=experiments.revision5.sweep
  run $S classical --seedset CAL --cells grid24 --jobs 3
  run experiments.revision5.thresholds
  run $S classical --seedset SEL --cells grid24 --jobs 3
  run $S rule --seedset SEL --cells grid24 --jobs 3
  run $S agent --seedset SEL --cells grid24 --runs $R5 --jobs 2
  run $S classical --seedset EVAL --cells grid24 --jobs 3
  run $S rule --seedset EVAL --cells grid24 --jobs 3
  run $S agent --seedset SEL --cells headline --runs ref_s5 ref_s6 ref_s7 ref_s8 ref_s9 --tag s5_9 --jobs 2
  run $S agent --seedset EVAL --cells headline --runs ref_s5 ref_s6 ref_s7 ref_s8 ref_s9 --tag s5_9 --jobs 4
  run $S agent --seedset EVAL --cells grid24 --runs $R5 --jobs 4
  BUDGET="ref_s0_u2000@1200 ref_s0_u2000@1500 ref_s0_u2000@1700 ref_s0_u2000@2000 ref_s1_u2000@1200 ref_s1_u2000@1500 ref_s1_u2000@1700 ref_s1_u2000@2000 ref_s2_u2000@1200 ref_s2_u2000@1500 ref_s2_u2000@1700 ref_s2_u2000@2000"
  VARIANTS="gcn_s0 gcn_s1 gcn_s2 gcn_s3 gcn_s4 mlp_s0 mlp_s1 mlp_s2 mlp_s3 mlp_s4 norel_s0 norel_s1 norel_s2 norel_s3 norel_s4"
  for set in SEL EVAL; do
    run $S agent --seedset $set --cells headline --runs rungB_s0 rungB_s1 rungB_s2 rungB_s3 rungB_s4 --tag ladderB --jobs 5
    run privileged.sweep_privileged agent --seedset $set --cells headline --runs rungC_s0 rungC_s1 rungC_s2 rungC_s3 rungC_s4 --tag ladderC --jobs 5
    run $S agent --seedset $set --cells headline --runs $VARIANTS --tag variants --jobs 5
    run $S agent --seedset $set --cells headline --runs $R10 --taus 0 --tag tau0 --jobs 5
  done
  for set in SEL EVAL; do run $S agent --seedset $set --cells headline --runs $R10 --random-gate-from SEL --tag rgate --jobs 5; done
  for set in SEL EVAL; do run $S agent --seedset $set --cells headline --runs $BUDGET --tag budget --jobs 5; done
  run $S classical --seedset CAL --cells grid24 --channel calibrated --jobs 3
  run experiments.revision5.thresholds --channel calibrated
  for set in SEL EVAL; do
    run $S classical --seedset $set --cells grid24 --channel calibrated --jobs 3
    run $S rule --seedset $set --cells grid24 --channel calibrated --jobs 3
  done
  run_sumo $S classical --seedset SEL --cells headline --channel default --backend sumo --jobs 3
  run_sumo $S classical --seedset EVAL --cells headline --channel default --backend sumo --jobs 2
  run_sumo $S rule --seedset SEL --cells headline --channel default --backend sumo --jobs 3
  run_sumo $S rule --seedset EVAL --cells headline --channel default --backend sumo --jobs 3
  run_sumo $S agent --seedset SEL --cells headline --channel default --backend sumo --runs $R10 --jobs 5
  run_sumo $S agent --seedset EVAL --cells headline --channel default --backend sumo --runs $R10 --jobs 5
  run $S agent --seedset SEL --cells grid24 --channel calibrated --runs $R5 --jobs 5
  run $S agent --seedset EVAL --cells grid24 --channel calibrated --runs $R5 --jobs 5
  run experiments.revision5.veins.custom_h3 --jobs 2
  run experiments.revision5.veins.d6_calibrate run --jobs 4
  run experiments.revision5.veins.d6_calibrate score
  run experiments.revision5.veins.d6_calibrate rerun --jobs 4
fi

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
A=experiments.revision5.analysis
run experiments.revision5.veins.d6_calibrate score
run $A.h1 --B 10000
run $A.h1 --B 10000 --D Dr
run $A.h2 --B 10000
run $A.h2 --B 10000 --D Dr
run $A.h2 --B 10000 --grid24
run $A.h3 --B 10000
run $A.h3 --mode sub_epoch
run $A.holm
run $A.secondary --B 10000
run $A.training_diag
run $A.compute_curves
run $A.ledger
run $A.tables
run $A.figures
run experiments.revision5.manifest
echo "done: results/revision5/analysis, results/revision5/figures, results/MANIFEST.md, results/manifest.json"
