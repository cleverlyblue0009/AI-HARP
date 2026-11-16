#!/usr/bin/env bash
# Revision 8: regenerate results/revision8/analysis/ from committed rows (results/revision8/ADDENDUM.md).
#
#   ./reproduce_revision8.sh                  analysis only (each step runs only when its committed inputs exist)
#   ./reproduce_revision8.sh --from-scratch   retrain (80 generations x 20 runs), rerun the custom evaluation and, on the
#                                             laptop with WSL/Veins, the Veins gate, replays and scoring first
#
# Interpreter: NumPy 1.26.4 only (AIHARP_PY; default the laptop venv seen from Git Bash; in the cloud /root/r8env/bin/python).
# Training and STRUCT-rand need SUMO 1.19.0 (AIHARP_SUMO_HOME, exported as SUMO_HOME only for those steps; analysis runs
# with SUMO_HOME unset so backend "auto" cannot pick SUMO). --from-scratch MOVES existing outputs to *.bak-<timestamp>.
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
  for p in results/revision8/train results/revision8/runs results/revision8/veins/h5A_rows.jsonl \
           results/revision8/veins/h5B_rows.jsonl results/revision8/veins/expC_rows.jsonl results/revision8/veins/expD_rows.jsonl \
           cache/veins_runs/r8_gate cache/veins_runs/r8_h5A cache/veins_runs/r8_h5B cache/veins_runs/r8_expC cache/veins_runs/r8_expD; do
    if [[ -e "$p" ]]; then mv "$p" "$p.bak-$ts"; echo "moved $p -> $p.bak-$ts"; fi
  done
  run experiments.revision8.warm_start
  run experiments.revision8.veins_thresholds
  SUMO_HOME="${AIHARP_SUMO_HOME:?set AIHARP_SUMO_HOME to SUMO 1.19.0}" run experiments.revision8.traces --backend sumo
  run experiments.revision8.traces --backend fallback
  SUMO_HOME="$AIHARP_SUMO_HOME" PATH="$(dirname "$PY"):$PATH" bash experiments/revision8/train_all.sh
  for arm in plain rand; do for ss in SEL EVAL; do run experiments.revision8.sweep_struct --arm $arm --seedset $ss --jobs 4; done; done
  if command -v wsl >/dev/null 2>&1; then
    SUMO_HOME="$AIHARP_SUMO_HOME" run experiments.revision8.veins.validate8 unit --run plain_s0
    SUMO_HOME="$AIHARP_SUMO_HOME" run experiments.revision8.veins.validate8 unit --run rand_s0
    SUMO_HOME="$AIHARP_SUMO_HOME" run experiments.revision8.veins.validate8 geometric
    run experiments.revision8.veins.queue8 gate
    for m in h5B h5A expC expD; do run experiments.revision8.veins.queue8 $m --jobs 8; run experiments.revision8.veins.score8 $m; done
  fi
fi

# ---- analysis (no simulation) ---------------------------------------------------------------------------------------
[[ -f results/revision8/runs/struct_plain__EVAL__headline__default__fallback.csv ]] && run experiments.revision8.analysis.h4 --B 10000
[[ -f results/revision8/runs/struct_rand__EVAL__headline__default__fallback.csv ]] && run experiments.revision8.analysis.h4 --B 10000 --arm rand
[[ -f results/revision8/veins/h5B_rows.jsonl ]] && run experiments.revision8.analysis.h5 h5
run experiments.revision8.analysis.h5 expcd
[[ -f results/revision8/analysis/h4.json && -f results/revision8/analysis/h5.json ]] && run experiments.revision8.analysis.h5 holm
echo "done: results/revision8/analysis"
