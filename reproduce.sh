#!/usr/bin/env bash
# Regenerate every number, figure and table in the paper from scratch.
#
#   ./reproduce.sh            full run (hours)
#   ./reproduce.sh --smoke    end-to-end on tiny settings (minutes)
#   ./reproduce.sh --no-train reuse the committed checkpoint instead of training
#
# Nothing here takes a number on faith: every stage writes into results/ with a
# config hash and a metrics_version, and the figures read only those files.

set -euo pipefail
cd "$(dirname "$0")"

# The learning stack lives in the D: environment (see ENVIRONMENT.md); C: was
# full when torch was installed. Phases 1-4 run under either interpreter.
PY="${AIHARP_PYTHON:-D:/aiharp-env/python.exe}"
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  echo "!! $PY is not usable. Set AIHARP_PYTHON to a working interpreter." >&2
  exit 1
fi

SMOKE=0
TRAIN=1
for arg in "$@"; do
  case "$arg" in
    --smoke)    SMOKE=1 ;;
    --no-train) TRAIN=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

SEEDS=10
RUN_DIR=checkpoints/reproduce
TRAIN_FLAGS=()
if [[ $SMOKE -eq 1 ]]; then SEEDS=2; RUN_DIR=checkpoints/reproduce_smoke; TRAIN_FLAGS=(--smoke); fi

step() { printf '\n=== %s ===\n' "$1"; }

step "0. Environment and test suite"
"$PY" -c "import sys, numpy, torch; print(sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)" || true
"$PY" -m pytest tests/ -q

step "1. Pipeline smoke test"
"$PY" -m experiments.run_sim --smoke

step "2. Constants provenance table"
"$PY" -m analysis.constants_table | tee results/constants_provenance.txt

step "3. Baseline comparison (paired, $SEEDS seeds) -> results/runs.csv"
"$PY" -m experiments.compare --quiet --seeds "$SEEDS"

step "4. Operating curves and reference points -> results/pareto_cells.json"
"$PY" -m analysis.comparator --seeds "$SEEDS" --quiet

if [[ $TRAIN -eq 1 ]]; then
  step "5. Train the agent: dense pretrain, then sparse-weighted finetune -> $RUN_DIR"
  # Warm the trace cache first, or every rollout worker regenerates the same
  # traces during the first updates.
  if [[ $SMOKE -eq 0 ]]; then "$PY" -m experiments.warm_traces --jobs 8; fi
  # --keep-awake: without it, idle sleep dominated a multi-hour run on the
  # development laptop (one PPO update took 13,828 s instead of ~170 s).
  # A run interrupted part-way (ckpt_latest.pt but no run_summary.json) is
  # resumed exactly rather than restarted.
  if [[ -f "$RUN_DIR/ckpt_latest.pt" && ! -f "$RUN_DIR/run_summary.json" ]]; then
    "$PY" -m agents.train --resume "$RUN_DIR/ckpt_latest.pt" --workers auto --quiet --keep-awake
  else
    "$PY" -m agents.train "${TRAIN_FLAGS[@]}" --out "$RUN_DIR" --workers auto --quiet --keep-awake
  fi
  cat "$RUN_DIR/run_summary.json"
else
  step "5. Training skipped (--no-train); reusing $RUN_DIR"
fi

step "5b. Evaluate the trained agent against the baselines -> results/agent"
"$PY" -m experiments.evaluate_agent --checkpoint "$RUN_DIR/ckpt_final.pt" --seeds "$SEEDS" \
  --out-dir results/agent

step "6. Figures and tables -> results/figures, results/tables"
"$PY" -m analysis.report

step "Done"
echo "results/  figures/ tables/ runs.csv pareto_cells.json constants_provenance.txt"
