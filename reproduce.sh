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

# Training runs on the GPU interpreter when there is one (ENVIRONMENT.md).
TRAIN_PY="${AIHARP_TRAIN_PYTHON:-D:/aiharp-gpu/Scripts/python.exe}"
if ! "$TRAIN_PY" -c "import torch" >/dev/null 2>&1; then TRAIN_PY="$PY"; fi

if [[ $SMOKE -eq 0 ]]; then
  step "4b. Training coverage targets on all 32 training seeds -> results/coverage_targets.json"
  "$PY" -m experiments.coverage_targets --seeds 32 --jobs 11
  step "4c. Sparse feasibility: every baseline setting at d=2 -> results/feasibility/"
  "$PY" -m experiments.sparse_feasibility --jobs 4
  step "4d. Lambda-cap follow-up (run8 pretrain, cap 500, sparse-only) -> results/feasibility/"
  for arm in "feas_A_cap500:" "feas_B_init500:--lambda-init 500"; do
    dir="checkpoints/${arm%%:*}"; extra="${arm#*:}"
    n=300; [[ "$dir" == *B_init500 ]] && n=600
    if [[ ! -f "$dir/run_summary.json" ]]; then
      # shellcheck disable=SC2086
      "$TRAIN_PY" -m agents.train --stage finetune --init-from checkpoints/run8/ckpt_pretrain.pt \
        --sparse-only --lambda-max 500 $extra --updates "$n" --out "$dir" --quiet --keep-awake
    fi
  done
  "$PY" -m experiments.lambda_cap_summary
fi

if [[ $TRAIN -eq 1 ]]; then
  step "5. Train the agent: dense pretrain, then sparse-weighted finetune"
  # Warm the trace cache first, or every rollout worker regenerates the same
  # traces during the first updates.
  if [[ $SMOKE -eq 0 ]]; then "$PY" -m experiments.warm_traces --jobs 8; fi
  # --keep-awake: without it, idle sleep dominated a multi-hour run on the
  # development laptop (one PPO update took 13,828 s instead of ~170 s).
  # Interrupted runs (ckpt_latest.pt but no run_summary.json) resume exactly.
  if [[ $SMOKE -eq 1 ]]; then
    "$PY" -m agents.train "${TRAIN_FLAGS[@]}" --out "$RUN_DIR" --workers auto --quiet --keep-awake
    cat "$RUN_DIR/run_summary.json"
  else
    # Reference agent, then one retrain per architectural ablation.
    "$TRAIN_PY" -m experiments.campaign_train
    RUN_DIR=checkpoints/campaign/ref
    cat "$RUN_DIR/run_summary.json"
  fi
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
