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

step "3. Full factorial baseline grid (paired, $SEEDS seeds) -> results/runs.csv"
# The paper's evidence base: 3 scenarios x 8 densities x 4 weathers x 5 hazards
# x 9 baselines x seeds, with train/held-out labels on every row.
# experiments/compare.py writes the SAME file from one cell family, so it is
# deliberately not used here -- it would leave runs.csv holding a different grid
# from the one analysis/grid_stats.py and the figures assume.
if [[ $SMOKE -eq 1 ]]; then
  "$PY" -m experiments.full_sweep --jobs 2 --seeds "$SEEDS" \
    --scenarios rural_highway --densities 2 20 --weathers clear --hazards fog_bank
else
  "$PY" -m experiments.full_sweep --jobs 3 --seeds "$SEEDS"
fi

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

  step "4e. Causal-vs-oracle risk estimation agreement -> results/risk_estimation.csv"
  # Independent of any policy, so it is one sweep rather than a runs.csv column.
  "$PY" -m experiments.risk_estimation --jobs 3

  step "4f. Simulator validation against a published curve -> results/validation/"
  "$PY" -m experiments.validate_amador --seeds 30 --jobs 3
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
# Sampled is the headline: it is the policy the coverage constraint trained.
"$PY" -m experiments.evaluate_agent --checkpoint "$RUN_DIR/ckpt_final.pt" --seeds "$SEEDS" \
  --policy-mode sampled --out-dir results/agent

if [[ $SMOKE -eq 0 ]]; then
  # Argmax beside it: the mode moves a cell by up to ~0.03 RWCR and decides
  # whether urban d=20 clears the matched-quality bar at all.
  "$PY" -m experiments.evaluate_agent --checkpoint "$RUN_DIR/ckpt_final.pt" --seeds "$SEEDS" \
    --policy-mode argmax --out-dir results/agent_argmax

  step "5c. Agent rows for the full grid -> results/runs.csv"
  "$PY" -m experiments.full_sweep --no-baselines --checkpoint "$RUN_DIR/ckpt_final.pt" \
    --taus 0 0.5 --jobs 3 --seeds "$SEEDS"

  step "5d. Ablation evaluations -> results/agent_<ablation>/"
  "$PY" -m experiments.evaluate_ablations --skip-ref --seeds "$SEEDS"

  step "5e. Slot-granularity sensitivity -> results/slot_granularity.csv"
  "$PY" -m experiments.slot_granularity --jobs 2 --seeds "$SEEDS"

  step "5f. Paired statistics over the grid -> results/stats/"
  "$PY" -m analysis.grid_stats

  if [[ -n "${SUMO_HOME:-}" ]]; then
    step "5g. Headline cells on SUMO, then on the real OSM maps"
    # Separate cell files: the committed results/pareto_cells.json is the
    # fallback-backend evidence the rest of the paper is built on.
    "$PY" -m analysis.comparator --seeds "$SEEDS" --quiet --backend sumo \
      --out results/pareto_cells_sumo.json
    "$PY" -m analysis.comparator --seeds "$SEEDS" --quiet --backend sumo \
      --scenario-map rural_highway=rural_highway_osm,urban_nlos=urban_grid_osm \
      --out results/pareto_cells_osm.json

    step "5g2. The sparse band on SUMO -> results/runs_sparse_sumo.csv"
    # 14,400 runs. The point is not the absolute numbers but whether the
    # ORDERING of policies survives the backend change, which is what every
    # comparative claim in the paper depends on.
    "$PY" -m experiments.full_sweep --backend sumo --jobs 2 --seeds "$SEEDS" \
      --scenarios rural_highway urban_nlos --densities 1 2 3 5 \
      --out results/runs_sparse_sumo.csv
    "$PY" -m analysis.backend_compare
  else
    echo "SUMO_HOME is not set: skipping the SUMO and real-map headline re-runs."
  fi

  step "5h. Attention over one dissemination event -> results/attention_event.json"
  # Figure 7 is drawn from this file. It used to be made by hand, which meant
  # this script could not regenerate a figure the paper publishes.
  "$PY" -m experiments.attention_event --checkpoint "$RUN_DIR/ckpt_final.pt"
fi

step "6. Figures and tables -> results/figures, results/tables"
"$PY" -m analysis.report

step "Done"
echo "results/  figures/ tables/ runs.csv risk_estimation.csv slot_granularity.csv"
echo "          stats/ validation/ feasibility/ pareto_cells*.json agent*/"
