#!/usr/bin/env bash
# Regenerate every revision-1 number, figure and table (results/revision1/).
#
#   bash experiments/revision1/reproduce_revision1.sh           # everything (many hours)
#   bash experiments/revision1/reproduce_revision1.sh --no-train # reuse checkpoints/revision1/
#
# Also run by `./reproduce.sh --revision1` after the main pipeline. Every step
# is resumable: per-run CSVs are appended and cached by task hash.

set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${AIHARP_PYTHON:-D:/aiharp-gpu/Scripts/python.exe}"
TRAIN_PY="${AIHARP_TRAIN_PYTHON:-$PY}"
JOBS="${AIHARP_JOBS:-4}"
TRAIN=1
for arg in "$@"; do
  case "$arg" in
    --no-train) TRAIN=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done
step() { printf '\n=== revision-1 %s ===\n' "$1"; }
CK=checkpoints/campaign/ref/ckpt_final.pt

step "A. facts from committed results"
"$PY" -m experiments.revision1.stage_a

step "B sweeps: headline (63 settings), slots, training seeds, generalization, 24 cells"
for s in headline slots train gen grid24; do "$PY" -m experiments.revision1.sweeps "$s" --jobs "$JOBS"; done

step "agent evaluations of the reference checkpoint (per seed)"
"$PY" -m experiments.revision1.agent_eval --checkpoint "$CK" --name ref_s0 --taus 0.5 0 --modes sampled --jobs "$JOBS"
"$PY" -m experiments.revision1.agent_eval --checkpoint "$CK" --name ref_s0 --taus 0.5 --modes argmax --jobs "$JOBS"
"$PY" -m experiments.revision1.agent_eval --checkpoint "$CK" --name ref_s0_randomgate \
  --random-gate-from results/revision1/agent_runs/ref_s0.csv --jobs "$JOBS"
"$PY" -m experiments.revision1.agent_eval --checkpoint "$CK" --name ref_s0_gen \
  --cells generalization --taus 0.5 --jobs "$JOBS"

step "B1-B5, C3"
"$PY" -m experiments.revision1.b1_latency
"$PY" -m experiments.revision1.b2_slots
"$PY" -m experiments.revision1.b3_feasibility
"$PY" -m experiments.revision1.b4_stats bootstrap
"$PY" -m experiments.revision1.b4_stats wilcoxon
"$PY" -m experiments.revision1.b5_modes
"$PY" -m experiments.revision1.c3_fallback

if [[ $TRAIN -eq 1 ]]; then
  step "C1/C2 training (extra seeds; final-checkpoint evaluation launched per run)"
  "$TRAIN_PY" -m experiments.revision1.train_queue --workers 5 --eval-jobs "$JOBS"
fi
for d in checkpoints/revision1/*/; do
  n=$(basename "$d")
  [[ -f "$d/ckpt_final.pt" ]] && "$PY" -m experiments.revision1.agent_eval --checkpoint "$d/ckpt_final.pt" \
    --name "$n" --taus 0.5 0 --modes sampled --jobs "$JOBS"
done
if [[ -f experiments/revision1/c_seeds.py ]]; then
  step "C1/C2/C4 tables"
  "$PY" -m experiments.revision1.c_seeds
fi
[[ -f experiments/revision1/c4_generalization.py ]] && "$PY" -m experiments.revision1.c4_generalization

step "D. figures and tables"
"$PY" -m experiments.revision1.stage_d
echo "results/revision1/ regenerated"
