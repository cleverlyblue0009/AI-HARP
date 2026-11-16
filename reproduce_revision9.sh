#!/usr/bin/env bash
# Revision 9: regenerate results/revision9 analysis outputs from committed rows (results/revision9/ADDENDUM.md).
#
#   ./reproduce_revision9.sh                  analysis only (each step runs only when its committed inputs exist)
#   ./reproduce_revision9.sh --from-scratch   print how to rerun the runner work (gates, training, Veins, S2)
#
# Interpreter: NumPy 1.26.4 only (AIHARP_PY; cloud default /root/r8env/bin/python). SUMO_HOME is unset so backend
# "auto" cannot pick SUMO.
set -euo pipefail
cd "$(dirname "$0")"
PY="${AIHARP_PY:-/root/r8env/bin/python}"
export PYTHONIOENCODING=utf-8
unset SUMO_HOME
envline="$("$PY" -c 'import sys, numpy; print(sys.executable, numpy.__version__)' | tr -d '\r')"
echo "[env] $envline"
[[ "$envline" == *" 1.26.4" ]] || { echo "wrong interpreter or NumPy version (need NumPy 1.26.4): $envline" >&2; exit 1; }
run() { echo ">> python -m $*"; "$PY" -u -m "$@"; }
if [[ "${1:-}" == "--from-scratch" ]]; then
  cat <<'TXT'
The heavy work runs on GitHub-hosted runners (D14 of revision 8; ADDENDUM section 7). To rerun it, bump "request" in
the request file and push to revision-9; outputs land on branch r9-ci:
  ci/r9-gates.json  -> 7,552 training-pool episodes (gates, miss targets)
  ci/r9-train.json  -> 10 STRUCT9 runs
  ci/r9-veins.json  -> {"mode": "h6" | "c1", "seeds": [220..229]}: 20 Veins shards each
  ci/r9-eval.json   -> S2 custom-simulator rows (20 slices per arm)
then copy the outputs from r9-ci (git checkout origin/r9-ci -- <paths>; never commit bridge_logs) and run this script.
TXT
  exit 0
fi
ls results/revision9/gates_runs_slice*of20.csv >/dev/null 2>&1 && run experiments.revision9.gates analyse
[[ -f results/revision9/veins/h6_rows.jsonl ]] && run experiments.revision9.analysis.veins9 h6
[[ -f results/revision9/veins/c1_rows.jsonl ]] && run experiments.revision9.analysis.veins9 c1
[[ -f results/revision9/veins/h6_rows.jsonl && -f results/revision9/veins/c1_rows.jsonl ]] && run experiments.revision9.analysis.veins9 s1
[[ -f results/revision9/runs/struct9_rand__EVAL__headline__default__fallback.csv ]] && run experiments.revision9.analysis.s2 --B 10000
echo "done: results/revision9"
