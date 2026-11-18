#!/usr/bin/env bash
# Revision 10: regenerate results/revision10 analysis outputs from committed files (results/revision10/ADDENDUM.md).
#
#   ./reproduce_revision10.sh                  analysis only (each step runs only when its committed inputs exist)
#   ./reproduce_revision10.sh --from-scratch   print how to rerun the runner work (gates, Veins shards)
#
# Interpreter: NumPy 1.26.4 only (AIHARP_PY; cloud default /root/r8env/bin/python). SUMO_HOME is unset.
# Not rerun here: experiments.revision10.seed_inventory (it fetches branches r8-ci / r9-ci; run it by hand to check
# seeds.json). Outputs are identical to the committed files apart from their "provenance" stamps.
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
The heavy work runs on GitHub-hosted runners. Bump "request" in the request file and push to revision-10; outputs land
on branch r10-ci:
  ci/r10-gates.json -> step-1 gates: pipeline (2 runners), checkpoint, bundles for seeds 230-249 (no replay)
  ci/r10-veins.json -> {"mode": "h7" | "s3" | "s5", "seeds": [...]}: one Veins shard per seed (60 replays each)
then copy the outputs (git checkout origin/r10-ci -- results/revision10/gate results/revision10/veins/<mode>; then
git rm -r --cached any bridge_logs / gate2_logs) and run this script.
TXT
  exit 0
fi
ls results/revision10/gate/pipeline_*.json >/dev/null 2>&1 && run experiments.revision10.veins.gate10 assemble
for m in h7 s3 s5; do
  [[ -d results/revision10/veins/$m ]] && run experiments.revision10.analysis.veins10 assemble $m
done
[[ -f results/revision10/veins/h7_rows.jsonl ]] && run experiments.revision10.analysis.veins10 h7
[[ -f results/revision10/veins/h7_rows.jsonl ]] && run experiments.revision10.analysis.veins10 s4
[[ -f results/revision10/veins/h7_rows.jsonl && -f results/revision10/veins/s3_rows.jsonl ]] && run experiments.revision10.analysis.veins10 s3
[[ -f results/revision10/veins/s5_rows.jsonl ]] && run experiments.revision10.analysis.veins10 s5
echo "done: results/revision10"
