#!/usr/bin/env bash
# On a VM, in tmux:  bash run_vm.sh <slice> <commit>
# Update to <commit>, run the gate, and only if it passes run this VM's Veins slice (restarted until complete).
set -u
SLICE=$1; COMMIT=$2
cd "$HOME/AI-HARP" && git fetch -q origin revision-5 && git checkout -q "$COMMIT" || exit 1
tr -d '\r' < experiments/revision5/veins/run_veins.py > "$HOME/veins-r5/run_veins.py"
set +u; source "$HOME/veins-r5/env.sh"; set -u
export SUMO_HOME="$HOME/veins-r5/sumo-1.19.0" AIHARP_REPO="$HOME/AI-HARP"
cd "$HOME/AI-HARP"
echo "$(date -Is) gate start (commit $COMMIT)" >> "$HOME/run_vm.log"
if ! "$HOME/venv/bin/python" -m experiments.revision5.cloud.gate --ref "$HOME/gate_ref" > "$HOME/gate.log" 2>&1; then
  echo "$(date -Is) GATE FAILED -- no work started (see ~/GATE.json, ~/gate.log)" >> "$HOME/run_vm.log"; exit 1
fi
echo "$(date -Is) gate PASSED; starting Veins slice $SLICE/3" >> "$HOME/run_vm.log"
until grep -q "slice complete" "$HOME/vm_veins.log" 2>/dev/null; do
  "$HOME/venv/bin/python" -u -m experiments.revision5.cloud.vm_veins --slice "$SLICE" --n 3 --seeds 200-229 --jobs 3 \
      --skip "$HOME/skip.txt" >> "$HOME/vm_veins.log" 2>&1
  sleep 30
done
echo "$(date -Is) Veins slice $SLICE complete" >> "$HOME/run_vm.log"
