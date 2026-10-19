#!/usr/bin/env bash
# Revision-5 offload VM setup (Ubuntu 24.04 x64). Idempotent; log in ~/setup.log; ~/SETUP_DONE when finished.
#   COMMIT=<sha> bash setup_cloud_vm.sh
# Python 3.12 venv with ENVIRONMENT.md pins (numpy 1.26.4, torch 2.2.2 CPU, torch-geometric 2.8.0, and the
# exact versions of the laptop environment for the rest); OMNeT++ 6.0.3 / SUMO 1.19.0 / Veins 5.3 built by
# the same installer as the WSL toolchain; the public repo cloned at COMMIT (no credentials on the VM).
set -euo pipefail
: "${COMMIT:?set COMMIT to the revision-5 commit to check out}"
exec > >(tee -a "$HOME/setup.log") 2>&1
echo "== setup start $(date -Is) commit $COMMIT"

sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-venv python3-pip git rsync tmux

if [ ! -d "$HOME/AI-HARP/.git" ]; then
  git clone -q --branch revision-5 https://github.com/cleverlyblue0009/AI-HARP.git "$HOME/AI-HARP"
fi
cd "$HOME/AI-HARP"
git fetch -q origin revision-5
git checkout -q "$COMMIT"
git config --global --add safe.directory "$HOME/AI-HARP"

if [ ! -x "$HOME/venv/bin/python" ]; then python3.12 -m venv "$HOME/venv"; fi
V="$HOME/venv/bin/pip"
$V install -q --upgrade pip
$V install -q numpy==1.26.4
$V install -q torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
$V install -q torch-geometric==2.8.0 scipy==1.13.1 pandas==2.3.3 PyYAML==6.0.3 networkx==3.6.1 psutil==7.2.2 \
              tqdm==4.70.1 pytest
$V install -q matplotlib==3.11.2 || echo "WARN: matplotlib==3.11.2 not installable (not used on the VM)"
"$HOME/venv/bin/python" -c "import numpy, torch, torch_geometric, scipy, pandas; print('python ok', numpy.__version__, torch.__version__, torch_geometric.__version__)"

sudo bash experiments/revision5/veins/install_wsl.sh deps
J=4 bash experiments/revision5/veins/install_wsl.sh build
tr -d '\r' < experiments/revision5/veins/env.sh > "$HOME/veins-r5/env.sh"
tr -d '\r' < experiments/revision5/veins/run_veins.py > "$HOME/veins-r5/run_veins.py"
AIHARP_SRC="$HOME/AI-HARP" bash experiments/revision5/veins/build_app.sh

echo "== setup done $(date -Is)"
touch "$HOME/SETUP_DONE"
