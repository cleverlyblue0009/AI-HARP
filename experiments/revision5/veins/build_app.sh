#!/usr/bin/env bash
# Build the H3 Veins application (experiments/revision5/veins/aiharp) into ~/veins-r5/aiharp/aiharp.
set -euo pipefail
source /tmp/env.sh
SRC=/mnt/c/Users/UPASANA/AI-HARP/experiments/revision5/veins/aiharp
DST=$VR5/aiharp
mkdir -p "$DST"
for f in "$SRC"/*; do tr -d '\r' < "$f" > "$DST/$(basename "$f")"; done
cd "$DST"
opp_makemake -f -o aiharp -I"$VEINS/src" -I"$VEINS/src/veins" -L"$VEINS/src" -lveins -DVEINS_IMPORT
make -j4 MODE=release
ls -l aiharp
