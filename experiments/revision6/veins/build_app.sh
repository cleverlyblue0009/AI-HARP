#!/usr/bin/env bash
# Build the revision-6 Veins application (experiments/revision6/veins/aiharp) into ~/veins-r6/aiharp/aiharp.
# The revision-5 app (~/veins-r5/aiharp) and its sources are left untouched; both use the same Veins/OMNeT++ build.
set -euo pipefail
source "$HOME/veins-r5/env.sh"
SRC=${AIHARP_SRC:-/mnt/c/Users/UPASANA/AI-HARP}/experiments/revision6/veins/aiharp
DST=$HOME/veins-r6/aiharp
mkdir -p "$DST"
for f in "$SRC"/*; do tr -d '\r' < "$f" > "$DST/$(basename "$f")"; done
cd "$DST"
opp_makemake -f -o aiharp -I"$VEINS/src" -I"$VEINS/src/veins" -L"$VEINS/src" -lveins -DVEINS_IMPORT
make -j4 MODE=release
ls -l aiharp
sha256sum aiharp
