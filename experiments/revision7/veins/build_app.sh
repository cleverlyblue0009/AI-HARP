#!/usr/bin/env bash
# Build the revision-7 Veins application (experiments/revision7/veins/aiharp) into ~/veins-r7/aiharp/aiharp.
# The revision-5 and revision-6 apps and their sources are left untouched; both use the same Veins/OMNeT++ build.
set -euo pipefail
source "$HOME/veins-r5/env.sh"
SRC=${AIHARP_SRC:-/mnt/c/Users/UPASANA/AI-HARP}/experiments/revision7/veins/aiharp
DST=$HOME/veins-r7/aiharp
mkdir -p "$DST"
for f in "$SRC"/*; do tr -d '\r' < "$f" > "$DST/$(basename "$f")"; done
cd "$DST"
opp_makemake -f -o aiharp -I"$VEINS/src" -I"$VEINS/src/veins" -L"$VEINS/src" -lveins -DVEINS_IMPORT
make -j4 MODE=release
ls -l aiharp
sha256sum aiharp
