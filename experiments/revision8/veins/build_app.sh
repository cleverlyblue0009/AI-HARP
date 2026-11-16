#!/usr/bin/env bash
# Build the revision-8 Veins application (experiments/revision8/veins/aiharp) into ~/veins-r8/aiharp/aiharp.
# The revision-5, -6 and -7 apps and their sources are left untouched; both use the same Veins/OMNeT++ build.
set -euo pipefail
source "$HOME/veins-r5/env.sh"
SRC=${AIHARP_SRC:-/mnt/c/Users/UPASANA/AI-HARP}/experiments/revision8/veins/aiharp
DST=$HOME/veins-r8/aiharp
mkdir -p "$DST"
for f in "$SRC"/*; do tr -d '\r' < "$f" > "$DST/$(basename "$f")"; done
cd "$DST"
opp_makemake -f -o aiharp -I"$VEINS/src" -I"$VEINS/src/veins" -L"$VEINS/src" -lveins -DVEINS_IMPORT
make -j4 MODE=release
ls -l aiharp
sha256sum aiharp
