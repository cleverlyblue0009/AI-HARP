#!/usr/bin/env bash
# Copy the H3 helper scripts into ~/veins-r5 (WSL clears /tmp when it idles), LF line endings.
R=/mnt/c/Users/UPASANA/AI-HARP/experiments/revision5/veins
for f in env.sh run_veins.py build_app.sh; do tr -d '\r' < "$R/$f" > "$HOME/veins-r5/$f"; done
