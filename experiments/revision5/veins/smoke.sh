#!/usr/bin/env bash
# Toolchain smoke test: Veins' bundled Erlangen example, 60 s of simulated time, Cmdenv.
source "$(dirname "$0")/env.sh"
cd "$VEINS"
bin/veins_launchd -vv -c sumo -L /tmp/launchd.log -d
sleep 2
cd examples/veins && timeout 900 ./run -u Cmdenv -c Default --sim-time-limit=60s 2>&1 | tail -5
pkill -f veins_launchd; ls -1 results | head -5
