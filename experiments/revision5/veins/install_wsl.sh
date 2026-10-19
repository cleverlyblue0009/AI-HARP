#!/usr/bin/env bash
# Revision-5 H3 toolchain in WSL2 Ubuntu 24.04: OMNeT++ 6.0.3 (core, no GUI), Veins 5.3, SUMO 1.19.0.
#   wsl -d Ubuntu -u root -- bash experiments/revision5/veins/install_wsl.sh deps
#   wsl -d Ubuntu        -- bash experiments/revision5/veins/install_wsl.sh build
# Idempotent: each step is skipped when its product exists. Everything lives in ~/veins-r5.
set -euo pipefail
J=${J:-4}
P=$HOME/veins-r5
step=${1:-build}
# Windows PATH leaks into WSL (CMake found miniconda's Windows gtest.lib): keep Linux paths only.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)

if [ "$step" = deps ]; then
  apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential clang lld gdb bison flex perl \
    python3 python3-pip python3-venv python3-numpy python3-scipy python3-pandas python3-matplotlib \
    libxml2-dev zlib1g-dev doxygen graphviz cmake git wget \
    libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
  exit 0
fi

mkdir -p "$P" && cd "$P"
if [ ! -x "$P/omnetpp-6.0.3/bin/opp_run" ]; then
  [ -f omnetpp-6.0.3-core.tgz ] || wget -q https://github.com/omnetpp/omnetpp/releases/download/omnetpp-6.0.3/omnetpp-6.0.3-core.tgz
  [ -d omnetpp-6.0.3 ] || tar xzf omnetpp-6.0.3-core.tgz
  cd omnetpp-6.0.3
  sed -i 's/^WITH_QTENV=yes/WITH_QTENV=no/; s/^WITH_OSG=yes/WITH_OSG=no/; s/^WITH_OSGEARTH=yes/WITH_OSGEARTH=no/' configure.user
  set +u; source setenv -f; set -u
  ./configure WITH_QTENV=no WITH_OSG=no WITH_OSGEARTH=no
  make -j"$J" MODE=release base
  cd "$P"
fi

if [ ! -x "$P/sumo-1.19.0/bin/duarouter" ]; then
  [ -f sumo-src-1.19.0.tar.gz ] || wget -q https://sumo.dlr.de/releases/1.19.0/sumo-src-1.19.0.tar.gz
  [ -d sumo-1.19.0 ] || tar xzf sumo-src-1.19.0.tar.gz
  cmake -S sumo-1.19.0 -B sumo-1.19.0/build/cmake-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON -DCMAKE_IGNORE_PREFIX_PATH=/mnt/c
  cmake --build sumo-1.19.0/build/cmake-build -j"$J"
fi

if [ ! -f "$P/veins-veins-5.3/src/libveins.so" ]; then
  [ -f veins-5.3.tar.gz ] || wget -q -O veins-5.3.tar.gz https://github.com/sommer/veins/archive/refs/tags/veins-5.3.tar.gz
  [ -d veins-veins-5.3 ] || tar xzf veins-5.3.tar.gz
  cd omnetpp-6.0.3 && set +u; source setenv -f; set -u && cd "$P/veins-veins-5.3"
  ./configure
  make -j"$J" MODE=release
fi

echo "versions:"
"$P/omnetpp-6.0.3/bin/opp_run" -v | head -2 || true
"$P/sumo-1.19.0/bin/sumo" --version | head -1 || true
ls -l "$P/veins-veins-5.3/src/libveins.so"
