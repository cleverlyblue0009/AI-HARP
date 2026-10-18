# Source inside WSL: OMNeT++ 6.0.3, SUMO 1.19.0 and Veins 5.3 from install_wsl.sh, Linux-only PATH.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
VR5=$HOME/veins-r5
set +u; cd "$VR5/omnetpp-6.0.3" && source setenv -f >/dev/null; cd - >/dev/null
export SUMO_HOME=$VR5/sumo-1.19.0 PATH=$VR5/sumo-1.19.0/bin:$PATH VEINS=$VR5/veins-veins-5.3
