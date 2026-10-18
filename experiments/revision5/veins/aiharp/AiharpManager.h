// Revision-5 H3: TraCI manager that skips the SUMO warm-up's per-step overhead.
#pragma once

#include "veins/modules/mobility/traci/TraCIScenarioManagerForker.h"

namespace veins {

class AiharpManager : public TraCIScenarioManagerForker {
protected:
    bool caughtUp = false;
    double t0 = 0, minEnd = 0;
    void initialize(int stage) override;
    void handleSelfMsg(cMessage* msg) override;
};

} // namespace veins
