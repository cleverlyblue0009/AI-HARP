// Revision-5 H3: TraCI manager for the replay.
//
// 1. Warm-up skip. firstStepAt = t0 - 1 s makes the first TraCI step advance SUMO through the whole
//    warm-up in one call (SUMO's own stepping and RNG are untouched; the per-run FCD identity check
//    verifies it). The departed-vehicles subscription only reports the last step, so after that first
//    step every vehicle already in SUMO is subscribed explicitly -- which creates its Veins node.
// 2. Early stop. Once per step: when no hazard transmission is pending anywhere, the last one is at
//    least 1 s old, and the measurement window has reached max(t0 + 20 s, origin + 10 s), nothing in the
//    hazard log can change any more for the implemented (non-carrying) schemes; the run ends there.

#include "AiharpManager.h"

#include "AiharpApp.h"

using namespace veins;

Define_Module(veins::AiharpManager);

void AiharpManager::initialize(int stage)
{
    TraCIScenarioManagerForker::initialize(stage);
    if (stage == 0) {
        t0 = par("t0").doubleValue();
        minEnd = par("minEnd").doubleValue();
    }
}

void AiharpManager::handleSelfMsg(cMessage* msg)
{
    bool step = msg == executeOneTimestepTrigger;
    TraCIScenarioManagerForker::handleSelfMsg(msg);
    if (!step || !isConnected()) return;
    if (!caughtUp) {
        caughtUp = true;
        for (auto& id : getCommandInterface()->getVehicleIds())
            if (subscribedVehicles.find(id) == subscribedVehicles.end()) subscribeToVehicleVariables(id);
    }
    if (AiharpApp::quiescent(simTime(), minEnd)) {
        recordScalar("earlyStopAt", simTime().dbl());
        endSimulation();
    }
}
