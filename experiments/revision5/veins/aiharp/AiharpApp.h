// Revision-5 H3: classical dissemination schemes in Veins, mirroring sim/engine.py's decision rules.
#pragma once

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"
#include "AiharpMsg_m.h"

namespace veins {

class AiharpApp : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
    void finish() override;
    void receiveSignal(cComponent* source, simsignal_t signalID, bool b, cObject* details) override;
    using DemoBaseApplLayer::receiveSignal;
    /** No hazard transmission (or, before the TTL, carry timer) pending; origin done; window long enough. */
    static bool quiescent(simtime_t now, double minEnd);
    /** One "Q" log line with the early-stop state (diagnostics). */
    static void logState(simtime_t now, double minEnd);

protected:
    enum Kind { FLOODING, P_PERSISTENCE, SLOTTED_1P, WEIGHTED_P, COUNTER_BASED, GREEDY_FARTHEST, DVCAST };
    enum Ev { EV_ORIGIN = 100, EV_TX, EV_BEACON, EV_TIMER };
    enum Act { SUPPRESS, BROADCAST, DEFER, CARRY, RELAY };
    static constexpr int NO_STEP = -1;
    static constexpr double NEIGHBOUR_TTL_S = 1.05;   // beacon neighbour-table entry lifetime

    struct Shared {                       // one per run (single-process simulation)
        bool loaded = false;
        double t0 = 0, dt = 0.1, commRange = 0, ttl = 60;
        int nSteps = 0, maxHops = 32, originIdx = -1, originStep = -1;
        std::map<std::string, int> indexOf;                          // SUMO id -> trace index
        std::map<int, std::vector<std::pair<int, int>>> active;      // trace index -> [first, last] steps
        std::ofstream log;
        double beaconHz = 10, beaconDur = 0, csRange = 0, dccTarget = 0.62, dccMin = 1;
        bool dcc = true;
        int cwSlots = 1;
        double slotTime = 13e-6;
        std::map<int, Coord> pos;         // current position of every vehicle with a node (DCC neighbour count)
        int txPending = 0;                // armed transmission events, all vehicles
        int timerPending = 0;             // armed carry/re-decision timers, all vehicles
        bool originDone = false;
        double lastTx = -1;
    };
    static Shared S;
    static void loadShared(const std::string& nodesFile, const std::string& logFile);

    struct Neighbour { int idx; Coord pos; Coord vel; double dist; };
    struct NbEntry { Coord pos; Coord vel; double t; };
    struct Decision { Act act = BROADCAST; int delay = 1; int cancel = NO_STEP; int relays[2] = {-1, -1}; };

    // policy
    Kind kind = FLOODING;
    double knob = 0;

    // per-vehicle message state (sim/engine.py arrays, one entry per vehicle)
    int idx = -1;
    bool holds = false;
    int informedStep = NO_STEP, hops = 0, dupCount = 0, txSched = NO_STEP, timerSched = NO_STEP;
    int cancelAtDups = NO_STEP, txCount = 0;
    double maxSenderDist = 0;
    int pendingRelays[2] = {-1, -1};
    std::map<int, NbEntry> nbTable;       // from decoded beacons
    double busySince = -1, busyActive = 0, activeTime = 0, activeSince = -1;
    long beaconsSent = 0;
    cMessage* txEvt = nullptr;
    cMessage* timerEvt = nullptr;
    cMessage* beaconEvt = nullptr;
    cMessage* originEvt = nullptr;

    void onWSM(BaseFrame1609_4* wsm) override;
    void onBSM(DemoSafetyMessage* bsm) override;
    void handleSelfMsg(cMessage* msg) override;
    void handlePositionUpdate(cObject* obj) override;

    int stepAt(simtime_t t) const;
    bool isActive(int step) const;
    static bool activeAt(int i, int step);
    simtime_t timeOf(int step) const;
    double beaconRate(int step) const;
    std::vector<Neighbour> neighbours() const;
    Decision policyDecide(const char* trigger, double senderDist, bool hasSender, Coord senderPos, bool duplicate,
                          bool designated);
    void decide(int step, const char* trigger, double senderDist, bool hasSender = false, Coord senderPos = Coord(),
                bool duplicate = false, bool designated = false);
    void scheduleTx(int step);
    void disarmTx();
    void scheduleTimer(int step);
    void disarmTimer();
    void transmit(int step);
};

} // namespace veins
