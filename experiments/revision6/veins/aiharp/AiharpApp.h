// Revision-5 H3: classical dissemination schemes in Veins, mirroring sim/engine.py's decision rules.
// Revision 6: plus the rule baselines (agents/rule_baselines.py) and the learned policy via a decision server.
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
    enum Kind { FLOODING, P_PERSISTENCE, SLOTTED_1P, WEIGHTED_P, COUNTER_BASED, GREEDY_FARTHEST, DVCAST,
                RAR, SCF_SLOTTED, AI_HARP };
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
        // revision 6: risk field (hazard/risk_field.py) and hazard (hazard/model.py), exported per bundle
        bool riskLoaded = false;
        bool grid = false;                                    // GridRiskGeometry, else HighwayRiskGeometry
        double etaFull = 60, etaTau = 60, insideW = 1, minSpeed = 1, gamma = 1, oppRel = 0;
        double spanStart = 0, spanEnd = 0, sev0 = 1, onset = 0, decayTau = 0, decayTtl = 0;
        int affected = 0, decay = 0;                          // decay: 0 none, 1 exponential, 2 linear_ttl
        double hx = 0, hy = 0, hHeading = 0, spanTol = 25, headingTol = M_PI / 2;
        bool hHeadingKnown = false;
        bool anyDirNonzero = true;                            // RiskField.direction_gate: np.any(direction != 0)
        std::map<int, int> dirOf;                             // trace index -> static carriageway sign
        std::map<int, std::string> vclassOf;                  // trace index -> vehicle class
    };
    static Shared S;
    static void loadShared(const std::string& nodesFile, const std::string& logFile);

    struct Neighbour { int idx; Coord pos; Coord vel; double dist; };
    struct NbEntry { Coord pos; Coord vel; double t; int dir = 0; };
    struct RelOut { double rel; double eta; };
    struct Decision { Act act = BROADCAST; int delay = 1; int cancel = NO_STEP; int relays[2] = {-1, -1}; };

    // policy
    Kind kind = FLOODING;
    double knob = 0;
    double rarA = 0, rarB = 0, rarC = 0, rarNmax = 30;       // rar: p = clip(a*rho + b*(1 - n/n_max) + c, 0, 1)
    int rarSlots = 5, scfSlots = 5;
    double scfTimeout = 5, carryStart = -1;                  // scf_slotted: carry start time (s), -1 = not carrying
    std::string bridgeAddr;
    int replaySeed = 0;
    int bridgeFd = -1;
    std::map<int, bool> heardFrom;                           // trace indices whose hazard frame this vehicle decoded
    int curSender = -1;                                      // sender of the frame that triggered the decision

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
    // revision 6
    bool wantsDupCallbacks() const { return kind == DVCAST || kind == SCF_SLOTTED || kind == AI_HARP; }
    Coord toTrace(const Coord& omnetPos) const;              // OMNeT++ -> SUMO network (trace) coordinates
    static Coord velToTrace(const Coord& omnetVel) { return Coord(omnetVel.x, -omnetVel.y); }
    static RelOut relevanceAt(double x, double y, double vx, double vy, int dir, double t);
    RelOut ownRelevance(int step, double& x, double& y, double& vx, double& vy) const;
    Decision bridgeDecide(int step, const char* trigger, double senderDist, int senderIdx, Coord senderPos,
                          bool designated);
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
