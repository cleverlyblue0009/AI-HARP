// Revision-5 H3: classical dissemination schemes in Veins, mirroring sim/engine.py.
//
// Timing (engine semantics): decisions happen in 100 ms epochs aligned to the trace (step k <-> SUMO time
// t0 + k*dt). BROADCAST -> transmit at step+1; DEFER(d) -> step+max(1,d); a pending transmission is
// cancelled once dupCount >= cancelAtDups. TTL (60 s) and the hop limit (32) are checked before sending.
// A vehicle takes part only in steps where it is active in the trace (in the corridor); elsewhere it
// neither transmits, beacons nor processes receptions. Policy decisions (agents/*.py) are reproduced
// rule for rule; the policy random draws come from OMNeT++'s RNG (seed-set = run seed).
//
// Log lines (CSV): O,t,step,idx | T,t,step,idx,hops,sched_t | R,t,step,idx,sender,dist,new,hops
//                  D,t,step,idx,trigger,action,delay | X,t,step,idx,reason

#include "AiharpApp.h"

#include <cmath>
#include <sstream>

#include "veins/modules/mac/ieee80211p/Mac1609_4.h"

using namespace veins;

Define_Module(veins::AiharpApp);

AiharpApp::Shared AiharpApp::S;

void AiharpApp::loadShared(const std::string& nodesFile, const std::string& logFile)
{
    if (S.loaded) return;
    std::ifstream in(nodesFile);
    if (!in) throw cRuntimeError("cannot open %s", nodesFile.c_str());
    std::string line, key;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        ss >> key;
        if (key == "t0") ss >> S.t0;
        else if (key == "dt") ss >> S.dt;
        else if (key == "n_steps") ss >> S.nSteps;
        else if (key == "comm_range_m") ss >> S.commRange;
        else if (key == "max_hops") ss >> S.maxHops;
        else if (key == "ttl_s") ss >> S.ttl;
        else if (key == "origin") ss >> S.originIdx >> S.originStep;
        else if (key == "v") {          // v <index> <sumo_id> <first> <last> [<first> <last> ...]
            int i, a, b;
            std::string id;
            ss >> i >> id;
            S.indexOf[id] = i;
            while (ss >> a >> b) S.active[i].push_back({a, b});
        }
    }
    S.log.open(logFile);
    if (!S.log) throw cRuntimeError("cannot open %s", logFile.c_str());
    S.log.precision(9);
    S.loaded = true;
}

void AiharpApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        loadShared(par("nodesFile").stdstringValue(), par("logFile").stdstringValue());
        std::string p = par("policy").stdstringValue();
        if (p == "flooding") kind = FLOODING;
        else if (p == "p_persistence") kind = P_PERSISTENCE;
        else if (p == "slotted_1p") kind = SLOTTED_1P;
        else if (p == "weighted_p") kind = WEIGHTED_P;
        else if (p == "counter_based") kind = COUNTER_BASED;
        else throw cRuntimeError("unknown policy %s", p.c_str());
        knob = par("knob").doubleValue();
        std::string sumoId = mobility->getExternalId();
        auto it = S.indexOf.find(sumoId);
        idx = it == S.indexOf.end() ? -1 : it->second;   // vehicles never in the corridor stay silent
        txEvt = new cMessage("tx", EV_TX);
        beaconEvt = new cMessage("beacon", EV_BEACON);
        originEvt = new cMessage("origin", EV_ORIGIN);
        if (cModule* m = getParentModule()->getSubmodule("nic")->getSubmodule("mac1609_4"))
            m->subscribe(Mac1609_4::sigChannelBusy, this);
    }
    else if (stage == 1 && idx >= 0) {
        if (par("beacons").boolValue())
            scheduleAt(simTime() + dblrand() * beaconInterval, beaconEvt);   // random phase, as Veins does
        if (idx == S.originIdx && timeOf(S.originStep) >= simTime())
            scheduleAt(timeOf(S.originStep), originEvt);
    }
}

bool AiharpApp::quiescent(simtime_t now, double minEnd)
{
    return S.loaded && S.originDone && S.pending == 0 && now.dbl() >= minEnd && now.dbl() - S.lastTx >= 1.0;
}

simtime_t AiharpApp::timeOf(int step) const { return S.t0 + step * S.dt; }

int AiharpApp::stepAt(simtime_t t) const { return (int) std::floor((t.dbl() - S.t0) / S.dt + 1e-6); }

bool AiharpApp::isActive(int step) const
{
    if (idx < 0 || step < 0 || step >= S.nSteps) return false;
    auto it = S.active.find(idx);
    if (it == S.active.end()) return false;
    for (auto& r : it->second)
        if (step >= r.first && step <= r.second) return true;
    return false;
}

void AiharpApp::receiveSignal(cComponent* source, simsignal_t signalID, bool b, cObject* details)
{
    if (signalID != Mac1609_4::sigChannelBusy) return cListener::receiveSignal(source, signalID, b, details);
    double now = simTime().dbl();
    if (b) busySince = now;
    else if (busySince >= 0) {
        if (isActive(stepAt(busySince)) && isActive(stepAt(now))) busyActive += now - busySince;
        busySince = -1;
    }
}

void AiharpApp::handleSelfMsg(cMessage* msg)
{
    int step = stepAt(simTime());
    switch (msg->getKind()) {
    case EV_BEACON: {
        if (isActive(step)) {
            DemoSafetyMessage* bsm = new DemoSafetyMessage("beacon");
            populateWSM(bsm);
            sendDown(bsm);
            beaconsSent++;
            if (activeSince < 0) activeSince = simTime().dbl();
        }
        else if (activeSince >= 0) {
            activeTime += simTime().dbl() - activeSince;
            activeSince = -1;
        }
        scheduleAt(simTime() + beaconInterval, beaconEvt);
        break;
    }
    case EV_ORIGIN: {
        holds = true;
        informedStep = step;
        hops = 0;
        S.log << "O," << simTime() << "," << step << "," << idx << "\n";
        decide(step, "originate", 0.0);
        break;
    }
    case EV_TX:
        S.pending--;                                         // the event has fired
        transmit(step);
        break;
    default:
        DemoBaseApplLayer::handleSelfMsg(msg);
    }
}

void AiharpApp::decide(int step, const char* trigger, double senderDist)
{
    // agents/*.py decide(); ORIGINATE always broadcasts.
    std::string action = "broadcast";
    int delay = 1, cancel = NO_STEP;
    bool originate = std::string(trigger) == "originate";
    if (!originate) {
        double R = S.commRange;
        switch (kind) {
        case FLOODING: break;
        case P_PERSISTENCE:
            if (!(uniform(0, 1) < knob)) action = "suppress";
            break;
        case WEIGHTED_P: {
            double p = R <= 0 ? knob : std::min(std::max(senderDist / R, 0.0), knob);   // min_p = 0, max_p = knob
            if (!(uniform(0, 1) < p)) action = "suppress";
            break;
        }
        case SLOTTED_1P: {                                   // p = 1, slot_epochs = 1, cancel_on_duplicates = 1
            int n = (int) knob;
            double frac = R <= 0 ? 1.0 : std::min(std::max(senderDist, 0.0), R) / R;
            int slot = std::min(std::max((int) std::floor(n * (1.0 - frac)), 0), n - 1);
            action = "defer"; delay = 1 + slot; cancel = 1;
            break;
        }
        case COUNTER_BASED: {                                // max_delay_epochs = 5, min_distance_fraction = 0.35
            if (senderDist < 0.35 * R) { action = "suppress"; break; }
            action = "defer"; delay = 1 + intuniform(0, 4); cancel = (int) knob - 1;
            break;
        }
        }
    }
    S.log << "D," << simTime() << "," << step << "," << idx << "," << trigger << "," << action << "," << delay << "\n";
    if (cancel != NO_STEP) cancelAtDups = cancel;
    if (action == "suppress") {
        txSched = NO_STEP;
        disarm();
    }
    else
        schedule(step + std::max(1, delay));
}

void AiharpApp::schedule(int step)
{
    txSched = step;
    disarm();
    scheduleAt(timeOf(step), txEvt);
    S.pending++;
}

void AiharpApp::disarm()
{
    if (txEvt->isScheduled()) {
        cancelEvent(txEvt);
        S.pending--;
    }
}

void AiharpApp::transmit(int step)
{
    if (txSched != step || !holds) return;
    txSched = NO_STEP;
    if (!isActive(step)) return;                             // engine: pending requires act[step]
    double age = (step - S.originStep) * S.dt;
    if (hops >= S.maxHops || age > S.ttl) {
        S.log << "X," << simTime() << "," << step << "," << idx << ",ttl\n";
        return;
    }
    AiharpMsg* m = new AiharpMsg("hazard");
    populateWSM(m);
    m->setHops(hops);
    m->setSenderIdx(idx);
    m->setSenderPos(curPosition);
    m->setSchedTime(timeOf(step).dbl());
    sendDown(m);
    txCount++;
    S.lastTx = simTime().dbl();
    S.log << "T," << simTime() << "," << step << "," << idx << "," << hops << "," << timeOf(step) << "\n";
}

void AiharpApp::onWSM(BaseFrame1609_4* wsm)
{
    AiharpMsg* m = dynamic_cast<AiharpMsg*>(wsm);
    if (!m) return;
    int step = stepAt(simTime());
    if (!isActive(step)) return;                             // engine: receivers are active vehicles only
    double dist = curPosition.distance(m->getSenderPos());
    bool fresh = !holds;
    S.log << "R," << simTime() << "," << step << "," << idx << "," << m->getSenderIdx() << "," << dist << ","
          << (fresh ? 1 : 0) << "," << m->getHops() << "\n";
    if (fresh) {
        holds = true;
        informedStep = step;
        hops = m->getHops() + 1;
        decide(step, "receive", dist);
    }
    else {
        dupCount++;
        if (cancelAtDups != NO_STEP && dupCount >= cancelAtDups && txSched > step) {
            txSched = NO_STEP;
            cancelAtDups = NO_STEP;
            disarm();
            S.log << "X," << simTime() << "," << step << "," << idx << ",cancel\n";
        }
    }
}

void AiharpApp::finish()
{
    if (activeSince >= 0) activeTime += simTime().dbl() - activeSince;
    if (idx >= 0) {
        recordScalar("traceIndex", idx);
        recordScalar("activeTime", activeTime);
        recordScalar("busyTimeWhileActive", busyActive);
        recordScalar("beaconsSent", beaconsSent);
        recordScalar("hazardTx", txCount);
    }
    disarm();
    cancelAndDelete(txEvt);
    cancelAndDelete(beaconEvt);
    cancelAndDelete(originEvt);
    txEvt = beaconEvt = originEvt = nullptr;
    DemoBaseApplLayer::finish();
    S.log.flush();
}
