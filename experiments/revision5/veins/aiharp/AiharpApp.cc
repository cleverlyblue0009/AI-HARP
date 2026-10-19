// Revision-5 H3: classical dissemination schemes in Veins, mirroring sim/engine.py.
//
// Timing (engine semantics): decisions happen in 100 ms epochs aligned to the trace (step k <-> SUMO time
// t0 + k*dt). BROADCAST/RELAY -> transmit at step+1; DEFER(d) -> step+max(1,d); CARRY(d) -> re-decide
// (TIMER) at step+max(1,d), skipped while a transmission is pending; SUPPRESS clears both. A pending
// transmission is cancelled once dupCount >= cancelAtDups; otherwise a policy with duplicate callbacks
// (DV-CAST) re-decides on a duplicate when no transmission is pending. TTL (60 s) and the hop limit (32)
// are checked before sending. A vehicle takes part only in steps where it is active in the trace (in the
// corridor). Policy decisions (agents/*.py) are reproduced rule for rule; random draws come from
// OMNeT++'s RNG (seed-set = run seed).
//
// Neighbour tables (greedy_farthest, DV-CAST): the custom engine takes them from trace geometry within the
// nominal range (with an NLOS link-budget test in urban_nlos), standing in for "vehicles whose beacons
// arrive". Here they are what a real stack has: the latest decoded beacon (position, velocity) per sender,
// entries expiring after 1.05 s (covers the 1 Hz DCC floor), positions extrapolated to now.
// Greedy designations ride in the frame (relay1/relay2) and take effect when the designated vehicle
// decodes that frame as its first copy.
//
// Channel access: each hazard frame is handed to the MAC after a uniform backoff slot in [0, CWmin]
// (the custom MAC's per-epoch draw); 802.11 would otherwise send epoch-aligned frames simultaneously.
//
// Log lines (CSV): Q,t,originDone,txPending,timerPending,lastTx,minEnd (manager, every 10 s)
//   O,t,step,idx | T,t,step,idx,hops,sched_t | R,t,step,idx,sender,dist,new,hops,designated
//   D,t,step,idx,trigger,action,delay | X,t,step,idx,reason

#include "AiharpApp.h"

#include <cmath>
#include <sstream>

#include "veins/modules/mac/ieee80211p/Mac1609_4.h"

using namespace veins;

Define_Module(veins::AiharpApp);

AiharpApp::Shared AiharpApp::S;

static const char* ACT_NAME[] = {"suppress", "broadcast", "defer", "carry", "relay"};

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
        else if (key == "access") ss >> S.cwSlots >> S.slotTime;   // access <cw_slots> <slot_s>
        else if (key == "beacon") {     // beacon <hz> <duration_s> <cs_range_m> <dcc 0|1> <target_cbr> <min_hz>
            int on;
            ss >> S.beaconHz >> S.beaconDur >> S.csRange >> on >> S.dccTarget >> S.dccMin;
            S.dcc = on != 0;
        }
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
        else if (p == "greedy_farthest") kind = GREEDY_FARTHEST;
        else if (p == "dvcast") kind = DVCAST;
        else throw cRuntimeError("unknown policy %s", p.c_str());
        knob = par("knob").doubleValue();
        std::string sumoId = mobility->getExternalId();
        auto it = S.indexOf.find(sumoId);
        idx = it == S.indexOf.end() ? -1 : it->second;   // vehicles never in the corridor stay silent
        txEvt = new cMessage("tx", EV_TX);
        timerEvt = new cMessage("timer", EV_TIMER);
        beaconEvt = new cMessage("beacon", EV_BEACON);
        originEvt = new cMessage("origin", EV_ORIGIN);
        if (cModule* m = getParentModule()->getSubmodule("nic")->getSubmodule("mac1609_4"))
            m->subscribe(Mac1609_4::sigChannelBusy, this);
    }
    else if (stage == 1 && idx >= 0) {
        if (par("beacons").boolValue() && S.beaconHz > 0)
            scheduleAt(simTime() + dblrand() * beaconInterval, beaconEvt);   // random phase, as Veins does
        if (idx == S.originIdx && timeOf(S.originStep) >= simTime())
            scheduleAt(timeOf(S.originStep), originEvt);
    }
}

bool AiharpApp::quiescent(simtime_t now, double minEnd)
{
    if (!S.loaded || !S.originDone || S.txPending != 0) return false;
    double tOrigin = S.t0 + S.originStep * S.dt;
    bool timersIdle = S.timerPending == 0 || now.dbl() > tOrigin + S.ttl + S.dt;   // after the TTL nothing can send
    return timersIdle && now.dbl() >= minEnd && now.dbl() - S.lastTx >= 1.0;
}

void AiharpApp::logState(simtime_t now, double minEnd)
{
    if (!S.loaded) return;
    S.log << "Q," << now << "," << S.originDone << "," << S.txPending << "," << S.timerPending << "," << S.lastTx
          << "," << minEnd << "\n";
}

simtime_t AiharpApp::timeOf(int step) const { return S.t0 + step * S.dt; }

int AiharpApp::stepAt(simtime_t t) const { return (int) std::floor((t.dbl() - S.t0) / S.dt + 1e-6); }

bool AiharpApp::isActive(int step) const { return activeAt(idx, step); }

bool AiharpApp::activeAt(int i, int step)
{
    if (i < 0 || step < 0 || step >= S.nSteps) return false;
    auto it = S.active.find(i);
    if (it == S.active.end()) return false;
    for (auto& r : it->second)
        if (step >= r.first && step <= r.second) return true;
    return false;
}

void AiharpApp::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);
    if (idx >= 0) S.pos[idx] = curPosition;
}

double AiharpApp::beaconRate(int step) const
{
    // sim/mac.py effective_beacon_hz: clip(target / (n * T_b), min_hz, beacon_hz), n = active vehicles
    // within the carrier-sense range
    if (!S.dcc) return S.beaconHz;
    int n = 0;
    for (auto& kv : S.pos)
        if (kv.first != idx && curPosition.distance(kv.second) <= S.csRange && activeAt(kv.first, step)) n++;
    double allowed = S.dccTarget / std::max(n * S.beaconDur, 1e-12);
    return std::min(std::max(allowed, S.dccMin), S.beaconHz);
}

std::vector<AiharpApp::Neighbour> AiharpApp::neighbours() const
{
    std::vector<Neighbour> out;
    double now = simTime().dbl();
    for (auto& kv : nbTable) {
        double age = now - kv.second.t;
        if (age > NEIGHBOUR_TTL_S) continue;
        Coord p = kv.second.pos + kv.second.vel * age;
        out.push_back({kv.first, p, kv.second.vel, curPosition.distance(p)});
    }
    return out;
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
            AiharpBeacon* bsm = new AiharpBeacon("beacon");
            populateWSM(bsm);
            bsm->setSenderIdx(idx);
            sendDown(bsm);
            beaconsSent++;
            if (activeSince < 0) activeSince = simTime().dbl();
        }
        else if (activeSince >= 0) {
            activeTime += simTime().dbl() - activeSince;
            activeSince = -1;
        }
        double rate = isActive(step) ? beaconRate(step) : 0.0;
        scheduleAt(simTime() + (rate > 0 ? 1.0 / rate : beaconInterval.dbl()), beaconEvt);
        break;
    }
    case EV_ORIGIN: {
        holds = true;
        informedStep = step;
        hops = 0;
        S.log << "O," << simTime() << "," << step << "," << idx << "\n";
        decide(step, "originate", 0.0);
        S.originDone = true;
        break;
    }
    case EV_TX:
        S.txPending--;                                       // the event has fired
        transmit(step);
        break;
    case EV_TIMER:
        S.timerPending--;
        if (timerSched == step) {
            timerSched = NO_STEP;
            // engine: due timers need holds & active; skipped while a transmission is queued
            if (holds && isActive(step) && txSched == NO_STEP) decide(step, "timer", maxSenderDist);
        }
        break;
    default:
        DemoBaseApplLayer::handleSelfMsg(msg);
    }
}

AiharpApp::Decision AiharpApp::policyDecide(const char* trigger, double senderDist, bool hasSender,
                                            Coord senderPos, bool duplicate, bool designated)
{
    // agents/*.py decide()
    Decision d;
    std::string trig(trigger);
    bool originate = trig == "originate";
    double R = S.commRange;
    auto slotOf = [&](int n) {
        double frac = R <= 0 ? 1.0 : std::min(std::max(senderDist, 0.0), R) / R;
        return std::min(std::max((int) std::floor(n * (1.0 - frac)), 0), n - 1);
    };
    // progress of each neighbour along the propagation direction (DecisionContext.progress)
    auto progress = [&](const std::vector<Neighbour>& nb) {
        std::vector<double> pr(nb.size(), 0.0);
        if (!hasSender) return pr;
        Coord sd = senderPos - curPosition;
        double norm = std::sqrt(sd.x * sd.x + sd.y * sd.y);
        if (norm < 1e-9) return pr;
        double ux = -sd.x / norm, uy = -sd.y / norm;
        for (size_t k = 0; k < nb.size(); ++k) {
            Coord dd = nb[k].pos - curPosition;
            pr[k] = dd.x * ux + dd.y * uy;
        }
        return pr;
    };
    switch (kind) {
    case FLOODING:
        break;
    case P_PERSISTENCE:
        if (!originate && !(uniform(0, 1) < knob)) d.act = SUPPRESS;
        break;
    case WEIGHTED_P: {
        if (originate) break;
        double p = R <= 0 ? knob : std::min(std::max(senderDist / R, 0.0), knob);   // min_p = 0, max_p = knob
        if (!(uniform(0, 1) < p)) d.act = SUPPRESS;
        break;
    }
    case SLOTTED_1P:                                         // p = 1, slot_epochs = 1, cancel_on_duplicates = 1
        if (originate) break;
        d.act = DEFER; d.delay = 1 + slotOf((int) knob); d.cancel = 1;
        break;
    case COUNTER_BASED:                                      // max_delay_epochs = 5, min_distance_fraction = 0.35
        if (originate) break;
        if (senderDist < 0.35 * R) { d.act = SUPPRESS; break; }
        d.act = DEFER; d.delay = 1 + intuniform(0, 4); d.cancel = (int) knob - 1;
        break;
    case GREEDY_FARTHEST: {                                  // fallback_epochs = 3, seed_both_directions
        if (originate || designated) {
            auto nb = neighbours();
            if (nb.empty()) break;                           // no relay to name: BROADCAST_NOW
            if (!hasSender) {                                // origination: farthest on each side of the corridor
                double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
                for (auto& n : nb) {
                    Coord dd = n.pos - curPosition;
                    minx = std::min(minx, dd.x); maxx = std::max(maxx, dd.x);
                    miny = std::min(miny, dd.y); maxy = std::max(maxy, dd.y);
                }
                bool useX = (maxx - minx) >= (maxy - miny);
                int k = 0;
                for (int side : {+1, -1}) {
                    int best = -1;
                    double bestD = -1;
                    for (auto& n : nb) {
                        Coord dd = n.pos - curPosition;
                        double a = useX ? dd.x : dd.y;
                        if ((side > 0 ? a > 0 : a < 0) && n.dist > bestD) { bestD = n.dist; best = n.idx; }
                    }
                    if (best >= 0) d.relays[k++] = best;
                }
            }
            else {
                auto pr = progress(nb);
                int best = -1;
                double bestP = 0.0;
                for (size_t k = 0; k < nb.size(); ++k)
                    if (pr[k] > 0.0 && (best < 0 || pr[k] > bestP)) { bestP = pr[k]; best = nb[k].idx; }
                if (best >= 0) d.relays[0] = best;
            }
            if (d.relays[0] >= 0) d.act = RELAY;
            break;
        }
        d.act = DEFER; d.delay = 3 + slotOf((int) knob); d.cancel = 1;
        break;
    }
    case DVCAST: {                                           // slot_epochs = 1, cancel 1, carry_recheck_epochs = 10
        if (originate) break;
        if (txCount > 0) { d.act = SUPPRESS; break; }
        if (trig == "receive" && duplicate) { d.act = SUPPRESS; break; }
        auto nb = neighbours();
        enum { WELL, SPARSE, DISC } regime;
        if (nb.empty()) regime = DISC;
        else {
            bool ahead = !hasSender;
            if (hasSender)
                for (double p : progress(nb)) ahead |= p > 0.0;
            if (ahead) regime = WELL;
            else {
                Coord own = mobility->getHeading().toCoord();
                bool opposing = false;
                for (auto& n : nb) {
                    double sp = std::sqrt(n.vel.x * n.vel.x + n.vel.y * n.vel.y);
                    if (sp <= 0.5) continue;
                    double dot = (n.vel.x * own.x + n.vel.y * own.y) / std::max(sp, 1e-9);
                    if (std::acos(std::min(std::max(dot, -1.0), 1.0)) > M_PI / 2) opposing = true;
                }
                regime = opposing ? SPARSE : DISC;
            }
        }
        if (regime == WELL) {
            if (trig == "timer") break;                      // BROADCAST_NOW
            d.act = DEFER; d.delay = 1 + slotOf((int) knob); d.cancel = 1;
        }
        else if (regime == DISC) {
            d.act = CARRY; d.delay = 10;
        }
        break;                                               // SPARSE: BROADCAST_NOW
    }
    }
    return d;
}

void AiharpApp::decide(int step, const char* trigger, double senderDist, bool hasSender, Coord senderPos,
                       bool duplicate, bool designated)
{
    Decision d = policyDecide(trigger, senderDist, hasSender, senderPos, duplicate, designated);
    S.log << "D," << simTime() << "," << step << "," << idx << "," << trigger << "," << ACT_NAME[d.act] << ","
          << d.delay << "\n";
    // sim/engine.py _apply
    if (d.cancel != NO_STEP) cancelAtDups = d.cancel;
    switch (d.act) {
    case SUPPRESS:
        txSched = NO_STEP; disarmTx();
        timerSched = NO_STEP; disarmTimer();
        break;
    case BROADCAST:
        pendingRelays[0] = pendingRelays[1] = -1;
        scheduleTx(step + 1);
        break;
    case DEFER:
        pendingRelays[0] = pendingRelays[1] = -1;
        scheduleTx(step + std::max(1, d.delay));
        break;
    case CARRY:
        scheduleTimer(step + std::max(1, d.delay));
        break;
    case RELAY:
        pendingRelays[0] = d.relays[0];
        pendingRelays[1] = d.relays[1];
        scheduleTx(step + 1);
        break;
    }
}

void AiharpApp::scheduleTx(int step)
{
    txSched = step;
    disarmTx();
    scheduleAt(timeOf(step), txEvt);
    S.txPending++;
}

void AiharpApp::disarmTx()
{
    if (txEvt && txEvt->isScheduled()) {
        cancelEvent(txEvt);
        S.txPending--;
    }
}

void AiharpApp::scheduleTimer(int step)
{
    timerSched = step;
    disarmTimer();
    scheduleAt(timeOf(step), timerEvt);
    S.timerPending++;
}

void AiharpApp::disarmTimer()
{
    if (timerEvt && timerEvt->isScheduled()) {
        cancelEvent(timerEvt);
        S.timerPending--;
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
    m->setRelay1(pendingRelays[0]);
    m->setRelay2(pendingRelays[1]);
    pendingRelays[0] = pendingRelays[1] = -1;
    sendDelayedDown(m, intuniform(0, S.cwSlots - 1) * S.slotTime);   // custom MAC's per-epoch backoff draw
    txCount++;
    S.lastTx = simTime().dbl();
    S.log << "T," << simTime() << "," << step << "," << idx << "," << hops << "," << timeOf(step) << "\n";
}

void AiharpApp::onBSM(DemoSafetyMessage* bsm)
{
    AiharpBeacon* b = dynamic_cast<AiharpBeacon*>(bsm);
    if (!b || b->getSenderIdx() < 0 || !isActive(stepAt(simTime()))) return;
    nbTable[b->getSenderIdx()] = {b->getSenderPos(), b->getSenderSpeed(), simTime().dbl()};
}

void AiharpApp::onWSM(BaseFrame1609_4* wsm)
{
    AiharpMsg* m = dynamic_cast<AiharpMsg*>(wsm);
    if (!m) return;
    int step = stepAt(simTime());
    if (!isActive(step)) return;                             // engine: receivers are active vehicles only
    double dist = curPosition.distance(m->getSenderPos());
    maxSenderDist = std::max(maxSenderDist, dist);
    bool fresh = !holds;
    bool designated = fresh && (m->getRelay1() == idx || m->getRelay2() == idx);
    S.log << "R," << simTime() << "," << step << "," << idx << "," << m->getSenderIdx() << "," << dist << ","
          << (fresh ? 1 : 0) << "," << m->getHops() << "," << (designated ? 1 : 0) << "\n";
    if (fresh) {
        holds = true;
        informedStep = step;
        hops = m->getHops() + 1;
        decide(step, "receive", dist, true, m->getSenderPos(), false, designated);
    }
    else {
        dupCount++;
        if (cancelAtDups != NO_STEP && dupCount >= cancelAtDups && txSched > step) {
            txSched = NO_STEP;
            cancelAtDups = NO_STEP;
            disarmTx();
            S.log << "X," << simTime() << "," << step << "," << idx << ",cancel\n";
        }
        else if (kind == DVCAST && txSched == NO_STEP)       // wants_duplicate_callbacks
            decide(step, "receive", dist, true, m->getSenderPos(), true, false);
    }
}

void AiharpApp::finish()
{
    if (activeSince >= 0) activeTime += simTime().dbl() - activeSince;
    S.pos.erase(idx);
    if (idx >= 0) {
        recordScalar("traceIndex", idx);
        recordScalar("activeTime", activeTime);
        recordScalar("busyTimeWhileActive", busyActive);
        recordScalar("beaconsSent", beaconsSent);
        recordScalar("hazardTx", txCount);
    }
    disarmTx();
    disarmTimer();
    cancelAndDelete(txEvt);
    cancelAndDelete(timerEvt);
    cancelAndDelete(beaconEvt);
    cancelAndDelete(originEvt);
    txEvt = timerEvt = beaconEvt = originEvt = nullptr;
    DemoBaseApplLayer::finish();
    S.log.flush();
}
