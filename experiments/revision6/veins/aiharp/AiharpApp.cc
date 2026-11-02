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
// arrive". Here the information source is what a real stack has: the latest decoded beacon (position,
// velocity) per sender, entries expiring after 1.05 s (covers the 1 Hz DCC floor), positions extrapolated
// to now -- restricted, as the DecisionContext contract defines neighbours, to the nominal range. (The first
// port omitted that restriction; greedy then named relays beyond the nominal range from beacons that got
// through on favourable fading, none decoded the designation, and the 2026-10-19 validation replay was
// re-run after the fix.)
// Greedy designations ride in the frame (relay1/relay2) and take effect when the designated vehicle
// decodes that frame as its first copy.
//
// Channel access: each hazard frame is handed to the MAC after a uniform backoff slot in [0, CWmin]
// (the custom MAC's per-epoch draw); 802.11 would otherwise send epoch-aligned frames simultaneously.
//
// Log lines (CSV): Q,t,originDone,txPending,timerPending,lastTx,minEnd (manager, every 10 s)
//   O,t,step,idx | T,t,step,idx,hops,sched_t,relay1,relay2 | R,t,step,idx,sender,dist,new,hops,designated
//   D,t,step,idx,trigger,action,delay | X,t,step,idx,reason

#include "AiharpApp.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <sstream>

#include "veins/modules/mac/ieee80211p/Mac1609_4.h"
#include "veins/modules/mobility/traci/TraCIConnection.h"
#include "veins/modules/mobility/traci/TraCIScenarioManager.h"

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
        // revision 6 (experiments/revision6/veins/r6_bundle.py): risk field and hazard, exact Python values
        else if (key == "risk") {       // risk <highway|grid> eta_full eta_tau inside_w min_speed gamma opp_rel
            std::string g;
            ss >> g >> S.etaFull >> S.etaTau >> S.insideW >> S.minSpeed >> S.gamma >> S.oppRel;
            S.grid = g == "grid";
            S.riskLoaded = true;
        }
        else if (key == "hazard")       // hazard span_start span_end affected decay sev0 onset tau ttl
            ss >> S.spanStart >> S.spanEnd >> S.affected >> S.decay >> S.sev0 >> S.onset >> S.decayTau >> S.decayTtl;
        else if (key == "gridgeom") {   // gridgeom hx hy heading heading_known span_tol heading_tol
            int known;
            ss >> S.hx >> S.hy >> S.hHeading >> known >> S.spanTol >> S.headingTol;
            S.hHeadingKnown = known != 0;
        }
        else if (key == "anydir") { int a; ss >> a; S.anyDirNonzero = a != 0; }
        else if (key == "dir") {        // dir <index> <sign> <vclass>
            int i, sgn;
            std::string vc;
            ss >> i >> sgn >> vc;
            S.dirOf[i] = sgn;
            S.vclassOf[i] = vc;
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
        else if (p == "rar") kind = RAR;
        else if (p == "scf_slotted") kind = SCF_SLOTTED;
        else if (p == "ai_harp") kind = AI_HARP;
        else throw cRuntimeError("unknown policy %s", p.c_str());
        knob = par("knob").doubleValue();
        std::vector<double> kv;
        {
            std::string ks = par("knobs").stdstringValue();
            for (char& ch : ks) if (ch == ',') ch = ' ';
            std::istringstream ss(ks);
            double v;
            while (ss >> v) kv.push_back(v);
        }
        if (kind == RAR) {
            if (kv.size() != 3) throw cRuntimeError("rar needs knobs a,b,c");
            rarA = kv[0]; rarB = kv[1]; rarC = kv[2];
        }
        if (kind == SCF_SLOTTED) {
            if (kv.size() != 2) throw cRuntimeError("scf_slotted needs knobs n_slots,timeout_s");
            scfSlots = (int) kv[0]; scfTimeout = kv[1];
        }
        if ((kind == RAR || kind == SCF_SLOTTED || kind == AI_HARP) && !S.riskLoaded)
            throw cRuntimeError("policy %s needs the revision-6 risk/hazard lines in %s", p.c_str(),
                                par("nodesFile").stdstringValue().c_str());
        bridgeAddr = par("bridge").stdstringValue();
        replaySeed = par("replaySeed").intValue();
        if (kind == AI_HARP && bridgeAddr.empty()) throw cRuntimeError("ai_harp needs bridge=host:port");
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
        double d = curPosition.distance(p);
        if (d > S.commRange) continue;      // DecisionContext: neighbours are within the nominal range
        out.push_back({kv.first, p, kv.second.vel, d});
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
            auto di = S.dirOf.find(idx);
            bsm->setSenderDir(di == S.dirOf.end() ? 0 : di->second);
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
    case RAR: {                                              // agents/rule_baselines.py RelevanceAwareRule
        if (originate) break;                                // BROADCAST_NOW
        if (txCount > 0) { d.act = SUPPRESS; break; }
        int step = stepAt(simTime());
        double x, y, vx, vy;
        RelOut r = ownRelevance(step, x, y, vx, vy);
        int n = (int) neighbours().size();
        double dens = 1.0 - std::min((double) n, rarNmax) / rarNmax;
        double p = std::min(std::max(rarA * r.rel + rarB * dens + rarC, 0.0), 1.0);
        double u = uniform(0, 1);
        {   // decision inputs at full precision (port validation recomputes rho_c and the decision from them)
            std::ostringstream fl;
            fl.precision(17);
            fl << "F," << simTime() << "," << step << "," << idx << "," << trig << "," << x << "," << y << "," << vx << ","
               << vy << "," << S.dirOf[idx] << "," << r.rel << "," << r.eta << "," << n << "," << senderDist << "," << u
               << "," << p << "\n";
            S.log << fl.str();
        }
        if (u >= p) { d.act = SUPPRESS; break; }
        d.act = DEFER; d.delay = 1 + slotOf(rarSlots); d.cancel = 1;
        break;
    }
    case SCF_SLOTTED: {                                      // agents/rule_baselines.py StoreCarryForwardSlotted
        if (originate) break;
        if (txCount > 0) { d.act = SUPPRESS; break; }
        int step = stepAt(simTime());
        double timeS = step * S.dt;                          // DecisionContext.time_s = step * dt
        if (trig == "receive" && duplicate) { carryStart = -1; d.act = SUPPRESS; break; }
        int n = (int) neighbours().size();
        {
            std::ostringstream fl;
            fl.precision(17);
            fl << "F," << simTime() << "," << step << "," << idx << "," << trig << ",,,,," << S.dirOf[idx] << ",,," << n
               << "," << senderDist << ",," << carryStart << "\n";
            S.log << fl.str();
        }
        if (n > 0) {
            if (trig == "timer") { carryStart = -1; break; } // a carried message meets a neighbour: BROADCAST_NOW
            d.act = DEFER; d.delay = 1 + slotOf(scfSlots); d.cancel = 1;
            break;
        }
        if (carryStart < 0) carryStart = timeS;
        if (timeS - carryStart >= scfTimeout) { carryStart = -1; d.act = SUPPRESS; break; }
        d.act = CARRY; d.delay = 10;                         // carry_recheck_epochs
        break;
    }
    case AI_HARP:
        return bridgeDecide(stepAt(simTime()), trigger, senderDist, hasSender ? curSender : -1, senderPos, designated);
    }
    return d;
}

// ---------------------------------------------------------------------------------------------- revision 6 --
Coord AiharpApp::toTrace(const Coord& omnetPos) const
{
    TraCICoord c = mobility->getManager()->getConnection()->omnet2traci(omnetPos);
    return Coord(c.x, c.y);
}

static double sgn(double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }        // np.sign
static double pymod(double a, double m) { return a - m * std::floor(a / m); }   // Python %

AiharpApp::RelOut AiharpApp::relevanceAt(double x, double y, double vx, double vy, int dir, double t)
{
    // hazard/risk_field.py RiskField.evaluate for one active vehicle
    enum { APPROACHING = 0, INSIDE = 1, AWAY = 2 };
    double dist;
    int state;
    if (!S.grid) {                                           // HighwayRiskGeometry
        double dd = dir != 0 ? (double) dir : sgn(vx != 0 ? vx : 1.0);
        bool inside = x >= S.spanStart && x <= S.spanEnd;
        bool approaching = dd > 0 ? x < S.spanStart : x > S.spanEnd;
        dist = dd > 0 ? S.spanStart - x : x - S.spanEnd;
        dist = inside ? 0.0 : std::max(dist, 0.0);
        state = inside ? INSIDE : (approaching ? APPROACHING : AWAY);
    }
    else {                                                   // GridRiskGeometry
        double dx = S.hx - x, dy = S.hy - y;
        dist = std::fabs(dx) + std::fabs(dy);
        double closing = sgn(dx) * vx + sgn(dy) * vy;
        bool inside = dist <= S.spanTol + (S.spanEnd - S.spanStart) / 2.0;
        state = inside ? INSIDE : (closing > 0.1 ? APPROACHING : AWAY);
        if (S.affected != 0 && S.hHeadingKnown) {
            double delta = std::fabs(pymod(std::atan2(vy, vx) - S.hHeading + M_PI, 2 * M_PI) - M_PI);
            if (!(delta <= S.headingTol)) state = AWAY;
        }
        dist = std::max(dist, 0.0);
    }
    double speed = std::hypot(vx, vy);
    double eta = dist / std::max(speed, S.minSpeed);
    double geom = state == INSIDE ? S.insideW : (state == APPROACHING ? 1.0 : 0.0);
    double dtt = t - S.onset, sev;
    if (S.decay == 0) sev = S.sev0;
    else if (S.decay == 1) sev = S.sev0 * std::exp(-std::max(dtt, 0.0) / S.decayTau);
    else sev = S.sev0 * std::max(0.0, 1.0 - std::max(dtt, 0.0) / S.decayTtl);
    if (dtt < 0.0) sev = 0.0;
    double gate = 1.0;
    if (S.affected != 0 && S.anyDirNonzero) gate = dir == S.affected ? 1.0 : S.oppRel;
    double w = std::exp(-std::max(eta - S.etaFull, 0.0) / S.etaTau);
    double rel = gate * geom * w * std::pow(std::max(sev, 0.0), S.gamma);
    if (std::isnan(rel)) rel = 0.0;
    return {std::min(std::max(rel, 0.0), 1.0), eta};
}

AiharpApp::RelOut AiharpApp::ownRelevance(int step, double& x, double& y, double& vx, double& vy) const
{
    // own state as the trace defines it: SUMO position; velocity = speed * (cos h, sin h), h = Veins heading
    // (radians, x east / y north, i.e. radians(90 - SUMO angle) as mobility/fcd.py)
    Coord p = toTrace(curPosition);
    double h = mobility->getHeading().getRad(), sp = mobility->getSpeed();
    x = p.x; y = p.y; vx = sp * std::cos(h); vy = sp * std::sin(h);
    auto di = S.dirOf.find(idx);
    return relevanceAt(x, y, vx, vy, di == S.dirOf.end() ? 0 : di->second, step * S.dt);
}

static int bridgeSock = -1;                                  // one connection per replay (single process)
static std::string bridgeBuf;

static void bridgeConnect(const std::string& addr)
{
    size_t c = addr.rfind(':');
    if (c == std::string::npos) throw cRuntimeError("bridge must be host:port, got %s", addr.c_str());
    std::string host = addr.substr(0, c), port = addr.substr(c + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
        throw cRuntimeError("bridge: cannot resolve %s", addr.c_str());
    bridgeSock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (bridgeSock < 0 || connect(bridgeSock, res->ai_addr, res->ai_addrlen) != 0) {
        freeaddrinfo(res);
        throw cRuntimeError("bridge: cannot connect to %s (%s)", addr.c_str(), strerror(errno));
    }
    freeaddrinfo(res);
    int one = 1;
    setsockopt(bridgeSock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static std::string bridgeCall(const std::string& addr, const std::string& line)
{
    // synchronous request/reply, newline-delimited: simulation time does not advance while the server decides
    if (bridgeSock < 0) bridgeConnect(addr);
    std::string out = line + "\n";
    for (size_t off = 0; off < out.size();) {
        ssize_t k = send(bridgeSock, out.data() + off, out.size() - off, 0);
        if (k <= 0) throw cRuntimeError("bridge: send failed (%s)", strerror(errno));
        off += (size_t) k;
    }
    for (;;) {
        size_t nl = bridgeBuf.find('\n');
        if (nl != std::string::npos) {
            std::string reply = bridgeBuf.substr(0, nl);
            bridgeBuf.erase(0, nl + 1);
            return reply;
        }
        char buf[4096];
        ssize_t k = recv(bridgeSock, buf, sizeof(buf), 0);
        if (k <= 0) throw cRuntimeError("bridge: connection closed while waiting for a decision");
        bridgeBuf.append(buf, (size_t) k);
    }
}

AiharpApp::Decision AiharpApp::bridgeDecide(int step, const char* trigger, double senderDist, int senderIdx,
                                            Coord senderPos, bool designated)
{
    // The holder's DecisionContext (agents/base.py) as Veins has it: own state from mobility, neighbours from the
    // beacon table (neighbours()), their relevance/ETA from their beacon state and the hazard the holder carries,
    // informed = overheard. The server runs the unchanged agent code and returns the action.
    double x, y, vx, vy;
    RelOut own = ownRelevance(step, x, y, vx, vy);
    double t = step * S.dt;
    std::ostringstream q;
    q.precision(17);
    q << "{\"seed\":" << replaySeed << ",\"step\":" << step << ",\"time_s\":" << t << ",\"dt\":" << S.dt
      << ",\"index\":" << idx << ",\"trigger\":\"" << trigger << "\",\"hop_count\":" << hops
      << ",\"duplicate_count\":" << dupCount << ",\"own_tx_count\":" << txCount << ",\"sender_index\":" << senderIdx
      << ",\"sender_distance_m\":" << senderDist << ",\"max_sender_distance_m\":" << maxSenderDist
      << ",\"age_s\":" << (S.originStep >= 0 ? (step - S.originStep) * S.dt : 0.0) << ",\"x\":" << x << ",\"y\":" << y
      << ",\"speed_ms\":" << mobility->getSpeed() << ",\"heading\":" << mobility->getHeading().getRad()
      << ",\"direction\":" << S.dirOf[idx] << ",\"relevance\":" << own.rel << ",\"eta_s\":" << own.eta
      << ",\"vclass\":\"" << S.vclassOf[idx] << "\",\"comm_range_m\":" << S.commRange
      << ",\"was_designated\":" << (designated ? "true" : "false");
    double sdx = 0, sdy = 0;
    if (senderIdx >= 0) {
        Coord sp = toTrace(senderPos);
        sdx = sp.x - x; sdy = sp.y - y;
    }
    q << ",\"sender_dx\":" << sdx << ",\"sender_dy\":" << sdy;
    std::ostringstream ids, ds, dxs, dys, vxs, vys, rels, etas, infs;
    for (auto* o : {&ids, &ds, &dxs, &dys, &vxs, &vys, &rels, &etas, &infs}) o->precision(17);
    bool first = true;
    for (auto& n : neighbours()) {
        Coord p = toTrace(n.pos), v = velToTrace(n.vel);
        int nd = nbTable.at(n.idx).dir;
        RelOut r = relevanceAt(p.x, p.y, v.x, v.y, nd, t);
        const char* c = first ? "" : ",";
        ids << c << n.idx; ds << c << n.dist; dxs << c << p.x - x; dys << c << p.y - y; vxs << c << v.x;
        vys << c << v.y; rels << c << r.rel; etas << c << r.eta; infs << c << (heardFrom.count(n.idx) ? 1 : 0);
        first = false;
    }
    q << ",\"nb\":[" << ids.str() << "],\"nb_dist\":[" << ds.str() << "],\"nb_dx\":[" << dxs.str() << "],\"nb_dy\":["
      << dys.str() << "],\"nb_vx\":[" << vxs.str() << "],\"nb_vy\":[" << vys.str() << "],\"nb_rel\":[" << rels.str()
      << "],\"nb_eta\":[" << etas.str() << "],\"nb_informed\":[" << infs.str() << "]}";
    std::istringstream rep(bridgeCall(bridgeAddr, q.str()));
    Decision d;
    int act;
    if (!(rep >> act >> d.delay >> d.cancel >> d.relays[0] >> d.relays[1]) || act < SUPPRESS || act > RELAY)
        throw cRuntimeError("bridge: malformed reply");
    d.act = (Act) act;
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
    int r1 = pendingRelays[0], r2 = pendingRelays[1];
    pendingRelays[0] = pendingRelays[1] = -1;
    sendDelayedDown(m, intuniform(0, S.cwSlots - 1) * S.slotTime);   // custom MAC's per-epoch backoff draw
    txCount++;
    S.lastTx = simTime().dbl();
    S.log << "T," << simTime() << "," << step << "," << idx << "," << hops << "," << timeOf(step) << "," << r1 << ","
          << r2 << "\n";
}

void AiharpApp::onBSM(DemoSafetyMessage* bsm)
{
    AiharpBeacon* b = dynamic_cast<AiharpBeacon*>(bsm);
    if (!b || b->getSenderIdx() < 0 || !isActive(stepAt(simTime()))) return;
    nbTable[b->getSenderIdx()] = {b->getSenderPos(), b->getSenderSpeed(), simTime().dbl(), b->getSenderDir()};
}

void AiharpApp::onWSM(BaseFrame1609_4* wsm)
{
    AiharpMsg* m = dynamic_cast<AiharpMsg*>(wsm);
    if (!m) return;
    int step = stepAt(simTime());
    if (!isActive(step)) return;                             // engine: receivers are active vehicles only
    double dist = curPosition.distance(m->getSenderPos());
    maxSenderDist = std::max(maxSenderDist, dist);
    heardFrom[m->getSenderIdx()] = true;                     // DecisionContext.neighbour_informed (overheard only)
    curSender = m->getSenderIdx();
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
        else if (wantsDupCallbacks() && txSched == NO_STEP)  // wants_duplicate_callbacks
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
