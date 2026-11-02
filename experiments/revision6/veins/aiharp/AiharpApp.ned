// Revision-6 application: the revision-5 classical ports plus rar, scf_slotted and ai_harp (see AiharpApp.cc).
import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

simple AiharpApp extends DemoBaseApplLayer
{
    parameters:
        @class(veins::AiharpApp);
        string nodesFile;               // nodes.txt written by run_veins.py from bundle.json
        string logFile;                 // event log (CSV lines, see AiharpApp.cc)
        string policy;                  // flooding | p_persistence | slotted_1p | weighted_p | counter_based
        double knob = default(0);       // p | n_slots | max_p | counter_threshold
        bool beacons = default(true);   // 10 Hz CAM/BSM load from active vehicles
        string knobs = default("");     // rar: "a,b,c"; scf_slotted: "n_slots,timeout_s"
        string bridge = default("");    // ai_harp: decision server "host:port"
        int replaySeed = default(0);    // ai_harp: names the per-(seed, vehicle, epoch) RNG streams
}
