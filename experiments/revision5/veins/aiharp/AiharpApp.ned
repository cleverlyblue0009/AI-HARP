// Revision-5 H3 application (see AiharpApp.cc).
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
}
