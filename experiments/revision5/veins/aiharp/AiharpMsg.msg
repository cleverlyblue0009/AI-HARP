// Revision-5 H3: the hazard message and the CAM/BSM beacon replayed in Veins.
import veins.base.utils.Coord;
import veins.modules.messages.BaseFrame1609_4;
import veins.modules.messages.DemoSafetyMessage;

namespace veins;

packet AiharpMsg extends BaseFrame1609_4 {
    int hops;            // sender's hop count; receiver takes hops + 1
    int senderIdx;       // trace index of the sender
    Coord senderPos;     // sender position at hand-off to the MAC
    double schedTime;    // epoch start the transmission was scheduled for
    int relay1 = -1;     // greedy forwarding: designated relays (trace indices), carried in the frame
    int relay2 = -1;
}

packet AiharpBeacon extends DemoSafetyMessage {
    int senderIdx;       // trace index (neighbour tables are keyed by it)
}
