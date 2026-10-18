// Revision-5 H3: the hazard message replayed in Veins.
import veins.base.utils.Coord;
import veins.modules.messages.BaseFrame1609_4;

namespace veins;

packet AiharpMsg extends BaseFrame1609_4 {
    int hops;            // sender's hop count; receiver takes hops + 1
    int senderIdx;       // trace index of the sender
    Coord senderPos;     // sender position at hand-off to the MAC
    double schedTime;    // epoch start the transmission was scheduled for
}
