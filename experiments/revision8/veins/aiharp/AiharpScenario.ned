// Revision-5 H3 network: Veins' Scenario with a forking TraCI manager (SUMO runs in the run directory
// with the exported veins.sumocfg, whose seed is NOT overridden).
import org.car2x.veins.base.connectionManager.ConnectionManager;
import org.car2x.veins.base.modules.BaseWorldUtility;
import org.car2x.veins.modules.mobility.traci.TraCIScenarioManagerForker;
import org.car2x.veins.modules.obstacle.ObstacleControl;
import org.car2x.veins.modules.world.annotations.AnnotationManager;
import org.car2x.veins.nodes.Car;

network AiharpScenario
{
    parameters:
        double playgroundSizeX @unit(m);
        double playgroundSizeY @unit(m);
        double playgroundSizeZ @unit(m);
    submodules:
        obstacles: ObstacleControl;
        annotations: AnnotationManager;
        connectionManager: ConnectionManager;
        world: BaseWorldUtility {
            playgroundSizeX = parent.playgroundSizeX;
            playgroundSizeY = parent.playgroundSizeY;
            playgroundSizeZ = parent.playgroundSizeZ;
        }
        manager: AiharpManager;
        node[0]: Car;
    connections allowunconnected:
}

simple AiharpManager extends TraCIScenarioManagerForker
{
    parameters:
        @class(veins::AiharpManager);
        double t0;          // SUMO time of trace step 0
        double minEnd;      // earliest early-stop time (CBR window), s
}
