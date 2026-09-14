"""Step 6 speed-ups must not change a single transition.

Only changes that measured a gain are kept (results/profile_step6_cheap.txt).
Vectorising the star-edge construction in build_decision_graph was tried and
reverted: graph construction went from 6.3 s to 6.0 s in the instrumented
profile, within noise, because the cost is in the feature arithmetic, not the
edge-list loop.
"""

from __future__ import annotations

import pytest


def test_per_step_feature_cache_changes_nothing(hz_cfg):
    """A whole learned-agent episode with the cache on and off."""
    pytest.importorskip("torch")
    import copy
    from pathlib import Path

    import torch

    from agents.constrained_reward import build_training_objective
    from agents.gat_drl import build_network
    from agents.train import EpisodeSpec, _training_policy, collect_rollouts
    from common.config import load_yaml

    cfg = load_yaml("agent.yaml")
    base = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    torch.manual_seed(0)
    net = build_network(cfg)
    objective = build_training_objective(cfg, Path("does-not-exist"))
    spec = EpisodeSpec("urban_nlos", 10.0, "clear", "fog_bank", 101)

    def run(cache):
        cfgs = copy.deepcopy(base)
        cfgs["experiment"]["simulation"]["cache_step_features"] = cache
        return collect_rollouts([spec], [9], net, _training_policy(net, cfg, None),
                                objective, cfgs)[0]

    (t_on, i_on), (t_off, i_off) = run(True), run(False)
    assert len(t_on) == len(t_off) > 0
    for a, b in zip(t_on, t_off):
        assert (a.vehicle, a.step, a.action, a.reward, a.log_prob, a.value) == \
               (b.vehicle, b.step, b.action, b.reward, b.log_prob, b.value)
        assert a.graph.x.tobytes() == b.graph.x.tobytes()
    assert (i_on["obj_coverage"], i_on["transmissions"]) == \
           (i_off["obj_coverage"], i_off["transmissions"])
