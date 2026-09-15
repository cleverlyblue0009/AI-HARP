"""CLI overrides used by the sparse-feasibility runs (lambda cap test).

run8 left rural d=2 and urban d=2 short with lambda pinned at its cap of 50.
The follow-up finetunes on the sparse densities alone with a raised cap; these
pin that the flags do what the write-up says they do.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from agents.constrained_reward import build_training_objective  # noqa: E402
from agents.train import stage_densities  # noqa: E402
from common.config import PROJECT_ROOT, load_yaml  # noqa: E402


def test_sparse_only_finetune_draws_only_sparse_densities():
    cfg = load_yaml("agent.yaml")
    cfg["training"]["curriculum"]["finetune"]["dense_densities"] = []
    d, p = stage_densities(cfg, "finetune")
    assert d == [float(x) for x in cfg["training"]["curriculum"]["finetune"]["sparse_densities"]]
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p, 1.0 / len(d))


def test_default_finetune_is_unchanged():
    cfg = load_yaml("agent.yaml")
    d, p = stage_densities(cfg, "finetune")
    assert len(d) == 8 and p.sum() == pytest.approx(1.0)
    assert p[:4].sum() == pytest.approx(0.75)


def test_lambda_overrides_reach_new_groups_but_not_restored_ones():
    cfg = load_yaml("agent.yaml")
    cfg["objective"].update(lambda_max=500.0, lambda_init=500.0)
    obj = build_training_objective(cfg, PROJECT_ROOT)
    obj.load_state_dict({"multipliers": {"group_by": ["scenario", "density"],
                                         "values": {"rural_highway|20": 2.6}}})
    assert obj.multipliers.value("rural_highway|20") == pytest.approx(2.6)
    assert obj.multipliers.value("rural_highway|2") == pytest.approx(500.0)
    obj.update([("rural_highway|2", 0.1, 0.9)])
    assert obj.multipliers.value("rural_highway|2") == pytest.approx(500.0)   # capped at 500
    obj.update([("rural_highway|20", 0.1, 0.9)])
    assert obj.multipliers.value("rural_highway|20") > 2.6
