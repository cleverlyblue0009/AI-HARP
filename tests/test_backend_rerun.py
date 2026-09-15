"""SUMO / OSM re-runs of the headline cells: backend threading and scenario remapping."""

from __future__ import annotations

import pytest

from analysis.comparator import main as comparator_main
from analysis.comparator import parse_scenario_map, remap_specs


def test_scenario_map_parses_and_remaps_only_named_scenarios():
    m = parse_scenario_map("rural_highway=rural_highway_osm, urban_nlos=urban_grid_osm")
    assert m == {"rural_highway": "rural_highway_osm", "urban_nlos": "urban_grid_osm"}
    specs = [{"scenario": "rural_highway", "density": 2}, {"scenario": "urban_grid", "density": 20}]
    out = remap_specs(specs, m)
    assert [s["scenario"] for s in out] == ["rural_highway_osm", "urban_grid"]
    assert specs[0]["scenario"] == "rural_highway"          # input untouched
    assert parse_scenario_map(None) == {}
    with pytest.raises(ValueError):
        parse_scenario_map("rural_highway")


def test_forced_backend_reaches_every_run(monkeypatch):
    import analysis.pareto as pareto

    seen = []

    def fake_run_single(spec, **kw):
        seen.append(spec.backend)
        return {"rwcr": 0.5, "tx_per_at_risk_informed": 1.0}, None

    monkeypatch.setattr(pareto, "run_single", fake_run_single)
    pareto.build_cells([{"scenario": "rural_highway", "density": 2}], range(2),
                       policies=["flooding", "slotted_1p"], backend="sumo")
    assert seen and set(seen) == {"sumo"}


def test_backend_rerun_refuses_to_overwrite_the_committed_cells():
    with pytest.raises(SystemExit):
        comparator_main(["--backend", "sumo"])
    with pytest.raises(SystemExit):
        comparator_main(["--scenario-map", "rural_highway=rural_highway_osm"])


def test_evaluate_agent_loads_the_cells_it_is_pointed_at(monkeypatch, tmp_path):
    from pathlib import Path

    import experiments.evaluate_agent as ev

    ckpt = tmp_path / "placeholder.pt"         # main() refuses a missing checkpoint first
    ckpt.write_bytes(b"not a real checkpoint")

    class Loaded(Exception):
        pass

    def fake_load(path=None):
        raise Loaded(path)

    monkeypatch.setattr(ev, "load_cells", fake_load)
    with pytest.raises(Loaded) as info:
        ev.main(["--checkpoint", str(ckpt), "--cells-path", "results/sumo_cells.json",
                 "--backend", "sumo"])
    assert info.value.args[0] == Path("results/sumo_cells.json")


def test_sumo_trace_keys_are_versioned_but_fallback_keys_are_not():
    import mobility.generate as gen

    args = ({"name": "x"}, 20.0, 0)
    fb = gen.trace_cache_key(*args, "fallback", 1.0, 1.0, None)
    sumo = gen.trace_cache_key(*args, "sumo", 1.0, 1.0, None)
    from common.config import config_hash

    # fallback key is exactly the pre-versioning payload: cached fallback traces still hit
    assert fb == config_hash({"scenario": {"name": "x"}, "density": 20.0, "seed": 0,
                              "backend": "fallback", "speed_factor": 1.0,
                              "headway_factor": 1.0, "duration_s": None})
    old_sumo = config_hash({"scenario": {"name": "x"}, "density": 20.0, "seed": 0,
                            "backend": "sumo", "speed_factor": 1.0,
                            "headway_factor": 1.0, "duration_s": None})
    assert sumo != old_sumo                      # pre-fix SUMO traces are never served
