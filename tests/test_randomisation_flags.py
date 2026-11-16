"""Revision 8 section 3: the audit-informed randomisation axes are flags, OFF by default.

* Flag-off engine reproduces committed revision-5 rows (CAL classical, fallback; EVAL classical on SUMO mobility, the
  backend of the revision-7 custom runs) metric for metric, as written (string-identical CSV values).
* Default settings and explicit flag-off settings give byte-identical engine output.
* Zero-effect settings of the stochastic axes (R2 with p_miss = p_stale = 0, R5 with u = 0) are byte-identical to off:
  they draw only from their own streams.
* Each axis, when on, does what it says (R1 cancels more in a dense cell; R2 changes neighbour counts; R3 shows the
  decision the trace one epoch earlier; R5 raises PDR when the gap is positive).
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import os

import numpy as np
import pytest

from common.config import PROJECT_ROOT, load_yaml
from experiments.revision1.runner import METRICS, run_task, task
from sim.engine import SimSettings
from tests.test_structured import _fingerprint

RUNS = PROJECT_ROOT / "results" / "revision5" / "runs"
EXACT = ("tx_per_at_risk_informed", "transmissions", "tir_median_s", "tir_p95_s", "actionable_deadline_miss_rate",
         "deadline_miss_rate")
R7_COMMIT = "a6dea21"


def _committed(fname, picks):
    out = []
    with open(RUNS / fname, newline="") as f:
        for r in csv.DictReader(f):
            k = (r["scenario"], r["density"], r["policy"], r["value"], r["seed"])
            if k in picks:
                out.append(r)
    assert len(out) == len(picks)
    return out


CAL_PICKS = {("rural_highway", "2.0", "slotted_1p", "5", "300"), ("rural_highway", "20.0", "counter_based", "3", "301"),
             ("rural_highway", "80.0", "weighted_p", "0.5", "302"), ("urban_nlos", "20.0", "dvcast", "8", "303"),
             ("urban_nlos", "20.0", "greedy_farthest", "4", "304"), ("rural_highway", "20.0", "flooding", "", "305")}
SUMO_PICKS = {("rural_highway", "20.0", "slotted_1p", "8", "200"), ("urban_nlos", "20.0", "counter_based", "3", "201")}


def _rerun(r, backend):
    val = r["value"]
    v = None if val == "" else (int(val) if val.lstrip("-").isdigit() else float(val))
    t = task(r["scenario"], float(r["density"]), r["policy"], int(r["seed"]), param=r["param"], value=v,
             backend=backend)
    if t["task_hash"] != r["task_hash"] and backend == "fallback":
        # rows written before revision 5 made the backend explicit (D2) carry backend "auto" in their identity;
        # "auto" resolves to fallback here because SUMO is not findable (the test unsets SUMO_HOME for these rows)
        t = task(r["scenario"], float(r["density"]), r["policy"], int(r["seed"]), param=r["param"], value=v)
    assert t["task_hash"] == r["task_hash"]
    return run_task(t)


@pytest.mark.parametrize("fname,picks,backend", [
    ("classical__CAL__grid24__default__fallback.csv", CAL_PICKS, "fallback"),
    ("classical__EVAL__headline__default__sumo.csv", SUMO_PICKS, "sumo")])
def test_flag_off_engine_reproduces_committed_rows(fname, picks, backend, monkeypatch):
    if backend == "fallback":
        monkeypatch.delenv("SUMO_HOME", raising=False)
        from mobility.sumo_runner import find_sumo
        assert find_sumo() is None
    if backend == "sumo":
        from mobility.sumo_runner import find_sumo
        if find_sumo() is None:
            pytest.skip("SUMO not installed (SUMO_HOME unset)")
    logging.disable(logging.WARNING)
    for r in _committed(fname, picks):
        new = _rerun(r, backend)
        for k in METRICS + ("causal_coverage",):
            if r[k] in ("", "nan"):
                assert str(new[k]) in ("", "nan"), (k, r[k], new[k])
            elif k in EXACT:
                assert str(new[k]) == r[k], (k, r[k], new[k])
            else:
                # Floating aggregates differ in the last bit between this Linux machine and the Windows laptop that
                # wrote the rows, with and without the revision-8 changes alike (measured: rwcr <= 1 ulp,
                # risk_peak_corr ~1e-15); the byte-level check is test_flag_off_engine_equals_revision7_engine.
                assert float(new[k]) == pytest.approx(float(r[k]), rel=1e-12, abs=1e-15), (k, r[k], new[k])


def test_defaults_are_off():
    s = SimSettings()
    assert s.dup_count_per_frame is False and s.nb_table_error is None and s.obs_lag_steps == 0
    assert s.pdr_perturb is None
    s2 = SimSettings.from_config(load_yaml("experiment.yaml"))
    assert s2.dup_count_per_frame is False and s2.nb_table_error is None and s2.obs_lag_steps == 0
    assert s2.pdr_perturb is None


# ------------------------------------------------------------------------------------- direct engine runs --
def _engine_run(settings, policy="slotted_1p", params=None, scenario="rural_highway", density=20.0, seed=100,
                spy=None):
    from agents.registry import build_policy
    from common.seeding import SeedBundle
    from hazard.model import hazard_from_config
    from hazard.risk_field import build_risk_field
    from mobility.generate import get_trace, load_scenario
    from sim.engine import DisseminationEngine
    from sim.mac import build_mac
    from sim.phy import build_phy
    logging.disable(logging.WARNING)
    phy_cfg, hz_cfg = load_yaml("phy.yaml"), load_yaml("hazard.yaml")
    sc = load_scenario(scenario)
    tr = get_trace(sc, density, seed, weather="clear", backend="fallback", phy_cfg=phy_cfg)
    hz = hazard_from_config(hz_cfg, tr.meta, overrides={"type": "fog_bank"})
    phy = build_phy(phy_cfg, scenario=sc.get("phy_profile", sc["name"]), weather="clear", seed=seed, trace_meta=tr.meta)
    pol = build_policy(policy, **(params or {"n_slots": 5}))
    if spy is not None:
        inner = pol.decide

        def decide(ctx):
            spy.append(ctx)
            return inner(ctx)
        pol.decide = decide
    eng = DisseminationEngine(tr, phy, build_mac(phy_cfg, phy), build_risk_field(hz_cfg, tr, hz), hz, pol,
                              SeedBundle(master_seed=seed), settings)
    return eng.run(), tr


def _fp(res):
    return _fingerprint(res, {})


BASE = SimSettings.from_config(load_yaml("experiment.yaml"))


@pytest.mark.parametrize("cell", [("rural_highway", 20.0), ("urban_nlos", 20.0)])
def test_explicit_off_and_zero_effect_settings_are_byte_identical(cell):
    sc, d = cell
    ref, _ = _engine_run(BASE, scenario=sc, density=d)
    for kw in ({"dup_count_per_frame": False, "nb_table_error": None, "obs_lag_steps": 0, "pdr_perturb": None},
               {"nb_table_error": (0.0, 0.0, 1.15)},
               {"pdr_perturb": {"u": 0.0, "bin_m": 50.0, "gap": [0.3] * 13, "pdr_custom": [0.5] * 13}}):
        res, _ = _engine_run(dataclasses.replace(BASE, **kw), scenario=sc, density=d)
        assert _fp(res) == _fp(ref), kw


def test_r1_per_frame_counting_cancels_more_in_a_dense_cell():
    off, _ = _engine_run(BASE, density=80.0)
    on, _ = _engine_run(dataclasses.replace(BASE, dup_count_per_frame=True), density=80.0,
                        params={"n_slots": 50, "cancel_on_duplicates": 2})
    off2, _ = _engine_run(BASE, density=80.0, params={"n_slots": 50, "cancel_on_duplicates": 2})
    assert on.n_transmissions < off2.n_transmissions
    assert off.n_transmissions > 0


def test_r2_changes_neighbour_counts():
    a, b = [], []
    _engine_run(BASE, spy=a)
    _engine_run(dataclasses.replace(BASE, nb_table_error=(0.3, 0.3, 1.15)), spy=b)
    na = [c.n_neighbours for c in a if c.trigger.value == "receive"][:50]
    nb = [c.n_neighbours for c in b if c.trigger.value == "receive"][:50]
    assert na != nb


def test_r3_lag_shows_the_previous_epoch():
    ctxs = []
    _, tr = _engine_run(dataclasses.replace(BASE, obs_lag_steps=1), spy=ctxs)
    checked = 0
    for c in ctxs:
        if c.trigger.value != "receive" or not tr.active[c.step - 1, c.index]:
            continue
        assert c.x == float(tr.x[c.step - 1, c.index]) and c.y == float(tr.y[c.step - 1, c.index])
        checked += 1
    assert checked > 10


def test_r5_positive_gap_raises_delivery():
    off, _ = _engine_run(BASE)
    on, _ = _engine_run(dataclasses.replace(
        BASE, pdr_perturb={"u": 1.0, "bin_m": 50.0, "gap": [0.4] * 13, "pdr_custom": [0.5] * 13}))
    assert on.n_rx_success / on.n_rx_attempts > off.n_rx_success / off.n_rx_attempts


def _r7_engine_module():
    import importlib.util
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    try:
        src = subprocess.check_output(["git", "show", f"{R7_COMMIT}:sim/engine.py"], cwd=PROJECT_ROOT, text=True)
    except Exception:
        pytest.skip("revision-7 engine source not available from git")
    p = Path(tempfile.mkdtemp()) / "engine_r7.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("engine_r7", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine_r7"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("cell", [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0),
                                  ("urban_nlos", 20.0)])
@pytest.mark.parametrize("policy,params", [("slotted_1p", {"n_slots": 5}), ("rar", {"abc": [1.0, 0.5, 0.0]}),
                                           ("counter_based", {"counter_threshold": 3})])
def test_flag_off_engine_equals_revision7_engine(cell, policy, params, monkeypatch):
    """Byte-identical RunResult: the revision-8 engine with every flag off vs the revision-7 engine source (a6dea21)."""
    import sim.engine as cur
    r7 = _r7_engine_module()
    sc, d = cell
    a, _ = _engine_run(BASE, policy=policy, params=params, scenario=sc, density=d, seed=101)
    monkeypatch.setattr(cur, "DisseminationEngine", r7.DisseminationEngine)
    b, _ = _engine_run(r7.SimSettings.from_config(load_yaml("experiment.yaml")), policy=policy, params=params,
                       scenario=sc, density=d, seed=101)
    assert a.n_transmissions > 0
    assert _fp(a) == _fp(b)
