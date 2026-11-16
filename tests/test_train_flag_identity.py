"""Revision 9: with the deadline-miss flag off, the STRUCT trainer is revision 8's.

* Function level, against the revision-8 source (git b67f18c): run_episode and fitness give identical results on
  plain and randomised (fallback-backend) episodes of a real batch.
* One full generation (slow, opt-in with AIHARP_SLOW=1): the first history line of plain_s0 equals the committed
  revision-8 line except wall and CPU time (plain_s0 was trained in this cloud container type).
* Flag on: always-suppress (p = 0) is infeasible under results/revision9/miss_targets.json.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from common.config import PROJECT_ROOT

R8 = "b67f18c"


def _r8_trainer():
    try:
        src = subprocess.check_output(["git", "show", f"{R8}:agents/train_structured.py"], cwd=PROJECT_ROOT, text=True)
    except Exception:
        pytest.skip("revision-8 trainer source not available from git")
    p = Path(tempfile.mkdtemp()) / "train_structured_r8.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("train_structured_r8", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_structured_r8"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_flag_off_episode_and_fitness_equal_revision8():
    import agents.train_structured as cur
    old = _r8_trainer()
    cfg = cur.load_yaml("agent.yaml")["objective"]
    targets = cur.CoverageTargets.load(PROJECT_ROOT / cfg["targets_path"], float(cfg["target_fraction"]),
                                       float(cfg["fallback_target"]))
    theta = cur.warm_theta()
    r5 = cur.r5_table()
    eps = [e for e in cur.batch(0, 0, "plain", r5) if e["cell"][1] != 80.0][:4]
    eps += [e for e in cur.batch(1, 3, "rand", r5) if e["axes"]["backend"] == "fallback" and e["cell"][1] != 80.0][:3]
    assert len(eps) >= 5
    a = [cur.run_episode(theta, e) for e in eps]
    b = [old.run_episode(theta, e) for e in eps]
    assert a == b
    assert cur.fitness(a, eps, targets) == old.fitness(b, eps, targets)


@pytest.mark.skipif(os.environ.get("AIHARP_SLOW") != "1", reason="slow: one full generation (set AIHARP_SLOW=1)")
def test_one_generation_equals_committed_revision8_plain_s0():
    import agents.train_structured as cur
    out = Path(tempfile.mkdtemp()) / "plain_s0"
    cur.train("plain", 0, generations=1, workers=4, out=out)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("wall_s", "main_cpu_s")}  # noqa: E731
    got = json.loads((out / "history.jsonl").read_text().splitlines()[0])
    ref = json.loads((PROJECT_ROOT / "results/revision8/train/plain_s0/history.jsonl").read_text().splitlines()[0])
    assert strip(got) == strip(ref)


def test_flag_on_always_suppress_infeasible():
    import agents.train_structured as cur
    mt = PROJECT_ROOT / "results" / "revision9" / "miss_targets.json"
    if not mt.exists():
        pytest.skip("results/revision9/miss_targets.json not yet computed")
    bounds = {k: float(v["b"]) for k, v in json.loads(mt.read_text())["groups"].items()}
    cfg = cur.load_yaml("agent.yaml")["objective"]
    targets = cur.CoverageTargets.load(PROJECT_ROOT / cfg["targets_path"], float(cfg["target_fraction"]),
                                       float(cfg["fallback_target"]))
    theta = np.zeros(21)                     # p = clip(0) = 0: only the originator transmits
    eps = [e for e in cur.batch(0, 0, "plain", cur.r5_table()) if e["cell"][1] != 80.0]
    outc = [cur.run_episode(theta, e, True) for e in eps]
    f = cur.fitness(outc, eps, targets, bounds)
    assert not f["feasible"] and f["f"] >= cur.INFEASIBLE
