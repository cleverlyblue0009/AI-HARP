"""Revision 6: accept and score the rule / policy Veins replays (results/revision6/veins/veins6_rows.jsonl).

    python -m experiments.revision6.veins.veins6_score [--runs cache/veins_runs/r6]

Acceptance (the revision-5 merge rule, experiments/revision5/cloud/merge.py, for replays made on this laptop): exit
code 0, an identical SUMO trace, and the app binary that passed the revision-6 identity gate (its sha256 in
results/revision6/veins/app_gate.json). Accepted replays are scored with the revision-5 scorer's own functions,
unchanged (experiments/revision5/veins/veins_metrics.py: context, parse_events, reconstruct, scalars, busy_ratio ->
analysis.metrics.compute_metrics on the canonical trace, hazard and risk field of the cell). Rejected replays are
counted and listed, never scored; a rerun of the replay replaces them. Resumable by replay name.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from common.config import PROJECT_ROOT, load_yaml
from experiments.revision1.provenance import stamp
from experiments.revision5.veins import veins_metrics as vm

RUNS = PROJECT_ROOT / "cache" / "veins_runs" / "r6"
OUT = PROJECT_ROOT / "results" / "revision6" / "veins"
GATE = OUT / "app_gate.json"


def accepted(rec: dict, app_sha: str) -> tuple[bool, str]:
    if rec.get("exit_code") != 0:
        return False, f"exit {rec.get('exit_code')}"
    if not rec.get("fcd_check", {}).get("identical"):
        return False, "trace not identical"
    if rec.get("app_sha256") != app_sha:
        return False, "app binary differs from the gated one"
    return True, ""


def score(run_json: Path) -> dict:
    rec = json.loads(run_json.read_text())
    name = run_json.name[: -len(".run.json")]
    base = run_json.parent / name
    scen, rest = rec["bundle"].split("_d")
    dens, seed = rest.split("_s")
    row = {"bundle": rec["bundle"], "scenario": scen, "density": float(dens), "seed": int(seed),
           "policy": rec["policy"], "knobs": rec["knobs"], "veins_seed": rec["seed"], "wall_s": rec["wall_s"]}
    ev = vm.parse_events(Path(str(base) + ".events.csv.gz"))
    res, raw = vm.reconstruct(ev, vm.context(scen, float(dens), int(seed)))
    from analysis.metrics import compute_metrics
    m = compute_metrics(res, load_yaml("hazard.yaml"))
    sca = vm.scalars(Path(str(base) + ".sca.gz"))
    raw["channel_busy_ratio"] = vm.busy_ratio(sca)
    for k in ("rwcr", "causal_coverage", "tx_per_at_risk_informed", "actionable_deadline_miss_rate", "tir_median_s",
              "tir_p95_s", "pdr", "transmissions", "deadline_miss_rate", "stop_deadline_warned_share_actionable"):
        v = m.get(k, math.nan)
        row[k] = float(v) if isinstance(v, (int, float)) else v
    row["channel_busy_ratio"] = raw["channel_busy_ratio"]
    row["action_mix"] = dict(Counter(d[4] for d in ev.get("D", [])))   # D: t, step, idx, trigger, action, delay
    return row


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT / "veins6_rows.jsonl")
    a = ap.parse_args(argv)
    os.environ.setdefault("SUMO_HOME", "D:/sumo/sumo-1.19.0")
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    app_sha = json.loads(GATE.read_text())["app_sha256"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["name"] for l in a.out.open(encoding="utf-8")} if a.out.exists() else set()
    rejected = {}
    n_new = 0
    with a.out.open("a", encoding="utf-8") as fh:
        for p in sorted(a.runs.glob("*.run.json")):
            name = p.name[: -len(".run.json")]
            if name in done:
                continue
            ok, why = accepted(json.loads(p.read_text()), app_sha)
            if not ok:
                rejected[name] = why
                continue
            try:
                row = score(p)
            except Exception as exc:  # noqa: BLE001 -- logged; a rerun scores it again
                rejected[name] = f"scoring failed: {exc!r}"
                continue
            fh.write(json.dumps({"name": name, "row": row}, default=str) + "\n")
            fh.flush()
            n_new += 1
    total = len(done) + n_new
    summary = {"merged": total, "new": n_new, "rejected": len(rejected), "rejected_replays": rejected}
    (a.out.parent / f"{a.out.stem}_merge.json").write_text(json.dumps(
        {"provenance": stamp("revision6/veins/veins6_score", {"runs": str(a.runs.name), "app_sha256": app_sha}),
         **summary}, indent=1))
    print(json.dumps({k: summary[k] for k in ("merged", "new", "rejected")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
