"""Is the sparse coverage constraint feasible for ANY baseline, at any cost?

::

    D:/aiharp-env/python.exe -m experiments.sparse_feasibility --jobs 11

run8 finished 0.15 short of target on rural d=2 and urban d=2 with lambda at
its cap. Before that shortfall can be blamed on the learned policy, the target
itself has to be shown reachable. This sweeps every baseline over every knob
setting (analysis/pareto.py POLICY_SWEEPS, plus each registry default) in every
training weather x hazard cell of the sparse groups, and measures two things:

* TRAINING side -- CAUSAL coverage (what training optimises) on the whole
  training seed pool, against the training target (0.95 x a ceiling measured
  on only ``objective.target_seeds`` = 2 seeds). Reported as the pooled
  shortfall the multiplier sees, (sum target - sum coverage) / sum target,
  for the best fixed setting and for the per-episode hindsight envelope (the
  best setting chosen separately for every episode -- no fixed baseline can
  beat it).
* EVALUATION side -- ORACLE RWCR on the evaluation seeds, against the
  comparator's target (0.95 x the best setting's mean).

Every run is independently seeded, so ``--jobs`` changes wall-clock time only.
Rows are appended as they finish and skipped on restart, so an interrupted
sweep resumes. Writes results/feasibility/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from agents.constrained_reward import CoverageTargets, causal_coverage, warned_at_risk
from agents.registry import BASELINE_POLICIES, build_policy
from analysis.metrics import compute_metrics
from analysis.pareto import POLICY_SWEEPS
from common.config import PROJECT_ROOT, load_yaml
from common.seeding import SeedBundle
from hazard.model import hazard_from_config
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace, load_scenario
from sim.engine import DisseminationEngine, SimSettings
from sim.mac import build_mac
from sim.phy import build_phy

FIELDS = ("scenario", "density", "weather", "hazard_type", "split", "policy", "param",
          "value", "seed", "causal_coverage", "rwcr", "cost", "miss", "n_transmissions")
KEY_FIELDS = ("scenario", "density", "weather", "hazard_type", "policy", "param", "value", "seed")

_CFGS: dict[str, Any] = {}


def settings() -> list[tuple[str, str, Any]]:
    """Every knob setting of every sweep, plus each baseline's registry default."""
    out = [(pol, param, v) for pol, (param, values) in POLICY_SWEEPS.items() for v in values]
    out += [(pol, "", None) for pol in BASELINE_POLICIES]
    return out


def _init() -> None:
    logging.disable(logging.INFO)
    _CFGS.update(phy=load_yaml("phy.yaml"), hazard=load_yaml("hazard.yaml"),
                 experiment=load_yaml("experiment.yaml"))


def run_one(task: tuple) -> dict[str, Any]:
    """One episode; built exactly as experiments/coverage_targets.py builds it."""
    sc, d, w, h, split, pol, param, value, seed = task
    if not _CFGS:
        _init()
    cfgs = _CFGS
    scenario_cfg = load_scenario(sc)
    trace = get_trace(scenario_cfg, d, seed, weather=w, phy_cfg=cfgs["phy"])
    hazard = hazard_from_config(cfgs["hazard"], trace.meta, overrides={"type": h})
    phy = build_phy(cfgs["phy"], scenario_cfg["name"], w, seed, trace_meta=trace.meta)
    mac = build_mac(cfgs["phy"], phy)
    risk = build_risk_field(cfgs["hazard"], trace, hazard)
    policy = build_policy(pol, **({param: value} if param else {}))
    res = DisseminationEngine(trace, phy, mac, risk, hazard, policy, SeedBundle(master_seed=seed),
                              SimSettings.from_config(cfgs["experiment"])).run()
    m = compute_metrics(res, cfgs["hazard"])
    rel = risk.relevance_matrix(trace, hazard)
    peak, at_risk, warned = warned_at_risk(rel, res.informed_step, risk.at_risk_threshold)
    return {"scenario": sc, "density": d, "weather": w, "hazard_type": h, "split": split,
            "policy": pol, "param": param, "value": "" if value is None else value, "seed": seed,
            "causal_coverage": causal_coverage(peak, at_risk, warned),
            "rwcr": m.get("rwcr", np.nan), "cost": m.get("tx_per_at_risk_informed", np.nan),
            "miss": m.get("actionable_deadline_miss_rate", np.nan),
            "n_transmissions": int(res.n_transmissions)}


def _key(row: dict[str, Any]) -> tuple:
    return tuple(str(row[k]) if k not in ("density",) else f"{float(row[k]):g}" for k in KEY_FIELDS)


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _label(pol: str, param: str, value: Any) -> str:
    return f"{pol}({param}={value})" if param else f"{pol}(default)"


def summarise(rows: list[dict[str, Any]], cfg: dict[str, Any],
              targets: CoverageTargets, run8_history: Path | None) -> tuple[dict, str]:
    lines: list[str] = []
    out: dict[str, Any] = {"training": {}, "evaluation": {}}
    groups = sorted({(r["scenario"], f"{float(r['density']):g}") for r in rows})

    run8 = {}
    if run8_history is not None and run8_history.exists():
        hist = [json.loads(ln) for ln in run8_history.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        for sc, d in groups:
            g = f"{sc}|{d}"
            s = [r["shortfall_by_group"][g] for r in hist
                 if r.get("stage") == "finetune" and g in r.get("shortfall_by_group", {})
                 and int(r["update"]) > 800 and np.isfinite(r["shortfall_by_group"][g])]
            if s:
                run8[g] = float(np.mean(s))

    lines.append("=" * 100)
    lines.append(" TRAINING SIDE -- causal coverage on the training seed pool vs the training target")
    lines.append(" pooled shortfall = (sum target - sum coverage) / sum target over every episode of")
    lines.append(" the group (training weather x training hazard x seed), as the multiplier sees it")
    lines.append("=" * 100)
    for sc, d in groups:
        g = f"{sc}|{d}"
        tr = [r for r in rows if r["split"] == "train" and r["scenario"] == sc
              and f"{float(r['density']):g}" == d]
        if not tr:
            continue
        by_setting: dict[str, list[tuple[float, float]]] = defaultdict(list)
        envelope: dict[tuple, float] = {}
        tgt_of: dict[tuple, float] = {}
        for r in tr:
            cov = _f(r["causal_coverage"])
            t = targets.target(sc, float(d), r["weather"], r["hazard_type"])
            ep = (r["weather"], r["hazard_type"], int(r["seed"]))
            tgt_of[ep] = t
            if np.isfinite(cov):
                by_setting[_label(r["policy"], r["param"], r["value"])].append((cov, t))
                envelope[ep] = max(envelope.get(ep, -1.0), cov)

        def pooled(pairs):
            tt = sum(t for _, t in pairs)
            return (tt - sum(c for c, _ in pairs)) / tt if tt > 0 else float("nan")

        ranked = sorted(((pooled(p), lab, float(np.mean([c for c, _ in p])), len(p))
                         for lab, p in by_setting.items()))
        env_pairs = [(c, tgt_of[ep]) for ep, c in envelope.items()]
        env_short = pooled(env_pairs)
        mean_target = float(np.mean(list(tgt_of.values())))
        n_seeds = len({ep[2] for ep in tgt_of})
        out["training"][g] = {
            "mean_target": mean_target, "n_seeds": n_seeds, "n_episodes": len(tgt_of),
            "best_fixed": {"setting": ranked[0][1], "pooled_shortfall": ranked[0][0],
                           "mean_coverage": ranked[0][2]},
            "hindsight_envelope": {"pooled_shortfall": env_short,
                                   "mean_coverage": float(np.mean([c for c, _ in env_pairs]))},
            "settings_meeting_target": [lab for s, lab, _, _ in ranked if s <= 0],
            "run8_mean_shortfall_last200": run8.get(g),
            "ranked": [{"setting": lab, "pooled_shortfall": s, "mean_coverage": c, "n": n}
                       for s, lab, c, n in ranked],
        }
        lines.append(f"\n{g}: {len(tgt_of)} episodes ({n_seeds} seeds x 6 cells), "
                     f"mean target {mean_target:.3f}")
        for s, lab, c, n in ranked[:5]:
            lines.append(f"   {lab:40s} coverage {c:.3f}  pooled shortfall {s:+.3f}")
        lines.append(f"   {'hindsight envelope (best per episode)':40s} coverage "
                     f"{out['training'][g]['hindsight_envelope']['mean_coverage']:.3f}  "
                     f"pooled shortfall {env_short:+.3f}")
        lines.append(f"   fixed settings meeting the target: "
                     f"{len(out['training'][g]['settings_meeting_target'])} of {len(ranked)}")
        if g in run8:
            lines.append(f"   run8 agent, mean pooled shortfall over updates 801-1000: {run8[g]:+.3f}")

    lines.append("")
    lines.append("=" * 100)
    lines.append(" EVALUATION SIDE -- oracle RWCR on the evaluation seeds, per cell")
    lines.append(" comparator target = 0.95 x best setting's mean (so some setting meets it by")
    lines.append(" construction); the point is how far above the typical setting that ceiling sits")
    lines.append("=" * 100)
    cells = sorted({(r["scenario"], f"{float(r['density']):g}", r["weather"], r["hazard_type"])
                    for r in rows if r["split"] == "eval"})
    for cell in cells:
        ev = [r for r in rows if r["split"] == "eval" and
              (r["scenario"], f"{float(r['density']):g}", r["weather"], r["hazard_type"]) == cell]
        by_setting = defaultdict(list)
        env: dict[int, float] = {}
        for r in ev:
            q = _f(r["rwcr"])
            if np.isfinite(q):
                by_setting[_label(r["policy"], r["param"], r["value"])].append(
                    (q, _f(r["cost"]), _f(r["miss"])))
                env[int(r["seed"])] = max(env.get(int(r["seed"]), -1.0), q)
        stats = sorted(((float(np.mean([x[0] for x in v])), float(np.std([x[0] for x in v])),
                         float(np.nanmean([x[1] for x in v])), float(np.nanmean([x[2] for x in v])),
                         lab) for lab, v in by_setting.items()), reverse=True)
        ceiling = stats[0][0]
        target = 0.95 * ceiling
        meeting = [s for s in stats if s[0] >= target]
        name = "|".join(cell)
        out["evaluation"][name] = {
            "ceiling": ceiling, "ceiling_setting": stats[0][4], "ceiling_std": stats[0][1],
            "target": target, "n_settings": len(stats), "n_meeting": len(meeting),
            "median_setting_rwcr": float(np.median([s[0] for s in stats])),
            "per_seed_envelope_mean": float(np.mean(list(env.values()))),
            "cheapest_meeting": min(meeting, key=lambda s: s[2])[4] if meeting else None,
        }
        e = out["evaluation"][name]
        lines.append(f"\n{name}: ceiling {ceiling:.3f} +/- {stats[0][1]:.3f} ({stats[0][4]}), "
                     f"target {target:.3f}")
        lines.append(f"   settings meeting target: {len(meeting)} of {len(stats)} | median setting "
                     f"{e['median_setting_rwcr']:.3f} | per-seed envelope {e['per_seed_envelope_mean']:.3f}")
        for q, sd, c, miss, lab in stats[:3]:
            lines.append(f"   {lab:40s} RWCR {q:.3f} +/- {sd:.3f}  cost {c:.2f}  miss {miss:.3f}")
    return out, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sparse-constraint feasibility over every baseline setting")
    ap.add_argument("--scenarios", nargs="*", default=["rural_highway", "urban_nlos"])
    ap.add_argument("--densities", nargs="*", type=float, default=[2.0])
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default="results/feasibility")
    ap.add_argument("--run8-history", default="checkpoints/run8/history.jsonl")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_yaml("agent.yaml")
    t = cfg["training"]
    pool = t["train_seed_pool"]
    train_seeds = list(range(int(pool["start"]), int(pool["start"]) + int(pool["count"])))
    eval_seeds = list(range(args.eval_seeds))
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sparse_feasibility_runs.csv"

    tasks = [(sc, d, w, h, split, pol, param, value, seed)
             for sc in args.scenarios for d in args.densities
             for w in t["train_weather"] for h in t["train_hazards"]
             for split, seeds in (("train", train_seeds), ("eval", eval_seeds))
             for pol, param, value in settings() for seed in seeds]

    done: set[tuple] = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            done = {_key(r) for r in csv.DictReader(fh)}
    task_fields = ("scenario", "density", "weather", "hazard_type", "split", "policy", "param",
                   "value", "seed")

    def task_key(tk: tuple) -> tuple:
        row = dict(zip(task_fields, tk))
        row["value"] = "" if row["value"] is None else row["value"]
        return _key(row)

    todo = [tk for tk in tasks if task_key(tk) not in done]
    print(f"{len(tasks)} runs in total, {len(done)} already on disk, {len(todo)} to go "
          f"({len(settings())} settings, train seeds {train_seeds[0]}-{train_seeds[-1]}, "
          f"eval seeds 0-{eval_seeds[-1]})", flush=True)

    if todo and not args.summarise_only:
        new = not csv_path.exists()
        t0 = time.time()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            if args.jobs > 1:
                from concurrent.futures import ProcessPoolExecutor

                ex = ProcessPoolExecutor(max_workers=args.jobs, initializer=_init)
                it = ex.map(run_one, todo, chunksize=16)
            else:
                _init()
                ex, it = None, map(run_one, todo)
            try:
                for i, row in enumerate(it, 1):
                    wr.writerow(row)
                    if i % 500 == 0 or i == len(todo):
                        fh.flush()
                        el = time.time() - t0
                        print(f"  {i}/{len(todo)}  {el:.0f}s  eta {el / i * (len(todo) - i):.0f}s",
                              flush=True)
            finally:
                if ex is not None:
                    ex.shutdown(cancel_futures=True)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    o = cfg["objective"]
    targets = CoverageTargets.load(PROJECT_ROOT / o["targets_path"], o["target_fraction"],
                                   o["fallback_target"])
    summary, text = summarise(rows, cfg, targets, PROJECT_ROOT / args.run8_history)
    summary["provenance"] = {"n_runs": len(rows), "settings": len(settings()),
                             "train_seeds": train_seeds, "eval_seeds": eval_seeds,
                             "targets_path": o["targets_path"],
                             "target_seeds_used_for_ceiling": o.get("target_seeds")}
    (out_dir / "sparse_feasibility.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "sparse_feasibility.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
