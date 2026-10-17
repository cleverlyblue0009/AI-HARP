"""R3: hierarchical bootstrap of rural d=20 regret / margin and of cells matched.

::

    python -m experiments.revision2.r3_hier_bootstrap --seeds 0 1 2 3 4
    python -m experiments.revision2.r3_hier_bootstrap --seeds 0 1 2

Each resample draws training seeds (with replacement, from the given list) and the 10
eval seeds (with replacement) JOINTLY: one eval-seed draw is applied to the baselines and
to every agent, so pairing is kept. Algorithm 1 is then run in full on the resampled
means -- per-cell oracle and best fixed setting are re-selected inside each resample --
and the statistic is averaged over the drawn training seeds. 10,000 resamples,
percentile 95% CIs, fixed RNG seed.

Rural d=20 regret / margin are defined only when the agent qualifies; the share of
(resample, training seed) draws where it does not is reported, and the CI is over
resamples of the mean among qualifying draws.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from analysis.comparator import Cell, CellKey, compare_policy
from analysis.pareto import OperatingPoint, PolicyCurve
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision2.common import agent_rows, baseline_rows, SEED_FILES

N_BOOT = 10_000
RNG_SEED = 20261018
R20 = ("rural_highway", 20.0)
FIELDS = ("rwcr", "tx_per_at_risk_informed", "actionable_deadline_miss_rate", "tir_median_s")

_DATA: dict = {}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def pack(rows):
    """{cell: {(policy, param, value): array[10 seeds, len(FIELDS)]}}."""
    out: dict = defaultdict(dict)
    tmp: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        ck = (r["scenario"], float(r["density"]))
        tmp[ck][(r["policy"], r["param"], r["value"])][int(r["seed"])] = [_num(r[f]) for f in FIELDS]
    for ck, settings in tmp.items():
        for s, by_seed in settings.items():
            out[ck][s] = np.array([by_seed[i] for i in range(10)], dtype=float)
    return out


def make_cells(packed_list, idx):
    """Comparator cells from per-seed arrays, aggregated over eval-seed indices idx."""
    cells = {}
    for packed in packed_list:
        for ck, settings in packed.items():
            curves = cells.setdefault(ck, {})
            for (pol, param, value), arr in settings.items():
                a = arr[idx]
                m = np.nanmean(a, axis=0)
                val = None if value in ("", None) else (int(float(value)) if float(value).is_integer()
                                                       and "." not in str(value) else float(value))
                curves.setdefault(pol, []).append(OperatingPoint(
                    policy=pol, param=param, value=val, quality=m[0], quality_std=0.0, cost=m[1],
                    cost_std=0.0, n_seeds=len(idx),
                    extras={"actionable_deadline_miss_rate": m[2], "tir_median_s": m[3]}))
    return [Cell(key=CellKey(sc, d, "clear", "fog_bank"), curves={p: PolicyCurve(p, pts) for p, pts in c.items()})
            for (sc, d), c in cells.items()]


def one(b):
    rng = np.random.default_rng([RNG_SEED, b])
    seeds = _DATA["seeds"]
    tdraw = rng.choice(seeds, size=len(seeds), replace=True)
    idx = rng.integers(0, 10, size=10)
    regs, mars, matched, fails = [], [], [], 0
    cache = {}
    for s in tdraw:
        if s not in cache:
            res = compare_policy(make_cells([_DATA["base"], _DATA["agent"][s]], idx), "ai_harp")
            k = next(k for k in res.agent_cost if (k.scenario, k.density) == R20)
            cache[s] = (res.regret_per_cell[k], res.margin_per_cell[k], 4 - res.cells_failed)
        r, m, n = cache[s]
        matched.append(n)
        if np.isfinite(r):
            regs.append(r); mars.append(m)
        else:
            fails += 1
    return (np.mean(regs) if regs else np.nan, np.mean(mars) if mars else np.nan,
            float(np.mean(matched)), fails / len(tdraw))


def _init(base, agents, seeds):
    _DATA.update(base=base, agent=agents, seeds=seeds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=N_BOOT)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    base = pack(baseline_rows("headline"))
    agents = {s: pack(agent_rows(SEED_FILES["headline"][s], 0.5)) for s in args.seeds}
    # point estimate (no resampling)
    full = np.arange(10)
    point = {}
    for s in args.seeds:
        res = compare_policy(make_cells([base, agents[s]], full), "ai_harp")
        k = next(k for k in res.agent_cost if (k.scenario, k.density) == R20)
        point[s] = {"regret": res.regret_per_cell[k], "margin": res.margin_per_cell[k],
                    "cells_matched": 4 - res.cells_failed}
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(base, agents, args.seeds)) as ex:
        draws = np.array(list(ex.map(one, range(args.n), chunksize=50)), dtype=float)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    finite = lambda v: v[np.isfinite(v)]  # noqa: E731
    out = {"training_seeds": args.seeds, "n_boot": args.n, "rng_seed": RNG_SEED, "point_per_seed": point,
           "point_mean": {q: float(np.mean([p[q] for p in point.values() if np.isfinite(p[q])]))
                          for q in ("regret", "margin", "cells_matched")},
           "rural_d20_regret_ci95": ci(finite(draws[:, 0])), "rural_d20_margin_ci95": ci(finite(draws[:, 1])),
           "cells_matched_ci95": ci(draws[:, 2]), "cells_matched_boot_mean": float(draws[:, 2].mean()),
           "share_draws_agent_fails_rural_d20": float(draws[:, 3].mean()),
           "share_resamples_regret_negative": float((finite(draws[:, 0]) < 0).mean()),
           "share_resamples_margin_positive": float((finite(draws[:, 1]) > 0).mean())}
    out["provenance"] = stamp("R3", {"training_seeds": args.seeds, "n_boot": args.n, "rng_seed": RNG_SEED,
                                     "resampling": "training seeds and eval seeds jointly, with replacement; "
                                                   "Algorithm 1 re-run per resample"},
                              seeds={"eval": list(range(10)), "train": args.seeds})
    tag = "".join(map(str, args.seeds))
    write_json(out_dir("R3", 2) / f"r3_hier_bootstrap_seeds{tag}.json", out)
    print({k: v for k, v in out.items() if k != "provenance"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
