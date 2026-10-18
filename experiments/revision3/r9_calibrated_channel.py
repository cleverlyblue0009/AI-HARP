"""R9: calibrated-channel stress test.

::

    python -m experiments.revision3.r9_calibrated_channel fit     --jobs 3
    python -m experiments.revision3.r9_calibrated_channel rerun   --jobs 3
    python -m experiments.revision3.r9_calibrated_channel analyse

fit
    The density-dependent extra reception loss of sim/mac.py,
    p(n) = p_max * clip((n - n0) / (n1 - n0), 0, 1), n = receiver's neighbours within
    carrier-sense range. The ramp is placed from the validation geometry (Amador et al.:
    8 lanes, 778 m range, so n ~ 12.4 x density: 124 / 249 / 373 / 498 / 622 at 10-50
    veh/km/lane), where the published PDR steps down between 20 and 30: n0 = 250,
    n1 = 373. p_max is fitted by bisection so the mean |ours - paper| over 30-50 is
    minimised (10 seeds), then the full curve is re-run with 30 seeds and must be within
    0.01 at every density.

rerun
    With the fitted knob: all 55 baseline settings x 4 headline cells x 10 eval seeds,
    the agent (full model seeds 0-4, tau 0.5, sampled, biases -2..+3), and the R4 adaptive
    pick (already one of the 55 settings).

analyse
    Algorithm 1 on the calibrated runs; for each headline finding, same vs changed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json

R = PROJECT_ROOT / "results"
D = out_dir("R9", 3)
N0, N1 = 250.0, 373.0
PAPER = {10.0: 0.9998, 20.0: 0.9961, 30.0: 0.9280, 40.0: 0.9371, 50.0: 0.9372}


def _val_run(args):
    density, seed, p_max = args
    import experiments.validate_amador as V

    orig = V.configs

    def patched(*a, **k):
        c = orig(*a, **k)
        c["phy"]["mac"]["extra_loss"] = {"p_max": p_max, "n0": N0, "n1": N1}
        return c

    V.configs = patched
    r = V.run_one((density, seed))
    return {"density": density, "seed": seed, "p_max": p_max, "pdr": r["paper"], "n_area": r["n_area"]}


def curve(p_max, densities, seeds, jobs):
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(_val_run, [(d, s, p_max) for d in densities for s in range(seeds)]))
    return {d: float(np.nanmean([r["pdr"] for r in rows if r["density"] == d])) for d in densities}, rows


def fit(jobs):
    hi_d = (30.0, 40.0, 50.0)
    lo, hi, hist = 0.0, 1.0, []
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        c, _ = curve(mid, hi_d, 10, jobs)
        gap = float(np.mean([c[d] - PAPER[d] for d in hi_d]))     # >0: still too optimistic
        hist.append({"p_max": mid, "curve": c, "mean_gap": gap})
        print(f"p_max {mid:.4f} -> {c} gap {gap:+.4f}", flush=True)
        lo, hi = (mid, hi) if gap > 0 else (lo, mid)
    p_best = min(hist, key=lambda h: abs(h["mean_gap"]))["p_max"]
    full, rows = curve(p_best, sorted(PAPER), 30, jobs)
    dev = {d: full[d] - PAPER[d] for d in PAPER}
    out = {"function": "p(n) = p_max * clip((n - n0)/(n1 - n0), 0, 1); n = neighbours within carrier-sense range",
           "rng_stream": "extra_loss (own named stream)",
           "n0": N0, "n1": N1, "p_max": p_best, "bisection": hist, "verification_30_seeds": full,
           "deviation": dev, "max_abs_deviation": max(abs(v) for v in dev.values()),
           "within_0.01_everywhere": all(abs(v) <= 0.01 for v in dev.values()), "rows": rows,
           "provenance": stamp("R9/fit", {"n0": N0, "n1": N1, "paper": PAPER}, seeds=list(range(30)))}
    write_json(D / "r9_fit.json", out)
    print("fitted p_max", p_best, "verification", full, "max |dev|", out["max_abs_deviation"])
    return out


def patch():
    f = json.loads((D / "r9_fit.json").read_text())
    return {"mac": {"extra_loss": {"p_max": f["p_max"], "n0": f["n0"], "n1": f["n1"]}}}


def rerun(jobs):
    from experiments.evaluate_agent import DEFAULT_BIASES, agent_params
    from experiments.revision1.runner import run_tasks, task
    from experiments.revision1.sweeps import HEADLINE, knob_settings

    pp = patch()
    base = [task(sc, d, p, s, param=a, value=v, phy_patch=pp)
            for sc, d in HEADLINE for p, a, v in knob_settings() for s in range(10)]
    run_tasks(base, D / "r9_baseline_runs.csv", jobs=jobs)
    for seed in range(5):
        ck = (PROJECT_ROOT / "checkpoints" / "campaign" / "ref" / "ckpt_final.pt" if seed == 0
              else PROJECT_ROOT / "checkpoints" / "revision1" / f"ref_s{seed}" / "ckpt_final.pt")
        fixed = agent_params(ck, tau=0.5, deterministic=False)
        ag = [task(sc, d, "ai_harp", s, param="suppression_bias", value=float(b), fixed=fixed, phy_patch=pp)
              for sc, d in HEADLINE for b in DEFAULT_BIASES for s in range(10)]
        run_tasks(ag, D / f"r9_agent_ref_s{seed}.csv", jobs=jobs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "rerun", "analyse"])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    if a.stage == "fit":
        fit(a.jobs)
    elif a.stage == "rerun":
        rerun(a.jobs)
    else:
        from experiments.revision3.r9_analyse import analyse

        analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
