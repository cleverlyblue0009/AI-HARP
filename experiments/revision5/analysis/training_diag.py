"""Training diagnostics from every revision-5-relevant history.jsonl (analysis/training_diagnostics.json).

    python -m experiments.revision5.analysis.training_diag

Per run: final Lagrange multiplier lambda per curriculum group and its slope over the last 100 updates (OLS, 95 %
CI), final shortfall per group, the consecutive-compliant counter (early_stop_run: final and max), lambda
saturation, and lambda/shortfall trajectories every 10 updates (for the figure). rural_highway|2 is highlighted.
Histories are read from results/revision1/checkpoints/<run>_history.jsonl; runs whose history is not committed
there yet are copied in from checkpoints/ first (provenance: committed inputs only).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision5.analysis.common import write

R1 = PROJECT_ROOT / "results" / "revision1" / "checkpoints"
RUNS = ([f"ref_s{k}" for k in range(10)] + [f"{v}_s{k}" for v in ("rungB", "rungC", "gcn", "mlp", "norel") for k in range(5)]
        + [f"ref_s{k}_u2000" for k in range(3)])
FOCUS = "rural_highway|2"


def source(run: str) -> Path | None:
    cands = {"ref_s0": "checkpoints/campaign/ref", "gcn_s0": "checkpoints/campaign/gcn", "mlp_s0": "checkpoints/campaign/mlp",
             "norel_s0": "checkpoints/campaign/no_relevance"}
    for d in ([cands[run]] if run in cands else []) + [f"checkpoints/revision1/{run}", f"checkpoints/revision3/{run}"]:
        p = PROJECT_ROOT / d / "history.jsonl"
        if p.exists():
            return p
    return None


def slope_ci(y: np.ndarray) -> tuple[float, float, float]:
    from scipy import stats
    x = np.arange(len(y), dtype=float)
    r = stats.linregress(x, y)
    t = stats.t.ppf(0.975, len(y) - 2)
    return float(r.slope), float(r.slope - t * r.stderr), float(r.slope + t * r.stderr)


def main() -> int:
    out, copied, missing = {}, [], []
    for run in RUNS:
        dst = R1 / f"{run}_history.jsonl"
        if not dst.exists():
            src = source(run)
            if src is None:
                missing.append(run)
                continue
            shutil.copy2(src, dst)
            copied.append(run)
        H = [json.loads(l) for l in dst.open(encoding="utf-8")]
        groups = sorted({g for h in H for g in (h.get("lambda_by_group") or {})})
        last = H[-100:]
        per = {}
        for g in groups:
            lam = np.array([h["lambda_by_group"].get(g, np.nan) for h in last], dtype=float)
            ok = lam[np.isfinite(lam)]
            s, lo, hi = slope_ci(ok) if ok.size > 10 else (math.nan,) * 3
            # the curriculum samples a subset of groups per update: "final" = last value recorded for the group
            lam_all = [h["lambda_by_group"][g] for h in H if g in (h.get("lambda_by_group") or {})]
            sf_all = [h["shortfall_by_group"][g] for h in H if g in (h.get("shortfall_by_group") or {})]
            per[g] = {"final_lambda": float(lam_all[-1]) if lam_all else math.nan,
                      "final_shortfall": float(sf_all[-1]) if sf_all else math.nan,
                      "n_updates_with_group": len(lam_all),
                      "slope_last100": s, "slope_ci95": [lo, hi],
                      "slope_ci_excludes_0": bool(math.isfinite(lo) and (lo > 0 or hi < 0))}
        traj = {"update": [h["update"] for h in H[::10]],
                "lambda": {g: [h["lambda_by_group"].get(g) for h in H[::10]] for g in groups},
                "shortfall": {g: [(h.get("shortfall_by_group") or {}).get(g) for h in H[::10]] for g in groups},
                "early_stop_run": [h.get("early_stop_run") for h in H[::10]]}
        out[run] = {"n_updates": len(H), "groups": per,
                    "early_stop_run_final": H[-1].get("early_stop_run"),
                    "early_stop_run_max": max(h.get("early_stop_run", 0) for h in H),
                    "lambda_saturated_groups_final": H[-1].get("lambda_saturated_groups"),
                    "focus_" + FOCUS: per.get(FOCUS), "trajectory_every_10": traj}
    payload = {"runs": out, "histories_copied_into_results": copied, "missing_history": missing, "focus_group": FOCUS}
    print(write("training_diagnostics.json", payload, {"runs": RUNS}))
    print("copied", copied, "missing", missing)
    for run, v in out.items():
        f = v.get("focus_" + FOCUS)
        if f:
            print(run, "rural d=2 final lambda %.2f slope %.4f CI [%.4f, %.4f]" % (f["final_lambda"], f["slope_last100"], *f["slope_ci95"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
