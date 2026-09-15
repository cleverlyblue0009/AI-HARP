"""Summarise the lambda-cap follow-up runs against run8.

::

    D:/aiharp-env/python.exe -m experiments.lambda_cap_summary

run8 plateaued with rural d=2 and urban d=2 short of target and lambda at its
cap (50). Two sparse-only finetunes from run8's pretrain checkpoint test
whether the cap was binding:

* A: cap 500, lambda starting at its usual value (dual ascent must climb);
* B: cap 500, sparse lambdas starting AT 500 (the whole run at the raised price).

Commands (D:/aiharp-gpu, deterministic CUDA, bit-reproducible)::

    python -m agents.train --stage finetune --init-from checkpoints/run8/ckpt_pretrain.pt \
        --sparse-only --lambda-max 500 --updates 300 --out checkpoints/feas_A_cap500
    python -m agents.train --stage finetune --init-from checkpoints/run8/ckpt_pretrain.pt \
        --sparse-only --lambda-max 500 --lambda-init 500 --updates 600 \
        --out checkpoints/feas_B_init500

(B was run to 300, then resumed to 600 with --resume; resume is exact.)
Writes results/feasibility/lambda_cap_arms.{json,txt}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT

GROUPS = ("rural_highway|1", "rural_highway|2", "rural_highway|3", "rural_highway|5",
          "urban_nlos|1", "urban_nlos|2", "urban_nlos|3", "urban_nlos|5")


def _history(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def group_stats(rows: list[dict], group: str, last: int, stage: str = "finetune") -> dict:
    """Mean and standard error of the group's pooled shortfall over the last
    ``last`` finetune updates, and the lambda at the end."""
    ft = [r for r in rows if r.get("stage") == stage]
    if not ft:
        return {}
    end = max(int(r["stage_update"]) for r in ft)
    pts = [r for r in ft if int(r["stage_update"]) > end - last
           and group in r.get("shortfall_by_group", {})
           and np.isfinite(r["shortfall_by_group"][group])]
    if not pts:
        return {}
    s = np.array([r["shortfall_by_group"][group] for r in pts])
    return {"shortfall_mean": float(s.mean()), "shortfall_se": float(s.std() / np.sqrt(len(s))),
            "n_updates_sampled": len(s), "window": f"finetune {end - last + 1}-{end}",
            "lambda_end": float(pts[-1]["lambda_next_by_group"][group])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run8", default="checkpoints/run8/history.jsonl")
    ap.add_argument("--arm-a", default="checkpoints/feas_A_cap500/history.jsonl")
    ap.add_argument("--arm-b", default="checkpoints/feas_B_init500/history.jsonl")
    ap.add_argument("--last", type=int, default=200)
    args = ap.parse_args(argv)

    runs = {"run8 (cap 50, mixed finetune)": args.run8,
            "A (cap 500, climbs from init)": args.arm_a,
            "B (cap 500, starts at 500)": args.arm_b}
    out: dict = {"window_updates": args.last, "runs": {}}
    lines = [f"Pooled shortfall per group, mean +/- s.e. over the last {args.last} finetune "
             "updates (>0 = short of target)", ""]
    header = f"{'group':<18}" + "".join(f"{name:>34}" for name in runs)
    lines += [header, "-" * len(header)]
    for name, path in runs.items():
        p = PROJECT_ROOT / path
        rows = _history(p) if p.exists() else []
        last = args.last if "A (" not in name else min(args.last, 100)
        out["runs"][name] = {"history": path, "window": last,
                             "groups": {g: group_stats(rows, g, last) for g in GROUPS}}
    for g in GROUPS:
        cells = []
        for name in runs:
            st = out["runs"][name]["groups"].get(g) or {}
            cells.append(f"{st['shortfall_mean']:+.3f}+/-{st['shortfall_se']:.3f} (lam {st['lambda_end']:.0f})"
                         if st else "n/a")
        lines.append(f"{g:<18}" + "".join(f"{c:>34}" for c in cells))
    lines += ["", "Run A uses its last 100 updates: its lambda was still climbing and reached "
              "~50, never the raised cap."]
    text = "\n".join(lines)
    dst = PROJECT_ROOT / "results" / "feasibility"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "lambda_cap_arms.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (dst / "lambda_cap_arms.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
