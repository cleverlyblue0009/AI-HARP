"""Rule 7 bit-exactness gate: re-run ref_s0 at rural d=20 (bias -2 and +1, tau=0.5, sampled,
eval seeds 0-9) and assert all 20 rows match results/revision1/agent_runs/ref_s0.csv to <= 1e-12.

::

    python -m experiments.revision5.bitexact

Exit code 0 = pass, 1 = fail (and the differing rows are printed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.evaluate_agent import agent_params
from experiments.revision1.runner import load_rows, run_task, task

FIELDS = ("rwcr", "tx_per_at_risk_informed", "tir_median_s", "tir_p95_s", "actionable_deadline_miss_rate",
          "transmissions", "gate_fallback_rate")
TOL = 1e-12


def main() -> int:
    ck = (PROJECT_ROOT / "checkpoints" / "campaign" / "ref" / "ckpt_final.pt").resolve()
    ref = {}
    for r in load_rows(PROJECT_ROOT / "results" / "revision1" / "agent_runs" / "ref_s0.csv"):
        q = json.loads(r["params_json"])
        if (r["scenario"], float(r["density"])) == ("rural_highway", 20.0) and float(q.get("tau", -1)) == 0.5 \
                and not q.get("deterministic") and float(r["value"]) in (-2.0, 1.0):
            ref[(float(r["value"]), int(r["seed"]))] = r
    assert len(ref) == 20, f"expected 20 reference rows, found {len(ref)}"
    bad, worst = [], 0.0
    for (b, s), r in sorted(ref.items()):
        new = run_task(task("rural_highway", 20.0, "ai_harp", s, param="suppression_bias", value=b,
                            fixed=agent_params(ck, tau=0.5, deterministic=False)))
        for f in FIELDS:
            x, y = float(r[f]), float(new[f])
            d = 0.0 if (x != x and y != y) else abs(x - y)
            worst = max(worst, d)
            if d > TOL:
                bad.append((b, s, f, x, y))
    print(f"bit-exactness: 20 rows x {len(FIELDS)} fields, worst |diff| = {worst:.3e} -> "
          f"{'PASS' if not bad else 'FAIL'}")
    for b in bad[:20]:
        print("  differs:", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
