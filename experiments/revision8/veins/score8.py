"""Revision 8: accept and score the Veins replays with the existing scorer (experiments/revision6/veins/veins6_score.py).

    python -m experiments.revision8.veins.score8 h5B     results/revision8/veins/h5B_rows.jsonl
    python -m experiments.revision8.veins.score8 h5A | expC | expD

Only the gate differs (the revision-8 binary, results/revision8/veins/app_gate.json). Resumable by replay name.
Scoring reconstructs from the logged steps, so Experiment D's shifted step <-> time mapping is scored on trace steps.
"""

from __future__ import annotations

import sys

from common.config import PROJECT_ROOT
from experiments.revision6.veins import veins6_score as s

OUT = PROJECT_ROOT / "results" / "revision8" / "veins"
ROOT = PROJECT_ROOT / "cache" / "veins_runs"


def main(argv=None) -> int:
    mode = (argv if argv is not None else sys.argv[1:])[0]
    if mode not in ("h5A", "h5B", "expC", "expD"):
        raise SystemExit(f"unknown mode {mode}")
    s.GATE = OUT / "app_gate.json"
    return s.main(["--runs", str(ROOT / f"r8_{mode}"), "--out", str(OUT / f"{mode}_rows.jsonl")])


if __name__ == "__main__":
    raise SystemExit(main())
