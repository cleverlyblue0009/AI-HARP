"""Revision 7: accept and score the Veins replays with the existing scorer (experiments/revision6/veins/veins6_score.py).

    python -m experiments.revision7.veins.score7 capped      results/revision7/veins/capped_rows.jsonl
    python -m experiments.revision7.veins.score7 d9          results/revision7/veins/d9_rows.jsonl

Only the gate differs (the revision-7 binary), and capped rows also count `C` lines. Resumable by replay name.
"""

from __future__ import annotations

import sys
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.revision5.veins import veins_metrics as vm
from experiments.revision6.veins import veins6_score as s

OUT = PROJECT_ROOT / "results" / "revision7" / "veins"
ROOT = PROJECT_ROOT / "cache" / "veins_runs"
_score = s.score


def score(run_json: Path) -> dict:
    row = _score(run_json)
    base = str(run_json)[: -len(".run.json")]
    row["capped_callbacks"] = len(vm.parse_events(Path(base + ".events.csv.gz")).get("C", []))
    return row


def main(argv=None) -> int:
    mode = (argv if argv is not None else sys.argv[1:])[0]
    s.GATE = OUT / "app_gate.json"
    s.score = score
    return s.main(["--runs", str(ROOT / f"r7_{mode}"), "--out", str(OUT / f"{mode}_rows.jsonl")])


if __name__ == "__main__":
    raise SystemExit(main())
