"""Committed map run name -> checkpoint_sha for every checkpoint behind a revision-5 policy row.
Writes results/revision5/checkpoint_shas.json.

    python -m experiments.revision5.checkpoint_shas

The analysis tells training runs apart by the checkpoint_sha recorded in each row's params_json. The checkpoints
themselves are not committed (size), so a clean clone reads this map instead (h2.run_sha); where the checkpoint file
exists, its hash must equal the map's. Every hash is also checked to occur in the committed rows.
"""

from __future__ import annotations

import json
import sys

from common.config import PROJECT_ROOT
from experiments.evaluate_agent import checkpoint_sha
from experiments.revision1.provenance import stamp, write_json
from experiments.revision5.sweep import checkpoint_for

RUNS = ([f"ref_s{k}" for k in range(10)] + [f"{v}_s{k}" for v in ("rungB", "rungC", "gcn", "mlp", "norel") for k in range(5)]
        + [f"ref_s{k}_u2000@{u}" for k in range(3) for u in (1200, 1500, 1700, 2000)])
OUT = PROJECT_ROOT / "results" / "revision5" / "checkpoint_shas.json"


def main(argv=None) -> int:
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    shas = {r: checkpoint_sha(checkpoint_for(r)) for r in RUNS}
    seen = set()
    for f in (PROJECT_ROOT / "results" / "revision5" / "runs").glob("agent__*.csv"):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                i = line.find("checkpoint_sha")
                if i >= 0:
                    seen.add(line[i:i + 60].split('"')[4] if line[i:i + 60].count('"') >= 5 else "")
    missing = [r for r, s in shas.items() if s not in seen]
    if missing:
        print(f"[shas] not found in any committed row: {missing}", flush=True)
        return 1
    write_json(OUT, {"provenance": stamp("revision5/checkpoint_shas", {"runs": RUNS}), "shas": shas})
    print(OUT, len(shas), "runs, all found in committed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
