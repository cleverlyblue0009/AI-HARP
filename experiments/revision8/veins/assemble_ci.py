"""Revision 8 (D14): assemble the H5 Tier B shards produced on GitHub runners into the H5 input.

    python -m experiments.revision8.veins.assemble_ci      -> results/revision8/veins/h5B_rows.jsonl (+ _merge.json)

Reads results/revision8/veins/ci/<cell>_s<seed>/ (copied from branch r8-ci) for rural d=20 and urban NLOS d=20 x seeds
210-219 (D12). A shard is accepted only if (a) its gate passed (classical replay byte-identical to the revision-6 gate log
707d6533...), (b) its offline trace check reproduced every logged STRUCT decision, and (c) all 60 of its replays were
accepted (exit 0, identical SUMO trace, that runner's gated binary). Rows of an accepted shard are written unchanged;
a shard that fails any of these contributes no row and is listed. Written before any shard result existed.
"""

from __future__ import annotations

import json

from common.config import PROJECT_ROOT

V = PROJECT_ROOT / "results" / "revision8" / "veins"
CELLS = ("rural_highway_d20", "urban_nlos_d20")
SEEDS = range(210, 220)
EXPECTED = 60


def main() -> int:
    accepted, rejected, n_rows, shas = [], {}, 0, set()
    with open(V / "h5B_rows.jsonl", "w", encoding="utf-8") as fo:
        for c in CELLS:
            for s in SEEDS:
                d = V / "ci" / f"{c}_s{s}"
                why = []
                g = json.loads((d / "gate.json").read_text()) if (d / "gate.json").exists() else {}
                t = json.loads((d / "trace_check.json").read_text()) if (d / "trace_check.json").exists() else {}
                m = json.loads((d / "merge.json").read_text()) if (d / "merge.json").exists() else {}
                if not g.get("pass"):
                    why.append("gate not passed")
                if not t.get("all_pass") or t.get("n") != EXPECTED:
                    why.append(f"trace check {t.get('all_pass')} over {t.get('n')} logs")
                if m.get("merged") != EXPECTED:
                    why.append(f"merged {m.get('merged')} of {EXPECTED} (rejected {m.get('rejected')})")
                if why:
                    rejected[d.name] = why
                    continue
                rows = [json.loads(l) for l in (d / "rows.jsonl").open(encoding="utf-8")]
                assert len(rows) == EXPECTED, d
                for r in rows:
                    fo.write(json.dumps(r, default=str) + "\n")
                n_rows += len(rows)
                accepted.append(d.name)
                shas.add(m.get("app_sha256"))
    from experiments.revision1.provenance import stamp, write_json
    write_json(V / "h5B_rows_merge.json", {
        "provenance": stamp("revision8/veins/assemble_ci", {"cells": CELLS, "seeds": [SEEDS[0], SEEDS[-1]]}),
        "shards_accepted": len(accepted), "shards_rejected": rejected, "rows": n_rows,
        "runner_app_sha256": sorted(x for x in shas if x)})
    print(f"accepted {len(accepted)} / {len(CELLS) * len(SEEDS)} shards, {n_rows} rows; rejected: {rejected}")
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
