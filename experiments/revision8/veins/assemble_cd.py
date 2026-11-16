"""Revision 8 (D15): assemble the Experiment C and D shards (results/revision8/veins/ci_cd/rural_highway_d20_s<seed>/,
from branch r8-ci) into results/revision8/veins/expC_rows.jsonl and expD_rows.jsonl (+ _merge.json).

A shard is accepted only if both gate checks passed (classical 707d6533..., AI-HARP 220dd1fd...), the checkpoints'
hashes matched, and for each experiment all 30 replays were accepted and all 30 decision logs reproduced offline.
Written before any C/D result existed.

    python -m experiments.revision8.veins.assemble_cd
"""

from __future__ import annotations

import json

from common.config import PROJECT_ROOT

V = PROJECT_ROOT / "results" / "revision8" / "veins"
SEEDS = range(200, 210)


def main() -> int:
    acc, rej = [], {}
    rows = {"expC": [], "expD": []}
    for s in SEEDS:
        d = V / "ci_cd" / f"rural_highway_d20_s{s}"
        g = json.loads((d / "gate.json").read_text()) if (d / "gate.json").exists() else {}
        m = json.loads((d / "merge.json").read_text()) if (d / "merge.json").exists() else {}
        why = []
        if not g.get("pass"):
            why.append(f"gate {g.get('check1', {}).get('pass')}/{g.get('check2', {}).get('pass')}")
        for e in ("expC", "expD"):
            x = m.get(e, {})
            if x.get("merged") != 30 or not x.get("trace_all_pass") or x.get("trace_logs") != 30:
                why.append(f"{e}: merged {x.get('merged')}, trace {x.get('trace_all_pass')} ({x.get('trace_logs')})")
        if why:
            rej[d.name] = why
            continue
        acc.append(d.name)
        for e in rows:
            rows[e] += [json.loads(l) for l in (d / f"rows_{e}.jsonl").open(encoding="utf-8")]
    from experiments.revision1.provenance import stamp, write_json
    for e, rs in rows.items():
        with open(V / f"{e}_rows.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, default=str) + "\n")
        write_json(V / f"{e}_rows_merge.json", {"provenance": stamp(f"revision8/veins/assemble_cd/{e}", {}),
                                                "shards_accepted": len(acc), "shards_rejected": rej, "rows": len(rs)})
    print(f"accepted {len(acc)} / {len(SEEDS)} shards; rows C {len(rows['expC'])}, D {len(rows['expD'])}; rejected {rej}")
    return 0 if not rej else 1


if __name__ == "__main__":
    raise SystemExit(main())
