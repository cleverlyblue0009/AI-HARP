"""Revision-8 manifest: adds the revision-8 outputs that exist to results/MANIFEST.md and results/manifest.json.

    python -m experiments.revision8.manifest

Entries of earlier revisions are kept exactly as they are; earlier revision-8 entries are replaced. Outputs that do not
exist yet (e.g. Veins rows before the laptop runs) are listed with "exists": false and no hash.
"""

from __future__ import annotations

import hashlib
import json

from common.config import PROJECT_ROOT

R = PROJECT_ROOT / "results"
R8 = "results/revision8/"
RUNS5 = "results/revision5/runs/"
E = "experiments/revision8/"
SPEC = [
    (R8 + "probe_heads.json", E + "probe_heads.py", "python -m experiments.revision8.probe_heads", ["training-pool seeds 100-103"]),
    (R8 + "warm_start.json", E + "warm_start.py", "python -m experiments.revision8.warm_start",
     [RUNS5 + "classical__SEL__grid24__default__fallback.csv", RUNS5 + "rule__SEL__grid24__default__fallback.csv",
      "results/revision5/thresholds.json"]),
    (R8 + "veins_thresholds.json", E + "veins_thresholds.py", "python -m experiments.revision8.veins_thresholds",
     ["results/revision5/veins/veins_rows.jsonl (seeds 200-209)"]),
    (R8 + "train/", "agents/train_structured.py", "bash experiments/revision8/train_all.sh",
     ["results/coverage_targets.json", R8 + "warm_start.json", "results/revision5/analysis/h3.json (R5)",
      "training-pool traces 100-131"]),
    (R8 + "runs/struct_plain__SEL__headline__default__fallback.csv", E + "sweep_struct.py",
     "python -m experiments.revision8.sweep_struct --arm plain --seedset SEL", [R8 + "train/plain_s*/theta.json"]),
    (R8 + "runs/struct_plain__EVAL__headline__default__fallback.csv", E + "sweep_struct.py",
     "python -m experiments.revision8.sweep_struct --arm plain --seedset EVAL", [R8 + "train/plain_s*/theta.json"]),
    (R8 + "runs/struct_rand__SEL__headline__default__fallback.csv", E + "sweep_struct.py",
     "python -m experiments.revision8.sweep_struct --arm rand --seedset SEL", [R8 + "train/rand_s*/theta.json"]),
    (R8 + "runs/struct_rand__EVAL__headline__default__fallback.csv", E + "sweep_struct.py",
     "python -m experiments.revision8.sweep_struct --arm rand --seedset EVAL", [R8 + "train/rand_s*/theta.json"]),
    (R8 + "analysis/h4.json", E + "analysis/h4.py", None,
     [R8 + "runs/struct_plain__{SEL,EVAL}__headline__default__fallback.csv", RUNS5 + "classical__{SEL,EVAL}__grid24__default__fallback.csv",
      RUNS5 + "rule__{SEL,EVAL}__grid24__default__fallback.csv", "results/revision5/thresholds.json"]),
    (R8 + "analysis/h4_rand_descriptive.json", E + "analysis/h4.py", None,
     [R8 + "runs/struct_rand__{SEL,EVAL}__headline__default__fallback.csv"]),
    (R8 + "veins/port_validation_geometric.json", E + "veins/validate8.py", "python -m experiments.revision8.veins.validate8 geometric", []),
    (R8 + "veins/port_validation_unit_plain_s0.json", E + "veins/validate8.py",
     "python -m experiments.revision8.veins.validate8 unit --run plain_s0", [R8 + "train/plain_s0/theta.json"]),
    (R8 + "veins/port_validation_unit_rand_s0.json", E + "veins/validate8.py",
     "python -m experiments.revision8.veins.validate8 unit --run rand_s0", [R8 + "train/rand_s0/theta.json"]),
    (R8 + "veins/app_gate.json", E + "veins/queue8.py", "python -m experiments.revision8.veins.queue8 gate", ["laptop"]),
    (R8 + "veins/h5B_rows.jsonl", E + "veins/score8.py", "python -m experiments.revision8.veins.score8 h5B", ["laptop"]),
    (R8 + "veins/h5A_rows.jsonl", E + "veins/score8.py", "python -m experiments.revision8.veins.score8 h5A", ["laptop"]),
    (R8 + "veins/expC_rows.jsonl", E + "veins/score8.py", "python -m experiments.revision8.veins.score8 expC", ["laptop"]),
    (R8 + "veins/expD_rows.jsonl", E + "veins/score8.py", "python -m experiments.revision8.veins.score8 expD", ["laptop"]),
    (R8 + "analysis/h5.json", E + "analysis/h5.py", None, [R8 + "veins/h5{A,B}_rows.jsonl", R8 + "veins_thresholds.json"]),
    (R8 + "analysis/exp_cd.json", E + "analysis/h5.py", None, [R8 + "veins/exp{C,D}_rows.jsonl", R8 + "veins_thresholds.json"]),
    (R8 + "analysis/holm.json", E + "analysis/h5.py", None, [R8 + "analysis/h4.json", R8 + "analysis/h5.json"]),
]
MARK = "## Revision 8"


def digest(p) -> str | None:
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode() + b"\0" + f.read_bytes())
        return h.hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main(argv=None) -> int:
    man = json.loads((R / "manifest.json").read_text(encoding="utf-8"))
    keep = [o for o in man["outputs"] if not o["output"].startswith(R8)]
    new = []
    for name, script, cmd, src in SPEC:
        p = PROJECT_ROOT / name
        pv = {}
        if p.suffix == ".json" and p.exists():
            pv = json.loads(p.read_text(encoding="utf-8")).get("provenance", {}) or {}
        new.append({"output": name, "exists": p.exists(), "script": script, "command": cmd or pv.get("command"),
                    "commit": pv.get("git_commit"), "config_hash": pv.get("config_hash"), "generated": pv.get("generated"),
                    "sources": src, "sha256": digest(p)})
    man["outputs"] = keep + new
    (R / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = (R / "MANIFEST.md").read_text(encoding="utf-8").split(MARK)[0].rstrip() + "\n\n"
    lines = [MARK, "", "Generated by `python -m experiments.revision8.manifest`. Regenerate with `reproduce_revision8.ps1` (or `.sh`).",
             "Pre-registration: `results/revision8/ADDENDUM.md`. Outputs marked *(not yet)* are produced on the laptop",
             "(`results/revision8/LAPTOP_RUNBOOK.md`) or after training.", "",
             "| Output | Script / command | Commit | Config hash | Source rows |", "|---|---|---|---|---|"]
    for o in new:
        src = "<br>".join(f"`{s}`" for s in o["sources"]) or "–"
        out = f"`{o['output']}`" + ("" if o["exists"] else " *(not yet)*")
        lines.append(f"| {out} | `{o['command']}` | {(o['commit'] or '–')[:12]} | {o['config_hash'] or '–'} | {src} |")
    (R / "MANIFEST.md").write_text(text + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(R / "MANIFEST.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
