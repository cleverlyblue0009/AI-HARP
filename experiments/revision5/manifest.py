"""Revision-5 manifest: every output in results/revision5/analysis/ (and its tables and figures) -> source rows, script,
command, commit, config hash, SHA-256. Writes results/MANIFEST.md and results/manifest.json.

    python -m experiments.revision5.manifest

Command, commit and config hash are read from each JSON output's provenance stamp; tables and figures carry no stamp
and inherit the command of their script. Source rows are the committed files each script reads (paths repo-relative).
"""

from __future__ import annotations

import hashlib
import json

from common.config import PROJECT_ROOT

R = PROJECT_ROOT / "results"
AN = R / "revision5" / "analysis"
RUNS = "results/revision5/runs/"
VE = "results/revision5/veins/"
HIST = "results/revision1/checkpoints/*_history.jsonl"
DEF = [RUNS + f"{k}__{s}__grid24__default__fallback.csv" for k in ("classical", "rule") for s in ("SEL", "EVAL")]
AG = [RUNS + f"agent__{s}__grid24__default__fallback.csv" for s in ("SEL", "EVAL")]
AG10 = AG + [RUNS + f"agent__{s}__headline__default__fallback__s5_9.csv" for s in ("SEL", "EVAL")]
TAGS = ("ladderB", "ladderC", "variants", "tau0", "rgate", "budget")
SECONDARY = (DEF + AG10 + ["results/revision5/thresholds.json", "results/revision5/thresholds_stress.json"]
             + [RUNS + f"agent__{s}__headline__default__fallback__{t}.csv" for t in TAGS for s in ("SEL", "EVAL")]
             + [RUNS + f"{k}__{s}__grid24__calibrated__fallback.csv" for k in ("classical", "rule", "agent")
                for s in ("SEL", "EVAL")]
             + [RUNS + f"{k}__{s}__headline__default__sumo.csv" for k in ("classical", "rule", "agent") for s in ("SEL", "EVAL")])
A = "experiments/revision5/analysis/"
SPEC = {
    "h1.json": (A + "h1.py", [RUNS + f"classical__{s}__grid24__default__fallback.csv" for s in ("CAL", "SEL", "EVAL")]),
    "h1_Dr.json": (A + "h1.py", [RUNS + f"classical__{s}__grid24__default__fallback.csv" for s in ("CAL", "SEL", "EVAL")]),
    "h2.json": (A + "h2.py", DEF + AG10 + ["results/revision5/thresholds.json"]),
    "h2_Dr.json": (A + "h2.py", DEF + AG10 + ["results/revision5/thresholds.json"]),
    "h2_grid24.json": (A + "h2.py", DEF + AG + ["results/revision5/thresholds.json"]),
    "h3.json": (A + "h3.py", [VE + "veins_rows.jsonl", VE + "custom_h3.jsonl", RUNS + "classical__EVAL__headline__default__sumo.csv"]),
    "h3_subepoch.json": (A + "h3.py", [VE + "veins_rows.jsonl", VE + "custom_h3_subepoch.jsonl", "configs/phy_patch_d6_frozen.json"]),
    "d6_calibration.json": ("experiments/revision5/veins/d6_calibrate.py", [VE + "veins_rows.jsonl", VE + "d6_calibration/*.jsonl"]),
    "holm.json": (A + "holm.py", ["results/revision5/analysis/" + f for f in ("h1.json", "h2.json", "h3.json", "h3_subepoch.json")]),
    "secondary.json": (A + "secondary.py", SECONDARY),
    "training_diagnostics.json": (A + "training_diag.py", [HIST]),
    "compute_curves.json": (A + "compute_curves.py", DEF[:1] + DEF[1:2] + [HIST, "results/revision5/analysis/secondary.json"]),
    "ledger.json": (A + "ledger.py", ["results/revision5/analysis/" + f for f in
                                       ("h1.json", "h2.json", "h2_grid24.json", "h3.json", "h3_subepoch.json", "holm.json",
                                        "secondary.json")] + [RUNS + f"classical__{s}__grid24__calibrated__fallback.csv"
                                                              for s in ("CAL", "SEL", "EVAL")]
                    + [RUNS + f"classical__{s}__headline__default__sumo.csv" for s in ("SEL", "EVAL")]),
}
TABLE_SRC = {"T-heldout": ["secondary.json", "h2.json"], "T-veins": ["h3.json", "h3_subepoch.json"],
             "T-ablation": ["secondary.json"], "T-ladder": ["secondary.json"], "T-gate": ["secondary.json"],
             "T-interp": ["secondary.json"], "T-sens": ["secondary.json"]}
FIG_SRC = {"fig_veins_pdr": ["h3.json", "h3_subepoch.json"],
           "fig_veins_delay": [VE + "veins_rows.jsonl", VE + "custom_h3.jsonl", VE + "custom_h3_subepoch.jsonl"],
           "fig_compute": ["compute_curves.json"], "fig_training": ["training_diagnostics.json", HIST],
           "fig_heldout": [RUNS + "classical__EVAL__grid24__default__fallback.csv", "secondary.json"]}


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _src(lst):
    return [s if "/" in s else "results/revision5/analysis/" + s for s in lst]


def main(argv=None) -> int:
    out = []
    for name, (script, src) in SPEC.items():
        p = AN / name
        pv = json.loads(p.read_text(encoding="utf-8")).get("provenance", {}) if p.exists() else {}
        out.append({"output": f"results/revision5/analysis/{name}", "exists": p.exists(), "script": script,
                    "command": pv.get("command"), "commit": pv.get("git_commit"), "config_hash": pv.get("config_hash"),
                    "generated": pv.get("generated"), "sources": src, "sha256": sha(p) if p.exists() else None})
    for kind, d, ext, spec, script, cmd in (
            ("table", AN / "tables", "tex", TABLE_SRC, A + "tables.py", "python -m experiments.revision5.analysis.tables"),
            ("figure", R / "revision5" / "figures", "pdf", FIG_SRC, A + "figures.py",
             "python -m experiments.revision5.analysis.figures")):
        for n, src in spec.items():
            p = d / f"{n}.{ext}"
            out.append({"output": p.relative_to(PROJECT_ROOT).as_posix(), "exists": p.exists(), "script": script,
                        "command": cmd, "commit": None, "config_hash": None, "generated": None, "sources": _src(src),
                        "sha256": sha(p) if p.exists() else None})
    (R / "manifest.json").write_text(json.dumps({"description": __doc__.splitlines()[0], "outputs": out}, indent=1) + "\n",
                                     encoding="utf-8", newline="\n")
    lines = ["# Revision-5 results manifest", "",
             "Generated by `python -m experiments.revision5.manifest` (machine-readable copy: `results/manifest.json`).",
             "Every output in `results/revision5/analysis/` and its tables and figures, with the committed rows it is",
             "computed from, the script and command that wrote it, and the commit and config hash from its provenance",
             "stamp. Regenerate everything with `reproduce_revision5.ps1` (or `.sh`); `--from-scratch` reruns the",
             "simulations first. Tables and figures have no provenance stamp of their own.", "",
             "| Output | Script / command | Commit | Config hash | Source rows |", "|---|---|---|---|---|"]
    for o in out:
        src = "<br>".join(f"`{s}`" for s in o["sources"])
        lines.append(f"| `{o['output']}` | `{o['command']}` | {(o['commit'] or '–')[:12]} | {o['config_hash'] or '–'} | {src} |")
    lines += ["", "SHA-256 of each output at manifest time: `results/manifest.json` (field `sha256`).", ""]
    (R / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(R / "MANIFEST.md")
    print(R / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
