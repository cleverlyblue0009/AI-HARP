"""R5: reproducibility table -- commit, config hash and checkpoint SHA for every table row.

::

    python -m experiments.revision2.r5_repro_table

One row per row of every table the paper draws from results/: Table 2 / per-cell main tables
(results/tables/table_main*.tex <- results/runs.csv), the reference table
(table_reference.tex <- pareto_cells.json), the ablation table (table_ablation.tex <-
results/agent_<variant>/sampled/agent_evaluation.json), Table 3 (risk_estimation.csv),
Table 5 (feasibility/lambda_cap_arms.json), Table 6 (validation/amador2022.json), and the
revision-1/2 ablation-over-seeds table (C1C2/table_ablation_seeds.tex).

Columns: source file; ``source_commit`` = the last commit that changed that file;
``config_hash`` = for runs.csv rows, a hash of the sorted per-run config hashes behind the
row (with the run count), otherwise the hash of the configuration where one exists (training
checkpoints) or ``not recorded`` with ``content_sha256`` of the exact source rows instead;
``checkpoint_sha`` = experiments.evaluate_agent.checkpoint_sha of every checkpoint behind the
row; ``metrics_version``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from common.config import PROJECT_ROOT, config_hash
from experiments.revision1.provenance import out_dir, stamp, write_json

R = PROJECT_ROOT / "results"


def commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path.relative_to(PROJECT_ROOT))],
                                       cwd=PROJECT_ROOT, text=True).strip() or "uncommitted"
    except Exception:
        return "unknown"


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def ckpt_info(path: Path) -> tuple[str, str]:
    import torch

    from experiments.evaluate_agent import checkpoint_sha

    if not path.exists():
        return "missing", "missing"
    st = torch.load(str(path), map_location="cpu", weights_only=False)
    return checkpoint_sha(path), config_hash(st["config"])


def rows_main(runs: pd.DataFrame, rows: list) -> None:
    src = R / "runs.csv"
    c = commit(src)
    for (sc, d), sub in runs.groupby(["scenario", "density_veh_km_lane"]):
        table = "table_main.tex (Table 2)" if (sc, d) == ("rural_highway", 20.0) else f"table_main_{sc}_d{d:g}.tex"
        for (pol, tau), g in sub.fillna({"tau": -1}).groupby(["policy", "tau"]):
            label = pol if tau < 0 else f"{pol} (tau={tau:g})"
            shas = sorted(set(g["checkpoint_sha"].dropna().astype(str))) if "checkpoint_sha" in g else []
            rows.append({"table": table, "row": label, "values": f"RWCR {g.rwcr.mean():.3f}; cost {g.tx_per_at_risk_informed.mean():.2f}",
                         "source_file": "results/runs.csv", "source_commit": c,
                         "config_hash": f"set:{sha(sorted(g.config_hash.astype(str)))} ({len(g)} runs)",
                         "content_sha256": "", "checkpoint_sha": ";".join(s for s in shas if s and s != "nan") or "-",
                         "metrics_version": ",".join(map(str, sorted(g.metrics_version.unique())))})


def main() -> int:
    rows: list[dict] = []
    runs = pd.read_csv(R / "runs.csv", low_memory=False)
    rows_main(runs, rows)
    # reference table <- pareto_cells.json
    pc = json.loads((R / "pareto_cells.json").read_text())
    for cell in pc:
        k = cell["key"]
        rows.append({"table": "table_reference.tex", "row": f"{k['scenario']} d={k['density']:g}",
                     "values": "oracle-best / fixed-best / ceiling", "source_file": "results/pareto_cells.json",
                     "source_commit": commit(R / "pareto_cells.json"), "config_hash": "not recorded (seed means only)",
                     "content_sha256": sha(cell), "checkpoint_sha": "-", "metrics_version": "3"})
    # ablation table <- agent_evaluation.json (committed; seed 0)
    for v in ("ref", "gcn", "mlp", "star"):
        p = R / f"agent_{v}" / "sampled" / "agent_evaluation.json"
        e = json.loads(p.read_text())
        ck = PROJECT_ROOT / "checkpoints" / "campaign" / v / "ckpt_final.pt"
        s, ch = ckpt_info(ck)
        rows.append({"table": "table_ablation.tex", "row": v, "values": f"regret {e['axes']['cost']['mean_regret']:.3f}; margin {e['axes']['cost']['mean_margin']:.3f}",
                     "source_file": f"results/agent_{v}/sampled/agent_evaluation.json", "source_commit": commit(p),
                     "config_hash": ch, "content_sha256": sha(e["axes"]["cost"]),
                     "checkpoint_sha": e["checkpoint_sha"] + ("" if s == e["checkpoint_sha"] else f" (file now {s})"),
                     "metrics_version": "3"})
    # Table 3 <- risk_estimation.csv
    re_ = pd.read_csv(R / "risk_estimation.csv")
    for sc, g in re_.groupby("scenario"):
        rows.append({"table": "Table 3 (risk estimation)", "row": sc, "values": f"peak corr {g.risk_peak_corr.mean():.3f}",
                     "source_file": "results/risk_estimation.csv", "source_commit": commit(R / "risk_estimation.csv"),
                     "config_hash": "not recorded in the CSV", "content_sha256": sha(g.to_dict("list")),
                     "checkpoint_sha": "- (policy-independent)", "metrics_version": "3"})
    # Table 5 <- lambda_cap_arms.json (training runs)
    lam = R / "feasibility" / "lambda_cap_arms.json"
    for name, d in (("run8 (cap 50)", "run8"), ("arm A (cap 500)", "feas_A_cap500"), ("arm B (init 500)", "feas_B_init500")):
        s, ch = ckpt_info(PROJECT_ROOT / "checkpoints" / d / "ckpt_final.pt")
        rows.append({"table": "Table 5 (lambda arms)", "row": name, "values": "per-group shortfall / lambda",
                     "source_file": "results/feasibility/lambda_cap_arms.json", "source_commit": commit(lam),
                     "config_hash": ch, "content_sha256": sha(json.loads(lam.read_text())),
                     "checkpoint_sha": s, "metrics_version": "-"})
    # Table 6 <- amador2022.json
    am = R / "validation" / "amador2022.json"
    for dens, row in json.loads(am.read_text())["per_density"].items():
        rows.append({"table": "Table 6 (validation)", "row": f"{dens} veh/km/lane",
                     "values": f"PDR {row['pdr_mean']:.4f} vs paper {row['paper']}", "source_file": "results/validation/amador2022.json",
                     "source_commit": commit(am), "config_hash": "not recorded", "content_sha256": sha(row),
                     "checkpoint_sha": "- (CBF baseline)", "metrics_version": "-"})
    # revision tables: ablation over training seeds
    shas = json.loads((R / "revision1" / "checkpoints" / "SHA.json").read_text())["sha"]
    for v in ("ref", "gcn", "mlp", "star", "norel"):
        ck = sorted(f"{k.split('_ckpt')[0]}={s}" for k, s in shas.items() if k.startswith(f"{v}_s"))
        cfgs = []
        for f in sorted((R / "revision1" / "C_train").glob(f"{v}_s*.json")):
            cfgs.append(f"{f.stem}={json.loads(f.read_text()).get('config_hash')}")
        rows.append({"table": "C1C2/table_ablation_seeds.tex", "row": v, "values": "see table",
                     "source_file": "results/revision1/C1C2/c_seeds.json", "source_commit": commit(R / "revision1" / "C1C2" / "c_seeds.json"),
                     "config_hash": ";".join(cfgs) or "seed 0: campaign run (config in checkpoint)",
                     "content_sha256": "", "checkpoint_sha": ";".join(ck), "metrics_version": "3"})
    d = out_dir("R5", 2)
    with (d / "r5_repro_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")  # noqa: E731
    tex = [r"% Reproducibility table (revision-2 R5). Generated by experiments/revision2/r5_repro_table.py.",
           r"\begin{tabular}{llllll}", r"\toprule",
           r"Table & Row & Source & Commit & Config hash & Checkpoint SHA \\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join(esc(x) for x in (r["table"], r["row"], r["source_file"].replace("results/", ""),
                                               r["source_commit"][:10], (r["config_hash"] or r["content_sha256"])[:40],
                                               r["checkpoint_sha"][:60])) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (d / "r5_repro_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    write_json(d / "r5_meta.json", stamp("R5", {"n_rows": len(rows), "tables": sorted({r["table"] for r in rows})}))
    print(len(rows), "rows;", sorted({r["table"] for r in rows})[:6], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
