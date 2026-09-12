"""The headline comparator: regret against oracle-best, margin over fixed-best.

Why "vs flooding" is dead
-------------------------
Measured on rural at 20 veh/km/lane: reaching RWCR 0.95 costs flooding 2.11
transmissions per at-risk vehicle informed, and DV-CAST 0.41. Beating flooding
by 5x is therefore not a result -- it is a restatement of the fact that nobody
deploys flooding. Four baselines already cluster within 6% of each other at
0.41-0.43, and that cluster is the real bar.

Two reference points, and they answer different questions
---------------------------------------------------------
**(a) Per-cell oracle-best baseline.** For each cell of the factorial, the best
any baseline achieves *with its knob tuned for that cell specifically*. This is
not deployable -- it requires knowing the density, weather and topology in
advance and retuning per cell -- which is exactly why it is the right upper
bound. The agent's **regret** against it says: how much of the achievable
performance does a single learned policy recover, against a family of schemes
each hand-tuned with hindsight?

**(b) Best single fixed baseline.** One (policy, knob) setting, chosen once and
held constant across every cell. This *is* deployable, and it is what an
engineer would actually ship. The agent's **margin** over it says: is a learned
policy worth the complexity, compared with picking a good static scheme?

A learned policy that beats (b) but has large regret against (a) is adapting,
but not as well as per-cell tuning would. One that approaches (a) is
recovering the value of retuning without needing to retune. Reporting only one
of the two hides which is happening.

Both are computed at *matched quality*, so a policy can never look good by
transmitting more.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from analysis.pareto import OperatingPoint, PolicyCurve, pareto_front
from common.config import RESULTS_DIR, ensure_dir
from common.logging_utils import get_logger

logger = get_logger("analysis.comparator")

#: Cost axes. Both are reported: RWCR saturates in every scenario once measured
#: against the oracle at-risk set, so overhead and latency are co-primary.
COST_AXES: dict[str, str] = {
    "cost": "transmissions per at-risk vehicle informed",
    "tir_median_s": "median time-to-informed-at-risk (s)",
    "tir_p95_s": "p95 time-to-informed-at-risk (s)",
}


@dataclass(frozen=True)
class CellKey:
    scenario: str
    density: float
    weather: str
    hazard_type: str

    def __str__(self) -> str:
        return f"{self.scenario}/d={self.density:g}/{self.weather}/{self.hazard_type}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Cell:
    """One factorial cell and every policy's operating curve within it."""

    key: CellKey
    curves: dict[str, PolicyCurve]

    def point_cost(self, point: OperatingPoint, axis: str) -> float:
        return point.cost if axis == "cost" else point.extras.get(axis, float("nan"))

    def settings(self) -> list[tuple[str, str, Any]]:
        """Every (policy, param, value) setting present in this cell."""
        return [(p.policy, p.param, p.value)
                for c in self.curves.values() for p in c.points]

    def find(self, policy: str, param: str, value: Any) -> OperatingPoint | None:
        for p in self.curves.get(policy, PolicyCurve(policy, [])).points:
            if p.param == param and p.value == value:
                return p
        return None

    def achievable_quality(self, exclude: Iterable[str] = ()) -> float:
        """Best RWCR any (non-excluded) policy reaches in this cell."""
        excluded = set(exclude)
        best = -np.inf
        for name, curve in self.curves.items():
            if name in excluded:
                continue
            for pt in curve.points:
                if np.isfinite(pt.quality):
                    best = max(best, pt.quality)
        return float(best) if np.isfinite(best) else float("nan")

    def resolve_target(
        self, target: float, mode: str = "relative", exclude: Iterable[str] = ()
    ) -> float:
        """Turn a target specification into a concrete RWCR for this cell.

        ``absolute`` uses the number as given. ``relative`` treats it as a
        fraction of what is achievable *here*, which is the only workable
        choice across cells of very different difficulty. Measured ceilings
        range from RWCR 0.689 on rural at 2 veh/km/lane to 0.980 on
        urban_nlos at 20, so no single absolute threshold is both feasible
        in the sparse cell and demanding in the easy one: a fixed 0.95 is
        unreachable in three cells out of four and reports n/a everywhere.

        The ceiling excludes the policy being scored, so a stronger agent
        can never raise its own bar.
        """
        if mode == "absolute":
            return float(target)
        ceiling = self.achievable_quality(exclude)
        return float(target * ceiling) if np.isfinite(ceiling) else float("nan")

    def cost_at_matched(
        self, policy: str, target_quality: float, axis: str = "cost"
    ) -> float:
        """Cheapest cost on ``axis`` at which ``policy`` attains the target.

        For the transmission axis this is the interpolated Pareto crossing. For
        a latency axis, interpolation is not meaningful (latency is not
        monotone in the suppression knob), so the value of the cheapest
        qualifying setting is taken instead.
        """
        curve = self.curves.get(policy)
        if curve is None:
            return float("inf")
        if axis == "cost":
            return curve.overhead_at(target_quality)
        qualifying = [p for p in curve.points if p.quality >= target_quality]
        if not qualifying:
            return float("inf")
        best = min(qualifying, key=lambda p: p.cost)
        return self.point_cost(best, axis)


# ---------------------------------------------------------------------------
# (a) per-cell oracle-best
# ---------------------------------------------------------------------------
@dataclass
class OracleBest:
    """Best baseline per cell, with its knob tuned for that cell."""

    per_cell: dict[CellKey, tuple[str, float]] = field(default_factory=dict)
    axis: str = "cost"
    target: float = 0.95

    def cost(self, key: CellKey) -> float:
        return self.per_cell.get(key, ("", float("inf")))[1]

    def label(self, key: CellKey) -> str:
        return self.per_cell.get(key, ("none", float("inf")))[0]


def per_cell_oracle_best(
    cells: Sequence[Cell], target: float = 0.95, axis: str = "cost",
    exclude: Iterable[str] = (), mode: str = "relative",
) -> OracleBest:
    """The best (policy, knob) *within each cell*, tuned with hindsight."""
    excluded = set(exclude)
    out = OracleBest(axis=axis, target=target)
    for cell in cells:
        tq = cell.resolve_target(target, mode, excluded)
        best_label, best_cost = "none", float("inf")
        for policy in cell.curves:
            if policy in excluded:
                continue
            c = cell.cost_at_matched(policy, tq, axis)
            if np.isfinite(c) and c < best_cost:
                best_cost, best_label = c, policy
        out.per_cell[cell.key] = (best_label, best_cost)
    return out


# ---------------------------------------------------------------------------
# (b) best single fixed baseline
# ---------------------------------------------------------------------------
@dataclass
class FixedBest:
    """One setting held constant across every cell -- the deployable reference."""

    policy: str
    param: str
    value: Any
    per_cell_cost: dict[CellKey, float] = field(default_factory=dict)
    score: float = float("inf")
    axis: str = "cost"
    target: float = 0.95

    @property
    def label(self) -> str:
        return self.policy if not self.param else f"{self.policy}({self.param}={self.value})"

    def cost(self, key: CellKey) -> float:
        return self.per_cell_cost.get(key, float("inf"))


def best_fixed_baseline(
    cells: Sequence[Cell], target: float = 0.95, axis: str = "cost",
    exclude: Iterable[str] = (), penalty: float = 10.0, mode: str = "relative",
) -> FixedBest:
    """Pick the single setting that does best averaged over all cells.

    Scored on cost *normalised per cell* by that cell's oracle-best, so cells
    with intrinsically different cost scales contribute comparably. A setting
    that fails to reach the target in a cell is charged ``penalty`` times the
    oracle-best there rather than infinity -- otherwise one hard cell would
    eliminate every candidate and the comparison would have no reference at
    all. The number of cells a candidate fails in is reported alongside.
    """
    excluded = set(exclude)
    oracle = per_cell_oracle_best(cells, target, axis, exclude=excluded, mode=mode)

    candidates: set[tuple[str, str, Any]] = set()
    for cell in cells:
        for policy, param, value in cell.settings():
            if policy not in excluded:
                candidates.add((policy, param, value))

    best: FixedBest | None = None
    for policy, param, value in sorted(candidates, key=lambda c: (c[0], str(c[2]))):
        per_cell: dict[CellKey, float] = {}
        ratios: list[float] = []
        for cell in cells:
            pt = cell.find(policy, param, value)
            ref = oracle.cost(cell.key)
            tq = cell.resolve_target(target, mode, excluded)
            if pt is None or pt.quality < tq:
                per_cell[cell.key] = float("inf")
                ratios.append(penalty)
                continue
            c = cell.point_cost(pt, axis)
            per_cell[cell.key] = c
            ratios.append(c / ref if np.isfinite(ref) and ref > 0 else penalty)
        score = float(np.mean(ratios)) if ratios else float("inf")
        if best is None or score < best.score:
            best = FixedBest(policy=policy, param=param, value=value,
                             per_cell_cost=per_cell, score=score, axis=axis, target=target)
    assert best is not None, "no candidate settings"
    return best


# ---------------------------------------------------------------------------
# Scoring a candidate policy (the agent)
# ---------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    axis: str
    target: float
    regret_per_cell: dict[CellKey, float]
    margin_per_cell: dict[CellKey, float]
    oracle: OracleBest
    fixed: FixedBest
    agent_cost: dict[CellKey, float]

    @property
    def mean_regret(self) -> float:
        vals = [v for v in self.regret_per_cell.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_margin(self) -> float:
        vals = [v for v in self.margin_per_cell.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def cells_failed(self) -> int:
        return sum(1 for v in self.agent_cost.values() if not np.isfinite(v))


def compare_policy(
    cells: Sequence[Cell], policy: str, target: float = 0.95, axis: str = "cost",
    mode: str = "relative",
) -> ComparisonResult:
    """Score ``policy`` as regret vs oracle-best and margin over fixed-best.

    ``regret`` is the fractional excess cost over the per-cell oracle-best:
    0.0 means it matched hindsight tuning, 0.25 means it cost 25% more.
    ``margin`` is the fractional saving against the deployable fixed baseline:
    positive means the learned policy is cheaper.

    Both exclude ``policy`` itself from the reference sets, so an agent cannot
    become its own baseline.
    """
    oracle = per_cell_oracle_best(cells, target, axis, exclude=(policy,), mode=mode)
    fixed = best_fixed_baseline(cells, target, axis, exclude=(policy,), mode=mode)

    agent_cost, regret, margin = {}, {}, {}
    for cell in cells:
        tq = cell.resolve_target(target, mode, {policy})
        a = cell.cost_at_matched(policy, tq, axis)
        agent_cost[cell.key] = a
        o, f = oracle.cost(cell.key), fixed.cost(cell.key)
        regret[cell.key] = (a / o - 1.0) if np.isfinite(a) and np.isfinite(o) and o > 0 \
            else float("inf")
        margin[cell.key] = (f / a - 1.0) if np.isfinite(a) and np.isfinite(f) and a > 0 \
            else float("-inf") if np.isfinite(f) else float("nan")
    return ComparisonResult(axis=axis, target=target, regret_per_cell=regret,
                            margin_per_cell=margin, oracle=oracle, fixed=fixed,
                            agent_cost=agent_cost)


# ---------------------------------------------------------------------------
# Reporting and persistence
# ---------------------------------------------------------------------------
def format_reference_table(
    cells: Sequence[Cell], target: float = 0.95, axis: str = "cost",
    mode: str = "relative",
) -> str:
    """The two reference points per cell, before any agent exists."""
    oracle = per_cell_oracle_best(cells, target, axis, mode=mode)
    fixed = best_fixed_baseline(cells, target, axis, mode=mode)
    unit = COST_AXES.get(axis, axis)
    band = ("RWCR >= " + format(target, ".2f") if mode == "absolute"
            else "RWCR >= " + format(target, ".0%") + " of each cell ceiling")

    hdr = (f"{'cell':<38}{'ceil':>7}{'target':>8}"
           f"{'oracle-best':>24}{'fixed':>8}{'penalty':>9}")
    lines = ["=" * len(hdr),
             f" REFERENCE POINTS -- {unit}, at {band}",
             "=" * len(hdr),
             f" (b) best single fixed baseline across all cells: {fixed.label}",
             f"     mean normalised score {fixed.score:.3f} "
             f"({sum(1 for v in fixed.per_cell_cost.values() if not np.isfinite(v))} "
             f"cell(s) where it misses the target)",
             "-" * len(hdr), hdr, "-" * len(hdr)]
    for cell in cells:
        o_label, o_cost = oracle.per_cell[cell.key]
        f_cost = fixed.cost(cell.key)
        ceil = cell.achievable_quality()
        tq = cell.resolve_target(target, mode)
        pen = (f_cost / o_cost) if np.isfinite(f_cost) and np.isfinite(o_cost) and o_cost > 0 \
            else float("inf")
        lines.append(
            f"{str(cell.key):<38}{ceil:>7.3f}{tq:>8.3f}"
            f"{o_label + ' ' + _fmt(o_cost):>24}{_fmt(f_cost):>8}"
            f"{_fmt(pen, 'x'):>9}"
        )
    lines.append("=" * len(hdr))
    lines.append(" 'penalty' is what a deployable fixed choice costs against per-cell")
    lines.append(" hindsight tuning. It is the headroom a learned policy has to recover.")
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def _fmt(v: float, suffix: str = "") -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.2f}{suffix}"


def save_cells(cells: Sequence[Cell], path: Path | None = None) -> Path:
    """Persist curves so the agent can be scored later without re-running."""
    path = path or (RESULTS_DIR / "pareto_cells.json")
    ensure_dir(path.parent)
    payload = [
        {"key": c.key.as_dict(),
         "curves": {name: [asdict(p) for p in curve.points]
                    for name, curve in c.curves.items()}}
        for c in cells
    ]
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    logger.info("Saved %d cell(s) -> %s", len(cells), path)
    return path


def load_cells(path: Path | None = None) -> list[Cell]:
    path = path or (RESULTS_DIR / "pareto_cells.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for entry in payload:
        curves = {
            name: PolicyCurve(name, [OperatingPoint(**pt) for pt in pts])
            for name, pts in entry["curves"].items()
        }
        cells.append(Cell(key=CellKey(**entry["key"]), curves=curves))
    return cells


def main(argv: list[str] | None = None) -> int:
    """Run the multi-cell sweep and report the two reference points."""
    import argparse
    import logging

    from analysis.pareto import build_cells
    from common.config import load_yaml

    ap = argparse.ArgumentParser(description="AI-HARP headline comparator")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--mode", default="relative",
                    choices=["relative", "absolute"])
    ap.add_argument("--axis", default="cost", choices=list(COST_AXES))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="load results/pareto_cells.json instead of re-running")
    args = ap.parse_args(argv)
    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    if args.reuse:
        cells = load_cells()
    else:
        exp = load_yaml("experiment.yaml")
        specs = exp["comparator"]["cells"]
        cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
                "experiment": exp}
        cells = build_cells(specs, range(args.seeds), cfgs=cfgs)
        save_cells(cells)

    for axis in ([args.axis] if args.axis != "all" else list(COST_AXES)):
        print()
        print(format_reference_table(cells, args.target, axis, args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
