"""Phase 7b: does any conclusion flip between mobility backends?

The single biggest reviewer risk in this project is that the network layer is
custom Python and the mobility was, until now, a pure-Python fallback rather
than SUMO. This runs the same cells under both backends, paired by seed, and
reports where they agree and where they do not.

The honest outcome is not "they agree". It is whichever of these is true:

* the *ordering* of policies is preserved -> the comparative claims hold, and
  the paper reports the absolute numbers from SUMO;
* the ordering changes -> the paper reports both backends and says so.

Ordering is what matters, because every claim in the paper is comparative
(regret against oracle-best, margin over fixed-best). Absolute RWCR differing
between backends is expected: the fallback pre-places vehicles and has no lane
changing, so its traffic is more platooned.

::

    SUMO_HOME=D:/sumo-1.19.0 D:/aiharp-env/python.exe -m experiments.backend_validation
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from common.config import load_yaml
from common.logging_utils import get_logger
from experiments.run_sim import RunSpec, run_single

logger = get_logger("experiments.backend_validation")

DEFAULT_POLICIES = ("flooding", "slotted_1p", "greedy_farthest", "dvcast")
DEFAULT_METRICS = ("rwcr", "tir_median_s", "tx_per_at_risk_informed", "pdr")


@dataclass
class BackendComparison:
    metric: str
    fallback: dict[str, float]
    sumo: dict[str, float]
    #: Per-policy seed samples, kept so "did the ordering change" can be
    #: answered against the noise rather than against the point estimates.
    fallback_samples: dict[str, list[float]] = None
    sumo_samples: dict[str, list[float]] = None

    def spread(self, which: str) -> float:
        d = self.fallback if which == "fallback" else self.sumo
        v = [x for x in d.values() if np.isfinite(x)]
        return float(max(v) - min(v)) if len(v) > 1 else float("nan")

    def typical_std(self, which: str) -> float:
        samples = self.fallback_samples if which == "fallback" else self.sumo_samples
        if not samples:
            return float("nan")
        stds = [float(np.nanstd(v)) for v in samples.values() if len(v) > 1]
        return float(np.nanmean(stds)) if stds else float("nan")

    def moved(self) -> list[str]:
        """Policies whose rank differs between the two backends."""
        a, b = self.ranking("fallback"), self.ranking("sumo")
        return [p for p in a if p in b and a.index(p) != b.index(p)]

    def _spread_and_std(self, which: str, policies: Sequence[str]) -> tuple[float, float]:
        d = self.fallback if which == "fallback" else self.sumo
        samples = self.fallback_samples if which == "fallback" else self.sumo_samples
        vals = [d[p] for p in policies if np.isfinite(d.get(p, np.nan))]
        spread = float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        stds = [float(np.nanstd(samples[p])) for p in policies
                if samples and p in samples and len(samples[p]) > 1]
        return spread, (float(np.nanmean(stds)) if stds else float("nan"))

    @property
    def separable(self) -> bool:
        """Is the rank change a real flip rather than noise?

        Judged only on the policies that MOVED, and only if their gap exceeds
        the seed-to-seed std under BOTH backends. The first version used the
        spread across all policies, so one outlier that did not move (flooding,
        ~2.3 on cost against ~0.6 for the rest) made a reshuffle among three
        near-identical schemes look separable. Requiring both backends
        distinguishes a genuine reversal from "ordered in one, tied in the
        other".
        """
        movers = self.moved()
        if len(movers) < 2:
            return False
        for which in ("fallback", "sumo"):
            sp, sd = self._spread_and_std(which, movers)
            if not (np.isfinite(sp) and np.isfinite(sd) and sp > sd):
                return False
        return True

    @property
    def verdict(self) -> str:
        if self.ordering_preserved:
            return "PRESERVED"
        return "CHANGED (real)" if self.separable else "changed within noise"

    def ranking(self, which: str) -> list[str]:
        from analysis.metrics import METRIC_DIRECTION

        d = self.fallback if which == "fallback" else self.sumo
        usable = {k: v for k, v in d.items() if np.isfinite(v)}
        return sorted(usable, key=lambda k: -METRIC_DIRECTION.get(self.metric, 1) * usable[k])

    @property
    def ordering_preserved(self) -> bool:
        return self.ranking("fallback") == self.ranking("sumo")

    def spearman(self) -> float:
        a, b = self.ranking("fallback"), self.ranking("sumo")
        common = [p for p in a if p in b]
        if len(common) < 3:
            return float("nan")
        ra = np.array([a.index(p) for p in common], dtype=float)
        rb = np.array([b.index(p) for p in common], dtype=float)
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])


def run_backend_cells(
    scenario: str, density: float, seeds: Sequence[int],
    policies: Sequence[str] = DEFAULT_POLICIES,
    metrics: Sequence[str] = DEFAULT_METRICS,
    duration_s: float | None = None, corridor_length_m: float | None = None,
) -> list[BackendComparison]:
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    results: dict[str, dict[str, list[float]]] = {
        b: {p: [] for p in policies} for b in ("fallback", "sumo")
    }

    for backend in ("fallback", "sumo"):
        for policy in policies:
            for seed in seeds:
                spec = RunSpec(
                    scenario=scenario, density_veh_km_lane=density, policy=policy,
                    seed=seed, backend=backend, duration_s=duration_s,
                    corridor_length_m=corridor_length_m,
                )
                m, _ = run_single(spec, phy_cfg=cfgs["phy"], hz_cfg=cfgs["hazard"],
                                  exp_cfg=cfgs["experiment"])
                results[backend][policy].append(m)

    out = []
    for metric in metrics:
        fb_s = {p: [float(r.get(metric, np.nan)) for r in results["fallback"][p]]
                for p in policies}
        su_s = {p: [float(r.get(metric, np.nan)) for r in results["sumo"][p]]
                for p in policies}
        fb = {p: float(np.nanmean(v)) for p, v in fb_s.items()}
        su = {p: float(np.nanmean(v)) for p, v in su_s.items()}
        out.append(BackendComparison(metric=metric, fallback=fb, sumo=su,
                                     fallback_samples=fb_s, sumo_samples=su_s))
    return out


def format_comparison(comps: Sequence[BackendComparison], title: str) -> str:
    hdr = f"{'metric':<26}{'policy':<18}{'fallback':>12}{'sumo':>12}{'delta':>12}"
    lines = ["=" * len(hdr), f" BACKEND VALIDATION -- {title}", "=" * len(hdr),
             hdr, "-" * len(hdr)]
    for c in comps:
        for i, pol in enumerate(c.fallback):
            f, s = c.fallback[pol], c.sumo.get(pol, float("nan"))
            d = s - f
            lines.append(
                f"{(c.metric if i == 0 else ''):<26}{pol:<18}"
                f"{f:>12.4f}{s:>12.4f}{d:>+12.4f}"
            )
        lines.append("-" * len(hdr))

    lines += ["", "=" * len(hdr), " ORDERING (what the comparative claims depend on)",
              "=" * len(hdr)]
    for c in comps:
        lines.append(
            f"  {c.metric:<28}{c.verdict:<24}spearman={c.spearman():+.3f}  "
            f"spread={c.spread('fallback'):.4f} vs seed-std={c.typical_std('fallback'):.4f}"
        )
        if not c.ordering_preserved:
            lines.append(f"      fallback: {' > '.join(c.ranking('fallback'))}")
            lines.append(f"      sumo    : {' > '.join(c.ranking('sumo'))}")
    lines.append("=" * len(hdr))

    real = [c.metric for c in comps if not c.ordering_preserved and c.separable]
    noise = [c.metric for c in comps if not c.ordering_preserved and not c.separable]
    if real:
        lines.append(f" {len(real)} metric(s) flipped with policies separable: {real}.")
        lines.append(" The paper MUST report both backends for these.")
    if noise:
        lines.append(f" {len(noise)} metric(s) reshuffled WITHIN NOISE: {noise}.")
        lines.append(" The policies are not separable on these at this cell, so the rank")
        lines.append(" change is not a conclusion flipping -- but it does mean the metric")
        lines.append(" cannot support a ranking claim here either, under either backend.")
    if not real and not noise:
        lines.append(" No ordering changed. Comparative claims hold under both backends;")
        lines.append(" report absolute numbers from SUMO.")
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fallback vs SUMO backend validation")
    ap.add_argument("--scenario", default="rural_highway")
    ap.add_argument("--density", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    from mobility.sumo_runner import find_sumo

    if find_sumo() is None:
        print("No SUMO installation found. Set SUMO_HOME (see ENVIRONMENT.md).")
        return 1

    comps = run_backend_cells(
        args.scenario, args.density, list(range(args.seeds)),
        duration_s=args.duration, corridor_length_m=args.length,
    )
    print(format_comparison(
        comps, f"{args.scenario} @ {args.density:g} veh/km/lane, {args.seeds} seeds"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
