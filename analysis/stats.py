"""Phase 7: paired significance testing with effect sizes.

Wilcoxon signed-rank, because the samples are paired by construction (seed *k*
gives every policy identical mobility, hazard placement, shadowing and fading)
and the per-cell metric distributions are not remotely normal -- RWCR is
bounded and often piles up near its ceiling, and transmission counts are
heavy-tailed.

Effect sizes are reported alongside every p-value, and this is not a formality.
With 10 paired seeds and a deterministic simulator, tiny differences reach
significance easily: a 1% improvement that is "highly significant" is still a
1% improvement. The rank-biserial correlation says how *consistently* one
policy beats another, and the median paired difference says by how much, in the
metric's own units. Report all three or none.

Multiple comparisons are corrected with Holm-Bonferroni across the metric
family within a cell. Holm rather than plain Bonferroni because it is uniformly
more powerful at the same family-wise error rate, and uniformly more powerful
is free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.metrics import METRIC_DIRECTION
from common.logging_utils import get_logger

logger = get_logger("analysis.stats")

#: Below this many paired samples, a two-sided Wilcoxon test cannot reach
#: p < 0.05 at all (2^-n > 0.025 for n < 6), so reporting one is meaningless.
MIN_PAIRS_FOR_WILCOXON = 6


@dataclass
class PairedTest:
    """One paired comparison of a candidate against a reference."""

    metric: str
    n_pairs: int
    median_difference: float
    mean_difference: float
    rank_biserial: float
    p_value: float
    p_adjusted: float = float("nan")
    significant: bool = False
    direction: int = 1
    note: str = ""

    @property
    def candidate_better(self) -> bool:
        """Is the candidate better, in this metric's own preferred direction?"""
        return self.median_difference * self.direction > 0

    @property
    def marker(self) -> str:
        """Significance marker for a LaTeX table."""
        if not np.isfinite(self.p_adjusted):
            return ""
        if self.p_adjusted < 0.001:
            return "***"
        if self.p_adjusted < 0.01:
            return "**"
        if self.p_adjusted < 0.05:
            return "*"
        return ""

    def describe(self) -> str:
        return (
            f"{self.metric}: n={self.n_pairs} median_diff={self.median_difference:+.4f} "
            f"rrb={self.rank_biserial:+.3f} p={self.p_value:.4g} "
            f"p_adj={self.p_adjusted:.4g}{self.marker} {self.note}".strip()
        )


def rank_biserial_correlation(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: the Wilcoxon effect size.

    ``(W+ - W-) / (W+ + W-)`` over the signed ranks of the non-zero
    differences. +1 means the candidate wins every pair, -1 loses every pair,
    0 is an even split. Unlike the p-value it does not grow with sample size.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return 0.0
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total > 0 else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (what the signed-rank statistic requires)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average over tied groups.
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def paired_test(
    candidate: Sequence[float], reference: Sequence[float], metric: str
) -> PairedTest:
    """Wilcoxon signed-rank of ``candidate`` against ``reference``.

    Pairs where either side is non-finite are dropped, and the count of
    surviving pairs is reported: a metric that is undefined in half the seeds
    (TIR when nothing is informed, say) must not silently look like a clean
    comparison.
    """
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must be the same length: {a.shape} vs {b.shape}")

    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    direction = METRIC_DIRECTION.get(metric, 1)

    result = PairedTest(
        metric=metric, n_pairs=int(d.size),
        median_difference=float(np.median(d)) if d.size else float("nan"),
        mean_difference=float(d.mean()) if d.size else float("nan"),
        rank_biserial=rank_biserial_correlation(d),
        p_value=float("nan"), direction=direction,
    )
    if d.size < MIN_PAIRS_FOR_WILCOXON:
        result.note = f"too few usable pairs (n={d.size}); no p-value reported"
        return result
    if np.allclose(d, 0):
        result.p_value = 1.0
        result.note = "identical in every pair"
        return result

    try:
        from scipy.stats import wilcoxon

        result.p_value = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception as exc:  # pragma: no cover
        result.note = f"wilcoxon failed: {exc}"
    return result


def holm_bonferroni(tests: Sequence[PairedTest], alpha: float = 0.05) -> list[PairedTest]:
    """Holm-Bonferroni step-down correction over a family of tests.

    Sorts ascending by p, compares the k-th to ``alpha / (m - k)``, and stops
    at the first failure -- every test after it is non-significant regardless
    of its own p-value. Adjusted p-values are made monotone so they can be
    reported directly.
    """
    usable = [t for t in tests if np.isfinite(t.p_value)]
    m = len(usable)
    if m == 0:
        return list(tests)

    order = sorted(range(m), key=lambda i: usable[i].p_value)
    running = 0.0
    still_significant = True
    for k, i in enumerate(order):
        adj = min(1.0, usable[i].p_value * (m - k))
        running = max(running, adj)          # enforce monotonicity
        usable[i].p_adjusted = running
        if still_significant and running > alpha:
            still_significant = False
        usable[i].significant = still_significant and running <= alpha
    return list(tests)


def compare_against_reference(
    candidate_rows: dict[str, Sequence[float]],
    reference_rows: dict[str, Sequence[float]],
    metrics: Sequence[str],
    alpha: float = 0.05,
) -> list[PairedTest]:
    """Test a candidate against a reference across a family of metrics."""
    tests = [
        paired_test(candidate_rows.get(m, []), reference_rows.get(m, []), m)
        for m in metrics
    ]
    return holm_bonferroni(tests, alpha)


def strongest_baseline(
    per_policy: dict[str, Sequence[float]], metric: str
) -> tuple[str, float]:
    """The best baseline for one metric, in that metric's preferred direction.

    Phase 7 tests the agent against *this* rather than against a fixed choice,
    so the comparison is never flattered by picking a weak opponent.
    """
    direction = METRIC_DIRECTION.get(metric, 1)
    best_name, best_val = "", -np.inf
    for name, vals in per_policy.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        score = float(arr.mean()) * direction
        if score > best_val:
            best_name, best_val = name, score
    return best_name, best_val * direction


def format_tests(tests: Sequence[PairedTest], title: str) -> str:
    hdr = (f"{'metric':<30}{'n':>4}{'median diff':>14}{'r_rb':>8}"
           f"{'p':>11}{'p_holm':>11}{'sig':>6}")
    lines = ["=" * len(hdr), f" {title}", "=" * len(hdr), hdr, "-" * len(hdr)]
    for t in tests:
        p = "n/a" if not np.isfinite(t.p_value) else f"{t.p_value:.4g}"
        pa = "n/a" if not np.isfinite(t.p_adjusted) else f"{t.p_adjusted:.4g}"
        lines.append(
            f"{t.metric:<30}{t.n_pairs:>4}{t.median_difference:>+14.4f}"
            f"{t.rank_biserial:>+8.3f}{p:>11}{pa:>11}{t.marker:>6}"
        )
        if t.note:
            lines.append(f"{'':<30}{t.note}")
    lines.append("=" * len(hdr))
    lines.append(" r_rb is the matched-pairs rank-biserial correlation: +1 means the")
    lines.append(" candidate wins every seed. Unlike p, it does not grow with n --")
    lines.append(" a significant 1% improvement is still a 1% improvement.")
    lines.append("=" * len(hdr))
    return "\n".join(lines)
