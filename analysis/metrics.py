"""Phase 6 metrics.

Two of these are proposed by the paper (RWCR, TIR) and are therefore defined
here in full, with their equations, rather than being left to prose.

Every metric is computed from one :class:`~sim.engine.RunResult`. Nothing here
aggregates across seeds -- that is Phase 7's job, and keeping the split sharp
is what stops a single-run number from ever reaching a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hazard.model import Hazard
from hazard.risk_field import RiskField
from sim.engine import NO_STEP, RunResult

#: +1 = higher is better, -1 = lower is better. Used for significance markers
#: and for picking the "strongest baseline" in Phase 7.
METRIC_DIRECTION: dict[str, int] = {
    "pdr": +1,
    "rwcr": +1,
    "coverage": +1,
    "at_risk_coverage": +1,
    "tir_median_s": -1,
    "tir_p95_s": -1,
    "tir_uninformed_frac": -1,
    "latency_mean_s": -1,
    "latency_p95_s": -1,
    "redundancy_ratio": -1,
    "transmissions": -1,
    "collisions_per_delivered": -1,
    "deadline_miss_rate": -1,
    "max_hops": +1,
    "spatial_reach_m": +1,
}


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


@dataclass
class MetricContext:
    """Precomputed quantities shared by several metrics."""

    relevance: np.ndarray        # [T, N] relevance over the whole run
    peak_relevance: np.ndarray   # [N]
    at_risk: np.ndarray          # [N] bool
    informed_in_time: np.ndarray # [N] bool
    latency_s: np.ndarray        # [N] seconds from hazard onset, inf if never


def _build_context(res: RunResult, risk: RiskField, hazard: Hazard) -> MetricContext:
    tr = res.trace
    rel = risk.relevance_matrix(tr, hazard)
    peak, at_risk = risk.at_risk_set(rel)

    informed = res.informed_step >= 0
    idx = np.flatnonzero(informed)
    # A warning only counts if it arrives while the vehicle is still at risk:
    # relevance is exactly zero once a vehicle has passed the hazard span or is
    # travelling away from it.
    rel_at_info = np.zeros(tr.n_vehicles)
    rel_at_info[idx] = rel[res.informed_step[idx], idx]
    informed_in_time = informed & (rel_at_info > 0.0)

    latency = np.full(tr.n_vehicles, np.inf)
    latency[idx] = res.informed_step[idx] * tr.dt - hazard.onset_time_s
    latency = np.maximum(latency, 0.0)

    return MetricContext(
        relevance=rel, peak_relevance=peak, at_risk=at_risk,
        informed_in_time=informed_in_time, latency_s=latency,
    )


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------
def packet_delivery_ratio(res: RunResult) -> float:
    """PDR for unacknowledged broadcast.

    .. math:: \\mathrm{PDR} = \\frac{\\sum_{\\text{tx}} |\\{r : \\text{decoded}\\}|}
                                   {\\sum_{\\text{tx}} |\\{r : d(tx,r) \\le R_{nom}\\}|}

    The denominator is every active vehicle inside the *nominal* communication
    range (the median link budget at receiver sensitivity) at the moment of
    transmission, excluding the sender. Counting only in-range receivers is
    what makes PDR a channel-quality measure rather than a re-statement of
    network density.
    """
    return res.n_rx_success / res.n_rx_attempts if res.n_rx_attempts else float("nan")


def risk_weighted_coverage_ratio(ctx: MetricContext) -> float:
    """**RWCR** -- the paper's headline metric.

    .. math::

        \\mathrm{RWCR} \\;=\\;
        \\frac{\\sum_{v \\in \\mathcal{A}} \\rho^{*}(v)\\,
               \\mathbb{1}[\\,v \\text{ informed while still at risk}\\,]}
             {\\sum_{v \\in \\mathcal{A}} \\rho^{*}(v)}

    where :math:`\\rho^{*}(v) = \\max_t \\mathrm{rel}(v,h,t)` is the vehicle's
    peak relevance over the run and
    :math:`\\mathcal{A} = \\{v : \\rho^{*}(v) > \\theta_{\\text{at-risk}}\\}`.

    Two properties distinguish it from plain coverage:

    1. Informing a vehicle that is speeding towards the hazard contributes far
       more than informing one that is driving away from it -- the weights are
       the risk field, not the node count.
    2. A message that arrives after the vehicle has already passed the hazard
       contributes **nothing**, because the indicator requires the vehicle to
       still be at risk when it is informed. Late delivery is not partial
       credit.

    RWCR is 1.0 only when every at-risk vehicle is warned in time, and it is
    insensitive to how many irrelevant vehicles were reached -- which is
    exactly the behaviour that makes coverage-maximising flooding score badly.
    """
    w = ctx.peak_relevance[ctx.at_risk]
    if w.sum() <= 0:
        return float("nan")
    hit = ctx.informed_in_time[ctx.at_risk]
    return float((w * hit).sum() / w.sum())


def time_to_informed_at_risk(ctx: MetricContext, risk: RiskField) -> dict[str, float]:
    """**TIR** -- latency for the vehicles that actually needed the message.

    Measured from *hazard onset* (the safety clock), over vehicles whose peak
    relevance exceeds ``high_relevance_threshold`` (0.5 by default). Vehicles
    never informed are reported separately as ``tir_uninformed_frac`` rather
    than being dropped: excluding them would let a policy that reaches three
    nearby vehicles quickly beat one that reaches everyone slightly slower.
    """
    high = ctx.peak_relevance > risk.high_relevance_threshold
    n_high = int(high.sum())
    if n_high == 0:
        return {"tir_median_s": float("nan"), "tir_p95_s": float("nan"),
                "tir_uninformed_frac": float("nan"), "n_high_relevance": 0}
    got = high & ctx.informed_in_time
    lat = ctx.latency_s[got]
    return {
        "tir_median_s": _percentile(lat, 50),
        "tir_p95_s": _percentile(lat, 95),
        "tir_uninformed_frac": float(1.0 - got.sum() / n_high),
        "n_high_relevance": n_high,
    }


def deadline_miss_rate(
    ctx: MetricContext, res: RunResult, hazard: Hazard, hz_cfg: dict[str, Any]
) -> float:
    """Fraction of at-risk vehicles not usefully warned before the deadline.

    A miss is any of:

    * never informed, or informed only after passing the hazard;
    * informed later than ``hazard.safety_deadline_s`` after onset;
    * (when ``deadlines.require_actionable_eta``) informed when the vehicle's
      ETA to the hazard is already below the driver reaction time -- a warning
      that arrives too late to act on is not a warning.
    """
    tr = res.trace
    at_risk = ctx.at_risk
    n = int(at_risk.sum())
    if n == 0:
        return float("nan")

    ok = ctx.informed_in_time & at_risk
    ok &= ctx.latency_s <= hazard.safety_deadline_s

    dl = hz_cfg.get("deadlines", {})
    if dl.get("require_actionable_eta", False):
        reaction = float(dl.get("reaction_time_s", 2.5))
        idx = np.flatnonzero(ok)
        if idx.size:
            steps = res.informed_step[idx]
            eta = np.array([_eta_at(res, int(s), int(v)) for s, v in zip(steps, idx)])
            ok[idx] = eta >= reaction
    return float(1.0 - ok.sum() / n)


def _eta_at(res: RunResult, step: int, vehicle: int) -> float:
    tr, risk, hz = res.trace, res.risk, res.hazard
    f = risk.evaluate(
        tr.x[step, vehicle:vehicle + 1], tr.y[step, vehicle:vehicle + 1],
        tr.vx[step, vehicle:vehicle + 1], tr.vy[step, vehicle:vehicle + 1],
        tr.direction[vehicle:vehicle + 1], hz, step * tr.dt,
    )
    return float(f["eta_s"][0])


def overhead_metrics(res: RunResult, ctx: MetricContext) -> dict[str, float]:
    """Rebroadcast overhead and redundancy."""
    n_informed = int((res.informed_step >= 0).sum())
    n_at_risk_informed = int((ctx.informed_in_time & ctx.at_risk).sum())
    return {
        "transmissions": float(res.n_transmissions),
        "redundancy_ratio": res.n_transmissions / n_informed if n_informed else float("nan"),
        "tx_per_at_risk_informed": (
            res.n_transmissions / n_at_risk_informed if n_at_risk_informed else float("inf")
        ),
        "collisions_per_delivered": (
            res.n_fail_sinr / res.n_rx_success if res.n_rx_success else float("nan")
        ),
    }


def message_lifetime(res: RunResult) -> dict[str, float]:
    """How far the message travelled, in hops and in metres."""
    tr = res.trace
    informed = np.flatnonzero(res.informed_step >= 0)
    if informed.size == 0 or res.origin_step < 0:
        return {"max_hops": 0.0, "spatial_reach_m": 0.0}
    ox = float(tr.x[res.origin_step, res.origin_index])
    oy = float(tr.y[res.origin_step, res.origin_index])
    steps = res.informed_step[informed]
    dx = tr.x[steps, informed] - ox
    dy = tr.y[steps, informed] - oy
    return {
        "max_hops": float(res.hops[informed].max()),
        "spatial_reach_m": float(np.nanmax(np.hypot(dx, dy))),
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
def compute_metrics(res: RunResult, hz_cfg: dict[str, Any]) -> dict[str, Any]:
    """All Phase 6 metrics for one run."""
    risk, hazard, tr = res.risk, res.hazard, res.trace
    ctx = _build_context(res, risk, hazard)

    n_active_ever = int(tr.active.any(axis=0).sum())
    informed = res.informed_step >= 0
    lat_all = ctx.latency_s[np.isfinite(ctx.latency_s)]

    out: dict[str, Any] = {
        "pdr": packet_delivery_ratio(res),
        "rwcr": risk_weighted_coverage_ratio(ctx),
        "coverage": float(informed.sum() / n_active_ever) if n_active_ever else float("nan"),
        "at_risk_coverage": (
            float((ctx.informed_in_time & ctx.at_risk).sum() / ctx.at_risk.sum())
            if ctx.at_risk.any() else float("nan")
        ),
        "latency_mean_s": float(lat_all.mean()) if lat_all.size else float("nan"),
        "latency_p95_s": _percentile(lat_all, 95),
        "deadline_miss_rate": deadline_miss_rate(ctx, res, hazard, hz_cfg),
        "n_at_risk": int(ctx.at_risk.sum()),
        "n_informed": int(informed.sum()),
        "n_vehicles": n_active_ever,
        "origin_step": int(res.origin_step),
        "origin_time_s": float(res.origin_step * tr.dt) if res.origin_step >= 0 else float("nan"),
    }
    out.update(time_to_informed_at_risk(ctx, risk))
    out.update(overhead_metrics(res, ctx))
    out.update(message_lifetime(res))

    # Channel diagnostics -- not paper metrics, but the first thing to look at
    # when a number surprises you.
    out.update({
        "rx_attempts": res.n_rx_attempts,
        "rx_success": res.n_rx_success,
        "fail_sensitivity": res.n_fail_sensitivity,
        "fail_sinr": res.n_fail_sinr,
        "fail_beacon": res.n_fail_beacon,
        "backoff_ties": res.n_backoff_ties,
        "hidden_overlaps": res.n_hidden_overlaps,
        "busy_deferrals": res.n_busy_deferrals,
        "dropped_ttl": res.n_dropped_ttl,
    })
    # Which actions the policy actually chose. `n_action_carry` is the one that
    # tells you whether store-carry-forward ever fired, or whether the network
    # was connected enough that the policy never reached that branch.
    for action, count in (res.action_counts or {}).items():
        out[f"n_action_{action}"] = int(count)
    return out


_LABELS: dict[str, tuple[str, str]] = {
    "rwcr": ("Risk-Weighted Coverage Ratio (RWCR)", "{:.4f}"),
    "at_risk_coverage": ("At-risk coverage (unweighted)", "{:.4f}"),
    "coverage": ("Coverage, all vehicles", "{:.4f}"),
    "pdr": ("Packet delivery ratio", "{:.4f}"),
    "tir_median_s": ("TIR median (s)", "{:.3f}"),
    "tir_p95_s": ("TIR p95 (s)", "{:.3f}"),
    "tir_uninformed_frac": ("TIR uninformed fraction", "{:.4f}"),
    "latency_mean_s": ("End-to-end latency mean (s)", "{:.3f}"),
    "latency_p95_s": ("End-to-end latency p95 (s)", "{:.3f}"),
    "transmissions": ("Transmissions", "{:.0f}"),
    "redundancy_ratio": ("Redundancy ratio (tx / informed)", "{:.3f}"),
    "collisions_per_delivered": ("Collisions per delivered frame", "{:.4f}"),
    "deadline_miss_rate": ("Safety-deadline miss rate", "{:.4f}"),
    "max_hops": ("Message lifetime (hops)", "{:.0f}"),
    "spatial_reach_m": ("Spatial reach (m)", "{:.0f}"),
}


def format_metrics(metrics: dict[str, Any], title: str = "METRICS") -> str:
    """Human-readable block for the console. Not a paper table."""
    width = 58
    lines = [f"{'=' * width}", f" {title}", f"{'=' * width}"]
    for key, (label, fmt) in _LABELS.items():
        if key not in metrics:
            continue
        v = metrics[key]
        s = "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)
        lines.append(f" {label:<42}{s:>14}")
    lines.append("-" * width)
    lines.append(
        f" {'at-risk set / informed / vehicles':<42}"
        f"{metrics['n_at_risk']}/{metrics['n_informed']}/{metrics['n_vehicles']:>4}".rjust(14)
    )
    lines.append(
        f" {'frame losses  sens / SINR / beacon':<42}"
        f"{metrics['fail_sensitivity']}/{metrics['fail_sinr']}/{metrics['fail_beacon']:>4}".rjust(14)
    )
    lines.append(
        f" {'MAC  ties / hidden / busy-defer':<42}"
        f"{metrics['backoff_ties']}/{metrics['hidden_overlaps']}/"
        f"{metrics['busy_deferrals']:>4}".rjust(14)
    )
    lines.append("=" * width)
    return "\n".join(lines)
