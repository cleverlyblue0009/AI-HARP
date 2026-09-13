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
from hazard.oracle import OracleRiskField, build_oracle_risk_field, estimation_agreement
from hazard.risk_field import RiskField
from sim.engine import NO_STEP, RunResult

#: Bumped whenever a metric DEFINITION changes, and stamped into every results
#: row. `config_hash` covers the configuration but not the code, so without
#: this a row computed under an older definition is indistinguishable from a
#: current one -- which is how 360 pre-oracle rows survived in results/runs.csv
#: and were nearly used to draw a figure.
#:
#: 1  original causal at-risk set
#: 2  oracle at-risk set for RWCR/TIR; actionable_deadline_miss_rate; regime;
#:    risk-estimation agreement; per-transmitter collision attribution
METRICS_VERSION = 2

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
    "actionable_deadline_miss_rate": -1,
    # Cost metrics. `tx_per_at_risk_informed` -- the headline cost axis -- was
    # missing, so it defaulted to +1 (higher is better). The backend
    # validation printout then ranked flooding BEST on cost, and
    # analysis.stats.strongest_baseline would have picked flooding as the
    # strongest cost baseline: the weakest opponent, chosen as the reference.
    "tx_per_at_risk_informed": -1,
    "airtime_ms": -1,
    "airtime_per_at_risk_informed_ms": -1,
    "dissemination_cbr": -1,
    "max_hops": +1,
    "spatial_reach_m": +1,
}


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


@dataclass
class MetricContext:
    """Precomputed quantities shared by several metrics.

    The relevance here is the **oracle** field: the at-risk set, RWCR and TIR
    are evaluation-time judgements about who actually needed the warning, and
    answering that from a vehicle's instantaneous heading was what made grid
    RWCR meaningless. The causal field is never used for scoring, and the
    oracle is never used for deciding -- see hazard/oracle.py and
    tests/test_oracle_isolation.py.
    """

    relevance: np.ndarray        # [T, N] ORACLE relevance over the whole run
    peak_relevance: np.ndarray   # [N]
    at_risk: np.ndarray          # [N] bool
    informed_in_time: np.ndarray # [N] bool
    latency_s: np.ndarray        # [N] seconds from hazard onset, inf if never
    oracle: OracleRiskField | None = None


def _build_context(res: RunResult, risk: RiskField, hazard: Hazard) -> MetricContext:
    tr = res.trace
    oracle = build_oracle_risk_field(risk, hazard)
    rel = oracle.relevance_matrix(tr, hazard)
    peak, at_risk = oracle.at_risk_set(rel)

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
        informed_in_time=informed_in_time, latency_s=latency, oracle=oracle,
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


def actionable_deadline_miss_rate(
    ctx: MetricContext, res: RunResult, hazard: Hazard, hz_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Deadline misses restricted to vehicles a policy could actually have saved.

    ``deadline_miss_rate`` is dominated by vehicles that were *already* inside
    the hazard span, or already within reaction time of it, at the instant the
    hazard was first detected. No dissemination scheme can help those vehicles:
    the message did not exist yet, and no relay decision changes that. Including
    them adds a large policy-independent constant to every result, which is why
    the plain metric sits at 0.27-0.39 for all nine baselines and discriminates
    nothing.

    The actionable set is the at-risk vehicles whose ETA to the hazard at
    origination still exceeded the driver reaction time -- i.e. those for whom a
    timely warning was physically possible. This is the metric to report as
    primary; the unrestricted one is kept for comparability with prior work.
    """
    tr = res.trace
    dl = hz_cfg.get("deadlines", {})
    reaction = float(dl.get("reaction_time_s", 2.5))

    if res.origin_step < 0 or not ctx.at_risk.any():
        return {"actionable_deadline_miss_rate": float("nan"),
                "n_actionable": 0, "n_unreachable_at_origin": 0}

    # ETA of every vehicle at the moment the message came into existence.
    s = res.origin_step
    feats = res.risk.evaluate(
        np.nan_to_num(tr.x[s]), np.nan_to_num(tr.y[s]), tr.vx[s], tr.vy[s],
        tr.direction, hazard, s * tr.dt, active=tr.active[s],
    )
    eta_at_origin = feats["eta_s"]

    # A vehicle counts as actionable if it was at risk and still had more than
    # a reaction time in hand when the hazard was detected. Vehicles that enter
    # the corridor later are actionable by construction (they were nowhere near
    # the hazard at origination), so absent-at-origin counts as actionable.
    present = tr.active[s]
    had_time = (~present) | (eta_at_origin >= reaction)
    actionable = ctx.at_risk & had_time
    n_actionable = int(actionable.sum())
    n_unreachable = int((ctx.at_risk & ~had_time).sum())

    if n_actionable == 0:
        return {"actionable_deadline_miss_rate": float("nan"),
                "n_actionable": 0, "n_unreachable_at_origin": n_unreachable}

    ok = ctx.informed_in_time & actionable
    ok &= ctx.latency_s <= hazard.safety_deadline_s
    if dl.get("require_actionable_eta", False):
        idx = np.flatnonzero(ok)
        if idx.size:
            eta = np.array([_eta_at(res, int(st), int(v))
                            for st, v in zip(res.informed_step[idx], idx)])
            ok[idx] = eta >= reaction

    return {
        "actionable_deadline_miss_rate": float(1.0 - ok.sum() / n_actionable),
        "n_actionable": n_actionable,
        "n_unreachable_at_origin": n_unreachable,
    }


def traffic_regime(res: RunResult) -> dict[str, Any]:
    """Label the traffic state from measured speed against free-flow speed.

    At 120 veh/km/lane the rural corridor runs at ~2.5 m/s: that is a jam, not
    high-density free flow, and presenting it as a density point without
    saying so would misrepresent it. Every figure carries this label.

    Thresholds follow the usual speed-ratio convention: above 0.85 of free-flow
    speed is uncongested, below 0.5 is breakdown.
    """
    tr = res.trace
    speed = np.hypot(tr.vx, tr.vy)
    moving = speed[tr.active]
    if moving.size == 0:
        return {"mean_speed_ms": float("nan"), "free_flow_speed_ms": float("nan"),
                "speed_ratio": float("nan"), "regime": "unknown"}
    mean_speed = float(moving.mean())
    # Free-flow reference: the 95th percentile of achieved speed, which is what
    # unobstructed vehicles in this same trace manage. Using the trace itself
    # keeps the reference consistent with the weather speed factor.
    free_flow = float(np.percentile(moving, 95))
    ratio = mean_speed / free_flow if free_flow > 0 else float("nan")
    if not np.isfinite(ratio):
        regime = "unknown"
    elif ratio > 0.85:
        regime = "free_flow"
    elif ratio > 0.5:
        regime = "congested"
    else:
        regime = "jammed"
    return {
        "mean_speed_ms": round(mean_speed, 3),
        "free_flow_speed_ms": round(free_flow, 3),
        "speed_ratio": round(ratio, 4),
        "regime": regime,
    }


def _eta_at(res: RunResult, step: int, vehicle: int) -> float:
    tr, risk, hz = res.trace, res.risk, res.hazard
    f = risk.evaluate(
        tr.x[step, vehicle:vehicle + 1], tr.y[step, vehicle:vehicle + 1],
        tr.vx[step, vehicle:vehicle + 1], tr.vy[step, vehicle:vehicle + 1],
        tr.direction[vehicle:vehicle + 1], hz, step * tr.dt,
    )
    return float(f["eta_s"][0])


def overhead_metrics(res: RunResult, ctx: MetricContext) -> dict[str, float]:
    """Rebroadcast overhead, redundancy and channel load.

    ``tx_per_at_risk_informed`` is the cost axis of the Pareto front in
    :mod:`analysis.pareto`: how many transmissions the network spent per
    at-risk vehicle actually warned in time. It is the quantity the paper's
    headline result minimises at matched RWCR.
    """
    n_informed = int((res.informed_step >= 0).sum())
    n_at_risk_informed = int((ctx.informed_in_time & ctx.at_risk).sum())

    # Channel occupancy contributed by the dissemination itself, in airtime.
    frame_s = res.mac.frame_duration_s if res.mac is not None else float("nan")
    airtime_ms = res.n_transmissions * frame_s * 1e3
    duration_s = res.trace.duration_s if res.trace is not None else float("nan")

    # Background load from CAM/BSM beaconing at the mean neighbourhood size,
    # after DCC rate adaptation. This is the floor the dissemination sits on.
    beacon_cbr = float("nan")
    if res.mac is not None and res.trace is not None:
        mean_concurrent = float(res.trace.active.sum(axis=1).mean())
        # Neighbours within carrier sense, approximated by the share of the
        # network inside one CS radius on a corridor of known length.
        span = float(res.trace.meta.get("length_m", 0.0)) or float(
            res.trace.meta.get("total_lane_km", 1.0) * 1000.0
        )
        cs_r = float(res.meta.get("cs_range_m", 0.0))
        frac = min(1.0, (2 * cs_r / span) if span > 0 else 1.0)
        beacon_cbr = res.mac.channel_busy_ratio(mean_concurrent * frac)

    return {
        "transmissions": float(res.n_transmissions),
        "redundancy_ratio": res.n_transmissions / n_informed if n_informed else float("nan"),
        "tx_per_at_risk_informed": (
            res.n_transmissions / n_at_risk_informed if n_at_risk_informed else float("inf")
        ),
        "collisions_per_delivered": (
            res.n_fail_sinr / res.n_rx_success if res.n_rx_success else float("nan")
        ),
        "airtime_ms": airtime_ms,
        "airtime_per_at_risk_informed_ms": (
            airtime_ms / n_at_risk_informed if n_at_risk_informed else float("inf")
        ),
        "dissemination_cbr": airtime_ms / (duration_s * 1e3) if duration_s else float("nan"),
        "beacon_cbr": beacon_cbr,
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
        "metrics_version": METRICS_VERSION,
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
    out.update(estimation_agreement(tr, hazard, risk, ctx.oracle))
    out.update(actionable_deadline_miss_rate(ctx, res, hazard, hz_cfg))
    out.update(traffic_regime(res))
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
    "actionable_deadline_miss_rate": ("Deadline miss rate, actionable set", "{:.4f}"),
    "deadline_miss_rate": ("Deadline miss rate, all at-risk", "{:.4f}"),
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
