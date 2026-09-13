"""Single-run driver.

::

    python -m experiments.run_sim --scenario rural_highway --density 20 \\
        --weather clear --policy flooding --seed 0

One run is never a result. This entry point exists to inspect one cell of the
sweep, to smoke-test the pipeline, and to produce the event log behind a
figure; Phase 7's sweep runner is what produces anything that goes in a table.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from agents.registry import available_policies, build_policy
from analysis.metrics import compute_metrics, format_metrics
from common.config import config_hash, load_yaml
from common.logging_utils import get_logger
from common.seeding import SeedBundle
from hazard.model import hazard_from_config, sample_hazard
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace, load_scenario
from sim.engine import DisseminationEngine, RunResult, SimSettings
from sim.mac import build_mac
from sim.phy import build_phy

logger = get_logger("experiments.run")


@dataclass
class RunSpec:
    """Everything that defines one run. Hashed into the results row."""

    scenario: str = "rural_highway"
    density_veh_km_lane: float = 20.0
    weather: str = "clear"
    policy: str = "flooding"
    policy_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    hazard_type: str | None = None
    sample_hazard_instance: bool = False
    backend: str = "auto"
    duration_s: float | None = None
    corridor_length_m: float | None = None

    def key(self) -> str:
        return config_hash(self.__dict__)


def run_single(
    spec: RunSpec,
    *,
    phy_cfg: dict[str, Any] | None = None,
    hz_cfg: dict[str, Any] | None = None,
    exp_cfg: dict[str, Any] | None = None,
    return_result: bool = False,
) -> tuple[dict[str, Any], RunResult | None]:
    """Execute one (mobility, hazard, policy) run and return its metrics."""
    phy_cfg = phy_cfg or load_yaml("phy.yaml")
    hz_cfg = hz_cfg or load_yaml("hazard.yaml")
    exp_cfg = exp_cfg or load_yaml("experiment.yaml")

    scenario_cfg = load_scenario(spec.scenario)
    if spec.duration_s is not None:
        scenario_cfg["simulation"]["duration_s"] = float(spec.duration_s)
    if spec.corridor_length_m is not None:
        scenario_cfg["geometry"]["length_m"] = float(spec.corridor_length_m)

    seeds = SeedBundle(master_seed=spec.seed)

    trace = get_trace(
        scenario_cfg, spec.density_veh_km_lane, spec.seed,
        weather=spec.weather, backend=spec.backend, phy_cfg=phy_cfg,
    )

    if spec.sample_hazard_instance:
        hazard = sample_hazard(hz_cfg, trace.meta, seeds.rng("hazard"), htype=spec.hazard_type)
    else:
        overrides = {"type": spec.hazard_type} if spec.hazard_type else None
        hazard = hazard_from_config(hz_cfg, trace.meta, overrides=overrides)

    phy = build_phy(phy_cfg, scenario=scenario_cfg["name"], weather=spec.weather,
                    seed=spec.seed, trace_meta=trace.meta)
    mac = build_mac(phy_cfg, phy)
    risk = build_risk_field(hz_cfg, trace, hazard)
    policy = build_policy(spec.policy, **spec.policy_params)
    # A learned policy computes its predicted-RSSI edge feature from the PHY.
    # Training sets this; evaluation must too, or the agent is scored on
    # features computed differently from the ones it was trained on.
    if hasattr(policy, "phy"):
        policy.phy = phy
    settings = SimSettings.from_config(exp_cfg)

    logger.info("PHY  | %s", json.dumps(phy.summary()))
    logger.info("WX   | %s", phy.weather.summary())
    logger.info("MAC  | %s", json.dumps(mac.summary()))
    logger.info("HAZ  | %s", hazard.describe())
    logger.info("POL  | %s", policy.describe())

    engine = DisseminationEngine(trace, phy, mac, risk, hazard, policy, seeds, settings)
    result = engine.run()

    if result.origin_step < 0:
        logger.warning(
            "No vehicle ever detected the hazard -- no message was originated. "
            "Check hazard placement (position_frac), onset_time_s and detection_range_m."
        )

    metrics = compute_metrics(result, hz_cfg)
    # Policy-level statistics (the confidence gate's fallback rate is a
    # first-class reported metric) would otherwise never reach a results row.
    if hasattr(policy, "stats"):
        metrics.update(policy.stats())
    metrics.update({
        "scenario": spec.scenario,
        "density_veh_km_lane": spec.density_veh_km_lane,
        "weather": spec.weather,
        "policy": spec.policy,
        "seed": spec.seed,
        "hazard_type": hazard.htype.value,
        "hazard_severity": hazard.severity0,
        "backend": trace.backend,
        "config_hash": config_hash(spec.__dict__, phy_cfg, hz_cfg, exp_cfg["simulation"]),
        "comm_range_m": round(engine.comm_range_m, 1),
        "cs_range_m": round(engine.cs_range_m, 1),
    })
    return metrics, (result if return_result else None)


def run_smoke(exp_cfg: dict[str, Any] | None = None) -> int:
    """End-to-end pipeline check on tiny settings.

    Exercises every stage that currently exists -- mobility generation and
    caching, PHY/MAC construction, hazard placement, the risk field, the
    engine, and metrics -- across every scenario/weather combination in the
    ``smoke:`` block, and asserts the invariants that must hold before a sweep
    is worth launching. Designed to finish in well under two minutes.
    """
    import time

    exp_cfg = exp_cfg or load_yaml("experiment.yaml")
    sm = exp_cfg["smoke"]
    phy_cfg, hz_cfg = load_yaml("phy.yaml"), load_yaml("hazard.yaml")
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    combos = [
        (sc, d, w, h, p, s)
        for sc in sm["scenarios"]
        for d in sm["densities_veh_km_lane"]
        for w in sm["weather"]
        for h in sm["hazard_types"]
        for p in sm["policies"]
        for s in sm["seeds"]
    ]
    print(f"[smoke] {len(combos)} runs\n")
    for sc, d, w, h, p, s in combos:
        spec = RunSpec(
            scenario=sc, density_veh_km_lane=d, weather=w, policy=p, seed=s, hazard_type=h,
            duration_s=sm.get("duration_s"), corridor_length_m=sm.get("corridor_length_m"),
        )
        m, res = run_single(spec, phy_cfg=phy_cfg, hz_cfg=hz_cfg, exp_cfg=exp_cfg,
                            return_result=True)
        rows.append(m)
        tag = f"{sc}/{p}/d={d}/{w}/{h}/seed={s}"
        if res.origin_step < 0:
            failures.append(f"{tag}: hazard was never detected")
        elif res.n_transmissions == 0:
            # A detected hazard that produces no transmission at all means the
            # channel model has locked up (this is how the missing-DCC
            # saturation bug showed itself), not that the policy chose silence.
            failures.append(
                f"{tag}: hazard detected but zero transmissions "
                f"({res.n_busy_deferrals} busy deferrals) -- channel saturated?"
            )
        if not (0.0 <= m["rwcr"] <= 1.0):
            failures.append(f"{tag}: RWCR out of range ({m['rwcr']})")
        if res.n_rx_success + res.n_fail_sensitivity + res.n_fail_sinr + res.n_fail_beacon \
                != res.n_rx_attempts:
            failures.append(f"{tag}: frame accounting does not balance")

    hdr = f"{'scenario':<14}{'pol':<10}{'wx':<14}{'hazard':<14}{'sd':>3}" \
          f"{'RWCR':>8}{'PDR':>8}{'TIRp50':>8}{'tx':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        print(
            f"{m['scenario']:<14}{m['policy']:<10}{m['weather']:<14}{m['hazard_type']:<14}"
            f"{m['seed']:>3}{m['rwcr']:>8.3f}{m['pdr']:>8.3f}"
            f"{m['tir_median_s']:>8.2f}{m['transmissions']:>6.0f}"
        )

    dt = time.time() - t0
    print(f"\n[smoke] {len(rows)} runs in {dt:.1f}s")
    if failures:
        print(f"[smoke] FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke] OK -- all invariants held")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-HARP single-run driver")
    p.add_argument("--scenario", default="rural_highway",
                   choices=["rural_highway", "urban_grid", "urban_nlos"])
    p.add_argument("--density", type=float, default=20.0, help="veh/km/lane")
    p.add_argument("--weather", default="clear",
                   choices=["clear", "moderate_rain", "heavy_rain", "dense_fog"])
    p.add_argument("--policy", default="flooding", choices=available_policies())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hazard", default=None, help="hazard type (default: hazard.yaml)")
    p.add_argument("--sample-hazard", action="store_true",
                   help="sample severity/extent/placement instead of the fixed default")
    p.add_argument("--backend", default="auto", choices=["auto", "sumo", "fallback"])
    p.add_argument("--duration", type=float, default=None, help="override duration (s)")
    p.add_argument("--json", action="store_true", help="emit metrics as JSON")
    p.add_argument("--smoke", action="store_true",
                   help="run the whole pipeline end to end on tiny settings (<2 min)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        return run_smoke()
    spec = RunSpec(
        scenario=args.scenario, density_veh_km_lane=args.density, weather=args.weather,
        policy=args.policy, seed=args.seed, hazard_type=args.hazard,
        sample_hazard_instance=args.sample_hazard, backend=args.backend,
        duration_s=args.duration,
    )
    metrics, _ = run_single(spec)
    if args.json:
        print(json.dumps({k: v for k, v in metrics.items()}, default=str, indent=2))
    else:
        title = (
            f"{spec.policy} | {spec.scenario} | {spec.density_veh_km_lane:g} veh/km/lane "
            f"| {spec.weather} | seed {spec.seed}"
        )
        print(format_metrics(metrics, title))
        print(
            "\nNOTE: this is a SINGLE RUN. No number here belongs in the paper; "
            "Phase 7 aggregates 10 seeds per cell with mean +/- std."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
