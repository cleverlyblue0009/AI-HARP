"""Capture the attention coefficients of one real dissemination event.

::

    python -m experiments.attention_event --checkpoint checkpoints/campaign/ref/ckpt_final.pt

Figure 7 shows what the GATv2 layer attended to while a message spread. That
figure used to be produced by hand, which meant ``reproduce.sh`` could not
rebuild it; this script is the reproducible replacement. It writes
``results/attention_event.json`` and the figure is drawn from that file.

The coefficients are the ones that actually selected the relays -- the policy
ranks neighbours by attention onto the holder -- not a separate visualisation
head, so the figure describes the decision rather than illustrating it.

One run is not a result. This is a qualitative figure and its caption says so;
nothing in the paper's claims rests on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.registry import build_policy
from common.config import PROJECT_ROOT, RESULTS_DIR, ensure_dir, load_yaml
from common.logging_utils import get_logger
from common.seeding import SeedBundle
from hazard.model import hazard_from_config
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace, load_scenario
from sim.engine import DisseminationEngine, SimSettings
from sim.mac import build_mac
from sim.phy import build_phy

logger = get_logger("experiments.attention")


def capture(
    checkpoint: str, scenario: str = "rural_highway", density: float = 20.0,
    weather: str = "clear", hazard_type: str | None = None, seed: int = 0,
    tau: float = 0.5, backend: str = "auto",
) -> dict[str, Any]:
    """Run one (trace, hazard, agent) episode and return its attention log."""
    phy_cfg = load_yaml("phy.yaml")
    hz_cfg = load_yaml("hazard.yaml")
    exp_cfg = load_yaml("experiment.yaml")
    scenario_cfg = load_scenario(scenario)
    seeds = SeedBundle(master_seed=seed)

    trace = get_trace(scenario_cfg, density, seed, weather=weather, backend=backend,
                      phy_cfg=phy_cfg)
    overrides = {"type": hazard_type} if hazard_type else None
    hazard = hazard_from_config(hz_cfg, trace.meta, overrides=overrides)
    phy = build_phy(phy_cfg, scenario=scenario_cfg.get("phy_profile", scenario_cfg["name"]),
                    weather=weather, seed=seed, trace_meta=trace.meta)
    mac = build_mac(phy_cfg, phy)
    risk = build_risk_field(hz_cfg, trace, hazard)

    path = Path(checkpoint)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    policy = build_policy("ai_harp", checkpoint=str(path), tau=tau,
                          deterministic=True, capture_attention=True)
    policy.phy = phy                      # same edge features as training

    result = DisseminationEngine(trace, phy, mac, risk, hazard, policy, seeds,
                                 SimSettings.from_config(exp_cfg)).run()
    if result.origin_step < 0:
        raise SystemExit("no vehicle detected the hazard: no event to show")

    log = list(policy.attention_log)
    logger.info("captured %d network decisions over %d steps", len(log),
                len({r["step"] for r in log}))
    return {
        "checkpoint": str(path), "scenario": scenario, "density_veh_km_lane": density,
        "weather": weather, "hazard_type": hazard.htype, "seed": seed, "tau": tau,
        "backend": trace.meta.get("backend", backend),
        "origin_step": int(result.origin_step),
        "dt_s": float(trace.meta.get("dt_s", np.nan)),
        "decisions": log,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="checkpoints/campaign/ref/ckpt_final.pt")
    ap.add_argument("--scenario", default="rural_highway")
    ap.add_argument("--density", type=float, default=20.0)
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--hazard", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default=str(RESULTS_DIR / "attention_event.json"))
    args = ap.parse_args(argv)

    payload = capture(args.checkpoint, scenario=args.scenario, density=args.density,
                      weather=args.weather, hazard_type=args.hazard, seed=args.seed,
                      tau=args.tau, backend=args.backend)
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{len(payload['decisions'])} decisions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
