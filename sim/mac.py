"""Approximate IEEE 802.11p CSMA/CA for *broadcast* frames.

Broadcast frames are unacknowledged. There is no RTS/CTS, no ACK, no
retransmission, and no binary exponential backoff -- with no ACK there is no
timeout to trigger a CW increase, so the window stays at ``CWmin``. A lost
broadcast is simply lost, which is precisely why relay selection matters.

Three loss mechanisms are modelled, and they are kept distinct because they
behave differently as density grows:

1. **Backoff-slot collisions between concurrent relays.** Relays that hear each
   other contend. Each draws a backoff slot in ``[0, CW]``; two that draw the
   same slot transmit simultaneously and their frames overlap in time. The
   probability that a given frame ties with at least one of ``n-1`` others is
   the standard ``1 - (1 - 1/(CW+1))^(n-1)``.

2. **Hidden terminals.** Relays outside each other's carrier-sense range never
   defer for one another. They overlap whenever their transmissions fall within
   a mutual vulnerable window, ``2 * T_frame`` out of the decision epoch.
   Whether an overlap actually destroys the frame is left to the SINR/capture
   test in the engine, which is what makes hidden-terminal loss distance- and
   power-dependent rather than a flat probability.

3. **Background beacon load.** Every vehicle also transmits CAM/BSM position
   beacons at ``background_beacon_hz``. At the *receiver*, these arrive as a
   Poisson stream; a frame survives them with probability
   ``exp(-lambda * T_vuln)`` where ``lambda`` is the beacon rate summed over
   the receiver's neighbours. This is the floor of channel load that hazard
   dissemination has to share the medium with.

The engine computes explicit SINR for (1) and (2) because it knows where every
interferer is; (3) is a per-reception Bernoulli because the beacon senders are
not individually tracked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MacModel:
    """Resolved MAC parameters and the derived channel-access probabilities."""

    slot_time_s: float
    sifs_s: float
    difs_s: float
    cw_min: int
    cw_max: int
    aifsn: int
    frame_bytes: int
    carrier_sense_threshold_dbm: float
    interference_margin_db: float
    channel_busy_suppression: bool
    background_beacon_enabled: bool
    background_beacon_hz: float
    background_beacon_bytes: int
    vulnerable_window_frames: float
    frame_duration_s: float
    beacon_duration_s: float
    dcc_enabled: bool = True
    dcc_target_cbr: float = 0.62
    dcc_min_beacon_hz: float = 1.0

    # ------------------------------------------------------------- properties --
    @property
    def cw_slots(self) -> int:
        """Number of distinct backoff slots: the CW spans ``[0, CWmin]``."""
        return int(self.cw_min) + 1

    @property
    def vulnerable_window_s(self) -> float:
        """Window in which an interferer's start destroys our frame."""
        return self.vulnerable_window_frames * self.frame_duration_s

    # ------------------------------------------------------------- collisions --
    def tie_probability(self, n_contenders: int) -> float:
        """P(a frame shares its backoff slot with >= 1 of ``n_contenders - 1``).

        ``1 - (1 - 1/W)^(n-1)`` with ``W = CWmin + 1``. This is the textbook
        approximation and is exact for a single contention round in which all
        contenders are mutually in carrier-sense range.
        """
        if n_contenders <= 1:
            return 0.0
        return 1.0 - (1.0 - 1.0 / self.cw_slots) ** (n_contenders - 1)

    def draw_backoff_slots(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform backoff slots in ``[0, CWmin]`` -- one per contending relay."""
        return rng.integers(0, self.cw_slots, size=n)

    def hidden_overlap_probability(self, epoch_s: float) -> float:
        """P(two mutually hidden relays' frames overlap within one epoch).

        Both transmit once somewhere in the epoch with no coordination, so the
        overlap probability is the vulnerable window as a fraction of the epoch.
        """
        return float(min(1.0, self.vulnerable_window_s / max(epoch_s, 1e-12)))

    # -------------------------------------------------------- background load --
    def effective_beacon_hz(self, n_neighbours: np.ndarray | int) -> np.ndarray:
        """Per-vehicle beacon rate after DCC rate adaptation.

        ETSI TS 102 687 adapts the beacon rate to hold the channel busy ratio
        near a target. Modelling the limit directly: with ``n`` neighbours each
        sending frames of duration ``T_b``, the rate that lands on the target
        CBR is ``target / (n * T_b)``, floored at ``dcc_min_beacon_hz`` and
        capped at the unloaded rate.

        Omitting this does not make the model conservative -- it makes it
        degenerate: the medium saturates, every frame defers, and no hazard
        message is transmitted at any density above a few hundred neighbours.
        """
        n = np.asarray(n_neighbours, dtype=float)
        if not self.background_beacon_enabled:
            return np.zeros_like(n)
        if not self.dcc_enabled:
            return np.full_like(n, self.background_beacon_hz)
        with np.errstate(divide="ignore", invalid="ignore"):
            allowed = self.dcc_target_cbr / np.maximum(n * self.beacon_duration_s, 1e-12)
        return np.clip(allowed, self.dcc_min_beacon_hz, self.background_beacon_hz)

    def beacon_arrival_rate(self, n_neighbours: np.ndarray | int) -> np.ndarray:
        """Aggregate beacon frame rate seen at a receiver, in frames/s."""
        if not self.background_beacon_enabled:
            return np.zeros_like(np.asarray(n_neighbours, dtype=float))
        n = np.asarray(n_neighbours, dtype=float)
        return n * self.effective_beacon_hz(n)

    def beacon_survival_probability(self, n_neighbours: np.ndarray | int) -> np.ndarray:
        """P(no background beacon overlaps our frame at this receiver)."""
        lam = self.beacon_arrival_rate(n_neighbours)
        return np.exp(-lam * self.vulnerable_window_s)

    def channel_busy_probability(self, n_neighbours: np.ndarray | int) -> np.ndarray:
        """P(CCA finds the medium busy) = offered background load, capped at 1.

        Channel utilisation from beacons alone: ``lambda * T_beacon``.
        """
        if not (self.background_beacon_enabled and self.channel_busy_suppression):
            return np.zeros_like(np.asarray(n_neighbours, dtype=float))
        lam = self.beacon_arrival_rate(n_neighbours)
        return np.clip(lam * self.beacon_duration_s, 0.0, 0.99)

    def channel_busy_ratio(self, n_neighbours: float) -> float:
        """Reported channel busy ratio (CBR) from background load alone."""
        return float(
            np.clip(self.beacon_arrival_rate(n_neighbours) * self.beacon_duration_s, 0.0, 1.0)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "cw_slots": self.cw_slots,
            "frame_bytes": self.frame_bytes,
            "frame_duration_us": round(self.frame_duration_s * 1e6, 1),
            "vulnerable_window_us": round(self.vulnerable_window_s * 1e6, 1),
            "beacon_hz": self.background_beacon_hz if self.background_beacon_enabled else 0.0,
            "dcc": f"target_cbr={self.dcc_target_cbr}" if self.dcc_enabled else "off",
            "cs_threshold_dbm": self.carrier_sense_threshold_dbm,
        }


def build_mac(phy_cfg: dict[str, Any], phy_model: Any) -> MacModel:
    """Resolve the ``mac:`` block of ``configs/phy.yaml``.

    ``phy_model`` supplies the frame airtime, which depends on the PHY data
    rate, so the two are always consistent.
    """
    m = phy_cfg["mac"]
    frame_bytes = int(m["frame_bytes"])
    beacon_bytes = int(m.get("background_beacon_bytes", frame_bytes))
    return MacModel(
        slot_time_s=float(m["slot_time_us"]) * 1e-6,
        sifs_s=float(m["sifs_us"]) * 1e-6,
        difs_s=float(m["difs_us"]) * 1e-6,
        cw_min=int(m["cw_min"]),
        cw_max=int(m["cw_max"]),
        aifsn=int(m["aifsn"]),
        frame_bytes=frame_bytes,
        carrier_sense_threshold_dbm=float(m["carrier_sense_threshold_dbm"]),
        interference_margin_db=float(m["interference_margin_db"]),
        channel_busy_suppression=bool(m["channel_busy_suppression"]),
        background_beacon_enabled=bool(m.get("background_beacon_enabled", False)),
        background_beacon_hz=float(m.get("background_beacon_hz", 0.0)),
        background_beacon_bytes=beacon_bytes,
        vulnerable_window_frames=float(m.get("vulnerable_window_frames", 2.0)),
        frame_duration_s=phy_model.frame_duration_s(frame_bytes),
        beacon_duration_s=phy_model.frame_duration_s(beacon_bytes),
        dcc_enabled=bool(m.get("dcc_enabled", True)),
        dcc_target_cbr=float(m.get("dcc_target_cbr", 0.62)),
        dcc_min_beacon_hz=float(m.get("dcc_min_beacon_hz", 1.0)),
    )
