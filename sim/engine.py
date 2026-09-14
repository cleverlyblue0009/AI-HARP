"""The time-stepped dissemination engine (100 ms epochs).

Per epoch:

1. **Detection.** A vehicle within ``detection_range_m`` of an active hazard,
   and heading into it, originates the hazard message.
2. **Channel access.** Vehicles with a transmission scheduled for this epoch
   perform CCA; a busy medium delays the frame within the epoch (802.11p
   backoff slots are 13 us) and never drops it.
3. **Contention.** Surviving transmitters draw backoff slots. Transmitters that
   hear each other and tie -- plus mutually hidden transmitters whose frames
   fall in each other's vulnerable window -- overlap in time.
4. **Reception.** For every (transmitter, receiver) pair: log-distance path loss
   + weather + correlated shadowing + per-frame Nakagami fading gives received
   power; overlapping transmitters contribute interference; the frame decodes
   if it clears both receiver sensitivity and the SINR threshold, survives the
   background beacon load, and the receiver is not itself transmitting.
5. **Decisions.** Newly informed vehicles, and vehicles whose deferred timers
   have come due, consult the policy.

A rebroadcast can never go out in the same epoch as the reception that
triggered it: there is always at least one 100 ms epoch of turnaround for
processing and channel contention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.base import Action, ActionType, DecisionContext, Policy, Trigger
from common.logging_utils import get_logger
from common.seeding import SeedBundle
from hazard.model import Hazard
from hazard.risk_field import APPROACHING, INSIDE, RiskField
from mobility.trace import Trace
from sim.mac import MacModel
from sim.phy import PhyModel

logger = get_logger("sim.engine")

NO_STEP = -1


@dataclass
class SimSettings:
    """Engine-level knobs (from ``configs/experiment.yaml``)."""

    max_hops: int = 32
    message_ttl_s: float = 60.0
    carry_recheck_steps: int = 10
    max_originators: int = 1
    rx_prune_margin_db: float = 15.0
    record_transmissions: bool = True

    #: Keys that used to exist and must not come back silently.
    REMOVED_KEYS = {
        "max_busy_deferrals": "a busy medium no longer defers a frame by whole "
                              "100 ms epochs or drops it; see _channel_access",
    }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SimSettings":
        s = cfg.get("simulation", {})
        removed = sorted(k for k in s if k in cls.REMOVED_KEYS)
        if removed:
            raise ValueError(f"simulation keys {removed} were removed: "
                             + "; ".join(cls.REMOVED_KEYS[k] for k in removed))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in s.items() if k in known})


@dataclass
class RunResult:
    """Raw per-run output. :mod:`analysis.metrics` turns this into metrics."""

    # Per-vehicle
    informed_step: np.ndarray      # [N] int32, NO_STEP if never informed
    hops: np.ndarray               # [N] int16
    informed_by: np.ndarray        # [N] int32
    tx_count: np.ndarray           # [N] int32

    # Message
    origin_index: int
    origin_step: int

    # Channel accounting
    n_transmissions: int
    n_rx_attempts: int             # (tx, in-nominal-range receiver) pairs
    n_rx_success: int
    n_fail_sensitivity: int
    n_fail_sinr: int               # interference-caused: the collision losses
    n_fail_beacon: int             # lost to background CAM/BSM load
    n_backoff_ties: int            # pairs of relays that drew the same slot
    n_hidden_overlaps: int         # pairs of mutually hidden overlapping relays
    n_busy_deferrals: int
    n_dropped_ttl: int

    # Event log (for figures and the Phase 5 attention heatmap)
    transmissions: list[dict[str, Any]] = field(default_factory=list)

    #: Per-vehicle share of the blame for receptions lost to interference.
    #: A reception that failed the SINR test is blamed on the concurrent
    #: transmitters that actually overlapped it, split equally between
    #: them, so the total equals n_fail_sinr. Needed by the Phase 5 reward:
    #: charging every transmitter the NETWORK-MEAN collision rate instead
    #: carries no per-vehicle signal and silently multiplies the
    #: calibrated transmission cost.
    collisions_caused: np.ndarray | None = None

    #: How many times each action type was chosen. This is what shows *why* a
    #: policy behaved as it did -- e.g. whether DV-CAST's store-carry-forward
    #: branch ever actually fired, or whether the network was connected enough
    #: that it degenerated into slotted persistence.
    action_counts: dict[str, int] = field(default_factory=dict)

    # Context
    trace: Trace | None = None
    hazard: Hazard | None = None
    risk: RiskField | None = None
    phy: PhyModel | None = None
    mac: MacModel | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_informed(self) -> int:
        return int((self.informed_step >= 0).sum())


class DisseminationEngine:
    """Runs one (trace, hazard, policy) triple to completion."""

    def __init__(
        self,
        trace: Trace,
        phy: PhyModel,
        mac: MacModel,
        risk: RiskField,
        hazard: Hazard,
        policy: Policy,
        seeds: SeedBundle,
        settings: SimSettings | None = None,
    ) -> None:
        self.trace = trace
        self.phy = phy
        self.mac = mac
        self.risk = risk
        self.hazard = hazard
        self.policy = policy
        self.seeds = seeds
        self.cfg = settings or SimSettings()

        self.N = trace.n_vehicles
        self.dt = trace.dt
        # Median-link range at receiver sensitivity: the nominal broadcast
        # range, used as the PDR denominator and as the neighbourhood radius
        # policies are allowed to see.
        self.comm_range_m = phy.nominal_range_m
        self.cs_range_m = phy.range_for_power_m(mac.carrier_sense_threshold_dbm)
        # Fading can pull a link well above the median, so candidate receivers
        # are pruned at a generous margin rather than at the nominal range.
        self.rx_range_m = phy.range_for_power_m(phy.sensitivity_dbm - self.cfg.rx_prune_margin_db)

    # ------------------------------------------------------------------- run --
    def run(self) -> RunResult:
        tr, cfg = self.trace, self.cfg
        rng_mac = self.seeds.rng("mac")
        rng_fade = self.seeds.rng("fading")
        rng_pol = self.seeds.rng("policy")
        self.policy.reset(self.N, rng_pol)

        N = self.N
        holds = np.zeros(N, dtype=bool)
        informed_step = np.full(N, NO_STEP, dtype=np.int32)
        informed_by = np.full(N, NO_STEP, dtype=np.int32)
        hops = np.zeros(N, dtype=np.int16)
        dup_count = np.zeros(N, dtype=np.int32)
        max_sender_dist = np.zeros(N, dtype=np.float32)
        tx_sched = np.full(N, NO_STEP, dtype=np.int32)
        timer_sched = np.full(N, NO_STEP, dtype=np.int32)
        cancel_at_dups = np.full(N, NO_STEP, dtype=np.int32)
        forced_broadcast = np.zeros(N, dtype=bool)
        tx_count = np.zeros(N, dtype=np.int32)
        collisions_caused = np.zeros(N, dtype=np.float64)

        origin_index = NO_STEP
        origin_step = NO_STEP
        counters = dict(
            n_transmissions=0, n_rx_attempts=0, n_rx_success=0, n_fail_sensitivity=0,
            n_fail_sinr=0, n_fail_beacon=0, n_backoff_ties=0, n_hidden_overlaps=0,
            n_busy_deferrals=0, n_dropped_ttl=0,
        )
        transmissions: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {a.value: 0 for a in ActionType}
        self._action_counts = action_counts
        notify_dups = getattr(self.policy, "wants_duplicate_callbacks", False)

        for step in range(tr.n_steps):
            t = step * self.dt
            act = tr.active[step]
            if not act.any():
                continue

            # -- 1. hazard detection / message origination -------------------
            if origin_index == NO_STEP and t >= self.hazard.onset_time_s:
                cand = self._detect(step)
                if cand is not None:
                    origin_index, origin_step = cand, step
                    holds[cand] = True
                    informed_step[cand] = step
                    hops[cand] = 0
                    self._decide_and_apply(
                        cand, step, Trigger.ORIGINATE, holds, informed_step, hops, dup_count,
                        max_sender_dist, tx_sched, timer_sched, cancel_at_dups, forced_broadcast,
                        origin_step, rng_pol, sender=None, sender_dist=0.0,
                        tx_count=tx_count,
                    )
                    logger.debug("t=%.1fs hazard detected by vehicle %d", t, cand)
            if origin_index == NO_STEP:
                continue

            age = t - origin_step * self.dt

            # -- 2. transmissions scheduled for this epoch --------------------
            pending = np.flatnonzero((tx_sched == step) & holds & act)
            if pending.size:
                # TTL / hop-limit enforcement happens before the frame goes out.
                expired = pending[(hops[pending] >= cfg.max_hops) | (age > cfg.message_ttl_s)]
                if expired.size:
                    tx_sched[expired] = NO_STEP
                    counters["n_dropped_ttl"] += int(expired.size)
                    pending = np.setdiff1d(pending, expired, assume_unique=True)

            if pending.size:
                self._channel_access(pending, step, act, counters, rng_mac)

            if pending.size:
                self._transmit(
                    pending, step, act, holds, informed_step, informed_by, hops, dup_count,
                    max_sender_dist, tx_sched, timer_sched, cancel_at_dups, forced_broadcast,
                    tx_count, counters, transmissions, origin_step, rng_mac, rng_fade, rng_pol,
                    notify_dups, collisions_caused,
                )

            # -- 5. deferred re-decisions (carry / store-carry-forward) -------
            due = np.flatnonzero((timer_sched == step) & holds & act)
            self._begin_batch()
            for i in due:
                timer_sched[i] = NO_STEP
                if tx_sched[i] != NO_STEP:
                    continue  # a transmission is already queued
                self._decide_and_apply(
                    int(i), step, Trigger.TIMER, holds, informed_step, hops, dup_count,
                    max_sender_dist, tx_sched, timer_sched, cancel_at_dups, forced_broadcast,
                    origin_step, rng_pol, sender=None,
                    sender_dist=float(max_sender_dist[i]), tx_count=tx_count,
                )
            self._flush_batch(step, tx_sched, timer_sched, cancel_at_dups, forced_broadcast)

        return RunResult(
            informed_step=informed_step, hops=hops, informed_by=informed_by,
            tx_count=tx_count, origin_index=int(origin_index), origin_step=int(origin_step),
            transmissions=transmissions, action_counts=action_counts,
            collisions_caused=collisions_caused,
            trace=tr, hazard=self.hazard, risk=self.risk,
            phy=self.phy, mac=self.mac,
            meta={
                "policy": self.policy.describe(),
                "comm_range_m": round(self.comm_range_m, 1),
                "cs_range_m": round(self.cs_range_m, 1),
                "rx_prune_range_m": round(self.rx_range_m, 1),
            },
            **counters,
        )

    # -------------------------------------------------------------- detection --
    def _detect(self, step: int) -> int | None:
        """First active vehicle inside the detection envelope of the hazard."""
        tr = self.trace
        f = self.risk.features_at(tr, step, self.hazard)
        near = (
            tr.active[step]
            & (f["distance_m"] <= self.hazard.detection_range_m)
            & np.isin(f["state"], (APPROACHING, INSIDE))
        )
        # Only gate on carriageway sign when the scenario has one. Grid traces
        # carry direction == 0 for every vehicle, and their bearing-alignment
        # gate is already folded into the geometry state above; applying a sign
        # comparison here as well would mean no vehicle ever detects a
        # directional hazard in the grid.
        if self.hazard.affected_direction != 0 and np.any(tr.direction != 0):
            near &= tr.direction == self.hazard.affected_direction
        idx = np.flatnonzero(near)
        if idx.size == 0:
            return None
        # The vehicle closest to the hazard detects it first.
        return int(idx[np.argmin(f["distance_m"][idx])])

    # ---------------------------------------------------------- channel access --
    def _channel_access(
        self, pending: np.ndarray, step: int, act: np.ndarray,
        counters: dict[str, int], rng: np.random.Generator,
    ) -> None:
        """CCA. A busy medium delays a frame within its epoch; it is never dropped.

        802.11p defers in 13 us backoff slots behind background beacons that
        last well under a millisecond, so a frame that finds the medium busy
        still goes out long before a 100 ms epoch ends. Contention with other
        relays and beacon overlap at the receiver are modelled separately.

        This used to defer a busy frame by a whole epoch and discard it after
        five busy draws, uncounted. Local busy probability reaches 0.81 around
        an urban d=80 originator, so 0.81^6 = 28% of episodes never transmitted
        at all (9 of 32 training seeds under flooding, for every policy), and
        dense cells silently lost 6-9% of flooding's relay frames and added
        ~100-200 ms of spurious latency. The busy draw is kept only to count
        how many frames had to defer.
        """
        n_nb = self._neighbour_counts(step, act, pending, self.cs_range_m)
        p_busy = self.mac.channel_busy_probability(n_nb)
        busy = rng.random(pending.size) < p_busy
        counters["n_busy_deferrals"] += int(busy.sum())

    # ------------------------------------------------------------- transmission --
    def _transmit(
        self, tx: np.ndarray, step: int, act: np.ndarray, holds, informed_step, informed_by,
        hops, dup_count, max_sender_dist, tx_sched, timer_sched, cancel_at_dups,
        forced_broadcast, tx_count, counters, transmissions, origin_step,
        rng_mac, rng_fade, rng_pol, notify_dups: bool,
        collisions_caused: np.ndarray | None = None,
    ) -> None:
        tr, phy, mac = self.trace, self.phy, self.mac
        n_tx = tx.size
        tx_sched[tx] = NO_STEP
        tx_count[tx] += 1
        counters["n_transmissions"] += n_tx
        for i in tx:
            self.policy.on_transmit(int(i), step)

        rx_idx = np.flatnonzero(act)
        pos = np.stack([tr.x[step], tr.y[step]], axis=1)
        pos_tx, pos_rx = pos[tx], pos[rx_idx]

        # --- which concurrent frames actually overlap in time ---------------
        overlap = self._overlap_matrix(pos_tx, n_tx, counters, rng_mac)

        # --- received power for every (tx, rx) link -------------------------
        d = np.linalg.norm(pos_tx[:, None, :] - pos_rx[None, :, :], axis=2)
        d = np.maximum(d, phy.reference_distance_m)
        tx_ids = np.broadcast_to(tx[:, None], d.shape)
        rx_ids = np.broadcast_to(rx_idx[None, :], d.shape)
        p_dbm = phy.rx_power_dbm(tx_ids, rx_ids, d, rng_fade)
        # Buildings: links that do not share a street pay the detour around the
        # corner plus a diffraction loss. Zero for open-road scenarios.
        p_dbm = p_dbm - phy.nlos_excess_db(pos_tx, pos_rx)
        # Out-of-range links are not simulated at all.
        in_prune = d <= self.rx_range_m
        p_lin = np.where(in_prune, 10.0 ** (p_dbm / 10.0), 0.0)

        # --- SINR per link --------------------------------------------------
        interference = overlap.astype(float) @ p_lin          # [n_tx, n_rx]
        sinr_db = p_dbm - 10.0 * np.log10(phy.noise_mw + interference)

        is_tx = np.isin(rx_idx, tx)                            # half-duplex
        above_sens = (p_dbm >= phy.sensitivity_dbm) & in_prune
        above_sinr = sinr_db >= phy.sinr_threshold_db
        if not phy.capture_enabled:
            above_sinr &= interference <= 0.0

        # Background CAM/BSM load, evaluated at the receiver.
        nb_rx = self._neighbour_counts(step, act, rx_idx, self.cs_range_m)
        survive_p = mac.beacon_survival_probability(nb_rx)     # [n_rx]
        beacon_ok = rng_mac.random(d.shape) < survive_p[None, :]

        decoded = above_sens & above_sinr & beacon_ok & ~is_tx[None, :]

        # --- accounting, restricted to the nominal-range denominator --------
        in_nominal = (d <= self.comm_range_m) & ~is_tx[None, :]
        in_nominal[np.arange(n_tx), np.searchsorted(rx_idx, tx)] = False  # exclude self
        counters["n_rx_attempts"] += int(in_nominal.sum())
        counters["n_rx_success"] += int((decoded & in_nominal).sum())
        counters["n_fail_sensitivity"] += int((in_nominal & ~above_sens).sum())
        counters["n_fail_sinr"] += int((in_nominal & above_sens & ~above_sinr).sum())
        counters["n_fail_beacon"] += int(
            (in_nominal & above_sens & above_sinr & ~beacon_ok).sum()
        )

        # Blame each interference loss on the transmitters that overlapped
        # it, split equally, so the attributed total equals n_fail_sinr.
        if collisions_caused is not None and n_tx > 1:
            lost = (in_nominal & above_sens & ~above_sinr).sum(axis=1).astype(float)
            n_interferers = overlap.sum(axis=1).astype(float)
            share = np.divide(lost, np.maximum(n_interferers, 1.0),
                              out=np.zeros_like(lost), where=n_interferers > 0)
            np.add.at(collisions_caused, tx, overlap.T.astype(float) @ share)

        # --- deliver ---------------------------------------------------------
        any_decoded = decoded.any(axis=0)
        self._begin_batch()
        for r_local in np.flatnonzero(any_decoded):
            r = int(rx_idx[r_local])
            senders = np.flatnonzero(decoded[:, r_local])
            # Prefer the lowest-hop sender; ties broken by the nearest.
            best = senders[np.lexsort((d[senders, r_local], hops[tx[senders]]))[0]]
            s = int(tx[best])
            dist = float(d[best, r_local])
            max_sender_dist[r] = max(float(max_sender_dist[r]), dist)

            if not holds[r]:
                holds[r] = True
                informed_step[r] = step
                informed_by[r] = s
                hops[r] = hops[s] + 1
                # A designation is an input to the decision, not a bypass of
                # it: a designated relay still runs its policy, which is what
                # lets greedy forwarding designate the *next* hop in turn.
                designated = bool(forced_broadcast[r])
                forced_broadcast[r] = False
                self._decide_and_apply(
                    r, step, Trigger.RECEIVE, holds, informed_step, hops, dup_count,
                    max_sender_dist, tx_sched, timer_sched, cancel_at_dups,
                    forced_broadcast, origin_step, rng_pol, sender=s, sender_dist=dist,
                    designated=designated, tx_count=tx_count,
                )
            else:
                dup_count[r] += 1
                # Counter-based and p-persistent schemes cancel a pending
                # rebroadcast once enough duplicates have been overheard.
                if (
                    cancel_at_dups[r] != NO_STEP
                    and dup_count[r] >= cancel_at_dups[r]
                    and tx_sched[r] > step
                ):
                    tx_sched[r] = NO_STEP
                    cancel_at_dups[r] = NO_STEP
                elif notify_dups and tx_sched[r] == NO_STEP:
                    # Note a pending *timer* does not block this callback: a
                    # vehicle carrying the message across a gap is exactly the
                    # one that needs to know the gap has closed. A pending
                    # *transmission* does block it, since `cancel_on_duplicates`
                    # already handles that case.
                    self._decide_and_apply(
                        r, step, Trigger.RECEIVE, holds, informed_step, hops, dup_count,
                        max_sender_dist, tx_sched, timer_sched, cancel_at_dups,
                        forced_broadcast, origin_step, rng_pol, sender=s, sender_dist=dist,
                        tx_count=tx_count,
                    )

        self._flush_batch(step, tx_sched, timer_sched, cancel_at_dups, forced_broadcast)

        if self.cfg.record_transmissions:
            for k, i in enumerate(tx):
                transmissions.append({
                    "step": step,
                    "time_s": round(step * self.dt, 3),
                    "vehicle": int(i),
                    "x": float(tr.x[step, i]),
                    "y": float(tr.y[step, i]),
                    "hop": int(hops[i]),
                    "n_in_range": int(in_nominal[k].sum()),
                    "n_decoded": int((decoded[k] & in_nominal[k]).sum()),
                    "n_new": int((decoded[k] & in_nominal[k]).sum()),
                })

    # --------------------------------------------------------------- overlap --
    def _overlap_matrix(
        self, pos_tx: np.ndarray, n_tx: int, counters: dict[str, int], rng: np.random.Generator
    ) -> np.ndarray:
        """Which concurrent transmissions overlap in time (see :mod:`sim.mac`)."""
        overlap = np.zeros((n_tx, n_tx), dtype=bool)
        if n_tx < 2:
            return overlap

        dtx = np.linalg.norm(pos_tx[:, None, :] - pos_tx[None, :, :], axis=2)
        dtx = np.maximum(dtx, self.phy.reference_distance_m)
        sense = (
            self.phy.median_rx_power_dbm(dtx) - self.phy.nlos_excess_db(pos_tx, pos_tx)
        ) >= self.mac.carrier_sense_threshold_dbm
        np.fill_diagonal(sense, False)

        slots = self.mac.draw_backoff_slots(n_tx, rng)
        tie = slots[:, None] == slots[None, :]
        np.fill_diagonal(tie, False)

        # Mutually hidden relays never defer for one another; they overlap when
        # their frames land in each other's vulnerable window.
        p_hidden = self.mac.hidden_overlap_probability(self.dt)
        draw = rng.random((n_tx, n_tx))
        draw = np.triu(draw, 1)
        draw = draw + draw.T
        hidden_overlap = (~sense) & (draw < p_hidden)
        np.fill_diagonal(hidden_overlap, False)

        overlap = (sense & tie) | hidden_overlap
        counters["n_backoff_ties"] += int(np.triu(sense & tie, 1).sum())
        counters["n_hidden_overlaps"] += int(np.triu(hidden_overlap, 1).sum())
        return overlap

    # ------------------------------------------------------------- neighbours --
    def _neighbour_counts(
        self, step: int, act: np.ndarray, of: np.ndarray, radius: float
    ) -> np.ndarray:
        """Active neighbours within ``radius`` of each vehicle in ``of``."""
        tr = self.trace
        rx_idx = np.flatnonzero(act)
        pos = np.stack([tr.x[step, rx_idx], tr.y[step, rx_idx]], axis=1)
        q = np.stack([tr.x[step, of], tr.y[step, of]], axis=1)
        d = np.linalg.norm(q[:, None, :] - pos[None, :, :], axis=2)
        return (d <= radius).sum(axis=1) - 1  # exclude self

    # -------------------------------------------------------------- decisions --
    def _decide_and_apply(
        self, i: int, step: int, trigger: Trigger, holds, informed_step, hops, dup_count,
        max_sender_dist, tx_sched, timer_sched, cancel_at_dups, forced_broadcast,
        origin_step, rng_pol, *, sender: int | None, sender_dist: float,
        designated: bool = False, tx_count: np.ndarray | None = None,
    ) -> None:
        ctx = self._context(
            i, step, trigger, hops, dup_count, max_sender_dist, informed_step,
            origin_step, rng_pol, sender, sender_dist, designated,
            0 if tx_count is None else int(tx_count[i]),
        )
        pending = getattr(self, "_pending", None)
        if pending is not None:
            # Batched: the context is built NOW, so it sees exactly the state a
            # sequential decision would (e.g. which neighbours are already
            # informed this epoch); evaluation and application are deferred.
            pending.append((i, ctx))
            return
        action = self.policy.decide(ctx)
        self._apply(action, i, step, tx_sched, timer_sched, cancel_at_dups, forced_broadcast)

    def _begin_batch(self) -> None:
        """Start collecting decisions, if the policy evaluates them in batches.

        Deferring evaluation within one loop is exact for a policy whose
        decisions depend only on its own context: applying an action changes
        only the decider's own schedule, except a RELAY designation, which
        reaches a later receiver solely through ``ctx.was_designated``. A
        policy that reads ``was_designated`` (greedy forwarding) must not set
        ``batch_decisions``; the learned agent's graph never reads it.
        """
        self._pending = [] if getattr(self.policy, "batch_decisions", False) else None

    def _flush_batch(self, step: int, tx_sched, timer_sched, cancel_at_dups,
                     forced_broadcast) -> None:
        pending = getattr(self, "_pending", None)
        self._pending = None
        if not pending:
            return
        actions = self.policy.decide_batch([ctx for _, ctx in pending])
        for (i, _), action in zip(pending, actions):
            self._apply(action, i, step, tx_sched, timer_sched, cancel_at_dups, forced_broadcast)

    def _context(
        self, i: int, step: int, trigger: Trigger, hops, dup_count, max_sender_dist,
        informed_step, origin_step, rng_pol, sender: int | None, sender_dist: float,
        designated: bool = False, own_tx_count: int = 0,
    ) -> DecisionContext:
        tr = self.trace
        act = tr.active[step]
        feats = self.risk.features_at(tr, step, self.hazard)

        rx_idx = np.flatnonzero(act)
        dx = tr.x[step, rx_idx] - tr.x[step, i]
        dy = tr.y[step, rx_idx] - tr.y[step, i]
        dist = np.hypot(dx, dy)
        near = (dist <= self.comm_range_m) & (rx_idx != i)

        if self.phy.has_buildings:
            # A neighbour behind a building is not a neighbour. The neighbour
            # table comes from overheard beacons, so it can only contain
            # vehicles whose beacons actually arrive -- which means the NLOS
            # excess loss has to be applied here too, not only to the
            # dissemination frames.
            #
            # Without this, policies in urban_nlos "see" vehicles across a
            # street corner that they cannot reach: DV-CAST concludes it is
            # well connected and never carries, and greedy forwarding
            # designates relays that never hear the designation.
            self_pos = np.array([[tr.x[step, i], tr.y[step, i]]])
            nb_pos = np.stack([tr.x[step, rx_idx], tr.y[step, rx_idx]], axis=1)
            budget = (
                self.phy.median_rx_power_dbm(np.maximum(dist, self.phy.reference_distance_m))
                - self.phy.nlos_excess_db(self_pos, nb_pos)[0]
            )
            near &= budget >= self.phy.sensitivity_dbm

        nb = rx_idx[near]

        sender_dx = sender_dy = 0.0
        if sender is not None:
            sender_dx = float(tr.x[step, sender] - tr.x[step, i])
            sender_dy = float(tr.y[step, sender] - tr.y[step, i])

        return DecisionContext(
            step=step, time_s=step * self.dt, dt=self.dt, index=i, trigger=trigger,
            hop_count=int(hops[i]), duplicate_count=int(dup_count[i]),
            own_tx_count=int(own_tx_count),
            sender_index=sender, sender_distance_m=sender_dist,
            max_sender_distance_m=float(max_sender_dist[i]),
            age_s=(step - origin_step) * self.dt if origin_step >= 0 else 0.0,
            x=float(tr.x[step, i]), y=float(tr.y[step, i]),
            speed_ms=float(np.hypot(tr.vx[step, i], tr.vy[step, i])),
            heading=float(tr.heading[step, i]), direction=int(tr.direction[i]),
            relevance=float(feats["relevance"][i]), eta_s=float(feats["eta_s"][i]),
            vclass=str(tr.vclass[i]),
            neighbours=nb, neighbour_distances=dist[near],
            neighbour_dx=dx[near], neighbour_dy=dy[near],
            neighbour_vx=tr.vx[step, nb], neighbour_vy=tr.vy[step, nb],
            neighbour_relevance=feats["relevance"][nb],
            # A vehicle only knows a neighbour is informed if it heard it send.
            neighbour_informed=(informed_step[nb] >= 0) & (informed_step[nb] <= step),
            comm_range_m=self.comm_range_m, rng=rng_pol,
            sender_dx=sender_dx, sender_dy=sender_dy, was_designated=designated,
            extras={"eta_all": feats["eta_s"], "relevance_all": feats["relevance"]},
        )

    def _apply(
        self, action: Action, i: int, step: int, tx_sched, timer_sched, cancel_at_dups,
        forced_broadcast,
    ) -> None:
        """Apply a policy action. Rebroadcasts always cost >= one epoch."""
        counts = getattr(self, "_action_counts", None)
        if counts is not None:
            counts[action.kind.value] = counts.get(action.kind.value, 0) + 1
        if action.cancel_on_duplicates is not None:
            cancel_at_dups[i] = int(action.cancel_on_duplicates)

        if action.kind is ActionType.SUPPRESS:
            # Suppress means "done with this message": it clears a pending
            # carry timer as well as a pending transmission, so a carrier that
            # learns the gap has closed actually stops carrying.
            tx_sched[i] = NO_STEP
            timer_sched[i] = NO_STEP
        elif action.kind is ActionType.BROADCAST:
            tx_sched[i] = step + 1
        elif action.kind is ActionType.DEFER:
            tx_sched[i] = step + max(1, action.delay_steps)
        elif action.kind is ActionType.CARRY:
            timer_sched[i] = step + max(1, action.delay_steps or self.cfg.carry_recheck_steps)
        elif action.kind is ActionType.RELAY:
            # A designating relay transmits itself; the designation rides in the
            # frame, so it only takes effect if the designated vehicle actually
            # decodes it. Nothing guarantees that -- broadcast is unacknowledged
            # -- which is exactly why greedy relay selection needs a fallback.
            tx_sched[i] = step + 1
            for r in action.relay_indices:
                forced_broadcast[int(r)] = True
