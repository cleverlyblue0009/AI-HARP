"""Phase 4 tests: every baseline's suppression logic.

Each policy's defining behaviour is pinned here. The invariant common to all of
them is that the *engine*, not the policy, owns timing -- a policy can never
sneak a zero-latency rebroadcast past the 100 ms turnaround.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.base import Action, ActionType, DecisionContext, Policy, Trigger
from agents.counter import DistanceCounterPolicy
from agents.dvcast import DISCONNECTED, SPARSELY_CONNECTED, WELL_CONNECTED, DvCast
from agents.flooding import BlindFlooding
from agents.greedy import GreedyFarthestRelay
from agents.persistence import (
    ProbabilisticPPersistence,
    SlottedPersistence,
    WeightedPPersistence,
    distance_slot,
)
from agents.registry import BASELINE_POLICIES, available_policies, build_policy


def make_ctx(**kw) -> DecisionContext:
    """A vehicle 300 m from the sender with three neighbours ahead of it."""
    n_dx = np.array([-100.0, 200.0, 450.0])
    base = dict(
        step=10, time_s=1.0, dt=0.1, index=3, trigger=Trigger.RECEIVE,
        hop_count=1, duplicate_count=0, own_tx_count=0, sender_index=2,
        sender_distance_m=300.0,
        max_sender_distance_m=300.0, age_s=0.4,
        x=1000.0, y=1.75, speed_ms=25.0, heading=0.0, direction=1,
        relevance=0.8, eta_s=40.0, vclass="car",
        neighbours=np.array([1, 2, 5]),
        neighbour_distances=np.abs(n_dx),
        neighbour_dx=n_dx, neighbour_dy=np.zeros(3),
        neighbour_vx=np.array([25.0, 24.0, 26.0]), neighbour_vy=np.zeros(3),
        neighbour_relevance=np.array([0.7, 0.9, 0.2]),
        neighbour_informed=np.array([False, True, False]),
        comm_range_m=560.0, rng=np.random.default_rng(0),
        # Sender lies behind us (-x), so the message propagates towards +x.
        sender_dx=-300.0, sender_dy=0.0, was_designated=False,
    )
    base.update(kw)
    return DecisionContext(**base)


def empty_ctx(**kw) -> DecisionContext:
    """A vehicle with no neighbours at all."""
    return make_ctx(
        neighbours=np.array([], dtype=int),
        neighbour_distances=np.array([]),
        neighbour_dx=np.array([]), neighbour_dy=np.array([]),
        neighbour_vx=np.array([]), neighbour_vy=np.array([]),
        neighbour_relevance=np.array([]),
        neighbour_informed=np.array([], dtype=bool),
        **kw,
    )


# ============================================================ 1. flooding ====
def test_flooding_rebroadcasts_on_first_reception():
    assert BlindFlooding().decide(make_ctx()).kind is ActionType.BROADCAST


def test_flooding_rebroadcasts_on_origination():
    assert BlindFlooding().decide(make_ctx(trigger=Trigger.ORIGINATE)).kind \
        is ActionType.BROADCAST


def test_flooding_never_suppresses_regardless_of_context():
    """Blind flooding is blind: no neighbour count, distance or relevance
    changes its mind. This is what makes it the broadcast-storm baseline."""
    p = BlindFlooding()
    for rel in (0.0, 0.5, 1.0):
        for dist in (5.0, 550.0):
            assert p.decide(make_ctx(relevance=rel, sender_distance_m=dist)).kind \
                is ActionType.BROADCAST
    assert p.decide(empty_ctx()).kind is ActionType.BROADCAST


def test_flooding_does_not_ask_for_duplicate_callbacks():
    """The engine calls back on duplicates only when asked; flooding must not
    ask, otherwise a vehicle would rebroadcast once per duplicate heard."""
    assert BlindFlooding.wants_duplicate_callbacks is False


# ================================================ 2. probabilistic p-persist ==
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
def test_p_persistence_rebroadcast_rate_matches_p(p):
    pol = ProbabilisticPPersistence(p=p)
    rng = np.random.default_rng(0)
    n = 20_000
    hits = sum(
        pol.decide(make_ctx(rng=rng)).kind is ActionType.BROADCAST for _ in range(n)
    )
    assert hits / n == pytest.approx(p, abs=0.015)


def test_p_persistence_originator_always_speaks():
    """A suppressed originator means the hazard is detected and never announced."""
    pol = ProbabilisticPPersistence(p=0.0)
    assert pol.decide(make_ctx(trigger=Trigger.ORIGINATE)).kind is ActionType.BROADCAST
    assert pol.decide(make_ctx(trigger=Trigger.RECEIVE)).kind is ActionType.SUPPRESS


def test_p_persistence_ignores_distance():
    """The defining weakness: p is fixed, so position is irrelevant."""
    pol = ProbabilisticPPersistence(p=0.5)
    rng = np.random.default_rng(1)
    near = sum(pol.decide(make_ctx(sender_distance_m=20.0, rng=rng)).kind
               is ActionType.BROADCAST for _ in range(4000))
    far = sum(pol.decide(make_ctx(sender_distance_m=540.0, rng=rng)).kind
              is ActionType.BROADCAST for _ in range(4000))
    assert abs(near - far) / 4000 < 0.03


def test_p_persistence_rejects_invalid_p():
    with pytest.raises(ValueError):
        ProbabilisticPPersistence(p=1.4)


# ==================================================== 3. slotted 1-persistence =
def test_distance_slot_orders_farthest_first():
    """The farthest band speaks first; the nearest waits longest."""
    R, n = 500.0, 5
    assert distance_slot(500.0, R, n) == 0
    assert distance_slot(0.0, R, n) == n - 1
    slots = [distance_slot(d, R, n) for d in np.linspace(0, R, 50)]
    assert slots == sorted(slots, reverse=True)


def test_distance_slot_is_clamped_to_the_window():
    for d in (-50.0, 0.0, 250.0, 500.0, 5000.0):
        assert 0 <= distance_slot(d, 500.0, 5) <= 4


def test_slotted_defers_with_a_distance_ranked_delay():
    pol = SlottedPersistence(p=1.0, n_slots=5, slot_epochs=1)
    far = pol.decide(make_ctx(sender_distance_m=550.0))
    near = pol.decide(make_ctx(sender_distance_m=30.0))
    assert far.kind is ActionType.DEFER and near.kind is ActionType.DEFER
    assert far.delay_steps < near.delay_steps


def test_slotted_never_defers_by_zero_epochs():
    """Even slot 0 must respect the one-epoch turnaround."""
    pol = SlottedPersistence()
    assert pol.decide(make_ctx(sender_distance_m=560.0)).delay_steps >= 1


def test_slotted_arms_duplicate_cancellation():
    """Suppression comes from cancelling on overheard duplicates, not chance."""
    a = SlottedPersistence(cancel_on_duplicates=1).decide(make_ctx())
    assert a.cancel_on_duplicates == 1


def test_slotted_1p_is_deterministic():
    pol = SlottedPersistence(p=1.0)
    delays = {pol.decide(make_ctx(rng=np.random.default_rng(s))).delay_steps
              for s in range(20)}
    assert len(delays) == 1


def test_slotted_p_below_one_suppresses_some():
    pol = SlottedPersistence(p=0.5)
    rng = np.random.default_rng(2)
    kinds = [pol.decide(make_ctx(rng=rng)).kind for _ in range(2000)]
    assert ActionType.SUPPRESS in kinds and ActionType.DEFER in kinds


def test_slot_epochs_scales_the_wait():
    a = SlottedPersistence(slot_epochs=1).decide(make_ctx(sender_distance_m=100.0))
    b = SlottedPersistence(slot_epochs=3).decide(make_ctx(sender_distance_m=100.0))
    assert b.delay_steps > a.delay_steps


# ================================================= 4. weighted p-persistence ==
def test_weighted_p_probability_is_distance_over_range():
    pol = WeightedPPersistence()
    assert pol.rebroadcast_probability(make_ctx(sender_distance_m=280.0,
                                                comm_range_m=560.0)) == pytest.approx(0.5)
    assert pol.rebroadcast_probability(make_ctx(sender_distance_m=560.0,
                                                comm_range_m=560.0)) == pytest.approx(1.0)
    assert pol.rebroadcast_probability(make_ctx(sender_distance_m=0.0,
                                                comm_range_m=560.0)) == pytest.approx(0.0)


def test_weighted_p_favours_distant_receivers():
    """The whole point: far receivers add new coverage, near ones do not."""
    pol = WeightedPPersistence()
    rng = np.random.default_rng(3)
    far = sum(pol.decide(make_ctx(sender_distance_m=500.0, rng=rng)).kind
              is ActionType.BROADCAST for _ in range(4000))
    near = sum(pol.decide(make_ctx(sender_distance_m=60.0, rng=rng)).kind
               is ActionType.BROADCAST for _ in range(4000))
    assert far > 5 * near


def test_weighted_p_probability_is_clamped_beyond_range():
    pol = WeightedPPersistence()
    assert pol.rebroadcast_probability(make_ctx(sender_distance_m=5000.0)) == 1.0


# ================================================ 5. distance-based counter ===
def test_counter_suppresses_when_the_sender_was_too_close():
    """No worthwhile new coverage -> do not bother deferring at all."""
    pol = DistanceCounterPolicy(min_distance_fraction=0.35)
    assert pol.decide(make_ctx(sender_distance_m=50.0)).kind is ActionType.SUPPRESS


def test_counter_defers_when_the_sender_was_far():
    pol = DistanceCounterPolicy(min_distance_fraction=0.35)
    a = pol.decide(make_ctx(sender_distance_m=500.0))
    assert a.kind is ActionType.DEFER


def test_counter_threshold_maps_to_duplicates():
    """A threshold of C copies received is C - 1 duplicates overheard."""
    a = DistanceCounterPolicy(counter_threshold=3).decide(make_ctx(sender_distance_m=500.0))
    assert a.cancel_on_duplicates == 2


def test_counter_assessment_delay_is_randomised():
    """Without a random delay every receiver decides at once, nobody has heard
    anybody, and the counter is always zero."""
    pol = DistanceCounterPolicy(max_delay_epochs=5)
    rng = np.random.default_rng(4)
    delays = {pol.decide(make_ctx(sender_distance_m=500.0, rng=rng)).delay_steps
              for _ in range(200)}
    assert len(delays) > 1
    assert min(delays) >= 1


def test_counter_delay_stays_within_the_configured_window():
    pol = DistanceCounterPolicy(max_delay_epochs=5)
    rng = np.random.default_rng(5)
    for _ in range(200):
        d = pol.decide(make_ctx(sender_distance_m=500.0, rng=rng)).delay_steps
        assert 1 <= d <= 5


def test_counter_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        DistanceCounterPolicy(counter_threshold=0)


# ==================================================== 6. greedy farthest relay =
def test_greedy_designates_the_farthest_neighbour_ahead():
    pol = GreedyFarthestRelay()
    a = pol.decide(make_ctx(was_designated=True))
    assert a.kind is ActionType.RELAY
    # Neighbour 5 is at +450 m (progress 450); neighbour 1 at -100 m is behind.
    assert a.relay_indices == (5,)


def test_greedy_never_designates_a_neighbour_behind_the_sender():
    """Relaying backwards re-covers ground the sender already reached."""
    pol = GreedyFarthestRelay()
    ctx = make_ctx(
        was_designated=True,
        neighbours=np.array([7, 8]),
        neighbour_dx=np.array([-400.0, -50.0]),
        neighbour_dy=np.zeros(2),
        neighbour_distances=np.array([400.0, 50.0]),
        neighbour_vx=np.zeros(2), neighbour_vy=np.zeros(2),
        neighbour_relevance=np.zeros(2),
        neighbour_informed=np.zeros(2, dtype=bool),
    )
    # Everything lies back towards the sender, so the chain ends here.
    assert pol.decide(ctx).kind is ActionType.BROADCAST


def test_greedy_originator_seeds_both_directions():
    """A hazard message has to travel both ways along a two-way corridor."""
    pol = GreedyFarthestRelay(seed_both_directions=True)
    a = pol.decide(make_ctx(trigger=Trigger.ORIGINATE, sender_index=None,
                            sender_dx=0.0, sender_dy=0.0))
    assert a.kind is ActionType.RELAY
    assert len(a.relay_indices) == 2
    assert set(a.relay_indices) == {1, 5}   # farthest each side


def test_greedy_non_designated_receiver_stays_silent_without_fallback():
    pol = GreedyFarthestRelay(fallback_enabled=False)
    assert pol.decide(make_ctx(was_designated=False)).kind is ActionType.SUPPRESS


def test_greedy_fallback_waits_past_the_designated_relays_turn():
    """Implicit-ACK recovery: a lost designation must not kill the chain."""
    pol = GreedyFarthestRelay(fallback_enabled=True, fallback_epochs=3)
    a = pol.decide(make_ctx(was_designated=False))
    assert a.kind is ActionType.DEFER
    assert a.delay_steps >= 3
    assert a.cancel_on_duplicates == 1


def test_greedy_fallback_is_distance_ranked():
    pol = GreedyFarthestRelay(fallback_enabled=True)
    far = pol.decide(make_ctx(sender_distance_m=550.0)).delay_steps
    near = pol.decide(make_ctx(sender_distance_m=30.0)).delay_steps
    assert far < near


def test_greedy_broadcasts_when_it_has_no_neighbours():
    pol = GreedyFarthestRelay()
    assert pol.decide(empty_ctx(was_designated=True)).kind is ActionType.BROADCAST


# ========================================================= 7. DV-CAST / SCF ===
def test_dvcast_classifies_neighbours_ahead_as_well_connected():
    assert DvCast().classify(make_ctx()) == WELL_CONNECTED


def test_dvcast_classifies_no_neighbours_as_disconnected():
    assert DvCast().classify(empty_ctx()) == DISCONNECTED


def test_dvcast_classifies_only_oncoming_traffic_as_sparsely_connected():
    ctx = make_ctx(
        neighbours=np.array([9]),
        neighbour_dx=np.array([-200.0]), neighbour_dy=np.array([0.0]),
        neighbour_distances=np.array([200.0]),
        neighbour_vx=np.array([-22.0]), neighbour_vy=np.array([0.0]),
        neighbour_relevance=np.array([0.1]),
        neighbour_informed=np.array([False]),
    )
    assert DvCast().classify(ctx) == SPARSELY_CONNECTED


def test_dvcast_carries_when_disconnected():
    """The third option no other baseline has: hold it and keep driving."""
    a = DvCast(carry_recheck_epochs=10).decide(empty_ctx())
    assert a.kind is ActionType.CARRY
    assert a.delay_steps == 10


def test_dvcast_suppresses_via_slots_when_well_connected():
    a = DvCast().decide(make_ctx())
    assert a.kind is ActionType.DEFER
    assert a.cancel_on_duplicates == 1


def test_dvcast_broadcasts_immediately_to_oncoming_carriers():
    ctx = make_ctx(
        neighbours=np.array([9]),
        neighbour_dx=np.array([-200.0]), neighbour_dy=np.array([0.0]),
        neighbour_distances=np.array([200.0]),
        neighbour_vx=np.array([-22.0]), neighbour_vy=np.array([0.0]),
        neighbour_relevance=np.array([0.1]),
        neighbour_informed=np.array([False]),
    )
    assert DvCast().decide(ctx).kind is ActionType.BROADCAST


def test_dvcast_sends_on_timer_once_the_gap_has_closed():
    """A carried message whose neighbourhood has reconnected goes out now --
    the waiting is already paid for."""
    a = DvCast().decide(make_ctx(trigger=Trigger.TIMER, sender_index=None,
                                 sender_dx=0.0, sender_dy=0.0))
    assert a.kind is ActionType.BROADCAST


def test_dvcast_keeps_carrying_while_still_disconnected():
    a = DvCast().decide(empty_ctx(trigger=Trigger.TIMER, sender_index=None,
                                  sender_dx=0.0, sender_dy=0.0))
    assert a.kind is ActionType.CARRY


def test_dvcast_wants_duplicate_callbacks():
    """A carrier must learn when the gap it was carrying across has closed."""
    assert DvCast.wants_duplicate_callbacks is True


def test_dvcast_never_relays_the_same_message_twice():
    """Regression: DV-CAST asks to be re-invoked on duplicates, so without an
    'already relayed' check it re-armed a rebroadcast on every duplicate heard
    and emitted ~40x more transmissions than blind flooding -- the opposite of
    what a store-carry-forward scheme is for."""
    pol = DvCast()
    for trigger in (Trigger.RECEIVE, Trigger.TIMER):
        a = pol.decide(make_ctx(trigger=trigger, own_tx_count=1))
        assert a.kind is ActionType.SUPPRESS


def test_dvcast_stops_carrying_once_it_hears_a_duplicate():
    """Overhearing the message means someone else is relaying it; the gap we
    were carrying across has closed, so drop it rather than adding traffic."""
    a = DvCast().decide(empty_ctx(trigger=Trigger.RECEIVE, duplicate_count=1))
    assert a.kind is ActionType.SUPPRESS


@pytest.mark.parametrize("name", BASELINE_POLICIES)
def test_no_baseline_relays_twice(name):
    """Whole-family invariant behind the DV-CAST regression: once a vehicle has
    transmitted, no policy may schedule another transmission of the same
    message."""
    pol = build_policy(name)
    if not pol.wants_duplicate_callbacks:
        return  # the engine never re-invokes these after the first decision
    rng = np.random.default_rng(11)
    for trigger in (Trigger.RECEIVE, Trigger.TIMER):
        a = pol.decide(make_ctx(trigger=trigger, own_tx_count=1, rng=rng))
        assert a.kind is ActionType.SUPPRESS, f"{name} re-relays on {trigger}"


def test_dvcast_can_disable_opposite_direction_carriers():
    ctx = make_ctx(
        neighbours=np.array([9]),
        neighbour_dx=np.array([-200.0]), neighbour_dy=np.array([0.0]),
        neighbour_distances=np.array([200.0]),
        neighbour_vx=np.array([-22.0]), neighbour_vy=np.array([0.0]),
        neighbour_relevance=np.array([0.1]),
        neighbour_informed=np.array([False]),
    )
    assert DvCast(use_opposite_direction_carriers=False).classify(ctx) == DISCONNECTED


# ======================================================== shared invariants ===
@pytest.mark.parametrize("name", BASELINE_POLICIES)
def test_every_baseline_originator_transmits(name):
    """No policy may silence the vehicle that detected the hazard."""
    a = build_policy(name).decide(make_ctx(trigger=Trigger.ORIGINATE, sender_index=None,
                                           sender_dx=0.0, sender_dy=0.0))
    assert a.kind in (ActionType.BROADCAST, ActionType.RELAY)


@pytest.mark.parametrize("name", BASELINE_POLICIES)
def test_no_baseline_can_transmit_in_the_reception_epoch(name):
    """Timing belongs to the engine: any deferral is at least one epoch."""
    pol = build_policy(name)
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = pol.decide(make_ctx(rng=rng))
        if a.kind in (ActionType.DEFER, ActionType.CARRY):
            assert a.delay_steps >= 1


@pytest.mark.parametrize("name", BASELINE_POLICIES)
def test_every_baseline_handles_an_empty_neighbourhood(name):
    """Sparse traffic is the paper's regime; no policy may crash with K = 0."""
    pol = build_policy(name)
    assert pol.decide(empty_ctx()).kind in set(ActionType)


@pytest.mark.parametrize("name", BASELINE_POLICIES)
def test_baseline_is_registered_and_named(name):
    pol = build_policy(name)
    assert pol.name == name
    assert name in available_policies()


def test_all_seven_baseline_families_are_present():
    """The brief specifies seven schemes; p-persistence contributes three."""
    assert len(BASELINE_POLICIES) == 9      # 7 schemes, p-persistence x3
    for family in ("flooding", "p_persistence", "slotted_1p", "weighted_p",
                   "counter_based", "greedy_farthest", "dvcast"):
        assert any(b.startswith(family) for b in BASELINE_POLICIES)


def test_policy_defaults_come_from_yaml():
    """No magic numbers in agents/*.py."""
    from agents.registry import policy_defaults

    assert policy_defaults("p_persistence_03")["p"] == 0.3
    assert policy_defaults("p_persistence_07")["p"] == 0.7
    assert policy_defaults("counter_based")["counter_threshold"] == 3


# ------------------------------------------------------------- action space --
def test_relay_action_requires_a_relay_index():
    with pytest.raises(ValueError):
        Action(ActionType.RELAY)
    assert Action(ActionType.RELAY, relay_indices=(4,)).relay_indices == (4,)


def test_negative_delay_is_rejected():
    with pytest.raises(ValueError):
        Action(ActionType.DEFER, delay_steps=-1)


def test_context_reports_neighbourhood_size():
    assert make_ctx().n_neighbours == 3
    assert make_ctx(duplicate_count=2).is_duplicate


def test_progress_is_positive_ahead_and_negative_behind():
    p = make_ctx().progress()
    assert p[2] > p[1] > 0 > p[0]


def test_progress_is_zero_without_a_sender():
    assert np.allclose(make_ctx(sender_dx=0.0, sender_dy=0.0).progress(), 0.0)


def test_registry_rejects_unknown_policy_with_a_helpful_message():
    with pytest.raises(KeyError) as exc:
        build_policy("definitely_not_a_policy")
    assert "Available" in str(exc.value)
