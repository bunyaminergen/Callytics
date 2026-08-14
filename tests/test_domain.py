"""Domain rules — pure functions, no database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from callytics.contracts.lead import Consent, LeadIdentity
from callytics.contracts.vocabulary import Channel
from callytics.domain import consent as consent_rules
from callytics.domain import identity, routing, scoring, sla, stages
from callytics.util import uuid7

LEAD_1 = uuid7()


# --- identity -------------------------------------------------------------


def test_exact_phone_match_attaches():
    incoming = LeadIdentity(phone="+91 98765 43210", full_name="Amal Raj")
    existing = identity.ExistingLead(
        lead_id=LEAD_1, phones=("919876543210",), emails=(), full_name="Amal R"
    )
    decision = identity.resolve(incoming, [existing])
    assert decision.action == "attach"
    assert decision.matched_lead_id == LEAD_1
    assert identity.should_auto_attach(decision)


def test_name_match_alone_never_auto_attaches():
    """Two people called 'Amal' are not one person."""
    incoming = LeadIdentity(phone="919999900000", full_name="Amal Raj")
    existing = identity.ExistingLead(
        lead_id=LEAD_1, phones=("919876543210",), emails=(), full_name="amal raj"
    )
    decision = identity.resolve(incoming, [existing])
    assert decision.action == "review"
    assert not identity.should_auto_attach(decision)


def test_merged_leads_are_not_candidates():
    incoming = LeadIdentity(phone="919876543210")
    existing = identity.ExistingLead(
        lead_id=LEAD_1, phones=("919876543210",), emails=(), full_name=None, is_merged=True
    )
    assert identity.resolve(incoming, [existing]).action == "create"


def test_email_match_attaches_when_phone_differs():
    incoming = LeadIdentity(phone="919999900000", email="A.Raj@Example.COM")
    existing = identity.ExistingLead(
        lead_id=LEAD_1, phones=("919876543210",), emails=("a.raj@example.com",), full_name=None
    )
    decision = identity.resolve(incoming, [existing])
    assert decision.action == "attach"


# --- consent --------------------------------------------------------------


def test_dnc_blocks_calls_only():
    c = Consent(do_not_call=True, whatsapp_opt_in_at=datetime.now(UTC))
    assert not consent_rules.check(c, Channel.CALL).allowed
    assert consent_rules.check(c, Channel.WHATSAPP).allowed


def test_whatsapp_requires_explicit_opt_in():
    assert not consent_rules.check(Consent(), Channel.WHATSAPP).allowed


def test_marketable_requires_every_channel_clear():
    assert consent_rules.marketable(Consent())
    assert not consent_rules.marketable(Consent(do_not_email=True))


def test_require_raises():
    with pytest.raises(consent_rules.ConsentDenied):
        consent_rules.require(Consent(do_not_sms=True), Channel.SMS)


# --- stages ---------------------------------------------------------------


def test_machine_cannot_set_revenue_stage():
    result = stages.can_transition("Interested", "JOINED", actor_is_human=False)
    assert not result.allowed
    assert "must be set by a person" in result.reason


def test_human_can_set_revenue_stage():
    assert stages.can_transition("Interested", "JOINED", actor_is_human=True).allowed


def test_reopening_terminal_stage_requires_reason():
    result = stages.can_transition("Not interested", "Interested", actor_is_human=True)
    assert result.allowed and result.requires_reason


def test_import_is_not_a_reachable_stage():
    assert not stages.can_transition("Open", "Import", actor_is_human=True).allowed


def test_unknown_stage_rejected():
    with pytest.raises(stages.InvalidStage):
        stages.validate("Nonsense")


# --- scoring --------------------------------------------------------------


def _facts(**kw):
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    base = {
        "stage": "Open",
        "source": "Select Source",
        "created_at": now - timedelta(days=1),
        "now": now,
    }
    base.update(kw)
    return scoring.ScoringFacts(**base)


def test_score_is_bounded_and_explained():
    score, band, contributions = scoring.compute(
        _facts(stage="Interested", source="Direct Walk-in", connected_calls=3, appointments_kept=1)
    )
    assert 0 <= score <= 100
    assert band in {"cold", "warm", "hot"}
    assert contributions, "a score must always explain itself"
    # The published score must be the sum of its stated reasons, clamped to
    # 0..100 — otherwise the explanation is decorative.
    assert score == max(0, min(100, round(sum(c.points for c in contributions))))
    assert all(c.detail for c in contributions)


def test_walk_in_outscores_bulk_data():
    walk_in, _, _ = scoring.compute(_facts(source="Direct Walk-in"))
    bulk, _, _ = scoring.compute(_facts(source="College Data purchase"))
    assert walk_in > bulk


def test_engagement_decays_with_time():
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    fresh, _, _ = scoring.compute(
        _facts(connected_calls=2, last_engagement_at=now - timedelta(days=1))
    )
    stale, _, _ = scoring.compute(
        _facts(connected_calls=2, last_engagement_at=now - timedelta(days=90))
    )
    assert fresh > stale


def test_suppressed_lead_is_penalised():
    plain, _, _ = scoring.compute(_facts(stage="Interested", connected_calls=2))
    suppressed, _, _ = scoring.compute(_facts(stage="Interested", connected_calls=2, is_suppressed=True))
    assert suppressed < plain


def test_unreachable_after_repeated_attempts():
    _, _, contributions = scoring.compute(_facts(call_attempts=6, connected_calls=0))
    assert any(c.feature == "unreachable" for c in contributions)


# --- SLA ------------------------------------------------------------------


def test_paid_source_gets_tightest_clock():
    assert sla.rule_for("FB Lead Ads").first_touch < sla.rule_for("Organic Search").first_touch


def test_bulk_source_has_no_first_touch_sla():
    assert sla.first_touch_due("College Data purchase", datetime.now(UTC)) is None


def test_out_of_hours_arrival_is_due_next_morning():
    captured = datetime(2026, 8, 14, 22, 40, tzinfo=UTC)
    due = sla.first_touch_due("FB Lead Ads", captured, day_start_hour=9, day_end_hour=20)
    assert due is not None
    assert due.day == 15 and due.hour == 9


def test_in_hours_arrival_uses_plain_offset():
    captured = datetime(2026, 8, 14, 10, 0, tzinfo=UTC)
    due = sla.first_touch_due("FB Lead Ads", captured, day_start_hour=9, day_end_hour=20)
    assert due == datetime(2026, 8, 14, 10, 15, tzinfo=UTC)


def test_breach_detection():
    due = datetime(2026, 8, 14, 10, 0, tzinfo=UTC)
    late = datetime(2026, 8, 14, 11, 0, tzinfo=UTC)
    assert sla.is_breached(due, None, late)
    assert not sla.is_breached(due, datetime(2026, 8, 14, 9, 30, tzinfo=UTC), late)
    assert not sla.is_breached(None, None, late)


# --- routing --------------------------------------------------------------


def _candidate(name, **kw):
    return routing.Candidate(user_id=name, name=name, teams=kw.pop("teams", frozenset({"Calicut"})), **kw)


def test_routes_to_least_loaded():
    decision = routing.choose(
        routing.RoutingRequest(source="FB Lead Ads"),
        [_candidate("Anju", open_leads=40), _candidate("Nifthab", open_leads=5)],
    )
    assert decision.user_id == "Nifthab"


def test_capacity_is_respected():
    decision = routing.choose(
        routing.RoutingRequest(source="FB Lead Ads"),
        [_candidate("Anju", open_leads=150, capacity=150)],
    )
    assert decision.user_id is None
    assert "capacity" in decision.reason


def test_language_filter_excludes_ineligible():
    decision = routing.choose(
        routing.RoutingRequest(source="FB Lead Ads", language="malayalam"),
        [_candidate("Deepa", languages=frozenset({"kannada"}))],
    )
    assert decision.user_id is None
    assert "malayalam" in decision.reason


def test_preferred_team_wins_when_available():
    decision = routing.choose(
        routing.RoutingRequest(source="FB Lead Ads", preferred_team="TVM"),
        [
            _candidate("Anju", teams=frozenset({"Calicut"}), open_leads=0),
            _candidate("Sahala", teams=frozenset({"TVM"}), open_leads=20),
        ],
    )
    assert decision.user_id == "Sahala"
