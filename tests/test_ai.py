"""Intelligence layer: engagement timing, stage inference, next-best-action."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from callytics.ai import engagement_timing, next_best_action, stage_inference
from callytics.contracts.intelligence import CallIntelligence
from callytics.contracts.lead import Consent
from callytics.contracts.vocabulary import CallDisposition, Channel
from callytics.util import uuid7

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
LEAD = uuid7()


# --- engagement timing ----------------------------------------------------


def _responses(hours, channel=Channel.WHATSAPP):
    return [
        engagement_timing.Response(occurred_at=NOW.replace(hour=h) - timedelta(days=i), channel=channel)
        for i, h in enumerate(hours)
    ]


def test_refuses_to_guess_from_thin_history():
    """Three data points is not a pattern, and saying so beats inventing one."""
    profile = engagement_timing.build_profile(LEAD, _responses([19, 19, 20]), now=NOW)
    assert not profile.sufficient_data
    assert profile.windows == ()
    assert profile.total_observations == 3


def test_finds_evening_window_with_enough_history():
    profile = engagement_timing.build_profile(LEAD, _responses([19, 19, 20, 19, 20, 19, 20, 21]), now=NOW)
    assert profile.sufficient_data
    assert profile.windows
    best = profile.windows[0]
    assert best.start.hour >= 19
    assert best.observations >= 5
    assert "responses arrived between" in best.rationale


def test_preferred_channel_is_the_observed_one():
    profile = engagement_timing.build_profile(
        LEAD, _responses([19, 19, 20, 19, 20, 19], channel=Channel.WHATSAPP), now=NOW
    )
    assert profile.preferred_channel == Channel.WHATSAPP


def test_responsiveness_uses_attempts():
    attempts = [
        engagement_timing.Attempt(occurred_at=NOW, channel=Channel.CALL, connected=(i % 4 == 0))
        for i in range(8)
    ]
    profile = engagement_timing.build_profile(LEAD, _responses([19] * 6), attempts, now=NOW)
    assert profile.responsiveness == 0.25


def test_next_window_start_returns_now_when_already_inside():
    profile = engagement_timing.build_profile(LEAD, _responses([12, 12, 12, 13, 12, 13]), now=NOW)
    assert engagement_timing.next_window_start(profile, NOW) == NOW


# --- stage inference ------------------------------------------------------


def _intel(**kw) -> CallIntelligence:
    base = {
        "call_id": uuid7(),
        "lead_phone": "919876543210",
        "occurred_at": NOW,
    }
    base.update(kw)
    return CallIntelligence(**base)


def test_no_answer_maps_to_did_not_pick():
    signal = stage_inference.infer(_intel(), "Open", disposition=CallDisposition.NO_ANSWER)
    assert signal is not None
    assert signal.to_stage == "Did Not Pick"
    assert signal.evidence


def test_invalid_number_maps_to_junk():
    signal = stage_inference.infer(_intel(), "Open", disposition=CallDisposition.INVALID_NUMBER)
    assert signal is not None and signal.to_stage == "Junk leads"


def test_commitment_language_proposes_confirmed_stage():
    signal = stage_inference.infer(
        _intel(promised_actions=["Student will pay the admission fee on Monday"]),
        "Interested",
        disposition=CallDisposition.CONNECTED,
    )
    assert signal is not None
    assert signal.to_stage == "Confirmed - Waiting for payment"
    assert signal.evidence[0].detail.startswith("Student will pay")


def test_rejection_language_proposes_not_interested():
    signal = stage_inference.infer(
        _intel(objections=["Not interested, the fee is too high"]),
        "Interested",
        disposition=CallDisposition.CONNECTED,
    )
    assert signal is not None and signal.to_stage == "Not interested"


def test_callback_language_proposes_callback_stage():
    signal = stage_inference.infer(
        _intel(suggested_next_steps=["Call back later this week after they discuss with their parents"]),
        "Open",
        disposition=CallDisposition.CONNECTED,
    )
    assert signal is not None and signal.to_stage == "Call back later"


def test_no_signal_when_call_is_unremarkable():
    signal = stage_inference.infer(
        _intel(summary="Discussed the syllabus and duration."),
        "Interested",
        disposition=CallDisposition.CONNECTED,
    )
    assert signal is None


def test_never_proposes_the_current_stage():
    signal = stage_inference.infer(
        _intel(objections=["Not interested"]),
        "Not interested",
        disposition=CallDisposition.CONNECTED,
    )
    assert signal is None


def test_corroboration_raises_confidence():
    single = stage_inference.infer(
        _intel(objections=["Not interested"]), "Interested", disposition=CallDisposition.CONNECTED
    )
    multiple = stage_inference.infer(
        _intel(
            objections=["Not interested"],
            summary="Lead said they are not interested.",
            points_to_remember=["Not interested in the ACCA programme"],
        ),
        "Interested",
        disposition=CallDisposition.CONNECTED,
    )
    assert single and multiple
    assert multiple.confidence > single.confidence
    assert multiple.confidence <= 0.95


def test_auto_apply_never_touches_revenue_stages():
    """The core guard rail: a model may never book revenue."""
    signal = stage_inference.Signal(
        to_stage="JOINED", confidence=0.99, rationale="test", evidence=()
    )
    assert not stage_inference.may_auto_apply(signal, enabled=True, min_confidence=0.5)


def test_auto_apply_is_off_by_default_and_floor_gated():
    signal = stage_inference.Signal(
        to_stage="Call back later", confidence=0.70, rationale="test", evidence=()
    )
    assert not stage_inference.may_auto_apply(signal, enabled=False, min_confidence=0.5)
    assert not stage_inference.may_auto_apply(signal, enabled=True, min_confidence=0.80)
    assert stage_inference.may_auto_apply(signal, enabled=True, min_confidence=0.60)


def test_proposal_dedupe_key_is_bound_to_the_call():
    call_id = uuid7()
    signal = stage_inference.Signal("Call back later", 0.7, "test", ())
    row = stage_inference.to_proposal(signal, LEAD, "Open", call_id=call_id)
    assert row["dedupe_key"] == f"finecho:call_intelligence:{call_id}:Call back later"
    assert row["status"] == "pending"


# --- next best action -----------------------------------------------------


def _facts(**kw):
    base = {
        "lead_id": LEAD,
        "stage": "Interested",
        "score": 60,
        "consent": Consent(whatsapp_opt_in_at=NOW),
        "now": NOW,
    }
    base.update(kw)
    return next_best_action.ActionFacts(**base)


def test_unkept_promise_outranks_everything():
    actions = next_best_action.recommend(
        _facts(
            open_promises=["Send the fee structure PDF"],
            unanswered_inbound_at=NOW - timedelta(hours=2),
        )
    )
    assert actions[0].priority == 0
    assert "Send the fee structure" in actions[0].action


def test_unanswered_inbound_surfaces():
    actions = next_best_action.recommend(_facts(unanswered_inbound_at=NOW - timedelta(hours=3)))
    assert actions[0].action == "Reply to the lead's message"
    assert "3h ago" in actions[0].reason


def test_recent_inbound_is_within_grace():
    actions = next_best_action.recommend(_facts(unanswered_inbound_at=NOW - timedelta(minutes=5)))
    assert not any(a.action.startswith("Reply") for a in actions)


def test_sla_breach_surfaces_first_contact():
    actions = next_best_action.recommend(
        _facts(sla_due_at=NOW - timedelta(minutes=20), first_touch_at=None)
    )
    assert any("first contact" in a.action.lower() for a in actions)
    assert any("breached" in a.reason for a in actions)


def test_terminal_lead_gets_no_actions():
    assert next_best_action.recommend(_facts(is_terminal=True)) == []


def test_fully_suppressed_lead_gets_no_actions():
    consent = Consent(do_not_call=True, do_not_sms=True, do_not_email=True, do_not_whatsapp=True)
    assert next_best_action.recommend(_facts(consent=consent)) == []


def test_actions_respect_consent_channel():
    """A DNC lead must never be told to call."""
    consent = Consent(do_not_call=True, whatsapp_opt_in_at=NOW)
    actions = next_best_action.recommend(
        _facts(consent=consent, sla_due_at=NOW - timedelta(minutes=10))
    )
    assert all(a.channel != Channel.CALL for a in actions)


def test_stale_lead_gets_routine_follow_up():
    actions = next_best_action.recommend(_facts(last_contact_at=NOW - timedelta(days=10)))
    assert actions and actions[0].priority == 5
    assert "10d" in actions[0].reason
