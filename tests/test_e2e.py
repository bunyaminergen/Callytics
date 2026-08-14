"""End-to-end: capture a lead, land a call from FinEcho, decide the proposal.

This walks the exact path production takes — HTTP in, database out — with no
FinEcho process, no Kafka and no MySQL.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from callytics.db.models import Activity, Lead, Proposal
from callytics.util import uuid7

SECRET = "test-secret"
PHONE = "+91 98765 43210"
NORMALIZED = "919876543210"


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _capture(client, auth, **overrides):
    payload = {
        "identity": {"phone": PHONE, "full_name": "Amal Raj", "email": "amal@example.com"},
        "source": "FB Lead Ads",
        "fields": {"course_category": "ACCA", "district": "Kozhikode"},
        "external_ids": {"meta": "form-sub-1"},
    }
    payload.update(overrides)
    return client.post("/api/leads", json=payload, headers=auth)


def _finecho_payload(call_id, **summary):
    body = {
        "call": {
            "call_id": str(call_id),
            "lead_phone": NORMALIZED,
            "employee_id": "anju",
            "recorded_at": datetime.now(UTC).isoformat(),
            "duration_seconds": 412.5,
            "language_primary": "ml",
            "audio_uri": "s3://recordings/call.wav",
        },
        "summary": {
            "summary": "Discussed the ACCA programme, fees and batch timings.",
            "points_to_remember": ["Prefers evening batch"],
            "objections": [],
            "hooks": [],
            "promised_actions": [],
            "suggested_next_steps": [],
            "model_tag": "finecho-l3.2-3b-sft1",
        },
        "evaluation": {
            "call_type": "first_contact",
            "overall_score": 7.5,
            "detected_program": "ACCA",
            "criteria": [],
            "compliance_flags": [],
        },
    }
    body["summary"].update(summary)
    return body


# --- capture --------------------------------------------------------------


def test_capture_creates_lead_with_timeline_and_score(client, auth, db):
    response = _capture(client, auth)
    assert response.status_code == 201
    body = response.json()
    assert body["is_new"] is True
    assert body["sla_due_at"] is not None, "a paid source must start an SLA clock"

    lead = db.get(Lead, __import__("uuid").UUID(body["lead_id"]))
    assert lead.phone == NORMALIZED, "phone must be normalised on the way in"
    assert lead.stage == "Open"
    assert lead.score >= 0

    events = db.execute(select(Activity).where(Activity.lead_id == lead.id)).scalars().all()
    assert any(a.type == "lead_created" for a in events)


def test_repeat_submission_attaches_instead_of_duplicating(client, auth, db):
    first = _capture(client, auth).json()
    second = _capture(client, auth, external_ids={"meta": "form-sub-2"}).json()

    assert second["is_new"] is False
    assert second["lead_id"] == first["lead_id"]
    assert second["action"] == "attach"
    assert db.execute(select(Lead)).scalars().all().__len__() == 1


def test_capture_is_idempotent_on_webhook_redelivery(client, auth, db):
    """Same external id twice must not double the timeline."""
    _capture(client, auth)
    _capture(client, auth)
    lead = db.execute(select(Lead)).scalars().one()
    created = db.execute(
        select(Activity).where(Activity.lead_id == lead.id, Activity.type == "lead_created")
    ).scalars().all()
    assert len(created) == 1


def test_unknown_stage_is_rejected(client, auth):
    assert _capture(client, auth, stage="Nonsense").status_code == 422


def test_unauthenticated_capture_is_refused(client):
    assert _capture(client, {}).status_code == 401
    assert _capture(client, {"Authorization": "Bearer wrong"}).status_code == 401


# --- FinEcho webhook ------------------------------------------------------


def test_finecho_webhook_lands_summary_on_timeline(client, auth, db):
    lead_id = _capture(client, auth).json()["lead_id"]
    body = json.dumps(_finecho_payload(uuid7())).encode()

    response = client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )
    assert response.status_code == 202
    result = response.json()
    assert result["matched"] is True
    assert result["lead_id"] == lead_id

    summaries = db.execute(select(Activity).where(Activity.type == "call_summary")).scalars().all()
    assert len(summaries) == 1
    assert "ACCA programme" in summaries[0].body
    assert summaries[0].payload["overall_score"] == 7.5


def test_finecho_webhook_rejects_bad_signature(client, auth):
    _capture(client, auth)
    body = json.dumps(_finecho_payload(uuid7())).encode()
    response = client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": "sha256=deadbeef", "Content-Type": "application/json"},
    )
    assert response.status_code == 401


def test_finecho_webhook_is_idempotent(client, auth, db):
    _capture(client, auth)
    call_id = uuid7()
    body = json.dumps(_finecho_payload(call_id)).encode()
    headers = {"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"}

    client.post("/webhooks/finecho/call", content=body, headers=headers)
    second = client.post("/webhooks/finecho/call", content=body, headers=headers)

    assert second.status_code == 202
    assert second.json()["detail"] == "call already ingested"
    assert len(db.execute(select(Activity).where(Activity.type == "call_summary")).scalars().all()) == 1


def test_unmatched_call_is_reported_not_stored(client, db):
    payload = _finecho_payload(uuid7())
    payload["call"]["lead_phone"] = "919000000000"
    body = json.dumps(payload).encode()
    response = client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )
    assert response.status_code == 202
    assert response.json()["matched"] is False
    assert db.execute(select(Activity).where(Activity.type == "call_summary")).scalars().all() == []


# --- proposals ------------------------------------------------------------


def test_commitment_call_raises_a_proposal_but_does_not_apply_it(client, auth, db, counsellor):
    """The central guarantee: AI suggests, the pipeline does not move by itself."""
    lead_id = _capture(client, auth).json()["lead_id"]
    body = json.dumps(
        _finecho_payload(uuid7(), promised_actions=["Will pay the admission fee on Monday"])
    ).encode()
    client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )

    lead = db.get(Lead, __import__("uuid").UUID(lead_id))
    db.refresh(lead)
    assert lead.stage == "Open", "stage must not move without a human decision"

    pending = client.get("/api/proposals", headers=auth).json()
    assert len(pending) == 1
    assert pending[0]["proposed"]["stage"] == "Confirmed - Waiting for payment"
    assert pending[0]["evidence"], "a proposal must cite its evidence"
    assert 0 < pending[0]["confidence"] <= 1


def test_accepting_a_proposal_moves_the_stage_and_is_attributed(client, auth, db, counsellor):
    lead_id = _capture(client, auth).json()["lead_id"]
    body = json.dumps(
        _finecho_payload(uuid7(), objections=["Not interested, joining another institute"])
    ).encode()
    client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )

    proposal_id = client.get("/api/proposals", headers=auth).json()[0]["id"]
    human = {**auth, "X-User-Id": str(counsellor.id)}
    decision = client.post(
        f"/api/proposals/{proposal_id}/decision",
        json={"accept": True, "note": "confirmed on the call"},
        headers=human,
    )
    assert decision.status_code == 200
    assert decision.json()["applied"] is True

    lead = db.get(Lead, __import__("uuid").UUID(lead_id))
    db.refresh(lead)
    assert lead.stage in {"Not interested", "Lost to competitor"}

    stored = db.execute(select(Proposal)).scalars().one()
    assert stored.status == "accepted"
    assert stored.decided_by == counsellor.id


def test_rejecting_keeps_the_stage_and_records_the_label(client, auth, db, counsellor):
    lead_id = _capture(client, auth).json()["lead_id"]
    body = json.dumps(_finecho_payload(uuid7(), objections=["Not interested"])).encode()
    client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )

    proposal_id = client.get("/api/proposals", headers=auth).json()[0]["id"]
    human = {**auth, "X-User-Id": str(counsellor.id)}
    response = client.post(
        f"/api/proposals/{proposal_id}/decision",
        json={"accept": False, "note": "misread — they were asking about a different course"},
        headers=human,
    )
    assert response.status_code == 200
    assert response.json()["applied"] is False

    lead = db.get(Lead, __import__("uuid").UUID(lead_id))
    db.refresh(lead)
    assert lead.stage == "Open"

    stored = db.execute(select(Proposal)).scalars().one()
    assert stored.status == "rejected"
    assert "misread" in stored.decision_note


def test_proposal_decision_requires_a_named_human(client, auth, db):
    _capture(client, auth)
    body = json.dumps(_finecho_payload(uuid7(), objections=["Not interested"])).encode()
    client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )
    proposal_id = client.get("/api/proposals", headers=auth).json()[0]["id"]

    # Service token alone is not enough — decisions must be attributable.
    response = client.post(
        f"/api/proposals/{proposal_id}/decision", json={"accept": True}, headers=auth
    )
    assert response.status_code == 403


def test_a_decided_proposal_cannot_be_decided_twice(client, auth, db, counsellor):
    _capture(client, auth)
    body = json.dumps(_finecho_payload(uuid7(), objections=["Not interested"])).encode()
    client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )
    proposal_id = client.get("/api/proposals", headers=auth).json()[0]["id"]
    human = {**auth, "X-User-Id": str(counsellor.id)}

    client.post(f"/api/proposals/{proposal_id}/decision", json={"accept": True}, headers=human)
    second = client.post(
        f"/api/proposals/{proposal_id}/decision", json={"accept": True}, headers=human
    )
    assert second.status_code == 409


# --- reading surfaces -----------------------------------------------------


def test_timeline_returns_history_newest_first(client, auth, db):
    lead_id = _capture(client, auth).json()["lead_id"]
    body = json.dumps(_finecho_payload(uuid7())).encode()
    client.post(
        "/webhooks/finecho/call",
        content=body,
        headers={"X-FinEcho-Signature": _sign(body), "Content-Type": "application/json"},
    )

    timeline = client.get(f"/api/leads/{lead_id}/timeline", headers=auth).json()
    assert len(timeline) >= 2
    assert timeline[0]["type"] == "call_summary"
    timestamps = [t["occurred_at"] for t in timeline]
    assert timestamps == sorted(timestamps, reverse=True)


def test_stage_change_requires_reason_when_reopening(client, auth, db, counsellor):
    lead_id = _capture(client, auth).json()["lead_id"]
    human = {**auth, "X-User-Id": str(counsellor.id)}

    client.post(
        f"/api/leads/{lead_id}/stage",
        json={"to_stage": "Not interested", "reason": "declined on call"},
        headers=human,
    )
    blocked = client.post(f"/api/leads/{lead_id}/stage", json={"to_stage": "Interested"}, headers=human)
    assert blocked.status_code == 409
    assert "requires an explicit reason" in blocked.json()["detail"]


def test_next_best_action_surfaces_sla_breach(client, auth, db, counsellor):
    lead_id = _capture(client, auth).json()["lead_id"]
    lead = db.get(Lead, __import__("uuid").UUID(lead_id))
    lead.captured_at = datetime.now(UTC) - timedelta(days=2)
    db.commit()

    actions = client.get(f"/api/leads/{lead_id}/next-best-action", headers=auth).json()
    assert actions
    assert any(a["reason"] for a in actions)


def test_engagement_profile_reports_insufficient_data(client, auth):
    lead_id = _capture(client, auth).json()["lead_id"]
    profile = client.get(f"/api/leads/{lead_id}/engagement", headers=auth).json()
    assert profile["sufficient_data"] is False
    assert profile["windows"] == []
