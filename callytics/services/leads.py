"""Lead capture and lifecycle orchestration.

This is where the domain rules, the repository and the event bus meet. The
capture path is the one that has to be right: it runs on every ad webhook, web
form and import row, and a mistake here creates duplicate leads or drops a paid
enquiry on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from ..contracts.activity import ActivityCreate, StageChange
from ..contracts.events import LeadCaptured, LeadStageChanged
from ..contracts.lead import Consent, LeadCreate
from ..contracts.vocabulary import STAGE_OPEN, ActivityType, Channel, Direction
from ..db import repo
from ..db.models import Lead
from ..domain import identity, routing, scoring, sla, stages
from ..settings import get_settings
from ..util import utcnow, uuid7


@dataclass
class CaptureResult:
    lead_id: UUID
    is_new: bool
    action: str
    reason: str
    owner_id: UUID | None
    sla_due_at: datetime | None


def capture(session: Session, payload: LeadCreate, *, producer: str = "api") -> CaptureResult:
    """Create or attach an inbound lead.

    Order matters: resolve identity *before* writing, so a redelivered webhook
    attaches to the existing lead instead of creating a twin.
    """
    settings = get_settings()
    now = utcnow()
    captured_at = payload.captured_at or now

    candidates = repo.identity_candidates(
        session,
        phones=payload.identity.phones,
        email=payload.identity.email,
        full_name=payload.identity.full_name,
    )
    decision = identity.resolve(payload.identity, candidates)

    if identity.should_auto_attach(decision) and decision.matched_lead_id is not None:
        lead_id = decision.matched_lead_id  # type: ignore[assignment]
        lead = repo.get_lead(session, lead_id)
        if lead is not None:
            _attach_to_existing(session, lead, payload, captured_at)
            return CaptureResult(
                lead_id=lead.id,
                is_new=False,
                action="attach",
                reason=decision.reason,
                owner_id=lead.owner_id,
                sla_due_at=None,
            )

    # New lead. Route it, start its SLA clock, and seed the timeline.
    stage = payload.stage or STAGE_OPEN
    stages.validate(stage)

    owner_id = payload.owner_id
    if owner_id is None:
        candidates_raw = repo.routing_candidates(session, payload.team)
        pool = [
            routing.Candidate(
                user_id=c["user_id"],
                name=c["name"],
                teams=c["teams"],
                open_leads=c["open_leads"],
            )
            for c in candidates_raw
        ]
        request = routing.RoutingRequest(
            source=payload.source,
            language=payload.fields.get("language"),
            district=payload.fields.get("district"),
            course_category=payload.fields.get("course_category"),
            preferred_team=payload.team,
        )
        owner_id = routing.choose(request, pool).user_id  # type: ignore[assignment]

    sla_due_at = sla.first_touch_due(
        payload.source,
        captured_at,
        day_start_hour=settings.working_hours_start,
        day_end_hour=settings.working_hours_end,
    )

    lead = Lead(
        id=uuid7(),
        phone=payload.identity.phone,
        alternate_phone=payload.identity.alternate_phone,
        email=payload.identity.email,
        full_name=payload.identity.full_name,
        stage=stage,
        source=payload.source,
        source_campaign=payload.source_campaign,
        source_medium=payload.source_medium,
        source_referrer=payload.source_referrer,
        owner_id=owner_id,
        do_not_call=payload.consent.do_not_call,
        do_not_sms=payload.consent.do_not_sms,
        do_not_email=payload.consent.do_not_email,
        do_not_whatsapp=payload.consent.do_not_whatsapp,
        consent_timestamp=payload.consent.consent_timestamp,
        consent_source=payload.consent.consent_source,
        fields=payload.fields,
        external_ids=payload.external_ids,
        captured_at=captured_at,
    )
    session.add(lead)
    session.flush()

    for phone in payload.identity.phones:
        repo.add_identifier(session, lead.id, "phone", phone, primary=phone == payload.identity.phone)
    if payload.identity.email:
        repo.add_identifier(session, lead.id, "email", payload.identity.email, primary=True)

    repo.record_activity(
        session,
        lead_id=lead.id,
        type=ActivityType.LEAD_CREATED.value,
        channel=Channel.SYSTEM.value,
        direction=Direction.INTERNAL.value,
        occurred_at=captured_at,
        subject=f"Lead captured from {payload.source}",
        payload={"source": payload.source, "campaign": payload.source_campaign},
        dedupe_key=_capture_dedupe_key(payload),
    )
    refresh_score(session, lead)
    return CaptureResult(
        lead_id=lead.id,
        is_new=True,
        action=decision.action if decision.action == "review" else "create",
        reason=decision.reason,
        owner_id=owner_id,
        sla_due_at=sla_due_at,
    )


def _capture_dedupe_key(payload: LeadCreate) -> str | None:
    """Stable key for a capture event, so webhook redelivery is harmless."""
    for system, external_id in sorted(payload.external_ids.items()):
        return f"capture:{system}:{external_id}"
    return None


def _attach_to_existing(session: Session, lead: Lead, payload: LeadCreate, captured_at: datetime) -> None:
    """Fold a repeat enquiry into the lead we already have.

    A re-enquiry is a real buying signal, so it lands on the timeline. Existing
    field values are never overwritten by a thinner later capture — only blanks
    are filled in.
    """
    for phone in payload.identity.phones:
        repo.add_identifier(session, lead.id, "phone", phone)
    if payload.identity.email:
        repo.add_identifier(session, lead.id, "email", payload.identity.email)
        if not lead.email:
            lead.email = payload.identity.email

    merged_fields = dict(lead.fields or {})
    for key, value in payload.fields.items():
        if value not in (None, "") and not merged_fields.get(key):
            merged_fields[key] = value
    lead.fields = merged_fields
    lead.last_activity_at = captured_at

    repo.record_activity(
        session,
        lead_id=lead.id,
        type=ActivityType.LEAD_CREATED.value,
        channel=Channel.SYSTEM.value,
        direction=Direction.INBOUND.value,
        occurred_at=captured_at,
        subject=f"Repeat enquiry from {payload.source}",
        payload={"source": payload.source, "campaign": payload.source_campaign, "repeat": True},
        dedupe_key=_capture_dedupe_key(payload),
    )
    refresh_score(session, lead)


def change_stage(
    session: Session,
    lead: Lead,
    change: StageChange,
    *,
    actor_is_human: bool = True,
) -> LeadStageChanged:
    """Apply a stage change after checking the stage machine."""
    result = stages.can_transition(lead.stage, change.to_stage, actor_is_human=actor_is_human)
    if not result.allowed:
        raise ValueError(result.reason)
    if result.requires_reason and change.reason in ("", "manual"):
        raise ValueError(f"reopening from {lead.stage!r} requires an explicit reason")

    previous = lead.stage
    lead.stage = change.to_stage
    now = utcnow()

    repo.record_activity(
        session,
        lead_id=lead.id,
        type=ActivityType.STAGE_CHANGED.value,
        channel=Channel.SYSTEM.value,
        direction=Direction.INTERNAL.value,
        occurred_at=now,
        actor_id=change.changed_by,
        subject=f"{previous} → {change.to_stage}",
        payload={
            "from": previous,
            "to": change.to_stage,
            "reason": change.reason,
            "proposal_id": str(change.proposal_id) if change.proposal_id else None,
        },
    )
    refresh_score(session, lead)
    return LeadStageChanged(
        lead_id=lead.id,
        from_stage=previous,
        to_stage=change.to_stage,
        changed_by=change.changed_by,
        reason=change.reason,
        occurred_at=now,
    )


def record(session: Session, lead: Lead, activity: ActivityCreate) -> None:
    """Append an activity and update the lead's denormalised counters."""
    written = repo.record_activity(
        session,
        lead_id=lead.id,
        type=activity.type.value,
        channel=activity.channel.value,
        direction=activity.direction.value,
        occurred_at=activity.occurred_at,
        actor_id=activity.actor_id,
        actor_name=activity.actor_name,
        subject=activity.subject,
        body=activity.body,
        payload=activity.payload,
        dedupe_key=activity.dedupe_key,
    )
    if written is None:
        return  # already recorded; counters must not double-count

    if lead.last_activity_at is None or activity.occurred_at > lead.last_activity_at:
        lead.last_activity_at = activity.occurred_at
    if activity.direction == Direction.INBOUND and (
        lead.last_inbound_at is None or activity.occurred_at > lead.last_inbound_at
    ):
        lead.last_inbound_at = activity.occurred_at
    if activity.type == ActivityType.CALL_LOGGED:
        lead.call_attempts += 1
        if activity.payload.get("connected"):
            lead.connected_calls += 1
            if lead.first_touch_at is None:
                lead.first_touch_at = activity.occurred_at
    refresh_score(session, lead)


def refresh_score(session: Session, lead: Lead) -> None:
    """Recompute and persist the lead's score from current facts."""
    settings = get_settings()
    aggregates = repo.scoring_facts_for(session, lead.id)
    field_keys = repo.field_definition_keys(session)
    filled = sum(1 for k in field_keys if (lead.fields or {}).get(k) not in (None, "", []))

    facts = scoring.ScoringFacts(
        stage=lead.stage,
        source=lead.source,
        created_at=lead.created_at or utcnow(),
        now=utcnow(),
        connected_calls=lead.connected_calls,
        call_attempts=lead.call_attempts,
        filled_field_count=filled,
        total_field_count=len(field_keys),
        is_suppressed=any((lead.do_not_call, lead.do_not_sms, lead.do_not_email, lead.do_not_whatsapp)),
        **aggregates,
    )
    score, band, contributions = scoring.compute(facts, half_life_days=settings.score_half_life_days)
    if score != lead.score or band != lead.score_band:
        repo.save_score(session, lead.id, score, band, [c.model_dump(mode="json") for c in contributions])
    lead.score = score
    lead.score_band = band


def captured_event(result: CaptureResult, payload: LeadCreate) -> LeadCaptured:
    return LeadCaptured(
        lead_id=result.lead_id,
        source=payload.source,
        stage=payload.stage or STAGE_OPEN,
        owner_id=result.owner_id,
        is_new=result.is_new,
        captured_at=payload.captured_at or utcnow(),
    )


def consent_of(lead: Lead) -> Consent:
    return Consent(
        do_not_call=lead.do_not_call,
        do_not_sms=lead.do_not_sms,
        do_not_email=lead.do_not_email,
        do_not_whatsapp=lead.do_not_whatsapp,
        consent_timestamp=lead.consent_timestamp,
        consent_source=lead.consent_source,
    )
