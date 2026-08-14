"""Lead endpoints: capture, read, timeline, stage change, recommendations."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...ai import engagement_timing, next_best_action
from ...contracts.activity import ActivityCreate, StageChange
from ...contracts.intelligence import EngagementProfile, NextBestAction
from ...contracts.lead import LeadCreate, LeadView
from ...contracts.vocabulary import Channel
from ...db import repo
from ...db.models import Lead
from ...domain import sla, stages
from ...services import leads as lead_service
from ...settings import get_settings
from ...util import utcnow
from ..deps import Caller, get_db, require_caller, require_human

router = APIRouter(prefix="/api/leads", tags=["leads"])


class CaptureResponse(BaseModel):
    lead_id: UUID
    is_new: bool
    action: str
    reason: str
    owner_id: UUID | None
    sla_due_at: datetime | None


class StageChangeRequest(BaseModel):
    to_stage: str
    reason: str = "manual"


def _load(db: Session, lead_id: UUID) -> Lead:
    lead = repo.get_lead(db, lead_id)
    if lead is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="lead not found")
    if lead.merged_into is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"lead was merged into {lead.merged_into}",
        )
    return lead


def _view(lead: Lead) -> LeadView:
    from ...contracts.lead import LeadIdentity

    return LeadView(
        id=lead.id,
        identity=LeadIdentity(
            phone=lead.phone,
            alternate_phone=lead.alternate_phone,
            email=lead.email,
            full_name=lead.full_name,
        ),
        stage=lead.stage,
        source=lead.source,
        owner_id=lead.owner_id,
        owner_name=None,
        team=None,
        consent=lead_service.consent_of(lead),
        fields=lead.fields or {},
        score=lead.score,
        score_band=lead.score_band,
        created_at=lead.created_at,
        updated_at=lead.updated_at,
        last_activity_at=lead.last_activity_at,
        first_touch_at=lead.first_touch_at,
        touched=lead.first_touch_at is not None,
        external_ids=lead.external_ids or {},
    )


@router.post("", response_model=CaptureResponse, status_code=status.HTTP_201_CREATED)
def capture_lead(
    payload: LeadCreate,
    db: Session = Depends(get_db),
    caller: Caller = Depends(require_caller),
) -> CaptureResponse:
    """Capture an inbound lead. Safe to retry — identity resolution deduplicates."""
    try:
        stages.validate(payload.stage) if payload.stage else None
    except stages.InvalidStage as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    result = lead_service.capture(db, payload, producer=caller.label)
    return CaptureResponse(
        lead_id=result.lead_id,
        is_new=result.is_new,
        action=result.action,
        reason=result.reason,
        owner_id=result.owner_id,
        sla_due_at=result.sla_due_at,
    )


@router.get("/{lead_id}", response_model=LeadView)
def get_lead(
    lead_id: UUID,
    db: Session = Depends(get_db),
    _: Caller = Depends(require_caller),
) -> LeadView:
    return _view(_load(db, lead_id))


@router.get("/{lead_id}/timeline")
def get_timeline(
    lead_id: UUID,
    limit: int = Query(default=50, le=200),
    before: datetime | None = None,
    db: Session = Depends(get_db),
    _: Caller = Depends(require_caller),
) -> list[dict]:
    """The lead's full history, newest first.

    This is the answer to "a counsellor picks up a six-month-old lead and needs
    the background" — one call, no note archaeology.
    """
    _load(db, lead_id)
    rows = repo.timeline(db, lead_id, limit=limit, before=before)
    return [
        {
            "id": str(a.id),
            "type": a.type,
            "channel": a.channel,
            "direction": a.direction,
            "occurred_at": a.occurred_at,
            "actor_name": a.actor_name,
            "subject": a.subject,
            "body": a.body,
            "payload": a.payload,
        }
        for a in rows
    ]


@router.post("/{lead_id}/activities", status_code=status.HTTP_202_ACCEPTED)
def add_activity(
    lead_id: UUID,
    payload: ActivityCreate,
    db: Session = Depends(get_db),
    caller: Caller = Depends(require_caller),
) -> dict:
    lead = _load(db, lead_id)
    if payload.lead_id != lead_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="lead_id mismatch")
    if payload.actor_id is None and caller.user_id is not None:
        payload = payload.model_copy(update={"actor_id": caller.user_id, "actor_name": caller.label})
    lead_service.record(db, lead, payload)
    return {"status": "recorded", "score": lead.score, "band": lead.score_band}


@router.post("/{lead_id}/stage")
def change_stage(
    lead_id: UUID,
    payload: StageChangeRequest,
    db: Session = Depends(get_db),
    caller: Caller = Depends(require_human),
) -> dict:
    """Move a lead's stage. Always attributable to a person."""
    lead = _load(db, lead_id)
    try:
        event = lead_service.change_stage(
            db,
            lead,
            StageChange(
                from_stage=lead.stage,
                to_stage=payload.to_stage,
                changed_by=caller.user_id,
                reason=payload.reason,
            ),
            actor_is_human=True,
        )
    except (ValueError, stages.InvalidStage) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return event.model_dump(mode="json")


@router.get("/{lead_id}/engagement", response_model=EngagementProfile)
def get_engagement(
    lead_id: UUID,
    db: Session = Depends(get_db),
    _: Caller = Depends(require_caller),
) -> EngagementProfile:
    """When this lead actually responds. Reports its own sample size."""
    _load(db, lead_id)
    activities = repo.responses_for(db, lead_id)
    responses = [
        engagement_timing.Response(occurred_at=a.occurred_at, channel=Channel(a.channel))
        for a in activities
        if a.channel in {c.value for c in Channel}
    ]
    return engagement_timing.build_profile(lead_id, responses)


@router.get("/{lead_id}/next-best-action", response_model=list[NextBestAction])
def get_next_best_action(
    lead_id: UUID,
    db: Session = Depends(get_db),
    _: Caller = Depends(require_caller),
) -> list[NextBestAction]:
    lead = _load(db, lead_id)
    settings = get_settings()

    activities = repo.responses_for(db, lead_id)
    profile = engagement_timing.build_profile(
        lead_id,
        [
            engagement_timing.Response(occurred_at=a.occurred_at, channel=Channel(a.channel))
            for a in activities
            if a.channel in {c.value for c in Channel}
        ],
    )

    latest_calls = repo.timeline(db, lead_id, limit=1, types=["call_summary"])
    promises: list[str] = []
    promise_at = None
    if latest_calls:
        promises = list(latest_calls[0].payload.get("promised_actions") or [])
        promise_at = latest_calls[0].occurred_at

    facts = next_best_action.ActionFacts(
        lead_id=lead_id,
        stage=lead.stage,
        score=lead.score,
        consent=lead_service.consent_of(lead),
        now=utcnow(),
        unanswered_inbound_at=lead.last_inbound_at,
        open_promises=promises,
        promise_made_at=promise_at,
        sla_due_at=sla.first_touch_due(
            lead.source,
            lead.captured_at,
            day_start_hour=settings.working_hours_start,
            day_end_hour=settings.working_hours_end,
        ),
        first_touch_at=lead.first_touch_at,
        last_contact_at=lead.last_activity_at,
        profile=profile,
        is_terminal=stages.is_terminal(lead.stage),
    )
    return next_best_action.recommend(facts)
