"""Landing FinEcho call intelligence onto a lead.

FinEcho owns transcription, translation, summarisation and rubric scoring. It
stays a separate service with its own release cadence and its own model stack;
Callytics never imports it. What crosses the boundary is this contract, and
what Callytics does with it is three things:

1. attach the summary to the lead's timeline so the six-month-history problem
   is solved by reading one screen,
2. store the structured intelligence for scoring and search,
3. run stage inference and raise a *proposal* if the call implies a pipeline
   move.

Matching is by phone. If no lead matches, the call is not dropped silently —
it is reported back so the caller can decide (personal call, colleague, or a
lead that should exist and does not).
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from ..ai import stage_inference
from ..contracts.intelligence import CallIntelligence
from ..contracts.vocabulary import ActivityType, CallDisposition, Channel, Direction
from ..db import repo
from ..db.models import CallIntelligenceRow, Lead
from ..services import leads as lead_service
from ..settings import get_settings
from ..util import normalize_phone, phone_variants


@dataclass
class IngestResult:
    matched: bool
    lead_id: UUID | None
    activity_id: UUID | None
    proposal_id: UUID | None
    auto_applied: bool
    reason: str


def ingest(session: Session, intel: CallIntelligence) -> IngestResult:
    """Attach one call's intelligence to its lead."""
    settings = get_settings()

    lead = _match_lead(session, intel.lead_phone)
    if lead is None:
        return IngestResult(
            matched=False,
            lead_id=None,
            activity_id=None,
            proposal_id=None,
            auto_applied=False,
            reason=f"no lead matches {intel.lead_phone}",
        )

    # Idempotency is keyed on FinEcho's call id: the webhook is at-least-once.
    existing = session.get(CallIntelligenceRow, intel.call_id)
    if existing is not None:
        return IngestResult(
            matched=True,
            lead_id=lead.id,
            activity_id=existing.activity_id,
            proposal_id=None,
            auto_applied=False,
            reason="call already ingested",
        )

    activity = repo.record_activity(
        session,
        lead_id=lead.id,
        type=ActivityType.CALL_SUMMARY.value,
        channel=Channel.CALL.value,
        direction=Direction.OUTBOUND.value,
        occurred_at=intel.occurred_at,
        actor_name=intel.agent_key,
        subject=_headline(intel),
        body=intel.summary,
        payload={
            "call_id": str(intel.call_id),
            "points_to_remember": intel.points_to_remember,
            "objections": intel.objections,
            "promised_actions": intel.promised_actions,
            "suggested_next_steps": intel.suggested_next_steps,
            "detected_program": intel.detected_program,
            "call_type": intel.call_type,
            "overall_score": intel.overall_score,
            "compliance_flags": intel.compliance_flags,
            "recording_uri": intel.recording_uri,
            "connected": True,
        },
        dedupe_key=f"finecho:summary:{intel.call_id}",
    )
    session.flush()

    session.add(
        CallIntelligenceRow(
            call_id=intel.call_id,
            lead_id=lead.id,
            activity_id=activity.id if activity else None,
            agent_key=intel.agent_key,
            occurred_at=intel.occurred_at,
            duration_seconds=intel.duration_seconds,
            language=intel.language,
            summary=intel.summary,
            points_to_remember=intel.points_to_remember,
            objections=intel.objections,
            hooks=intel.hooks,
            promised_actions=intel.promised_actions,
            suggested_next_steps=intel.suggested_next_steps,
            detected_program=intel.detected_program,
            call_type=intel.call_type,
            overall_score=intel.overall_score,
            criteria=intel.criteria,
            compliance_flags=intel.compliance_flags,
            recording_uri=intel.recording_uri,
            model_tag=intel.model_tag,
        )
    )

    if lead.first_touch_at is None:
        lead.first_touch_at = intel.occurred_at
    lead.connected_calls += 1
    if lead.last_activity_at is None or intel.occurred_at > lead.last_activity_at:
        lead.last_activity_at = intel.occurred_at

    proposal_id: UUID | None = None
    auto_applied = False
    signal = stage_inference.infer(intel, lead.stage, disposition=CallDisposition.CONNECTED, lead_id=lead.id)
    if signal is not None:
        row = stage_inference.to_proposal(signal, lead.id, lead.stage, call_id=intel.call_id)
        may_apply = stage_inference.may_auto_apply(
            signal,
            enabled=settings.stage_proposal_auto_apply,
            min_confidence=settings.stage_proposal_min_confidence,
        )
        if may_apply:
            row["status"] = "auto_applied"
        proposal = repo.create_proposal(session, row)
        if proposal is not None:
            proposal_id = proposal.id
            if may_apply:
                from ..contracts.activity import StageChange

                lead_service.change_stage(
                    session,
                    lead,
                    StageChange(
                        from_stage=lead.stage,
                        to_stage=signal.to_stage,
                        changed_by=None,
                        reason=f"auto-applied: {signal.rationale}",
                        proposal_id=proposal.id,
                    ),
                    actor_is_human=False,
                )
                auto_applied = True

    lead_service.refresh_score(session, lead)
    return IngestResult(
        matched=True,
        lead_id=lead.id,
        activity_id=activity.id if activity else None,
        proposal_id=proposal_id,
        auto_applied=auto_applied,
        reason="ingested",
    )


def _match_lead(session: Session, phone: str) -> Lead | None:
    """Find the lead this call belongs to, trying each phone variant."""
    normalized = normalize_phone(phone)
    if normalized:
        lead = repo.find_lead_by_identifier(session, "phone", normalized)
        if lead is not None:
            return lead
    for variant in phone_variants(phone):
        lead = repo.find_lead_by_identifier(session, "phone", variant)
        if lead is not None:
            return lead
    return None


def _headline(intel: CallIntelligence) -> str:
    parts = ["Call summary"]
    if intel.call_type:
        parts.append(f"({intel.call_type})")
    if intel.overall_score is not None:
        parts.append(f"— scored {intel.overall_score}/10")
    return " ".join(parts)[:255]
