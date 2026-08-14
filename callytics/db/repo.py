"""Repository: all SQL lives here.

Services call this module; nothing else constructs queries. That boundary is
what lets the domain layer stay pure and lets the whole pipeline be tested
against SQLite while production runs MySQL.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..contracts.vocabulary import ENGAGEMENT_ACTIVITIES, ActivityType, Direction
from ..domain.identity import ExistingLead
from ..util import utcnow, uuid7
from .models import (
    Activity,
    Appointment,
    AuditLog,
    CallIntelligenceRow,
    EngagementProfileRow,
    FieldDefinition,
    Lead,
    LeadIdentifier,
    ProcessedEvent,
    Proposal,
    ScoreSnapshot,
    Task,
    Team,
    TeamMember,
    User,
)

# --- leads ----------------------------------------------------------------


def get_lead(session: Session, lead_id: UUID) -> Lead | None:
    return session.get(Lead, lead_id)


def find_lead_by_identifier(session: Session, kind: str, value: str) -> Lead | None:
    stmt = (
        select(Lead)
        .join(LeadIdentifier, LeadIdentifier.lead_id == Lead.id)
        .where(LeadIdentifier.kind == kind, LeadIdentifier.value == value)
        .limit(1)
    )
    return session.execute(stmt).scalars().first()


def identity_candidates(
    session: Session,
    *,
    phones: Sequence[str],
    email: str | None,
    full_name: str | None,
) -> list[ExistingLead]:
    """Narrow the dedupe search to leads plausibly matching the incoming record.

    Only exact identifier matches and an exact name match are fetched — the
    resolver deliberately refuses to act on anything looser, so a wider net
    would just cost query time.
    """
    values = [v for v in phones if v]
    lead_ids: set[UUID] = set()

    if values or email:
        stmt = select(LeadIdentifier.lead_id).where(
            LeadIdentifier.value.in_([*values, *([email] if email else [])])
        )
        lead_ids.update(session.execute(stmt).scalars().all())

    if full_name:
        stmt2 = select(Lead.id).where(Lead.full_name == full_name).limit(25)
        lead_ids.update(session.execute(stmt2).scalars().all())

    if not lead_ids:
        return []

    leads = session.execute(select(Lead).where(Lead.id.in_(lead_ids))).scalars().all()
    out: list[ExistingLead] = []
    for lead in leads:
        idents = session.execute(
            select(LeadIdentifier).where(LeadIdentifier.lead_id == lead.id)
        ).scalars().all()
        out.append(
            ExistingLead(
                lead_id=lead.id,
                phones=tuple(i.value for i in idents if i.kind == "phone"),
                emails=tuple(i.value for i in idents if i.kind == "email"),
                full_name=lead.full_name,
                is_merged=lead.merged_into is not None,
            )
        )
    return out


def add_identifier(session: Session, lead_id: UUID, kind: str, value: str, *, primary: bool = False) -> None:
    """Idempotent: re-adding an identifier the lead already has is a no-op."""
    existing = session.execute(
        select(LeadIdentifier).where(LeadIdentifier.kind == kind, LeadIdentifier.value == value)
    ).scalars().first()
    if existing is not None:
        return
    session.add(
        LeadIdentifier(id=uuid7(), lead_id=lead_id, kind=kind, value=value, is_primary=primary)
    )


# --- activities -----------------------------------------------------------


def record_activity(session: Session, **kwargs: Any) -> Activity | None:
    """Append one activity. Returns None if ``dedupe_key`` was already seen."""
    dedupe_key = kwargs.get("dedupe_key")
    if dedupe_key:
        existing = session.execute(
            select(Activity).where(Activity.dedupe_key == dedupe_key)
        ).scalars().first()
        if existing is not None:
            return None
    activity = Activity(id=kwargs.pop("id", None) or uuid7(), **kwargs)
    session.add(activity)
    return activity


def timeline(
    session: Session,
    lead_id: UUID,
    *,
    limit: int = 50,
    before: datetime | None = None,
    types: Sequence[str] | None = None,
) -> list[Activity]:
    stmt = select(Activity).where(Activity.lead_id == lead_id)
    if before is not None:
        stmt = stmt.where(Activity.occurred_at < before)
    if types:
        stmt = stmt.where(Activity.type.in_(list(types)))
    stmt = stmt.order_by(Activity.occurred_at.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


def scoring_facts_for(session: Session, lead_id: UUID) -> dict[str, Any]:
    """Aggregate the counts the scorer needs in one pass per lead."""
    rows = session.execute(
        select(Activity.type, Activity.direction, func.count(), func.max(Activity.occurred_at))
        .where(Activity.lead_id == lead_id)
        .group_by(Activity.type, Activity.direction)
    ).all()

    inbound_messages = 0
    email_opens = 0
    last_engagement: datetime | None = None
    for type_, direction, count, latest in rows:
        if type_ == ActivityType.WHATSAPP_MESSAGE.value and direction == Direction.INBOUND.value:
            inbound_messages += count
        if type_ == ActivityType.EMAIL_OPENED.value:
            email_opens += count
        if type_ in {a.value for a in ENGAGEMENT_ACTIVITIES} and latest is not None:
            last_engagement = latest if last_engagement is None else max(last_engagement, latest)

    kept = session.execute(
        select(func.count()).select_from(Appointment).where(
            Appointment.lead_id == lead_id, Appointment.status == "completed"
        )
    ).scalar_one()
    missed = session.execute(
        select(func.count()).select_from(Appointment).where(
            Appointment.lead_id == lead_id, Appointment.status == "no_show"
        )
    ).scalar_one()

    latest_call = session.execute(
        select(CallIntelligenceRow)
        .where(CallIntelligenceRow.lead_id == lead_id)
        .order_by(CallIntelligenceRow.occurred_at.desc())
        .limit(1)
    ).scalars().first()

    return {
        "inbound_messages": inbound_messages,
        "email_opens": email_opens,
        "last_engagement_at": last_engagement,
        "appointments_kept": int(kept or 0),
        "appointments_missed": int(missed or 0),
        "latest_call_score": float(latest_call.overall_score) if latest_call and latest_call.overall_score else None,
        "has_objection": bool(latest_call.objections) if latest_call else False,
    }


def save_score(session: Session, lead_id: UUID, score: int, band: str, contributions: list[Any]) -> None:
    session.add(
        ScoreSnapshot(
            id=uuid7(),
            lead_id=lead_id,
            score=score,
            band=band,
            contributions=contributions,
        )
    )


# --- proposals ------------------------------------------------------------


def create_proposal(session: Session, row: dict[str, Any]) -> Proposal | None:
    """Insert a proposal, or return None if its dedupe key already exists."""
    dedupe_key = row.get("dedupe_key")
    if dedupe_key:
        existing = session.execute(
            select(Proposal).where(Proposal.dedupe_key == dedupe_key)
        ).scalars().first()
        if existing is not None:
            return None
    proposal = Proposal(**row)
    session.add(proposal)
    return proposal


def pending_proposals(session: Session, lead_id: UUID | None = None, *, limit: int = 50) -> list[Proposal]:
    stmt = select(Proposal).where(Proposal.status == "pending")
    if lead_id is not None:
        stmt = stmt.where(Proposal.lead_id == lead_id)
    return list(session.execute(stmt.order_by(Proposal.created_at.desc()).limit(limit)).scalars().all())


# --- engagement -----------------------------------------------------------


def responses_for(session: Session, lead_id: UUID, *, since_days: int = 180) -> list[Activity]:
    """Inbound activity only — what the lead did, not what we did to them."""
    cutoff = utcnow() - timedelta(days=since_days)
    stmt = select(Activity).where(
        Activity.lead_id == lead_id,
        Activity.occurred_at >= cutoff,
        Activity.direction == Direction.INBOUND.value,
    )
    return list(session.execute(stmt).scalars().all())


def save_engagement_profile(session: Session, lead_id: UUID, payload: dict[str, Any]) -> None:
    row = session.get(EngagementProfileRow, lead_id)
    if row is None:
        row = EngagementProfileRow(lead_id=lead_id)
        session.add(row)
    for key, value in payload.items():
        setattr(row, key, value)
    row.computed_at = utcnow()


# --- org ------------------------------------------------------------------


def routing_candidates(session: Session, team_name: str | None = None) -> list[dict[str, Any]]:
    """Active counsellors with their team membership and current open load."""
    stmt = select(User).where(User.is_active.is_(True), User.role == "counsellor")
    users = list(session.execute(stmt).scalars().all())

    open_counts = dict(
        session.execute(
            select(Lead.owner_id, func.count())
            .where(Lead.owner_id.is_not(None), Lead.merged_into.is_(None))
            .group_by(Lead.owner_id)
        ).all()
    )

    memberships = session.execute(
        select(TeamMember.user_id, Team.name).join(Team, Team.id == TeamMember.team_id)
    ).all()
    teams_by_user: dict[UUID, set[str]] = {}
    for user_id, name in memberships:
        teams_by_user.setdefault(user_id, set()).add(name)

    out = []
    for user in users:
        user_teams = teams_by_user.get(user.id, set())
        if team_name and team_name not in user_teams:
            continue
        out.append(
            {
                "user_id": user.id,
                "name": user.name,
                "teams": frozenset(user_teams),
                "open_leads": int(open_counts.get(user.id, 0)),
            }
        )
    return out


def field_definition_keys(session: Session) -> list[str]:
    return list(
        session.execute(select(FieldDefinition.key).where(FieldDefinition.is_active.is_(True))).scalars().all()
    )


# --- ops ------------------------------------------------------------------


def mark_processed(session: Session, event_id: UUID, group: str) -> bool:
    """Return True if this consumer had not already handled ``event_id``."""
    existing = session.get(ProcessedEvent, {"event_id": event_id, "consumer_group": group})
    if existing is not None:
        return False
    session.add(ProcessedEvent(event_id=event_id, consumer_group=group))
    return True


def audit(
    session: Session,
    *,
    actor_id: UUID | None,
    actor_label: str,
    action: str,
    target_type: str,
    target_id: str,
    detail: dict[str, Any] | None = None,
) -> None:
    session.add(
        AuditLog(
            id=uuid7(),
            actor_id=actor_id,
            actor_label=actor_label,
            action=action,
            target_type=target_type,
            target_id=target_id,
            detail=detail,
        )
    )


def open_tasks_for(session: Session, assignee_id: UUID, *, limit: int = 100) -> list[Task]:
    stmt = (
        select(Task)
        .where(Task.assignee_id == assignee_id, Task.status == "open")
        .order_by(Task.due_at)
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())
