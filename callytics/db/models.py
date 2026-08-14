"""SQLAlchemy models. Mirrors ``docs/design/03-database-schema.md``.

Two shape decisions worth stating up front, because everything else follows
from them:

1. **The timeline is append-only.** ``activities`` is never updated in place.
   A lead's current state (stage, owner, fields) is a projection you can always
   rebuild by replaying its activities. This is what makes "a counsellor picks
   up a six-month-old lead and needs the background" a solved problem rather
   than a note-reading exercise.

2. **Core columns are promoted; the long tail is JSON.** The live LeadSquared
   account carries 117 lead fields, but only ~15 are queried in hot paths.
   Those get real indexed columns; the rest live in ``leads.fields`` with a
   ``field_definitions`` table describing type and validation. Promoting a
   field later is a migration, not a redesign.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy import (
    DateTime as SaDateTime,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..util import utcnow


class UtcDateTime(TypeDecorator):
    """A datetime column that is always timezone-aware UTC in Python.

    Neither MySQL nor SQLite stores an offset, so a naive value comes back out
    and any comparison against ``utcnow()`` raises ``TypeError``. Rather than
    sprinkling defensive conversions through the domain layer, the boundary
    normalises: aware values are converted to UTC on the way in, and every
    value read back is tagged UTC on the way out.
    """

    impl = SaDateTime
    cache_ok = True

    def __init__(self, timezone: bool = True) -> None:
        # Accepts (and ignores) ``timezone`` so Alembic autogenerate, which
        # renders this type as ``UtcDateTime(timezone=True)``, round-trips.
        super().__init__(timezone=True)

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def DateTime(timezone: bool = True) -> UtcDateTime:  # noqa: N802 - drop-in for sa.DateTime
    """Shadow ``sqlalchemy.DateTime`` so every column in this module is UTC-aware."""
    return UtcDateTime()

PROPOSAL_STATUSES = ("pending", "accepted", "rejected", "expired", "auto_applied")
TASK_STATUSES = ("open", "completed", "cancelled", "overdue")
IDENTIFIER_KINDS = ("phone", "email")


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


# --- org ------------------------------------------------------------------


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)  # folder-safe, matches FinEcho counsellor_key
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(190), unique=True, nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="counsellor")  # counsellor|manager|admin|system
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    external_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Team(Base, TimestampMixin):
    __tablename__ = "teams"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    manager_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class TeamMember(Base):
    __tablename__ = "team_members"

    team_id: Mapped[UUID] = mapped_column(ForeignKey("teams.id"), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), primary_key=True)


# --- configurable vocabulary ---------------------------------------------


class LeadStage(Base):
    """Stages are data, not an enum — admins reorder and rename them."""

    __tablename__ = "lead_stages"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    is_revenue: Mapped[bool] = mapped_column(Boolean, default=False)
    is_nurture: Mapped[bool] = mapped_column(Boolean, default=False)
    is_artefact: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class LeadSource(Base):
    __tablename__ = "lead_sources"

    name: Mapped[str] = mapped_column(String(120), primary_key=True)
    is_paid: Mapped[bool] = mapped_column(Boolean, default=False)
    is_high_intent: Mapped[bool] = mapped_column(Boolean, default=False)
    is_bulk: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class FieldDefinition(Base, TimestampMixin):
    """Describes one entry in ``leads.fields``. Replaces LSQ's 117 mx_ columns."""

    __tablename__ = "field_definitions"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    label: Mapped[str] = mapped_column(String(160))
    data_type: Mapped[str] = mapped_column(String(24))  # text|number|date|select|multiselect|boolean|phone|email
    options: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    is_required: Mapped[bool] = mapped_column(Boolean, default=False)
    is_pii: Mapped[bool] = mapped_column(Boolean, default=False)
    legacy_schema_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# --- lead -----------------------------------------------------------------


class Lead(Base, TimestampMixin):
    __tablename__ = "leads"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)

    # identity (normalised; see util.normalize_phone)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    alternate_phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    email: Mapped[str | None] = mapped_column(String(190), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(160), nullable=True)

    # pipeline
    stage: Mapped[str] = mapped_column(ForeignKey("lead_stages.name"))
    source: Mapped[str] = mapped_column(ForeignKey("lead_sources.name"))
    source_campaign: Mapped[str | None] = mapped_column(String(190), nullable=True)
    source_medium: Mapped[str | None] = mapped_column(String(80), nullable=True)
    source_referrer: Mapped[str | None] = mapped_column(String(500), nullable=True)

    owner_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    team_id: Mapped[UUID | None] = mapped_column(ForeignKey("teams.id"), nullable=True)

    # consent gates
    do_not_call: Mapped[bool] = mapped_column(Boolean, default=False)
    do_not_sms: Mapped[bool] = mapped_column(Boolean, default=False)
    do_not_email: Mapped[bool] = mapped_column(Boolean, default=False)
    do_not_whatsapp: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consent_source: Mapped[str | None] = mapped_column(String(80), nullable=True)

    # denormalised for list views — rebuilt from activities, never hand-edited
    score: Mapped[int] = mapped_column(Integer, default=0)
    score_band: Mapped[str] = mapped_column(String(8), default="cold")
    first_touch_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_activity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_inbound_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    call_attempts: Mapped[int] = mapped_column(Integer, default=0)
    connected_calls: Mapped[int] = mapped_column(Integer, default=0)

    # long-tail custom fields (see FieldDefinition)
    fields: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    external_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    merged_into: Mapped[UUID | None] = mapped_column(ForeignKey("leads.id"), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_leads_phone", "phone"),
        Index("idx_leads_email", "email"),
        Index("idx_leads_owner_stage", "owner_id", "stage"),
        Index("idx_leads_stage_updated", "stage", "updated_at"),
        Index("idx_leads_team_stage", "team_id", "stage"),
        Index("idx_leads_score", "score"),
        Index("idx_leads_last_activity", "last_activity_at"),
        Index("idx_leads_source_captured", "source", "captured_at"),
    )


class LeadIdentifier(Base):
    """Unique contactable identifiers, one row each.

    Kept separate from ``leads`` because a person legitimately has several
    numbers, and because a UNIQUE constraint here is what actually prevents the
    duplicate-lead problem at write time rather than in a nightly cleanup job.
    """

    __tablename__ = "lead_identifiers"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    kind: Mapped[str] = mapped_column(Enum(*IDENTIFIER_KINDS, name="identifier_kind"))
    value: Mapped[str] = mapped_column(String(190))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("kind", "value", name="uq_identifier_kind_value"),
        Index("idx_identifier_lead", "lead_id"),
    )


# --- timeline -------------------------------------------------------------


class Activity(Base):
    __tablename__ = "activities"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    type: Mapped[str] = mapped_column(String(40))
    channel: Mapped[str] = mapped_column(String(20))
    direction: Mapped[str] = mapped_column(String(12))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    actor_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    actor_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dedupe_key: Mapped[str | None] = mapped_column(String(190), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_activity_lead_time", "lead_id", "occurred_at"),
        Index("idx_activity_type_time", "type", "occurred_at"),
        Index("idx_activity_actor_time", "actor_id", "occurred_at"),
    )


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    title: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(24), default="call")
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    assignee_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(Enum(*TASK_STATUSES, name="task_status"), default="open")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(255), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    watchers: Mapped[list[Any]] = mapped_column(JSON, default=list)

    __table_args__ = (
        Index("idx_task_assignee_due", "assignee_id", "status", "due_at"),
        Index("idx_task_lead", "lead_id"),
    )


class Appointment(Base, TimestampMixin):
    """Walk-ins, campus visits and calls booked for a future slot.

    Double-booking is prevented here rather than in the UI: ``uq_appointment_slot``
    means two counsellors cannot promise the same manager the same time.
    """

    __tablename__ = "appointments"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    host_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(24), default="walk_in")  # walk_in|video|phone
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    location: Mapped[str | None] = mapped_column(String(190), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="scheduled")
    outcome: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        UniqueConstraint("host_id", "starts_at", name="uq_appointment_slot"),
        Index("idx_appointment_host_time", "host_id", "starts_at"),
        Index("idx_appointment_lead", "lead_id"),
    )


# --- intelligence ---------------------------------------------------------


class Proposal(Base):
    """A machine suggestion awaiting a human decision. See contracts.intelligence."""

    __tablename__ = "proposals"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    kind: Mapped[str] = mapped_column(String(32))
    proposed: Mapped[dict[str, Any]] = mapped_column(JSON)
    current: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3))
    rationale: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(Enum(*PROPOSAL_STATUSES, name="proposal_status"), default="pending")
    dedupe_key: Mapped[str | None] = mapped_column(String(190), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("idx_proposal_lead_status", "lead_id", "status"),
        Index("idx_proposal_status_created", "status", "created_at"),
    )


class ScoreSnapshot(Base):
    """History of score changes — needed to prove the model is improving."""

    __tablename__ = "score_snapshots"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    score: Mapped[int] = mapped_column(Integer)
    band: Mapped[str] = mapped_column(String(8))
    contributions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model_version: Mapped[str] = mapped_column(String(40), default="rules/v1")
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("idx_score_lead_time", "lead_id", "computed_at"),)


class EngagementProfileRow(Base):
    __tablename__ = "engagement_profiles"

    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"), primary_key=True)
    windows: Mapped[list[Any]] = mapped_column(JSON, default=list)
    preferred_channel: Mapped[str | None] = mapped_column(String(20), nullable=True)
    responsiveness: Mapped[float] = mapped_column(Numeric(4, 3), default=0)
    total_observations: Mapped[int] = mapped_column(Integer, default=0)
    sufficient_data: Mapped[bool] = mapped_column(Boolean, default=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CallIntelligenceRow(Base):
    """FinEcho output landed against a lead. FinEcho remains the system of record."""

    __tablename__ = "call_intelligence"

    call_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    lead_id: Mapped[UUID] = mapped_column(ForeignKey("leads.id"))
    activity_id: Mapped[UUID | None] = mapped_column(ForeignKey("activities.id"), nullable=True)
    agent_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Numeric(8, 1), nullable=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    points_to_remember: Mapped[list[Any]] = mapped_column(JSON, default=list)
    objections: Mapped[list[Any]] = mapped_column(JSON, default=list)
    hooks: Mapped[list[Any]] = mapped_column(JSON, default=list)
    promised_actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    suggested_next_steps: Mapped[list[Any]] = mapped_column(JSON, default=list)
    detected_program: Mapped[str | None] = mapped_column(String(120), nullable=True)
    call_type: Mapped[str | None] = mapped_column(String(24), nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Numeric(3, 1), nullable=True)
    criteria: Mapped[list[Any]] = mapped_column(JSON, default=list)
    compliance_flags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    recording_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_tag: Mapped[str | None] = mapped_column(String(80), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("idx_ci_lead_time", "lead_id", "occurred_at"),)


class AudienceSegmentRow(Base, TimestampMixin):
    __tablename__ = "audience_segments"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    filter_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    destination: Mapped[str] = mapped_column(String(16))
    lead_count: Mapped[int] = mapped_column(Integer, default=0)
    external_audience_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class SavedView(Base, TimestampMixin):
    """A named filter a manager can share with a team (LSQ 'shared view' parity)."""

    __tablename__ = "saved_views"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    filter_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    shared_with_team_id: Mapped[UUID | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    is_shared: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_view_owner_name"),)


# --- operations -----------------------------------------------------------


class ProcessedEvent(Base):
    """Consumer-side idempotency ledger; the bus is at-least-once."""

    __tablename__ = "processed_events"

    event_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    consumer_group: Mapped[str] = mapped_column(String(48), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("idx_pe_age", "processed_at"),)


class AuditLog(Base):
    """Who changed what. Every write that a human could dispute lands here."""

    __tablename__ = "audit_log"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    actor_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    actor_label: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(80))
    target_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[str] = mapped_column(String(80))
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_audit_created", "created_at"),
        Index("idx_audit_target", "target_type", "target_id"),
    )


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    recipient_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(40))
    lead_id: Mapped[UUID | None] = mapped_column(ForeignKey("leads.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(190))
    body: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("idx_notif_recipient_read", "recipient_id", "read_at"),)
