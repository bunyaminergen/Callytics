"""Timeline activity contracts — the append-only history of a lead."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .vocabulary import ActivityType, CallDisposition, Channel, Direction


class ActivityCreate(BaseModel):
    """One thing that happened to a lead.

    ``dedupe_key`` is what makes ingestion safely retryable: telephony vendors
    and webhook senders redeliver, and every producer here is at-least-once.
    Two activities with the same key collapse to one row.
    """

    lead_id: UUID
    type: ActivityType
    channel: Channel
    direction: Direction = Direction.OUTBOUND
    occurred_at: datetime
    actor_id: UUID | None = None
    actor_name: str | None = None
    subject: str | None = None
    body: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str | None = None


class ActivityView(BaseModel):
    id: UUID
    lead_id: UUID
    type: ActivityType
    channel: Channel
    direction: Direction
    occurred_at: datetime
    actor_id: UUID | None
    actor_name: str | None
    subject: str | None
    body: str | None
    payload: dict[str, Any]


class CallLog(BaseModel):
    """Telephony CDR. Arrives from the IVR vendor, independent of FinEcho."""

    lead_phone: str
    agent_id: UUID | None = None
    agent_key: str | None = None
    direction: Direction
    started_at: datetime
    duration_seconds: float = 0.0
    disposition: CallDisposition
    recording_uri: str | None = None
    provider: str = "unknown"
    provider_call_id: str | None = None


class StageChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_stage: str | None
    to_stage: str
    changed_by: UUID | None
    reason: str = "manual"
    proposal_id: UUID | None = None


class TaskCreate(BaseModel):
    lead_id: UUID
    title: str
    due_at: datetime
    assignee_id: UUID
    kind: str = "call"  # call|meeting|walk_in|follow_up
    notes: str | None = None
    created_by: UUID | None = None
    watchers: list[UUID] = Field(default_factory=list)


class TaskView(BaseModel):
    id: UUID
    lead_id: UUID
    title: str
    kind: str
    due_at: datetime
    assignee_id: UUID
    status: str
    notes: str | None
    completed_at: datetime | None
    outcome: str | None
